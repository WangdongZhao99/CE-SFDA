import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.models import classifier, Temporal_Imputer, masking, classifier_aad
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl, SupConLoss
from scipy.spatial.distance import cdist
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import copy
from numpy import linalg as LA
import loss
import sys
import torch.optim as optim


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

class SHOT(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SHOT, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # Freeze the classifier
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                # prevent gradient accumulation
                self.optimizer.zero_grad()

                # Extract features
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # pseudo labeling loss
                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)

                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)

        preds = torch.cat((preds))
        feas = torch.cat((feas))

        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)

        self.feature_extractor.train()
        self.classifier.train()
        return pred_label


class CESFDA(Algorithm):

    def __init__(self, backbone, configs, hparams, device, is_wegited=False):
        super(CESFDA, self).__init__(configs)
        self.is_wegited = is_wegited
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier_aad = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_aad = torch.optim.Adam(
            self.classifier_aad.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.supcon = SupConLoss(self.device)
    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y, None)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()
        classifier_state_dict = self.classifier.state_dict()
        self.classifier_aad.load_state_dict(classifier_state_dict)
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        num_samples = len(trg_dataloader.dataset)
        fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len) 
        score_bank = torch.randn(num_samples, self.configs.num_classes).cuda()                              
        temp = True   
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (trg_x, y, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                if temp:
                    self.feature_extractor.eval()
                    self.classifier_aad.eval()
                    with torch.no_grad():
                        iter_test = iter(trg_dataloader)
                        for i in range(len(trg_dataloader)):
                            data = next(iter_test)
                            inputs = data[0]
                            indx = data[-1]
                            inputs = inputs.cuda()
                            output, _ = self.feature_extractor(inputs)
                            output_norm = F.normalize(output)
                            outputs = self.classifier_aad(output)
                            outputs = nn.Softmax(-1)(outputs)
                            fea_bank[indx] = output_norm.detach().clone().cpu()
                            score_bank[indx] = outputs.detach().clone()  # .cpu()
                    self.feature_extractor.train()
                    self.classifier_aad.train()
                    temp = False
                self.optimizer.zero_grad()
                self.optimizer_aad.zero_grad()
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)#shot logits
                trg_pred_aad = self.classifier_aad(trg_feat)#AaD logits
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                softmax_out_aad = nn.Softmax(dim=1)(trg_pred_aad)
                with torch.no_grad():
                    output_f_norm = F.normalize(trg_feat)
                    output_f_ = output_f_norm.cpu().detach().clone()
                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out_aad.detach().clone()
                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C
                softmax_out_un = softmax_out_aad.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C
                loss_aad = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
                with torch.no_grad():
                    entropy1 = -torch.sum(softmax_out * torch.log2(softmax_out + 1e-5), dim=1)
                    entropy2 = -torch.sum(softmax_out_aad * torch.log2(softmax_out_aad + 1e-5), dim=1)
                    weight = torch.exp(-entropy1) / (torch.exp(-entropy1) + torch.exp(-entropy2))
                    weighted_logits = weight[:, None] * trg_pred + (1 - weight)[:, None] * trg_pred_aad
                    weighted_softmax = nn.Softmax(dim=1)(weighted_logits)
                    max_entropy = torch.log2(torch.tensor(self.configs.num_classes))
                    # w = -torch.sum(weighted_softmax * torch.log2(weighted_softmax + 1e-5), dim=1)
                    # w = w / max_entropy
                    # w = torch.exp(-w)

                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))
                if self.is_wegited:
                    _, pseudo_label = weighted_softmax.max(dim=1)
                else:
                    with torch.no_grad():
                        max_logits = []
                        for i in range(softmax_out.shape[0]):
                            v1 = softmax_out[i].max()
                            v2 = softmax_out_aad[i].max()
                            if v1 > v2:
                                max_logits.append(softmax_out[i])
                            else:
                                max_logits.append(softmax_out_aad[i])
                        max_logits = torch.cat([i.unsqueeze(0) for i in max_logits], dim=0)
                        pseudo_label = max_logits.argmax(-1)
                target_loss = self.cross_entropy(trg_pred.squeeze(), pseudo_label.long())
                loss = entropy_loss + self.hparams['aad_wt'] * loss_aad + self.hparams['target_cls_wt'] * target_loss
                
                loss.backward()
                self.optimizer.step()
                self.optimizer_aad.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model

class AaD(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(AaD, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        fea_bank, score_bank = self.build_feat_score_bank(trg_dataloader)
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            # defining best and last model
            best_src_risk = float('inf')
            best_model = self.network.state_dict()
            last_model = self.network.state_dict()



            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                num_samples = len(trg_dataloader.dataset)

                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)

                # output softmax probs
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model

    def build_feat_score_bank(self, data_loader):
        fea_bank = torch.empty(0).cuda()
        score_bank = torch.empty(0).cuda()

        self.feature_extractor.eval()
        self.classifier.eval()

        # Process data batch by batch
        for data in data_loader:
            batch_data = data[0].cuda()  # Assuming the first element in your batch is the data. Adjust as needed.

            batch_feat,_ = self.feature_extractor(batch_data)
            norm_feats = F.normalize(batch_feat)
            batch_pred = self.classifier(batch_feat)
            batch_probs = nn.Softmax(dim=-1)(batch_pred)

            # Update the banks
            fea_bank = torch.cat((fea_bank, norm_feats.detach()), 0)
            score_bank = torch.cat((score_bank, batch_probs.detach()), 0)
        self.feature_extractor.train()
        self.classifier.train()
        return fea_bank, score_bank


class NRC(Algorithm):
    """
    Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NIPS 2021)
    https://github.com/Albert0147/NRC_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(NRC, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).cuda()
                softmax_out = nn.Softmax(dim=1)(predictions)

                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                    fea_near = fea_bank[idx_near]  # batch x K x num_dim
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                    distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=5 + 1)  # M near neighbors for each of above K ones
                    idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                    trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                    match = (
                            idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(0.1))  # batch x K

                    weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                            5)  # batch x K x M
                    weight_kk = weight_kk.fill_(0.1)

                    # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                    # weight_kk[idx_near_near == trg_idx_]=0

                    score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                    # print(weight_kk.shape)
                    weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                            -1)  # batch x KM

                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                    self.configs.num_classes)  # batch x KM x C

                    score_self = score_bank[trg_idx]

                # start gradients
                output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                            -1)  # batch x C x 1
                const = torch.mean(
                    (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                     weight_kk.cuda()).sum(
                        1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                loss = torch.mean(const)

                # nn
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss += torch.mean(
                    (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

                # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                # loss += -torch.mean((softmax_out * score_self).sum(-1))

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax *
                                          torch.log(msoftmax + self.hparams['epsilon']))
                loss += gentropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences


                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss
                total_loss.backward()
                self.pre_optimizer.step()
                # self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(),}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False


        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                # self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent

                loss.backward()
                self.optimizer.step()
                # self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model
    
class SCLM(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(SCLM, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences


                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss
                total_loss.backward()
                self.pre_optimizer.step()
                # self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(),}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
              # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            ent_old_val = 0
            self.feature_extractor.eval()
          
            
            mem_label, ent_new_val, feas_SNTg_dic, feas_SNTl_dic = self.obtain_label(trg_dataloader, self.feature_extractor, self.classifier, ent_old_val)
            mem_label = torch.from_numpy(mem_label).cuda()
            self.feature_extractor.train()
            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                
                ent_old_val = ent_new_val
                
#                 ada_dic_num = feas_SNTg_dic.size(0)
                
                fea_f_ori,_ = self.feature_extractor(trg_x)
                ada_dic_num = feas_SNTg_dic.size(0)
                fea_f_snt_g = self.SNT_global_detect(fea_f_ori, feas_SNTg_dic)
                fea_f_snt_g = torch.from_numpy(fea_f_snt_g).cuda()

                fea_f_snt_l = self.SNT_local_detect(fea_f_ori, feas_SNTl_dic, ada_dic_num)
                fea_f_snt_l = fea_f_snt_l.cuda()

                outputs_test_ori = self.classifier(fea_f_ori)
                outputs_test_snt_g = self.classifier(fea_f_snt_g)
                outputs_test_snt_l = self.classifier(fea_f_snt_l)

                softmax_out_ori = nn.Softmax(dim=1)(outputs_test_ori)
                softmax_out_snt_g = nn.Softmax(dim=1)(outputs_test_snt_g)
                softmax_out_snt_l = nn.Softmax(dim=1)(outputs_test_snt_l)

                output_ori_re = softmax_out_ori.unsqueeze(1)
                output_snt_g_re = softmax_out_snt_g.unsqueeze(1)
                output_snt_l_re = softmax_out_snt_l.unsqueeze(1)

                output_snt_g_re = output_snt_g_re.permute(0,2,1)
                output_snt_l_re = output_snt_l_re.permute(0,2,1)

                classifier_loss_snt_g = torch.log(torch.bmm(output_ori_re,output_snt_g_re)).sum(-1)
                classifier_loss_snt_l = torch.log(torch.bmm(output_ori_re,output_snt_l_re)).sum(-1)

                loss_const_snt_g = -torch.mean(classifier_loss_snt_g)
                loss_const_snt_l = -torch.mean(classifier_loss_snt_l)


                pred = mem_label[trg_idx]
                ss_ori_loss = nn.CrossEntropyLoss()(outputs_test_ori, pred)
                ss_ori_loss *= 0.3

                ss_snt_loss = loss_const_snt_g + loss_const_snt_l
                ss_snt_loss *= 0.1

                classifier_loss = ss_ori_loss + ss_snt_loss
                
                
                softmax_out = softmax_out_ori + softmax_out_snt_g + softmax_out_snt_l
                entropy_loss = torch.mean(loss.Entropy(softmax_out))


                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss
                
                
                '''
                Overall objective loss
                '''
                # removing trg ent
                im_loss = entropy_loss 
                classifier_loss += im_loss
                self.optimizer.zero_grad()
                classifier_loss.backward()
#                 optimizer.step()
                self.optimizer.step()
                # self.tov_optimizer.step()

                losses = {'entropy_loss': entropy_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model
  
    def obtain_label(self, loader, netF, netC, ent_old_val_):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                
                feas_f,_ = netF(inputs)
               
                outputs = netC(feas_f)
                if start_test:
                    all_fea_f = feas_f.float().cpu()
                    
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea_f = torch.cat((all_fea_f, feas_f.float().cpu()), 0)
                    
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
        unknown_weight = 1 - ent / np.log(3)                           # class num
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        # entropy with momenum
        #===============================================================================
        SNTl_dic_ = copy.deepcopy(all_fea_f)
        ent_cur = ent.cpu().numpy()
        ent_cur_val = ent_cur
        ent_var_tmp = ent_old_val_ - ent_cur_val
        ent_var_ = np.maximum(ent_var_tmp, -ent_var_tmp)
        ent_with_mom = 0.3 * ent_cur_val + (1.0 - 0.3) * ent_var_
        #===============================================================================

        
#         all_fea_f = torch.cat((all_fea_f, 1)
        all_fea = (all_fea_f.t() / torch.norm(all_fea_f, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()


        # EntMomClustering
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        net_pred_label = predict.cpu().numpy()
        confi_net_pred_lst = []
        confi_all_fea_lst = []
        confi_aff_lst = []
        for i in range(K):
            idx_i = np.where(net_pred_label == i)[0]
            ent_fin_slt_cls = ent_with_mom[idx_i]
            all_fea_cls = all_fea[idx_i, :]
            aff_cls = aff[idx_i, :]
            pred_cls = net_pred_label[idx_i]
            if idx_i.shape[0] > 0:
                confi_all_fea_i, confi_aff_i, pred_i = self.get_confi_fea_and_op(ent_fin_slt_cls, all_fea_cls, aff_cls, pred_cls)
                confi_all_fea_lst.append(confi_all_fea_i)
                confi_aff_lst.append(confi_aff_i)
                confi_net_pred_lst.append(pred_i)

        confi_all_fea = np.vstack(tuple(confi_all_fea_lst))
        confi_aff = np.vstack(tuple(confi_aff_lst))
        confi_pred_slt = np.hstack(tuple(confi_net_pred_lst))

        initc_confi = confi_aff.transpose().dot(confi_all_fea) 
        initc_confi = initc_confi / (1e-8 + confi_aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[confi_pred_slt].sum(axis=0)
        labelset = np.where(cls_count>0.0)
        labelset = labelset[0]

        initc_ori = aff.transpose().dot(all_fea)
        initc_ori = initc_ori / (1e-8 + aff.sum(axis=0)[:,None])

        initc = 0.3 * initc_confi + (1.0 - 0.3) * initc_ori

        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        # construct SNTg dictionary
        #-------------------------------------------------------------------------------
        K_pred = K
        confi_cls_lst = []
        all_fea_f = all_fea_f.cpu().numpy()
        for i in range(K_pred):
            idx_i = np.where(pred_label == i)[0]
            ent_fin_slt_cls = ent_with_mom[idx_i]
            all_fea_f_cls = all_fea_f[idx_i, :]
            if idx_i.shape[0] > 0:
                confi_cls_i = self.SNTg_dic_cls(ent_fin_slt_cls, all_fea_f_cls, K_pred)
                confi_cls_lst.append(confi_cls_i)

        feas_SNTg_dic_ = np.vstack(tuple(confi_cls_lst))
        feas_SNTg_dic_ = torch.from_numpy(feas_SNTg_dic_).cpu()
        #-------------------------------------------------------------------------------
        
        

        return pred_label.astype('int'), ent_cur, feas_SNTg_dic_, SNTl_dic_
        
    def SNT_global_detect(self, data_t_batch, data_s_confi):
        data_t = data_t_batch.detach()
        data_s = data_s_confi.detach()
        data_t_ = data_t.cpu().numpy()
        data_s_ = data_s.cpu().numpy()

        X = np.transpose(data_s_)
        Y = np.transpose(data_t_)
        beta = 20.0

        Xt = np.transpose(X)
        I = np.identity(Xt.shape[0])

        par_1 = np.matmul(Xt, X)
        par_2 = np.multiply(beta, I)

        B = par_1 + par_2
        Binv = np.linalg.inv(B)
        C = np.matmul(Binv, Xt)
        
        recon_fea = np.matmul(C, Y)

        idx_recon = np.argmax(recon_fea, axis = 0)
        recon_from_confi = data_s_[idx_recon, :]
        
        return recon_from_confi
    def SNT_local_detect(self, data_q, data_all, ada_num_):
        data_q_ = data_q.detach()
        data_all_ = data_all.detach()
        data_q_ = data_q_.cpu().numpy()
        data_all_ = data_all_.cpu().numpy()

        sim_slt, fea_nh_nval = self.get_sim_in_batch(data_q_, data_all_, ada_num_)
        mask_fea_in_batch = self.get_mask_in_batch(sim_slt, fea_nh_nval)
        re_tmp = self.get_similar_fea_in_batch(mask_fea_in_batch, data_all_)

        re = torch.from_numpy(re_tmp)
        return re
    
    def SNTg_dic_cls(self, ent_cls, fea_cls, K):
    
        balance_num = int(2 * fea_cls.shape[1] / K)
        len_confi = int(ent_cls.shape[0] * 0.3)
        ent_fin = torch.from_numpy(ent_cls)

        if len_confi > balance_num:
            len_confi = balance_num
#             print("== cls is not balance ==")

        idx_confi = ent_fin.topk(len_confi, largest = False)[-1]
        fea_confi = fea_cls[idx_confi, :]
        
        return fea_confi
    
    def get_confi_fea_and_op(self, ent_cls, fea_cls, aff_cls, pred_cls):
    
        len_confi = int(ent_cls.shape[0] * 0.3) + 1
        ent_cls_tensor = torch.from_numpy(ent_cls)
        idx_confi = ent_cls_tensor.topk(len_confi, largest = False)[-1]
        fea_confi = fea_cls[idx_confi, :]
        aff_confi = aff_cls[idx_confi, :]
        pred_slt = pred_cls[idx_confi]

        return fea_confi, aff_confi, pred_slt
    
    def get_sim_in_batch(self, Q, X, basis_num_):
        Xt = np.transpose(X)
        Simo = np.dot(Q, Xt)               
        nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
        nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
        Nor = np.dot(nq, nx)
        Sim_f = 1 - (Simo / Nor) 

        indices_min = np.argmin(Sim_f, axis=1)
        indices_row = np.arange(0, Q.shape[0], 1)
        Sim_f[indices_row, indices_min] = 999
        Sim_f_sorted = np.sort(Sim_f, axis = 1)

        threshold_num = X.shape[0]//basis_num_
        get_nh_nval = Sim_f_sorted[:, threshold_num]

        return Sim_f, get_nh_nval

    def get_mask_in_batch(self, Sim_f, fea_nh_nval):

        fea_nh_nval_f = np.expand_dims(fea_nh_nval, axis = 1)

        fea_nh_nval_zerof = np.zeros_like(Sim_f)
        fea_nh_nval_ff = fea_nh_nval_f + fea_nh_nval_zerof

        fea_nh_nval_slt = Sim_f - fea_nh_nval_ff

        all_1 = np.ones_like(Sim_f)
        fea_nh_nval_slt = torch.from_numpy(fea_nh_nval_slt)
        all_1 = torch.from_numpy(all_1)
        fea_nh_nval_zerof = torch.from_numpy(fea_nh_nval_zerof)

        mask_fea = torch.where(fea_nh_nval_slt <= 0.0, all_1, fea_nh_nval_zerof)
        mask_fea = mask_fea.cpu().numpy()

        return mask_fea

    def get_similar_fea_in_batch(self, mask_fea_f, fea_all_f):
        ln = mask_fea_f.shape[0]
        ext_fea_list = []

        for k in range(ln):
            idx_hunter_feas = np.where(mask_fea_f[k] == 1.0)[0]
            fea_hunter_k = fea_all_f[idx_hunter_feas]

            if fea_hunter_k.shape[0] > 1:
                fea_hunter = np.mean(fea_hunter_k, axis=0)
            else:
                fea_hunter = fea_hunter_k
            ext_fea_list.append(fea_hunter) 

        ext_fea_arr = np.vstack(tuple(ext_fea_list))
        return ext_fea_arr
    
class TPDS(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(TPDS, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
    def op_copy(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']
        return optimizer
    
    
    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return optimizer
    
   

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                # self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences


                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss
                total_loss.backward()
                self.pre_optimizer.step()
                # self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(),}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model
    
    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

#         # freeze both classifier and ood detector
#         for k, v in self.classifier.named_parameters():
#             v.requires_grad = False
#               # obtain pseudo labels
        param_group = []
        for k, v in self.feature_extractor.named_parameters():
            if 0.1 > 0:
                param_group += [{'params': v, 'lr': 1e-3 * 0.1}]
            else:
                v.requires_grad = False

        for k, v in self.classifier.named_parameters():
            if 0.1 > 0:
                param_group += [{'params': v, 'lr': 1e-3 * 0.1}]
            else:
                v.requires_grad = False


      
        
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            self.feature_extractor.eval()
            _, feas_all, label_confi, _, _ = self.obtain_label_ts(trg_dataloader, self.feature_extractor, self.classifier)
            self.feature_extractor.train()

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)


                # -----------------------------------data--------------------------------
    #                 inputs_test = inputs_test.cuda()
                # inputs_test_aug = inputs_test_aug.cuda()
                features_test_F, _ = self.feature_extractor(trg_x)

                outputs_test = self.classifier(features_test_F)
                softmax_out = nn.Softmax(dim=1)(outputs_test)

                features_test_N, _, _ = self.obtain_nearest_trace(features_test_F, feas_all, label_confi)
                features_test_N = torch.from_numpy(features_test_N).cuda()

                outputs_test_N = self.classifier(features_test_N)
                softmax_out_hyper = nn.Softmax(dim=1)(outputs_test_N)

                # -------------------------------objective------------------------------
                classifier_loss = torch.tensor(0.0).cuda()
                iic_loss = self.IID_loss(softmax_out, softmax_out_hyper)
                classifier_loss = classifier_loss + 1 * iic_loss




                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                gentropy_loss = gentropy_loss * 0.1
                classifier_loss = classifier_loss - gentropy_loss

                # elif cfg.dset == "office-home":
                #     msoftmax = softmax_out.mean(dim=0)
                #     gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.epsilon))
                #     gentropy_loss = gentropy_loss * 0.5
                #     classifier_loss = classifier_loss - gentropy_loss

                # --------------------------------------------------------------------    
                
                self.optimizer.zero_grad()
                classifier_loss.backward()
                self.optimizer.step()
        return last_model, best_model

    def obtain_label_ts(self,loader, netF, netC):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                feas_F, _ = netF(inputs)
#                 feas = netB(feas_F)
                outputs = netC(feas_F)
                if start_test:
                    all_fea_F = feas_F.float().cpu()
#                     all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
#                     all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        # all_logis = all_output
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
        unknown_weight = 1 - ent / np.log(6)
        _, predict = torch.max(all_output, 1)

        len_unconfi = int(ent.shape[0]*0.5)
        idx_unconfi = ent.topk(len_unconfi, largest=True)[-1]
        idx_unconfi_list_ent = idx_unconfi.cpu().numpy().tolist()

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
#         if 'cosine' == 'cosine':
#             all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
#             all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea_F = all_fea_F.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea_F)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>1e-5)
        labelset = labelset[0]
        # print(labelset)

        dd = cdist(all_fea_F, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        # --------------------use dd to get confi_idx and unconfi_idx-------------
        dd_min = dd.min(axis = 1)
        dd_min_tsr = torch.from_numpy(dd_min).detach()
        dd_t_confi = dd_min_tsr.topk(int((dd.shape[0]*0.6)), largest = False)[-1]
        dd_confi_list = dd_t_confi.cpu().numpy().tolist()
        dd_confi_list.sort()
        idx_confi = dd_confi_list

        idx_all_arr = np.zeros(shape = dd.shape[0], dtype = np.int64)
        idx_all_arr[idx_confi] = 1
        idx_unconfi_arr = np.where(idx_all_arr == 0)
        idx_unconfi_list_dd = list(idx_unconfi_arr[0])

        idx_unconfi_list = list(set(idx_unconfi_list_dd).intersection(set(idx_unconfi_list_ent)))
        # ------------------------------------------------------------------------
        # idx_unconfi_list = idx_unconfi_list_dd # idx_unconfi_list_dd

        label_confi = np.ones(ent.shape[0], dtype="int64")
        label_confi[idx_unconfi_list] = 0

        return pred_label.astype('int'), all_fea_F, label_confi, all_label, all_output



    def obtain_nearest_trace(self, data_q, data_all, lab_confi):
        data_q_ = data_q.detach()
#         data_all_ = data_all
        data_q_ = data_q_.cpu().numpy()
        data_all_ = data_all
        num_sam = data_q.shape[0]
        LN_MEM = 70

        flag_is_done = 0         # indicate whether the trace process has done over the target dataset 
        ctr_oper = 0             # counter the operation time
        idx_left = np.arange(0, num_sam, 1)
        mtx_mem_rlt = -3*np.ones((num_sam, LN_MEM), dtype='int64')
        mtx_mem_ignore = np.zeros((num_sam, LN_MEM), dtype='int64')
        is_mem = 0
        mtx_log = np.zeros((num_sam, LN_MEM), dtype='int64')
        indices_row = np.arange(0, num_sam, 1)
        flag_sw_bad = 0 
        nearest_idx_last = np.array([-7])

        while flag_is_done == 0:

            nearest_idx_tmp, idx_last_tmp = self.get_nearest_sam_idx(data_q_, data_all_, is_mem, ctr_oper, mtx_mem_ignore, nearest_idx_last)
            is_mem = 1
            nearest_idx_last = nearest_idx_tmp

            if ctr_oper == (LN_MEM-1):    
                flag_sw_bad = 1
            else:
                flag_sw_bad = 0 

            mtx_mem_rlt[:, ctr_oper] = nearest_idx_tmp
            mtx_mem_ignore[:, ctr_oper] = idx_last_tmp

            lab_confi_tmp = lab_confi[nearest_idx_tmp]
            idx_done_tmp = np.where(lab_confi_tmp == 1)[0]
            idx_left[idx_done_tmp] = -1

            if flag_sw_bad == 1:
                idx_bad = np.where(idx_left >= 0)[0]
                mtx_log[idx_bad, 0] = 1
            else:
                mtx_log[:, ctr_oper] = lab_confi_tmp

            flag_len = len(np.where(idx_left >= 0)[0])
            # print("{}--the number of left:{}".format(str(ctr_oper), flag_len))

            if flag_len == 0 or flag_sw_bad == 1:
                # idx_nn_tmp = [list(mtx_log[k, :]).index(1) for k in range(num_sam)]
                idx_nn_step = []
                for k in range(num_sam):
                    try:
                        idx_ts = list(mtx_log[k, :]).index(1)
                        idx_nn_step.append(idx_ts)
                    except:
#                         print("ts:", k, mtx_log[k, :])
                        # mtx_log[k, 0] = 1
                        idx_nn_step.append(0)

                idx_nn_re = mtx_mem_rlt[indices_row, idx_nn_step]
                data_re = data_all[idx_nn_re, :]
                flag_is_done = 1
            else:
                data_q_ = data_all_[nearest_idx_tmp, :]
            ctr_oper += 1

        return data_re, idx_nn_re, idx_nn_step # array



    def get_nearest_sam_idx(self, Q, X, is_mem_f, step_num, mtx_ignore, nearest_idx_last_f): # Q、X arranged in format of row-vector
        Xt = np.transpose(X)
        Simo = np.dot(Q, Xt)               
        nq = np.expand_dims(LA.norm(Q, axis=1), axis=1)
        nx = np.expand_dims(LA.norm(X, axis=1), axis=0)
        Nor = np.dot(nq, nx)
        Sim = 1 - (Simo / Nor)

        # Sim = cdist(Q, X, "cosine") # too slow
        # print('eeeeee \n', Sim)

        indices_min = np.argmin(Sim, axis=1)
        indices_row = np.arange(0, Q.shape[0], 1)

        idx_change = np.where((indices_min - nearest_idx_last_f)!=0)[0] 
        if is_mem_f == 1:
            if idx_change.shape[0] != 0:
                indices_min[idx_change] = nearest_idx_last_f[idx_change]  
        Sim[indices_row, indices_min] = 1000

        # mytst = np.eye(795)[indices_min]
        # mytst_log = np.sum(mytst, axis=0)
        # haha = np.where(mytst_log > 1)[0]
        # if haha.size != 0:
        #     print(haha)

        # Ignore the history elements. 
        if is_mem_f == 1:
            for k in range(step_num):
                indices_ingore = mtx_ignore[:, k]
                Sim[indices_row, indices_ingore] = 1000

        indices_min_cur = np.argmin(Sim, axis=1)
        indices_self = indices_min
        return indices_min_cur, indices_self
    
    def IID_loss(self, x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
        # has had softmax applied
        _, k = x_out.size()
        # p_i_j = compute_joint(x_out, x_tf_out)
        bn_, k_ = x_out.size()
        assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)
        su_temp1 = x_out.unsqueeze(2)
        su_temp2 = x_tf_out.unsqueeze(1)
        p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k #这两个相乘有什么用？
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise 为什么要对称
        p_i_j = p_i_j / p_i_j.sum()  # normalise
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                               k)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        p_i_j[(p_i_j < EPS).data] = EPS
        # p_j[(p_j < EPS).data] = EPS
        # p_i[(p_i < EPS).data] = EPS

        loss = - p_i_j * (torch.log(p_i_j) \
                        - lamb * torch.log(p_j) \
                        - lamb * torch.log(p_i))

        loss = loss.sum()

        # loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
        #                           - torch.log(p_j) \
        #                           - torch.log(p_i))

        # loss_no_lamb = loss_no_lamb.sum()

        return loss

class GKD(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device):
        super(GKD, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()
        
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        
        
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            # defining best and last model
            best_src_risk = float('inf')
            best_model = self.network.state_dict()
            last_model = self.network.state_dict()
            
            self.feature_extractor.eval()
            mem_label_soft, mtx_infor_nh, feas_FC = self.obtain_label(trg_dataloader, self.feature_extractor,  self.classifier)
            mem_label_soft = torch.from_numpy(mem_label_soft).cuda()
            feas_all = feas_FC[0]
            ops_all = feas_FC[1]
            self.feature_extractor.train()
            
            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                
                    
                    

                # Extract features
                features, _ = self.feature_extractor(trg_x)
                
                features_F_nh = self.get_mtx_sam_wgt_nh(feas_all, mtx_infor_nh, trg_idx)
                features_F_nh = features_F_nh.cuda()
                features_F_mix = 0.8*features + 0.2*features_F_nh
                outputs_test_mix = self.classifier(features_F_mix)
                
                log_probs = nn.LogSoftmax(dim=1)(outputs_test_mix)
                targets = mem_label_soft[trg_idx]
                loss_soft = (- targets * log_probs).sum(dim=1)
                classifier_loss = loss_soft.mean() 

                classifier_loss *= 0.3

                softmax_out = nn.Softmax(dim=1)(outputs_test_mix) # outputs_test_mix
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss
                
                im_loss = entropy_loss *1
                classifier_loss += im_loss

                self.optimizer.zero_grad()
                classifier_loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(classifier_loss.item(), 32)

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        return last_model, best_model
    
    def obtain_label(self, loader, netF, netC):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                
                inputs = inputs.cuda()
                feas_F,_ = netF(inputs)
                feas = feas_F
                outputs = netC(feas)
                if start_test:
                    all_fea_F = feas_F.float().cpu()
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    
                    start_test = False
                else:
                    all_fea_F = torch.cat((all_fea_F, feas_F.float().cpu()), 0)
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)          # 498*256
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0) # 498*31
                    

        all_output_C = all_output
        all_output = nn.Softmax(dim=1)(all_output)

        ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
        unknown_weight = 1 - ent / np.log(3)                       # class num
#         unknown_weight = 1 - ent / np.log(cfg.class_num)
        _, predict = torch.max(all_output, 1)

        
        
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        xx = np.eye(K)[predict]
        cls_count = xx.sum(axis=0)
        labelset = np.where(cls_count>0)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset],'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea) 
            initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
            dd = cdist(all_fea, initc[labelset],'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]

        feas_re = (all_fea_F, all_output_C)
        pred_label_new, mtx_idxnn, mtx_wts = self.obtain_label_nh(all_fea, pred_label, K)
        pred_label_re = pred_label_new
        mtx_re = [mtx_idxnn, mtx_wts]
          
        return pred_label_re.astype('int'), mtx_re, feas_re
    
    def obtain_label_nh(self, feas, label_old, Kf):
        num_nn_max = 7
        VAL_MIN = -1000
        BETA = np.array(range(num_nn_max)) + 1
        ln_sam = feas.shape[0]
        idx_row = np.array(range(ln_sam))
        dd_fea = np.dot(feas, feas.T)
        oh_final = np.zeros((feas.shape[0], Kf))

        log_idx = []
        val_dd = []
        for k in range(num_nn_max):
            idx_col_max_k = dd_fea.argmax(axis=1)
            log_idx.append(idx_col_max_k)
            val_dd_k = dd_fea[idx_row, idx_col_max_k]
            val_dd.append(val_dd_k)
            dd_fea[idx_row, idx_col_max_k] = BETA[k]*VAL_MIN
        val_dd_arr = np.vstack(tuple(val_dd)).T

        oh_all = []
        for k in range(num_nn_max):
            idx_col_max_k = log_idx[k]
            lab_k = label_old[idx_col_max_k]
            one_hot_k = np.eye(Kf)[lab_k]
            wts_k = val_dd_arr[:, k][:, None]
            one_hot_w_k = one_hot_k*wts_k
            oh_final = oh_final + one_hot_w_k
            oh_all.append(oh_final)

        num_nn = 5
        oh_final_slt = oh_all[num_nn - 1]
        mtx_idx = np.vstack(tuple(log_idx)).T
        mtx_idx_re = mtx_idx[:, 0:num_nn]
        val_dd_re = val_dd_arr[:, 0:num_nn]
        return oh_final_slt, mtx_idx_re, val_dd_re
    def get_mtx_sam_wgt_nh(self, fea_all_f, mtx_infor_nh_f, tar_idx_f):
        mtx_idx_nh = mtx_infor_nh_f[0]
        mtx_wts_nh = mtx_infor_nh_f[1]
        idx_batch = tar_idx_f.cpu().numpy()
        fea_all_f = fea_all_f.cpu().numpy()
        ln = len(idx_batch)
        sam_wgt_nh_list = []
        for k in range(ln):
            idx_k = idx_batch[k]
            idx_nh_k = mtx_idx_nh[idx_k, 1:]
            wts_nh_k = mtx_wts_nh[idx_k, 1:][:, None]
            wts_nh_k[0] = 0.5
            wts_nh_k[1] = (0.5)*(0.5)
            wts_nh_k[2] = (0.5)*(0.5)*(0.5)
            wts_nh_k[3] = (0.5)*(0.5)*(0.5)*(0.5)
            mtx_fea_k = fea_all_f[idx_nh_k]
            mtx_fea_wgt_k = mtx_fea_k*wts_nh_k 
            sam_wgt_nh_k = np.sum(mtx_fea_wgt_k, axis=0)
            sam_wgt_nh_list.append(sam_wgt_nh_k)
        mtx_sam_wgt_nh = np.vstack(tuple(sam_wgt_nh_list))
        mtx_sam_wgt_nh_re = torch.from_numpy(mtx_sam_wgt_nh)
        return mtx_sam_wgt_nh_re
