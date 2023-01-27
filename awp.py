class TransformersModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion = nn.__dict__[cfg["criterion"]]()

        # awp
        if cfg["awp"] is not None:
            self.automatic_optimization = False
            self.adv_param = cfg["awp"]["adv_param"]
            self.adv_lr = cfg["awp"]["adv_lr"]
            self.adv_eps = cfg["awp"]["adv_eps"]
            self.adv_step = cfg["awp"]["adv_step"]
            self.backup = {}
            self.backup_eps = {}
            self.awp_accumulate_grad_batches = cfg["awp"]["accumulate_grad_batches"]
            self.awp_start_epoch = cfg["awp"]["start_epoch"]

########################################################################################

def training_step(self, batch, batch_idx):
        X, y = batch
        X = self.collate(X)
        if self.cfg["awp"] is not None:
            # awp step
            opt = self.optimizers()
            sch = self.lr_schedulers()
            
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)

            if self.awp_accumulate_grad_batches > 1:
                loss = loss / self.awp_accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.awp_accumulate_grad_batches == 0:
                if self.trainer.current_epoch >= self.awp_start_epoch:
                    self._awp_save()
                    for _ in range(self.adv_step):
                        self._awp_attack_step()
                        pred_y = self.forward(X)
                        adv_loss = self.criterion(pred_y, y)
                        opt.zero_grad()
                        self.manual_backward(adv_loss)
                    self._awp_restore()
                    
                opt.step()
                opt.zero_grad()
                sch.step()
        else:
            # normal step
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
        
########################################################################################

def _awp_attack_step(self):
        e = 1e-6
        for name, param in self.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

def _awp_save(self):
        for name, param in self.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

def _awp_restore(self):
        for name, param in self.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
