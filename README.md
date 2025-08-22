AUG 21:

- Resnet with decoder only model, adapted from Illia's work is working well. 
- Loss is reduced to 0.5 less. 
- Key issue was the normalization of the action values which was used for loss computation. 
- Now loss is computed as mse(loss, loss_act) both of which is normalized. Finally actions are unnormalized before executing. 
- Videos are not being recorded on evaluation. This needs to be fixed. 
- Running on different data sources other than PushT env (some sort of benchmarking).
- Documentation and clean up