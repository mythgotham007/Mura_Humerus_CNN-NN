"""Function for fitting into a model"""        #approved
def train_nn(shallow_nn,xtr,ytr, batchsize, epchs, shffle):
    shallow_nn.fit(xtr, ytr, batch_size=batchsize, epochs=epchs, shuffle=shffle)
    return shallow_nn
