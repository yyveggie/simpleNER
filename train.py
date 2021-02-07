import bilsm_crf_model
import process_data
import numpy as np

EPOCHS = 10
model, (train_x, train_y) = bilsm_crf_model.create_model()
model.fit(train_x, train_y, batch_size=64, epochs=EPOCHS)
model.save('model/crf.h5')
