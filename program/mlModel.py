from sklearn.externals import joblib
import os
import time 
import values as val

def LoadTrainModel():
    os.chdir(val.modelPath)
    print 'Loading ML Model...'
    t1 = time.time()
    model=joblib.load(val.modelName)
    print'ML Model Loaded ,Time Taken: ' + str(time.time() - t1) + ' sec'
    print'ML Model Name: ' + str(val.modelName)
    modelSize = os.stat(val.modelName).st_size / (1024 * 1024)
    print'ML Model size: ' + str(modelSize) + ' MB'
    return model
