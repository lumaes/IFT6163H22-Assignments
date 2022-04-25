import glob
import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':
    import glob

    question_number = 7

    for i in range(1, question_number+1):
        logdir = './results/q'+str(i)+'/*/data/hw3_*/events*'
        count = 0
        for eventfile in sorted(glob.glob(logdir)):
            print(eventfile)
            count += 1
            X, Y, Z = get_section_results(eventfile)
            print(len(X), len(Y))
            data = {'X': X, 'Y': Y, 'Z': Z}
            import json
            with open('./data/q_'+str(i)+'_'+str(count)+'_data.json', 'w') as file:
                file.write(json.dumps(data))
