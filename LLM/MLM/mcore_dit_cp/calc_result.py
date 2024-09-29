a = 'Epoch 0: :   1%|▏                 | 57/4900 [10:58<15:32:28, v_num=qhyq, reduced_train_loss=6.830, global_step=56.00, consumed_samples=7296.0, train_step_timing in s=11.30]'
b = 'Epoch 0: :   2%|▍               | 116/4900 [21:47<14:58:56, v_num=qhyq, reduced_train_loss=5.340, global_step=115.0, consumed_samples=14848.0, train_step_timing in s=11.20]'

a = '57/4900 [10:58<15:32:28'
b = '116/4900 [21:47<14:58:56'

# return consumed_samples, time
def calc_consumed(a):
    iter_num = int(a.split('/')[0])
    time_beg = a.split('<')[0].split('[')[-1]
    ss = time_beg.split(':')
    time_elapsed = 0
    if len(ss) == 2:
        time_elapsed = int(ss[0])*60 + int(ss[1])
    elif len(ss) == 3:
        time_elapsed = int(ss[0])*60*60 + int(ss[1])*60 + int(ss[2])
        
    consumed_samples = iter_num * time_elapsed
    
    return consumed_samples, time_elapsed

def calc_throughput(a, b):
    c1, t1 = calc_consumed(a)
    c2, t2 = calc_consumed(b)
    return (c2-c1)/(t2-t1)
