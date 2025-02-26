from blink import *
from controller import control
import time
# Load the data

mode = True # mode True means EEG-IO, otherwise v/r (EEG-VV or EEG-VR) data
data_path = 'EEG-IO' if mode else 'EEG-VV' # or replace w/ EEG-VR

# ---
    # Reading data files

list_of_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and '_data' in f]

file_idx = 0
file_sig = list_of_files[file_idx]
file_stim = list_of_files[file_idx].replace('_data','_labels')
print ("File Name: ", file_sig, file_stim)

# Loading Data
if mode:
    data_sig = np.loadtxt(open(os.path.join(data_path,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2))
else:
    data_sig = np.loadtxt(open(os.path.join(data_path,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2))
    data_sig = data_sig[0:(int(200*fs)+1),:]
    data_sig = data_sig[:,0:3]
    data_sig[:,0] = np.array(range(0,len(data_sig)))/fs

# Step1: Low Pass Filter
data_sig[:,1] = lowpass(data_sig[:,1], 10, fs, 4)
data_sig[:,2] = lowpass(data_sig[:,2], 10, fs, 4)

data_len = len(data_sig)

# Split the data into training and testing sets
split = data_len//5 * 4
data_train = data_sig[:split]
data_test = data_sig[split:]

time_train_min = data_train[0,0]
time_train_max = data_train[-1,0]

time_test_min = data_test[0,0]
time_test_max = data_test[-1,0]

# decoding stimulations
#interval_corrupt, gt_blinks = decode_stim(data_sig, file_stim) #look into

args_chan1 = args_init(delta_init)

running_std = compute_running_std(data_train, chan_id, fs)


#training the model

for idx in range(len(data_train[:,0])):
    peakdet(data_train[idx,0], data_train[idx, chan_id], args_chan1)

min_pts = np.array(args_chan1['mintab'])
p_blinks_t, p_blinks_val = find_expoints(min_pts, data_train, chan_id)
corr_matrix, pow_matrix = compute_correlation(p_blinks_t, data_train, chan_id, fs)

       
# fingerprint
blink_fp_idx = np.argmax(sum(corr_matrix))
t = corr_matrix[blink_fp_idx,:] > corr_threshold_1
blink_index = [i for i, x in enumerate(t) if x]

blink_template_corrmat = corr_matrix[np.ix_(blink_index,blink_index)]
blink_template_powmat = pow_matrix[np.ix_(blink_index,blink_index)]
blink_templates_corrWpower = blink_template_corrmat/blink_template_powmat

blink_var = []
for idx in blink_index:
    blink_var.append(np.var(data_train[int(fs*p_blinks_t[idx,0]):int(fs*p_blinks_t[idx,2]), chan_id]))

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

Z = linkage(blink_templates_corrWpower, 'complete', 'correlation')
groups = fcluster(Z,2,'maxclust')

grp_1_blinks_var = [blink_var[i] for i, x in enumerate(groups==1) if x]
grp_2_blinks_var = [blink_var[i] for i, x in enumerate(groups==2) if x]
if np.mean(grp_1_blinks_var) > np.mean(grp_2_blinks_var):
    selected_group = 1
else:
    selected_group = 2
template_blink_idx = [blink_index[i] for i, x in enumerate(groups==selected_group) if x]

# computing delta new
delta_new = 0
for idx in template_blink_idx:
    delta_new = delta_new + min(p_blinks_val[idx,0], p_blinks_val[idx,2]) - p_blinks_val[idx,1]
delta_new = delta_new/len(template_blink_idx)

args_chan1 = args_init(delta_new/3.0)

for idx in range(len(data_train[:,0])):
    peakdet(data_train[idx,0], data_train[idx, chan_id], args_chan1)

min_pts = np.array(args_chan1['mintab'])
p_blinks_t, p_blinks_val = find_expoints(min_pts, data_train, chan_id)
corr_matrix, pow_matrix = compute_correlation(p_blinks_t, data_train, chan_id, fs)

       
s_fc = (sum(corr_matrix))
sort_idx = sorted(range(len(s_fc)), key=lambda k: s_fc[k])

t = corr_matrix[sort_idx[-1],:] > corr_threshold_2        
blink_index1 = set([i for i, x in enumerate(t) if x])
t = corr_matrix[sort_idx[-2],:] > corr_threshold_2        
blink_index2 = set([i for i, x in enumerate(t) if x])
t = corr_matrix[sort_idx[-3],:] > corr_threshold_2        
blink_index3 = set([i for i, x in enumerate(t) if x])

blink_index = list(blink_index1.union(blink_index2).union(blink_index3))

blink_template_corrmat = corr_matrix[np.ix_(blink_index,blink_index)]
blink_template_powmat = pow_matrix[np.ix_(blink_index,blink_index)]
blink_templates_corrWpower = blink_template_corrmat/blink_template_powmat

blink_var = []
for idx in blink_index:
    blink_var.append(np.var(data_train[int(fs*p_blinks_t[idx,0]):int(fs*p_blinks_t[idx,2]), chan_id]))


Z = linkage(blink_templates_corrWpower, 'complete', 'correlation')
groups = fcluster(Z,2,'maxclust')

grp_1_blinks_var = [blink_var[i] for i, x in enumerate(groups==1) if x]
grp_2_blinks_var = [blink_var[i] for i, x in enumerate(groups==2) if x]


#selection algorithm for group1 vs group2
if np.mean(grp_1_blinks_var) > np.mean(grp_2_blinks_var) and np.mean(grp_1_blinks_var)/np.mean(grp_2_blinks_var) > 10:
    blink_index = [blink_index[i] for i, x in enumerate(groups==1) if x]
elif np.mean(grp_2_blinks_var) > np.mean(grp_1_blinks_var) and np.mean(grp_2_blinks_var)/np.mean(grp_1_blinks_var) > 10:
    blink_index = [blink_index[i] for i, x in enumerate(groups==2) if x]

final_blinks_t = p_blinks_t[blink_index,:]
final_blinks_val = p_blinks_val[blink_index,:]

# Define the queue here


#Test window
#doesn't account for corrupted intervals in the test data lol
for idx in range(0, len(data_test), 500):
    window = data_test[idx:idx+500, :]
    # ^^ this for example is the chunk of the data that we just read or received from the head set

    #define args
    args_test = args_init(delta_new/3.0) #delta_new is coming from the training of the data

    #run peakdet
    for idx in range(len(window[:,0])):
        foundPeak = peakdet(window[idx,0], window[idx, chan_id], args_test)
        if foundPeak:
            # potentially do more stuff to not classify noise 
                # min_pts_test = np.array(args_test['mintab'])
                # p_blinks_t, p_blinks_val = find_expoints(min_pts_test, window, chan_id)
            #do stuff
            print('Here')
            control('play_pause')
            time.sleep(2)