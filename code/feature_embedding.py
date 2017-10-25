import robosims
import json
import numpy as np
import recog_stream
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm


architecture = 'VGG'
num_samples = 400
if architecture == 'ResNet':
    num_features = 2048
else:
    num_features = 4096

# initialize environment
env = robosims.controller.ChallengeController(
    # Use unity_path=thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64 for OSX
    unity_path='thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64',
    x_display="0.0", # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
    mode='continuous'
)
env.start()
with open("thor-challenge-targets/targets-train.json") as f:
    t = json.loads(f.read())
env.initialize_target(t[0])
event = env.step(action=dict(action='Look', horizon=0.0))
event = env.step(action=dict(action='Rotate', rotation=180.0))
event = env.step(action=dict(action='MoveBack', moveMagnitude=6))
event = env.step(action=dict(action='MoveRight', moveMagnitude=0.3))
event = env.step(action=dict(action='MoveAhead', moveMagnitude=0.1))
counter = 0

# initialize network
recog_net = recog_stream.RecogNet(architecture)
image_feature = np.zeros((num_samples, num_features))

# extract features
counter = 0
for i in range(100):
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=(3.5/100)))
    frame = event.frame
    image_feat = recog_net.feat_extract(frame)
    image_feature[counter, :] = image_feat.data.squeeze().unsqueeze_(0).numpy()
    counter += 1
event = env.step(action=dict(action='RotateRight', rotation=90.0))
for i in range(100):
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=(2.7/100)))
    frame = event.frame
    image_feat = recog_net.feat_extract(frame)
    image_feature[counter, :] = image_feat.data.squeeze().unsqueeze_(0).numpy()
    counter += 1
event = env.step(action=dict(action='RotateRight', rotation=90.0))
for i in range(100):
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=(3.5/100)))
    frame = event.frame
    image_feat = recog_net.feat_extract(frame)
    image_feature[counter, :] = image_feat.data.squeeze().unsqueeze_(0).numpy()
    counter += 1
event = env.step(action=dict(action='RotateRight', rotation=90.0))
for i in range(100):
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=(2.7/100)))
    frame = event.frame
    image_feat = recog_net.feat_extract(frame)
    image_feature[counter, :] = image_feat.data.squeeze().unsqueeze_(0).numpy()
    counter += 1
event = env.step(action=dict(action='RotateRight', rotation=90.0))

# low dimensional embedding
svd = TruncatedSVD(n_components=50, n_iter=15)
image_feature_SVD = svd.fit_transform(image_feature)
image_embedded = TSNE(n_components=2).fit_transform(image_feature_SVD)

# plot embeddings
ff1 = plt.figure(figsize=(8, 8))
counter = 0
for i in range(100):
    plt.scatter(image_embedded[counter,0], image_embedded[counter,1],c=cm.Purples_r(i / 100.0))
    counter += 1
for i in range(100):
    plt.scatter(image_embedded[counter,0], image_embedded[counter,1],c=cm.Greens_r(i / 100.0))
    counter += 1
for i in range(100):
    plt.scatter(image_embedded[counter,0], image_embedded[counter,1],c=cm.Blues_r(i / 100.0))
    counter += 1
for i in range(100):
    plt.scatter(image_embedded[counter,0], image_embedded[counter,1],c=cm.Oranges_r(i / 100.0))
    counter += 1
plt.savefig('embedding3_VGG.png')
plt.close(ff1)

