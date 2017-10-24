import robosims
import json
import numpy as np
import recog_stream
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

architecture = 'ResNet'
num_samples = 135
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

# initialize network
recog_net = recog_stream.RecogNet(architecture)
image_feature = np.zeros(num_samples, num_features)

# extract features
for i in range(num_samples):
    target = t[i]
    env.initialize_target(target)
    event = env.step(action=dict(action='MoveAhead', moveMagnitude=0.00))
    frame = event.frame
    image_feat = recog_net.feat_extract(frame)
    image_feature[i, :] = image_feat.data.squeeze().unsqueeze_(0).numpy()

# low dimensional embedding
svd = TruncatedSVD(n_components=50, n_iter=7)
image_feature_SVD = svd.fit_transform(image_feature)
image_embedded = TSNE(n_components=2).fit_transform(image_feature_SVD)

# low dimensional embedding
svd = TruncatedSVD(n_components=50, n_iter=7)
image_feature_SVD = svd.fit_transform(image_feature)
image_embedded = TSNE(n_components=2).fit_transform(image_feature_SVD)
