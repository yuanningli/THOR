import robosims
import cv2
import recog_stream


architecture = 'ResNet'
num_samples = 400
if architecture == 'ResNet':
    num_features = 2048
else:
    num_features = 4096

# initialize environment
env = robosims.controller.ChallengeController(
    unity_path='thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64',
    x_display="0.0" # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
)
env.start()

recog_net = recog_stream.RecogNet(architecture)

for target in t:
    # initialize
    env.initialize_target(target)
    # path to the target image (e.g. apple, lettuce, keychain, etc.)
    print(target['targetImage'])

    # convert target image
    target_im = cv2.imread("thor-challenge-targets/" + target['targetImage'])
    target_img = cv2.resize(target_im, (300, 300))
    target_feat = recog_net.feat_extract(target_img).squeeze()

    event = env.step(action=dict(action='MoveAhead'))
    print(event.metadata['lastActionSuccess'])
    frame = event.frame
    image_feature = recog_net.feat_extract(image).squeeze()

    step_count = 0
    while (not env.target_found()) and step_count < max_steps:
        # Possible actions are: MoveLeft, MoveRight, MoveAhead, MoveBack, LookUp, LookDown, RotateRight, RotateLeft
        # to plugin agent action here
        event = env.step(action=dict(action='MoveLeft'))

        # image of the current frame from the agent - numpy array of shape (300,300,3) in RGB order
        image = event.frame
        image_feature = recog_net.feat_extract(image).squeeze()

        # LookUp/Down beyond the allowed range.
        print(event.metadata['lastActionSuccess'])

