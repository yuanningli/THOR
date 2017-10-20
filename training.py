#!/usr/bin/env python
import robosims
import json
import recog_stream

env = robosims.controller.ChallengeController(
    # Use unity_path=thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64 for OSX
    # unity_path='projects/thor-201705011400-Linux64',
    unity_path='thor-201705011400-OSXIntel64.app/Contents/MacOS/thor-201705011400-OSXIntel64',
    x_display="0.0",  # this parameter is ignored on OSX, but you must set this to the appropriate display on Linux
    mode='continuous'
)
env.start()
recog_net = recog_stream.RecogNet()
max_steps = 500
with open("thor-challenge-targets/targets-train.json") as f:
    t = json.loads(f.read())
    for target in t:
        # initialize
        env.initialize_target(target)
        # path to the target image (e.g. apple, lettuce, keychain, etc.)
        print(target['targetImage'])
        event = env.step(action=dict(action='Look', horizon=0.0))
        print(event.metadata['lastActionSuccess'])
        frame = event.frame

        step_count = 0
        while (not env.target_found()) and step_count < max_steps:
            # Possible actions are: MoveLeft, MoveRight, MoveAhead, MoveBack, LookUp, LookDown, RotateRight, RotateLeft
            # to be plugin agent action here
            event = env.step(action=dict(action='MoveLeft', moveMagnitude=0.25))

            # image of the current frame from the agent - numpy array of shape (300,300,3) in RGB order
            image = event.frame
            image_feature = recog_net.feat_extract(image)
            
            # LookUp/Down beyond the allowed range.
            print(event.metadata['lastActionSuccess'])

