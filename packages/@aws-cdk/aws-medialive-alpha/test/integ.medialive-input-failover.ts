// Tests automatic input failover with primary/secondary SRT inputs and rich audio/caption/video selectors.
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as medialive from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-medialive-input-failover');

const outputBucket = new s3.Bucket(stack, 'OutputBucket', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

// --- Primary and secondary inputs (same input class, required for failover) ---
const primaryInput = new medialive.Input(stack, 'PrimaryInput', {
  inputName: 'primary-srt',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.100',
    srtListenerPort: 5000,
  }]),
});

const secondaryInput = new medialive.Input(stack, 'SecondaryInput', {
  inputName: 'secondary-srt',
  input: medialive.InputConfiguration.srtCaller([{
    srtListenerAddress: '203.0.113.101',
    srtListenerPort: 5001,
  }]),
});

// --- Encodes ---
const video = medialive.EncodeConfiguration.video({
  name: 'video',
  width: 1920,
  height: 1080,
  codec: medialive.VideoCodecSettings.h264({
    rateControl: medialive.H264RateControl.cbr({ bitrate: cdk.Bitrate.mbps(5) }),
    framerate: medialive.Framerate.FPS_29_97,
  }),
});
const audio = medialive.EncodeConfiguration.audio({ name: 'audio', audioSelectorName: 'english', codec: medialive.AudioCodecSettings.aac() });

// Shared audio selector — referenced both on the attachment and by the audio-silence
// failover condition (type-safe linkage, no stringly-typed name).
const englishAudio = medialive.AudioSelector.byLanguage('english', 'eng', medialive.AudioLanguageSelectionPolicy.LOOSE);

const channel = new medialive.Channel(stack, 'Channel', {
  channelName: 'input-failover-channel',
  inputs: [{
    input: primaryInput,
    inputAttachmentName: 'primary',
    audioSelectors: [englishAudio],
    captionSelectors: [
      medialive.CaptionSelector.embedded('embedded', {
        convert608To708: medialive.Convert608To708.UPCONVERT,
      }),
    ],
    videoSelector: {
      colorSpace: medialive.VideoColorSpace.REC_709,
      selectBy: medialive.VideoSelection.byProgramId(1),
    },
    automaticInputFailover: {
      secondaryInput,
      inputPreference: medialive.InputPreference.PRIMARY_INPUT_PREFERRED,
      // Must be at least 2000 msec greater than the largest failover-condition threshold (2s here).
      errorClearTime: cdk.Duration.seconds(5),
      failoverConditions: [
        medialive.FailoverCondition.inputLoss({ threshold: cdk.Duration.millis(1500) }),
        medialive.FailoverCondition.audioSilence({ audioSelector: englishAudio, threshold: cdk.Duration.seconds(2) }),
        medialive.FailoverCondition.videoBlack({ blackDetectThreshold: 0.1, threshold: cdk.Duration.seconds(1) }),
      ],
    },
  }, {
    // The secondary input of a failover pair must also be attached to the channel, and MediaLive
    // requires identical audio-selector names and an equal caption-selector count across the pair.
    input: secondaryInput,
    inputAttachmentName: 'secondary',
    audioSelectors: [englishAudio],
    captionSelectors: [
      medialive.CaptionSelector.embedded('embedded', {
        convert608To708: medialive.Convert608To708.UPCONVERT,
      }),
    ],
  }],
  outputGroups: [
    medialive.OutputGroupConfiguration.hls({
      name: 'hls-to-s3',
      destinations: [medialive.OutputDestination.toBucket(outputBucket, 'live/stream')],
      segment: medialive.Segment.seconds(6),
      keepSegments: 10,
      outputs: [{ encodes: [video, audio], outputName: 'hls_output' }],
    }),
  ],
});

new cdk.CfnOutput(stack, 'ChannelArn', { value: channel.channelArn });

new IntegTest(app, 'cdk-integ-medialive-input-failover', {
  testCases: [stack],
});

app.synth();
