import { App, Lazy, Stack } from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import * as mediapackagev2 from '../lib';

let app: App;
let stack: Stack;
beforeEach(() => {
  app = new App();
  stack = new Stack(app, undefined, {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

test('MediaPackagev2 Channel Group Configuration', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
    channelGroupName: 'test',
    description: 'my test channel',
    tags: {
      env: 'dev',
    },
  });

  new mediapackagev2.Channel(stack, 'myChannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
    channelName: 'my-test-channel',
    description: 'a channel test',
    tags: {
      env: 'dev',
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::ChannelGroup', {
    ChannelGroupName: 'test',
    Description: 'my test channel',
    Tags: [{
      Key: 'env',
      Value: 'dev',
    }],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::Channel', {
    ChannelGroupName: 'test',
    ChannelName: 'my-test-channel',
    Description: 'a channel test',
    Tags: [{
      Key: 'env',
      Value: 'dev',
    }],
  });
});

test('MediaPackagev2 Channel Configuration - no names', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');

  new mediapackagev2.Channel(stack, 'mychannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::Channel', {
    ChannelName: 'mychannel',
  });
});

test('MediaPackagev2 Channel Configuration - policy test', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');
  new mediapackagev2.Channel(stack, 'mychannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::Channel', {
    ChannelName: 'mychannel',
  });
});

test('existing Channel can be imported', () => {
  const importedChannel = mediapackagev2.Channel.fromChannelAttributes(stack, 'ImportedChannel', {
    channelName: 'test',
    channelGroupName: 'MyChannelGroup',
  });

  expect(importedChannel.channelArn).toMatch(/^arn:.*:mediapackagev2:us-east-1:123456789012:channelGroup\/MyChannelGroup\/channel\/test$/);
});

test('imported Channel with cross-region override', () => {
  const importedChannel = mediapackagev2.Channel.fromChannelAttributes(stack, 'ImportedChannel', {
    channelName: 'test',
    channelGroupName: 'MyChannelGroup',
    region: 'eu-west-1',
  });

  expect(importedChannel.channelArn).toMatch(/^arn:.*:mediapackagev2:eu-west-1:123456789012:channelGroup\/MyChannelGroup\/channel\/test$/);
});

test('Channel can be imported from ARN', () => {
  const importedChannel = mediapackagev2.Channel.fromChannelArn(stack, 'ImportedChannel', 'arn:aws:mediapackagev2:eu-west-1:123456789012:channelGroup/MyGroup/channel/MyChannel');

  expect(importedChannel.channelGroupName).toBe('MyGroup');
  expect(importedChannel.channelName).toBe('MyChannel');
  expect(importedChannel.channelArn).toMatch(/mediapackagev2:eu-west-1:123456789012/);
  expect(importedChannel.region).toBe('eu-west-1');
});

test('Channel.fromChannelArn throws on invalid ARN', () => {
  expect(() => {
    mediapackagev2.Channel.fromChannelArn(stack, 'Bad', 'arn:aws:mediapackagev2:us-east-1:123456789012:channelGroup/MyGroup');
  }).toThrow(/Could not parse channel ARN/);
});

test('Channel.fromChannelArn throws on token ARN', () => {
  expect(() => {
    mediapackagev2.Channel.fromChannelArn(stack, 'TokenArn', Lazy.string({ produce: () => 'arn:aws:mediapackagev2:us-east-1:123456789012:channelGroup/G/channel/C' }));
  }).toThrow(/Cannot parse a token ARN/);
});

test('Channel has accessible ingest URLs - Tokens returned in Array', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
    channelGroupName: 'test',
    description: 'my test channel',
    tags: {
      env: 'dev',
    },
  });
  const channel = new mediapackagev2.Channel(stack, 'myChannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
    channelName: 'my-test-channel',
    description: 'a channel test',
    tags: {
      env: 'dev',
    },
  });

  expect(channel.ingestEndpointUrls[0]).toMatch(/\${Token\[TOKEN\.[0-9]*\]}/);
  expect(channel.ingestEndpointUrls[1]).toMatch(/\${Token\[TOKEN\.[0-9]*\]}/);
});

test('Grant write to MediaLive Role created seperately', () => {
  const role = new Role(stack, 'MediaLiveRole', {
    assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
  });

  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');
  const channel = new mediapackagev2.Channel(stack, 'mychannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
  });
  channel.grants.ingest(role);

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: [
          'mediapackagev2:GetChannel',
          'mediapackagev2:PutObject',
        ],
        Effect: 'Allow',
        Resource: {
          'Fn::GetAtt': ['mychannel130AE695', 'Arn'],
        },
      }],
    },
  });
});

test('MediaPackagev2 Channel Group CMSD Header Config', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
    channelGroupName: 'test',
    description: 'my test channel',
    tags: {
      env: 'dev',
    },
  });

  new mediapackagev2.Channel(stack, 'myChannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf({
      outputHeaders: [mediapackagev2.HeadersCMSD.MQCS],
    }),
    channelName: 'my-test-channel',
    description: 'a channel test',
    tags: {
      env: 'dev',
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::ChannelGroup', {
    ChannelGroupName: 'test',
    Description: 'my test channel',
    Tags: [{
      Key: 'env',
      Value: 'dev',
    }],
  });

  Template.fromStack(stack).hasResourceProperties('AWS::MediaPackageV2::Channel', {
    ChannelGroupName: 'test',
    ChannelName: 'my-test-channel',
    Description: 'a channel test',
    Tags: [{
      Key: 'env',
      Value: 'dev',
    }],
  });
});

test('Channel name validation - too short', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');

  expect(() => {
    new mediapackagev2.Channel(stack, 'myChannel', {
      channelGroup: group,
      channelName: '',
      input: mediapackagev2.InputConfiguration.hls(),
    });
  }).toThrow('Channel name must be between 1 and 256 characters in length.');
});

test('Channel name validation - too long', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');

  expect(() => {
    new mediapackagev2.Channel(stack, 'myChannel', {
      channelGroup: group,
      channelName: 'a'.repeat(257),
      input: mediapackagev2.InputConfiguration.hls(),
    });
  }).toThrow('Channel name must be between 1 and 256 characters in length.');
});

test('Channel name validation - invalid characters', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');

  expect(() => {
    new mediapackagev2.Channel(stack, 'myChannel', {
      channelGroup: group,
      channelName: 'invalid@name',
      input: mediapackagev2.InputConfiguration.hls(),
    });
  }).toThrow('Channel name must only contain alphanumeric characters, hyphens, and underscores.');
});

test('Channel description validation - too long', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');

  expect(() => {
    new mediapackagev2.Channel(stack, 'myChannel', {
      channelGroup: group,
      description: 'a'.repeat(1025),
      input: mediapackagev2.InputConfiguration.hls(),
    });
  }).toThrow('Channel description must not exceed 1024 characters.');
});

test('Channel Group name validation - too short', () => {
  expect(() => {
    new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
      channelGroupName: '',
    });
  }).toThrow('Channel group name must be between 1 and 256 characters in length.');
});

test('Channel Group name validation - too long', () => {
  expect(() => {
    new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
      channelGroupName: 'a'.repeat(257),
    });
  }).toThrow('Channel group name must be between 1 and 256 characters in length.');
});

test('Channel Group name validation - invalid characters', () => {
  expect(() => {
    new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
      channelGroupName: 'invalid@name',
    });
  }).toThrow('Channel group name must only contain alphanumeric characters, hyphens, and underscores.');
});

test('Channel Group description validation - too long', () => {
  expect(() => {
    new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
      description: 'a'.repeat(1025),
    });
  }).toThrow('Channel group description must not exceed 1024 characters.');
});

test('channel metrics', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup', {
    channelGroupName: 'test-group',
  });
  const channel = new mediapackagev2.Channel(stack, 'myChannel', {
    channelGroup: group,
    channelName: 'test-channel',
    input: mediapackagev2.InputConfiguration.cmaf(),
  });

  expect(stack.resolve(channel.metricIngressBytes())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'IngressBytes',
    statistic: 'Sum',
    unit: 'Bytes',
  }));

  expect(stack.resolve(channel.metricEgressBytes())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'EgressBytes',
    statistic: 'Sum',
    unit: 'Bytes',
  }));

  expect(stack.resolve(channel.metricIngressResponseTime())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'IngressResponseTime',
    statistic: 'Average',
    unit: 'Milliseconds',
  }));

  expect(stack.resolve(channel.metricEgressResponseTime())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'EgressResponseTime',
    statistic: 'Average',
    unit: 'Milliseconds',
  }));

  expect(stack.resolve(channel.metricIngressRequestCount())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'IngressRequestCount',
    statistic: 'Sum',
    unit: 'Count',
  }));

  expect(stack.resolve(channel.metricEgressRequestCount())).toEqual(expect.objectContaining({
    dimensions: { ChannelGroup: 'test-group', Channel: 'test-channel' },
    namespace: 'AWS/MediaPackage',
    metricName: 'EgressRequestCount',
    statistic: 'Sum',
    unit: 'Count',
  }));
});

test('Token channel name skips validation', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');
  // Should not throw even though the token string doesn't match the pattern
  expect(() => {
    new mediapackagev2.Channel(stack, 'mychannel', {
      channelGroup: group,
      channelName: Lazy.string({ produce: () => 'resolved-later' }),
    });
  }).not.toThrow();
});

test('grants.ingest() adds correct IAM policy', () => {
  const role = new Role(stack, 'MediaLiveRole', {
    assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
  });

  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');
  const channel = new mediapackagev2.Channel(stack, 'mychannel', {
    channelGroup: group,
  });
  channel.grants.ingest(role);

  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: [
          'mediapackagev2:GetChannel',
          'mediapackagev2:PutObject',
        ],
        Effect: 'Allow',
        Resource: {
          'Fn::GetAtt': ['mychannel130AE695', 'Arn'],
        },
      }],
    },
  });
});

test('imported channel has undefined for createdAt and modifiedAt, throws for ingestEndpointUrls', () => {
  const imported = mediapackagev2.Channel.fromChannelAttributes(stack, 'ImportedChannel2', {
    channelName: 'test',
    channelGroupName: 'MyChannelGroup',
  });
  expect(imported.createdAt).toBeUndefined();
  expect(imported.modifiedAt).toBeUndefined();
  expect(() => imported.ingestEndpointUrls).toThrow(/ingestEndpointUrls.*is not available/);
});

test('Channel exposes region from Stack', () => {
  const group = new mediapackagev2.ChannelGroup(stack, 'MyChannelGroup');
  const channel = new mediapackagev2.Channel(stack, 'myChannel', {
    channelGroup: group,
    input: mediapackagev2.InputConfiguration.cmaf(),
  });

  expect(channel.region).toBe('us-east-1');
});

test('imported channel region falls back to importing stack region', () => {
  const imported = mediapackagev2.Channel.fromChannelAttributes(stack, 'ImportedChannel3', {
    channelName: 'test',
    channelGroupName: 'MyChannelGroup',
  });

  expect(imported.region).toBe('us-east-1');
});

test('imported channel uses explicit region from attributes', () => {
  const imported = mediapackagev2.Channel.fromChannelAttributes(stack, 'ImportedChannel4', {
    channelName: 'test',
    channelGroupName: 'MyChannelGroup',
    region: 'eu-west-1',
  });

  expect(imported.region).toBe('eu-west-1');
});
