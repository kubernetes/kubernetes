/*
 * Integration test for Bedrock Agent Core Memory with Stream Delivery
 *
 * Covers both content levels, since the level is what determines whether memory
 * record bodies leave the service: FULL_CONTENT on one memory, METADATA_ONLY on
 * another. Each memory gets its own Kinesis stream so the rendered
 * StreamDeliveryResources of the two are independently verifiable.
 */

/// !cdk-integ aws-cdk-agentcore-memory-stream-delivery

import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as kinesis from 'aws-cdk-lib/aws-kinesis';
import * as agentcore from 'aws-cdk-lib/aws-bedrockagentcore';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'aws-cdk-agentcore-memory-stream-delivery');

// Create a Kinesis Data Stream for memory event delivery
const memoryEventStream = new kinesis.Stream(stack, 'MemoryEventStream', {
  encryption: kinesis.StreamEncryption.MANAGED,
});

// A memory that streams complete memory record bodies
const fullContentMemory = new agentcore.Memory(stack, 'MemoryWithStreamDelivery', {
  memoryName: 'memory_with_stream_delivery',
  description: 'A test memory with Kinesis stream delivery',
  expirationDuration: cdk.Duration.days(90),
  streamDeliveryResources: [
    agentcore.StreamDeliveryResource.kinesis(memoryEventStream, {
      contentConfigurations: [
        {
          type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
          level: agentcore.StreamDeliveryContentLevel.FULL_CONTENT,
        },
      ],
    }),
  ],
});

// A memory that streams only event metadata, on its own stream
const metadataEventStream = new kinesis.Stream(stack, 'MetadataEventStream', {
  encryption: kinesis.StreamEncryption.MANAGED,
});

const metadataOnlyMemory = new agentcore.Memory(stack, 'MemoryWithMetadataOnlyStreamDelivery', {
  memoryName: 'memory_with_metadata_only_stream_delivery',
  description: 'A test memory delivering only memory record metadata',
  expirationDuration: cdk.Duration.days(90),
  streamDeliveryResources: [
    agentcore.StreamDeliveryResource.kinesis(metadataEventStream, {
      contentConfigurations: [
        {
          type: agentcore.StreamDeliveryContentType.MEMORY_RECORDS,
          level: agentcore.StreamDeliveryContentLevel.METADATA_ONLY,
        },
      ],
    }),
  ],
});

const test = new integ.IntegTest(app, 'MemoryStreamDelivery', {
  testCases: [stack],
  regions: ['us-east-1', 'us-east-2', 'us-west-2', 'ca-central-1', 'eu-central-1', 'eu-north-1', 'eu-west-1', 'eu-west-2', 'eu-west-3', 'ap-northeast-1', 'ap-northeast-2', 'ap-south-1', 'ap-southeast-1', 'ap-southeast-2'],
});

// Verify each deployed memory reports back the content level it was configured
// with, so a silently dropped level fails the test instead of passing unnoticed.
const fullContentCall = test.assertions.awsApiCall('bedrock-agentcore-control', 'getMemory', {
  memoryId: fullContentMemory.memoryId,
}).expect(integ.ExpectedResult.objectLike({
  memory: {
    streamDeliveryResources: {
      resources: [
        {
          kinesis: {
            contentConfigurations: [
              { type: 'MEMORY_RECORDS', level: 'FULL_CONTENT' },
            ],
          },
        },
      ],
    },
  },
}));

const metadataOnlyCall = test.assertions.awsApiCall('bedrock-agentcore-control', 'getMemory', {
  memoryId: metadataOnlyMemory.memoryId,
}).expect(integ.ExpectedResult.objectLike({
  memory: {
    streamDeliveryResources: {
      resources: [
        {
          kinesis: {
            contentConfigurations: [
              { type: 'MEMORY_RECORDS', level: 'METADATA_ONLY' },
            ],
          },
        },
      ],
    },
  },
}));

// The IAM action prefix (bedrock-agentcore) differs from the SDK client name
// (bedrock-agentcore-control), so grant the assertion provider explicitly
// rather than relying on the action being derived from the service name.
for (const call of [fullContentCall, metadataOnlyCall]) {
  call.provider.addToRolePolicy({
    Effect: 'Allow',
    Action: ['bedrock-agentcore:GetMemory'],
    Resource: ['*'],
  });
}
