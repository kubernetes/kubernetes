/**
 * Integration test for Bedrock AgentCore Runtime observability with
 * manageDeliveryResourcePolicy disabled
 *
 * This test creates a runtime with observability enabled but opts out of
 * automatic resource policy creation. This is useful when deploying many
 * runtimes per account/region to avoid hitting resource policy quotas.
 */

/// !cdk-integ aws-cdk-bedrock-agentcore-runtime-observability-no-resource-policy

import * as path from 'path';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as agentcore from 'aws-cdk-lib/aws-bedrockagentcore';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-cdk-bedrock-agentcore-runtime-observability-no-resource-policy');

// Use fromAsset to build and push Docker image to ECR automatically
const runtimeArtifact = agentcore.AgentRuntimeArtifact.fromAsset(
  path.join(__dirname, 'testArtifact'),
  { platform: cdk.aws_ecr_assets.Platform.LINUX_ARM64 },
);

// Create log group for logging destination
const logGroup = new logs.LogGroup(stack, 'RuntimeLogGroup', {
  removalPolicy: cdk.RemovalPolicy.DESTROY,
});

// Runtime with observability but NO resource policy management
// This avoids consuming account-level quota slots for
// AWS::Logs::ResourcePolicy and AWS::XRay::ResourcePolicy
const runtime = new agentcore.Runtime(stack, 'ObservabilityRuntimeNoPolicy', {
  runtimeName: 'integ_observability_no_policy_runtime',
  description: 'Runtime with observability but no resource policy management',
  agentRuntimeArtifact: runtimeArtifact,
  protocolConfiguration: agentcore.ProtocolType.HTTP,
  tracingEnabled: true,
  loggingConfigs: [
    {
      logType: agentcore.LogType.APPLICATION_LOGS,
      destination: agentcore.LoggingDestination.cloudWatchLogs(logGroup),
    },
    {
      logType: agentcore.LogType.USAGE_LOGS,
      destination: agentcore.LoggingDestination.cloudWatchLogs(logGroup),
    },
  ],
  // Opt out of automatic resource policy creation
  manageDeliveryResourcePolicy: false,
  tags: {
    TestCase: 'ObservabilityNoResourcePolicy',
  },
});

// Output runtime information for verification
new cdk.CfnOutput(stack, 'RuntimeId', {
  value: runtime.agentRuntimeId,
  description: 'Runtime ID',
});

new cdk.CfnOutput(stack, 'RuntimeArn', {
  value: runtime.agentRuntimeArn,
  description: 'Runtime ARN',
});

new cdk.CfnOutput(stack, 'LogGroupArn', {
  value: logGroup.logGroupArn,
  description: 'Log group ARN',
});

new integ.IntegTest(app, 'BedrockAgentCoreRuntimeObservabilityNoResourcePolicyTest', {
  testCases: [stack],
});
