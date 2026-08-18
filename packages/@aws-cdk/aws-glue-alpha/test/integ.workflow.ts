import * as path from 'path';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as glue from '../lib';

const app = new cdk.App();
const stack = new cdk.Stack(app, 'GlueWorkflowTriggerStack');

const workflow = new glue.Workflow(stack, 'Workflow', {
  description: 'MyWorkflow',
});

const role = new iam.Role(stack, 'JobRole', {
  assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
});

const script = glue.Code.fromAsset(path.join(__dirname, 'job-script', 'hello_world.py'));

const OutboundJob = new glue.PySparkEtlJob(stack, 'OutboundJob', {
  script: script,
  role,
  glueVersion: glue.GlueVersion.V4_0,
  workerConfiguration: { workerType: glue.WorkerType.G_2X, numberOfWorkers: 2 },
});

const InboundJob = new glue.PySparkEtlJob(stack, 'InboundJob', {
  script: script,
  role,
  glueVersion: glue.GlueVersion.V4_0,
  workerConfiguration: { workerType: glue.WorkerType.G_2X, numberOfWorkers: 2 },
});

workflow.addOnDemandTrigger('OnDemandTrigger', {
  actions: [{ job: InboundJob }],
});

workflow.addConditionalTrigger('ConditionalTrigger', {
  actions: [{ job: OutboundJob }],
  predicate: {
    conditions: [
      {
        job: InboundJob,
        state: glue.JobState.SUCCEEDED,
      },
    ],
  },
});

new cdk.CfnOutput(stack, 'WorkflowName', {
  value: workflow.workflowName,
});

new integ.IntegTest(app, 'aws-cdk-glue-workflow-trigger-integ', {
  testCases: [stack],
});

app.synth();
