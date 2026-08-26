import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import { INTEG_TEST_LATEST_AURORA_MYSQL } from './db-versions';
import type { StackProps } from 'aws-cdk-lib';
import { App, CfnParameter } from 'aws-cdk-lib';
import { Vpc } from 'aws-cdk-lib/aws-ec2';
import * as rds from 'aws-cdk-lib/aws-rds';
import { ClusterInstance } from 'aws-cdk-lib/aws-rds';
import type { Construct } from 'constructs';
import { IntegTestBaseStack } from './integ-test-base-stack';

export class TestStack extends IntegTestBaseStack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const vpc = new Vpc(this, 'Integ-VPC');

    // serverlessV2 capacities supplied as deploy-time tokens (CfnParameter.valueAsNumber).
    // Before the fix these threw a synth-time ValidationError; they must now synthesize and deploy.
    const maxCapacity = new CfnParameter(this, 'MaxCapacity', { type: 'Number', default: 4 });
    const minCapacity = new CfnParameter(this, 'MinCapacity', { type: 'Number', default: 0.5 });

    new rds.DatabaseCluster(this, 'Integ-Cluster', {
      engine: rds.DatabaseClusterEngine.auroraMysql({ version: INTEG_TEST_LATEST_AURORA_MYSQL }),
      serverlessV2MaxCapacity: maxCapacity.valueAsNumber,
      serverlessV2MinCapacity: minCapacity.valueAsNumber,
      writer: ClusterInstance.serverlessV2('writer'),
      vpc: vpc,
    });
  }
}

const app = new App();
new IntegTest(app, 'integ-test-serverless-v2-capacity-token', {
  testCases: [new TestStack(app, 'integ-aurora-serverlessv2-cluster-capacity-token')],
});
