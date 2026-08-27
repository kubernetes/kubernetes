import { App, Stack } from 'aws-cdk-lib';
import type { Construct } from 'constructs';
import { LinuxArmLambdaBuildImage, Project, BuildSpec, ComputeType } from 'aws-cdk-lib/aws-codebuild';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';

class LinuxArmLambdaImageTestStack extends Stack {
  public readonly project: Project;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.project = new Project(this, 'MyProject', {
      buildSpec: BuildSpec.fromObject({
        version: '0.2',
        phases: {
          build: {
            commands: ['ls'],
          },
        },
      }),
      environment: {
        computeType: ComputeType.LAMBDA_1GB,
        buildImage: LinuxArmLambdaBuildImage.AMAZON_LINUX_2023_NODE_24,
      },
    });
  }
}

const app = new App();

const codebuildLinuxLambda = new LinuxArmLambdaImageTestStack(app, 'codebuild-project-linux-arm-lambda');

const integ = new IntegTest(app, 'linux-arm-lambda-codebuild', {
  testCases: [codebuildLinuxLambda],
  stackUpdateWorkflow: true,
  // Lambda compute (ARM_LAMBDA_CONTAINER) is not available in all regions
  regions: ['us-east-1', 'us-east-2', 'us-west-2', 'ap-southeast-2', 'eu-central-1'],
});

// Execute the `startBuild` API to confirm that the build can be done correctly using Lambda compute.
integ.assertions.awsApiCall('CodeBuild', 'startBuild', {
  projectName: codebuildLinuxLambda.project.projectName,
}).waitForAssertions();
