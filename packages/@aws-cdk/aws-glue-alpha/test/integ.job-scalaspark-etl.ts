import * as path from 'path';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as glue from '../lib';

/**
 * To verify the ability to run jobs created in this test
 *
 * Run the job using
 *   `aws glue start-job-run --region us-east-1 --job-name <job name>`
 * This will return a runId
 *
 * Get the status of the job run using
 *   `aws glue get-job-run --region us-east-1 --job-name <job name> --run-id <runId>`
 *
 * For example, to test the ETLJob
 * - Run: `aws glue start-job-run --region us-east-1 --job-name ETLJob`
 * - Get Status: `aws glue get-job-run --region us-east-1 --job-name ETLJob --run-id <runId output by the above command>`
 * - Check output: `aws logs get-log-events --region us-east-1 --log-group-name "/aws-glue/python-jobs/output" --log-stream-name "<runId>>` which should show "hello world"
 */

const app = new cdk.App();
const stack = new cdk.Stack(app, 'aws-glue-job-scalaspark-etl');

const jar_file = glue.Code.fromAsset(path.join(__dirname, 'job-jar', 'helloworld.jar'));
const job_class ='com.example.HelloWorld';

const iam_role = new iam.Role(stack, 'IAMServiceRole', {
  assumedBy: new iam.ServicePrincipal('glue.amazonaws.com'),
  managedPolicies: [iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSGlueServiceRole')],
});

new glue.ScalaSparkEtlJob(stack, 'BasicScalaSparkETLJob', {
  script: jar_file,
  role: iam_role,
  className: job_class,
});

new glue.ScalaSparkEtlJob(stack, 'OverrideScalaSparkETLJob', {
  script: jar_file,
  className: job_class,
  role: iam_role,
  description: 'Optional Override ScalaSpark ETL Job',
  glueVersion: glue.GlueVersion.V3_0,
  workerConfiguration: { workerType: glue.WorkerType.G_1X, numberOfWorkers: 20 },
  timeout: cdk.Duration.minutes(15),
  jobName: 'Optional Override ScalaSpark ETL Job',
  defaultArguments: {
    arg1: 'value1',
    arg2: 'value2',
  },
  tags: {
    key: 'value',
  },
  jobRunQueuingEnabled: true,
});

new integ.IntegTest(app, 'aws-glue-job-scalaspark-etl-integ-test', {
  testCases: [stack],
});

app.synth();
