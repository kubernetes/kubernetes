import * as cdk from 'aws-cdk-lib';
import { IntegTest } from '@aws-cdk/integ-tests-alpha';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';

import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';

/// !cdk-integ aws-StepFunctionsTasks-rest-api-JSONataExpression

const app = new cdk.App();
const stack = new cdk.Stack(app,
  'aws-StepFunctionsTasks-rest-api-JSONataExpression',
);

const restApi = new apigateway.RestApi(stack, 'example-stepfunctions-rest-api');
restApi.root.addMethod('GET');

const state = new tasks.CallApiGatewayRestApiEndpoint(stack, 'StateExample', {
  api: restApi,
  stageName: 'prod',
  method: tasks.HttpMethod.GET,
  apiPath: '{% "/path/" & $states.input.path_suffix %}',
  authType: tasks.AuthType.IAM_ROLE,
  taskTimeout: sfn.Timeout.duration(cdk.Duration.seconds(30)),
  queryLanguage: sfn.QueryLanguage.JSONATA,
});

new sfn.StateMachine(stack, 'StateMachine', {
  definitionBody: sfn.DefinitionBody.fromChainable(state),
  queryLanguage: sfn.QueryLanguage.JSONATA,
});

new IntegTest(app, 'StateMachineDefaultRoleTrust', { testCases: [stack] });

app.synth();
