import * as apigateway from '../../../aws-apigateway';
import * as sfn from '../../../aws-stepfunctions';
import * as cdk from '../../../core';
import { HttpMethod, CallApiGatewayRestApiEndpoint } from '../../lib';

describe('CallApiGatewayRestApiEndpoint', () => {
  test('default', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // WHEN
    const task = new CallApiGatewayRestApiEndpoint(stack, 'Call', {
      api: restApi,
      method: HttpMethod.GET,
      stageName: 'dev',
    });

    // THEN
    expect(stack.resolve(task.toStateJson())).toEqual({
      Type: 'Task',
      End: true,
      Parameters: {
        ApiEndpoint: {
          'Fn::Join': [
            '',
            [
              {
                Ref: 'RestApi0C43BF4B',
              },
              '.execute-api.',
              {
                Ref: 'AWS::Region',
              },
              '.',
              {
                Ref: 'AWS::URLSuffix',
              },
            ],
          ],
        },
        AuthType: 'NO_AUTH',
        Method: 'GET',
        Stage: 'dev',
      },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':states:::apigateway:invoke',
          ],
        ],
      },
    });
  });

  test('provide region override', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');
    const region = 'us-west-2';

    // WHEN
    const task = new CallApiGatewayRestApiEndpoint(stack, 'Call', {
      api: restApi,
      method: HttpMethod.GET,
      stageName: 'dev',
      region: region,
    });

    // THEN
    expect(stack.resolve(task.toStateJson())).toEqual({
      Type: 'Task',
      End: true,
      Parameters: {
        ApiEndpoint: {
          'Fn::Join': [
            '',
            [
              {
                Ref: 'RestApi0C43BF4B',
              },
              `.execute-api.${region}.`,
              {
                Ref: 'AWS::URLSuffix',
              },
            ],
          ],
        },
        AuthType: 'NO_AUTH',
        Method: 'GET',
        Stage: 'dev',
      },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':states:::apigateway:invoke',
          ],
        ],
      },
    });
  });

  test('wait for task token', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // WHEN
    const task = new CallApiGatewayRestApiEndpoint(stack, 'Call', {
      api: restApi,
      method: HttpMethod.GET,
      stageName: 'dev',
      integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
      headers: sfn.TaskInput.fromObject({ TaskToken: sfn.JsonPath.array(sfn.JsonPath.taskToken) }),
    });

    // THEN
    expect(stack.resolve(task.toStateJson())).toEqual({
      Type: 'Task',
      End: true,
      Parameters: {
        ApiEndpoint: {
          'Fn::Join': [
            '',
            [
              {
                Ref: 'RestApi0C43BF4B',
              },
              '.execute-api.',
              {
                Ref: 'AWS::Region',
              },
              '.',
              {
                Ref: 'AWS::URLSuffix',
              },
            ],
          ],
        },
        AuthType: 'NO_AUTH',
        Headers: {
          'TaskToken.$': 'States.Array($$.Task.Token)',
        },
        Method: 'GET',
        Stage: 'dev',
      },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':states:::apigateway:invoke.waitForTaskToken',
          ],
        ],
      },
    });
  });

  test('wait for task token - using JSONata', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // WHEN
    const task = CallApiGatewayRestApiEndpoint.jsonata(stack, 'Call', {
      api: restApi,
      method: HttpMethod.GET,
      stageName: 'dev',
      integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
      headers: sfn.TaskInput.fromObject({ TaskToken: '{% States.Array($states.context.Task.Token) %}' }),
    });

    // THEN
    expect(stack.resolve(task.toStateJson())).toEqual({
      Type: 'Task',
      QueryLanguage: 'JSONata',
      End: true,
      Arguments: {
        ApiEndpoint: {
          'Fn::Join': [
            '',
            [
              {
                Ref: 'RestApi0C43BF4B',
              },
              '.execute-api.',
              {
                Ref: 'AWS::Region',
              },
              '.',
              {
                Ref: 'AWS::URLSuffix',
              },
            ],
          ],
        },
        AuthType: 'NO_AUTH',
        Headers: {
          TaskToken: '{% States.Array($states.context.Task.Token) %}',
        },
        Method: 'GET',
        Stage: 'dev',
      },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':states:::apigateway:invoke.waitForTaskToken',
          ],
        ],
      },
    });
  });

  test('JSONata expression as apiPath is passed through to state JSON', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // WHEN
    const task = CallApiGatewayRestApiEndpoint.jsonata(stack, 'Call', {
      api: restApi,
      method: HttpMethod.GET,
      stageName: 'dev',
      apiPath: '{% "/path/" & $states.input.path_suffix %}',
    });

    // THEN
    expect(stack.resolve(task.toStateJson())).toMatchObject({
      Type: 'Task',
      QueryLanguage: 'JSONata',
      End: true,
      Arguments: {
        Method: 'GET',
        Path: '{% "/path/" & $states.input.path_suffix %}',
        Stage: 'dev',
      },
      Resource: {
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':states:::apigateway:invoke',
          ],
        ],
      },
    });
  });

  test('wait for task token - missing token', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // THEN
    expect(() => {
      new CallApiGatewayRestApiEndpoint(stack, 'Call', {
        api: restApi,
        method: HttpMethod.GET,
        stageName: 'dev',
        integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
      });
    }).toThrow(/Task Token is required in `headers` for WAIT_FOR_TASK_TOKEN pattern. Use JsonPath.taskToken to set the token./);
  });

  test('unsupported integration pattern - RUN_JOB', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const restApi = new apigateway.RestApi(stack, 'RestApi');

    // THEN
    expect(() => {
      new CallApiGatewayRestApiEndpoint(stack, 'Call', {
        api: restApi,
        method: HttpMethod.GET,
        stageName: 'dev',
        integrationPattern: sfn.IntegrationPattern.RUN_JOB,
      });
    }).toThrow(/Unsupported service integration pattern./);
  });
});
