import { Template } from '../../../assertions';
import * as iam from '../../../aws-iam';
import type * as sagemaker from '../../../aws-sagemaker';
import * as cdk from '../../../core';
import * as apigateway from '../../lib';

describe('SageMaker Integration', () => {
  test('minimal setup', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'my-api');
    const endpoint = new FakeEndpoint(stack, 'FakeEndpoint');

    // WHEN
    const integration = new apigateway.SagemakerIntegration(endpoint);
    api.root.addMethod('POST', integration);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ApiGateway::Method', {
      Integration: {
        IntegrationHttpMethod: 'POST',
        Type: 'AWS',
        Uri: {
          'Fn::Join': [
            '',
            [
              'arn:',
              {
                Ref: 'AWS::Partition',
              },
              ':apigateway:',
              {
                Ref: 'AWS::Region',
              },
              ':runtime.sagemaker:path/endpoints/endpointName/invocations',
            ],
          ],
        },
      },
    });
  });
});

class FakeEndpoint extends cdk.Resource implements sagemaker.IEndpoint {
  public readonly endpointArn = 'arn:aws:sagemaker:us-east-1:123456789012:endpoint/my-endpoint';

  public readonly endpointName = 'endpointName';

  public grantInvoke(grantee: iam.IGrantable) {
    return iam.Grant.addToPrincipal({
      grantee,
      actions: ['sagemaker:InvokeEndpoint'],
      resourceArns: [this.endpointArn],
    });
  }

  public get endpointRef(): sagemaker.EndpointReference {
    return { endpointArn: this.endpointArn };
  }
}
