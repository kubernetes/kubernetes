/**
 * Targeted tests to improve branch coverage
 */

import { Template, Match } from '../../../../assertions';
import * as apigateway from '../../../../aws-apigateway';
import * as cloudwatch from '../../../../aws-cloudwatch';
import * as iam from '../../../../aws-iam';
import * as kms from '../../../../aws-kms';
import * as lambda from '../../../../aws-lambda';
import * as cdk from '../../../../core';
import {
  ApiKeyCredentialProvider,
  ApiSchema,
  Gateway,
  GatewayCredentialProvider,
  GatewayExceptionLevel,
  OAuth2CredentialProvider,
} from '../../../lib';
import { CustomClaimOperator } from '../../../lib/common/types';
import { GatewayAuthorizer } from '../../../lib/gateway/inbound-auth/authorizer';
import { GatewayCustomClaim } from '../../../lib/gateway/inbound-auth/custom-claim';
import { LambdaInterceptor } from '../../../lib/gateway/interceptor';
import { ApiKeyCredentialLocation } from '../../../lib/gateway/outbound-auth/api-key';
import { GatewayProtocol, MCPProtocolVersion, McpGatewaySearchType } from '../../../lib/gateway/protocol';
import { ToolSchema, SchemaDefinitionType } from '../../../lib/gateway/targets/schema/tool-schema';
import { ApiGatewayHttpMethod } from '../../../lib/gateway/targets/target-configuration';

/**
 * Helper: wraps a Metric in a minimal Alarm so we can assert on the
 * synthesized CloudWatch::Alarm template properties.
 */
function alarmForMetric(stack: cdk.Stack, id: string, metric: cloudwatch.Metric): void {
  new cloudwatch.Alarm(stack, id, {
    metric,
    evaluationPeriods: 1,
    threshold: 1,
  });
}

describe('Gateway Coverage Tests', () => {
  let stack: cdk.Stack;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'TestStack');
  });

  test('Should create a ServiceRole when no custom role is provided', () => {
    new Gateway(stack, 'Gateway');

    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Principal: { Service: 'bedrock-agentcore.amazonaws.com' },
          }),
        ]),
      },
    });
  });

  test('Should use the provided role and not create a ServiceRole', () => {
    const role = new iam.Role(stack, 'CustomRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    new Gateway(stack, 'Gateway', { role });

    const template = Template.fromStack(stack);
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      RoleArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('CustomRole.*'), 'Arn'] },
    });
  });

  test('Should grant KMS permissions when both kmsKey and custom role provided', () => {
    const key = new kms.Key(stack, 'Key');
    const role = new iam.Role(stack, 'Role', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    new Gateway(stack, 'Gateway', {
      kmsKey: key,
      role: role,
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['kms:Decrypt', 'kms:Encrypt']),
            Effect: 'Allow',
          }),
        ]),
      },
      Roles: Match.arrayWith([{ Ref: Match.stringLikeRegexp('Role.*') }]),
    });
  });

  test('Should pass credentialProviderConfigurations when provided to Lambda target', () => {
    const gateway = new Gateway(stack, 'Gateway', {});

    const fn = new lambda.Function(stack, 'Function', {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => ({ statusCode: 200 });'),
    });

    const toolSchema = ToolSchema.fromInline([{
      name: 'test_tool',
      description: 'Test tool',
      inputSchema: { type: SchemaDefinitionType.OBJECT, properties: {} },
    }]);

    const creds = [GatewayCredentialProvider.fromIamRole()];

    gateway.addLambdaTarget('Target', {
      lambdaFunction: fn,
      toolSchema: toolSchema,
      credentialProviderConfigurations: creds,
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      CredentialProviderConfigurations: Match.arrayWith([
        Match.objectLike({ CredentialProviderType: 'GATEWAY_IAM_ROLE' }),
      ]),
    });
  });

  const minimalOpenApiSchema = ApiSchema.fromInline(
    JSON.stringify({
      openapi: '3.0.0',
      info: { title: 'Cov', version: '1.0.0' },
      servers: [{ url: 'https://api.example.com' }],
      paths: {
        '/test': {
          get: {
            operationId: 'getTest',
            responses: { 200: { description: 'OK' } },
          },
        },
      },
    }),
  );

  test('OpenAPI target uses Token Vault ApiKeyCredentialProvider via fromApiKeyIdentity', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'test-gateway' });
    const apiKey = new ApiKeyCredentialProvider(stack, 'KeyProv', {
      apiKeyCredentialProviderName: 'gw_cov_apikey',
      apiKey: cdk.SecretValue.unsafePlainText('secret'),
    });

    gateway.addOpenApiTarget('OpenApiKey', {
      gatewayTargetName: 'openapi-api-key-id',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [GatewayCredentialProvider.fromApiKeyIdentity(apiKey)],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      Name: 'openapi-api-key-id',
      CredentialProviderConfigurations: [
        {
          CredentialProviderType: 'API_KEY',
          CredentialProvider: {
            ApiKeyCredentialProvider: {
              ProviderArn: { 'Fn::GetAtt': ['KeyProv8A34724C', 'CredentialProviderArn'] },
              CredentialLocation: 'HEADER',
              CredentialParameterName: 'Authorization',
              CredentialPrefix: 'Bearer ',
            },
          },
        },
      ],
    });
  });

  test('OpenAPI target uses Token Vault OAuth2CredentialProvider via fromOauthIdentity', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'test-gateway' });
    const oauth = OAuth2CredentialProvider.usingGithub(stack, 'OAuthProvider', {
      oAuth2CredentialProviderName: 'gw_cov_oauth',
      clientId: 'cid',
      clientSecret: cdk.SecretValue.unsafePlainText('csec'),
    });

    gateway.addOpenApiTarget('OpenApiOauth', {
      gatewayTargetName: 'openapi-oauth-id',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromOauthIdentity(oauth, { scopes: ['read:user'], customParameters: { k: 'v' } }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      Name: 'openapi-oauth-id',
      CredentialProviderConfigurations: [
        {
          CredentialProviderType: 'OAUTH',
          CredentialProvider: {
            OauthCredentialProvider: {
              ProviderArn: { 'Fn::GetAtt': [Match.stringLikeRegexp('OAuthProvider.*'), 'CredentialProviderArn'] },
              Scopes: ['read:user'],
              CustomParameters: { k: 'v' },
            },
          },
        },
      ],
    });
  });

  test('Gateway role receives outbound auth grants for identity-backed OpenAPI targets', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'test-gateway' });
    const apiKey = new ApiKeyCredentialProvider(stack, 'KeyProv', {
      apiKeyCredentialProviderName: 'gw_cov_grant_key',
      apiKey: cdk.SecretValue.unsafePlainText('k'),
    });
    const oauth = OAuth2CredentialProvider.usingGithub(stack, 'Gh', {
      oAuth2CredentialProviderName: 'gw_cov_grant_oauth',
      clientId: 'c',
      clientSecret: cdk.SecretValue.unsafePlainText('s'),
    });

    gateway.addOpenApiTarget('A', {
      gatewayTargetName: 'openapi-grant-a',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [GatewayCredentialProvider.fromApiKeyIdentity(apiKey)],
    });
    gateway.addOpenApiTarget('B', {
      gatewayTargetName: 'openapi-grant-b',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [GatewayCredentialProvider.fromOauthIdentity(oauth, { scopes: ['read:user'] })],
    });

    const template = Template.fromStack(stack);
    // All grants go to the single gateway service role (1 role, 1 policy).
    // Separate hasResourceProperties calls because Match.arrayWith with multiple patterns
    // uses greedy matching that fails when statement shapes overlap.
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.resourceCountIs('AWS::IAM::Policy', 1);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['bedrock-agentcore:GetWorkloadAccessTokenForJWT', 'bedrock-agentcore:GetWorkloadAccessTokenForUserId']),
            Effect: 'Allow',
          }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({ Action: 'bedrock-agentcore:GetResourceApiKey', Effect: 'Allow' }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({ Action: 'bedrock-agentcore:GetResourceOauth2Token', Effect: 'Allow' }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({ Action: 'bedrock-agentcore:CompleteResourceTokenAuth', Effect: 'Allow' }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({ Action: 'secretsmanager:GetSecretValue', Effect: 'Allow' }),
        ]),
      },
    });
  });

  test('API key outbound auth produces three separate IAM statements with correct resource scoping', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'my-gateway' });
    const apiKey = new ApiKeyCredentialProvider(stack, 'Key', {
      apiKeyCredentialProviderName: 'key-prov',
      apiKey: cdk.SecretValue.unsafePlainText('k'),
    });

    gateway.addOpenApiTarget('T', {
      gatewayTargetName: 'target',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [GatewayCredentialProvider.fromApiKeyIdentity(apiKey)],
    });

    const template = Template.fromStack(stack);

    // Statement 1: GetWorkloadAccessToken scoped to workload-identity-directory
    // Statement 1: GetWorkloadAccessToken scoped to directory and identity wildcard
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:GetWorkloadAccessToken',
            Effect: 'Allow',
            Resource: [
              { 'Fn::Join': ['', Match.arrayWith([':workload-identity-directory/default'])] }, // directory ARN (arn:*:bedrock-agentcore:*:*:workload-identity-directory/default)
              { 'Fn::Join': ['', Match.arrayWith([':workload-identity-directory/default/workload-identity/my-gateway-*'])] }, // identity wildcard scoped to gateway name (arn:*:bedrock-agentcore:*:*:workload-identity-directory/default/workload-identity/my-gateway-*)
            ],
          }),
        ]),
      },
    });

    // Statement 2: GetResourceApiKey scoped to token vault, provider, directory, and identity
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:GetResourceApiKey',
            Effect: 'Allow',
            Resource: Match.arrayWith([
              { 'Fn::Join': ['', Match.arrayWith([':token-vault/default'])] }, // token vault ARN (arn:*:bedrock-agentcore:*:*:token-vault/default)
              { 'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp('Key.*'), 'CredentialProviderArn']) }, // the API key provider
            ]),
          }),
        ]),
      },
    });

    // Statement 3: Secrets Manager scoped to service-managed secret prefix
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'secretsmanager:GetSecretValue',
            Effect: 'Allow',
            Resource: { 'Fn::Join': ['', Match.arrayWith([':secret:bedrock-agentcore-identity!*'])] }, // wildcard for CDK-created provider secrets (arn:*:secretsmanager:*:*:secret:bedrock-agentcore-identity!*)
          }),
        ]),
      },
    });
  });

  test('OAuth outbound auth produces four separate IAM statements with correct resource scoping', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'my-gateway' });
    const oauth = OAuth2CredentialProvider.usingGithub(stack, 'Gh', {
      oAuth2CredentialProviderName: 'oauth-prov',
      clientId: 'c',
      clientSecret: cdk.SecretValue.unsafePlainText('s'),
    });

    gateway.addOpenApiTarget('T', {
      gatewayTargetName: 'target',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromOauthIdentity(oauth, { scopes: ['read'] }),
      ],
    });

    const template = Template.fromStack(stack);

    // Statement 1: GetWorkloadAccessToken + ForJWT + ForUserId scoped to directory
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['bedrock-agentcore:GetWorkloadAccessToken']),
            Effect: 'Allow',
            Resource: [
              { 'Fn::Join': ['', Match.arrayWith([':workload-identity-directory/default'])] }, // directory ARN (arn:*:bedrock-agentcore:*:*:workload-identity-directory/default)
              { 'Fn::Join': ['', Match.arrayWith([':workload-identity-directory/default/workload-identity/my-gateway-*'])] }, // identity wildcard scoped to gateway name (arn:*:bedrock-agentcore:*:*:workload-identity-directory/default/workload-identity/my-gateway-*)
            ],
          }),
        ]),
      },
    });

    // Statement 2: CompleteResourceTokenAuth scoped to token vault and provider
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:CompleteResourceTokenAuth',
            Effect: 'Allow',
            Resource: Match.arrayWith([
              { 'Fn::Join': ['', Match.arrayWith([':token-vault/default'])] }, // token vault ARN (arn:*:bedrock-agentcore:*:*:token-vault/default)
              { 'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp('Gh.*'), 'CredentialProviderArn']) }, // the OAuth provider
            ]),
          }),
        ]),
      },
    });

    // Statement 3: GetResourceOauth2Token scoped to provider
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:GetResourceOauth2Token',
            Effect: 'Allow',
            Resource: Match.arrayWith([
              { 'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp('Gh.*'), 'CredentialProviderArn']) }, // the OAuth provider
            ]),
          }),
        ]),
      },
    });

    // Statement 4: Secrets Manager scoped to service-managed secret prefix
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'secretsmanager:GetSecretValue',
            Effect: 'Allow',
            Resource: { 'Fn::Join': ['', Match.arrayWith([':secret:bedrock-agentcore-identity!*'])] }, // wildcard for CDK-created provider secrets (arn:*:secretsmanager:*:*:secret:bedrock-agentcore-identity!*)
          }),
        ]),
      },
    });
  });

  test('API key outbound auth with fromApiKeyIdentityArn uses provided secret ARN string', () => {
    const gateway = new Gateway(stack, 'Gateway', { gatewayName: 'my-gateway' });

    gateway.addOpenApiTarget('T', {
      gatewayTargetName: 'target',
      apiSchema: minimalOpenApiSchema,
      validateOpenApiSchema: false,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromApiKeyIdentityArn({
          providerArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/tv1/apikeycredentialprovider/my-key',
          secretArn: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret-abc123',
        }),
      ],
    });

    const template = Template.fromStack(stack);

    // All three statements present and separate
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:GetWorkloadAccessToken',
          }),
          Match.objectLike({
            Action: 'bedrock-agentcore:GetResourceApiKey',
            Resource: Match.arrayWith([
              'arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/tv1/apikeycredentialprovider/my-key',
            ]),
          }),
          Match.objectLike({
            Action: 'secretsmanager:GetSecretValue',
            Resource: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret-abc123',
          }),
        ]),
      },
    });
  });
});

describe('Gateway grant methods tests', () => {
  const GATEWAY_LOGICAL_ID = 'testgateway.*';

  let stack: cdk.Stack;
  let gateway: Gateway;
  let grantee: iam.Role;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    gateway = new Gateway(stack, 'test-gateway', {});
    grantee = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
  });

  test('Should grant custom actions to IAM principal scoped to gateway ARN', () => {
    gateway.grant(grantee, 'bedrock-agentcore:GetGateway', 'bedrock-agentcore:ListGatewayTargets');

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['bedrock-agentcore:GetGateway', 'bedrock-agentcore:ListGatewayTargets'],
            Effect: 'Allow',
            Resource: Match.objectLike({
              'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp(GATEWAY_LOGICAL_ID), 'GatewayArn']),
            }),
          }),
        ]),
      },
    });
  });

  test('Should grant read permissions with Get on gateway ARN and List on all resources', () => {
    gateway.grantRead(grantee);

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['bedrock-agentcore:GetGatewayTarget', 'bedrock-agentcore:GetGateway'],
            Effect: 'Allow',
            Resource: Match.objectLike({
              'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp(GATEWAY_LOGICAL_ID), 'GatewayArn']),
            }),
          }),
        ]),
      },
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['bedrock-agentcore:ListGateways', 'bedrock-agentcore:ListGatewayTargets'],
            Effect: 'Allow',
            Resource: '*',
          }),
        ]),
      },
    });
  });

  test('Should grant manage permissions with Create, Update, and Delete actions scoped to gateway ARN', () => {
    gateway.grantManage(grantee);

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith([
              'bedrock-agentcore:CreateGateway',
              'bedrock-agentcore:CreateGatewayTarget',
              'bedrock-agentcore:UpdateGateway',
              'bedrock-agentcore:UpdateGatewayTarget',
              'bedrock-agentcore:DeleteGateway',
              'bedrock-agentcore:DeleteGatewayTarget',
            ]),
            Effect: 'Allow',
            Resource: Match.objectLike({
              'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp(GATEWAY_LOGICAL_ID), 'GatewayArn']),
            }),
          }),
        ]),
      },
    });
  });

  test('Should grant invoke permission scoped to gateway ARN', () => {
    gateway.grantInvoke(grantee);

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeGateway',
            Effect: 'Allow',
            Resource: Match.objectLike({
              'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp(GATEWAY_LOGICAL_ID), 'GatewayArn']),
            }),
          }),
        ]),
      },
    });
  });
});

describe('Gateway metric methods tests', () => {
  let stack: cdk.Stack;
  let gateway: Gateway;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    gateway = new Gateway(stack, 'test-gateway-metrics', {});
  });

  test('metric() produces correct namespace, name, and dimensions', () => {
    const metric = gateway.metric('CustomMetric', { dimensionsMap: { CustomDimension: 'value' } });
    alarmForMetric(stack, 'CustomAlarm', metric);

    const template = Template.fromStack(stack);
    // Operation is always emitted.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Namespace: 'AWS/Bedrock-AgentCore',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeGateway' }),
      ]),
    });
    // Protocol comes from the gateway's protocol type.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Protocol', Value: 'MCP' }),
      ]),
    });
    // Resource is this gateway's ARN.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('.*'), 'GatewayArn'] } }),
      ]),
    });
    // A caller dimension merges over the defaults.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'CustomMetric',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'CustomDimension', Value: 'value' }),
      ]),
    });
  });

  test('metricInvocations() produces Invocations with Sum statistic', () => {
    alarmForMetric(stack, 'InvocationsAlarm', gateway.metricInvocations());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeGateway' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Protocol', Value: 'MCP' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('.*'), 'GatewayArn'] } }),
      ]),
    });
  });

  test('metricThrottles() produces Throttles with Sum statistic', () => {
    alarmForMetric(stack, 'ThrottlesAlarm', gateway.metricThrottles());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Throttles',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricSystemErrors() produces SystemErrors with Sum statistic', () => {
    alarmForMetric(stack, 'SystemErrorsAlarm', gateway.metricSystemErrors());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'SystemErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricUserErrors() produces UserErrors with Sum statistic', () => {
    alarmForMetric(stack, 'UserErrorsAlarm', gateway.metricUserErrors());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'UserErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricLatency() produces Latency with Average statistic', () => {
    alarmForMetric(stack, 'LatencyAlarm', gateway.metricLatency());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
    });
  });

  test('metricDuration() produces Duration with Average statistic', () => {
    alarmForMetric(stack, 'DurationAlarm', gateway.metricDuration());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Duration',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
    });
  });

  test('metricTargetExecutionTime() produces TargetExecutionTime with Average statistic', () => {
    alarmForMetric(stack, 'TargetExecAlarm', gateway.metricTargetExecutionTime());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TargetExecutionTime',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
    });
  });

  test('metricTargetType() produces TargetType with dimension and Sum statistic', () => {
    alarmForMetric(stack, 'TargetTypeAlarm', gateway.metricTargetType('Lambda'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TargetType',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'TargetType', Value: 'Lambda' }),
      ]),
    });
    // Operation and Protocol are also emitted on this metric.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TargetType',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeGateway' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TargetType',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Protocol', Value: 'MCP' }),
      ]),
    });
  });

  test('metric() on imported gateway emits expected dimensions', () => {
    const importedGateway = Gateway.fromGatewayAttributes(stack, 'ImportedGw', {
      gatewayArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/test-id',
      gatewayId: 'test-id',
      gatewayName: 'test-name',
      role: iam.Role.fromRoleArn(stack, 'ImportedRole', 'arn:aws:iam::123456789012:role/r'),
    });
    alarmForMetric(stack, 'ImportedGwAlarm', importedGateway.metric('Invocations'));

    const template = Template.fromStack(stack);
    // An imported gateway emits the same dimensions.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeGateway' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Protocol', Value: 'MCP' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:gateway/test-id' }),
      ]),
    });
  });

  test('metricTargetType() retains TargetType when caller supplies extra dimensionsMap', () => {
    alarmForMetric(stack, 'TargetTypeExtraDimAlarm', gateway.metricTargetType('Lambda', { dimensionsMap: { Foo: 'bar' } }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TargetType',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      // CloudWatch renders dimensions alphabetically, so the ordered subsequence [Foo, TargetType]
      // is intentional here (a future dimension rename could reorder this).
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Foo', Value: 'bar' }),
        Match.objectLike({ Name: 'TargetType', Value: 'Lambda' }),
      ]),
    });
  });

  test('custom statistic prop overrides the default', () => {
    alarmForMetric(stack, 'CustomStatAlarm', gateway.metricInvocations({ statistic: 'Average' }));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
    });
  });
});

describe('OAuth credential provider tests', () => {
  let stack: cdk.Stack;
  let gateway: Gateway;
  let fn: lambda.Function;
  let toolSchema: ToolSchema;

  const providerArn = 'arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/default/oauth2credentialprovider/test';
  const secretArn = 'arn:aws:secretsmanager:us-east-1:123456789012:secret:test-secret';

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    gateway = new Gateway(stack, 'Gateway', {});
    fn = new lambda.Function(stack, 'Fn', {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => ({});'),
    });
    toolSchema = ToolSchema.fromInline([{
      name: 'tool',
      description: 'tool',
      inputSchema: { type: SchemaDefinitionType.OBJECT, properties: {} },
    }]);
  });

  test('Should render OAuth credential provider in GatewayTarget template with scopes and customParameters', () => {
    gateway.addLambdaTarget('Target', {
      lambdaFunction: fn,
      toolSchema,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromOauthIdentityArn({
          providerArn,
          secretArn,
          scopes: ['openid', 'profile'],
          customParameters: { prompt: 'consent' },
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      CredentialProviderConfigurations: Match.arrayWith([
        Match.objectLike({
          CredentialProviderType: 'OAUTH',
          CredentialProvider: {
            OauthCredentialProvider: {
              ProviderArn: providerArn,
              Scopes: ['openid', 'profile'],
              CustomParameters: { prompt: 'consent' },
            },
          },
        }),
      ]),
    });
  });

  test('Should render OAuth credential provider without customParameters when not provided', () => {
    gateway.addLambdaTarget('Target', {
      lambdaFunction: fn,
      toolSchema,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromOauthIdentityArn({
          providerArn,
          secretArn,
          scopes: ['openid'],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      CredentialProviderConfigurations: Match.arrayWith([
        Match.objectLike({
          CredentialProviderType: 'OAUTH',
          CredentialProvider: {
            OauthCredentialProvider: {
              ProviderArn: providerArn,
              Scopes: ['openid'],
            },
          },
        }),
      ]),
    });
  });

  test('Should grant OAuth and Secrets Manager permissions to gateway role', () => {
    gateway.addLambdaTarget('Target', {
      lambdaFunction: fn,
      toolSchema,
      credentialProviderConfigurations: [
        GatewayCredentialProvider.fromOauthIdentityArn({
          providerArn,
          secretArn,
          scopes: ['openid'],
        }),
      ],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:GetResourceOauth2Token',
            Effect: 'Allow',
            Resource: Match.arrayWith([providerArn]),
          }),
        ]),
      },
    });
  });
});

describe('Gateway authorizer CFN rendering', () => {
  test('custom JWT authorizer with custom claims renders in template', () => {
    const stack = new cdk.Stack();

    new Gateway(stack, 'Gateway', {
      authorizerConfiguration: GatewayAuthorizer.usingCustomJwt({
        discoveryUrl: 'https://example.com/.well-known/openid-configuration',
        allowedAudience: ['my-app'],
        allowedClients: ['client-1'],
        customClaims: [
          GatewayCustomClaim.withStringValue('tenant', 'acme'),
          GatewayCustomClaim.withStringArrayValue('roles', ['admin'], CustomClaimOperator.CONTAINS),
        ],
      }),
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: 'https://example.com/.well-known/openid-configuration',
          AllowedAudience: ['my-app'],
          AllowedClients: ['client-1'],
          CustomClaims: Match.arrayWith([
            Match.objectLike({ InboundTokenClaimName: 'tenant', InboundTokenClaimValueType: 'STRING' }),
            Match.objectLike({ InboundTokenClaimName: 'roles', InboundTokenClaimValueType: 'STRING_ARRAY' }),
          ]),
        },
      },
    });
  });
});

describe('LambdaInterceptor tests', () => {
  let stack: cdk.Stack;
  let fn: lambda.Function;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    fn = new lambda.Function(stack, 'Interceptor', {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => ({});'),
    });
  });

  test('forRequest interceptor renders with REQUEST interception point and grants lambda:InvokeFunction', () => {
    new Gateway(stack, 'test-gateway', {
      interceptorConfigurations: [LambdaInterceptor.forRequest(fn)],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      InterceptorConfigurations: Match.arrayWith([
        Match.objectLike({
          InterceptionPoints: ['REQUEST'],
          InputConfiguration: { PassRequestHeaders: false },
        }),
      ]),
    });
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'lambda:InvokeFunction',
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('forResponse interceptor renders with RESPONSE interception point', () => {
    new Gateway(stack, 'test-gateway', {
      interceptorConfigurations: [LambdaInterceptor.forResponse(fn)],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      InterceptorConfigurations: Match.arrayWith([
        Match.objectLike({
          InterceptionPoints: ['RESPONSE'],
        }),
      ]),
    });
  });

  test('passRequestHeaders=true renders in the template', () => {
    new Gateway(stack, 'test-gateway', {
      interceptorConfigurations: [LambdaInterceptor.forRequest(fn, { passRequestHeaders: true })],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      InterceptorConfigurations: Match.arrayWith([
        Match.objectLike({
          InputConfiguration: { PassRequestHeaders: true },
        }),
      ]),
    });
  });

  test('passRequestHeaders defaults to false when not specified', () => {
    new Gateway(stack, 'test-gateway', {
      interceptorConfigurations: [LambdaInterceptor.forRequest(fn)],
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      InterceptorConfigurations: Match.arrayWith([
        Match.objectLike({
          InputConfiguration: { PassRequestHeaders: false },
        }),
      ]),
    });
  });
});

/**
 * ApiKeyCredentialLocation is a pure value-object factory that configures
 * how API key credentials are sent (header vs query parameter). It does not
 * produce CFN resources on its own — it only renders when attached to a
 * GatewayTarget's credential provider. We test the factory output directly
 * because there is no template to assert against at this level.
 */
describe('ApiKeyCredentialLocation.queryParameter tests', () => {
  test('Should default credentialParameterName to "api_key" and leave credentialPrefix undefined', () => {
    const location = ApiKeyCredentialLocation.queryParameter();

    expect(location.credentialLocationType).toBe('QUERY_PARAMETER');
    expect(location.credentialParameterName).toBe('api_key');
    expect(location.credentialPrefix).toBeUndefined();
  });

  test('Should accept custom credentialParameterName and credentialPrefix', () => {
    const location = ApiKeyCredentialLocation.queryParameter({
      credentialParameterName: 'myKey',
      credentialPrefix: 'Key-',
    });

    expect(location.credentialLocationType).toBe('QUERY_PARAMETER');
    expect(location.credentialParameterName).toBe('myKey');
    expect(location.credentialPrefix).toBe('Key-');
  });
});

describe('Gateway target convenience methods tests', () => {
  let stack: cdk.Stack;
  let gateway: Gateway;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    gateway = new Gateway(stack, 'test-gateway', {});
  });

  describe('addLambdaTarget', () => {
    let fn: lambda.Function;
    let toolSchema: ToolSchema;

    beforeEach(() => {
      fn = new lambda.Function(stack, 'Function', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => ({ statusCode: 200 });'),
      });
      toolSchema = ToolSchema.fromInline([{
        name: 'test_tool',
        description: 'Test tool',
        inputSchema: { type: SchemaDefinitionType.OBJECT, properties: {} },
      }]);
    });

    test('Should create a Lambda target with name and description', () => {
      gateway.addLambdaTarget('LambdaTarget', {
        gatewayTargetName: 'lambda-target',
        description: 'My Lambda target',
        lambdaFunction: fn,
        toolSchema: toolSchema,
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        Name: 'lambda-target',
        Description: 'My Lambda target',
      });
    });

    test('Should use default IAM role credentials when none provided', () => {
      gateway.addLambdaTarget('LambdaTarget', {
        lambdaFunction: fn,
        toolSchema: toolSchema,
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        CredentialProviderConfigurations: Match.arrayWith([
          Match.objectLike({ CredentialProviderType: 'GATEWAY_IAM_ROLE' }),
        ]),
      });
    });

    test('Should use provided credentials when supplied', () => {
      gateway.addLambdaTarget('LambdaTarget', {
        lambdaFunction: fn,
        toolSchema: toolSchema,
        credentialProviderConfigurations: [
          GatewayCredentialProvider.fromOauthIdentityArn({
            providerArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:token-vault/default/oauth2credentialprovider/test',
            secretArn: 'arn:aws:secretsmanager:us-east-1:123456789012:secret:test',
            scopes: ['openid'],
          }),
        ],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        CredentialProviderConfigurations: Match.arrayWith([
          Match.objectLike({ CredentialProviderType: 'OAUTH' }),
        ]),
      });
    });
  });

  describe('addOpenApiTarget', () => {
    test('Should create an OpenAPI target with inline schema', () => {
      const apiSchema = ApiSchema.fromInline(JSON.stringify({
        openapi: '3.0.0',
        info: { title: 'Test API', version: '1.0.0' },
        servers: [{ url: 'https://api.example.com' }],
        paths: {
          '/test': {
            get: {
              operationId: 'getTest',
              responses: { 200: { description: 'OK' } },
            },
          },
        },
      }));

      gateway.addOpenApiTarget('OpenApiTarget', {
        gatewayTargetName: 'openapi-target',
        apiSchema: apiSchema,
        validateOpenApiSchema: false,
        credentialProviderConfigurations: [GatewayCredentialProvider.fromIamRole()],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        Name: 'openapi-target',
        TargetConfiguration: {
          Mcp: {
            OpenApiSchema: {
              InlinePayload: Match.stringLikeRegexp('openapi.*3\\.0\\.0'),
            },
          },
        },
        CredentialProviderConfigurations: Match.arrayWith([
          Match.objectLike({ CredentialProviderType: 'GATEWAY_IAM_ROLE' }),
        ]),
      });
    });
  });

  describe('addSmithyTarget', () => {
    test('Should create a Smithy target with default IAM role credentials', () => {
      gateway.addSmithyTarget('SmithyTarget', {
        gatewayTargetName: 'smithy-target',
        smithyModel: ApiSchema.fromInline('{}'),
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        Name: 'smithy-target',
        TargetConfiguration: {
          Mcp: {
            SmithyModel: { InlinePayload: '{}' },
          },
        },
        CredentialProviderConfigurations: Match.arrayWith([
          Match.objectLike({ CredentialProviderType: 'GATEWAY_IAM_ROLE' }),
        ]),
      });
    });
  });

  describe('addMcpServerTarget', () => {
    test('Should create an MCP server target with required credentials', () => {
      gateway.addMcpServerTarget('McpTarget', {
        gatewayTargetName: 'mcp-target',
        endpoint: 'https://mcp-server.example.com',
        credentialProviderConfigurations: [GatewayCredentialProvider.fromIamRole()],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        Name: 'mcp-target',
        TargetConfiguration: {
          Mcp: {
            McpServer: { Endpoint: 'https://mcp-server.example.com' },
          },
        },
        CredentialProviderConfigurations: Match.arrayWith([
          Match.objectLike({ CredentialProviderType: 'GATEWAY_IAM_ROLE' }),
        ]),
      });
    });

    test('Should treat empty credentialProviderConfigurations array as no credentials', () => {
      gateway.addMcpServerTarget('McpTarget', {
        endpoint: 'https://mcp-server.example.com',
        credentialProviderConfigurations: [],
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        CredentialProviderConfigurations: Match.absent(),
      });
    });
  });

  describe('addApiGatewayTarget', () => {
    test('Should create an API Gateway target with tool configuration', () => {
      const restApi = new apigateway.RestApi(stack, 'RestApi', {
        restApiName: 'test-api',
        deployOptions: { stageName: 'prod' },
      });
      restApi.root.addResource('test').addMethod('GET');

      gateway.addApiGatewayTarget('ApiGwTarget', {
        gatewayTargetName: 'apigw-target',
        restApi: restApi,
        stage: 'prod',
        apiGatewayToolConfiguration: {
          toolFilters: [
            {
              filterPath: '/test',
              methods: [ApiGatewayHttpMethod.GET],
            },
          ],
        },
      });

      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
        Name: 'apigw-target',
        TargetConfiguration: {
          Mcp: {
            ApiGateway: {
              Stage: 'prod',
              ApiGatewayToolConfiguration: {
                ToolFilters: [
                  { FilterPath: '/test', Methods: ['GET'] },
                ],
              },
            },
          },
        },
      });
    });
  });
});

describe('MCPProtocolVersion.of() escape hatch', () => {
  test('renders custom version in the template', () => {
    const stack = new cdk.Stack();
    new Gateway(stack, 'TestGateway', {
      gatewayName: 'test-version',
      protocolConfiguration: GatewayProtocol.mcp({
        supportedVersions: [MCPProtocolVersion.of('2099-01-01')],
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      ProtocolConfiguration: Match.objectLike({
        Mcp: Match.objectLike({
          SupportedVersions: ['2099-01-01'],
        }),
      }),
    });
  });
});

describe('McpGatewaySearchType.of() escape hatch', () => {
  test('renders custom search type in the template', () => {
    const stack = new cdk.Stack();
    new Gateway(stack, 'TestGateway', {
      gatewayName: 'test-search',
      protocolConfiguration: GatewayProtocol.mcp({
        searchType: McpGatewaySearchType.of('CUSTOM_SEARCH'),
      }),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      ProtocolConfiguration: Match.objectLike({
        Mcp: Match.objectLike({
          SearchType: 'CUSTOM_SEARCH',
        }),
      }),
    });
  });
});

describe('GatewayExceptionLevel.of() escape hatch', () => {
  test('renders custom exception level in the template', () => {
    const stack = new cdk.Stack();
    new Gateway(stack, 'TestGateway', {
      gatewayName: 'test-exception',
      exceptionLevel: GatewayExceptionLevel.DEBUG,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      ExceptionLevel: 'DEBUG',
    });
  });
});

describe('InterceptionPoint rendering', () => {
  test('renders custom interception point in the template', () => {
    const stack = new cdk.Stack();
    const fn = new lambda.Function(stack, 'Fn', {
      runtime: lambda.Runtime.NODEJS_22_X,
      handler: 'index.handler',
      code: lambda.Code.fromInline('exports.handler = async () => {}'),
    });
    const gateway = new Gateway(stack, 'TestGateway', { gatewayName: 'test-intercept' });
    gateway.addInterceptor(LambdaInterceptor.forRequest(fn));
    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::Gateway', {
      InterceptorConfigurations: Match.arrayWith([
        Match.objectLike({
          InterceptionPoints: ['REQUEST'],
        }),
      ]),
    });
  });
});

describe('ApiGatewayHttpMethod.of() escape hatch', () => {
  test('renders custom HTTP method in the template', () => {
    const stack = new cdk.Stack();
    const api = new apigateway.RestApi(stack, 'Api');
    api.root.addResource('test').addMethod('GET');

    const gateway = new Gateway(stack, 'TestGateway', { gatewayName: 'test-method' });
    gateway.addApiGatewayTarget('ApiGwTarget', {
      gatewayTargetName: 'test-target',
      restApi: api,
      apiGatewayToolConfiguration: {
        toolFilters: [{ filterPath: '/test', methods: [ApiGatewayHttpMethod.of('CUSTOM')] }],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      TargetConfiguration: Match.objectLike({
        Mcp: Match.objectLike({
          ApiGateway: Match.objectLike({
            ApiGatewayToolConfiguration: Match.objectLike({
              ToolFilters: Match.arrayWith([Match.objectLike({
                Methods: ['CUSTOM'],
              })]),
            }),
          }),
        }),
      }),
    });
  });
});

describe('SchemaDefinitionType.of() escape hatch', () => {
  test('renders custom schema type in the template', () => {
    const stack = new cdk.Stack();
    const gateway = new Gateway(stack, 'TestGateway', { gatewayName: 'test-schema' });
    gateway.addLambdaTarget('SchemaTarget', {
      gatewayTargetName: 'schema-target',
      lambdaFunction: new lambda.Function(stack, 'Fn', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      }),
      toolSchema: ToolSchema.fromInline([{
        name: 'test_tool',
        description: 'test',
        inputSchema: { type: SchemaDefinitionType.of('custom_type'), properties: {} },
      }]),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::BedrockAgentCore::GatewayTarget', {
      TargetConfiguration: Match.objectLike({
        Mcp: Match.objectLike({
          Lambda: Match.objectLike({
            ToolSchema: Match.objectLike({
              InlinePayload: Match.arrayWith([Match.objectLike({
                InputSchema: Match.objectLike({ Type: 'custom_type' }),
              })]),
            }),
          }),
        }),
      }),
    });
  });
});
