import * as path from 'path';
import { Annotations, Template, Match } from '../../../../assertions';
import * as cloudwatch from '../../../../aws-cloudwatch';
import * as cognito from '../../../../aws-cognito';
import * as ec2 from '../../../../aws-ec2';
import * as ecr from '../../../../aws-ecr';
import * as iam from '../../../../aws-iam';
import * as firehose from '../../../../aws-kinesisfirehose';
import * as lambda from '../../../../aws-lambda';
import * as logs from '../../../../aws-logs';
import * as s3 from '../../../../aws-s3';
import { Duration } from '../../../../core';
import * as cdk from '../../../../core';
import { CustomClaimOperator } from '../../../lib/common/types';
import { RuntimeNetworkConfiguration } from '../../../lib/network/network-configuration';
import { RuntimeCustomClaim } from '../../../lib/runtime/inbound-auth/custom-claim';
import { RuntimeAuthorizerConfiguration } from '../../../lib/runtime/inbound-auth/runtime-authorizer-configuration';
import { LoggingDestination, LogType } from '../../../lib/runtime/observability';
import { Runtime } from '../../../lib/runtime/runtime';
import { AgentCoreRuntime, AgentRuntimeArtifact } from '../../../lib/runtime/runtime-artifact';
import {
  ProtocolType,
} from '../../../lib/runtime/types';

describe('Runtime default tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      description: 'A test runtime for agent execution',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    // With CloudFormation template, we have the native runtime resource
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::IAM::Role', 1); // Only Execution role
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 1);
  });

  test('Should have Runtime resource with expected properties', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime',
      Description: 'A test runtime for agent execution',
      ProtocolConfiguration: 'HTTP',
      NetworkConfiguration: { NetworkMode: 'PUBLIC' },
      AgentRuntimeArtifact: {
        ContainerConfiguration: {
          ContainerUri: { 'Fn::Join': ['', Match.arrayWith([{ Ref: Match.stringLikeRegexp('TestRepository.*') }, ':v1.0.0'])] },
        },
      },
      RoleArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('testruntimeExecutionRole*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have execution role with correct properties', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
      Description: 'Execution role for Bedrock Agent Core Runtime',
      MaxSessionDuration: 28800, // 8 hours in seconds
    });
  });

  test('Should have policy for execution role with correct permissions', () => {
    template.hasResourceProperties('AWS::IAM::Policy', expectedExecutionRolePolicy);
  });
});

describe('Runtime with custom execution role tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let customRole: iam.Role;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
      roleName: 'custom-runtime-execution-role',
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime-custom-role',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_custom_role',
      description: 'A test runtime with custom execution role',
      executionRole: customRole,
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::IAM::Role', 1); // Only CustomExecutionRole (no auto-created role)
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 1);
  });

  test('Should have Runtime with custom execution role', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime_custom_role',
      RoleArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('CustomExecutionRole*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have custom execution role with correct properties', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'custom-runtime-execution-role',
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
          },
        ],
        Version: '2012-10-17',
      },
    });
  });

  test('Should have policy for custom execution role with correct permissions', () => {
    template.hasResourceProperties('AWS::IAM::Policy', expectedExecutionRolePolicy);
  });
});

describe('Runtime with environment variables tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime-env',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_env',
      description: 'A test runtime with environment variables',
      agentRuntimeArtifact: agentRuntimeArtifact,
      environmentVariables: {
        API_KEY: 'test-api-key',
        DEBUG_MODE: 'true',
        MAX_RETRIES: '3',
        LOG_LEVEL: 'INFO',
      },
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::IAM::Role', 1); // Only execution role
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 1);
  });

  test('Should have Runtime with environment variables', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime_env',
      EnvironmentVariables: {
        API_KEY: 'test-api-key',
        DEBUG_MODE: 'true',
        MAX_RETRIES: '3',
        LOG_LEVEL: 'INFO',
      },
    });
  });
});

describe('Runtime with authorizer configuration tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime-auth',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_auth',
      description: 'A test runtime with authorizer configuration',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingJWT(
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1', 'client2'],
        ['audience1'],
        ['scope1', 'scope2'],
      ),
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::IAM::Role', 1); // Only execution role
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 1);
  });

  test('Should have Runtime with JWT authorizer configuration using PascalCase properties', () => {
    // Verify that the CloudFormation template has the correct PascalCase properties
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: 'https://auth.example.com/.well-known/openid-configuration',
          AllowedClients: ['client1', 'client2'],
          AllowedAudience: ['audience1'],
          AllowedScopes: ['scope1', 'scope2'],
        },
      },
    });
  });
});

describe('Runtime with Cognito authorizer configuration tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const userPool = new cognito.UserPool(stack, 'MyUserPool', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const userPoolClient = userPool.addClient('MyUserPoolClient', {});
    const anotherUserPoolClient = userPool.addClient('MyAnotherUserPoolClient', {});

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime-cognito',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_cognito',
      description: 'A test runtime with Cognito authorizer configuration',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingCognito(
        userPool,
        [userPoolClient, anotherUserPoolClient],
        ['cognito-audience'],
        ['read', 'write'],
      ),
    });

    template = Template.fromStack(stack);
  });

  test('Should have Runtime with Cognito authorizer configuration using PascalCase properties', () => {
    // Verify that the CloudFormation template has the correct PascalCase properties
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: {
            'Fn::Join': [
              '',
              [
                'https://cognito-idp.us-east-1.amazonaws.com/',
                { Ref: 'MyUserPoolD09D1D74' },
                '/.well-known/openid-configuration',
              ],
            ],
          },
          AllowedClients: Match.arrayWith([
            { Ref: 'MyUserPoolMyUserPoolClient01266CD6' },
            { Ref: 'MyUserPoolMyAnotherUserPoolClient4444CD16' },
          ]),
          AllowedAudience: ['cognito-audience'],
          AllowedScopes: ['read', 'write'],
        },
      },
    });
  });
});

describe('Runtime with MCP protocol tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime-mcp',
    });

    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_mcp',
      description: 'A test runtime with MCP protocol',
      agentRuntimeArtifact: agentRuntimeArtifact,
      protocolConfiguration: ProtocolType.MCP,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::IAM::Role', 1); // Only execution role
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 1);
  });

  test('Should have Runtime with MCP protocol configuration', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime_mcp',
      Description: 'A test runtime with MCP protocol',
      ProtocolConfiguration: 'MCP',
      NetworkConfiguration: { NetworkMode: 'PUBLIC' },
    });
  });
});

describe('Runtime name validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should throw error for name with hyphen', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test-runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).toThrow('Runtime name must start with a letter and contain only letters, numbers, and underscores');
  });

  test('Should throw error for empty name', () => {
    expect(() => {
      new Runtime(stack, 'empty-name', {
        runtimeName: '',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).toThrow(/The field Runtime name is 0 characters long but must be at least 1 characters/);
  });

  test('Should throw error for name with spaces', () => {
    expect(() => {
      new Runtime(stack, 'name-with-spaces', {
        runtimeName: 'test runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).toThrow('Runtime name must start with a letter and contain only letters, numbers, and underscores');
  });

  test('Should throw error for name with special characters', () => {
    expect(() => {
      new Runtime(stack, 'name-with-special-chars', {
        runtimeName: 'test@runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).toThrow('Runtime name must start with a letter and contain only letters, numbers, and underscores');
  });

  test('Should throw error for name exceeding 48 characters', () => {
    const longName = 'a'.repeat(49);
    expect(() => {
      new Runtime(stack, 'long-name', {
        runtimeName: longName,
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).toThrow(/The field Runtime name is 49 characters long but must be less than or equal to 48 characters/);
  });

  test('Should accept valid name with underscores', () => {
    expect(() => {
      new Runtime(stack, 'valid-name', {
        runtimeName: 'test_runtime_123',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).not.toThrow();
  });

  test('Should accept valid name with only letters and numbers', () => {
    expect(() => {
      new Runtime(stack, 'valid-name-2', {
        runtimeName: 'testRuntime123',
        agentRuntimeArtifact: agentRuntimeArtifact,
      });
    }).not.toThrow();
  });
});

describe('Runtime environment variables validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should throw error for too many environment variables', () => {
    const envVars: { [key: string]: string } = {};
    for (let i = 0; i < 51; i++) {
      envVars[`KEY_${i}`] = `value_${i}`;
    }

    expect(() => {
      new Runtime(stack, 'too-many-env-vars', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        environmentVariables: envVars,
      });
    }).toThrow('Too many environment variables: 51. Maximum allowed is 50');
  });

  test('Should throw error for invalid environment variable key format', () => {
    expect(() => {
      new Runtime(stack, 'invalid-env-key', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        environmentVariables: {
          '123_INVALID': 'value', // Starts with number
        },
      });
    }).toThrow(/Environment variable key '123_INVALID' must start with a letter or underscore and contain only letters, numbers, and underscores/);
  });

  test('Should throw error for environment variable key too long', () => {
    const longKey = 'a'.repeat(101);
    expect(() => {
      new Runtime(stack, 'long-env-key', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        environmentVariables: {
          [longKey]: 'value',
        },
      });
    }).toThrow(/is 101 characters long but must be less than or equal to 100 characters/);
  });

  test('Should throw error for environment variable value too long', () => {
    const longValue = 'a'.repeat(2049);
    expect(() => {
      new Runtime(stack, 'long-env-value', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        environmentVariables: {
          TEST_KEY: longValue,
        },
      });
    }).toThrow('Invalid environment variable value length for key \'TEST_KEY\': 2049 characters. Values must not exceed 2048 characters');
  });

  test('Should accept valid environment variables', () => {
    expect(() => {
      new Runtime(stack, 'valid-env-vars', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        environmentVariables: {
          API_KEY: 'test-api-key',
          DEBUG_MODE: 'true',
          _INTERNAL_FLAG: 'enabled',
        },
      });
    }).not.toThrow();
  });
});

// Tests for instance-based auth configuration removed as these methods no longer exist
// Use RuntimeAuthorizerConfiguration static factory methods instead

describe('Runtime with local asset tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let runtime: Runtime;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Use local asset from testArtifact folder
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromAsset(
      path.join(__dirname, 'testArtifact'),
      {
        buildArgs: {
          BUILDKIT_INLINE_CACHE: '1',
        },
      },
    );

    runtime = new Runtime(stack, 'test-runtime-local', {
      runtimeName: 'test_runtime_local',
      description: 'A test runtime with local asset for agent execution',
      agentRuntimeArtifact: agentRuntimeArtifact,
      networkConfiguration: RuntimeNetworkConfiguration.usingPublicNetwork(),
      protocolConfiguration: ProtocolType.HTTP,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
    template.resourceCountIs('AWS::Lambda::Function', 0); // No Lambda functions
    template.resourceCountIs('AWS::IAM::Role', 1); // Only execution role
    template.resourceCountIs('AWS::CloudFormation::CustomResource', 0); // No custom resources
    template.resourceCountIs('AWS::ECR::Repository', 0); // Local asset doesn't create ECR repo in stack
  });

  test('Should have Runtime with local asset properties', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime_local',
      Description: 'A test runtime with local asset for agent execution',
      ProtocolConfiguration: 'HTTP',
      NetworkConfiguration: { NetworkMode: 'PUBLIC' },
      AgentRuntimeArtifact: {
        ContainerConfiguration: {
          ContainerUri: Match.objectLike({ 'Fn::Sub': Match.stringLikeRegexp('.*\\.dkr\\.ecr\\..*') }),
        },
      },
      RoleArn: {
        'Fn::GetAtt': [
          Match.stringLikeRegexp('testruntimelocalExecutionRole*'),
          'Arn',
        ],
      },
    });
  });

  test('Should have execution role with correct permissions', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
          },
        ],
      },
    });
  });

  test('Should have execution role with correct trust policy', () => {
    // Check that the execution role has the correct trust policy with confused deputy conditions
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: {
              Service: 'bedrock-agentcore.amazonaws.com',
            },
            Condition: {
              StringEquals: { 'aws:SourceAccount': '123456789012' },
              ArnLike: {
                'aws:SourceArn': {
                  'Fn::Join': ['', Match.arrayWith([
                    ':bedrock-agentcore:us-east-1:123456789012:runtime/test_runtime_local*',
                  ])],
                },
              },
            },
          },
        ],
      },
    });
  });
});

describe('Runtime static methods tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should import runtime from attributes', () => {
    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    const imported = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntime', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
      agentRuntimeId: 'test-runtime-id',
      agentRuntimeName: 'test-runtime',
      roleArn: role.roleArn,
      agentRuntimeVersion: '1',
      description: 'Imported runtime',
    });

    expect(imported.agentRuntimeArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id');
    expect(imported.agentRuntimeId).toBe('test-runtime-id');
    expect(imported.agentRuntimeName).toBe('test-runtime');
    // Since we create the role from ARN, check the ARN instead of object reference
    expect(imported.role.roleArn).toBe(role.roleArn);
    expect(imported.agentRuntimeVersion).toBe('1');
    // description is optional and may not exist on the interface
    expect(imported.agentStatus).toBeUndefined();
    expect(imported.createdAt).toBeUndefined();
    expect(imported.lastUpdatedAt).toBeUndefined();
  });

  test('Should import runtime with optional attributes', () => {
    const imported = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntimeOptional', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
      agentRuntimeId: 'test-runtime-id',
      agentRuntimeName: 'test-runtime',
      roleArn: 'arn:aws:iam::123456789012:role/test-role',
      agentRuntimeVersion: '2',
      agentStatus: 'ACTIVE',
      createdAt: '2024-01-15T10:30:00Z',
      lastUpdatedAt: '2024-01-15T14:45:00Z',
    });

    expect(imported.agentRuntimeArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id');
    expect(imported.agentRuntimeId).toBe('test-runtime-id');
    expect(imported.agentRuntimeName).toBe('test-runtime');
    expect(imported.role.roleArn).toBe('arn:aws:iam::123456789012:role/test-role');
    expect(imported.agentRuntimeVersion).toBe('2');
    expect(imported.agentStatus).toBe('ACTIVE');
    expect(imported.createdAt).toBe('2024-01-15T10:30:00Z');
    expect(imported.lastUpdatedAt).toBe('2024-01-15T14:45:00Z');
  });
});

describe('Runtime with OAuth authorizer tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create runtime with OAuth authorizer configuration', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_oauth',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingOAuth(
        'https://oauth.example.com/.well-known/openid-configuration',
        'oauth-client-123',
        ['audience1'],
      ),
    });

    const template = Template.fromStack(stack);

    // Should have runtime resource
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);

    // Verify the runtime was created
    expect(runtime.agentRuntimeName).toBe('test_runtime_oauth');
  });

  test('Should render OAuth authorizer configuration correctly', () => {
    new Runtime(stack, 'test-runtime-oauth-render', {
      runtimeName: 'test_runtime_oauth_render',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingOAuth(
        'https://oauth.provider.com/.well-known/openid-configuration',
        'oauth-client-456',
        ['aud1', 'aud2'],
        ['openid', 'profile'],
      ),
    });

    const template = Template.fromStack(stack);

    // Verify the OAuth configuration is properly rendered
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: 'https://oauth.provider.com/.well-known/openid-configuration',
          AllowedClients: ['oauth-client-456'],
          AllowedAudience: ['aud1', 'aud2'],
          AllowedScopes: ['openid', 'profile'],
        },
      },
    });
  });

  test('Should create OAuth configuration without audience', () => {
    const runtime = new Runtime(stack, 'test-runtime-oauth-no-aud', {
      runtimeName: 'test_runtime_oauth_no_aud',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingOAuth(
        'https://oauth2.example.com/.well-known/openid-configuration',
        'client-789',
      ),
    });

    const template = Template.fromStack(stack);
    expect(runtime.agentRuntimeName).toBe('test_runtime_oauth_no_aud');
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);
  });
});

describe('Runtime with JWT authorizer tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create runtime with JWT authorizer', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_jwt',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingJWT(
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1'],
      ),
    });

    const template = Template.fromStack(stack);

    // Should have runtime resource
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);

    // Verify the runtime was created
    expect(runtime.agentRuntimeName).toBe('test_runtime_jwt');
  });

  test('Should create runtime with JWT authorizer including allowedScopes', () => {
    new Runtime(stack, 'test-runtime-scopes', {
      runtimeName: 'test_runtime_jwt_scopes',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingJWT(
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1'],
        ['audience1'],
        ['read', 'write', 'admin'],
      ),
    });

    const template = Template.fromStack(stack);

    // Verify the JWT configuration with scopes is properly rendered
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: 'https://auth.example.com/.well-known/openid-configuration',
          AllowedClients: ['client1'],
          AllowedAudience: ['audience1'],
          AllowedScopes: ['read', 'write', 'admin'],
        },
      },
    });
  });

  test('Should create runtime with JWT authorizer without allowedScopes (optional parameter)', () => {
    new Runtime(stack, 'test-runtime-no-scopes', {
      runtimeName: 'test_runtime_jwt_no_scopes',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingJWT(
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1'],
        ['audience1'],
        // allowedScopes not provided - should work fine
      ),
    });

    const template = Template.fromStack(stack);

    // Verify the JWT configuration without scopes is properly rendered
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: {
          DiscoveryUrl: 'https://auth.example.com/.well-known/openid-configuration',
          AllowedClients: ['client1'],
          AllowedAudience: ['audience1'],
          AllowedScopes: Match.absent(),
          CustomClaims: Match.absent(),
        },
      },
    });
  });
});

describe('Runtime with Custom Claims tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should render authorizer configuration with custom claims', () => {
    const stringClaim = RuntimeCustomClaim.withStringValue('department', 'engineering');
    const arrayClaim = RuntimeCustomClaim.withStringArrayValue('roles', ['admin']);

    new Runtime(stack, 'test-runtime-render-custom-claims', {
      runtimeName: 'test_runtime_render_custom_claims',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingJWT (
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1'],
        ['audience1'],
        ['read'],
        [stringClaim, arrayClaim],
      ),
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: Match.objectLike({
          CustomClaims: [
            {
              InboundTokenClaimName: 'department',
              InboundTokenClaimValueType: 'STRING',
              AuthorizingClaimMatchValue: {
                ClaimMatchOperator: 'EQUALS',
                ClaimMatchValue: { MatchValueString: 'engineering' },
              },
            },
            {
              InboundTokenClaimName: 'roles',
              InboundTokenClaimValueType: 'STRING_ARRAY',
              AuthorizingClaimMatchValue: {
                ClaimMatchOperator: 'CONTAINS',
                ClaimMatchValue: { MatchValueString: 'admin' },
              },
            },
          ],
        }),
      },
    });
  });

  test('Should create RuntimeCustomClaim with string array value (default CONTAINS)', () => {
    const claim = RuntimeCustomClaim.withStringArrayValue('roles', ['admin']);
    const rendered = claim._render();

    expect(rendered.inboundTokenClaimName).toBe('roles');
    expect(rendered.inboundTokenClaimValueType).toBe('STRING_ARRAY');
    const matchValue = rendered.authorizingClaimMatchValue as any;
    expect(matchValue.claimMatchOperator).toBe('CONTAINS');
    expect(matchValue.claimMatchValue.matchValueString).toBe('admin');
    expect(matchValue.claimMatchValue.matchValueStringList).toBeUndefined();
  });

  test('Should create RuntimeCustomClaim with string array value (CONTAINS_ANY)', () => {
    const claim = RuntimeCustomClaim.withStringArrayValue('permissions', ['read', 'write'], CustomClaimOperator.CONTAINS_ANY);
    const rendered = claim._render();

    expect(rendered.inboundTokenClaimName).toBe('permissions');
    expect(rendered.inboundTokenClaimValueType).toBe('STRING_ARRAY');
    const matchValue = rendered.authorizingClaimMatchValue as any;
    expect(matchValue.claimMatchOperator).toBe('CONTAINS_ANY');
    expect(matchValue.claimMatchValue.matchValueStringList).toEqual(['read', 'write']);
  });

  test('Should throw error for invalid operator with string array', () => {
    expect(() => {
      RuntimeCustomClaim.withStringArrayValue('roles', ['admin'], CustomClaimOperator.EQUALS);
    }).toThrow('STRING_ARRAY type only supports CONTAINS or CONTAINS_ANY operators');
  });

  test('Should throw error when CONTAINS operator is used with multiple values', () => {
    const claim = RuntimeCustomClaim.withStringArrayValue('roles', ['admin', 'user'], CustomClaimOperator.CONTAINS);
    expect(() => {
      claim._render();
    }).toThrow('CONTAINS operator requires exactly one value, got 2 values');
  });

  test('Should create runtime with JWT authorizer and custom claims', () => {
    const stringClaim = RuntimeCustomClaim.withStringValue('department', 'engineering');
    const arrayClaim = RuntimeCustomClaim.withStringArrayValue('roles', ['admin', 'user'], CustomClaimOperator.CONTAINS_ANY);

    new Runtime(stack, 'test-runtime-custom-claims', {
      runtimeName: 'test_runtime_custom_claims',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: (RuntimeAuthorizerConfiguration.usingJWT as any)(
        'https://auth.example.com/.well-known/openid-configuration',
        ['client1'],
        ['audience1'],
        ['read'],
        [stringClaim, arrayClaim],
      ),
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: Match.objectLike({
          CustomClaims: [
            {
              InboundTokenClaimName: 'department',
              InboundTokenClaimValueType: 'STRING',
              AuthorizingClaimMatchValue: {
                ClaimMatchOperator: 'EQUALS',
                ClaimMatchValue: { MatchValueString: 'engineering' },
              },
            },
            {
              InboundTokenClaimName: 'roles',
              InboundTokenClaimValueType: 'STRING_ARRAY',
              AuthorizingClaimMatchValue: {
                ClaimMatchOperator: 'CONTAINS_ANY',
                ClaimMatchValue: { MatchValueStringList: ['admin', 'user'] },
              },
            },
          ],
        }),
      },
    });
  });

  test('Should create runtime with Cognito authorizer and custom claims', () => {
    const userPool = new cognito.UserPool(stack, 'TestUserPool', {
      userPoolName: 'test-pool',
    });
    const userPoolClient = userPool.addClient('TestClient');

    const stringClaim = RuntimeCustomClaim.withStringValue('team', 'backend');
    const arrayClaim = RuntimeCustomClaim.withStringArrayValue('permissions', ['read', 'write'], CustomClaimOperator.CONTAINS_ANY);

    new Runtime(stack, 'test-runtime-cognito-claims', {
      runtimeName: 'test_runtime_cognito_claims',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: (RuntimeAuthorizerConfiguration.usingCognito as any)(
        userPool,
        [userPoolClient],
        ['audience1'],
        ['read'],
        [stringClaim, arrayClaim],
      ),
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: Match.objectLike({
          CustomClaims: [
            Match.objectLike({ InboundTokenClaimName: 'team', InboundTokenClaimValueType: 'STRING' }),
            Match.objectLike({ InboundTokenClaimName: 'permissions', InboundTokenClaimValueType: 'STRING_ARRAY' }),
          ],
        }),
      },
    });
  });

  test('Should create runtime with OAuth authorizer and custom claims', () => {
    const stringClaim = RuntimeCustomClaim.withStringValue('org', 'acme');

    new Runtime(stack, 'test-runtime-oauth-claims', {
      runtimeName: 'test_runtime_oauth_claims',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: (RuntimeAuthorizerConfiguration.usingOAuth as any)(
        'https://oauth.example.com/.well-known/openid-configuration',
        'oauth-client-123',
        ['audience1'],
        ['read'],
        [stringClaim],
      ),
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AuthorizerConfiguration: {
        CustomJWTAuthorizer: Match.objectLike({
          CustomClaims: [
            Match.objectLike({ InboundTokenClaimName: 'org', InboundTokenClaimValueType: 'STRING' }),
          ],
        }),
      },
    });
  });
});

describe('Runtime addEndpoint tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should add endpoint to runtime', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    runtime.addEndpoint('test_endpoint', {
      description: 'Test endpoint',
      version: '2',
    });

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', {
      Name: 'test_endpoint',
      Description: 'Test endpoint',
      AgentRuntimeVersion: '2',
    });
  });

  test('Should fall back to the runtime\'s agentRuntimeVersion when version is not provided', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    runtime.addEndpoint('test_endpoint');

    const template = Template.fromStack(stack);
    // When options.version is omitted, the endpoint uses the runtime resource's AgentRuntimeVersion attribute
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', {
      Name: 'test_endpoint',
      AgentRuntimeVersion: Match.objectLike({
        'Fn::GetAtt': Match.arrayWith([Match.stringLikeRegexp('testruntime.*'), 'AgentRuntimeVersion']),
      }),
    });
  });

  test('Should add multiple endpoints to the same runtime', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    runtime.addEndpoint('endpoint_a');
    runtime.addEndpoint('endpoint_b');
    runtime.addEndpoint('endpoint_c');

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', { Name: 'endpoint_a' });
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', { Name: 'endpoint_b' });
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', { Name: 'endpoint_c' });
  });

  test('Should create a DependsOn from the endpoint to the runtime resource', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    runtime.addEndpoint('dependent_endpoint');

    const template = Template.fromStack(stack);
    template.hasResource('AWS::BedrockAgentCore::RuntimeEndpoint', {
      Properties: { Name: 'dependent_endpoint' },
      DependsOn: Match.arrayWith([Match.stringLikeRegexp('testruntime.*')]),
    });
  });
});

describe('Runtime with tags tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create runtime with valid tags', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      tags: {
        'Environment': 'Production',
        'Team': 'Platform',
        'Cost-Center': '12345',
      },
    });

    const template = Template.fromStack(stack);
    expect(runtime.agentRuntimeName).toBe('test_runtime');

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'test_runtime',
      Tags: {
        'Environment': 'Production',
        'Team': 'Platform',
        'Cost-Center': '12345',
      },
    });
  });

  test('Should throw error for null tag value', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        tags: {
          TestKey: null as any,
        },
      });
    }).toThrow('Tag value for key "TestKey" cannot be null or undefined');
  });

  test('Should throw error for undefined tag value', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        tags: {
          TestKey: undefined as any,
        },
      });
    }).toThrow('Tag value for key "TestKey" cannot be null or undefined');
  });
});

// OAuth instance method tests removed as these methods no longer exist

describe('Runtime authentication configuration error cases', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should throw error for JWT with invalid discovery URL', () => {
    expect(() => {
      RuntimeAuthorizerConfiguration.usingJWT(
        'https://auth.example.com/invalid', // Doesn't end with /.well-known/openid-configuration
        ['client1'],
      );
    }).toThrow('JWT discovery URL must end with /.well-known/openid-configuration');
  });

  test('Should create runtime with Cognito auth using static factory', () => {
    const userPool = new cognito.UserPool(stack, 'MyUserPool', {
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const userPoolClient = userPool.addClient('MyUserPoolClient', {});

    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingCognito(
        userPool,
        [userPoolClient],
      ),
    });

    expect(runtime.agentRuntimeName).toBe('test_runtime');
  });

  test('Should throw error for OAuth with invalid discovery URL', () => {
    expect(() => {
      RuntimeAuthorizerConfiguration.usingOAuth(
        'custom',
        'https://oauth.example.com/invalid', // Doesn't end with /.well-known/openid-configuration
        ['oauth-client-123'],
      );
    }).toThrow('OAuth discovery URL must end with /.well-known/openid-configuration');
  });

  test('Should work with IAM authentication', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      authorizerConfiguration: RuntimeAuthorizerConfiguration.usingIAM(),
    });

    expect(runtime.agentRuntimeName).toBe('test_runtime');
  });

  test('does not fail validation if JWT discovery URL is a late-bound value', () => {
    // WHEN
    const discoveryUrlParam = new cdk.CfnParameter(stack, 'JWTDiscoveryUrl', {
      default: 'https://example.com/.well-known/openid-configuration',
      type: 'String',
    });

    // THEN
    expect(() => {
      RuntimeAuthorizerConfiguration.usingJWT(discoveryUrlParam.valueAsString);
    }).not.toThrow();
  });

  test('does not fail validation if OAuth discovery URL is a late-bound value', () => {
    // WHEN
    const discoveryUrlParam = new cdk.CfnParameter(stack, 'OAuthDiscoveryUrl', {
      default: 'https://oauth-provider.com/.well-known/openid-configuration',
      type: 'String',
    });

    // THEN
    expect(() => {
      RuntimeAuthorizerConfiguration.usingOAuth(
        discoveryUrlParam.valueAsString,
        'oauth-client-123',
      );
    }).not.toThrow();
  });
});

describe('RuntimeNetworkConfiguration tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create runtime with usingPublicNetwork() method', () => {
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      networkConfiguration: RuntimeNetworkConfiguration.usingPublicNetwork(),
    });

    const template = Template.fromStack(stack);

    // Should have runtime resource
    template.resourceCountIs('AWS::BedrockAgentCore::Runtime', 1);

    // Verify the runtime was created
    expect(runtime.agentRuntimeName).toBe('test_runtime');
  });
});

describe('Runtime metrics and grant methods tests', () => {
  let stack: cdk.Stack;
  let runtime: Runtime;

  function alarmForMetric(id: string, metric: cloudwatch.Metric): void {
    new cloudwatch.Alarm(stack, id, { metric, evaluationPeriods: 1, threshold: 1 });
  }

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');

    const repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });
  });

  test('metricInvocations() produces Invocations with Sum statistic', () => {
    alarmForMetric('InvocAlarm', runtime.metricInvocations());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      // Dimensions synth in alphabetical order by Name. Assert each dimension
      // in its own arrayWith, not in one ordered array.
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeAgentRuntime' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Name', Value: Match.stringLikeRegexp('.*::DEFAULT$') }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('.*'), 'AgentRuntimeArn'] } }),
      ]),
    });
    // Regression guard: the buggy shape emitted a `Service` dimension.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Service' }),
      ])),
    });
  });

  test('metricLatency() emits per-resource Operation/Name/Resource dimensions and no Service', () => {
    alarmForMetric('LatencyDimAlarm', runtime.metricLatency());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'InvokeAgentRuntime' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Name', Value: Match.stringLikeRegexp('.*::DEFAULT$') }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('.*'), 'AgentRuntimeArn'] } }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Service' }),
      ])),
    });
  });

  test('metricInvocationsAggregated() produces Invocations with only the AggregateOperation dimension', () => {
    alarmForMetric('InvocAggAlarm', runtime.metricInvocationsAggregated());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      // Aggregated metrics carry one dimension: AggregateOperation.
      Dimensions: [{ Name: 'AggregateOperation', Value: 'InvokeAgentRuntime' }],
    });
    // Guard: per-resource dimensions must be absent on the aggregated metric.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Operation' }),
      ])),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Name' }),
      ])),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Resource' }),
      ])),
    });
  });

  test('metricInvocationsAggregated() merges a caller-supplied dimension onto the AggregateOperation path', () => {
    alarmForMetric('InvocAggOverrideAlarm', runtime.metricInvocationsAggregated({
      dimensionsMap: { Foo: 'bar' },
    }));

    const template = Template.fromStack(stack);
    // The caller dimension merges onto the aggregated metric.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Foo', Value: 'bar' }),
      ]),
    });
    // The AggregateOperation dimension stays present with the override.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'AggregateOperation', Value: 'InvokeAgentRuntime' }),
      ]),
    });
  });

  test('metricThrottles() produces Throttles with Sum statistic', () => {
    alarmForMetric('ThrottlesAlarm', runtime.metricThrottles());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Throttles',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricSystemErrors() produces SystemErrors with Sum statistic', () => {
    alarmForMetric('SysErrAlarm', runtime.metricSystemErrors());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'SystemErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricUserErrors() produces UserErrors with Sum statistic', () => {
    alarmForMetric('UserErrAlarm', runtime.metricUserErrors());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'UserErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricLatency() produces Latency with Average statistic', () => {
    alarmForMetric('LatencyAlarm', runtime.metricLatency());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Latency',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Average',
    });
  });

  test('metricTotalErrors() produces TotalErrors with Sum statistic', () => {
    alarmForMetric('TotalErrAlarm', runtime.metricTotalErrors());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'TotalErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricSessionCount() produces SessionCount with Sum statistic', () => {
    alarmForMetric('SessionCountAlarm', runtime.metricSessionCount());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'SessionCount',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
    });
  });

  test('metricSessionsAggregated() produces Sessions with only the AggregateOperation dimension', () => {
    alarmForMetric('SessionsAggAlarm', runtime.metricSessionsAggregated());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Sessions',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: [{ Name: 'AggregateOperation', Value: 'InvokeAgentRuntime' }],
    });
    // Guard: per-resource dimensions must be absent on the aggregated metric.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Sessions',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Operation' }),
      ])),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Sessions',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Name' }),
      ])),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Sessions',
      Dimensions: Match.not(Match.arrayWith([
        Match.objectLike({ Name: 'Resource' }),
      ])),
    });
  });

  test('caller-supplied dimensionsMap overrides the per-resource defaults', () => {
    alarmForMetric('OverrideAlarm', runtime.metricInvocations({
      dimensionsMap: { Operation: 'CustomOp', Resource: 'custom-resource' },
    }));

    const template = Template.fromStack(stack);
    // Overridden dimensions win (order-independent per-dimension assertions).
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'CustomOp' }),
      ]),
    });
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Resource', Value: 'custom-resource' }),
      ]),
    });
    // The un-overridden default (Name) is still present.
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Name', Value: Match.stringLikeRegexp('.*::DEFAULT$') }),
      ]),
    });
  });

  test('unnamed runtime resolves Name to a concrete synth string ending ::DEFAULT', () => {
    // A runtime without runtimeName gets a Lazy token (Names.uniqueResourceName)
    // that resolves at synth to a plain string, NOT an Fn::Join.
    const repository = new ecr.Repository(stack, 'TokenRepository', {
      repositoryName: 'token-agent-runtime',
    });
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
    const unnamed = new Runtime(stack, 'unnamed-runtime', {
      agentRuntimeArtifact,
    });

    alarmForMetric('TokenNameAlarm', unnamed.metricInvocations());

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Invocations',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Name', Value: Match.stringLikeRegexp('^[^{]*::DEFAULT$') }),
      ]),
    });
  });

  test('Should grant invoke permissions', () => {
    const grantee = new iam.Role(stack, 'GranteeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    const grant = runtime.grantInvoke(grantee);
    expect(grant).toBeDefined();
  });

  test('Should grant invoke for user permissions', () => {
    const grantee = new iam.Role(stack, 'GranteeRoleForUser', {
      assumedBy: new iam.ServicePrincipal('apigateway.amazonaws.com'),
    });

    const grant = runtime.grantInvokeRuntimeForUser(grantee);
    expect(grant).toBeDefined();
  });

  test('Should grant all invoke permissions', () => {
    const grantee = new iam.Role(stack, 'GranteeRoleAll', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    const grant = runtime.grantInvoke(grantee);
    expect(grant).toBeDefined();
  });

  test('Should grant custom actions to runtime role', () => {
    const grant = runtime.grant(['bedrock:InvokeRuntime'], ['arn:aws:bedrock:*:*:*']);
    expect(grant).toBeDefined();
  });

  test('Should add policy statement to runtime role', () => {
    const result = runtime.addToRolePolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: ['arn:aws:s3:::bucket/*'],
    }));
    expect(result).toBe(runtime);

    // Can call multiple times
    const result2 = runtime.addToRolePolicy(new iam.PolicyStatement({
      actions: ['dynamodb:Query'],
      resources: ['arn:aws:dynamodb:us-east-1:123456789012:table/test-table'],
    }));
    expect(result2).toBe(runtime);
  });

  test('Should add policy to imported runtime role', () => {
    // Create an imported runtime with IRole (not Role)
    const importedRole = iam.Role.fromRoleArn(stack, 'ImportedRole',
      'arn:aws:iam::123456789012:role/imported-role');

    const imported = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntime', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
      agentRuntimeId: 'test-runtime-id',
      agentRuntimeName: 'test-runtime',
      roleArn: importedRole.roleArn,
      agentRuntimeVersion: '1',
    });

    const result = imported.addToRolePolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject'],
      resources: ['arn:aws:s3:::bucket/*'],
    }));
    expect(result).toBe(imported);

    // Can call multiple times
    const result2 = imported.addToRolePolicy(new iam.PolicyStatement({
      actions: ['dynamodb:Query'],
      resources: ['arn:aws:dynamodb:us-east-1:123456789012:table/test-table'],
    }));
    expect(result2).toBe(imported);
  });
});

describe('Runtime with VPC network configuration tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create runtime with VPC network configuration', () => {
    const vpc = new ec2.Vpc(stack, 'TestVpc', {
      maxAzs: 2,
    });

    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_vpc',
      agentRuntimeArtifact: agentRuntimeArtifact,
      networkConfiguration: RuntimeNetworkConfiguration.usingVpc(stack, {
        vpc: vpc,
      }),
    });

    expect(runtime.agentRuntimeName).toBe('test_runtime_vpc');
    expect(runtime.connections).toBeDefined();
    expect(runtime.connections?.securityGroups).toHaveLength(1);
  });

  test('Should import runtime with security groups', () => {
    const vpc = new ec2.Vpc(stack, 'TestVpc', {
      maxAzs: 2,
    });

    const sg = new ec2.SecurityGroup(stack, 'TestSG', {
      vpc: vpc,
    });

    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    const imported = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntime', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
      agentRuntimeId: 'test-runtime-id',
      agentRuntimeName: 'test-runtime',
      roleArn: role.roleArn,
      agentRuntimeVersion: '1',
      securityGroups: [sg],
    });

    expect(imported.connections).toBeDefined();
    expect(imported.connections?.securityGroups).toContain(sg);
  });
});

describe('Runtime description validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should throw error for description too long', () => {
    const longDescription = 'a'.repeat(1201);
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        description: longDescription,
      });
    }).toThrow(/The field Description is 1201 characters long but must be less than or equal to 1200 characters/);
  });

  test('Should accept valid description', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        description: 'Valid description for the runtime',
      });
    }).not.toThrow();
  });
});

describe('Runtime grantInvokeRuntime permission tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;
  let runtime: Runtime;
  let granteeRole: iam.Role;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');

    runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime_grant',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    granteeRole = new iam.Role(stack, 'GranteeRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });
  });

  test('Should grant InvokeAgentRuntime permission with grantInvokeRuntime', () => {
    runtime.grantInvokeRuntime(granteeRole);

    const template = Template.fromStack(stack);

    // Verify that a policy is created with the correct permission
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeAgentRuntime',
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  Match.stringLikeRegexp('testruntime.*'),
                  'AgentRuntimeArn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        Match.stringLikeRegexp('testruntime.*'),
                        'AgentRuntimeArn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            ],
          }),
        ]),
      },
    });
  });

  test('Should grant InvokeAgentRuntimeForUser permission with grantInvokeRuntimeForUser', () => {
    runtime.grantInvokeRuntimeForUser(granteeRole);

    const template = Template.fromStack(stack);

    // Verify that a policy is created with the correct permission
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeAgentRuntimeForUser',
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  Match.stringLikeRegexp('testruntime.*'),
                  'AgentRuntimeArn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        Match.stringLikeRegexp('testruntime.*'),
                        'AgentRuntimeArn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            ],
          }),
        ]),
      },
    });
  });

  test('Should grant both permissions with grantInvoke', () => {
    runtime.grantInvoke(granteeRole);

    const template = Template.fromStack(stack);

    // Verify that a policy is created with both permissions
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: [
              'bedrock-agentcore:InvokeAgentRuntime',
              'bedrock-agentcore:InvokeAgentRuntimeForUser',
            ],
            Effect: 'Allow',
            Resource: [
              {
                'Fn::GetAtt': [
                  Match.stringLikeRegexp('testruntime.*'),
                  'AgentRuntimeArn',
                ],
              },
              {
                'Fn::Join': [
                  '',
                  [
                    {
                      'Fn::GetAtt': [
                        Match.stringLikeRegexp('testruntime.*'),
                        'AgentRuntimeArn',
                      ],
                    },
                    '/*',
                  ],
                ],
              },
            ],
          }),
        ]),
      },
    });
  });

  test('Should attach policy to grantee role', () => {
    runtime.grantInvokeRuntime(granteeRole);

    const template = Template.fromStack(stack);

    // Verify that the policy is attached to the grantee role
    template.hasResourceProperties('AWS::IAM::Policy', {
      Roles: [
        {
          Ref: Match.stringLikeRegexp('GranteeRole.*'),
        },
      ],
    });
  });

  test('Should work with imported runtime', () => {
    const importedRuntime = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntime', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
      agentRuntimeId: 'test-runtime-id',
      agentRuntimeName: 'test-runtime',
      roleArn: 'arn:aws:iam::123456789012:role/test-role',
      agentRuntimeVersion: '1',
    });

    importedRuntime.grantInvokeRuntime(granteeRole);

    const template = Template.fromStack(stack);

    // Verify that a policy is created for the imported runtime
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeAgentRuntime',
            Effect: 'Allow',
            Resource: [
              'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id',
              'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime-id/*',
            ],
          }),
        ]),
      },
    });
  });

  test('Should grant to Lambda function', () => {
    const lambdaFunction = new lambda.Function(stack, 'TestLambda', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromInline('def handler(event, context): pass'),
    });

    runtime.grantInvokeRuntime(lambdaFunction);

    const template = Template.fromStack(stack);

    // Verify that the Lambda function's role has the permission
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeAgentRuntime',
            Effect: 'Allow',
          }),
        ]),
      },
      Roles: [
        {
          Ref: Match.stringLikeRegexp('TestLambdaServiceRole.*'),
        },
      ],
    });
  });

  test('Should grant multiple permissions to same role', () => {
    runtime.grantInvokeRuntime(granteeRole);
    runtime.grantInvokeRuntimeForUser(granteeRole);

    const template = Template.fromStack(stack);

    // Should have two separate policies
    template.resourceCountIs('AWS::IAM::Policy', 2);
  });

  test('Should return Grant object with success', () => {
    const grant = runtime.grantInvokeRuntime(granteeRole);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatement).toBeDefined();
  });

  test('Should work with runtime that has custom execution role', () => {
    const customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    const customRuntime = new Runtime(stack, 'custom-runtime', {
      runtimeName: 'custom_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      executionRole: customRole,
    });

    const lambdaRole = new iam.Role(stack, 'LambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
    });

    customRuntime.grantInvokeRuntime(lambdaRole);

    const template = Template.fromStack(stack);

    // Verify policy is created for the lambda role
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'bedrock-agentcore:InvokeAgentRuntime',
            Effect: 'Allow',
          }),
        ]),
      },
      Roles: [
        {
          Ref: Match.stringLikeRegexp('LambdaRole.*'),
        },
      ],
    });
  });
});

describe('Runtime role validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should validate invalid role ARN format', () => {
    const invalidRole = {
      roleArn: 'invalid-arn-format',
      roleName: 'test-role',
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: '123456789012',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Invalid IAM role ARN format/);
  });

  test('Should validate role ARN with invalid service', () => {
    const invalidRole = {
      roleArn: 'arn:aws:s3:::bucket/key',
      roleName: 'test-role',
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: '123456789012',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Invalid IAM role ARN format/);
  });

  test('Should validate role ARN with invalid account ID', () => {
    const invalidRole = {
      roleArn: 'arn:aws:iam::invalid:role/test-role',
      roleName: 'test-role',
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: 'invalid',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Invalid IAM role ARN format/);
  });

  test('Should validate role ARN with missing role name', () => {
    const invalidRole = {
      roleArn: 'arn:aws:iam::123456789012:role/',
      roleName: '',
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: '123456789012',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Invalid IAM role ARN format/);
  });

  test('Should validate role name exceeding maximum length', () => {
    const longRoleName = 'a'.repeat(65);
    const invalidRole = {
      roleArn: `arn:aws:iam::123456789012:role/${longRoleName}`,
      roleName: longRoleName,
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: '123456789012',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Role name exceeds maximum length of 64 characters/);
  });

  test('Should validate role ARN with invalid resource type', () => {
    const invalidRole = {
      roleArn: 'arn:aws:iam::123456789012:user/test-user',
      roleName: 'test-user',
      assumeRoleAction: 'sts:AssumeRole',
      grantPrincipal: {} as iam.IPrincipal,
      principalAccount: '123456789012',
      applyRemovalPolicy: () => {},
      node: {} as any,
      stack: stack,
      env: { account: '123456789012', region: 'us-east-1' },
    } as unknown as iam.IRole;

    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        executionRole: invalidRole,
      });
    }).toThrow(/Invalid IAM role ARN format/);
  });

  test('Should handle cross-account role with no warnings', () => {
    const crossAccountStack = new cdk.Stack(app, 'cross-account-stack', {
      env: {
        account: '111111111111',
        region: 'us-east-1',
      },
    });
    const crossAccountRole = new iam.Role(crossAccountStack, 'ImportedCrossAccountRole', {
      roleName: cdk.PhysicalName.GENERATE_IF_NEEDED,
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      executionRole: crossAccountRole,
    });

    // Should not throw
    expect(runtime.role).toBe(crossAccountRole);
  });

  test('Should handle cross-account imported role with warning', () => {
    const crossAccountRole = iam.Role.fromRoleArn(stack, 'ImportedCrossAccountRole',
      'arn:aws:iam::111111111111:role/test-cross-account-role');

    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      executionRole: crossAccountRole,
    });

    // Should not throw, just add warning
    expect(runtime.role).toBe(crossAccountRole);

    cdk.Validations.of(app).acknowledge({ id: 'CloudFormation-Validate::W9002', reason: 'Testing hardcoded ARN for cross-account role' });

    const annotations = Annotations.fromStack(stack).findWarning('*', Match.stringLikeRegexp('.*different account.*cross-account.*'));
    expect(annotations.length).toBe(1);

    Annotations.fromStack(stack).hasWarning('/test-stack/test-runtime', 'IAM role is from a different account (111111111111) than the stack account (123456789012). Ensure cross-account permissions are properly configured.');
  });
});

describe('Runtime lifecycle configuration tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should use default lifecycle configuration when not specified', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    // lifecycleConfigurationが指定されていない場合、undefinedプロパティは除外される
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      LifecycleConfiguration: {
        IdleRuntimeSessionTimeout: Match.absent(),
        MaxLifetime: Match.absent(),
      },
    });
  });

  test('Should set custom lifecycle configuration', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: Duration.minutes(10),
        maxLifetime: Duration.hours(4),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      LifecycleConfiguration: {
        IdleRuntimeSessionTimeout: 600,
        MaxLifetime: 14400,
      },
    });
  });

  test('Should set only idleRuntimeSessionTimeout', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: Duration.minutes(15),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      LifecycleConfiguration: {
        IdleRuntimeSessionTimeout: 900,
        MaxLifetime: Match.absent(),
      },
    });
  });

  test('Should set only maxLifetime', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      lifecycleConfiguration: {
        maxLifetime: Duration.hours(6),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      LifecycleConfiguration: {
        IdleRuntimeSessionTimeout: Match.absent(),
        MaxLifetime: 21600,
      },
    });
  });

  test('Should throw error for idleRuntimeSessionTimeout below minimum', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        lifecycleConfiguration: {
          idleRuntimeSessionTimeout: Duration.seconds(30),
        },
      });
    }).toThrow(/Idle runtime session timeout must be between 60 seconds and 28800 seconds/);
  });

  test('Should throw error for idleRuntimeSessionTimeout above maximum', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        lifecycleConfiguration: {
          idleRuntimeSessionTimeout: Duration.hours(9),
        },
      });
    }).toThrow(/Idle runtime session timeout must be between 60 seconds and 28800 seconds/);
  });

  test('Should throw error for maxLifetime below minimum', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        lifecycleConfiguration: {
          maxLifetime: Duration.seconds(30),
        },
      });
    }).toThrow(/Maximum lifetime must be between 60 seconds and 28800 seconds/);
  });

  test('does not fail validation if lifecycle configuration is a late-bound value', () => {
    // WHEN
    const idleTimeoutParam = new cdk.CfnParameter(stack, 'IdleTimeout', {
      default: 600,
      type: 'Number',
    });

    const maxLifetimeParam = new cdk.CfnParameter(stack, 'MaxLifetime', {
      default: 14400,
      type: 'Number',
    });

    // THEN
    expect(() => {
      new Runtime(stack, 'runtime-late-bound', {
        runtimeName: 'runtime_late_bound',
        agentRuntimeArtifact: agentRuntimeArtifact,
        lifecycleConfiguration: {
          idleRuntimeSessionTimeout: Duration.seconds(idleTimeoutParam.valueAsNumber),
          maxLifetime: Duration.seconds(maxLifetimeParam.valueAsNumber),
        },
      });
    }).not.toThrow();
  });
});

describe('Runtime request header configuration tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let repository: ecr.Repository;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });
    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should not include request header configuration when not specified', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      RequestHeaderConfiguration: Match.absent(),
    });
  });

  test('Should set request header configuration with allowList', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      requestHeaderConfiguration: {
        allowlistedHeaders: ['Authorization', 'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1'],
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      RequestHeaderConfiguration: {
        RequestHeaderAllowlist: ['Authorization', 'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1'],
      },
    });
  });

  test('Should set request header configuration with single header', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      requestHeaderConfiguration: {
        allowlistedHeaders: ['Authorization'],
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      RequestHeaderConfiguration: {
        RequestHeaderAllowlist: ['Authorization'],
      },
    });
  });

  test('Should set request header configuration with multiple custom headers', () => {
    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      requestHeaderConfiguration: {
        allowlistedHeaders: [
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1',
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header2',
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header3',
        ],
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      RequestHeaderConfiguration: {
        RequestHeaderAllowlist: [
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1',
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header2',
          'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header3',
        ],
      },
    });
  });

  test('Should throw error for empty allowList', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: [],
        },
      });
    }).toThrow(/Request header allow list contain between 1 and 20 headers/);
  });

  test('Should throw error for allowList exceeding 20 headers', () => {
    const longList = Array.from({ length: 21 }, (_, i) => `X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header${i + 1}`);
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: longList,
        },
      });
    }).toThrow(/Request header allow list contain between 1 and 20 headers/);
  });

  test('Should throw error for invalid header pattern', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: ['Invalid-Header@Name'],
        },
      });
    }).toThrow(/Request header must start with a letter and contain only letters, numbers, underscores, and hyphens/);
  });

  test('Should throw error for empty header name', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: [''],
        },
      });
    }).toThrow(/The field Request header is 0 characters long but must be at least 1 characters/);
  });

  test('Should accept headers beyond the X-Amzn-Bedrock-AgentCore-Runtime-Custom- prefix', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: [
            'X-Custom-Auth',
            'X-Request-Signature',
            'X-Amzn-Bedrock-AgentCore-Runtime-Custom-MyHeader',
            'Authorization',
          ],
        },
      });
    }).not.toThrow();
  });

  test('Should throw error for header with spaces', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: ['Invalid Header With Spaces'],
        },
      });
    }).toThrow(/Request header must start with a letter and contain only letters, numbers, underscores, and hyphens/);
  });

  test('Should throw error for header starting with a number', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: ['123-invalid'],
        },
      });
    }).toThrow(/Request header must start with a letter and contain only letters, numbers, underscores, and hyphens/);
  });

  test('Should accept headers with underscores', () => {
    expect(() => {
      new Runtime(stack, 'test-runtime', {
        runtimeName: 'test_runtime',
        agentRuntimeArtifact: agentRuntimeArtifact,
        requestHeaderConfiguration: {
          allowlistedHeaders: ['X-Custom_Header', 'My_Header_Name'],
        },
      });
    }).not.toThrow();
  });
});

describe('Runtime fromS3 artifact loading tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let bucket: s3.Bucket;

  beforeEach(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    bucket = new s3.Bucket(stack, 'CodeBucket', {
      bucketName: 'test-runtime-code-bucket',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
  });

  test('Should create runtime with fromS3 artifact', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: bucket.bucketName,
        objectKey: 'runtime-code.zip',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['main.handler'],
    );

    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    expect(runtime.agentRuntimeArtifact).toBe(agentRuntimeArtifact);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeArtifact: {
        CodeConfiguration: {
          Code: {
            S3: {
              Bucket: {
                Ref: Match.stringLikeRegexp('CodeBucket.*'),
              },
              Prefix: 'runtime-code.zip',
            },
          },
          Runtime: 'PYTHON_3_10',
          EntryPoint: ['main.handler'],
        },
      },
    });
  });

  test('Should create runtime with fromS3 artifact with object version', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: bucket.bucketName,
        objectKey: 'runtime-code.zip',
        objectVersion: 'version123',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['index.handler'],
    );

    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeArtifact: {
        CodeConfiguration: {
          Code: {
            S3: {
              Bucket: {
                Ref: Match.stringLikeRegexp('CodeBucket.*'),
              },
              Prefix: 'runtime-code.zip',
              VersionId: 'version123',
            },
          },
          Runtime: 'PYTHON_3_10',
          EntryPoint: ['index.handler'],
        },
      },
    });
  });

  test('Should grant S3 permissions to execution role when using fromS3 artifact', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: bucket.bucketName,
        objectKey: 'runtime-code.zip',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['main.handler'],
    );

    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    // Check that the execution role has S3 permissions
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: ['s3:GetObject*', 's3:GetBucket*', 's3:List*'],
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  test('Should work with fromS3 artifact and lifecycle configuration', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: bucket.bucketName,
        objectKey: 'runtime-code.zip',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['main.handler'],
    );

    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      lifecycleConfiguration: {
        idleRuntimeSessionTimeout: Duration.minutes(15),
        maxLifetime: Duration.hours(4),
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeArtifact: {
        CodeConfiguration: {
          Code: {
            S3: { Bucket: { Ref: Match.stringLikeRegexp('CodeBucket.*') }, Prefix: 'runtime-code.zip' },
          },
          Runtime: 'PYTHON_3_10',
          EntryPoint: ['main.handler'],
        },
      },
      LifecycleConfiguration: {
        IdleRuntimeSessionTimeout: 900,
        MaxLifetime: 14400,
      },
    });
  });

  test('Should work with fromS3 artifact and request header configuration', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: bucket.bucketName,
        objectKey: 'runtime-code.zip',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['main.handler'],
    );

    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
      requestHeaderConfiguration: {
        allowlistedHeaders: ['Authorization', 'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1'],
      },
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeArtifact: {
        CodeConfiguration: {
          Code: {
            S3: { Bucket: { Ref: Match.stringLikeRegexp('CodeBucket.*') }, Prefix: 'runtime-code.zip' },
          },
          Runtime: 'PYTHON_3_10',
          EntryPoint: ['main.handler'],
        },
      },
      RequestHeaderConfiguration: {
        RequestHeaderAllowlist: ['Authorization', 'X-Amzn-Bedrock-AgentCore-Runtime-Custom-Header1'],
      },
    });
  });

  test('Should work with fromS3 artifact using string bucket name', () => {
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromS3(
      {
        bucketName: 'my-custom-bucket',
        objectKey: 'runtime-code.zip',
      },
      AgentCoreRuntime.PYTHON_3_10,
      ['com.example.Handler::handleRequest'],
    );

    new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeArtifact: {
        CodeConfiguration: {
          Code: {
            S3: {
              Bucket: 'my-custom-bucket',
              Prefix: 'runtime-code.zip',
            },
          },
          Runtime: 'PYTHON_3_10',
          EntryPoint: ['com.example.Handler::handleRequest'],
        },
      },
    });
  });
});

const logGroupPolicyStatement = {
  Sid: 'LogGroupAccess',
  Action: ['logs:DescribeLogStreams', 'logs:CreateLogGroup'],
  Effect: 'Allow',
  Resource: {
    'Fn::Join': [
      '',
      [
        'arn:',
        { Ref: 'AWS::Partition' },
        ':logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/runtimes/*',
      ],
    ],
  },
};

const describeLogGroupsPolicyStatement = {
  Sid: 'DescribeLogGroups',
  Action: 'logs:DescribeLogGroups',
  Effect: 'Allow',
  Resource: {
    'Fn::Join': [
      '',
      [
        'arn:',
        { Ref: 'AWS::Partition' },
        ':logs:us-east-1:123456789012:log-group:*',
      ],
    ],
  },
};

const logStreamPolicyStatement = {
  Sid: 'LogStreamAccess',
  Action: ['logs:CreateLogStream', 'logs:PutLogEvents'],
  Effect: 'Allow',
  Resource: {
    'Fn::Join': [
      '',
      [
        'arn:',
        { Ref: 'AWS::Partition' },
        ':logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*',
      ],
    ],
  },
};

const xrayPolicyStatement = {
  Sid: 'XRayAccess',
  Action: [
    'xray:PutTraceSegments',
    'xray:PutTelemetryRecords',
    'xray:GetSamplingRules',
    'xray:GetSamplingTargets',
  ],
  Effect: 'Allow',
  Resource: '*',
};

const cloudWatchMetricsPolicyStatement = {
  Sid: 'CloudWatchMetrics',
  Action: 'cloudwatch:PutMetricData',
  Condition: {
    StringEquals: {
      'cloudwatch:namespace': 'bedrock-agentcore',
    },
  },
  Effect: 'Allow',
  Resource: '*',
};

const getAgentAccessTokenPolicyStatement = {
  Sid: 'GetAgentAccessToken',
  Action: [
    'bedrock-agentcore:GetWorkloadAccessToken',
    'bedrock-agentcore:GetWorkloadAccessTokenForJWT',
    'bedrock-agentcore:GetWorkloadAccessTokenForUserId',
  ],
  Effect: 'Allow',
  Resource: [
    {
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':bedrock-agentcore:us-east-1:123456789012:workload-identity-directory/default',
        ],
      ],
    },
    {
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':bedrock-agentcore:us-east-1:123456789012:workload-identity-directory/default/workload-identity/*',
        ],
      ],
    },
  ],
};

const ecrReadPolicyStatement = {
  Action: [
    'ecr:BatchCheckLayerAvailability',
    'ecr:GetDownloadUrlForLayer',
    'ecr:BatchGetImage',
  ],
  Effect: 'Allow',
  Resource: {
    'Fn::GetAtt': ['TestRepositoryC0DA8195', 'Arn'],
  },
};

const ecrAuthTokenPolicyStatement = {
  Action: 'ecr:GetAuthorizationToken',
  Effect: 'Allow',
  Resource: '*',
};

const expectedExecutionRolePolicy = {
  PolicyDocument: {
    Statement: [
      logGroupPolicyStatement,
      describeLogGroupsPolicyStatement,
      logStreamPolicyStatement,
      xrayPolicyStatement,
      cloudWatchMetricsPolicyStatement,
      getAgentAccessTokenPolicyStatement,
      ecrReadPolicyStatement,
      ecrAuthTokenPolicyStatement,
    ],
  },
};

describe('Runtime Optional Physical Names', () => {
  let stack: cdk.Stack;
  let agentRuntimeArtifact: AgentRuntimeArtifact;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'TestStack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });

    const repository = new ecr.Repository(stack, 'TestRepository', {
      repositoryName: 'test-agent-runtime',
    });

    agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0');
  });

  test('Should create Runtime without runtimeName (auto-generated)', () => {
    const runtime = new Runtime(stack, 'TestRuntime', {
      agentRuntimeArtifact: agentRuntimeArtifact,
    });

    expect(runtime.agentRuntimeName).toBeDefined();
    expect(runtime.agentRuntimeName).not.toBe('');
  });
});

describe('Runtime observability tests', () => {
  test('Should configure tracing delivery with runtime ARN as source', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'TracingTestStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const runtime = new Runtime(stack, 'TracingRuntime', {
      runtimeName: 'tracing_runtime',
      agentRuntimeArtifact,
      tracingEnabled: true,
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);

    // Verify delivery source uses runtime ARN as resource
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'TRACES',
      ResourceArn: resolvedRuntimeArn,
    });

    // Verify delivery destination is configured for X-Ray
    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'XRAY',
    });

    // Verify X-Ray resource policy allows logs delivery service
    template.hasResourceProperties('AWS::XRay::ResourcePolicy', {
      PolicyDocument: {
        'Fn::Join': [
          '',
          [
            '{"Statement":[{"Action":"xray:PutTraceSegments","Condition":{"ForAllValues:ArnLike":{"logs:LogGeneratingResourceArns":["',
            { 'Fn::GetAtt': ['TracingRuntime80A99119', 'AgentRuntimeArn'] },
            '"]},"StringEquals":{"aws:SourceAccount":"123456789012"},"ArnLike":{"aws:SourceArn":"arn:',
            { Ref: 'AWS::Partition' },
            ':logs:us-east-1:123456789012:delivery-source:*"}},"Effect":"Allow","Principal":{"Service":"delivery.logs.amazonaws.com"},"Resource":"*"}],"Version":"2012-10-17"}',
          ],
        ],
      },
    });
  });

  test('Should not create observability resources when not configured', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'NoObservabilityStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    new Runtime(stack, 'NoObservabilityRuntime', {
      runtimeName: 'no_observability_runtime',
      agentRuntimeArtifact,
    });

    const template = Template.fromStack(stack);

    expect(() => {
      template.hasResourceProperties('AWS::Logs::DeliverySource', {});
    }).toThrow();
  });

  test('Should configure CloudWatch Logs destination with log group ARN', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'LoggingCWLStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const logGroup = new logs.LogGroup(stack, 'AppLogGroup');

    const runtime = new Runtime(stack, 'LoggingRuntime', {
      runtimeName: 'logging_runtime',
      agentRuntimeArtifact,
      loggingConfigs: [
        {
          logType: LogType.APPLICATION_LOGS,
          destination: LoggingDestination.cloudWatchLogs(logGroup),
        },
      ],
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);
    const resolvedLogGroupArn = stack.resolve(logGroup.logGroupArn);

    // Verify delivery source uses runtime ARN and correct log type
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'APPLICATION_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });

    // Verify delivery destination points to the log group ARN
    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'CWL',
      DestinationResourceArn: resolvedLogGroupArn,
    });

    // Verify CloudWatch Logs resource policy allows logs delivery service
    template.hasResourceProperties('AWS::Logs::ResourcePolicy', {
      PolicyDocument: {
        'Fn::Join': [
          '',
          [
            '{"Statement":[{"Action":["logs:CreateLogStream","logs:PutLogEvents"],"Condition":{"StringEquals":{"aws:SourceAccount":"123456789012"},"ArnLike":{"aws:SourceArn":"arn:',
            { Ref: 'AWS::Partition' },
            ':logs:us-east-1:123456789012:*"}},"Effect":"Allow","Principal":{"Service":"delivery.logs.amazonaws.com"},"Resource":"',
            { 'Fn::GetAtt': ['AppLogGroup7D8CD952', 'Arn'] },
            ':log-stream:*"}],"Version":"2012-10-17"}',
          ],
        ],
      },
    });
  });

  test('Should configure S3 destination with bucket ARN and policy', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'LoggingS3Stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const bucket = new s3.Bucket(stack, 'LogBucket');

    const runtime = new Runtime(stack, 'LoggingRuntime', {
      runtimeName: 'logging_s3_runtime',
      agentRuntimeArtifact,
      loggingConfigs: [
        {
          logType: LogType.USAGE_LOGS,
          destination: LoggingDestination.s3(bucket),
        },
      ],
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);
    const resolvedBucketArn = stack.resolve(bucket.bucketArn);

    // Verify delivery source uses runtime ARN and USAGE_LOGS type
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'USAGE_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });

    // Verify delivery destination points to the S3 bucket ARN
    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'S3',
      DestinationResourceArn: resolvedBucketArn,
    });

    // Verify S3 bucket policy allows logs delivery service
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Allow',
            Principal: { Service: 'delivery.logs.amazonaws.com' },
            Action: 's3:PutObject',
            Condition: {
              StringEquals: {
                's3:x-amz-acl': 'bucket-owner-full-control',
                'aws:SourceAccount': '123456789012',
              },
            },
          }),
        ]),
      },
    });
  });

  test('Should create separate delivery sources for different log types', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'MultiLogTypeStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const appLogGroup = new logs.LogGroup(stack, 'AppLogGroup');
    const usageLogGroup = new logs.LogGroup(stack, 'UsageLogGroup');

    const runtime = new Runtime(stack, 'MultiLogRuntime', {
      runtimeName: 'multi_log_runtime',
      agentRuntimeArtifact,
      loggingConfigs: [
        {
          logType: LogType.APPLICATION_LOGS,
          destination: LoggingDestination.cloudWatchLogs(appLogGroup),
        },
        {
          logType: LogType.USAGE_LOGS,
          destination: LoggingDestination.cloudWatchLogs(usageLogGroup),
        },
      ],
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);

    // Verify APPLICATION_LOGS delivery source
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'APPLICATION_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });

    // Verify USAGE_LOGS delivery source
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'USAGE_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });
  });

  test('Should configure Firehose destination with stream ARN and tag', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'LoggingFirehoseStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    // Create an S3 bucket as the Firehose destination
    const destinationBucket = new s3.Bucket(stack, 'DestinationBucket');

    // Create a Firehose delivery stream
    const deliveryStream = new firehose.DeliveryStream(stack, 'LogDeliveryStream', {
      destination: new firehose.S3Bucket(destinationBucket),
    });

    const runtime = new Runtime(stack, 'LoggingRuntime', {
      runtimeName: 'logging_firehose_runtime',
      agentRuntimeArtifact,
      loggingConfigs: [
        {
          logType: LogType.APPLICATION_LOGS,
          destination: LoggingDestination.firehose(deliveryStream),
        },
      ],
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);
    const resolvedStreamArn = stack.resolve(deliveryStream.deliveryStreamArn);

    // Verify delivery source uses runtime ARN and APPLICATION_LOGS type
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'APPLICATION_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });

    // Verify delivery destination points to the Firehose stream ARN
    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'FH',
      DestinationResourceArn: resolvedStreamArn,
    });

    // Verify the Firehose stream is tagged with LogDeliveryEnabled
    template.hasResourceProperties('AWS::KinesisFirehose::DeliveryStream', {
      Tags: [
        {
          Key: 'LogDeliveryEnabled',
          Value: 'true',
        },
      ],
    });
  });

  test('Should not create X-Ray resource policy when manageDeliveryResourcePolicy is false', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'NoXRayPolicyStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const runtime = new Runtime(stack, 'TracingRuntime', {
      runtimeName: 'tracing_runtime',
      agentRuntimeArtifact,
      tracingEnabled: true,
      manageDeliveryResourcePolicy: false,
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);

    // Delivery source, destination, and delivery should still be created
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'TRACES',
      ResourceArn: resolvedRuntimeArn,
    });

    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'XRAY',
    });

    // X-Ray resource policy should NOT be created
    template.resourceCountIs('AWS::XRay::ResourcePolicy', 0);
  });

  test('Should not create CloudWatch Logs resource policy when manageDeliveryResourcePolicy is false', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'NoCwlPolicyStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const logGroup = new logs.LogGroup(stack, 'AppLogGroup');

    const runtime = new Runtime(stack, 'LoggingRuntime', {
      runtimeName: 'logging_runtime',
      agentRuntimeArtifact,
      loggingConfigs: [
        {
          logType: LogType.APPLICATION_LOGS,
          destination: LoggingDestination.cloudWatchLogs(logGroup),
        },
      ],
      manageDeliveryResourcePolicy: false,
    });

    const template = Template.fromStack(stack);
    const resolvedRuntimeArn = stack.resolve(runtime.agentRuntimeArn);
    const resolvedLogGroupArn = stack.resolve(logGroup.logGroupArn);

    // Delivery source, destination, and delivery should still be created
    template.hasResourceProperties('AWS::Logs::DeliverySource', {
      LogType: 'APPLICATION_LOGS',
      ResourceArn: resolvedRuntimeArn,
    });

    template.hasResourceProperties('AWS::Logs::DeliveryDestination', {
      DeliveryDestinationType: 'CWL',
      DestinationResourceArn: resolvedLogGroupArn,
    });

    // CloudWatch Logs resource policy should NOT be created
    template.resourceCountIs('AWS::Logs::ResourcePolicy', 0);
  });

  test('Should not create any delivery resource policies when manageDeliveryResourcePolicy is false with both tracing and logging', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'NoPoliciesStack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const repository = new ecr.Repository(stack, 'TestRepository');
    const agentRuntimeArtifact = AgentRuntimeArtifact.fromEcrRepository(repository, 'latest');

    const logGroup = new logs.LogGroup(stack, 'AppLogGroup');

    new Runtime(stack, 'FullObservabilityRuntime', {
      runtimeName: 'full_observability_runtime',
      agentRuntimeArtifact,
      tracingEnabled: true,
      loggingConfigs: [
        {
          logType: LogType.APPLICATION_LOGS,
          destination: LoggingDestination.cloudWatchLogs(logGroup),
        },
      ],
      manageDeliveryResourcePolicy: false,
    });

    const template = Template.fromStack(stack);

    // Neither X-Ray nor CloudWatch Logs resource policies should be created
    template.resourceCountIs('AWS::XRay::ResourcePolicy', 0);
    template.resourceCountIs('AWS::Logs::ResourcePolicy', 0);
  });
});

describe('Runtime applicationLogGroup tests', () => {
  test('Should expose applicationLogGroup pointing at the default endpoint log group', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });
    const repository = new ecr.Repository(stack, 'TestRepository');
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0'),
    });

    // Log group name resolves the AgentRuntimeId attribute
    const resolvedName = stack.resolve(runtime.applicationLogGroup.logGroupName);
    expect(resolvedName).toEqual({
      'Fn::Join': [
        '',
        [
          '/aws/bedrock-agentcore/runtimes/',
          { 'Fn::GetAtt': [expect.stringMatching(/^testruntime/), 'AgentRuntimeId'] },
          '-DEFAULT',
        ],
      ],
    });

    // ARN includes the stack region/account and uses the COLON_RESOURCE_NAME format with :*
    const resolvedArn = stack.resolve(runtime.applicationLogGroup.logGroupArn);
    expect(resolvedArn).toEqual({
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/runtimes/',
          { 'Fn::GetAtt': [expect.stringMatching(/^testruntime/), 'AgentRuntimeId'] },
          '-DEFAULT:*',
        ],
      ],
    });
  });

  test('Should accept the log group as the target of a MetricFilter without hardcoding its name', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });
    const repository = new ecr.Repository(stack, 'TestRepository');
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0'),
    });

    new logs.MetricFilter(stack, 'ToolErrors', {
      logGroup: runtime.applicationLogGroup,
      filterPattern: logs.FilterPattern.stringValue('$.tool_status', '=', 'error'),
      metricNamespace: 'MyApp',
      metricName: 'ToolExecutionErrors',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::Logs::MetricFilter', {
      LogGroupName: {
        'Fn::Join': [
          '',
          [
            '/aws/bedrock-agentcore/runtimes/',
            { 'Fn::GetAtt': [Match.stringLikeRegexp('^testruntime'), 'AgentRuntimeId'] },
            '-DEFAULT',
          ],
        ],
      },
      MetricTransformations: Match.arrayWith([
        Match.objectLike({
          MetricNamespace: 'MyApp',
          MetricName: 'ToolExecutionErrors',
        }),
      ]),
    });
  });

  test('Should memoize the imported log group across repeated accesses', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });
    const repository = new ecr.Repository(stack, 'TestRepository');
    const runtime = new Runtime(stack, 'test-runtime', {
      runtimeName: 'test_runtime',
      agentRuntimeArtifact: AgentRuntimeArtifact.fromEcrRepository(repository, 'v1.0.0'),
    });

    expect(runtime.applicationLogGroup).toBe(runtime.applicationLogGroup);
  });

  test('Should expose applicationLogGroup on imported runtimes', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });

    const imported = Runtime.fromAgentRuntimeAttributes(stack, 'ImportedRuntime', {
      agentRuntimeArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/runtime-abc123',
      agentRuntimeId: 'runtime-abc123',
      agentRuntimeName: 'imported-runtime',
      roleArn: 'arn:aws:iam::123456789012:role/test-role',
    });

    expect(imported.applicationLogGroup.logGroupName)
      .toBe('/aws/bedrock-agentcore/runtimes/runtime-abc123-DEFAULT');
    expect(stack.resolve(imported.applicationLogGroup.logGroupArn)).toEqual({
      'Fn::Join': [
        '',
        [
          'arn:',
          { Ref: 'AWS::Partition' },
          ':logs:us-east-1:123456789012:log-group:/aws/bedrock-agentcore/runtimes/runtime-abc123-DEFAULT:*',
        ],
      ],
    });
  });
});

