import { Template, Match } from '../../../../assertions';
import * as cloudwatch from '../../../../aws-cloudwatch';
import * as ec2 from '../../../../aws-ec2';
import * as iam from '../../../../aws-iam';
import * as s3 from '../../../../aws-s3';
import { App, Stack } from '../../../../core';
import * as cdk from '../../../../core';
import { BrowserNetworkConfiguration } from '../../../lib/network/network-configuration';
import { BrowserCustom, BrowserSigning } from '../../../lib/tools/browser';

describe('BrowserCustom default tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  // @ts-ignore
  let browser: BrowserCustom;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser for web automation',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::BrowserCustom', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have BrowserCustom resource with expected properties', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser',
      NetworkConfiguration: {
        NetworkMode: 'PUBLIC',
      },
    });
  });

  test('Should handle tags correctly when no tags are provided', () => {
    // Verify that the BrowserCustom resource exists with basic properties
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser',
      NetworkConfiguration: { NetworkMode: 'PUBLIC' },
      BrowserSigning: { Enabled: false },
    });
  });

  test('Should have service role with confused deputy conditions', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: [
          {
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Principal: { Service: 'bedrock-agentcore.amazonaws.com' },
            Condition: {
              StringEquals: { 'aws:SourceAccount': '123456789012' },
              ArnLike: {
                'aws:SourceArn': {
                  'Fn::Join': ['', Match.arrayWith([
                    ':bedrock-agentcore:us-east-1:123456789012:browser-custom/test_browser*',
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

describe('BrowserCustom with VPC config tests', () => {
  let template: Template;
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

  test('Provide VPC and security groups, no security group created', () => {
    const vpc = new ec2.Vpc(stack, 'testVPC');
    const sg = new ec2.SecurityGroup(stack, 'SG', { vpc });

    new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser for web automation',
      networkConfiguration: BrowserNetworkConfiguration.usingVpc(stack, {
        vpc: vpc,
        securityGroups: [sg],
      }),
    });

    template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      NetworkConfiguration: {
        NetworkMode: 'VPC',
        VpcConfig: {
          Subnets: Match.arrayWith([Match.objectLike({ Ref: Match.stringLikeRegexp('testVPC.*Subnet.*') })]),
          SecurityGroups: Match.arrayWith([Match.objectLike({ 'Fn::GetAtt': [Match.stringLikeRegexp('SG.*'), 'GroupId'] })]),
        },
      },
    });
  });

  test('Provide VPC and no security groups, a security group is created', () => {
    const vpc = new ec2.Vpc(stack, 'testVPC');

    new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser for web automation',
      networkConfiguration: BrowserNetworkConfiguration.usingVpc(stack, {
        vpc: vpc,
      }),
    });

    template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      NetworkConfiguration: {
        NetworkMode: 'VPC',
        VpcConfig: {
          Subnets: Match.arrayWith([Match.objectLike({ Ref: Match.stringLikeRegexp('testVPC.*Subnet.*') })]),
          SecurityGroups: Match.arrayWith([Match.objectLike({ 'Fn::GetAtt': [Match.stringLikeRegexp('SecurityGroup.*'), 'GroupId'] })]),
        },
      },
    });
  });

  test('Both security groups and allowAllOutbound are specified, an exception is thrown', () => {
    expect(() => {
      const vpc = new ec2.Vpc(stack, 'testVPC');
      const sg = new ec2.SecurityGroup(stack, 'SG', { vpc });

      new BrowserCustom(stack, 'test-browser', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingVpc(stack, {
          vpc: vpc,
          securityGroups: [sg],
          allowAllOutbound: false,
        }),
      });
    }).toThrow('Configure \'allowAllOutbound\' directly on the supplied SecurityGroups');
  });

  test('Vpc specified but no scope, an exception is thrown', () => {
    expect(() => {
      const vpc = new ec2.Vpc(stack, 'testVPC');
      const sg = new ec2.SecurityGroup(stack, 'SG', { vpc });

      new BrowserCustom(stack, 'test-browser', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingVpc(undefined as any, {
          vpc: vpc,
          securityGroups: [sg],
        }),
      });
    }).toThrow('Scope is required to create the security group');
  });

  test('Vpc not specified, an exception is thrown when accessing Connections object', () => {
    const browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });

    const when = () => browser.connections;
    expect(when).toThrow('Cannot manage network access without configuring a VPC');
  });

  test('When adding security group after browser instantiation, it is reflected in VpcConfig of Browser Custom', () => {
    const vpc = new ec2.Vpc(stack, 'testVPC');

    const browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingVpc(stack, {
        vpc: vpc,
      }),
    });

    expect(browser.connections.securityGroups.length).toBe(1);

    browser.connections.addSecurityGroup(new ec2.SecurityGroup(stack, 'AdditionalGroup', { vpc }));

    expect(browser.connections.securityGroups.length).toBe(2);

    template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      NetworkConfiguration: {
        NetworkMode: 'VPC',
        VpcConfig: {
          SecurityGroups: [
            {
              'Fn::GetAtt': [
                'SecurityGroupDD263621',
                'GroupId',
              ],
            },
            {
              'Fn::GetAtt': [
                'AdditionalGroup4973CFAA',
                'GroupId',
              ],
            },
          ],
        },
      },
    });
  });
});

describe('BrowserCustom static methods tests', () => {
  // @ts-ignore
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    template = Template.fromStack(stack);
  });

  test('fromBrowserCustomAttributes should create a BrowserCustom reference from existing attributes', () => {
    const browser = BrowserCustom.fromBrowserCustomAttributes(stack, 'test-browser', {
      browserArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:browser/test-browser',
      roleArn: 'arn:aws:iam::123456789012:role/test-browser-role',
      lastUpdatedAt: '2021-01-01T00:00:00Z',
      status: 'ACTIVE',
      createdAt: '2021-01-01T00:00:00Z',
    });

    expect(browser.browserArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:browser/test-browser');
    expect(browser.executionRole).toBeDefined();
    expect(browser.lastUpdatedAt).toBe('2021-01-01T00:00:00Z');
    expect(browser.status).toBe('ACTIVE');
    expect(browser.createdAt).toBe('2021-01-01T00:00:00Z');
  });

  test('fromBrowserCustomAttributes provides undefined values when not provided', () => {
    const browser = BrowserCustom.fromBrowserCustomAttributes(stack, 'test-browser-2', {
      browserArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:browser/test-browser',
      roleArn: 'arn:aws:iam::123456789012:role/test-browser-role',
    });

    expect(browser.browserArn).toBe('arn:aws:bedrock-agentcore:us-east-1:123456789012:browser/test-browser');
    expect(browser.executionRole).toBeDefined();
    expect(browser.lastUpdatedAt).toBeUndefined();
    expect(browser.status).toBeUndefined();
    expect(browser.createdAt).toBeUndefined();
  });

  test('fromBrowserCustomAttributes with no security groups specified, an exception is thrown', () => {
    // GIVEN
    const browser = BrowserCustom.fromBrowserCustomAttributes(stack, 'test-browser-3', {
      browserArn: 'arn:aws:bedrock-agentcore:us-east-1:123456789012:browser/test-browser',
      roleArn: 'arn:aws:iam::123456789012:role/test-browser-role',
      lastUpdatedAt: '2021-01-01T00:00:00Z',
      status: 'ACTIVE',
      createdAt: '2021-01-01T00:00:00Z',
    });

    // WHEN
    const when = () => browser.connections;

    // THEN
    expect(when).toThrow(/Cannot manage network access without configuring a VPC/);
  });
});

describe('BrowserCustom with recording config tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let recordingBucket: s3.Bucket;
  // @ts-ignore
  let browser: BrowserCustom;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    recordingBucket = new s3.Bucket(stack, 'RecordingBucket', {
      bucketName: 'test-browser-recordings',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser for web automation',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: recordingBucket.bucketName,
          objectKey: 'test-browser-recordings/',
        },
      },
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::BrowserCustom', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
    template.resourceCountIs('AWS::S3::Bucket', 1);
  });

  test('Should have BrowserCustom resource with recording config', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser',
      NetworkConfiguration: {
        NetworkMode: 'PUBLIC',
      },
    });
  });
});

describe('BrowserCustom with custom execution role tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;
  let customRole: iam.Role;
  // @ts-ignore
  let browser: BrowserCustom;

  beforeAll(() => {
    app = new cdk.App();

    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    // Create a custom execution role
    customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
      roleName: 'custom-browser-execution-role',
    });

    browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser with custom execution role',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      executionRole: customRole,
    });

    template = Template.fromStack(stack);
  });

  test('Should have the correct resources', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::BrowserCustom', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });

  test('Should have BrowserCustom resource with custom execution role', () => {
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser',
      NetworkConfiguration: {
        NetworkMode: 'PUBLIC',
      },
    });
  });

  test('Should have custom execution role with correct properties', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'custom-browser-execution-role',
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
});

describe('BrowserCustom name validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should accept name with hyphen (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'test-browser', {
        browserCustomName: 'test-browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).toThrow('The field Browser name with value "test-browser" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should accept empty name (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'empty-name', {
        browserCustomName: '',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).toThrow('The field Browser name is 0 characters long but must be at least 1 characters');
  });

  test('Should accept name with spaces (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'name-with-spaces', {
        browserCustomName: 'test browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).toThrow('The field Browser name with value "test browser" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should accept name with special characters (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'name-with-special-chars', {
        browserCustomName: 'test@browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).toThrow('The field Browser name with value "test@browser" does not match the required pattern /^[a-zA-Z][a-zA-Z0-9_]{0,47}$/');
  });

  test('Should accept name exceeding 48 characters (validation not enforced)', () => {
    const longName = 'a'.repeat(49);
    expect(() => {
      new BrowserCustom(stack, 'long-name', {
        browserCustomName: longName,
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).toThrow('The field Browser name is 49 characters long but must be less than or equal to 48 characters');
  });

  test('Should accept valid name with underscores', () => {
    expect(() => {
      new BrowserCustom(stack, 'valid-name', {
        browserCustomName: 'test_browser_123',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).not.toThrow();
  });

  test('Should accept valid name with only letters and numbers', () => {
    expect(() => {
      new BrowserCustom(stack, 'valid-name-2', {
        browserCustomName: 'testBrowser123',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      });
    }).not.toThrow();
  });

  test('Should use default PUBLIC network configuration when not provided', () => {
    const browser = new BrowserCustom(stack, 'default-network', {
      browserCustomName: 'test_browser_default',
    });

    expect(browser.networkConfiguration.networkMode).toBe('PUBLIC');
  });
});

describe('BrowserCustom tags validation tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });
  });

  test('Should accept valid tags', () => {
    expect(() => {
      new BrowserCustom(stack, 'valid-tags', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          'Environment': 'Production',
          'Team': 'AI/ML',
          'Project': 'AgentCore',
          'Cost-Center': '12345',
        },
      });
    }).not.toThrow();
  });

  test('Should accept tags with special characters', () => {
    expect(() => {
      new BrowserCustom(stack, 'special-chars-tags', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          'Environment': 'Production',
          'Team@Company': 'AI/ML',
          'Project:Name': 'AgentCore',
          'Cost-Center': '12345',
          'Description': 'Test browser with special chars',
        },
      });
    }).not.toThrow();
  });

  test('Should accept empty tag key (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'empty-tag-key', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          '': 'value',
        },
      });
    }).toThrow('The field Tag key is 0 characters long but must be at least 1 characters');
  });

  test('Should accept tag key exceeding 256 characters (validation not enforced)', () => {
    const longKey = 'a'.repeat(257);
    expect(() => {
      new BrowserCustom(stack, 'long-tag-key', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          [longKey]: 'value',
        },
      });
    }).toThrow('The field Tag key is 257 characters long but must be less than or equal to 256 characters');
  });

  test('Should accept tag value exceeding 256 characters (validation not enforced)', () => {
    const longValue = 'a'.repeat(257);
    expect(() => {
      new BrowserCustom(stack, 'long-tag-value', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          key: longValue,
        },
      });
    }).toThrow('The field Tag value is 257 characters long but must be less than or equal to 256 characters');
  });

  test('Should accept tag key with invalid characters (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'invalid-tag-key', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          'key#invalid': 'value',
        },
      });
    }).toThrow('The field Tag key with value "key#invalid" does not match the required pattern /^[a-zA-Z0-9\\s._:/=+@-]*$/');
  });

  test('Should accept tag value with invalid characters (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'invalid-tag-value', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          key: 'value#invalid',
        },
      });
    }).toThrow('The field Tag value with value "value#invalid" does not match the required pattern /^[a-zA-Z0-9\\s._:/=+@-]*$/');
  });

  test('Should accept null tag value (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'null-tag-value', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          key: null as any,
        },
      });
    }).not.toThrow();
  });

  test('Should accept undefined tag value (validation not enforced)', () => {
    expect(() => {
      new BrowserCustom(stack, 'undefined-tag-value', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {
          key: undefined as any,
        },
      });
    }).not.toThrow();
  });

  test('Should accept undefined tags', () => {
    expect(() => {
      new BrowserCustom(stack, 'undefined-tags', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: undefined,
      });
    }).not.toThrow();
  });

  test('Should accept empty tags object', () => {
    expect(() => {
      new BrowserCustom(stack, 'empty-tags', {
        browserCustomName: 'test_browser',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        tags: {},
      });
    }).not.toThrow();
  });
});

describe('BrowserCustom with tags CloudFormation template tests', () => {
  let template: Template;
  let app: cdk.App;
  let stack: cdk.Stack;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    new BrowserCustom(stack, 'test-browser-with-tags', {
      browserCustomName: 'test_browser_with_tags',
      description: 'A test browser with tags',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      tags: {
        Environment: 'Production',
        Team: 'AI/ML',
        Project: 'AgentCore',
      },
    });

    template = Template.fromStack(stack);
  });

  test('Should handle tags correctly when tags are provided', () => {
    // Verify that the BrowserCustom resource exists with basic properties
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser_with_tags',
      NetworkConfiguration: { NetworkMode: 'PUBLIC' },
    });
  });

  test('Should have correct resource count with tags', () => {
    template.resourceCountIs('AWS::BedrockAgentCore::BrowserCustom', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);
  });
});

describe('BrowserCustom CloudFormation parameter validation tests', () => {
  let app: App;
  let stack: Stack;
  let template: Template;

  test('Should pass bucket name string instead of S3 bucket resource', () => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    // Create an S3 bucket resource
    const bucket = new s3.Bucket(stack, 'TestBucket', {
      bucketName: 'test-bucket-name',
    });

    // Create browser with S3 bucket resource
    new BrowserCustom(stack, 'TestBrowser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: bucket.bucketName, // Extract bucket name string
          objectKey: 'recordings/',
        },
      },
    });

    template = Template.fromStack(stack);

    // Verify that the RecordingConfig is properly structured
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      RecordingConfig: {
        Enabled: true,
        S3Location: {
          Bucket: { Ref: 'TestBucket560B80BC' },
          Prefix: 'recordings/',
        },
      },
    });
  });

  test('Should handle empty recording config with conditional logic', () => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    new BrowserCustom(stack, 'TestBrowser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      // No recording config provided
    });

    template = Template.fromStack(stack);

    // Should have RecordingConfig with enabled: false when not provided
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      RecordingConfig: { Enabled: false },
    });
  });

  test('Should have recording disabled by default when not provided', () => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    const browser = new BrowserCustom(stack, 'TestBrowser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      // No recording config provided - should default to disabled
    });

    template = Template.fromStack(stack);

    // Verify that recordingConfig is set to disabled by default
    expect(browser.recordingConfig).toBeDefined();
    expect(browser.recordingConfig?.enabled).toBe(false);

    // Verify that the CloudFormation template includes RecordingConfig with enabled: false
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      RecordingConfig: { Enabled: false },
    });
  });

  test('Should validate CloudFormation template structure', () => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    new BrowserCustom(stack, 'TestBrowser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: 'test-bucket-name', // String bucket name
          objectKey: 'recordings/',
        },
      },
      tags: {
        Environment: 'Test',
        Team: 'AI/ML',
      },
    });

    template = Template.fromStack(stack);

    // Verify that all conditions reference parameters, not resources
    const conditions = template.findConditions('*');
    Object.values(conditions).forEach((condition: any) => {
      // Check that conditions don't reference resources
      const conditionStr = JSON.stringify(condition);
      expect(conditionStr).not.toMatch(/AWS::S3::Bucket/);
      expect(conditionStr).not.toMatch(/AWS::IAM::Role/);
      expect(conditionStr).not.toMatch(/AWS::BedrockAgentCore::BrowserCustom/);
    });

    // Verify that the template has the expected structure
    expect(template.toJSON()).toHaveProperty('Resources');
    // Conditions are not created in the current implementation
    // Outputs are no longer used - attributes are accessed directly from the resource
  });

  test('Should handle execution role ARN correctly', () => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
    });

    new BrowserCustom(stack, 'TestBrowser', {
      browserCustomName: 'test_browser',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      executionRole: role,
    });

    template = Template.fromStack(stack);

    // Verify that the execution role ARN is properly referenced
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      ExecutionRoleArn: { 'Fn::GetAtt': ['TestRole6C9272DF', 'Arn'] },
    });
  });

  describe('Recording Configuration Validation', () => {
    test('Should accept valid recording configuration', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'valid-bucket-name-123',
              objectKey: 'recordings/',
            },
          },
        });
      }).not.toThrow();
    });

    test('Should accept browser without recording configuration', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        });
      }).not.toThrow();
    });

    test('Should accept recording configuration without S3 location', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
          },
        });
      }).not.toThrow();
    });

    test('Should accept S3 location without bucket name (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: '', // Empty bucket name
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('S3 bucket name is required when S3 location is provided for recording configuration');
    });

    test('Should accept S3 location without object key (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'valid-bucket-name-123',
              objectKey: '', // Empty object key
            },
          },
        });
      }).toThrow('S3 object key (prefix) is required when S3 location is provided for recording configuration');
    });

    test('Should accept invalid bucket name - starts with uppercase (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'Invalid-Bucket-Name',
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "Invalid-Bucket-Name" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept invalid bucket name - starts with hyphen (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: '-invalid-bucket-name',
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "-invalid-bucket-name" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept invalid bucket name - ends with hyphen (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'invalid-bucket-name-',
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "invalid-bucket-name-" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept invalid bucket name - contains underscore (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'invalid_bucket_name',
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "invalid_bucket_name" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept invalid bucket name - too short (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'a',
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "a" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept invalid bucket name - too long (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      const longBucketName = 'a'.repeat(65); // 65 characters, exceeds the 63 character limit

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: longBucketName,
              objectKey: 'recordings/',
            },
          },
        });
      }).toThrow('The field S3 bucket name with value "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" does not match the required pattern /^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/');
    });

    test('Should accept empty object key (validation not enforced)', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      expect(() => {
        new BrowserCustom(stack, 'TestBrowser', {
          browserCustomName: 'test_browser',
          networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
          recordingConfig: {
            enabled: true,
            s3Location: {
              bucketName: 'valid-bucket-name-123',
              objectKey: '',
            },
          },
        });
      }).toThrow('S3 object key (prefix) is required when S3 location is provided for recording configuration');
    });

    test('Should accept valid bucket names with various valid characters', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      const validBucketNames = [
        'valid-bucket-name',
        'valid.bucket.name',
        'valid-bucket-name-123',
        'valid.bucket.name.123',
        'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6',
        'bucket123',
        '123bucket',
        'bucket-name-123',
      ];

      validBucketNames.forEach((bucketName, index) => {
        expect(() => {
          new BrowserCustom(stack, `TestBrowser${index}`, {
            browserCustomName: `test_browser_${index}`,
            networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
            recordingConfig: {
              enabled: true,
              s3Location: {
                bucketName: bucketName,
                objectKey: 'recordings/',
              },
            },
          });
        }).not.toThrow();
      });
    });

    test('Should accept valid object keys', () => {
      app = new App();
      stack = new Stack(app, 'TestStack');

      const validObjectKeys = [
        'recordings/',
        'recordings',
        'a',
        'recordings/subfolder/',
        'recordings-2024/',
        'recordings.with.dots/',
      ];

      validObjectKeys.forEach((objectKey, index) => {
        expect(() => {
          new BrowserCustom(stack, `TestBrowser${index}`, {
            browserCustomName: `test_browser_${index}`,
            networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
            recordingConfig: {
              enabled: true,
              s3Location: {
                bucketName: 'valid-bucket-name-123',
                objectKey: objectKey,
              },
            },
          });
        }).not.toThrow();
      });
    });
  });
});

describe('BrowserCustom grant method tests', () => {
  let app: cdk.App;
  let stack: cdk.Stack;
  let browser: BrowserCustom;

  beforeAll(() => {
    app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    browser = new BrowserCustom(stack, 'test-browser', {
      browserCustomName: 'test_browser',
      description: 'A test browser for grant testing',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });
  });

  test('Should grant custom actions to IAM principal', () => {
    const user = new iam.User(stack, 'TestUser');
    const grant = browser.grant(user, 'bedrock-agentcore:GetBrowser', 'bedrock-agentcore:ListBrowsers');

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });

  test('Should grant read permissions to IAM principal', () => {
    const user = new iam.User(stack, 'TestUser2');
    const grant = browser.grantRead(user);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });

  test('Should grant use permissions to IAM principal', () => {
    const user = new iam.User(stack, 'TestUser3');
    const grant = browser.grantUse(user);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });

  test('Should grant permissions to IAM role', () => {
    const role = new iam.Role(stack, 'TestRole', {
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
    });

    const grant = browser.grantRead(role);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });

  test('Should grant permissions to IAM group', () => {
    const group = new iam.Group(stack, 'TestGroup');
    const grant = browser.grantUse(group);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });

  test('Should return a valid Grant object', () => {
    const user = new iam.User(stack, 'TestUser4');
    const grant = browser.grantRead(user);

    expect(grant).toBeDefined();
    expect(grant.success).toBe(true);
    expect(grant.principalStatements).toBeDefined();
    expect(grant.principalStatements.length).toBeGreaterThan(0);
  });
});

describe('BrowserCustom recording configuration with S3 location tests', () => {
  test('Should grant S3 permissions when recording is enabled with S3 location', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const recordingBucket = new s3.Bucket(stack, 'RecordingBucket', {
      bucketName: 'test-browser-recordings',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    new BrowserCustom(stack, 'test-browser-with-recording', {
      browserCustomName: 'test_browser_with_recording',
      description: 'A test browser with recording enabled',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: recordingBucket.bucketName,
          objectKey: 'recordings/',
        },
      },
    });

    const template = Template.fromStack(stack);

    // Should have the browser resource
    template.resourceCountIs('AWS::BedrockAgentCore::BrowserCustom', 1);
    template.resourceCountIs('AWS::S3::Bucket', 1);
    template.resourceCountIs('AWS::IAM::Role', 1);

    // Should have RecordingConfig with S3 location
    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      RecordingConfig: {
        Enabled: true,
        S3Location: {
          Bucket: {
            Ref: Match.stringLikeRegexp('RecordingBucket.*'),
          },
          Prefix: 'recordings/',
        },
      },
    });
  });

  test('Should grant exactly the least-privilege recording actions scoped to the prefix', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    new BrowserCustom(stack, 'test-browser-with-recording', {
      browserCustomName: 'test_browser_with_recording',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: 'my-recording-bucket',
          objectKey: 'recordings/',
        },
      },
    });

    const template = Template.fromStack(stack);

    // Exact statement match: the auto-created execution role has no inline
    // statements of its own, so its default policy must contain exactly this
    // one least-privilege recording statement. Any added/removed action or
    // changed resource fails this assertion.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          {
            Action: [
              's3:PutObject',
              's3:ListMultipartUploadParts',
              's3:AbortMultipartUpload',
            ],
            Effect: 'Allow',
            Resource: {
              'Fn::Join': [
                '',
                ['arn:', { Ref: 'AWS::Partition' }, ':s3:::my-recording-bucket/recordings/*'],
              ],
            },
          },
        ],
        Version: '2012-10-17',
      },
    });

    // applyBefore() must make the browser resource depend on the recording policy,
    // so the policy exists before the browser is created.
    template.hasResource('AWS::BedrockAgentCore::BrowserCustom', {
      DependsOn: Match.arrayWith([Match.stringLikeRegexp('.*ServiceRoleDefaultPolicy.*')]),
    });
  });

  test('Should normalize a concrete prefix with no trailing slash to prefix/*', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });

    new BrowserCustom(stack, 'test-browser-no-slash', {
      browserCustomName: 'test_browser_no_slash',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: 'my-recording-bucket',
          objectKey: 'recordings', // no trailing slash
        },
      },
    });

    const template = Template.fromStack(stack);

    // A concrete object key without a trailing slash is normalized: a '/' is added
    // before the '*' so the grant is scoped to '.../recordings/*' (the folder),
    // not the broader '.../recordings*'.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          Match.objectLike({
            Action: [
              's3:PutObject',
              's3:ListMultipartUploadParts',
              's3:AbortMultipartUpload',
            ],
            Resource: {
              'Fn::Join': [
                '',
                ['arn:', { Ref: 'AWS::Partition' }, ':s3:::my-recording-bucket/recordings/*'],
              ],
            },
          }),
        ],
      },
    });
  });

  test('Should handle a tokenized recording prefix (current behavior, no normalization)', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });

    const prefixParam = new cdk.CfnParameter(stack, 'PrefixParam', { type: 'String' });

    new BrowserCustom(stack, 'test-browser-token-prefix', {
      browserCustomName: 'test_browser_token_prefix',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: 'my-recording-bucket',
          objectKey: prefixParam.valueAsString, // unresolved token
        },
      },
    });

    const template = Template.fromStack(stack);

    // A tokenized prefix cannot be inspected at synth, so it is embedded verbatim
    // and a '*' is appended after it. No trailing-slash normalization is possible.
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: [
          Match.objectLike({
            Action: [
              's3:PutObject',
              's3:ListMultipartUploadParts',
              's3:AbortMultipartUpload',
            ],
            Resource: {
              'Fn::Join': [
                '',
                [
                  'arn:',
                  { Ref: 'AWS::Partition' },
                  ':s3:::my-recording-bucket/',
                  { Ref: 'PrefixParam' },
                  '*',
                ],
              ],
            },
          }),
        ],
      },
    });
  });

  test('Should handle recording config with enabled true but no S3 location', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    expect(() => {
      new BrowserCustom(stack, 'test-browser-recording-no-s3', {
        browserCustomName: 'test_browser_recording_no_s3',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        recordingConfig: {
          enabled: true,
          // No s3Location provided
        },
      });
    }).not.toThrow();
  });

  test('Should handle recording config with S3 location but enabled false', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const recordingBucket = new s3.Bucket(stack, 'RecordingBucket2', {
      bucketName: 'test-browser-recordings-2',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    expect(() => {
      new BrowserCustom(stack, 'test-browser-recording-disabled', {
        browserCustomName: 'test_browser_recording_disabled',
        networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
        recordingConfig: {
          enabled: false,
          s3Location: {
            bucketName: recordingBucket.bucketName,
            objectKey: 'recordings/',
          },
        },
      });
    }).not.toThrow();
  });

  test('Should test metric methods with different configurations', () => {
    const app = new cdk.App();
    const stack = new cdk.Stack(app, 'test-stack', {
      env: {
        account: '123456789012',
        region: 'us-east-1',
      },
    });

    const browser = new BrowserCustom(stack, 'test-browser-metrics', {
      browserCustomName: 'test_browser_metrics',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });

    // Test various metric methods
    const latencyMetric = browser.metricLatencyForApiOperation('TestOperation');
    const invocationsMetric = browser.metricInvocationsForApiOperation('TestOperation');
    const errorsMetric = browser.metricErrorsForApiOperation('TestOperation');
    const sessionDurationMetric = browser.metricSessionDuration();
    const takeOverCountMetric = browser.metricTakeOverCount();
    const takeOverReleaseCountMetric = browser.metricTakeOverReleaseCount();
    const takeOverDurationMetric = browser.metricTakeOverDuration();

    expect(latencyMetric).toBeDefined();
    expect(invocationsMetric).toBeDefined();
    expect(errorsMetric).toBeDefined();
    expect(sessionDurationMetric).toBeDefined();
    expect(takeOverCountMetric).toBeDefined();
    expect(takeOverReleaseCountMetric).toBeDefined();
    expect(takeOverDurationMetric).toBeDefined();
  });
});

describe('BrowserCustom error metric methods tests', () => {
  let stack: cdk.Stack;
  let browser: BrowserCustom;

  function alarmForMetric(id: string, metric: cloudwatch.Metric): void {
    new cloudwatch.Alarm(stack, id, { metric, evaluationPeriods: 1, threshold: 1 });
  }

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'test-stack');
    browser = new BrowserCustom(stack, 'test-browser-error-metrics', {
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });
  });

  test('metricThrottlesForApiOperation() produces Throttles with Operation dimension', () => {
    alarmForMetric('ThrottlesAlarm', browser.metricThrottlesForApiOperation('TestOperation'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'Throttles',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'TestOperation' }),
        Match.objectLike({ Name: 'Resource', Value: { 'Fn::GetAtt': [Match.stringLikeRegexp('testbrowsererrormetrics.*'), 'BrowserArn'] } }),
      ]),
    });
  });

  test('metricSystemErrorsForApiOperation() produces SystemErrors with Operation dimension', () => {
    alarmForMetric('SysErrAlarm', browser.metricSystemErrorsForApiOperation('TestOperation'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'SystemErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'TestOperation' }),
      ]),
    });
  });

  test('metricUserErrorsForApiOperation() produces UserErrors with Operation dimension', () => {
    alarmForMetric('UserErrAlarm', browser.metricUserErrorsForApiOperation('TestOperation'));

    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::CloudWatch::Alarm', {
      MetricName: 'UserErrors',
      Namespace: 'AWS/Bedrock-AgentCore',
      Statistic: 'Sum',
      Dimensions: Match.arrayWith([
        Match.objectLike({ Name: 'Operation', Value: 'TestOperation' }),
      ]),
    });
  });
});

describe('BrowserCustom browser signing configuration tests', () => {
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

  test('Should default to DISABLED when browser signing is not specified', () => {
    const browser = new BrowserCustom(stack, 'test-browser-default', {
      browserCustomName: 'test_browser_default',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });

    expect(browser.browserSigning).toBe(BrowserSigning.DISABLED);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: false,
      },
    });
  });

  test('Should set browser signing to ENABLED when explicitly specified', () => {
    const browser = new BrowserCustom(stack, 'test-browser-enabled', {
      browserCustomName: 'test_browser_enabled',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.ENABLED,
    });

    expect(browser.browserSigning).toBe(BrowserSigning.ENABLED);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: true,
      },
    });
  });

  test('Should set browser signing to DISABLED when explicitly specified', () => {
    const browser = new BrowserCustom(stack, 'test-browser-disabled', {
      browserCustomName: 'test_browser_disabled',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.DISABLED,
    });

    expect(browser.browserSigning).toBe(BrowserSigning.DISABLED);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: false,
      },
    });
  });

  test('Should have BrowserSigning property in CloudFormation template when default', () => {
    new BrowserCustom(stack, 'test-browser-default-signing', {
      browserCustomName: 'test_browser_default_signing',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser_default_signing',
      BrowserSigning: { Enabled: false },
    });
  });

  test('Should have BrowserSigning property with Enabled true when ENABLED', () => {
    new BrowserCustom(stack, 'test-browser-enabled-signing', {
      browserCustomName: 'test_browser_enabled_signing',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.ENABLED,
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser_enabled_signing',
      BrowserSigning: { Enabled: true },
    });
  });

  test('Should have BrowserSigning property with Enabled false when DISABLED', () => {
    new BrowserCustom(stack, 'test-browser-disabled-signing', {
      browserCustomName: 'test_browser_disabled_signing',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.DISABLED,
    });

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      Name: 'test_browser_disabled_signing',
      BrowserSigning: { Enabled: false },
    });
  });

  test('Should work with browser signing ENABLED and recording config', () => {
    const recordingBucket = new s3.Bucket(stack, 'RecordingBucket', {
      bucketName: 'test-browser-recordings',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const browser = new BrowserCustom(stack, 'test-browser-signing-recording', {
      browserCustomName: 'test_browser_signing_recording',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.ENABLED,
      recordingConfig: {
        enabled: true,
        s3Location: {
          bucketName: recordingBucket.bucketName,
          objectKey: 'recordings/',
        },
      },
    });

    expect(browser.browserSigning).toBe(BrowserSigning.ENABLED);
    expect(browser.recordingConfig?.enabled).toBe(true);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: true,
      },
      RecordingConfig: {
        Enabled: true,
      },
    });
  });

  test('Should work with browser signing DISABLED and VPC configuration', () => {
    const vpc = new ec2.Vpc(stack, 'testVPC');

    const browser = new BrowserCustom(stack, 'test-browser-signing-vpc', {
      browserCustomName: 'test_browser_signing_vpc',
      networkConfiguration: BrowserNetworkConfiguration.usingVpc(stack, {
        vpc: vpc,
      }),
      browserSigning: BrowserSigning.DISABLED,
    });

    expect(browser.browserSigning).toBe(BrowserSigning.DISABLED);
    expect(browser.networkConfiguration.networkMode).toBe('VPC');

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: false,
      },
      NetworkConfiguration: {
        NetworkMode: 'VPC',
      },
    });
  });

  test('Should work with browser signing ENABLED and custom execution role', () => {
    const customRole = new iam.Role(stack, 'CustomExecutionRole', {
      assumedBy: new iam.ServicePrincipal('bedrock-agentcore.amazonaws.com'),
      roleName: 'custom-browser-execution-role',
    });

    const browser = new BrowserCustom(stack, 'test-browser-signing-role', {
      browserCustomName: 'test_browser_signing_role',
      networkConfiguration: BrowserNetworkConfiguration.usingPublicNetwork(),
      browserSigning: BrowserSigning.ENABLED,
      executionRole: customRole,
    });

    expect(browser.browserSigning).toBe(BrowserSigning.ENABLED);
    expect(browser.executionRole).toBe(customRole);

    const template = Template.fromStack(stack);

    template.hasResourceProperties('AWS::BedrockAgentCore::BrowserCustom', {
      BrowserSigning: {
        Enabled: true,
      },
    });
  });
});

describe('Browser Optional Physical Names', () => {
  let stack: cdk.Stack;

  beforeEach(() => {
    const app = new cdk.App();
    stack = new cdk.Stack(app, 'TestStack', {
      env: { account: '123456789012', region: 'us-east-1' },
    });
  });

  test('Should create BrowserCustom without browserCustomName (auto-generated)', () => {
    const browser = new BrowserCustom(stack, 'TestBrowser', {
    });

    expect(browser.browserCustomName).toBeDefined();
    expect(browser.browserCustomName).not.toBe('');
  });
});
