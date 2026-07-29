import { Template, Match } from '../../assertions';
import * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import * as cdk from '../../core';
import { Arn, ArnFormat } from '../../core';
import * as lambda from '../lib';

describe('capacity provider', () => {
  let stack: cdk.Stack;
  let vpc: ec2.Vpc;
  let subnets: ec2.ISubnet[];
  let securityGroup: ec2.SecurityGroup;

  beforeEach(() => {
    stack = new cdk.Stack();
    vpc = new ec2.Vpc(stack, 'Vpc');
    subnets = vpc.privateSubnets.slice(0, 2);
    securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', { vpc });
  });

  describe('When creating a capacity provider', () => {
    test('with minimal properties', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        VpcConfig: {
          SubnetIds: Match.arrayWith([
            { Ref: Match.stringLikeRegexp('VpcPrivateSubnet.*') },
          ]),
          SecurityGroupIds: [
            { 'Fn::GetAtt': [Match.stringLikeRegexp('SecurityGroup.*'), 'GroupId'] },
          ],
        },
        PermissionsConfig: {
          CapacityProviderOperatorRoleArn: {
            'Fn::GetAtt': [Match.stringLikeRegexp('MyCapacityProviderOperatorRole.*'), 'Arn'],
          },
        },
      });
    });

    test('with all properties configured', () => {
      // GIVEN
      const operatorRole = new iam.Role(stack, 'CustomOperatorRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      });
      const kmsKey = new kms.Key(stack, 'Key');

      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        capacityProviderName: 'my-capacity-provider',
        subnets,
        securityGroups: [securityGroup],
        operatorRole: operatorRole,
        architectures: [lambda.Architecture.X86_64],
        instanceTypeFilter: lambda.InstanceTypeFilter.allow([
          ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
          ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL),
        ]),
        maxVCpuCount: 12,
        scalingOptions: lambda.ScalingOptions.manual([
          lambda.TargetTrackingScalingPolicy.cpuUtilization(70),
        ]),
        kmsKey,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        CapacityProviderName: 'my-capacity-provider',
        VpcConfig: {
          SubnetIds: Match.arrayWith([
            { Ref: Match.stringLikeRegexp('VpcPrivateSubnet.*') },
          ]),
          SecurityGroupIds: [
            { 'Fn::GetAtt': [Match.stringLikeRegexp('SecurityGroup.*'), 'GroupId'] },
          ],
        },
        PermissionsConfig: {
          CapacityProviderOperatorRoleArn: {
            'Fn::GetAtt': [Match.stringLikeRegexp('CustomOperatorRole.*'), 'Arn'],
          },
        },
        InstanceRequirements: {
          AllowedInstanceTypes: ['t3.micro', 't3.small'],
          Architectures: ['x86_64'],
        },
        CapacityProviderScalingConfig: {
          MaxVCpuCount: 12,
          ScalingMode: 'Manual',
          ScalingPolicies: [
            {
              PredefinedMetricType: 'LambdaCapacityProviderAverageCPUUtilization',
              TargetValue: 70,
            },
          ],
        },
        KmsKeyArn: {
          'Fn::GetAtt': [Match.stringLikeRegexp('Key.*'), 'Arn'],
        },
      });
    });

    test('with excluded instance types', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        instanceTypeFilter: lambda.InstanceTypeFilter.exclude([
          ec2.InstanceType.of(ec2.InstanceClass.T2, ec2.InstanceSize.NANO),
        ]),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        InstanceRequirements: {
          ExcludedInstanceTypes: ['t2.nano'],
        },
      });
    });

    test('with Auto scaling mode', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        scalingOptions: lambda.ScalingOptions.auto(),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        CapacityProviderScalingConfig: {
          ScalingMode: 'Auto',
        },
      });
    });

    test('creates default operator role', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'lambda.amazonaws.com' },
            },
          ],
        },
        ManagedPolicyArns: [
          {
            'Fn::Join': ['', [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':iam::aws:policy/AWSLambdaManagedEC2ResourceOperator',
            ]],
          },
        ],
      });
    });

    test('uses custom operator role without creating default', () => {
      // GIVEN
      const customRole = new iam.Role(stack, 'CustomRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      });

      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        operatorRole: customRole,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        PermissionsConfig: {
          CapacityProviderOperatorRoleArn: {
            'Fn::GetAtt': [Match.stringLikeRegexp('CustomRole.*'), 'Arn'],
          },
        },
      });
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1); // Only custom role

      const template = Template.fromStack(stack);
      const roles = template.findResources('AWS::IAM::Role');
      const customRoleKey = Object.keys(roles).find(key => key.startsWith('CustomRole'));
      expect(roles[customRoleKey!].Properties.ManagedPolicyArns).toBeUndefined();
    });

    test('omits KMS key when not provided', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      const template = Template.fromStack(stack);
      const capacityProviders = template.findResources('AWS::Lambda::CapacityProvider');
      const cpResource = Object.values(capacityProviders)[0];
      expect(cpResource.Properties.KmsKeyArn).toBeUndefined();
    });

    test('default operator role trusts lambda service', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::IAM::Role', {
        AssumeRolePolicyDocument: {
          Statement: [
            {
              Action: 'sts:AssumeRole',
              Effect: 'Allow',
              Principal: { Service: 'lambda.amazonaws.com' },
            },
          ],
        },
      });
    });
  });

  describe('InstanceTypeFilter', () => {
    test('allow() sets only allowedInstanceTypes', () => {
      const filter = lambda.InstanceTypeFilter.allow([
        ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      ]);

      expect(filter.allowedInstanceTypes).toHaveLength(1);
      expect(filter.excludedInstanceTypes).toBeUndefined();
    });

    test('exclude() sets only excludedInstanceTypes', () => {
      const filter = lambda.InstanceTypeFilter.exclude([
        ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
      ]);

      expect(filter.excludedInstanceTypes).toHaveLength(1);
      expect(filter.allowedInstanceTypes).toBeUndefined();
    });
  });

  describe('validation', () => {
    const basicName = 'MyCapacityProvider';

    test('throws when instanceTypeList is supplied as empty', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          instanceTypeFilter: lambda.InstanceTypeFilter.allow([]),
        });
      }).toThrow();

      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          instanceTypeFilter: lambda.InstanceTypeFilter.exclude([]),
        });
      }).toThrow();
    });

    test('throws when maxVCpuCount is less than 2', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          maxVCpuCount: 1,
        });
      }).toThrow('maxVCpuCount must be between 12 and 15000');
    });

    test('throws when maxVCpuCount is greater than 15000', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          maxVCpuCount: 15001,
        });
      }).toThrow('maxVCpuCount must be between 12 and 15000');
    });

    test('throws when capacityProviderName contains invalid characters', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          capacityProviderName: 'invalid@name',
        });
      }).toThrow();
    });

    test('throws when capacityProviderName is too long', () => {
      // THEN
      const longName = 'a'.repeat(141);
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          capacityProviderName: longName,
        });
      }).toThrow(`Capacity provider name can not be longer than 140 characters but ${longName} has ${longName.length} characters.`);
    });

    test('throws when subnets array is empty', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets: [],
          securityGroups: [securityGroup],
          capacityProviderName: 'MyCapacityProvider',
        });
      }).toThrow(`subnets must contain between 1 and 16 items but ${basicName} has 0 items.`);
    });

    test('throws when subnets array has more than 16 items', () => {
      // GIVEN
      const manySubnets = Array.from({ length: 17 }, (_, i) =>
        ec2.Subnet.fromSubnetId(stack, `Subnet${i}`, `subnet-${i}`),
      );

      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets: manySubnets,
          securityGroups: [securityGroup],
          capacityProviderName: basicName,
        });
      }).toThrow(`subnets must contain between 1 and 16 items but ${basicName} has ${manySubnets.length} items.`);
    });

    test('throws when securityGroups array is empty', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [],
          capacityProviderName: basicName,
        });
      }).toThrow(`securityGroups must contain between 1 and 5 items but ${basicName} has 0 items.`);
    });

    test('throws when more than 5 security groups are specified', () => {
      // GIVEN
      const manySecurityGroups = Array.from({ length: 6 }, (_, i) =>
        ec2.SecurityGroup.fromSecurityGroupId(stack, `SG${i}`, `sg-${i}`),
      );

      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: manySecurityGroups,
          capacityProviderName: basicName,
        });
      }).toThrow(`securityGroups must contain between 1 and 5 items but ${basicName} has ${manySecurityGroups.length} items.`);
    });

    test('throws when no scalingPolicies are specified with Manual scaling mode', () => {
      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          scalingOptions: lambda.ScalingOptions.manual([]),
        });
      }).toThrow();
    });

    test('accepts tokens for all validated fields', () => {
      // GIVEN
      const tokenName = cdk.Fn.ref('CapacityProviderNameParam');
      const tokenMaxVCpu = cdk.Fn.ref('MaxVCpuParam');
      const tokenSubnets = cdk.Fn.split(',', cdk.Fn.ref('SubnetIdsParam'));
      const tokenSecurityGroups = cdk.Fn.split(',', cdk.Fn.ref('SecurityGroupIdsParam'));

      // WHEN - should not throw
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        capacityProviderName: tokenName,
        maxVCpuCount: cdk.Token.asNumber(tokenMaxVCpu),
        subnets: tokenSubnets.map((id, i) => ec2.Subnet.fromSubnetId(stack, `TokenSubnet${i}`, id)),
        securityGroups: tokenSecurityGroups.map((id, i) => ec2.SecurityGroup.fromSecurityGroupId(stack, `TokenSG${i}`, id)),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        CapacityProviderName: { Ref: 'CapacityProviderNameParam' },
        VpcConfig: {
          SubnetIds: { 'Fn::Split': [',', { Ref: 'SubnetIdsParam' }] },
          SecurityGroupIds: { 'Fn::Split': [',', { Ref: 'SecurityGroupIdsParam' }] },
        },
        CapacityProviderScalingConfig: {
          MaxVCpuCount: { Ref: 'MaxVCpuParam' },
        },
      });
    });
  });

  describe('static methods', () => {
    test('fromCapacityProviderName creates imported capacity provider', () => {
      // WHEN
      const imported = lambda.CapacityProvider.fromCapacityProviderName(
        stack,
        'ImportedCP',
        'my-existing-cp',
      );

      // THEN
      expect(imported.capacityProviderName).toBe('my-existing-cp');
      expect(imported.capacityProviderArn).toEqual(
        Arn.format({
          service: 'lambda',
          resource: 'capacity-provider',
          resourceName: 'my-existing-cp',
          arnFormat: ArnFormat.COLON_RESOURCE_NAME,
        }, stack),
      );
    });

    test('fromCapacityProviderArn creates imported capacity provider', () => {
      // GIVEN
      const arn = 'arn:aws:lambda:us-east-1:123456789012:capacity-provider:my-cp';

      // WHEN
      const imported = lambda.CapacityProvider.fromCapacityProviderArn(stack, 'ImportedCP', arn);

      // THEN
      expect(imported.capacityProviderArn).toBe(arn);
      expect(imported.capacityProviderName).toBe('my-cp');
    });
  });

  describe('resource count', () => {
    test('creates exactly one capacity provider resource', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::CapacityProvider', 1);
    });

    test('creates default IAM resources when not provided', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1); // Operator role
    });

    test('does not create IAM resources when provided', () => {
      // GIVEN
      const operatorRole = new iam.Role(stack, 'CustomOperatorRole', {
        assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      });

      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        operatorRole: operatorRole,
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::Lambda::CapacityProvider', 1);
      Template.fromStack(stack).resourceCountIs('AWS::IAM::Role', 1); // Only the custom roles
    });
  });

  describe('addFunction', () => {
    let capacityProvider: lambda.CapacityProvider;

    beforeEach(() => {
      capacityProvider = new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });
    });

    test('configures function with capacity provider', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // WHEN
      capacityProvider.addFunction(func);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        CapacityProviderConfig: {
          LambdaManagedInstancesCapacityProviderConfig: {
            CapacityProviderArn: {
              'Fn::GetAtt': [Match.stringLikeRegexp('MyCapacityProvider.*'), 'Arn'],
            },
          },
        },
      });
    });

    test('configures function with all options', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // WHEN
      capacityProvider.addFunction(func, {
        perExecutionEnvironmentMaxConcurrency: 10,
        executionEnvironmentMemoryGiBPerVCpu: 2,
        latestPublishedScalingConfig: {
          minExecutionEnvironments: 1,
          maxExecutionEnvironments: 5,
        },
        publishToLatestPublished: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        PublishToLatestPublished: true,
        CapacityProviderConfig: {
          LambdaManagedInstancesCapacityProviderConfig: {
            CapacityProviderArn: {
              'Fn::GetAtt': [Match.stringLikeRegexp('MyCapacityProvider.*'), 'Arn'],
            },
            PerExecutionEnvironmentMaxConcurrency: 10,
            ExecutionEnvironmentMemoryGiBPerVCpu: 2,
          },
        },
        FunctionScalingConfig: {
          MinExecutionEnvironments: 1,
          MaxExecutionEnvironments: 5,
        },
      });
    });

    test('configures function with only scaling config', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_LATEST,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // WHEN
      capacityProvider.addFunction(func, {
        latestPublishedScalingConfig: {
          minExecutionEnvironments: 2,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
        CapacityProviderConfig: {
          LambdaManagedInstancesCapacityProviderConfig: {
            CapacityProviderArn: {
              'Fn::GetAtt': [Match.stringLikeRegexp('MyCapacityProvider.*'), 'Arn'],
            },
          },
        },
        FunctionScalingConfig: {
          MinExecutionEnvironments: 2,
        },
      });
    });

    test('throws when minExecutionEnvironments is less than 0', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // THEN
      expect(() => {
        capacityProvider.addFunction(func, {
          latestPublishedScalingConfig: {
            minExecutionEnvironments: -1,
          },
        });
      }).toThrow('minExecutionEnvironments must be between 0 and 15000, but was -1');
    });

    test('throws when minExecutionEnvironments is greater than 15000', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // THEN
      expect(() => {
        capacityProvider.addFunction(func, {
          latestPublishedScalingConfig: {
            minExecutionEnvironments: 15001,
          },
        });
      }).toThrow('minExecutionEnvironments must be between 0 and 15000, but was 15001');
    });

    test('throws when maxExecutionEnvironments is less than 0', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // THEN
      expect(() => {
        capacityProvider.addFunction(func, {
          latestPublishedScalingConfig: {
            maxExecutionEnvironments: -1,
          },
        });
      }).toThrow('maxExecutionEnvironments must be between 0 and 15000, but was -1');
    });

    test('throws when maxExecutionEnvironments is greater than 15000', () => {
      // GIVEN
      const func = new lambda.Function(stack, 'MyFunction', {
        runtime: lambda.Runtime.NODEJS_22_X,
        handler: 'index.handler',
        code: lambda.Code.fromInline('exports.handler = async () => {}'),
      });

      // THEN
      expect(() => {
        capacityProvider.addFunction(func, {
          latestPublishedScalingConfig: {
            maxExecutionEnvironments: 15001,
          },
        });
      }).toThrow('maxExecutionEnvironments must be between 0 and 15000, but was 15001');
    });
  });

  describe('propagateTags', () => {
    test('renders PropagateTags with Mode Explicit when PropagateTags.explicit() used', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        propagateTags: lambda.PropagateTags.explicit({
          env: 'prod',
          team: 'platform',
        }),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        PropagateTags: {
          Mode: 'Explicit',
          ExplicitTags: [
            { Key: 'env', Value: 'prod' },
            { Key: 'team', Value: 'platform' },
          ],
        },
      });
    });

    test('renders PropagateTags with Mode None when PropagateTags.none() used', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
        propagateTags: lambda.PropagateTags.none(),
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        PropagateTags: {
          Mode: 'None',
        },
      });
    });

    test('does not render PropagateTags when prop is omitted', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
        subnets,
        securityGroups: [securityGroup],
      });

      // THEN
      const template = Template.fromStack(stack);
      template.hasResourceProperties('AWS::Lambda::CapacityProvider', {
        PropagateTags: Match.absent(),
      });
    });

    test('throws when explicit tags exceed 40', () => {
      // GIVEN
      const tooManyTags: { [key: string]: string } = {};
      for (let i = 0; i < 41; i++) {
        tooManyTags[`key${i}`] = `value${i}`;
      }

      // THEN
      expect(() => {
        new lambda.CapacityProvider(stack, 'MyCapacityProvider', {
          subnets,
          securityGroups: [securityGroup],
          propagateTags: lambda.PropagateTags.explicit(tooManyTags),
        });
      }).toThrow(/propagateTags explicit tags can have at most 40 tags/);
    });
  });

  describe('telemetry config', () => {
    beforeEach(() => {
      // TelemetryConfig is not yet in the bundled CFN schema — acknowledge the validation warning
      cdk.Validations.of(stack).acknowledge({ id: 'CloudFormation-Validate::F3002', reason: 'TelemetryConfig is a newly launched property not yet in the bundled schema' });
    });

    test('sets TelemetryConfig with systemLogLevel only', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'CP', {
        securityGroups: [securityGroup],
        subnets,
        systemLogLevel: lambda.SystemLogLevel.DEBUG,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        TelemetryConfig: {
          LoggingConfig: {
            SystemLogLevel: 'DEBUG',
          },
        },
      });
    });

    test('sets TelemetryConfig with logGroup only', () => {
      // GIVEN
      const logGroup = new logs.LogGroup(stack, 'LogGroup');

      // WHEN
      new lambda.CapacityProvider(stack, 'CP', {
        securityGroups: [securityGroup],
        subnets,
        logGroup,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        TelemetryConfig: {
          LoggingConfig: {
            LogGroup: { Ref: Match.stringLikeRegexp('LogGroup') },
          },
        },
      });
    });

    test('sets TelemetryConfig with both logGroup and systemLogLevel', () => {
      // GIVEN
      const logGroup = new logs.LogGroup(stack, 'LogGroup', {
        logGroupName: '/my-app/scaling',
      });

      // WHEN
      new lambda.CapacityProvider(stack, 'CP', {
        securityGroups: [securityGroup],
        subnets,
        logGroup,
        systemLogLevel: lambda.SystemLogLevel.WARN,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        TelemetryConfig: {
          LoggingConfig: {
            LogGroup: { Ref: Match.stringLikeRegexp('LogGroup') },
            SystemLogLevel: 'WARN',
          },
        },
      });
    });

    test('does not emit TelemetryConfig when neither logGroup nor systemLogLevel specified', () => {
      // WHEN
      new lambda.CapacityProvider(stack, 'CP', {
        securityGroups: [securityGroup],
        subnets,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::Lambda::CapacityProvider', {
        TelemetryConfig: Match.absent(),
      });
    });
  });
});
