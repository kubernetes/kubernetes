import { Match, Template } from '../../assertions';
import * as ec2 from '../../aws-ec2';
import * as kms from '../../aws-kms';
import * as logs from '../../aws-logs';
import * as cdk from '../../core';
import { ClusterParameterGroup, DatabaseCluster, DatabaseSecret, CaCertificate, StorageType } from '../lib';

describe('DatabaseCluster', () => {
  test('check that instantiation works', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DocDB::DBCluster', {
      Properties: {
        DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
        MasterUsername: 'admin',
        MasterUserPassword: 'tooshort',
        VpcSecurityGroupIds: [{ 'Fn::GetAtt': ['DatabaseSecurityGroup5C91FDCB', 'GroupId'] }],
        StorageEncrypted: true,
      },
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain',
    });

    Template.fromStack(stack).hasResource('AWS::DocDB::DBInstance', {
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain',
    });

    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBSubnetGroup', {
      SubnetIds: [
        { Ref: 'VPCPrivateSubnet1Subnet8BCA10E0' },
        { Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A' },
        { Ref: 'VPCPrivateSubnet3Subnet3EDCD457' },
      ],
    });
  });

  test('can create a cluster with a single instance', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
      MasterUsername: 'admin',
      MasterUserPassword: 'tooshort',
      VpcSecurityGroupIds: [{ 'Fn::GetAtt': ['DatabaseSecurityGroup5C91FDCB', 'GroupId'] }],
    });
  });

  test('can specify instance CA certificate', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      caCertificate: CaCertificate.RDS_CA_RSA4096_G1,
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBInstance', {
      DBClusterIdentifier: { Ref: 'DatabaseB269D8BB' },
      CACertificateIdentifier: 'rds-ca-rsa4096-g1',
    });
  });

  test('errors when less than one instance is specified', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    expect(() => {
      new DatabaseCluster(stack, 'Database', {
        instances: 0,
        masterUser: {
          username: 'admin',
        },
        vpc,
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.LARGE),
      });
    }).toThrow('At least one instance is required');
  });

  test('errors when only one subnet is specified', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC', {
      maxAzs: 1,
    });

    // WHEN
    expect(() => {
      new DatabaseCluster(stack, 'Database', {
        instances: 1,
        masterUser: {
          username: 'admin',
        },
        vpc,
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.LARGE),
        vpcSubnets: {
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      });
    }).toThrow('Cluster requires at least 2 subnets, got 1');
  });

  test('secret attachment target type is correct', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::SecretTargetAttachment', {
      SecretId: { Ref: 'DatabaseSecret3B817195' },
      TargetId: { Ref: 'DatabaseB269D8BB' },
      TargetType: 'AWS::DocDB::DBCluster',
    });
  });

  test('can create a cluster with imported vpc and security group', () => {
    // GIVEN
    const stack = testStack();
    const vpc = ec2.Vpc.fromLookup(stack, 'VPC', {
      vpcId: 'VPC12345',
    });
    const sg = ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG', 'SecurityGroupId12345');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      securityGroup: sg,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
      MasterUsername: 'admin',
      MasterUserPassword: 'tooshort',
      VpcSecurityGroupIds: ['SecurityGroupId12345'],
    });
  });

  test('can configure cluster deletion protection', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      deletionProtection: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      DeletionProtection: true,
    });
  });

  test('cluster with parameter group', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    const group = new ClusterParameterGroup(stack, 'Params', {
      family: 'hello',
      description: 'bye',
      parameters: {
        param: 'value',
      },
    });
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      parameterGroup: group,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      DBClusterParameterGroupName: { Ref: 'ParamsA8366201' },
    });
  });

  test('cluster with imported parameter group', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    const group = ClusterParameterGroup.fromParameterGroupName(stack, 'Params', 'ParamGroupName');

    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      parameterGroup: group,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      DBClusterParameterGroupName: 'ParamGroupName',
    });
  });

  test('creates a secret when master credentials are not specified', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      MasterUsername: {
        'Fn::Join': [
          '',
          [
            '{{resolve:secretsmanager:',
            {
              Ref: 'DatabaseSecret3B817195',
            },
            ':SecretString:username::}}',
          ],
        ],
      },
      MasterUserPassword: {
        'Fn::Join': [
          '',
          [
            '{{resolve:secretsmanager:',
            {
              Ref: 'DatabaseSecret3B817195',
            },
            ':SecretString:password::}}',
          ],
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::Secret', {
      GenerateSecretString: {
        ExcludeCharacters: '\"@/',
        GenerateStringKey: 'password',
        PasswordLength: 41,
        SecretStringTemplate: '{"username":"admin"}',
      },
    });
  });

  test('creates a secret with excludeCharacters', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        excludeCharacters: '"@/()[]',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::Secret', {
      GenerateSecretString: Match.objectLike({
        ExcludeCharacters: '\"@/()[]',
      }),
    });
  });

  test('creates a secret with secretName set', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        secretName: '/myapp/mydocdb/masteruser',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::Secret', {
      Name: '/myapp/mydocdb/masteruser',
    });
  });

  test('create an encrypted cluster with custom KMS key', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      kmsKey: new kms.Key(stack, 'Key'),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      KmsKeyId: {
        'Fn::GetAtt': [
          'Key961B73FD',
          'Arn',
        ],
      },
      StorageEncrypted: true,
    });
  });

  test('creating a cluster defaults to using encryption', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      StorageEncrypted: true,
    });
  });

  test('supplying a KMS key with storageEncryption false throws an error', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    function action() {
      new DatabaseCluster(stack, 'Database', {
        masterUser: {
          username: 'admin',
        },
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
        vpc,
        kmsKey: new kms.Key(stack, 'Key'),
        storageEncrypted: false,
      });
    }

    // THEN
    expect(action).toThrow();
  });

  test('cluster exposes different read and write endpoints', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    expect(stack.resolve(cluster.clusterEndpoint)).not.toBe(stack.resolve(cluster.clusterReadEndpoint));
  });

  test('instance identifier used when present', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    const instanceIdentifierBase = 'instanceidentifierbase-';
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      instanceIdentifierBase,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBInstance', {
      DBInstanceIdentifier: `${instanceIdentifierBase}1`,
    });
  });

  test('cluster identifier used', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    const clusterIdentifier = 'clusteridentifier-';
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      dbClusterName: clusterIdentifier,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBInstance', {
      DBInstanceIdentifier: `${clusterIdentifier}instance1`,
    });
  });

  test('imported cluster has supplied attributes', () => {
    // GIVEN
    const stack = testStack();

    // WHEN
    const cluster = DatabaseCluster.fromDatabaseClusterAttributes(stack, 'Database', {
      clusterEndpointAddress: 'addr',
      clusterIdentifier: 'identifier',
      instanceEndpointAddresses: ['addr'],
      instanceIdentifiers: ['identifier'],
      port: 3306,
      readerEndpointAddress: 'reader-address',
      securityGroup: ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG', 'sg-123456789', {
        allowAllOutbound: false,
      }),
    });

    // THEN
    expect(cluster.clusterEndpoint.hostname).toEqual('addr');
    expect(cluster.clusterEndpoint.port).toEqual(3306);
    expect(cluster.clusterIdentifier).toEqual('identifier');
    expect(cluster.instanceIdentifiers).toEqual(['identifier']);
    expect(cluster.clusterReadEndpoint.hostname).toEqual('reader-address');
    expect(cluster.securityGroupId).toEqual('sg-123456789');
  });

  test('imported cluster with imported security group honors allowAllOutbound', () => {
    // GIVEN
    const stack = testStack();

    const cluster = DatabaseCluster.fromDatabaseClusterAttributes(stack, 'Database', {
      clusterEndpointAddress: 'addr',
      clusterIdentifier: 'identifier',
      instanceEndpointAddresses: ['addr'],
      instanceIdentifiers: ['identifier'],
      port: 3306,
      readerEndpointAddress: 'reader-address',
      securityGroup: ec2.SecurityGroup.fromSecurityGroupId(stack, 'SG', 'sg-123456789', {
        allowAllOutbound: false,
      }),
    });

    // WHEN
    cluster.connections.allowToAnyIpv4(ec2.Port.tcp(443));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::EC2::SecurityGroupEgress', {
      GroupId: 'sg-123456789',
    });
  });

  test('minimal imported cluster throws on accessing attributes for unprovided parameters', () => {
    const stack = testStack();

    const cluster = DatabaseCluster.fromDatabaseClusterAttributes(stack, 'Database', {
      clusterIdentifier: 'identifier',
    });

    expect(cluster.clusterIdentifier).toEqual('identifier');
    expect(() => cluster.clusterEndpoint).toThrow(/Cannot access `clusterEndpoint` of an imported cluster/);
    expect(() => cluster.clusterReadEndpoint).toThrow(/Cannot access `clusterReadEndpoint` of an imported cluster/);
    expect(() => cluster.instanceIdentifiers).toThrow(/Cannot access `instanceIdentifiers` of an imported cluster/);
    expect(() => cluster.instanceEndpoints).toThrow(/Cannot access `instanceEndpoints` of an imported cluster/);
    expect(() => cluster.securityGroupId).toThrow(/Cannot access `securityGroupId` of an imported cluster/);
  });

  test('backup retention period respected', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      backup: {
        retention: cdk.Duration.days(20),
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      BackupRetentionPeriod: 20,
    });
  });

  test('backup maintenance window respected', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      backup: {
        retention: cdk.Duration.days(20),
        preferredWindow: '07:34-08:04',
      },
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      BackupRetentionPeriod: 20,
      PreferredBackupWindow: '07:34-08:04',
    });
  });

  test('regular maintenance window respected', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      preferredMaintenanceWindow: 'tue:07:34-tue:08:04',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      PreferredMaintenanceWindow: 'tue:07:34-tue:08:04',
    });
  });

  test('instanceMaintenanceWindow propagates to all auto-created instances', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      preferredMaintenanceWindow: 'tue:04:17-tue:04:47',
      instances: 2,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      instanceMaintenanceWindow: 'sat:09:00-sat:09:30',
    });

    // THEN
    const template = Template.fromStack(stack);
    template.hasResourceProperties('AWS::DocDB::DBCluster', {
      PreferredMaintenanceWindow: 'tue:04:17-tue:04:47',
    });
    template.resourceCountIs('AWS::DocDB::DBInstance', 2);
    template.allResources('AWS::DocDB::DBInstance', {
      Properties: Match.objectLike({
        PreferredMaintenanceWindow: 'sat:09:00-sat:09:30',
      }),
    });
  });

  test('maintenance window is omitted on instances when instanceMaintenanceWindow is not set', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBInstance', {
      PreferredMaintenanceWindow: Match.absent(),
    });
  });

  test.each([
    'mon:00:00-mon:00:30',
    'sun:23:45-mon:00:15',
    'wed:12:00-thu:11:59',
    'SAT:09:00-SAT:09:30', // case-insensitive
  ])('accepts valid maintenance window %s', (window) => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN / THEN
    expect(() => new DatabaseCluster(stack, 'Database', {
      masterUser: { username: 'admin' },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      preferredMaintenanceWindow: window,
      instanceMaintenanceWindow: window,
    })).not.toThrow();
  });

  test.each([
    ['07:34-08:04', 'missing day prefix'],
    ['tue:04:17', 'missing end range'],
    ['funday:04:17-tue:04:47', 'invalid day'],
    ['tue:24:00-tue:24:30', 'invalid hour'],
    ['tue:04:60-tue:05:30', 'invalid minute'],
    ['tue:04:17-tue:04:47 ', 'trailing whitespace'],
    ['', 'empty string'],
  ])('fails for invalid preferredMaintenanceWindow %s (%s)', (window, _reason) => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN / THEN
    expect(() => new DatabaseCluster(stack, 'Database', {
      masterUser: { username: 'admin' },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      preferredMaintenanceWindow: window,
    })).toThrow(/preferredMaintenanceWindow must be in the format ddd:hh24:mi-ddd:hh24:mi/);
  });

  test.each([
    ['07:34-08:04', 'missing day prefix'],
    ['funday:04:17-tue:04:47', 'invalid day'],
    ['tue:24:00-tue:24:30', 'invalid hour'],
    ['tue:04:60-tue:05:30', 'invalid minute'],
    ['', 'empty string'],
  ])('fails for invalid instanceMaintenanceWindow %s (%s)', (window, _reason) => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN / THEN
    expect(() => new DatabaseCluster(stack, 'Database', {
      masterUser: { username: 'admin' },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      instanceMaintenanceWindow: window,
    })).toThrow(/instanceMaintenanceWindow must be in the format ddd:hh24:mi-ddd:hh24:mi/);
  });

  test('skips maintenance window validation for tokenized values', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const clusterWindow = new cdk.CfnParameter(stack, 'ClusterWindow', { type: 'String' }).valueAsString;
    const instanceWindow = new cdk.CfnParameter(stack, 'InstanceWindow', { type: 'String' }).valueAsString;

    // WHEN / THEN
    expect(() => new DatabaseCluster(stack, 'Database', {
      masterUser: { username: 'admin' },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
      preferredMaintenanceWindow: clusterWindow,
      instanceMaintenanceWindow: instanceWindow,
    })).not.toThrow();
  });

  test('can configure CloudWatchLogs for audit', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      exportAuditLogsToCloudWatch: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      EnableCloudwatchLogsExports: ['audit'],
    });
  });

  test('can configure CloudWatchLogs for profiler', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      exportProfilerLogsToCloudWatch: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      EnableCloudwatchLogsExports: ['profiler'],
    });
  });

  test('can configure CloudWatchLogs for all logs', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      exportAuditLogsToCloudWatch: true,
      exportProfilerLogsToCloudWatch: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      EnableCloudwatchLogsExports: ['audit', 'profiler'],
    });
  });

  test('can set CloudWatch log retention', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      exportAuditLogsToCloudWatch: true,
      exportProfilerLogsToCloudWatch: true,
      cloudWatchLogsRetention: logs.RetentionDays.THREE_MONTHS,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetention', {
      ServiceToken: {
        'Fn::GetAtt': [
          'LogRetentionaae0aa3c5b4d4f87b02d85b201efdd8aFD4BFC8A',
          'Arn',
        ],
      },
      LogGroupName: { 'Fn::Join': ['', ['/aws/docdb/', { Ref: 'DatabaseB269D8BB' }, '/audit']] },
      RetentionInDays: 90,
    });
    Template.fromStack(stack).hasResourceProperties('Custom::LogRetention', {
      ServiceToken: {
        'Fn::GetAtt': [
          'LogRetentionaae0aa3c5b4d4f87b02d85b201efdd8aFD4BFC8A',
          'Arn',
        ],
      },
      LogGroupName: { 'Fn::Join': ['', ['/aws/docdb/', { Ref: 'DatabaseB269D8BB' }, '/profiler']] },
      RetentionInDays: 90,
    });
  });

  test('can enable Performance Insights on instances', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      enablePerformanceInsights: true,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DocDB::DBInstance', {
      Properties: {
        EnablePerformanceInsights: true,
      },
    });
  });

  test('single user rotation', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });

    // WHEN
    cluster.addRotationSingleUser(cdk.Duration.days(5));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Serverless::Application', {
      Location: {
        ApplicationId: { 'Fn::FindInMap': ['DatabaseRotationSingleUserSARMapping9AEB3E55', { Ref: 'AWS::Partition' }, 'applicationId'] },
        SemanticVersion: { 'Fn::FindInMap': ['DatabaseRotationSingleUserSARMapping9AEB3E55', { Ref: 'AWS::Partition' }, 'semanticVersion'] },
      },
      Parameters: {
        endpoint: {
          'Fn::Join': [
            '',
            [
              'https://secretsmanager.us-test-1.',
              { Ref: 'AWS::URLSuffix' },
            ],
          ],
        },
        functionName: 'DatabaseRotationSingleUser458A45BE',
        excludeCharacters: '\"@/',
        vpcSubnetIds: {
          'Fn::Join': [
            '',
            [
              { Ref: 'VPCPrivateSubnet1Subnet8BCA10E0' },
              ',',
              { Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A' },
              ',',
              { Ref: 'VPCPrivateSubnet3Subnet3EDCD457' },
            ],
          ],
        },
        vpcSecurityGroupIds: {
          'Fn::GetAtt': ['DatabaseRotationSingleUserSecurityGroupAC6E0E73', 'GroupId'],
        },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::RotationSchedule', {
      SecretId: { Ref: 'DatabaseSecretAttachmentE5D1B020' },
      RotationLambdaARN: {
        'Fn::GetAtt': ['DatabaseRotationSingleUser65F55654', 'Outputs.RotationLambdaARN'],
      },
      RotationRules: {
        ScheduleExpression: 'rate(5 days)',
      },
    });
  });

  test('single user rotation requires secret', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('secret'),
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });

    // WHEN
    function addSingleUserRotation() {
      cluster.addRotationSingleUser(cdk.Duration.days(10));
    }

    // THEN
    expect(addSingleUserRotation).toThrow();
  });

  test('no multiple single user rotations', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });

    // WHEN
    cluster.addRotationSingleUser(cdk.Duration.days(5));
    function addSecondRotation() {
      cluster.addRotationSingleUser(cdk.Duration.days(10));
    }

    // THEN
    expect(addSecondRotation).toThrow();
  });

  test('multi user rotation', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });
    const userSecret = new DatabaseSecret(stack, 'UserSecret', {
      username: 'seconduser',
      masterSecret: cluster.secret,
    });
    userSecret.attach(cluster);

    // WHEN
    cluster.addRotationMultiUser('Rotation', {
      secret: userSecret,
      automaticallyAfter: cdk.Duration.days(5),
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::Serverless::Application', {
      Location: {
        ApplicationId: { 'Fn::FindInMap': ['DatabaseRotationSARMappingE46CFA92', { Ref: 'AWS::Partition' }, 'applicationId'] },
        SemanticVersion: { 'Fn::FindInMap': ['DatabaseRotationSARMappingE46CFA92', { Ref: 'AWS::Partition' }, 'semanticVersion'] },
      },
      Parameters: {
        endpoint: {
          'Fn::Join': [
            '',
            [
              'https://secretsmanager.us-test-1.',
              { Ref: 'AWS::URLSuffix' },
            ],
          ],
        },
        functionName: 'DatabaseRotation0D47EBD2',
        excludeCharacters: '\"@/',
        vpcSubnetIds: {
          'Fn::Join': [
            '',
            [
              { Ref: 'VPCPrivateSubnet1Subnet8BCA10E0' },
              ',',
              { Ref: 'VPCPrivateSubnet2SubnetCFCDAA7A' },
              ',',
              { Ref: 'VPCPrivateSubnet3Subnet3EDCD457' },
            ],
          ],
        },
        vpcSecurityGroupIds: {
          'Fn::GetAtt': ['DatabaseRotationSecurityGroup17736B63', 'GroupId'],
        },
        masterSecretArn: { Ref: 'DatabaseSecretAttachmentE5D1B020' },
      },
    });
    Template.fromStack(stack).hasResourceProperties('AWS::SecretsManager::RotationSchedule', {
      SecretId: { Ref: 'UserSecret0463E4F5' },
      RotationLambdaARN: {
        'Fn::GetAtt': ['DatabaseRotation6B6E1D86', 'Outputs.RotationLambdaARN'],
      },
      RotationRules: {
        ScheduleExpression: 'rate(5 days)',
      },
    });
  });

  test('multi user rotation requires secret', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('secret'),
      },
      vpc,
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });
    const userSecret = new DatabaseSecret(stack, 'UserSecret', {
      username: 'seconduser',
      masterSecret: cluster.secret,
    });
    userSecret.attach(cluster);

    // WHEN
    function addMultiUserRotation() {
      cluster.addRotationMultiUser('Rotation', {
        secret: userSecret,
      });
    }

    // THEN
    expect(addMultiUserRotation).toThrow();
  });

  test('adds security groups', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');
    const cluster = new DatabaseCluster(stack, 'Database', {
      vpc,
      masterUser: {
        username: 'admin',
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
    });
    const securityGroup = new ec2.SecurityGroup(stack, 'SecurityGroup', {
      vpc,
    });

    // WHEN
    cluster.addSecurityGroups(securityGroup);

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      VpcSecurityGroupIds: Match.arrayWith([stack.resolve(securityGroup.securityGroupId)]),
    });
  });

  test('can create a cluster with snapshot removal policy', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      removalPolicy: cdk.RemovalPolicy.SNAPSHOT,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DocDB::DBCluster', {
      Properties: {
        DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
        MasterUsername: 'admin',
        MasterUserPassword: 'tooshort',
        VpcSecurityGroupIds: [{ 'Fn::GetAtt': ['DatabaseSecurityGroup5C91FDCB', 'GroupId'] }],
      },
      DeletionPolicy: 'Snapshot',
      UpdateReplacePolicy: 'Snapshot',
    });

    // Associated instance gets delete policy
    Template.fromStack(stack).hasResource('AWS::DocDB::DBInstance', {
      DeletionPolicy: 'Delete',
      UpdateReplacePolicy: 'Delete',
    });

    // Associated security group gets delete policy
    Template.fromStack(stack).hasResource('AWS::EC2::SecurityGroup', {
      DeletionPolicy: 'Delete',
      UpdateReplacePolicy: 'Delete',
    });
  });

  test('can specify instances removal policy', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      removalPolicy: cdk.RemovalPolicy.SNAPSHOT,
      instanceRemovalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DocDB::DBCluster', {
      Properties: {
        DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
        MasterUsername: 'admin',
        MasterUserPassword: 'tooshort',
        VpcSecurityGroupIds: [{ 'Fn::GetAtt': ['DatabaseSecurityGroup5C91FDCB', 'GroupId'] }],
      },
      DeletionPolicy: 'Snapshot',
      UpdateReplacePolicy: 'Snapshot',
    });

    // Associated instance gets specified policy
    Template.fromStack(stack).hasResource('AWS::DocDB::DBInstance', {
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain',
    });
  });

  test('can specify security group removal policy', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      instances: 1,
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      removalPolicy: cdk.RemovalPolicy.SNAPSHOT,
      securityGroupRemovalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::DocDB::DBCluster', {
      Properties: {
        DBSubnetGroupName: { Ref: 'DatabaseSubnets56F17B9A' },
        MasterUsername: 'admin',
        MasterUserPassword: 'tooshort',
        VpcSecurityGroupIds: [{ 'Fn::GetAtt': ['DatabaseSecurityGroup5C91FDCB', 'GroupId'] }],
      },
      DeletionPolicy: 'Snapshot',
      UpdateReplacePolicy: 'Snapshot',
    });

    // Associated security group gets specified policy
    Template.fromStack(stack).hasResource('AWS::EC2::SecurityGroup', {
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain',
    });
  });

  test('instances removal policy cannot be set to snapshot', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    // THEN
    expect(() => {
      new DatabaseCluster(stack, 'Database', {
        instances: 1,
        masterUser: {
          username: 'admin',
          password: cdk.SecretValue.unsafePlainText('tooshort'),
        },
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
        vpc,
        instanceRemovalPolicy: cdk.RemovalPolicy.SNAPSHOT,
      });
    }).toThrow(/AWS::DocDB::DBInstance does not support the SNAPSHOT removal policy/);
  });

  test('security group removal policy cannot be set to snapshot', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    // THEN
    expect(() => {
      new DatabaseCluster(stack, 'Database', {
        instances: 1,
        masterUser: {
          username: 'admin',
          password: cdk.SecretValue.unsafePlainText('tooshort'),
        },
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
        vpc,
        securityGroupRemovalPolicy: cdk.RemovalPolicy.SNAPSHOT,
      });
    }).toThrow(/AWS::EC2::SecurityGroup does not support the SNAPSHOT removal policy/);
  });

  test('cluster with copyTagsToSnapshot default', () => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      CopyTagsToSnapshot: Match.absent(),
    });
  });

  test.each([false, true])('cluster with copyTagsToSnapshot set', (value) => {
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new DatabaseCluster(stack, 'Database', {
      masterUser: {
        username: 'admin',
        password: cdk.SecretValue.unsafePlainText('tooshort'),
      },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE2, ec2.InstanceSize.SMALL),
      vpc,
      copyTagsToSnapshot: value,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
      CopyTagsToSnapshot: value,
    });
  });

  test.each(['1.0.0.1', '01.1.0', '1.0', '-1', '-0.1', 'abc', '1.0.a', 'a.b.c'])('throw error for invalid engine version %s', (engineVersion: string) => {
    // GIVEN
    const stack = testStack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // THEN
    expect(() => {
      new DatabaseCluster(stack, 'Database', {
        instances: 1,
        masterUser: {
          username: 'admin',
          password: cdk.SecretValue.unsafePlainText('tooshort'),
        },
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
        vpc,
        engineVersion,
      });
    }).toThrow(`Invalid engine version: '${engineVersion}'. Engine version must be in the format x.y.z`);
  });

  describe('storage type', () => {
    test('specify storage type', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN
      new DatabaseCluster(stack, 'Database', {
        instances: 1,
        masterUser: {
          username: 'admin',
          password: cdk.SecretValue.unsafePlainText('tooshort'),
        },
        instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
        vpc,
        storageType: StorageType.IOPT1,
        engineVersion: '5.0.0',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
        StorageType: 'iopt1',
      });
    });

    test('throw error for invalid engine version with I/O optimized storage type', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // THEN
      expect(() => {
        new DatabaseCluster(stack, 'Database', {
          instances: 1,
          masterUser: {
            username: 'admin',
            password: cdk.SecretValue.unsafePlainText('tooshort'),
          },
          instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.SMALL),
          vpc,
          storageType: StorageType.IOPT1,
          engineVersion: '3.6.0',
        });
      }).toThrow("I/O-optimized storage is supported starting with engine version 5.0.0, got '3.6.0'");
    });
  });

  describe('serverless clusters', () => {
    test('can create a serverless cluster', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN
      new DatabaseCluster(stack, 'Database', {
        masterUser: {
          username: 'admin',
          password: cdk.SecretValue.unsafePlainText('tooshort'),
        },
        vpc,
        serverlessV2ScalingConfiguration: {
          minCapacity: 0.5,
          maxCapacity: 1,
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
        ServerlessV2ScalingConfiguration: {
          MinCapacity: 0.5,
          MaxCapacity: 1,
        },
      });
      // Should not create any instances
      Template.fromStack(stack).resourceCountIs('AWS::DocDB::DBInstance', 0);
    });

    test('serverless cluster has empty instance arrays', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN
      const cluster = new DatabaseCluster(stack, 'Database', {
        masterUser: {
          username: 'admin',
        },
        vpc,
        serverlessV2ScalingConfiguration: {
          minCapacity: 0.5,
          maxCapacity: 2,
        },
      });

      // THEN
      expect(cluster.instanceIdentifiers).toEqual([]);
      expect(cluster.instanceEndpoints).toEqual([]);
    });

    test('cannot specify instanceType with serverless configuration', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN/THEN
      expect(() => {
        new DatabaseCluster(stack, 'Database', {
          masterUser: {
            username: 'admin',
          },
          vpc,
          instanceType: ec2.InstanceType.of(ec2.InstanceClass.R5, ec2.InstanceSize.LARGE),
          serverlessV2ScalingConfiguration: {
            minCapacity: 0.5,
            maxCapacity: 1,
          },
        });
      }).toThrow('Cannot specify both instanceType and serverlessV2ScalingConfiguration');
    });

    test('provisioned cluster requires instanceType', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN/THEN
      expect(() => {
        new DatabaseCluster(stack, 'Database', {
          masterUser: {
            username: 'admin',
          },
          vpc,
        });
      }).toThrow('Either instanceType (for provisioned clusters) or serverlessV2ScalingConfiguration (for serverless clusters) must be specified');
    });

    test('serverless cluster with all configuration options', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN
      new DatabaseCluster(stack, 'Database', {
        masterUser: {
          username: 'admin',
        },
        vpc,
        serverlessV2ScalingConfiguration: {
          minCapacity: 1,
          maxCapacity: 4,
        },
        engineVersion: '5.0.0',
        deletionProtection: true,
        exportAuditLogsToCloudWatch: true,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
        ServerlessV2ScalingConfiguration: {
          MinCapacity: 1,
          MaxCapacity: 4,
        },
        EngineVersion: '5.0.0',
        DeletionProtection: true,
        EnableCloudwatchLogsExports: ['audit'],
      });
    });

    test('serverless cluster requires engine version 5.0.0 or higher', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN/THEN
      expect(() => {
        new DatabaseCluster(stack, 'Database', {
          masterUser: {
            username: 'admin',
          },
          vpc,
          serverlessV2ScalingConfiguration: {
            minCapacity: 0.5,
            maxCapacity: 1,
          },
          engineVersion: '4.0.0',
        });
      }).toThrow("DocumentDB serverless requires engine version 5.0.0 or higher, got '4.0.0'");
    });

    test('serverless cluster allows engine version 5.0.0', () => {
      // GIVEN
      const stack = testStack();
      const vpc = new ec2.Vpc(stack, 'VPC');

      // WHEN
      new DatabaseCluster(stack, 'Database', {
        masterUser: {
          username: 'admin',
        },
        vpc,
        serverlessV2ScalingConfiguration: {
          minCapacity: 0.5,
          maxCapacity: 1,
        },
        engineVersion: '5.0.0',
      });

      // THEN - should not throw
      Template.fromStack(stack).hasResourceProperties('AWS::DocDB::DBCluster', {
        EngineVersion: '5.0.0',
        ServerlessV2ScalingConfiguration: {
          MinCapacity: 0.5,
          MaxCapacity: 1,
        },
      });
    });
  });
});

function testStack() {
  const stack = new cdk.Stack(undefined, undefined, { env: { account: '12345', region: 'us-test-1' } });
  stack.node.setContext('availability-zones:12345:us-test-1', ['us-test-1a', 'us-test-1b']);
  return stack;
}
