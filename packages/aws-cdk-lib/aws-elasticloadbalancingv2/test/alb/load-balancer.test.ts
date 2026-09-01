import type { Construct } from 'constructs';
import { Match, Template } from '../../../assertions';
import type { Metric } from '../../../aws-cloudwatch';
import * as ec2 from '../../../aws-ec2';
import { Key } from '../../../aws-kms';
import * as s3 from '../../../aws-s3';
import * as cdk from '../../../core';
import * as elbv2 from '../../lib';

describe('tests', () => {
  test('specify minimum capacity unit', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      minimumCapacityUnit: 1500,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      MinimumLoadBalancerCapacity: {
        CapacityUnits: 1500,
      },
    });
  });

  test.each([-1, 99, 100.5])('throw error for invalid range minimum capacity unit', (minimumCapacityUnit) => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'VPC');

    // THEN
    expect(() => {
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        minimumCapacityUnit,
      });
    }).toThrow(`'minimumCapacityUnit' must be a positive integer greater than or equal to 100 for Application Load Balancer, got: ${minimumCapacityUnit}.`);
  });

  test('Trivial construction: internet facing', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internet-facing',
      Subnets: [
        { Ref: 'StackPublicSubnet1Subnet0AD81D22' },
        { Ref: 'StackPublicSubnet2Subnet3C7D2288' },
      ],
      Type: 'application',
    });
  });

  test('internet facing load balancer has dependency on IGW', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: true,
    });

    // THEN
    Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      DependsOn: [
        'StackPublicSubnet1DefaultRoute16154E3D',
        'StackPublicSubnet1RouteTableAssociation74F1C1B6',
        'StackPublicSubnet2DefaultRoute0319539B',
        'StackPublicSubnet2RouteTableAssociation5E8F73F1',
      ],
    });
  });

  test('Trivial construction: internal', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', { vpc });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Scheme: 'internal',
      Subnets: [
        { Ref: 'StackPrivateSubnet1Subnet47AC2BC7' },
        { Ref: 'StackPrivateSubnet2SubnetA2F8EDD8' },
      ],
      Type: 'application',
    });
  });

  test('Attributes', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      deletionProtection: true,
      http2Enabled: false,
      idleTimeout: cdk.Duration.seconds(1000),
      dropInvalidHeaderFields: true,
      clientKeepAlive: cdk.Duration.seconds(200),
      preserveHostHeader: true,
      xAmznTlsVersionAndCipherSuiteHeaders: true,
      preserveXffClientPort: true,
      xffHeaderProcessingMode: elbv2.XffHeaderProcessingMode.PRESERVE,
      wafFailOpen: true,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: [
        {
          Key: 'deletion_protection.enabled',
          Value: 'true',
        },
        {
          Key: 'routing.http2.enabled',
          Value: 'false',
        },
        {
          Key: 'idle_timeout.timeout_seconds',
          Value: '1000',
        },
        {
          Key: 'routing.http.drop_invalid_header_fields.enabled',
          Value: 'true',
        },
        {
          Key: 'routing.http.preserve_host_header.enabled',
          Value: 'true',
        },
        {
          Key: 'routing.http.x_amzn_tls_version_and_cipher_suite.enabled',
          Value: 'true',
        },
        {
          Key: 'routing.http.xff_client_port.enabled',
          Value: 'true',
        },
        {
          Key: 'routing.http.xff_header_processing.mode',
          Value: 'preserve',
        },
        {
          Key: 'waf.fail_open.enabled',
          Value: 'true',
        },
        {
          Key: 'client_keep_alive.seconds',
          Value: '200',
        },
      ],
    });
  });

  test.each([59, 604801])('throw error for invalid clientKeepAlive in seconds', (duration) => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // THEN
    expect(() => {
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        clientKeepAlive: cdk.Duration.seconds(duration),
      });
    }).toThrow(`\'clientKeepAlive\' must be between 60 and 604800 seconds. Got: ${duration} seconds`);
  });

  test('throw errer for invalid clientKeepAlive in milliseconds', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // THEN
    expect(() => {
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        clientKeepAlive: cdk.Duration.millis(100),
      });
    }).toThrow('\'clientKeepAlive\' must be between 60 and 604800 seconds. Got: 100 milliseconds');
  });

  test.each([
    [false, undefined],
    [true, undefined],
    [false, elbv2.IpAddressType.IPV4],
    [true, elbv2.IpAddressType.IPV4],
  ])('throw error for denyAllIgwTraffic set to %s for Ipv4 (default) addressing.', (denyAllIgwTraffic, ipAddressType) => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // THEN
    expect(() => {
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        denyAllIgwTraffic: denyAllIgwTraffic,
        ipAddressType: ipAddressType,
      });
    }).toThrow(`'denyAllIgwTraffic' may only be set on load balancers with ${elbv2.IpAddressType.DUAL_STACK} addressing.`);
  });

  describe('Desync mitigation mode', () => {
    test('Defensive', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        desyncMitigationMode: elbv2.DesyncMitigationMode.DEFENSIVE,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'routing.http.desync_mitigation_mode',
            Value: 'defensive',
          },
        ],
      });
    });
    test('Monitor', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        desyncMitigationMode: elbv2.DesyncMitigationMode.MONITOR,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'routing.http.desync_mitigation_mode',
            Value: 'monitor',
          },
        ],
      });
    });
    test('Strictest', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        desyncMitigationMode: elbv2.DesyncMitigationMode.STRICTEST,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'routing.http.desync_mitigation_mode',
            Value: 'strictest',
          },
        ],
      });
    });
  });

  describe('http2Enabled', () => {
    test('http2Enabled is not set', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.not(
          Match.arrayWith([
            {
              Key: 'routing.http2.enabled',
              Value: Match.anyValue(),
            },
          ]),
        ),
      });
    });

    test.each([true, false])('http2Enabled is set to %s', (http2Enabled) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');
      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        http2Enabled,
      });
      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'routing.http2.enabled',
            Value: String(http2Enabled),
          },
        ]),
      });
    });
  });

  test('Deletion protection false', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      deletionProtection: false,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      LoadBalancerAttributes: Match.arrayWith([
        {
          Key: 'deletion_protection.enabled',
          Value: 'false',
        },
      ]),
    });
  });

  test('Can add and list listeners for an owned ApplicationLoadBalancer', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      internetFacing: true,
    });

    const listener = loadBalancer.addListener('listener', {
      protocol: elbv2.ApplicationProtocol.HTTP,
      defaultAction: elbv2.ListenerAction.fixedResponse(200),
    });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::Listener', 1);
    expect(loadBalancer.listeners).toContain(listener);
  });

  describe('logAccessLogs', () => {
    class ExtendedLB extends elbv2.ApplicationLoadBalancer {
      constructor(scope: Construct, id: string, vpc: ec2.IVpc) {
        super(scope, id, { vpc });

        const accessLogsBucket = new s3.Bucket(this, 'ALBAccessLogsBucket', {
          blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
          encryption: s3.BucketEncryption.S3_MANAGED,
          versioned: true,
          serverAccessLogsPrefix: 'selflog/',
          enforceSSL: true,
        });

        this.logAccessLogs(accessLogsBucket);
      }
    }

    function loggingSetup(withEncryption: boolean = false): { stack: cdk.Stack; bucket: s3.Bucket; lb: elbv2.ApplicationLoadBalancer } {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, undefined, { env: { region: 'us-east-1' } });
      const vpc = new ec2.Vpc(stack, 'Stack');
      let bucketProps = {};
      if (withEncryption) {
        const kmsKey = new Key(stack, 'TestKMSKey');
        bucketProps = { ...bucketProps, encryption: s3.BucketEncryption.KMS, encyptionKey: kmsKey };
      }
      const bucket = new s3.Bucket(stack, 'AccessLogBucket', { ...bucketProps });
      const lb = new elbv2.ApplicationLoadBalancer(stack, 'LB', { vpc });
      return { stack, bucket, lb };
    }

    test('sets load balancer attributes', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logAccessLogs(bucket);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'access_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'access_logs.s3.bucket',
            Value: { Ref: 'AccessLogBucketDA470295' },
          },
          {
            Key: 'access_logs.s3.prefix',
            Value: '',
          },
        ]),
      });
    });

    test('adds a dependency on the bucket', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logAccessLogs(bucket);

      // THEN
      // verify the ALB depends on the bucket policy
      Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        DependsOn: ['AccessLogBucketPolicyF52D2D01'],
      });
    });

    test('logging bucket permissions', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logAccessLogs(bucket);

      // THEN
      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'] }, '/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'] }, '/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'],
              },
            },
          ],
        },
      });
    });

    test('access logging with prefix', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logAccessLogs(bucket, 'prefix-of-access-logs');

      // THEN
      // verify that the LB attributes reference the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'access_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'access_logs.s3.bucket',
            Value: { Ref: 'AccessLogBucketDA470295' },
          },
          {
            Key: 'access_logs.s3.prefix',
            Value: 'prefix-of-access-logs',
          },
        ]),
      });

      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'] }, '/prefix-of-access-logs/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'] }, '/prefix-of-access-logs/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::GetAtt': ['AccessLogBucketDA470295', 'Arn'],
              },
            },
          ],
        },
      });
    });

    test('bucket with KMS throws validation error', () => {
      // GIVEN
      const { bucket, lb } = loggingSetup(true);

      // WHEN
      const logAccessLogFunctionTest = () => lb.logAccessLogs(bucket);

      // THEN
      // verify failure in case the access log bucket is encrypted with KMS
      expect(logAccessLogFunctionTest).toThrow('Encryption key detected. Bucket encryption using KMS keys is unsupported');
    });

    test('access logging on imported bucket', () => {
      // GIVEN
      const { stack, lb } = loggingSetup();

      const bucket = s3.Bucket.fromBucketName(stack, 'ImportedAccessLoggingBucket', 'imported-bucket');
      // Imported buckets have `autoCreatePolicy` disabled by default
      bucket.policy = new s3.BucketPolicy(stack, 'ImportedAccessLoggingBucketPolicy', {
        bucket,
      });

      // WHEN
      lb.logAccessLogs(bucket);

      // THEN
      // verify that the LB attributes reference the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'access_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'access_logs.s3.bucket',
            Value: 'imported-bucket',
          },
          {
            Key: 'access_logs.s3.prefix',
            Value: '',
          },
        ]),
      });

      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket/AWSLogs/',
                    { Ref: 'AWS::AccountId' },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket/AWSLogs/',
                    { Ref: 'AWS::AccountId' },
                    '/*',
                  ],
                ],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket',
                  ],
                ],
              },
            },
          ],
        },
      });

      // verify the ALB depends on the bucket policy
      Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        DependsOn: ['ImportedAccessLoggingBucketPolicy97AE3371'],
      });
    });

    test('does not add circular dependency on bucket with extended load balancer', () => {
      // GIVEN
      const { stack } = loggingSetup();
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new ExtendedLB(stack, 'ExtendedLB', vpc);

      // THEN
      Template.fromStack(stack).hasResource('AWS::S3::Bucket', {
        Type: 'AWS::S3::Bucket',
        Properties: {
          AccessControl: 'LogDeliveryWrite',
          BucketEncryption: {
            ServerSideEncryptionConfiguration: [
              {
                ServerSideEncryptionByDefault: {
                  SSEAlgorithm: 'AES256',
                },
              },
            ],
          },
          LoggingConfiguration: {
            LogFilePrefix: 'selflog/',
          },
          OwnershipControls: {
            Rules: [
              {
                ObjectOwnership: 'ObjectWriter',
              },
            ],
          },
          PublicAccessBlockConfiguration: {
            BlockPublicAcls: true,
            BlockPublicPolicy: true,
            IgnorePublicAcls: true,
            RestrictPublicBuckets: true,
          },
          VersioningConfiguration: {
            Status: 'Enabled',
          },
        },
        UpdateReplacePolicy: 'Retain',
        DeletionPolicy: 'Retain',
      });
    });
  });

  describe('logConnectionLogs', () => {
    class ExtendedLB extends elbv2.ApplicationLoadBalancer {
      constructor(scope: Construct, id: string, vpc: ec2.IVpc) {
        super(scope, id, { vpc });

        const connectionLogsBucket = new s3.Bucket(this, 'ALBConnectionLogsBucket', {
          blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
          encryption: s3.BucketEncryption.S3_MANAGED,
          versioned: true,
          serverAccessLogsPrefix: 'selflog/',
          enforceSSL: true,
        });

        this.logConnectionLogs(connectionLogsBucket);
      }
    }

    function loggingSetup(withEncryption: boolean = false ): { stack: cdk.Stack; bucket: s3.Bucket; lb: elbv2.ApplicationLoadBalancer } {
      const app = new cdk.App();
      const stack = new cdk.Stack(app, undefined, { env: { region: 'us-east-1' } });
      const vpc = new ec2.Vpc(stack, 'Stack');
      let bucketProps = {};
      if (withEncryption) {
        const kmsKey = new Key(stack, 'TestKMSKey');
        bucketProps = { ...bucketProps, encryption: s3.BucketEncryption.KMS, encyptionKey: kmsKey };
      }
      const bucket = new s3.Bucket(stack, 'ConnectionLogBucket', { ...bucketProps });
      const lb = new elbv2.ApplicationLoadBalancer(stack, 'LB', { vpc });
      return { stack, bucket, lb };
    }

    test('sets load balancer attributes', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logConnectionLogs(bucket);

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'connection_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'connection_logs.s3.bucket',
            Value: { Ref: 'ConnectionLogBucketFDE8490A' },
          },
          {
            Key: 'connection_logs.s3.prefix',
            Value: '',
          },
        ]),
      });
    });

    test('adds a dependency on the bucket', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logConnectionLogs(bucket);

      // THEN
      // verify the ALB depends on the bucket policy
      Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        DependsOn: ['ConnectionLogBucketPolicyF17C8635'],
      });
    });

    test('logging bucket permissions', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logConnectionLogs(bucket);

      // THEN
      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'] }, '/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'] }, '/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'],
              },
            },
          ],
        },
      });
    });

    test('connection logging with prefix', () => {
      // GIVEN
      const { stack, bucket, lb } = loggingSetup();

      // WHEN
      lb.logConnectionLogs(bucket, 'prefix-of-connection-logs');

      // THEN
      // verify that the LB attributes reference the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'connection_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'connection_logs.s3.bucket',
            Value: { Ref: 'ConnectionLogBucketFDE8490A' },
          },
          {
            Key: 'connection_logs.s3.prefix',
            Value: 'prefix-of-connection-logs',
          },
        ]),
      });

      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'] }, '/prefix-of-connection-logs/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': ['', [{ 'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'] }, '/prefix-of-connection-logs/AWSLogs/',
                  { Ref: 'AWS::AccountId' }, '/*']],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::GetAtt': ['ConnectionLogBucketFDE8490A', 'Arn'],
              },
            },
          ],
        },
      });
    });

    test('bucket with KMS throws validation error', () => {
      // GIVEN
      const { bucket, lb } = loggingSetup(true);

      // WHEN
      const logConnectionLogFunctionTest = () => lb.logConnectionLogs(bucket);

      // THEN
      // verify failure in case the connection log bucket is encrypted with KMS
      expect(logConnectionLogFunctionTest).toThrow('Encryption key detected. Bucket encryption using KMS keys is unsupported');
    });

    test('connection logging on imported bucket', () => {
      // GIVEN
      const { stack, lb } = loggingSetup();

      const bucket = s3.Bucket.fromBucketName(stack, 'ImportedConnectionLoggingBucket', 'imported-bucket');
      // Imported buckets have `autoCreatePolicy` disabled by default
      bucket.policy = new s3.BucketPolicy(stack, 'ImportedConnectionLoggingBucketPolicy', {
        bucket,
      });

      // WHEN
      lb.logConnectionLogs(bucket);

      // THEN
      // verify that the LB attributes reference the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'connection_logs.s3.enabled',
            Value: 'true',
          },
          {
            Key: 'connection_logs.s3.bucket',
            Value: 'imported-bucket',
          },
          {
            Key: 'connection_logs.s3.prefix',
            Value: '',
          },
        ]),
      });

      // verify the bucket policy allows the ALB to put objects in the bucket
      Template.fromStack(stack).hasResourceProperties('AWS::S3::BucketPolicy', {
        PolicyDocument: {
          Version: '2012-10-17',
          Statement: [
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { AWS: { 'Fn::Join': ['', ['arn:', { Ref: 'AWS::Partition' }, ':iam::127311923021:root']] } },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket/AWSLogs/',
                    { Ref: 'AWS::AccountId' },
                    '/*',
                  ],
                ],
              },
            },
            {
              Action: 's3:PutObject',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket/AWSLogs/',
                    { Ref: 'AWS::AccountId' },
                    '/*',
                  ],
                ],
              },
              Condition: { StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' } },
            },
            {
              Action: 's3:GetBucketAcl',
              Effect: 'Allow',
              Principal: { Service: 'delivery.logs.amazonaws.com' },
              Resource: {
                'Fn::Join': [
                  '',
                  [
                    'arn:',
                    { Ref: 'AWS::Partition' },
                    ':s3:::imported-bucket',
                  ],
                ],
              },
            },
          ],
        },
      });

      // verify the ALB depends on the bucket policy
      Template.fromStack(stack).hasResource('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        DependsOn: ['ImportedConnectionLoggingBucketPolicy548EEC12'],
      });
    });

    test('does not add circular dependency on bucket with extended load balancer', () => {
      // GIVEN
      const { stack } = loggingSetup();
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new ExtendedLB(stack, 'ExtendedLB', vpc);

      // THEN
      Template.fromStack(stack).hasResource('AWS::S3::Bucket', {
        Type: 'AWS::S3::Bucket',
        Properties: {
          AccessControl: 'LogDeliveryWrite',
          BucketEncryption: {
            ServerSideEncryptionConfiguration: [
              {
                ServerSideEncryptionByDefault: {
                  SSEAlgorithm: 'AES256',
                },
              },
            ],
          },
          LoggingConfiguration: {
            LogFilePrefix: 'selflog/',
          },
          OwnershipControls: {
            Rules: [
              {
                ObjectOwnership: 'ObjectWriter',
              },
            ],
          },
          PublicAccessBlockConfiguration: {
            BlockPublicAcls: true,
            BlockPublicPolicy: true,
            IgnorePublicAcls: true,
            RestrictPublicBuckets: true,
          },
          VersioningConfiguration: {
            Status: 'Enabled',
          },
        },
        UpdateReplacePolicy: 'Retain',
        DeletionPolicy: 'Retain',
        DependsOn: Match.absent(),
      });
    });
  });

  test('Exercise metrics', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');
    const lb = new elbv2.ApplicationLoadBalancer(stack, 'LB', { vpc });

    // WHEN
    const metrics = new Array<Metric>();
    metrics.push(lb.metrics.activeConnectionCount());
    metrics.push(lb.metrics.clientTlsNegotiationErrorCount());
    metrics.push(lb.metrics.consumedLCUs());
    metrics.push(lb.metrics.elbAuthError());
    metrics.push(lb.metrics.elbAuthFailure());
    metrics.push(lb.metrics.elbAuthLatency());
    metrics.push(lb.metrics.elbAuthSuccess());
    metrics.push(lb.metrics.httpCodeElb(elbv2.HttpCodeElb.ELB_3XX_COUNT));
    metrics.push(lb.metrics.httpCodeTarget(elbv2.HttpCodeTarget.TARGET_3XX_COUNT));
    metrics.push(lb.metrics.httpFixedResponseCount());
    metrics.push(lb.metrics.httpRedirectCount());
    metrics.push(lb.metrics.httpRedirectUrlLimitExceededCount());
    metrics.push(lb.metrics.ipv6ProcessedBytes());
    metrics.push(lb.metrics.ipv6RequestCount());
    metrics.push(lb.metrics.newConnectionCount());
    metrics.push(lb.metrics.processedBytes());
    metrics.push(lb.metrics.rejectedConnectionCount());
    metrics.push(lb.metrics.requestCount());
    metrics.push(lb.metrics.ruleEvaluations());
    metrics.push(lb.metrics.targetConnectionErrorCount());
    metrics.push(lb.metrics.targetResponseTime());
    metrics.push(lb.metrics.targetTLSNegotiationErrorCount());

    for (const metric of metrics) {
      expect(metric.namespace).toEqual('AWS/ApplicationELB');
      expect(stack.resolve(metric.dimensions)).toEqual({
        LoadBalancer: { 'Fn::GetAtt': ['LB8A12904C', 'LoadBalancerFullName'] },
      });
    }
  });

  test.each([
    elbv2.HttpCodeElb.ELB_500_COUNT,
    elbv2.HttpCodeElb.ELB_502_COUNT,
    elbv2.HttpCodeElb.ELB_503_COUNT,
    elbv2.HttpCodeElb.ELB_504_COUNT,
  ])('use specific load balancer generated 5XX metrics', (metricName) => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');
    const lb = new elbv2.ApplicationLoadBalancer(stack, 'LB', { vpc });

    // WHEN
    const metric = lb.metrics.httpCodeElb(metricName);

    // THEN
    expect(metric.namespace).toEqual('AWS/ApplicationELB');
    expect(metric.statistic).toEqual('Sum');
    expect(metric.metricName).toEqual(metricName);
    expect(stack.resolve(metric.dimensions)).toEqual({
      LoadBalancer: { 'Fn::GetAtt': ['LB8A12904C', 'LoadBalancerFullName'] },
    });
  });

  test('loadBalancerName', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    // WHEN
    new elbv2.ApplicationLoadBalancer(stack, 'ALB', {
      loadBalancerName: 'myLoadBalancer',
      vpc,
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      Name: 'myLoadBalancer',
    });
  });

  test('imported load balancer with no vpc throws error when calling addTargets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const sg = new ec2.SecurityGroup(stack, 'sg', {
      vpc,
      securityGroupName: 'mySg',
    });
    const alb = elbv2.ApplicationLoadBalancer.fromApplicationLoadBalancerAttributes(stack, 'ALB', {
      loadBalancerArn: albArn,
      securityGroupId: sg.securityGroupId,
    });

    // WHEN
    const listener = alb.addListener('Listener', { port: 80 });
    expect(() => listener.addTargets('Targets', { port: 8080 })).toThrow();
  });

  test('imported load balancer with vpc does not throw error when calling addTargets', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const sg = new ec2.SecurityGroup(stack, 'sg', {
      vpc,
      securityGroupName: 'mySg',
    });
    const alb = elbv2.ApplicationLoadBalancer.fromApplicationLoadBalancerAttributes(stack, 'ALB', {
      loadBalancerArn: albArn,
      securityGroupId: sg.securityGroupId,
      vpc,
    });

    // WHEN
    const listener = alb.addListener('Listener', { port: 80 });
    expect(() => listener.addTargets('Targets', { port: 8080 })).not.toThrow();
  });

  test('imported load balancer with vpc can add but not list listeners', () => {
    // GIVEN
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Vpc');
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const sg = new ec2.SecurityGroup(stack, 'sg', {
      vpc,
      securityGroupName: 'mySg',
    });
    const alb = elbv2.ApplicationLoadBalancer.fromApplicationLoadBalancerAttributes(stack, 'ALB', {
      loadBalancerArn: albArn,
      securityGroupId: sg.securityGroupId,
      vpc,
    });

    // WHEN
    const listener = alb.addListener('Listener', { port: 80 });
    listener.addTargets('Targets', { port: 8080 });

    // THEN
    Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::Listener', 1);
    expect(() => alb.listeners).toThrow();
  });

  test('imported load balancer knows its region', () => {
    const stack = new cdk.Stack();

    // WHEN
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const alb = elbv2.ApplicationLoadBalancer.fromApplicationLoadBalancerAttributes(stack, 'ALB', {
      loadBalancerArn: albArn,
      securityGroupId: 'sg-1234',
    });

    // THEN
    expect(alb.env.region).toEqual('us-west-2');
  });

  test('imported load balancer can produce metrics', () => {
    const stack = new cdk.Stack();

    // WHEN
    const albArn = 'arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-load-balancer/50dc6c495c0c9188';
    const alb = elbv2.ApplicationLoadBalancer.fromApplicationLoadBalancerAttributes(stack, 'ALB', {
      loadBalancerArn: albArn,
      securityGroupId: 'sg-1234',
    });

    // THEN
    const metric = alb.metrics.activeConnectionCount();
    expect(metric.namespace).toEqual('AWS/ApplicationELB');
    expect(stack.resolve(metric.dimensions)).toEqual({
      LoadBalancer: 'app/my-load-balancer/50dc6c495c0c9188',
    });
    expect(alb.env.region).toEqual('us-west-2');
  });

  test('can add secondary security groups', () => {
    const stack = new cdk.Stack();
    const vpc = new ec2.Vpc(stack, 'Stack');

    const alb = new elbv2.ApplicationLoadBalancer(stack, 'LB', {
      vpc,
      securityGroup: new ec2.SecurityGroup(stack, 'SecurityGroup1', { vpc }),
    });
    alb.addSecurityGroup(new ec2.SecurityGroup(stack, 'SecurityGroup2', { vpc }));

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
      SecurityGroups: [
        { 'Fn::GetAtt': ['SecurityGroup1F554B36F', 'GroupId'] },
        { 'Fn::GetAtt': ['SecurityGroup23BE86BB7', 'GroupId'] },
      ],
      Type: 'application',
    });
  });

  // test cases for crossZoneEnabled
  describe('crossZoneEnabled', () => {
    test('crossZoneEnabled can be true', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'alb', {
        vpc,
        crossZoneEnabled: true,
      });
      const t = Template.fromStack(stack);
      t.resourceCountIs('AWS::ElasticLoadBalancingV2::LoadBalancer', 1);
      t.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
          {
            Key: 'load_balancing.cross_zone.enabled',
            Value: 'true',
          },
        ],
      });
    });
    test('crossZoneEnabled can be undefined', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'alb', {
        vpc,
      });
      const t = Template.fromStack(stack);
      t.resourceCountIs('AWS::ElasticLoadBalancingV2::LoadBalancer', 1);
      t.hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: [
          {
            Key: 'deletion_protection.enabled',
            Value: 'false',
          },
        ],
      });
    });
    test('crossZoneEnabled cannot be false', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack');
      const vpc = new ec2.Vpc(stack, 'Vpc');

      // expect the error
      expect(() => {
        new elbv2.ApplicationLoadBalancer(stack, 'alb', {
          vpc,
          crossZoneEnabled: false,
        });
      }).toThrow('crossZoneEnabled cannot be false with Application Load Balancers.');
    });
  });

  describe('lookup', () => {
    test('Can look up an ApplicationLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      // WHEN
      const loadBalancer = elbv2.ApplicationLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::ApplicationLoadBalancer', 0);
      expect(loadBalancer.loadBalancerArn).toEqual('arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/application/my-load-balancer/50dc6c495c0c9188');
      expect(loadBalancer.loadBalancerCanonicalHostedZoneId).toEqual('Z3DZXE0EXAMPLE');
      expect(loadBalancer.loadBalancerDnsName).toEqual('my-load-balancer-123456789012.us-west-2.elb.amazonaws.com');
      expect(loadBalancer.ipAddressType).toEqual(elbv2.IpAddressType.DUAL_STACK);
      expect(loadBalancer.connections.securityGroups[0].securityGroupId).toEqual('sg-12345678');
      expect(loadBalancer.env.region).toEqual('us-west-2');
    });

    test('Can add but not list listeners for a looked-up ApplicationLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      const loadBalancer = elbv2.ApplicationLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      // WHEN
      loadBalancer.addListener('listener', {
        protocol: elbv2.ApplicationProtocol.HTTP,
        defaultAction: elbv2.ListenerAction.fixedResponse(200),
      });

      // THEN
      Template.fromStack(stack).resourceCountIs('AWS::ElasticLoadBalancingV2::Listener', 1);
      expect(() => loadBalancer.listeners).toThrow();
    });

    test('Can create metrics for a looked-up ApplicationLoadBalancer', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'stack', {
        env: {
          account: '123456789012',
          region: 'us-west-2',
        },
      });

      const loadBalancer = elbv2.ApplicationLoadBalancer.fromLookup(stack, 'a', {
        loadBalancerTags: {
          some: 'tag',
        },
      });

      // WHEN
      const metric = loadBalancer.metrics.activeConnectionCount();

      // THEN
      expect(metric.namespace).toEqual('AWS/ApplicationELB');
      expect(stack.resolve(metric.dimensions)).toEqual({
        LoadBalancer: 'application/my-load-balancer/50dc6c495c0c9188',
      });
    });
  });

  describe('dualstack', () => {
    test('Can create internet-facing dualstack ApplicationLoadBalancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        internetFacing: true,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internet-facing',
        Type: 'application',
        IpAddressType: 'dualstack',
      });
    });

    test('Can create internet-facing dualstack ApplicationLoadBalancer with denyAllIgwTraffic set to false', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        denyAllIgwTraffic: false,
        internetFacing: true,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internet-facing',
        Type: 'application',
        IpAddressType: 'dualstack',
      });
    });

    test('Can create internal dualstack ApplicationLoadBalancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internal',
        Type: 'application',
        IpAddressType: 'dualstack',
      });
    });

    test.each([undefined, false])('Can create internal dualstack ApplicationLoadBalancer with denyAllIgwTraffic set to true', (internetFacing) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        denyAllIgwTraffic: true,
        internetFacing: internetFacing,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internal',
        Type: 'application',
        IpAddressType: 'dualstack',
      });
    });
  });

  describe('dualstack without public ipv4', () => {
    test('Can create internet-facing dualstack without public ipv4 ApplicationLoadBalancer', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        internetFacing: true,
        ipAddressType: elbv2.IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        Scheme: 'internet-facing',
        Type: 'application',
        IpAddressType: 'dualstack-without-public-ipv4',
      });
    });

    test('Cannot create internal dualstack without public ipv4 ApplicationLoadBalancer', () => {
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      expect(() => {
        new elbv2.ApplicationLoadBalancer(stack, 'LB', {
          vpc,
          internetFacing: false,
          ipAddressType: elbv2.IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4,
        });
      }).toThrow('dual-stack without public IPv4 address can only be used with internet-facing scheme.');
    });
  });

  describe('Drop Invalid Header Fields', () => {
    test.each([true, false])('sets dropInvalidHeaderFields to %s', (dropInvalidHeaderFields) => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
        dropInvalidHeaderFields,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.arrayWith([
          {
            Key: 'routing.http.drop_invalid_header_fields.enabled',
            Value: String(dropInvalidHeaderFields),
          },
        ]),
      });
    });

    test('dropInvalidHeaderFields is not set when undefined', () => {
      // GIVEN
      const stack = new cdk.Stack();
      const vpc = new ec2.Vpc(stack, 'Stack');

      // WHEN
      new elbv2.ApplicationLoadBalancer(stack, 'LB', {
        vpc,
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties('AWS::ElasticLoadBalancingV2::LoadBalancer', {
        LoadBalancerAttributes: Match.not(
          Match.arrayWith([
            {
              Key: 'routing.http.drop_invalid_header_fields.enabled',
              Value: Match.anyValue(),
            },
          ]),
        ),
      });
    });
  });
});
