import { Template } from '../../assertions';
import * as cloudfront from '../../aws-cloudfront';
import * as ec2 from '../../aws-ec2';
import * as elbv2 from '../../aws-elasticloadbalancingv2';
import { App, Duration, Stack } from '../../core';
import { VpcOrigin } from '../lib';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'Stack', {
    env: { account: '1234', region: 'testregion' },
  });
});

test('VPC origin from an EC2 instance', () => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const instance = new ec2.Instance(stack, 'Instance', {
    vpc,
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.MICRO),
    machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  });

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withEc2Instance(instance),
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':ec2:testregion:1234:instance/',
            { Ref: 'InstanceC1063A87' },
          ],
        ],
      },
      Name: 'StackDistributionOrigin1VpcOriginB6F753F8',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: { 'Fn::GetAtt': ['InstanceC1063A87', 'PrivateDnsName'] },
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin1VpcOrigin1389D846', 'Id'] },
          },
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
    },
  });
});

test('VPC origin from an Application Load Balancer', () => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withApplicationLoadBalancer(loadBalancer),
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: { Ref: 'ALBAEE750D2' },
      Name: 'StackDistributionOrigin1VpcOriginB6F753F8',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: { 'Fn::GetAtt': ['ALBAEE750D2', 'DNSName'] },
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin1VpcOrigin1389D846', 'Id'] },
          },
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
    },
  });
});

test('VPC origin from a Network Load Balancer', () => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.NetworkLoadBalancer(stack, 'NLB', { vpc });

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withNetworkLoadBalancer(loadBalancer),
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: { Ref: 'NLB55158F82' },
      Name: 'StackDistributionOrigin1VpcOriginB6F753F8',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: { 'Fn::GetAtt': ['NLB55158F82', 'DNSName'] },
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin1VpcOrigin1389D846', 'Id'] },
          },
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
    },
  });
});

test('VPC origin from a VpcOrigin resource', () => {
  // GIVEN
  const vpcOrigin = new cloudfront.VpcOrigin(stack, 'VpcOrigin', {
    endpoint: { endpointArn: 'arn:opaque', domainName: 'vpcorigin.example.com' },
  });

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withVpcOrigin(vpcOrigin),
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: 'vpcorigin.example.com',
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['VpcOrigin65BCA67E', 'Id'] },
          },
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
    },
  });
});

test('VPC origin from an imported VpcOrigin resource', () => {
  // GIVEN
  const vpcOrigin = cloudfront.VpcOrigin.fromVpcOriginId(stack, 'VpcOrigin', 'vpc-origin-id');

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withVpcOrigin(vpcOrigin, {
        domainName: 'vpcorigin.example.com',
      }),
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: 'vpcorigin.example.com',
          VpcOriginConfig: {
            VpcOriginId: 'vpc-origin-id',
          },
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
    },
  });
});

test.each([
  Duration.seconds(0),
])('VPC origin throws when readTimeout is %s - below the minimum', (readTimeout) => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });

  // WHEN
  expect(() => {
    VpcOrigin.withApplicationLoadBalancer(loadBalancer, { readTimeout });
  }).toThrow(`readTimeout: Must be an int 1 seconds or greater; received ${readTimeout.toSeconds()}`);
});

test.each([
  Duration.seconds(121),
  Duration.minutes(5),
])('VPC origin accepts readTimeout of %s above the default quota', (readTimeout) => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });

  // WHEN
  expect(() => {
    VpcOrigin.withApplicationLoadBalancer(loadBalancer, { readTimeout });
  }).not.toThrow();
});

test.each([
  Duration.seconds(0),
])('VPC origin throws when keepaliveTimeout is %s - below the minimum', (keepaliveTimeout) => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });

  // WHEN
  expect(() => {
    VpcOrigin.withApplicationLoadBalancer(loadBalancer, { keepaliveTimeout });
  }).toThrow(`keepaliveTimeout: Must be an int 1 seconds or greater; received ${keepaliveTimeout.toSeconds()}`);
});

test.each([
  Duration.seconds(301),
  Duration.minutes(10),
])('VPC origin accepts keepaliveTimeout of %s above the default quota', (keepaliveTimeout) => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const loadBalancer = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });

  // WHEN
  expect(() => {
    VpcOrigin.withApplicationLoadBalancer(loadBalancer, { keepaliveTimeout });
  }).not.toThrow();
});

test('VPC origin throws when no domainName is specified', () => {
  // GIVEN
  const vpcOrigin = new cloudfront.VpcOrigin(stack, 'VpcOrigin', {
    endpoint: { endpointArn: 'arn:opaque' },
  });

  // WHEN
  expect(() => {
    VpcOrigin.withVpcOrigin(vpcOrigin);
  }).toThrow("'domainName' must be specified when no default domain name is defined.");
});

test('VPC origin with options configured', () => {
  // GIVEN
  const vpc = new ec2.Vpc(stack, 'Vpc');
  const instance = new ec2.Instance(stack, 'Instance', {
    vpc,
    instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE4_GRAVITON, ec2.InstanceSize.MICRO),
    machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  });
  const alb = new elbv2.ApplicationLoadBalancer(stack, 'ALB', { vpc });
  const nlb = new elbv2.NetworkLoadBalancer(stack, 'NLB', { vpc });
  const vpcOrigin = new cloudfront.VpcOrigin(stack, 'Opaque', {
    endpoint: { endpointArn: 'arn:opaque' },
  });

  // WHEN
  new cloudfront.Distribution(stack, 'Distribution', {
    defaultBehavior: {
      origin: VpcOrigin.withEc2Instance(instance, {
        domainName: 'ec2instance.example.com',
        vpcOriginName: 'Ec2InstanceOrigin',
        httpPort: 8080,
        httpsPort: 8443,
        protocolPolicy: cloudfront.OriginProtocolPolicy.MATCH_VIEWER,
        originSslProtocols: [cloudfront.OriginSslPolicy.TLS_V1_1, cloudfront.OriginSslPolicy.TLS_V1_2],
        readTimeout: Duration.seconds(60),
        keepaliveTimeout: Duration.seconds(120),
        connectionTimeout: Duration.seconds(5),
        connectionAttempts: 1,
      }),
    },
    additionalBehaviors: {
      '/alb/*': {
        origin: VpcOrigin.withApplicationLoadBalancer(alb, {
          domainName: 'alb.example.com',
          vpcOriginName: 'ALBOrigin',
          httpPort: 8080,
          protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
        }),
      },
      '/nlb/*': {
        origin: VpcOrigin.withNetworkLoadBalancer(nlb, {
          domainName: 'nlb.example.com',
          vpcOriginName: 'NLBOrigin',
          httpsPort: 8443,
          protocolPolicy: cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        }),
      },
      '/opaque/*': {
        origin: VpcOrigin.withVpcOrigin(vpcOrigin, {
          domainName: 'opaque.example.com',
          readTimeout: Duration.seconds(60),
          keepaliveTimeout: Duration.seconds(120),
          connectionTimeout: Duration.seconds(5),
          connectionAttempts: 1,
        }),
      },
    },
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: {
        'Fn::Join': [
          '',
          [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':ec2:testregion:1234:instance/',
            { Ref: 'InstanceC1063A87' },
          ],
        ],
      },
      Name: 'Ec2InstanceOrigin',
      HTTPPort: 8080,
      HTTPSPort: 8443,
      OriginProtocolPolicy: 'match-viewer',
      OriginSSLProtocols: ['TLSv1.1', 'TLSv1.2'],
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: { Ref: 'ALBAEE750D2' },
      Name: 'ALBOrigin',
      HTTPPort: 8080,
      OriginProtocolPolicy: 'http-only',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::VpcOrigin', {
    VpcOriginEndpointConfig: {
      Arn: { Ref: 'NLB55158F82' },
      Name: 'NLBOrigin',
      HTTPSPort: 8443,
      OriginProtocolPolicy: 'https-only',
    },
  });
  Template.fromStack(stack).hasResourceProperties('AWS::CloudFront::Distribution', {
    DistributionConfig: {
      Origins: [
        {
          Id: 'StackDistributionOrigin14F1B0A79',
          DomainName: 'ec2instance.example.com',
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin1VpcOrigin1389D846', 'Id'] },
            OriginReadTimeout: 60,
            OriginKeepaliveTimeout: 120,
          },
          ConnectionTimeout: 5,
          ConnectionAttempts: 1,
        },
        {
          Id: 'StackDistributionOrigin2322BFC29',
          DomainName: 'alb.example.com',
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin2VpcOrigin9CDFA022', 'Id'] },
          },
        },
        {
          Id: 'StackDistributionOrigin354BC8C0F',
          DomainName: 'nlb.example.com',
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['DistributionOrigin3VpcOrigin92647B05', 'Id'] },
          },
        },
        {
          Id: 'StackDistributionOrigin444BA487C',
          DomainName: 'opaque.example.com',
          VpcOriginConfig: {
            VpcOriginId: { 'Fn::GetAtt': ['OpaqueEB82B10F', 'Id'] },
            OriginReadTimeout: 60,
            OriginKeepaliveTimeout: 120,
          },
          ConnectionTimeout: 5,
          ConnectionAttempts: 1,
        },
      ],
      DefaultCacheBehavior: {
        TargetOriginId: 'StackDistributionOrigin14F1B0A79',
      },
      CacheBehaviors: [
        {
          PathPattern: '/alb/*',
          TargetOriginId: 'StackDistributionOrigin2322BFC29',
        },
        {
          PathPattern: '/nlb/*',
          TargetOriginId: 'StackDistributionOrigin354BC8C0F',
        },
        {
          PathPattern: '/opaque/*',
          TargetOriginId: 'StackDistributionOrigin444BA487C',
        },
      ],
    },
  });
});
