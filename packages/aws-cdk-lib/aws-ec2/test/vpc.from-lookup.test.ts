import type { Construct } from 'constructs';
import * as cxschema from '../../cloud-assembly-schema';
import type { GetContextValueOptions, GetContextValueResult } from '../../core';
import { ContextProvider, Lazy, Stack } from '../../core';
import * as cxapi from '../../cx-api';
import { GenericLinuxImage, Instance, InstanceType, SubnetType, Vpc } from '../lib';

describe('vpc from lookup', () => {
  describe('Vpc.fromLookup()', () => {
    test('requires concrete values', () => {
      // GIVEN
      const stack = new Stack();

      expect(() => {
        Vpc.fromLookup(stack, 'Vpc', {
          vpcId: Lazy.string({ produce: () => 'some-id' }),
        });
      }).toThrow('All arguments to Vpc.fromLookup() must be concrete');
    });

    test('selecting subnets by name from a looked-up VPC does not throw', () => {
      // GIVEN
      const stack = new Stack(undefined, undefined, { env: { region: 'us-east-1', account: '123456789012' } });
      const vpc = Vpc.fromLookup(stack, 'VPC', {
        vpcId: 'vpc-1234',
      });

      // WHEN
      vpc.selectSubnets({ subnetName: 'Bleep' });

      // THEN: no exception
    });

    test('accepts asymmetric subnets', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [
          {
            name: 'Public',
            type: cxapi.VpcSubnetGroupType.PUBLIC,
            subnets: [
              {
                subnetId: 'pub-sub-in-us-east-1a',
                availabilityZone: 'us-east-1a',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pub-sub-in-us-east-1b',
                availabilityZone: 'us-east-1b',
                routeTableId: 'rt-123',
              },
            ],
          },
          {
            name: 'Private',
            type: cxapi.VpcSubnetGroupType.PRIVATE,
            subnets: [
              {
                subnetId: 'pri-sub-1-in-us-east-1c',
                availabilityZone: 'us-east-1c',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-2-in-us-east-1c',
                availabilityZone: 'us-east-1c',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-1-in-us-east-1d',
                availabilityZone: 'us-east-1d',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-2-in-us-east-1d',
                availabilityZone: 'us-east-1d',
                routeTableId: 'rt-123',
              },
            ],
          },
        ],
      }, options => {
        expect(options.filter).toEqual({
          isDefault: 'true',
        });

        expect(options.subnetGroupNameTag).toEqual(undefined);
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        isDefault: true,
      });

      expect(vpc.availabilityZones).toEqual(['us-east-1a', 'us-east-1b', 'us-east-1c', 'us-east-1d']);
      expect(vpc.publicSubnets.length).toEqual(2);
      expect(vpc.privateSubnets.length).toEqual(4);
      expect(vpc.isolatedSubnets.length).toEqual(0);

      restoreContextProvider(previous);
    });

    test('selectSubnets onePerAz works on imported VPC', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [
          {
            name: 'Public',
            type: cxapi.VpcSubnetGroupType.PUBLIC,
            subnets: [
              {
                subnetId: 'pub-sub-in-us-east-1a',
                availabilityZone: 'us-east-1a',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pub-sub-in-us-east-1b',
                availabilityZone: 'us-east-1b',
                routeTableId: 'rt-123',
              },
            ],
          },
          {
            name: 'Private',
            type: cxapi.VpcSubnetGroupType.PRIVATE,
            subnets: [
              {
                subnetId: 'pri-sub-1-in-us-east-1c',
                availabilityZone: 'us-east-1c',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-2-in-us-east-1c',
                availabilityZone: 'us-east-1c',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-1-in-us-east-1d',
                availabilityZone: 'us-east-1d',
                routeTableId: 'rt-123',
              },
              {
                subnetId: 'pri-sub-2-in-us-east-1d',
                availabilityZone: 'us-east-1d',
                routeTableId: 'rt-123',
              },
            ],
          },
        ],
      }, options => {
        expect(options.filter).toEqual({
          isDefault: 'true',
        });

        expect(options.subnetGroupNameTag).toEqual(undefined);
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        isDefault: true,
      });

      // WHEN
      const subnets = vpc.selectSubnets({ subnetType: SubnetType.PRIVATE_WITH_EGRESS, onePerAz: true });

      // THEN: we got 2 subnets and not 4
      expect(subnets.subnets.map(s => s.availabilityZone)).toEqual(['us-east-1c', 'us-east-1d']);

      restoreContextProvider(previous);
    });

    test('AZ in dummy lookup VPC matches AZ in Stack', () => {
      // GIVEN
      const stack = new Stack(undefined, 'MyTestStack', { env: { account: '123456789012', region: 'dummy' } });
      const vpc = Vpc.fromLookup(stack, 'vpc', { isDefault: true });

      // WHEN
      const subnets = vpc.selectSubnets({
        availabilityZones: stack.availabilityZones,
      });

      // THEN
      expect(subnets.subnets.length).toEqual(2);
    });

    test('don\'t crash when using subnetgroup name in lookup VPC', () => {
      // GIVEN
      const stack = new Stack(undefined, 'MyTestStack', { env: { account: '123456789012', region: 'dummy' } });
      const vpc = Vpc.fromLookup(stack, 'vpc', { isDefault: true });

      // WHEN
      new Instance(stack, 'Instance', {
        vpc,
        instanceType: new InstanceType('t2.large'),
        machineImage: new GenericLinuxImage({ dummy: 'ami-1234' }),
        vpcSubnets: {
          subnetGroupName: 'application_layer',
        },
      });

      // THEN -- no exception occurred
    });
    test('subnets in imported VPC has all expected attributes', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [
          {
            name: 'Public',
            type: cxapi.VpcSubnetGroupType.PUBLIC,
            subnets: [
              {
                subnetId: 'pub-sub-in-us-east-1a',
                availabilityZone: 'us-east-1a',
                routeTableId: 'rt-123',
                cidr: '10.100.0.0/24',
              },
            ],
          },
        ],
      }, options => {
        expect(options.filter).toEqual({
          isDefault: 'true',
        });

        expect(options.subnetGroupNameTag).toEqual(undefined);
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        isDefault: true,
      });

      let subnet = vpc.publicSubnets[0];

      expect(subnet.availabilityZone).toEqual('us-east-1a');
      expect(subnet.subnetId).toEqual('pub-sub-in-us-east-1a');
      expect(subnet.routeTable.routeTableId).toEqual('rt-123');
      expect(subnet.ipv4CidrBlock).toEqual('10.100.0.0/24');

      restoreContextProvider(previous);
    });
    test('passes account and region', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
      }, options => {
        expect(options.region).toEqual('region-1234');
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
        region: 'region-1234',
      });

      expect(vpc.vpcId).toEqual('vpc-1234');

      restoreContextProvider(previous);
    });

    test('passes region to LookedUpVpc correctly', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
        region: 'region-1234',
      }, options => {
        expect(options.region).toEqual('region-1234');
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
        region: 'region-1234',
      });

      expect(vpc.env.region).toEqual('region-1234');
      restoreContextProvider(previous);
    });

    test('passes owner account id to LookedUpVpc correctly', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
        region: 'region-1234',
        ownerAccountId: '123456789012',
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
      });
      expect(vpc.env.account).toEqual('123456789012');
      restoreContextProvider(previous);
    });

    test('passes owner account id to context query correctly', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
        region: 'region-1234',
        ownerAccountId: '123456789012',
      }, options => {
        expect(options.filter['owner-id']).toEqual('123456789012');
      });

      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
        ownerAccountId: '123456789012',
      });
      expect(vpc.env.account).toEqual('123456789012');
      restoreContextProvider(previous);
    });

    test('a looked up VPC in a different region shared from an account has correct VPC', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
        region: 'region-1234',
        ownerAccountId: '123456789012',
      });
      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
      });
      expect(stack.resolve(vpc.vpcArn)).toEqual({
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':ec2:region-1234:123456789012:vpc/vpc-1234',
          ],
        ],
      });
      restoreContextProvider(previous);
    });

    test('a looked up VPC falls back to the parent stack\'s account and region', () => {
      const previous = mockVpcContextProviderWith({
        vpcId: 'vpc-1234',
        subnetGroups: [],
      });
      const stack = new Stack();
      const vpc = Vpc.fromLookup(stack, 'Vpc', {
        vpcId: 'vpc-1234',
      });
      expect(stack.resolve(vpc.vpcArn)).toEqual({
        'Fn::Join': [
          '',
          [
            'arn:',
            {
              Ref: 'AWS::Partition',
            },
            ':ec2:',
            {
              Ref: 'AWS::Region',
            },
            ':',
            {
              Ref: 'AWS::AccountId',
            },
            ':vpc/vpc-1234',
          ],
        ],
      });
      restoreContextProvider(previous);
    });
  });
});

interface MockVpcContextResponse {
  readonly vpcId: string;
  readonly subnetGroups: cxapi.VpcSubnetGroup[];
  readonly ownerAccountId?: string;
  readonly region?: string;
}

function mockVpcContextProviderWith(
  response: MockVpcContextResponse,
  paramValidator?: (options: cxschema.VpcContextQuery) => void) {
  const previous = ContextProvider.getValue;
  ContextProvider.getValue = (_scope: Construct, options: GetContextValueOptions) => {
    // do some basic sanity checks
    expect(options.provider).toEqual(cxschema.ContextProvider.VPC_PROVIDER);
    expect((options.props || {}).returnAsymmetricSubnets).toEqual(true);

    if (paramValidator) {
      paramValidator(options.props as any);
    }

    return {
      value: {
        availabilityZones: [],
        isolatedSubnetIds: undefined,
        isolatedSubnetNames: undefined,
        isolatedSubnetRouteTableIds: undefined,
        privateSubnetIds: undefined,
        privateSubnetNames: undefined,
        privateSubnetRouteTableIds: undefined,
        publicSubnetIds: undefined,
        publicSubnetNames: undefined,
        publicSubnetRouteTableIds: undefined,
        ...response,
      } as cxapi.VpcContextResponse,
    };
  };
  return previous;
}

function restoreContextProvider(previous: (scope: Construct, options: GetContextValueOptions) => GetContextValueResult): void {
  ContextProvider.getValue = previous;
}
