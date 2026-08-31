import { App, Stack } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { Role, ServicePrincipal } from 'aws-cdk-lib/aws-iam';
import {
  Cluster,
  ClusterType,
  ChannelPlacementGroup,
} from '../lib';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

describe('Cluster', () => {
  function instanceRole() {
    return new Role(stack, 'InstanceRole', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
    });
  }

  test('creates a minimal cluster', () => {
    new Cluster(stack, 'Cluster', {
      instanceRole: instanceRole(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Cluster', {
      InstanceRoleArn: {
        'Fn::GetAtt': [Match.stringLikeRegexp('^InstanceRole'), 'Arn'],
      },
    });
  });

  test.each([
    ClusterType.ON_PREMISES,
    ClusterType.OUTPOSTS_RACK,
    ClusterType.OUTPOSTS_SERVER,
    ClusterType.EC2,
  ])('renders clusterType %s', (clusterType) => {
    new Cluster(stack, 'Cluster', {
      clusterType,
      instanceRole: instanceRole(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Cluster', {
      ClusterType: clusterType.value,
    });
  });

  test('renders network settings with default route and interface mappings', () => {
    new Cluster(stack, 'Cluster', {
      instanceRole: instanceRole(),
      networkSettings: {
        defaultRoute: '10.0.0.1',
        interfaceMappings: [
          { logicalInterfaceName: 'my-interface', networkId: 'network-1' },
        ],
      },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Cluster', {
      NetworkSettings: {
        DefaultRoute: '10.0.0.1',
        InterfaceMappings: [
          { LogicalInterfaceName: 'my-interface', NetworkId: 'network-1' },
        ],
      },
    });
  });

  test('renders tags', () => {
    new Cluster(stack, 'Cluster', {
      instanceRole: instanceRole(),
      tags: { env: 'prod' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Cluster', {
      Tags: [{ Key: 'env', Value: 'prod' }],
    });
  });

  test('uses an explicit cluster name', () => {
    new Cluster(stack, 'Cluster', {
      clusterName: 'my-cluster',
      instanceRole: instanceRole(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::Cluster', {
      Name: 'my-cluster',
    });
  });

  test('fromClusterArn derives the cluster id from the ARN', () => {
    const imported = Cluster.fromClusterArn(stack, 'Imported', 'arn:aws:medialive:us-east-1:123456789012:cluster:cluster-123');

    expect(imported.clusterArn).toBe('arn:aws:medialive:us-east-1:123456789012:cluster:cluster-123');
    expect(imported.clusterId).toBe('cluster-123');
    expect(imported.clusterChannelIds).toBeUndefined();
    expect(imported.clusterState).toBeUndefined();
  });

  test('fromClusterArn throws when the ARN has no resource name segment', () => {
    expect(() => Cluster.fromClusterArn(stack, 'Imported', 'arn:aws:medialive:us-east-1:123456789012:cluster')).toThrow(/Could not parse MediaLive Cluster ARN/);
  });
});

describe('ChannelPlacementGroup', () => {
  function cluster() {
    const instanceRole = new Role(stack, 'InstanceRole', {
      assumedBy: new ServicePrincipal('medialive.amazonaws.com'),
    });
    return new Cluster(stack, 'Cluster', { instanceRole });
  }

  test('creates a minimal channel placement group', () => {
    new ChannelPlacementGroup(stack, 'Group', {
      cluster: cluster(),
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::ChannelPlacementGroup', {
      ClusterId: { Ref: Match.stringLikeRegexp('^Cluster') },
    });
  });

  test('renders nodes and tags', () => {
    new ChannelPlacementGroup(stack, 'Group', {
      cluster: cluster(),
      channelPlacementGroupName: 'my-group',
      nodes: ['node-1', 'node-2'],
      tags: { env: 'prod' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::ChannelPlacementGroup', {
      Name: 'my-group',
      Nodes: ['node-1', 'node-2'],
      Tags: [{ Key: 'env', Value: 'prod' }],
    });
  });

  test('fromChannelPlacementGroupAttributes wires the provided attributes', () => {
    const imported = ChannelPlacementGroup.fromChannelPlacementGroupAttributes(stack, 'Imported', {
      channelPlacementGroupArn: 'arn:aws:medialive:us-east-1:123456789012:channelPlacementGroup:cpg-123',
      channelPlacementGroupId: 'cpg-123',
      clusterId: 'cluster-123',
    });

    expect(imported.channelPlacementGroupArn).toBe('arn:aws:medialive:us-east-1:123456789012:channelPlacementGroup:cpg-123');
    expect(imported.channelPlacementGroupId).toBe('cpg-123');
    expect(imported.channelPlacementGroupRef).toEqual({
      channelPlacementGroupId: 'cpg-123',
      clusterId: 'cluster-123',
      channelPlacementGroupArn: 'arn:aws:medialive:us-east-1:123456789012:channelPlacementGroup:cpg-123',
    });
  });

  test('channelPlacementGroupRef resolves the cluster id from the cluster construct', () => {
    const clusterConstruct = cluster();
    const group = new ChannelPlacementGroup(stack, 'Group', { cluster: clusterConstruct });

    expect(group.channelPlacementGroupRef.clusterId).toBe(clusterConstruct.clusterId);
  });
});
