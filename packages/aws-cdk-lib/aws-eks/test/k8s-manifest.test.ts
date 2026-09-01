import { KubectlV31Layer } from '@aws-cdk/lambda-layer-kubectl-v31';
import { Template } from '../../assertions';
import { App, CfnResource, RemovalPolicy, Stack } from '../../core';
import { Cluster, KubernetesManifest, KubernetesVersion, HelmChart } from '../lib';

describe('k8s manifest', () => {
  let app: App;
  let stack: Stack;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app, 'Stack');
  });

  test('basic usage', () => {
    const cluster = new Cluster(stack, 'Cluster', {
      version: KubernetesVersion.V1_30,
      kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
    });
    const manifest = [
      {
        apiVersion: 'v1',
        kind: 'Service',
        metadata: {
          name: 'hello-kubernetes',
        },
        spec: {
          type: 'LoadBalancer',
          ports: [
            { port: 80, targetPort: 8080 },
          ],
          selector: {
            app: 'hello-kubernetes',
          },
        },
      },
      {
        apiVersion: 'apps/v1',
        kind: 'Deployment',
        metadata: {
          name: 'hello-kubernetes',
        },
        spec: {
          replicas: 2,
          selector: {
            matchLabels: {
              app: 'hello-kubernetes',
            },
          },
          template: {
            metadata: {
              labels: {
                app: 'hello-kubernetes',
              },
            },
            spec: {
              containers: [
                {
                  name: 'hello-kubernetes',
                  image: 'paulbouwer/hello-kubernetes:1.5',
                  ports: [
                    { containerPort: 8080 },
                  ],
                },
              ],
            },
          },
        },
      },
    ];

    // WHEN
    new KubernetesManifest(stack, 'manifest', {
      cluster,
      manifest,
    });

    Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
      Manifest: JSON.stringify(manifest),
    });
  });

  test('can be added to an imported cluster with minimal config', () => {
    // GIVEN
    const cluster = Cluster.fromClusterAttributes(stack, 'MyCluster', {
      clusterName: 'my-cluster-name',
      kubectlRoleArn: 'arn:aws:iam::111111111111:role/iam-role-that-has-masters-access',
    });

    // WHEN
    cluster.addManifest('foo', { bar: 2334 });
    cluster.addHelmChart('helm', { chart: 'hello-world' });

    // THEN
    Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
      Manifest: '[{"bar":2334}]',
      ClusterName: 'my-cluster-name',
      RoleArn: 'arn:aws:iam::111111111111:role/iam-role-that-has-masters-access',
    });

    Template.fromStack(stack).hasResourceProperties(HelmChart.RESOURCE_TYPE, {
      ClusterName: 'my-cluster-name',
      RoleArn: 'arn:aws:iam::111111111111:role/iam-role-that-has-masters-access',
      Release: 'stackmyclustercharthelmb653160c',
      Chart: 'hello-world',
      Namespace: 'default',
      CreateNamespace: true,
    });
  });

  test('default child is a CfnResource', () => {
    const cluster = Cluster.fromClusterAttributes(stack, 'MyCluster', {
      clusterName: 'my-cluster-name',
      kubectlRoleArn: 'arn:aws:iam::111111111111:role/iam-role-that-has-masters-access',
    });

    const manifest = cluster.addManifest('foo', { bar: 2334 });
    expect(manifest.node.defaultChild).toBeInstanceOf(CfnResource);
  });

  describe('prune labels', () => {
    test('base case', () => {
      // prune is enabled by default
      const cluster = new Cluster(stack, 'Cluster', {
        version: KubernetesVersion.V1_16,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });

      expect(cluster.prune).toEqual(true);

      // WHEN
      cluster.addManifest('m1', {
        apiVersion: 'v1beta1',
        kind: 'Foo',
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([{
          apiVersion: 'v1beta1',
          kind: 'Foo',
          metadata: {
            labels: {
              'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac': '',
            },
          },
        }]),
        PruneLabel: 'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac',
      });
    });

    test('multiple resources in the same manifest', () => {
      // GIVEN
      const cluster = new Cluster(stack, 'Cluster', {
        prune: true,
        version: KubernetesVersion.V1_30,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });

      // WHEN
      cluster.addManifest('m1',
        {
          apiVersion: 'v1beta',
          kind: 'Foo',
        },
        {
          apiVersion: 'v1',
          kind: 'Pod',
          metadata: {
            name: 'foo',
            labels: {
              bar: 1234,
            },
          },
          spec: {
            containers: [{ name: 'main', image: 'main' }],
          },
        },
      );

      // THEN
      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([
          {
            apiVersion: 'v1beta',
            kind: 'Foo',
            metadata: {
              labels: {
                'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac': '',
              },
            },
          },
          {
            apiVersion: 'v1',
            kind: 'Pod',
            metadata: {
              name: 'foo',
              labels: {
                'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac': '',
                'bar': 1234,
              },
            },
            spec: {
              containers: [
                {
                  name: 'main',
                  image: 'main',
                },
              ],
            },
          },
        ]),
        PruneLabel: 'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac',
      });
    });

    test('different KubernetesManifest resource use different prune labels', () => {
      // GIVEN
      const cluster = new Cluster(stack, 'Cluster', {
        version: KubernetesVersion.V1_30,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });

      // WHEN
      cluster.addManifest('m1', {
        apiVersion: 'v1beta',
        kind: 'Foo',
      });

      cluster.addManifest('m2', {
        apiVersion: 'v1',
        kind: 'Pod',
        metadata: {
          name: 'foo',
          labels: {
            bar: 1234,
          },
        },
        spec: {
          containers: [{ name: 'main', image: 'main' }],
        },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([
          {
            apiVersion: 'v1beta',
            kind: 'Foo',
            metadata: {
              labels: {
                'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac': '',
              },
            },
          },
        ]),
        PruneLabel: 'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac',
      });

      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([
          {
            apiVersion: 'v1',
            kind: 'Pod',
            metadata: {
              name: 'foo',
              labels: {
                'aws.cdk.eks/prune-c8aff6ac817006dd4d644e9d99b2cdbb8c8cd036d9': '',
                'bar': 1234,
              },
            },
            spec: {
              containers: [
                {
                  name: 'main',
                  image: 'main',
                },
              ],
            },
          },
        ]),
        PruneLabel: 'aws.cdk.eks/prune-c8aff6ac817006dd4d644e9d99b2cdbb8c8cd036d9',
      });
    });

    test('ignores resources without "kind"', () => {
      // GIVEN
      const cluster = new Cluster(stack, 'Cluster', {
        version: KubernetesVersion.V1_30,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });

      // WHEN
      cluster.addManifest('m1', {
        malformed: { resource: 'yes' },
      });

      // THEN
      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([{ malformed: { resource: 'yes' } }]),
        PruneLabel: 'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac',
      });
    });

    test('ignores entries that are not objects (invalid type)', () => {
      // GIVEN
      const cluster = new Cluster(stack, 'Cluster', {
        prune: true,
        version: KubernetesVersion.V1_30,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });
      expect(cluster.prune).toEqual(true);

      // WHEN
      cluster.addManifest('m1', ['foo']);

      // THEN
      Template.fromStack(stack).hasResourceProperties(KubernetesManifest.RESOURCE_TYPE, {
        Manifest: JSON.stringify([['foo']]),
        PruneLabel: 'aws.cdk.eks/prune-c89a5983505f58231ac2a9a86fd82735ccf2308eac',
      });
    });

    test('no prune labels when "prune" is disabled', () => {
      // GIVEN
      const cluster = new Cluster(stack, 'Cluster', {
        prune: false,
        version: KubernetesVersion.V1_30,
        kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
      });

      // WHEN
      cluster.addManifest('m1', { apiVersion: 'v1beta', kind: 'Foo' });

      // if "prune" is not specified at the manifest level, it is derived from the cluster settings.
      new KubernetesManifest(stack, 'm2', {
        cluster,
        manifest: [{ apiVersion: 'v1', kind: 'Pod' }],
      });

      // can be overridden at the manifest level
      new KubernetesManifest(stack, 'm3', {
        cluster,
        manifest: [{ apiVersion: 'v1', kind: 'Deployment' }],
        prune: true,
      });

      // THEN
      const template = Template.fromStack(stack).toJSON();

      const m1 = template.Resources.Clustermanifestm1E5FBE3C1.Properties;
      const m2 = template.Resources.m201F909C5.Properties;
      const m3 = template.Resources.m3B0AF9264.Properties;

      expect(m1.Manifest).toEqual(JSON.stringify([{ apiVersion: 'v1beta', kind: 'Foo' }]));
      expect(m2.Manifest).toEqual(JSON.stringify([{ apiVersion: 'v1', kind: 'Pod' }]));
      expect(m3.Manifest).toEqual(JSON.stringify([
        {
          apiVersion: 'v1',
          kind: 'Deployment',
          metadata: {
            labels: {
              'aws.cdk.eks/prune-c8971972440c5bb3661e468e4cb8069f7ee549414c': '',
            },
          },
        },
      ]));
      expect(m1.PruneLabel).toBeFalsy();
      expect(m2.PruneLabel).toBeFalsy();
      expect(m3.PruneLabel).toEqual('aws.cdk.eks/prune-c8971972440c5bb3661e468e4cb8069f7ee549414c');
    });
  });

  test('supports custom removal policy', () => {
    // GIVEN
    const cluster = new Cluster(stack, 'Cluster', {
      version: KubernetesVersion.V1_30,
      kubectlLayer: new KubectlV31Layer(stack, 'KubectlLayer'),
    });

    // WHEN
    new KubernetesManifest(stack, 'manifest', {
      cluster,
      manifest: [{ apiVersion: 'v1', kind: 'Pod' }],
      removalPolicy: RemovalPolicy.RETAIN,
    });

    // THEN
    Template.fromStack(stack).hasResource(KubernetesManifest.RESOURCE_TYPE, {
      DeletionPolicy: 'Retain',
    });
  });
});
