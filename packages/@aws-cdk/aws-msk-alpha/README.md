# Amazon Managed Streaming for Apache Kafka Construct Library
<!--BEGIN STABILITY BANNER-->

---

![cdk-constructs: Experimental](https://img.shields.io/badge/cdk--constructs-experimental-important.svg?style=for-the-badge)

> The APIs of higher level constructs in this module are experimental and under active development.
> They are subject to non-backward compatible changes or removal in any future version. These are
> not subject to the [Semantic Versioning](https://semver.org/) model and breaking changes will be
> announced in the release notes. This means that while you may use them, you may need to update
> your source code when upgrading to a newer version of this package.

---

<!--END STABILITY BANNER-->

[Amazon MSK](https://aws.amazon.com/msk/) is a fully managed service that makes it easy for you to build and run applications that use Apache Kafka to process streaming data.

The following example creates an MSK Cluster.

```ts
declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'Cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
});
```

## Allowing Connections

To control who can access the Cluster, use the `.connections` attribute. For a list of ports used by MSK, refer to the [MSK documentation](https://docs.aws.amazon.com/msk/latest/developerguide/client-access.html#port-info).

```ts
declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'Cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
});

cluster.connections.allowFrom(
  ec2.Peer.ipv4('1.2.3.4/8'),
  ec2.Port.tcp(2181),
);
cluster.connections.allowFrom(
  ec2.Peer.ipv4('1.2.3.4/8'),
  ec2.Port.tcp(9094),
);
```

## Cluster Endpoints

You can use the following attributes to get a list of the Kafka broker or ZooKeeper node endpoints

```ts
declare const cluster: msk.Cluster;
new CfnOutput(this, 'BootstrapBrokers', { value: cluster.bootstrapBrokers });
new CfnOutput(this, 'BootstrapBrokersTls', { value: cluster.bootstrapBrokersTls });
new CfnOutput(this, 'BootstrapBrokersSaslScram', { value: cluster.bootstrapBrokersSaslScram });
new CfnOutput(this, 'BootstrapBrokerStringSaslIam', { value: cluster.bootstrapBrokersSaslIam });
new CfnOutput(this, 'ZookeeperConnection', { value: cluster.zookeeperConnectionString });
new CfnOutput(this, 'ZookeeperConnectionTls', { value: cluster.zookeeperConnectionStringTls });
```

## Importing an existing Cluster

To import an existing MSK cluster into your CDK app use the `.fromClusterArn()` method.

```ts
const cluster = msk.Cluster.fromClusterArn(this, 'Cluster',
  'arn:aws:kafka:us-west-2:1234567890:cluster/a-cluster/11111111-1111-1111-1111-111111111111-1',
);
```

## Client Authentication

[MSK supports](https://docs.aws.amazon.com/msk/latest/developerguide/kafka_apis_iam.html) the following authentication mechanisms.

### TLS

To enable client authentication with TLS set the `certificateAuthorityArns` property to reference your ACM Private CA. [More info on Private CAs.](https://docs.aws.amazon.com/msk/latest/developerguide/msk-authentication.html)

```ts
import * as acmpca from 'aws-cdk-lib/aws-acmpca';

declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'Cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  encryptionInTransit: {
    clientBroker: msk.ClientBrokerEncryption.TLS,
  },
  clientAuthentication: msk.ClientAuthentication.tls({
    certificateAuthorities: [
      acmpca.CertificateAuthority.fromCertificateAuthorityArn(
        this,
        'CertificateAuthority',
        'arn:aws:acm-pca:us-west-2:1234567890:certificate-authority/11111111-1111-1111-1111-111111111111',
      ),
    ],
  }),
});
```

### SASL/SCRAM

Enable client authentication with [SASL/SCRAM](https://docs.aws.amazon.com/msk/latest/developerguide/msk-password.html):

```ts
declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  encryptionInTransit: {
    clientBroker: msk.ClientBrokerEncryption.TLS,
  },
  clientAuthentication: msk.ClientAuthentication.sasl({
    scram: true,
  }),
});
```

### IAM

Enable client authentication with [IAM](https://docs.aws.amazon.com/msk/latest/developerguide/iam-access-control.html):

```ts
declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  encryptionInTransit: {
    clientBroker: msk.ClientBrokerEncryption.TLS,
  },
  clientAuthentication: msk.ClientAuthentication.sasl({
    iam: true,
  }),
});
```


### SASL/IAM + TLS

Enable client authentication with [IAM](https://docs.aws.amazon.com/msk/latest/developerguide/iam-access-control.html)
as well as enable client authentication with TLS by setting the `certificateAuthorityArns` property to reference your ACM Private CA. [More info on Private CAs.](https://docs.aws.amazon.com/msk/latest/developerguide/msk-authentication.html)

```ts
import * as acmpca from 'aws-cdk-lib/aws-acmpca';

declare const vpc: ec2.Vpc;
const cluster = new msk.Cluster(this, 'Cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  encryptionInTransit: {
    clientBroker: msk.ClientBrokerEncryption.TLS,
  },
  clientAuthentication: msk.ClientAuthentication.saslTls({
    iam: true,
    certificateAuthorities: [
      acmpca.CertificateAuthority.fromCertificateAuthorityArn(
        this,
        'CertificateAuthority',
        'arn:aws:acm-pca:us-west-2:1234567890:certificate-authority/11111111-1111-1111-1111-111111111111',
      ),
    ],
  }),
});
```


## Logging

You can deliver Apache Kafka broker logs to one or more of the following destination types:
Amazon CloudWatch Logs, Amazon S3, Amazon Data Firehose.

To configure logs to be sent to an S3 bucket, provide a bucket in the `logging` config.

```ts
declare const vpc: ec2.Vpc;
declare const bucket: s3.IBucket;
const cluster = new msk.Cluster(this, 'cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  logging: {
    s3: {
      bucket,
    },
  },
});
```

When the S3 destination is configured, AWS will automatically create an S3 bucket policy
that allows the service to write logs to the bucket. This makes it impossible to later update
that bucket policy. To have CDK create the bucket policy so that future updates can be made,
the `@aws-cdk/aws-s3:createDefaultLoggingPolicy` [feature flag](https://docs.aws.amazon.com/cdk/v2/guide/featureflags.html) can be used. This can be set
in the `cdk.json` file.

```json
{
  "context": {
    "@aws-cdk/aws-s3:createDefaultLoggingPolicy": true
  }
}
```

## Storage Mode

You can configure an MSK cluster storage mode using the `storageMode` property.

Tiered storage is a low-cost storage tier for Amazon MSK that scales to virtually unlimited storage,
making it cost-effective to build streaming data applications.

> Visit [Tiered storage](https://docs.aws.amazon.com/msk/latest/developerguide/msk-tiered-storage.html)
to see the list of compatible Kafka versions and for more details.

```ts
declare const vpc: ec2.Vpc;
declare const bucket: s3.IBucket;

const cluster = new msk.Cluster(this, 'cluster', {
  clusterName: 'myCluster',
  kafkaVersion: msk.KafkaVersion.V4_1_X_KRAFT,
  vpc,
  storageMode: msk.StorageMode.TIERED,
});
```

## MSK Express Brokers

You can create an MSK cluster with Express Brokers by setting the `brokerType` property to `BrokerType.EXPRESS`. Express Brokers are a low-cost option for development, testing, and workloads that don't require the high availability guarantees of standard MSK cluster.
For more information, see [Amazon MSK Express Brokers](https://docs.aws.amazon.com/msk/latest/developerguide/msk-broker-types-express.html).

**Note:** When using Express Brokers, the following constraints apply:

- Apache Kafka version must be 3.6.x, 3.8.x, 3.9.x, or 4.2.x
- You must specify the `instanceType`
- The VPC must have at least 3 subnets (across 3 AZs)
- `ebsStorageInfo` is not supported
- `storageMode` is not supported
- `logging` is not supported
- Supported broker sizes: `m7g.xlarge`, `m7g.2xlarge`, `m7g.4xlarge`, `m7g.8xlarge`, `m7g.12xlarge`, `m7g.16xlarge`

```ts
declare const vpc: ec2.Vpc;

const expressCluster = new msk.Cluster(this, 'ExpressCluster', {
  clusterName: 'MyExpressCluster',
  kafkaVersion: msk.KafkaVersion.V3_8_X,
  vpc,
  brokerType: msk.BrokerType.EXPRESS,
  instanceType: ec2.InstanceType.of(
    ec2.InstanceClass.M7G,
    ec2.InstanceSize.XLARGE,
  ),
});
```

## MSK Serverless

You can also use MSK Serverless by using `ServerlessCluster` class.

MSK Serverless is a cluster type for Amazon MSK that makes it possible for you to run Apache Kafka without having to manage and scale cluster capacity.

MSK Serverless requires IAM access control for all clusters.

For more infomation, see [Use MSK Serverless clusters](https://docs.aws.amazon.com/msk/latest/developerguide/serverless-getting-started.html).

```ts
declare const vpc: ec2.Vpc;

const serverlessCluster = new msk.ServerlessCluster(this, 'ServerlessCluster', {
  clusterName: 'MyServerlessCluster',
  vpcConfigs: [
    { vpc },
  ],
});
```
