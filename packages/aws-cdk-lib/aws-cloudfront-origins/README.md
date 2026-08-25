# CloudFront Origins for the CDK CloudFront Library

This library contains convenience methods for defining origins for a CloudFront distribution. You can use this library to create origins from
S3 buckets, Elastic Load Balancing v2 load balancers, or any other domain name.

## S3 Bucket

An S3 bucket can be used as an origin. An S3 bucket origin can either be configured using a standard S3 bucket or using a S3 bucket that's configured as a website endpoint (see AWS docs for [Using an S3 Bucket](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/DownloadDistS3AndCustomOrigins.html#using-s3-as-origin)).

> Note: `S3Origin` has been deprecated. Use `S3BucketOrigin` for standard S3 origins and `S3StaticWebsiteOrigin` for static website S3 origins.

### Standard S3 Bucket

To set up an origin using a standard S3 bucket, use the `S3BucketOrigin` class. The bucket
is handled as a bucket origin and
CloudFront's redirect and error handling will be used. It is recommended to use `S3BucketOrigin.withOriginAccessControl()` to configure OAC for your origin.

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: origins.S3BucketOrigin.withOriginAccessControl(myBucket) },
});
```

> Note: When you use CloudFront OAC with Amazon S3 bucket origins, you must set Amazon S3 Object Ownership to Bucket owner enforced (the default for new Amazon S3 buckets). If you require ACLs, use the Bucket owner preferred setting to maintain control over objects uploaded via CloudFront.

### S3 Bucket Configured as a Website Endpoint

To set up an origin using an S3 bucket configured as a website endpoint, use the `S3StaticWebsiteOrigin` class. When the bucket is configured as a
website endpoint, the bucket is treated as an HTTP origin,
and the distribution can use built-in S3 redirects and S3 custom error pages.

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: new origins.S3StaticWebsiteOrigin(myBucket) },
});
```

### Restricting access to a standard S3 Origin

CloudFront provides two ways to send authenticated requests to a standard Amazon S3 origin:

* origin access control (OAC) and
* origin access identity (OAI)

OAI is considered legacy due to limited functionality and regional
limitations, whereas OAC is recommended because it supports all Amazon S3
buckets in all AWS Regions, Amazon S3 server-side encryption with AWS KMS (SSE-KMS), and dynamic requests (PUT and DELETE) to Amazon S3. Additionally,
OAC provides stronger security posture with short term credentials,
and more frequent credential rotations as compared to OAI. OAI and OAC can be used in conjunction with a bucket that is not public to
require that your users access your content using CloudFront URLs and not S3 URLs directly.

See AWS docs on [Restricting access to an Amazon S3 Origin](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html) for more details.

> Note: OAC and OAI can only be used with an regular S3 bucket origin (not a bucket configured as a website endpoint).

The `S3BucketOrigin` class supports creating a standard S3 origin with OAC, OAI, and no access control (using your bucket access settings) via
the `withOriginAccessControl()`, `withOriginAccessIdentity()`, and `withBucketDefaults()` methods respectively.

#### Setting up a new origin access control (OAC)

Setup a standard S3 origin with origin access control as follows:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withOriginAccessControl(myBucket) // Automatically creates a S3OriginAccessControl construct
  },
});
```

When creating a standard S3 origin using `origins.S3BucketOrigin.withOriginAccessControl()`, an [Origin Access Control resource](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-cloudfront-originaccesscontrol-originaccesscontrolconfig.html) is automatically created with the origin type set to `s3` and signing behavior set to `always`.

You can grant read, read versioned, list, write or delete access to the OAC using the `originAccessLevels` property:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
const s3Origin = origins.S3BucketOrigin.withOriginAccessControl(myBucket, { originAccessLevels: [cloudfront.AccessLevel.READ, cloudfront.AccessLevel.READ_VERSIONED, cloudfront.AccessLevel.WRITE, cloudfront.AccessLevel.DELETE],
});
```

The read versioned permission does contain the read permission, so it's required to set both `AccessLevel.READ` and
`AccessLevel.READ_VERSIONED`.

For details of list permission, see [Setting up OAC with LIST permission](#setting-up-oac-with-list-permission).

You can also pass in a custom S3 origin access control:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
const oac = new cloudfront.S3OriginAccessControl(this, 'MyOAC', { 
  signing: cloudfront.Signing.SIGV4_NO_OVERRIDE
});
const s3Origin = origins.S3BucketOrigin.withOriginAccessControl(myBucket, {
    originAccessControl: oac 
  }
)
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: s3Origin
  },
});
```

An existing S3 origin access control can be imported using the `fromOriginAccessControlId` method:

```ts
const importedOAC = cloudfront.S3OriginAccessControl.fromOriginAccessControlId(this, 'myImportedOAC', 'ABC123ABC123AB');
```

> [Note](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html): When you use OAC with S3
bucket origins, the bucket's object ownership must be either set to Bucket owner enforced (default for new S3 buckets) or Bucket owner preferred (only if you require ACLs).

#### Setting up OAC with a SSE-KMS encrypted S3 origin

If the objects in the S3 bucket origin are encrypted using server-side encryption with
AWS Key Management Service (SSE-KMS), the OAC must have permission to use the KMS key.

Setting up a standard S3 origin using `S3BucketOrigin.withOriginAccessControl()` will automatically add the statement to the KMS key policy
to give the OAC permission to use the KMS key.

```ts
import * as kms from 'aws-cdk-lib/aws-kms';

const myKmsKey = new kms.Key(this, 'myKMSKey');
const myBucket = new s3.Bucket(this, 'mySSEKMSEncryptedBucket', {
  encryption: s3.BucketEncryption.KMS,
  encryptionKey: myKmsKey,
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withOriginAccessControl(myBucket) // Automatically grants Distribution access to `myKmsKey`
  },
});
```

##### Scoping down the key policy

I saw this warning message during synth time. What do I do?

```text
To avoid a circular dependency between the KMS key, Bucket, and Distribution during the initial deployment, a wildcard is used in the Key policy condition to match all Distribution IDs.
After deploying once, it is strongly recommended to further scope down the policy for best security practices by following the guidance in the "Using OAC for a SSE-KMS encrypted S3 origin" section in the module README.
```

If the S3 bucket has an `encryptionKey` defined, `S3BucketOrigin.withOriginAccessControl()`
will automatically add the following policy statement to the KMS key policy to allow CloudFront read-only access (unless otherwise specified in the `originAccessLevels` property).

```json
{
    "Statement": {
        "Effect": "Allow",
        "Principal": {
            "Service": "cloudfront.amazonaws.com"
        },
        "Action": "kms:Decrypt",
        "Resource": "*",
        "Condition": {
            "ArnLike": {
                "AWS:SourceArn": "arn:aws:cloudfront::<account ID>:distribution/*"
            }
        }
    }
}
```

This policy uses a wildcard to match all distribution IDs in the account instead of referencing the specific distribution ID to resolve the circular dependency. The policy statement is not as scoped down as the example in the AWS CloudFront docs (see [SSE-KMS section](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#create-oac-overview-s3)).

After you have deployed the Distribution, you should follow these steps to only grant permissions to the specific distribution according to AWS best practices:

**Step 1.** Copy the key policy

**Step 2.** Use an escape hatch to update the policy statement condition so that

```json
  "Condition": {
      "ArnLike": {
          "AWS:SourceArn": "arn:aws:cloudfront::<account ID>:distribution/*"
      }
  }
```

...becomes...

```json
  "Condition": {
      "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::111122223333:distribution/<CloudFront distribution ID>"
      }
  }
```

> Note the change of condition operator from `ArnLike` to `StringEquals` in addition to replacing the wildcard (`*`) with the distribution ID.

To set the key policy using an escape hatch:

```ts
import * as kms from 'aws-cdk-lib/aws-kms';

const kmsKey = new kms.Key(this, 'myKMSKey');
const myBucket = new s3.Bucket(this, 'mySSEKMSEncryptedBucket', {
  encryption: s3.BucketEncryption.KMS,
  encryptionKey: kmsKey,
  objectOwnership: s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withOriginAccessControl(myBucket)
  },
});

// Add the following to scope down the key policy
const scopedDownKeyPolicy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::111122223333:root"
            },
            "Action": "kms:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "cloudfront.amazonaws.com"
            },
            "Action": [
                "kms:Decrypt",
                "kms:Encrypt",
                "kms:GenerateDataKey*"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "AWS:SourceArn": "arn:aws:cloudfront::111122223333:distribution/<CloudFront distribution ID>"
                }
            }
        }
    ]
};
const cfnKey = (kmsKey.node.defaultChild as kms.CfnKey);
cfnKey.keyPolicy = scopedDownKeyPolicy;
```

**Step 3.** Deploy the stack
> Tip: Run `cdk diff` before deploying to verify the
changes to your stack.

**Step 4.** Verify your final key policy includes the following statement after deploying:

```json
{
    "Effect": "Allow",
    "Principal": {
        "Service": [
            "cloudfront.amazonaws.com"
        ]
     },
    "Action": [
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey*"
    ],
    "Resource": "*",
    "Condition": {
            "StringEquals": {
                "AWS:SourceArn": "arn:aws:cloudfront::111122223333:distribution/<CloudFront distribution ID>"
            }
        }
}
```

##### Updating imported key policies

If you are using an imported KMS key to encrypt your S3 bucket and want to use OAC, you will need to update the
key policy manually to allow CloudFront to use the key. Like most imported resources, CDK apps cannot modify the configuration of imported keys.

After deploying the distribution, add the following policy statement to your key policy to allow CloudFront OAC to access your KMS key for SSE-KMS:

```json
{
    "Sid": "AllowCloudFrontServicePrincipalSSE-KMS",
    "Effect": "Allow",
    "Principal": {
        "Service": [
            "cloudfront.amazonaws.com"
        ]
     },
    "Action": [
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey*"
    ],
    "Resource": "*",
    "Condition": {
            "StringEquals": {
                "AWS:SourceArn": "arn:aws:cloudfront::111122223333:distribution/<CloudFront distribution ID>"
            }
        }
}
```

See CloudFront docs on [SSE-KMS](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#create-oac-overview-s3) for more details.

#### Setting up OAC with imported S3 buckets

If you are using an imported bucket for your S3 Origin and want to use OAC,
you will need to update
the S3 bucket policy manually to allow the OAC to access the S3 origin. Like most imported resources, CDK apps cannot modify the configuration of imported buckets.

After deploying the distribution, add the following
policy statement to your
S3 bucket to allow CloudFront read-only access
(or additional S3 permissions as required):

```json
{
    "Version": "2012-10-17",
    "Statement": {
        "Effect": "Allow",
        "Principal": {
            "Service": "cloudfront.amazonaws.com"
        },
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::<S3 bucket name>/*",
        "Condition": {
            "StringEquals": {
                "AWS:SourceArn": "arn:aws:cloudfront::111122223333:distribution/<CloudFront distribution ID>"
            }
        }
    }
}
```

See CloudFront docs on [Giving the origin access control permission to access the S3 bucket](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#create-oac-overview-s3) for more details.

> Note: If your bucket previously used OAI, you will need to manually remove the policy statement
that gives the OAI access to your bucket after setting up OAC.

#### Setting up OAC with LIST permission

By default, S3 origin returns 403 Forbidden HTTP response when the requested object does not exist.
When you want to receive 404 Not Found, specify `AccessLevel.LIST` in `originAccessLevels` to add `s3:ListBucket` permission in the bucket policy.

This is useful to distinguish between responses blocked by WAF (403) and responses where the file does not exist (404).

``` ts
const myBucket = new s3.Bucket(this, 'myBucket');
const s3Origin = origins.S3BucketOrigin.withOriginAccessControl(myBucket, {
  originAccessLevels: [cloudfront.AccessLevel.READ, cloudfront.AccessLevel.LIST],
});
new cloudfront.Distribution(this, 'distribution', {
  defaultBehavior: {
    origin: s3Origin,
  },
  defaultRootObject: 'index.html', // recommended to specify
});
```

When the origin is associated to the default behavior, it is highly recommended to specify `defaultRootObject` distribution property.
Without it, the root path `https://xxxx.cloudfront.net/` will return the list of the S3 object keys.

#### Setting up an OAI (legacy)

Setup an S3 origin with origin access identity (legacy) as follows:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withOriginAccessIdentity(myBucket) // Automatically creates an OAI
  },
});
```

You can also pass in a custom S3 origin access identity:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
const myOai = new cloudfront.OriginAccessIdentity(this, 'myOAI', {
  comment: 'My custom OAI'
});
const s3Origin = origins.S3BucketOrigin.withOriginAccessIdentity(myBucket, {
  originAccessIdentity: myOai
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: s3Origin
  },
});
```

#### Setting up OAI with imported S3 buckets (legacy)

If you are using an imported bucket for your S3 Origin and want to use OAI,
you will need to update
the S3 bucket policy manually to allow the OAI to access the S3 origin. Like most imported resources, CDK apps cannot modify the configuration of imported buckets.

Add the following
policy statement to your
S3 bucket to allow the OAI read access:

```json
{
    "Version": "2012-10-17",
    "Id": "PolicyForCloudFrontPrivateContent",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::cloudfront:user/CloudFront Origin Access Identity <origin access identity ID>"
            },
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::<S3 bucket name>/*"
        }
    ]
}
```

See AWS docs on [Giving an origin access identity permission to read files in the Amazon S3 bucket](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#private-content-restricting-access-to-s3-oai) for more details.

### Setting up a S3 origin with no origin access control

To setup a standard S3 origin with no access control (no OAI nor OAC), use `origins.S3BucketOrigin.withBucketDefaults()`:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: origins.S3BucketOrigin.withBucketDefaults(myBucket)
  },
});
```

### Migrating from OAI to OAC

If you are currently using OAI for your S3 origin and wish to migrate to OAC,
replace the `S3Origin` construct (deprecated) with `S3BucketOrigin.withOriginAccessControl()` which automatically
creates and sets up an OAC for you.

Existing setup using OAI and `S3Origin`:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
const s3Origin = new origins.S3Origin(myBucket);
const distribution = new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: s3Origin },
});
```

**Step 1:**

To ensure CloudFront doesn't lose access to the bucket during the transition, add a statement to bucket policy to grant OAC access to the S3 origin. Deploy the stack. If you are okay with downtime during the transition, you can skip this step.

> Tip: Run `cdk diff` before deploying to verify the
changes to your stack.

```ts
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';

const stack = new Stack();
const myBucket = new s3.Bucket(this, 'myBucket');
const s3Origin = new origins.S3Origin(myBucket);
const distribution = new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: s3Origin },
});

// Construct the bucket policy statement
const distributionArn = stack.formatArn(
  {
    service: 'cloudfront',
    region: '',
    resource: 'distribution',
    resourceName: distribution.distributionId,
    arnFormat: cdk.ArnFormat.SLASH_RESOURCE_NAME
  }
);

const cloudfrontSP = new iam.ServicePrincipal('cloudfront.amazonaws.com');

const oacBucketPolicyStatement = new iam.PolicyStatement(
  {
    effect: iam.Effect.ALLOW,
    principals: [cloudfrontSP],
    actions: ['s3:GetObject'],
    resources: [myBucket.arnForObjects('*')],
    conditions: {
      "StringEquals": {
        "AWS:SourceArn": distributionArn
      }
    }
  }
)

// Add statement to bucket policy
myBucket.addToResourcePolicy(oacBucketPolicyStatement);
```

The following changes will take place:

1. The bucket policy will be modified to grant the CloudFront distribution access. At this point the bucket policy allows both an OAI and an OAC to access the S3 origin.

**Step 2:**

Replace `S3Origin` with `S3BucketOrigin.withOriginAccessControl()`, which creates an OAC and attaches it to the distribution. You can remove the code from Step 1 which updated the bucket policy, as `S3BucketOrigin.withOriginAccessControl()` updates the bucket policy automatically with the same statement when defined in the `Distribution` (no net difference).

Run `cdk diff` before deploying to verify the changes to your stack.

```ts
const bucket = new s3.Bucket(this, 'Bucket');
const s3Origin = origins.S3BucketOrigin.withOriginAccessControl(bucket);
const distribution = new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin: s3Origin },
});
```

The following changes will take place:

1. A `AWS::CloudFront::OriginAccessControl` resource will be created.
2. The `Origin` property of the `AWS::CloudFront::Distribution` will set [`OriginAccessControlId`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-cloudfront-distribution-origin.html#cfn-cloudfront-distribution-origin-originaccesscontrolid) to the OAC ID after it is created. It will also set [`S3OriginConfig`](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-cloudfront-distribution-s3originconfig.html#aws-properties-cloudfront-distribution-s3originconfig-properties) to `{"OriginAccessIdentity": ""}`, which deletes the origin access identity from the existing distribution.
3. The `AWS::CloudFront::CloudFrontOriginAccessIdentity` resource will be deleted.

**Will migrating from OAI to OAC cause any resource replacement?**

No, following the migration steps does not cause any replacement of the existing `AWS::CloudFront::Distribution`, `AWS::S3::Bucket` nor `AWS::S3::BucketPolicy` resources. It will modify the bucket policy, create a `AWS::CloudFront::OriginAccessControl` resource, and delete the existing `AWS::CloudFront::CloudFrontOriginAccessIdentity`.

**Will migrating from OAI to OAC have any availability implications for my application?**

Updates to bucket policies are eventually consistent. Therefore, removing OAI permissions and setting up OAC in the same CloudFormation stack deployment is not recommended as it may cause downtime where CloudFront loses access to the bucket. Following the steps outlined above lowers the risk of downtime as the bucket policy is updated to have both OAI and OAC permissions, then in a subsequent deployment, the OAI permissions are removed.

For more information, see [Migrating from origin access identity (OAI) to origin access control (OAC)](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-restricting-access-to-s3.html#migrate-from-oai-to-oac).

### Adding Custom Headers

You can configure CloudFront to add custom headers to the requests that it sends to your origin. These custom headers enable you to send and gather information from your origin that you don’t get with typical viewer requests. These headers can even be customized for each origin. CloudFront supports custom headers for both for custom and Amazon S3 origins.

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: origins.S3BucketOrigin.withOriginAccessControl(myBucket, {
    customHeaders: {
      Foo: 'bar',
    },
  })},
});
```

## ELBv2 Load Balancer

An Elastic Load Balancing (ELB) v2 load balancer may be used as an origin. In order for a load balancer to serve as an origin, it must be publicly
accessible (`internetFacing` is true). Both Application and Network load balancers are supported.

```ts
declare const vpc: ec2.Vpc;
// Create an application load balancer in a VPC. 'internetFacing' must be 'true'
// for CloudFront to access the load balancer and use it as an origin.
const lb = new elbv2.ApplicationLoadBalancer(this, 'LB', {
  vpc,
  internetFacing: true,
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: new origins.LoadBalancerV2Origin(lb) },
});
```

The origin can also be customized to respond on different ports, have different connection properties, etc.

```ts
declare const loadBalancer: elbv2.ApplicationLoadBalancer;
const origin = new origins.LoadBalancerV2Origin(loadBalancer, {
  connectionAttempts: 3,
  connectionTimeout: Duration.seconds(5),
  readTimeout: Duration.seconds(45),
  responseCompletionTimeout: Duration.seconds(120),
  keepaliveTimeout: Duration.seconds(45),
  protocolPolicy: cloudfront.OriginProtocolPolicy.MATCH_VIEWER,
});
```

Note that `readTimeout` and `keepaliveTimeout` are governed by two separate CloudFront service quotas: `Response timeout per origin`, which defaults to
1-120 seconds, and `Keep-alive timeout per origin`, which defaults to 1-300 seconds. Neither is a hard limit. A value above the default requires an
approved quota increase in the target account, and without one CloudFront rejects it at deploy time.

## From an HTTP endpoint

Origins can also be created from any other HTTP endpoint, given the domain name, and optionally, other origin properties.

```ts
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: new origins.HttpOrigin('www.example.com') },
});
```

You can specify the IP address type for connecting to the origin:

```ts
const origin = new origins.HttpOrigin('www.example.com', {
  ipAddressType: cloudfront.OriginIpAddressType.IPV6, // IPv4, IPv6, or DUALSTACK
});

new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin },
});
```

The `ipAddressType` property allows you to specify whether CloudFront should use IPv4, IPv6, or both (dual-stack) when connecting to your origin.

The origin can be customized with timeout settings to handle different response scenarios:

```ts
new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: {
    origin: new origins.HttpOrigin('api.example.com', {
      readTimeout: Duration.seconds(60),
      responseCompletionTimeout: Duration.seconds(120),
      keepaliveTimeout: Duration.seconds(45),
    }),
  },
});
```

The `responseCompletionTimeout` property specifies the time that a request from CloudFront to the origin can stay open and wait for a response. If the complete response isn't received from the origin by this time, CloudFront ends the connection. Valid values are 1-3600 seconds, and if set, the value must be equal to or greater than the `readTimeout` value.

See the documentation of `aws-cdk-lib/aws-cloudfront` for more information.

## VPC origins

You can use CloudFront to deliver content from applications that are hosted in your virtual private cloud (VPC) private subnets.
You can use Application Load Balancers (ALBs), Network Load Balancers (NLBs), and EC2 instances in private subnets as VPC origins.

Learn more about [Restrict access with VPC origins](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/private-content-vpc-origins.html).

### From an Application Load Balancer

An Application Load Balancer (ALB) can be used as a VPC origin.
It is not needed to be publicly accessible.

``` ts
// Creates a distribution from an Application Load Balancer
declare const vpc: ec2.Vpc;
// Create an application load balancer in a VPC. 'internetFacing' can be 'false'.
const alb = new elbv2.ApplicationLoadBalancer(this, 'ALB', {
  vpc,
  internetFacing: false,
  vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: origins.VpcOrigin.withApplicationLoadBalancer(alb) },
});
```

### From a Network Load Balancer

A Network Load Balancer (NLB) can also be use as a VPC origin.
It is not needed to be publicly accessible.

- A Network Load Balancer must have a security group attached to it.
- Dual-stack Network Load Balancers and Network Load Balancers with TLS listeners can't be added as origins.

``` ts
// Creates a distribution from a Network Load Balancer
declare const vpc: ec2.Vpc;
// Create a network load balancer in a VPC. 'internetFacing' can be 'false'.
const nlb = new elbv2.NetworkLoadBalancer(this, 'NLB', {
  vpc,
  internetFacing: false,
  vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
  securityGroups: [new ec2.SecurityGroup(this, 'NLB-SG', { vpc })],
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: origins.VpcOrigin.withNetworkLoadBalancer(nlb) },
});
```

### From an EC2 instance

An EC2 instance can also be used directly as a VPC origin.
It can be in a private subnet.

``` ts
// Creates a distribution from an EC2 instance
declare const vpc: ec2.Vpc;
// Create an EC2 instance in a VPC. 'subnetType' can be private.
const instance = new ec2.Instance(this, 'Instance', {
  vpc,
  instanceType: ec2.InstanceType.of(ec2.InstanceClass.BURSTABLE3, ec2.InstanceSize.MICRO),
  machineImage: ec2.MachineImage.latestAmazonLinux2023(),
  vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
});
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: { origin: origins.VpcOrigin.withEc2Instance(instance) },
});
```

### Restrict traffic coming to the VPC origin

You may need to update the security group for your VPC private origin (Application Load Balancer, Network Load Balancer, or EC2 instance) to explicitly allow the traffic from your VPC origins.

#### The CloudFront managed prefix list

You can allow the traffic from the CloudFront managed prefix list named **com.amazonaws.global.cloudfront.origin-facing**. For more information, see [Use an AWS-managed prefix list](https://docs.aws.amazon.com/vpc/latest/userguide/working-with-aws-managed-prefix-lists.html#use-aws-managed-prefix-list).

``` ts
declare const alb: elbv2.ApplicationLoadBalancer;

const cfOriginFacing = ec2.PrefixList.fromLookup(this, 'CloudFrontOriginFacing', {
  prefixListName: 'com.amazonaws.global.cloudfront.origin-facing',
});
alb.connections.allowFrom(cfOriginFacing, ec2.Port.HTTP);
```

#### The VPC origin service security group

VPC origin will create a security group named `CloudFront-VPCOrigins-Service-SG`.
It can be further restricted to allow only traffic from your VPC origins.

The id of the security group is not provided by CloudFormation currently.
You can retrieve it dynamically using a custom resource.

``` ts
import * as cr from 'aws-cdk-lib/custom-resources';

declare const vpc: ec2.Vpc;
declare const distribution: cloudfront.Distribution;
declare const alb: elbv2.ApplicationLoadBalancer;

// Call ec2:DescribeSecurityGroups API to retrieve the VPC origins security group.
const getSg = new cr.AwsCustomResource(this, 'GetSecurityGroup', {
  onCreate: {
    service: 'ec2',
    action: 'describeSecurityGroups',
    parameters: {
      Filters: [
        { Name: 'vpc-id', Values: [vpc.vpcId] },
        { Name: 'group-name', Values: ['CloudFront-VPCOrigins-Service-SG'] },
      ],
    },
    physicalResourceId: cr.PhysicalResourceId.of('CloudFront-VPCOrigins-Service-SG'),
  },
  policy: cr.AwsCustomResourcePolicy.fromSdkCalls({ resources: ['*'] }),
});
// The security group may be available after the distributon is deployed
getSg.node.addDependency(distribution);
const sgVpcOrigins = ec2.SecurityGroup.fromSecurityGroupId(
  this,
  'VpcOriginsSecurityGroup',
  getSg.getResponseField('SecurityGroups.0.GroupId'),
);
// Allow connections from the security group
alb.connections.allowFrom(sgVpcOrigins, ec2.Port.HTTP);
```

## Failover Origins (Origin Groups)

You can set up CloudFront with origin failover for scenarios that require high availability.
To get started, you create an origin group with two origins: a primary and a secondary.
If the primary origin is unavailable, or returns specific HTTP response status codes that indicate a failure,
CloudFront automatically switches to the secondary origin.
You achieve that behavior in the CDK using the `OriginGroup` class:

```ts
const myBucket = new s3.Bucket(this, 'myBucket');
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: new origins.OriginGroup({
      primaryOrigin: origins.S3BucketOrigin.withOriginAccessControl(myBucket),
      fallbackOrigin: new origins.HttpOrigin('www.example.com'),
      // optional, defaults to: 500, 502, 503 and 504
      fallbackStatusCodes: [404],
    }),
  },
});
```

### Selection Criteria: Media Quality Based with AWS Elemental MediaPackageV2

You can setup your origin group to be configured for media quality based failover with your AWS Elemental MediaPackageV2 endpoints.
You can achieve this behavior in the CDK, again using the `OriginGroup` class:

```ts
new cloudfront.Distribution(this, 'myDist', {
  defaultBehavior: {
    origin: new origins.OriginGroup({
      primaryOrigin: new origins.HttpOrigin("<AWS Elemental MediaPackageV2 origin 1>"),
      fallbackOrigin: new origins.HttpOrigin("<AWS Elemental MediaPackageV2 origin 2>"),
      fallbackStatusCodes: [404],
      selectionCriteria: cloudfront.OriginSelectionCriteria.MEDIA_QUALITY_BASED,
    }),
  },
});
```

## From an API Gateway REST API

Origins can be created from an API Gateway REST API. It is recommended to use a
[regional API](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-endpoint-types.html) in this case. The origin path will automatically be set as the stage name.

```ts
declare const api: apigateway.RestApi;
new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin: new origins.RestApiOrigin(api) },
});
```

If you want to use a different origin path, you can specify it in the `originPath` property.

```ts
declare const api: apigateway.RestApi;
new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin: new origins.RestApiOrigin(api, { originPath: '/custom-origin-path' }) },
});
```

## From a Lambda Function URL

Lambda Function URLs enable direct invocation of Lambda functions via HTTP(S), without intermediaries. They can be set as CloudFront origins for streamlined function execution behind a CDN, leveraging caching and custom domains.

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';

declare const fn: lambda.Function;
const fnUrl = fn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });

new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { origin: new origins.FunctionUrlOrigin(fnUrl) },
});
```

You can also configure timeout settings for Lambda Function URL origins:

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';

declare const fn: lambda.Function;
const fnUrl = fn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });

new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: {
    origin: new origins.FunctionUrlOrigin(fnUrl, {
      readTimeout: Duration.seconds(30),
      responseCompletionTimeout: Duration.seconds(90),
      keepaliveTimeout: Duration.seconds(45),
    }),
  },
});
```

### Configuring IP Address Type

You can specify which IP protocol CloudFront uses when connecting to your Lambda Function URL origin. By default, CloudFront uses IPv4 only.

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';
import { OriginIpAddressType } from 'aws-cdk-lib/aws-cloudfront';

declare const fn: lambda.Function;
const fnUrl = fn.addFunctionUrl({ authType: lambda.FunctionUrlAuthType.NONE });

// Uses default IPv4 only
new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { 
    origin: new origins.FunctionUrlOrigin(fnUrl)
  },
});

// Explicitly specify IP address type
new cloudfront.Distribution(this, 'Distribution', {
  defaultBehavior: { 
    origin: new origins.FunctionUrlOrigin(fnUrl, {
      ipAddressType: OriginIpAddressType.DUALSTACK, // Use both IPv4 and IPv6
    })
  },
});
```

Supported values for `ipAddressType`:
- `OriginIpAddressType.IPV4` - CloudFront uses IPv4 only to connect to the origin (default)
- `OriginIpAddressType.IPV6` - CloudFront uses IPv6 only to connect to the origin  
- `OriginIpAddressType.DUALSTACK` - CloudFront uses both IPv4 and IPv6 to connect to the origin

### Lambda Function URL with Origin Access Control (OAC)
You can configure the Lambda Function URL with Origin Access Control (OAC) for enhanced security. When using OAC with Signing SIGV4_ALWAYS, it is recommended to set the Lambda Function URL authType to AWS_IAM to ensure proper authorization.

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';
declare const fn: lambda.Function;

const fnUrl = fn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.AWS_IAM,
});

new cloudfront.Distribution(this, 'MyDistribution', {
  defaultBehavior: {
    origin: origins.FunctionUrlOrigin.withOriginAccessControl(fnUrl),
  },
});
```

If you want to explicitly add OAC for more customized access control, you can use the originAccessControl option as shown below.

```ts
import * as lambda from 'aws-cdk-lib/aws-lambda';
declare const fn: lambda.Function;

const fnUrl = fn.addFunctionUrl({
  authType: lambda.FunctionUrlAuthType.AWS_IAM,
});

// Define a custom OAC
const oac = new cloudfront.FunctionUrlOriginAccessControl(this, 'MyOAC', {
  originAccessControlName: 'CustomLambdaOAC',
  signing: cloudfront.Signing.SIGV4_ALWAYS,
});

// Set up Lambda Function URL with OAC in CloudFront Distribution
new cloudfront.Distribution(this, 'MyDistribution', {
  defaultBehavior: {
    origin: origins.FunctionUrlOrigin.withOriginAccessControl(fnUrl, {
      originAccessControl: oac,
    }),
  },
});
```
