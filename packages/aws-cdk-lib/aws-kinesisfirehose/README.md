# Amazon Data Firehose Construct Library

[Amazon Data Firehose](https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html), [formerly known as Amazon Kinesis Data Firehose](https://aws.amazon.com/about-aws/whats-new/2024/02/amazon-data-firehose-formerly-kinesis-data-firehose/),
is a service for fully-managed delivery of real-time streaming data to storage services
such as Amazon S3, Amazon Redshift, Amazon Elasticsearch, Splunk, or any custom HTTP
endpoint or third-party services such as Datadog, Dynatrace, LogicMonitor, MongoDB, New
Relic, and Sumo Logic.

Amazon Data Firehose delivery streams are distinguished from Kinesis data streams in
their models of consumption. Whereas consumers read from a data stream by actively pulling
data from the stream, a delivery stream pushes data to its destination on a regular
cadence. This means that data streams are intended to have consumers that do on-demand
processing, like AWS Lambda or Amazon EC2. On the other hand, delivery streams are
intended to have destinations that are sources for offline processing and analytics, such
as Amazon S3 and Amazon Redshift.

This module is part of the [AWS Cloud Development Kit](https://github.com/aws/aws-cdk)
project. It allows you to define Amazon Data Firehose delivery streams.

## Defining a Delivery Stream

In order to define a Delivery Stream, you must specify a destination. S3, HTTP endpoint, and Datadog can be used as destinations, as covered [below](#destinations).

```ts
const bucket = new s3.Bucket(this, 'Bucket');
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: new firehose.S3Bucket(bucket),
});
```

The above example defines the following resources:

- An S3 bucket
- An Amazon Data Firehose delivery stream with Direct PUT as the source and CloudWatch
  error logging turned on.
- An IAM role which gives the delivery stream permission to write to the S3 bucket.

## Sources

An Amazon Data Firehose delivery stream can accept data from three main sources: Kinesis Data Streams, Managed Streaming for Apache Kafka (MSK), or via a "direct put" (API calls). Currently only Kinesis Data Streams and direct put are supported in the CDK. 

See: [Sending Data to a Delivery Stream](https://docs.aws.amazon.com/firehose/latest/dev/basic-write.html)
in the *Amazon Data Firehose Developer Guide*.

### Kinesis Data Stream

A delivery stream can read directly from a Kinesis data stream as a consumer of the data
stream. Configure this behaviour by passing in a data stream in the `source`
property via the `KinesisStreamSource` class when constructing a delivery stream:

```ts
declare const destination: firehose.IDestination;
const sourceStream = new kinesis.Stream(this, 'Source Stream');

new firehose.DeliveryStream(this, 'Delivery Stream', {
  source: new firehose.KinesisStreamSource(sourceStream),
  destination: destination,
});
```

### Direct Put

Data must be provided via "direct put", ie., by using a `PutRecord` or
`PutRecordBatch` API call. There are a number of ways of doing so, such as:

- Kinesis Agent: a standalone Java application that monitors and delivers files while
  handling file rotation, checkpointing, and retries. See: [Writing to Amazon Data Firehose Using Kinesis Agent](https://docs.aws.amazon.com/firehose/latest/dev/writing-with-agents.html)
  in the *Amazon Data Firehose Developer Guide*.
- AWS SDK: a general purpose solution that allows you to deliver data to a delivery stream
  from anywhere using Java, .NET, Node.js, Python, or Ruby. See: [Writing to Amazon Data Firehose Using the AWS SDK](https://docs.aws.amazon.com/firehose/latest/dev/writing-with-sdk.html)
  in the *Amazon Data Firehose Developer Guide*.
- CloudWatch Logs: subscribe to a log group and receive filtered log events directly into
  a delivery stream. See: [logs-destinations](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-logs-destinations-readme.html).
- Eventbridge: add an event rule target to send events to a delivery stream based on the
  rule filtering. See: [events-targets](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-events-targets-readme.html).
- SNS: add a subscription to send all notifications from the topic to a delivery
  stream. See: [sns-subscriptions](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-sns-subscriptions-readme.html).
- IoT: add an action to an IoT rule to send various IoT information to a delivery stream

## Destinations

Amazon Data Firehose supports multiple AWS and third-party services as destinations, including Amazon S3, Amazon Redshift, and more. You can find the full list of supported destination [here](https://docs.aws.amazon.com/firehose/latest/dev/create-destination.html).

Currently in the AWS CDK, S3, HTTP endpoint, and Datadog are implemented as L2 construct destinations. Other destinations can still be configured using L1 constructs.

### HTTP

Defining a delivery stream with an HTTP destination:

```ts
declare const endpointConfig: firehose.HttpEndpointConfig;
const httpDestination = new firehose.HttpEndpoint({
  endpointConfig,
});
```

### Datadog

Defining a delivery stream with a Datadog destination:

```ts
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';

declare const apiKey: secretsmanager.Secret;
const datadogDestination = new firehose.Datadog({
  apiKey,
  endpoint: firehose.DatadogEndpoint.LOGS_US1,
});
```

### S3

Defining a delivery stream with an S3 bucket destination:

```ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket);

new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

The S3 destination also supports custom dynamic prefixes. `dataOutputPrefix`
will be used for files successfully delivered to S3. `errorOutputPrefix` will be added to
failed records before writing them to S3.

```ts
import { TimeZone } from 'aws-cdk-lib';
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  dataOutputPrefix: 'myFirehose/DeliveredYear=!{timestamp:yyyy}/anyMonth/rand=!{firehose:random-string}',
  errorOutputPrefix: 'myFirehoseFailures/!{firehose:error-output-type}/!{timestamp:yyyy}/anyMonth/!{timestamp:dd}',
  // The time zone of timestamps (default UTC)
  timeZone: TimeZone.ASIA_TOKYO,
});
```

See: [Custom S3 Prefixes](https://docs.aws.amazon.com/firehose/latest/dev/s3-prefixes.html)
in the *Amazon Data Firehose Developer Guide*.

To override default file extension appended by Data Format Conversion or S3 compression features, specify `fileExtension`.

```ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  compression: firehose.Compression.GZIP,
  fileExtension: '.json.gz',
});
```

## Data Format Conversion

Data format conversion allows automatic conversion of inputs from JSON to either Parquet or ORC.
Converting JSON records to columnar formats like Parquet or ORC can help speed up analytical querying while also increasing compression efficiency.
When data format conversion is specified, it automatically enables Snappy compression on the output.

Only S3 Destinations support data format conversion.

An example of defining an S3 destination configured with data format conversion:

```ts
declare const bucket: s3.Bucket;
declare const schemaGlueTable: glue.CfnTable;
const s3Destination = new firehose.S3Bucket(bucket, {
  dataFormatConversion: {
    schemaConfiguration: firehose.SchemaConfiguration.fromCfnTable(schemaGlueTable),
    inputFormat: firehose.InputFormat.OPENX_JSON,
    outputFormat: firehose.OutputFormat.PARQUET,
  }
});
```

When data format conversion is enabled, the Delivery Stream's buffering size must be at least 64 MiB. 
Additionally, the default buffering size is changed from 5 MiB to 128 MiB. This mirrors the Cloudformation behavior.

You can only parse JSON and transform it into either Parquet or ORC:
- to read JSON using OpenX parser, choose `InputFormat.OPENX_JSON`.
- to read JSON using Hive parser, choose `InputFormat.HIVE_JSON`.
- to transform into Parquet, choose `OutputFormat.PARQUET`.
- to transform into ORC, choose `OutputFormat.ORC`.

The following subsections explain how to specify advanced configuration options for each input and output format if the defaults are not desirable

### Input Format: OpenX JSON

Example creation of custom OpenX JSON InputFormat:

```ts
const inputFormat = new firehose.OpenXJsonInputFormat({
  lowercaseColumnNames: false,
  columnToJsonKeyMappings: {"ts": "timestamp"},
  convertDotsInJsonKeysToUnderscores: true,
})
```

### Input Format: Hive JSON

Example creation of custom Hive JSON InputFormat:

```ts
const inputFormat = new firehose.HiveJsonInputFormat({
  timestampParsers: [
    firehose.TimestampParser.fromFormatString('yyyy-MM-dd'),
    firehose.TimestampParser.EPOCH_MILLIS,
  ]
})
```

Hive JSON allows you to specify custom timestamp formats to parse. The syntax of the format string is Joda Time.

To parse timestamps formatted as milliseconds since epoch, use the convenience constant `TimestampParser.EPOCH_MILLIS`.

### Output Format: Parquet

Example of a custom Parquet OutputFormat, with all values changed from the defaults.

```ts
const outputFormat = new firehose.ParquetOutputFormat({
  blockSize: Size.mebibytes(512),
  compression: firehose.ParquetCompression.UNCOMPRESSED,
  enableDictionaryCompression: true,
  maxPadding: Size.bytes(10),
  pageSize: Size.mebibytes(2),
  writerVersion: firehose.ParquetWriterVersion.V2,
})
```

### Output Format: ORC

Example creation of custom ORC OutputFormat, with all values changed from the defaults.

```ts
const outputFormat = new firehose.OrcOutputFormat({
  formatVersion: firehose.OrcFormatVersion.V0_11,
  blockSize: Size.mebibytes(256),
  compression: firehose.OrcCompression.NONE,
  bloomFilterColumns: ['columnA'],
  bloomFilterFalsePositiveProbability: 0.1,
  dictionaryKeyThreshold: 0.7,
  enablePadding: true,
  paddingTolerance: 0.2,
  rowIndexStride: 9000,
  stripeSize: Size.mebibytes(32),
})
```

## Server-side Encryption

Enabling server-side encryption (SSE) requires Amazon Data Firehose to encrypt all data
sent to delivery stream when it is stored at rest. This means that data is encrypted
before being written to the service's internal storage layer and decrypted after it is
received from the internal storage layer. The service manages keys and cryptographic
operations so that sources and destinations do not need to, as the data is encrypted and
decrypted at the boundaries of the service (i.e., before the data is delivered to a
destination). By default, delivery streams do not have SSE enabled.

The Key Management Service keys (KMS keys) used for SSE can either be AWS-owned or
customer-managed. AWS-owned KMS keys are created, owned and managed by AWS for use in
multiple AWS accounts. As a customer, you cannot view, use, track, or manage these keys,
and you are not charged for their use. On the other hand, customer-managed KMS keys are
created and owned within your account and managed entirely by you. As a customer, you are
responsible for managing access, rotation, aliases, and deletion for these keys, and you
are changed for their use.

See: [AWS KMS keys](https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html#kms_keys)
in the *KMS Developer Guide*.

```ts
declare const destination: firehose.IDestination;

// SSE with an AWS-owned key
new firehose.DeliveryStream(this, 'Delivery Stream with AWS Owned Key', {
  encryption: firehose.StreamEncryption.awsOwnedKey(),
  destination: destination,
});
// SSE with an customer-managed key that is created automatically by the CDK
new firehose.DeliveryStream(this, 'Delivery Stream with Customer Managed Key', {
  encryption: firehose.StreamEncryption.customerManagedKey(),
  destination: destination,
});
// SSE with an customer-managed key that is explicitly specified
declare const key: kms.Key;
new firehose.DeliveryStream(this, 'Delivery Stream with Customer Managed and Provided Key', {
  encryption: firehose.StreamEncryption.customerManagedKey(key),
  destination: destination,
});
```

See: [Data Protection](https://docs.aws.amazon.com/firehose/latest/dev/encryption.html)
in the *Amazon Data Firehose Developer Guide*.

## Monitoring

Amazon Data Firehose is integrated with CloudWatch, so you can monitor the performance of
your delivery streams via logs and metrics.

### Logs

Amazon Data Firehose will send logs to CloudWatch when data transformation or data
delivery fails. The CDK will enable logging by default and create a CloudWatch LogGroup
and LogStream with default settings for your Delivery Stream.

When creating a destination, you can provide an `ILoggingConfig`, which can either be an `EnableLogging` or `DisableLogging` instance.
If you use `EnableLogging`, the CDK will create a CloudWatch LogGroup and LogStream with all CloudFormation default settings for you, or you can optionally 
specify your own log group to be used for capturing and storing log events. For example:

```ts
import * as logs from 'aws-cdk-lib/aws-logs';

const logGroup = new logs.LogGroup(this, 'Log Group');
declare const bucket: s3.Bucket;
const destination = new firehose.S3Bucket(bucket, {
  loggingConfig: new firehose.EnableLogging(logGroup),
});

new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: destination,
});
```

Logging can also be disabled:

```ts
declare const bucket: s3.Bucket;
const destination = new firehose.S3Bucket(bucket, {
  loggingConfig: new firehose.DisableLogging(),
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: destination,
});
```

See: [Monitoring using CloudWatch Logs](https://docs.aws.amazon.com/firehose/latest/dev/monitoring-with-cloudwatch-logs.html)
in the *Amazon Data Firehose Developer Guide*.

### Metrics

Amazon Data Firehose sends metrics to CloudWatch so that you can collect and analyze the
performance of the delivery stream, including data delivery, data ingestion, data
transformation, format conversion, API usage, encryption, and resource usage. You can then
use CloudWatch alarms to alert you, for example, when data freshness (the age of the
oldest record in the delivery stream) exceeds the buffering limit (indicating that data is
not being delivered to your destination), or when the rate of incoming records exceeds the
limit of records per second (indicating data is flowing into your delivery stream faster
than it is configured to process).

CDK provides methods for accessing delivery stream metrics with default configuration,
such as `metricIncomingBytes`, and `metricIncomingRecords` (see [`IDeliveryStream`](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-cdk-lib.aws_kinesisfirehose.IDeliveryStream.html)
for a full list). CDK also provides a generic `metric` method that can be used to produce
metric configurations for any metric provided by Amazon Data Firehose; the configurations
are pre-populated with the correct dimensions for the delivery stream.

```ts
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';

declare const deliveryStream: firehose.DeliveryStream;

// Alarm that triggers when the per-second average of incoming bytes exceeds 90% of the current service limit
const incomingBytesPercentOfLimit = new cloudwatch.MathExpression({
  expression: 'incomingBytes / 300 / bytePerSecLimit',
  usingMetrics: {
    incomingBytes: deliveryStream.metricIncomingBytes({ statistic: cloudwatch.Statistic.SUM }),
    bytePerSecLimit: deliveryStream.metric('BytesPerSecondLimit'),
  },
});

new cloudwatch.Alarm(this, 'Alarm', {
  metric: incomingBytesPercentOfLimit,
  threshold: 0.9,
  evaluationPeriods: 3,
});
```

See: [Monitoring Using CloudWatch Metrics](https://docs.aws.amazon.com/firehose/latest/dev/monitoring-with-cloudwatch-metrics.html)
in the *Amazon Data Firehose Developer Guide*.

## Compression

Your data can automatically be compressed when it is delivered to S3 as either a final or
an intermediary/backup destination. Supported compression formats are: gzip, Snappy, 
Hadoop-compatible Snappy, and ZIP, except for Redshift destinations, where Snappy
(regardless of Hadoop-compatibility) and ZIP are not supported. By default, data is
delivered to S3 without compression.

```ts
// Compress data delivered to S3 using Snappy
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  compression: firehose.Compression.SNAPPY,
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

## Buffering

Incoming data is buffered before it is delivered to the specified destination. The
delivery stream will wait until the amount of incoming data has exceeded some threshold
(the "buffer size") or until the time since the last data delivery occurred exceeds some
threshold (the "buffer interval"), whichever happens first. You can configure these
thresholds based on the capabilities of the destination and your use-case. By default, the
buffer size is 5 MiB and the buffer interval is 5 minutes.

```ts
// Increase the buffer interval and size to 10 minutes and 8 MiB, respectively
declare const bucket: s3.Bucket;
const destination = new firehose.S3Bucket(bucket, {
  bufferingInterval: Duration.minutes(10),
  bufferingSize: Size.mebibytes(8),
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: destination,
});
```

See: [Data Delivery Frequency](https://docs.aws.amazon.com/firehose/latest/dev/basic-deliver.html#frequency)
in the *Amazon Data Firehose Developer Guide*.

Zero buffering, where Amazon Data Firehose stream can be configured to not buffer data before delivery, is supported by
setting the "buffer interval" to 0.

```ts
// Setup zero buffering
declare const bucket: s3.Bucket;
const destination = new firehose.S3Bucket(bucket, {
  bufferingInterval: Duration.seconds(0),
});
new firehose.DeliveryStream(this, 'ZeroBufferDeliveryStream', {
  destination: destination,
});
```

See: [Buffering Hints](https://docs.aws.amazon.com/firehose/latest/dev/buffering-hints.html).

## Destination Encryption

Your data can be automatically encrypted when it is delivered to S3 as a final or an
intermediary/backup destination. Amazon Data Firehose supports Amazon S3 server-side
encryption with AWS Key Management Service (AWS KMS) for encrypting delivered data in
Amazon S3. You can choose to not encrypt the data or to encrypt with a key from the list
of AWS KMS keys that you own. For more information,
see [Protecting Data Using Server-Side Encryption with AWS KMS–Managed Keys (SSE-KMS)](https://docs.aws.amazon.com/AmazonS3/latest/dev/UsingKMSEncryption.html).
By default, encryption isn’t directly enabled on the delivery stream; instead, it uses the default encryption settings of the destination S3 bucket.

```ts
declare const bucket: s3.Bucket;
declare const key: kms.Key;
const destination = new firehose.S3Bucket(bucket, {
  encryptionKey: key,
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: destination,
});
```

## Backup

A delivery stream can be configured to back up data to S3 that it attempted to deliver to
the configured destination. Backed up data can be all the data that the delivery stream
attempted to deliver or just data that it failed to deliver (Redshift and S3 destinations
can only back up all data). CDK can create a new S3 bucket where it will back up data, or
you can provide a bucket where data will be backed up. You can also provide a prefix under
which your backed-up data will be placed within the bucket. By default, source data is not
backed up to S3.

```ts
// Enable backup of all source records (to an S3 bucket created by CDK).
declare const bucket: s3.Bucket;
new firehose.DeliveryStream(this, 'Delivery Stream Backup All', {
  destination:
    new firehose.S3Bucket(bucket, {
      s3Backup: {
        mode: firehose.BackupMode.ALL,
      },
    }),
});
// Explicitly provide an S3 bucket to which all source records will be backed up.
declare const backupBucket: s3.Bucket;
new firehose.DeliveryStream(this, 'Delivery Stream Backup All Explicit Bucket', {
  destination: 
    new firehose.S3Bucket(bucket, {
      s3Backup: {
        bucket: backupBucket,
      },
    }),
});
// Explicitly provide an S3 prefix under which all source records will be backed up.
new firehose.DeliveryStream(this, 'Delivery Stream Backup All Explicit Prefix', {
  destination:
    new firehose.S3Bucket(bucket, {
      s3Backup: {
        mode: firehose.BackupMode.ALL,
        dataOutputPrefix: 'mybackup',
      },
    }),
});
```

If any Data Processing or Transformation is configured on your Delivery Stream, the source
records will be backed up in their original format.

## Data Processing/Transformation

Data can be transformed before being delivered to destinations. There are two types of
data processing for delivery streams: record transformation with AWS Lambda, and record
format conversion using a schema stored in an AWS Glue table. If both types of data
processing are configured, then the Lambda transformation is performed first. By default,
no data processing occurs.

This construct library currently only supports data
transformation with AWS Lambda and some built-in data processors.
See [#15501](https://github.com/aws/aws-cdk/issues/15501)
to track the status of adding support for record format conversion.

### Data transformation with AWS Lambda

To transform the data, Amazon Data Firehose will call a Lambda function that you provide
and deliver the data returned in place of the source record. The function must return a
result that contains records in a specific format, including the following fields:

- `recordId` -- the ID of the input record that corresponds the results.
- `result` -- the status of the transformation of the record: "Ok" (success), "Dropped"
  (not processed intentionally), or "ProcessingFailed" (not processed due to an error).
- `data` -- the transformed data, Base64-encoded.
- `metadata` -- the metadata used by dynamic partitioning.

The data is buffered up to 1 minute and up to 3 MiB by default before being sent to the
function, but can be configured using `bufferInterval` and `bufferSize`
in the processor configuration (see: [Buffering](#buffering)). If the function invocation
fails due to a network timeout or because of hitting an invocation limit, the invocation
is retried 3 times by default, but can be configured using `retries` in the processor
configuration.

```ts
// Provide a Lambda function that will transform records before delivery, with custom
// buffering and retry configuration
const lambdaFunction = new lambda.Function(this, 'Processor', {
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
  code: lambda.Code.fromAsset(path.join(__dirname, 'process-records')),
});
const lambdaProcessor = new firehose.LambdaFunctionProcessor(lambdaFunction, {
  bufferInterval: Duration.minutes(5),
  bufferSize: Size.mebibytes(5),
  retries: 5,
});
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  processors: [lambdaProcessor],
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

[Example Lambda data processor performing the identity transformation.](../../@aws-cdk-testing/framework-integ/test/aws-kinesisfirehose/test/integ.s3-bucket.lit.ts)

See: [Data Transformation](https://docs.aws.amazon.com/firehose/latest/dev/data-transformation.html)
in the *Amazon Data Firehose Developer Guide*.

### Add a new line delimiter when delivering data to Amazon S3

You can specify the `AppendDelimiterToRecordProcessor` built-in processor to add a new line delimiter between records in objects that are delivered to Amazon S3. This can be helpful for parsing objects in Amazon S3.
For details, see [Use Amazon S3 bucket prefix to deliver data](https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning-s3bucketprefix.html).

```ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  processors: [
    new firehose.AppendDelimiterToRecordProcessor(),
  ],
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

### Decompress and extract message of CloudWatch Logs

CloudWatch Logs events are sent to Firehose in compressed gzip format. If you want to deliver decompressed log events to Firehose destinations, you can use the `DecompressionProcessor` to automatically decompress CloudWatch Logs.
For details, see [Send CloudWatch Logs to Firehose](https://docs.aws.amazon.com/firehose/latest/dev/writing-with-cloudwatch-logs.html).

You may also needed to specify `AppendDelimiterToRecordProcessor`
because decompressed log events record has no trailing newline.

```ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  processors: [
    new firehose.DecompressionProcessor(),
    new firehose.AppendDelimiterToRecordProcessor(),
  ],
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

When you enable decompression, you have the option to also enable message extraction. When using message extraction, Firehose filters out all metadata, such as owner, loggroup, logstream, and others from the decompressed CloudWatch Logs records and delivers only the content inside the message fields.

```ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  processors: [
    new firehose.DecompressionProcessor(),
    new firehose.CloudWatchLogProcessor({ dataMessageExtraction: true }),
  ],
});
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: s3Destination,
});
```

## Dynamic Partitioning

Dynamic partitioning enables you to continuously partition streaming data in Firehose by using keys within data (for example, `customer_id` or `transaction_id`) and then deliver the data grouped by these keys into corresponding Amazon S3 prefixes.

For details, see [Partition streaming data in Amazon Data Firehose](https://docs.aws.amazon.com/firehose/latest/dev/dynamic-partitioning.html).

### Create partitioning keys with inline parsing

To enable dynamic partitioning with inline parsing, specify `MetadataExtractionProcessor` data processor in `processors`.

Only supported mechanism currently is [jq 1.6 parser](https://stedolan.github.io/jq/).

The partition keys can be referred as `!{partitionKeyFromQuery:key}` in `dataOutputPrefix`.

``` ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  dynamicPartitioning: { enabled: true },
  processors: [
    firehose.MetadataExtractionProcessor.jq16({
      customer_id: '.customer_id',
      device: '.type.device',
      year: '.event_timestamp|strftime("%Y")',
    }),
  ],
  dataOutputPrefix: '!{partitionKeyFromQuery:year}/!{partitionKeyFromQuery:device}/!{partitionKeyFromQuery:customer_id}/',
});
new firehose.DeliveryStream(this, 'DeliveryStream', {
  destination: s3Destination,
});
```

### Create partitioning keys with an AWS Lambda function

For compressed or encrypted data records, or data that is in any file format other than JSON, you can use the `LambdaFunctionDataProcessor` with your own custom code to decompress, decrypt, or transform the records in order to extract and return the data fields needed for partitioning.

The lambda function must return `metadata` in your result records.

The partition keys can be referred as `!{partitionKeyFromLambda:key}` in `dataOutputPrefix`.

``` ts
declare const bucket: s3.Bucket;
declare const lambdaFunction: lambda.Function;
const s3Destination = new firehose.S3Bucket(bucket, {
  dynamicPartitioning: { enabled: true },
  processors: [
    new firehose.LambdaFunctionProcessor(lambdaFunction),
  ],
  dataOutputPrefix: '!{partitionKeyFromLambda:year}/!{partitionKeyFromLambda:device}/!{partitionKeyFromLambda:customer_id}/',
});
new firehose.DeliveryStream(this, 'DeliveryStream', {
  destination: s3Destination,
});
```

### Multi record deaggregation

To apply dynamic partitioning to aggregated data (for example, multiple events, logs, or records aggregated into a single PutRecord and PutRecordBatch API call), use `RecordDeAggregationProcessor`.

When the input data is JSON objects on a single line with no delimiter or newline-delimited (JSONL), specify `RecordDeAggregationProcessor.json()`.

``` ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  dynamicPartitioning: { enabled: true },
  processors: [
    firehose.RecordDeAggregationProcessor.json(),
  ],
});
```

You can also specify custom delimiter using `RecordDeAggregationProcessor.delimited()`.

``` ts
declare const bucket: s3.Bucket;
const s3Destination = new firehose.S3Bucket(bucket, {
  dynamicPartitioning: { enabled: true },
  processors: [
    firehose.RecordDeAggregationProcessor.delimited('####'),
  ],
});
```

Record deaggregation by JSON or by delimiter is capped at 500 per record.

## Specifying an IAM role

The DeliveryStream class automatically creates IAM service roles with all the minimum
necessary permissions for Amazon Data Firehose to access the resources referenced by your
delivery stream. One service role is created for the delivery stream that allows Amazon
Data Firehose to read from a Kinesis data stream (if one is configured as the delivery
stream source) and for server-side encryption. Note that if the DeliveryStream is created 
without specifying a `source` or `encryptionKey`, this role is not created as it is not needed.

Another service role is created for each destination, which gives Amazon Data Firehose write 
access to the destination resource, as well as the ability to invoke data transformers and 
read schemas for record format conversion. If you wish, you may specify your own IAM role for 
either the delivery stream or the destination service role, or both. It must have the correct 
trust policy (it must allow Amazon Data Firehose to assume it) or delivery stream creation or 
data delivery will fail. Other required permissions to destination resources, encryption keys, etc.,
will be provided automatically.

```ts
// Create service roles for the delivery stream and destination.
// These can be used for other purposes and granted access to different resources.
// They must include the Amazon Data Firehose service principal in their trust policies.
// Two separate roles are shown below, but the same role can be used for both purposes.
const deliveryStreamRole = new iam.Role(this, 'Delivery Stream Role', {
  assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
});
const destinationRole = new iam.Role(this, 'Destination Role', {
  assumedBy: new iam.ServicePrincipal('firehose.amazonaws.com'),
});

// Specify the roles created above when defining the destination and delivery stream.
declare const bucket: s3.Bucket;
const destination = new firehose.S3Bucket(bucket, { role: destinationRole });
new firehose.DeliveryStream(this, 'Delivery Stream', {
  destination: destination,
  role: deliveryStreamRole,
});
```

See [Controlling Access](https://docs.aws.amazon.com/firehose/latest/dev/controlling-access.html)
in the *Amazon Data Firehose Developer Guide*.

## Granting application access to a delivery stream

IAM roles, users or groups which need to be able to work with delivery streams should be
granted IAM permissions.

Any object that implements the `IGrantable` interface (i.e., has an associated principal)
can be granted permissions to a delivery stream by calling:

- `grantPutRecords(principal)` - grants the principal the ability to put records onto the
  delivery stream
- `grant(principal, ...actions)` - grants the principal permission to a custom set of
  actions

```ts
const lambdaRole = new iam.Role(this, 'Role', {
  assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
});

// Give the role permissions to write data to the delivery stream
declare const deliveryStream: firehose.DeliveryStream;
deliveryStream.grantPutRecords(lambdaRole);
```

The following write permissions are provided to a service principal by the 
`grantPutRecords()` method:

- `firehose:PutRecord`
- `firehose:PutRecordBatch`

## Granting a delivery stream access to a resource

Conversely to the above, Amazon Data Firehose requires permissions in order for delivery
streams to interact with resources that you own. For example, if an S3 bucket is specified
as a destination of a delivery stream, the delivery stream must be granted permissions to
put and get objects from the bucket. When using the built-in AWS service destinations, the CDK grants the
permissions automatically. However, custom or third-party destinations may require custom
permissions. In this case, use the delivery stream as an `IGrantable`, as follows:

```ts
const fn = new lambda.Function(this, 'Function', {
  code: lambda.Code.fromInline('exports.handler = (event) => {}'),
  runtime: lambda.Runtime.NODEJS_LATEST,
  handler: 'index.handler',
});

declare const deliveryStream: firehose.DeliveryStream;
fn.grantInvoke(deliveryStream);
```
