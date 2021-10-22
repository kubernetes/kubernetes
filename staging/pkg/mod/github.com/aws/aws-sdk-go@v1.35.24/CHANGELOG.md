Release v1.35.24 (2020-11-09)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `service/dynamodb`: Updates service API, documentation, and paginators
  * This release adds supports for exporting Amazon DynamoDB table data to Amazon S3 to perform analytics at any scale.
* `service/ecs`: Updates service API and documentation
  * This release provides native support for specifying Amazon FSx for Windows File Server file systems as volumes in your Amazon ECS task definitions.
* `service/es`: Updates service API, documentation, and paginators
  * Adding support for package versioning in Amazon Elasticsearch Service
* `service/fsx`: Updates service API, documentation, paginators, and examples
* `service/iotanalytics`: Updates service API and documentation
* `service/macie2`: Updates service API and documentation
* `service/s3`: Updates service API, documentation, and examples
  * S3 Intelligent-Tiering adds support for Archive and Deep Archive Access tiers; S3 Replication adds replication metrics and failure notifications, brings feature parity for delete marker replication
* `service/ssm`: Updates service API and documentation
  * add a new filter to allow customer to filter automation executions by using resource-group which used for execute automation
* `service/storagegateway`: Updates service API, documentation, and paginators
  * Added bandwidth rate limit schedule for Tape and Volume Gateways

Release v1.35.23 (2020-11-06)
===

### Service Client Updates
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Network card support with four new attributes: NetworkCardIndex, NetworkPerformance, DefaultNetworkCardIndex, and MaximumNetworkInterfaces, added to the DescribeInstanceTypes API.
* `service/iotsitewise`: Updates service API and documentation
* `service/medialive`: Updates service API and documentation
  * Support for SCTE35 ad markers in OnCuePoint style in RTMP outputs.
* `service/ssm`: Updates service documentation
  * Documentation updates for Systems Manager

Release v1.35.22 (2020-11-05)
===

### Service Client Updates
* `service/appmesh`: Updates service API, documentation, paginators, and examples
* `service/dynamodb`: Updates service API and documentation
  * This release adds a new ReplicaStatus INACCESSIBLE_ENCRYPTION_CREDENTIALS for the Table description, indicating when a key used to encrypt a regional replica table is not accessible.
* `service/ec2`: Updates service API and documentation
  * Documentation updates for EC2.
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service now provides the ability to define a custom endpoint for your domain and link an SSL certificate from ACM, making it easier to refer to Kibana and the domain endpoint.
* `service/eventbridge`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * With this release, customers can now reprocess past events by storing the events published on event bus in an encrypted archive.
* `service/frauddetector`: Updates service API and documentation
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now supports providing user context in your query requests, Tokens can be JSON or JWT format. This release also introduces support for Confluence cloud datasources.
* `service/lambda`: Updates service API and documentation
  * Support Amazon MQ as an Event Source.
* `service/rds`: Updates service API and documentation
  * Supports a new parameter to set the max allocated storage in gigabytes for the CreateDBInstanceReadReplica API.

Release v1.35.21 (2020-11-04)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Capacity Rebalance helps you manage and maintain workload availability during Spot interruptions by proactively augmenting your Auto Scaling group with a new instance before interrupting an old one.
* `service/ec2`: Updates service API and documentation
  * Added support for Client Connect Handler for AWS Client VPN. Fleet supports launching replacement instances in response to Capacity Rebalance recommendation.
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service now supports native SAML authentication that seamlessly integrates with the customers' existing SAML 2.0 Identity Provider (IdP).
* `service/iot`: Updates service API, documentation, and paginators
  * Updated API documentation and added paginator for AWS Iot Registry ListThingPrincipals API.
* `service/meteringmarketplace`: Updates service API and documentation
  * Adding Vendor Tagging Support in MeterUsage and BatchMeterUsage API.
* `service/monitoring`: Updates service documentation
  * Documentation updates for monitoring
* `service/mq`: Updates service API and documentation
  * Amazon MQ introduces support for RabbitMQ, a popular message-broker with native support for AMQP 0.9.1. You can now create fully-managed RabbitMQ brokers in the cloud.
* `service/servicecatalog`: Updates service API and documentation
  * Service Catalog API ListPortfolioAccess can now support a maximum PageSize of 100.
* `service/transcribe-streaming`: Updates service API
* `service/xray`: Updates service API, documentation, and paginators
  * Releasing new APIs GetInsightSummaries, GetInsightEvents, GetInsight, GetInsightImpactGraph and updating GetTimeSeriesServiceStatistics API for AWS X-Ray Insights feature

Release v1.35.20 (2020-11-02)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds support for the following features: 1. P4d instances based on NVIDIA A100 GPUs.  2. NetworkCardIndex attribute to support multiple network cards.

Release v1.35.19 (2020-10-30)
===

### Service Client Updates
* `service/braket`: Updates service API and documentation
* `service/dms`: Updates service API and documentation
  * Adding DocDbSettings to support DocumentDB as a source.
* `service/elasticache`: Updates service documentation
  * Documentation updates for AWS ElastiCache
* `service/imagebuilder`: Updates service API and documentation
* `service/macie2`: Updates service API and documentation
* `service/medialive`: Updates service API and documentation
  * Support for HLS discontinuity tags in the child manifests. Support for incomplete segment behavior in the media output. Support for automatic input failover condition settings.
* `service/sns`: Updates service documentation
  * Documentation updates for Amazon SNS

Release v1.35.18 (2020-10-29)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Support disabling the default execute-api endpoint for REST APIs.
* `service/codeartifact`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Support for Appliance mode on Transit Gateway that simplifies deployment of stateful network appliances. Added support for AWS Client VPN Self-Service Portal.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/marketplacecommerceanalytics`: Updates service documentation
  * Documentation updates for marketplacecommerceanalytics to specify four data sets which are deprecated.
* `service/sesv2`: Updates service API, documentation, and paginators
* `service/storagegateway`: Updates service API and documentation
  * Adding support for access based enumeration on SMB file shares, file share visibility on SMB file shares, and file upload notifications for all file shares

Release v1.35.17 (2020-10-28)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * AWS Nitro Enclaves general availability. Added support to RunInstances for creating enclave-enabled EC2 instances. New APIs to associate an ACM certificate with an IAM role, for enclave consumption.
* `service/iot`: Updates service API and documentation
  * This release adds support for GG-Managed Job Namespace
* `service/workmail`: Updates service documentation
  * Documentation update for Amazon WorkMail

Release v1.35.16 (2020-10-27)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * AWS Glue machine learning transforms now support encryption-at-rest for labels and trained models.

Release v1.35.15 (2020-10-26)
===

### Service Client Updates
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now supports indexing data from Confluence Server.
* `service/neptune`: Updates service API, documentation, and paginators
  * This feature enables custom endpoints for Amazon Neptune clusters. Custom endpoints simplify connection management when clusters contain instances with different capacities and configuration settings.
* `service/sagemaker`: Updates service API, documentation, and paginators
  * This release enables customers to bring custom images for use with SageMaker Studio notebooks.

Release v1.35.14 (2020-10-23)
===

### Service Client Updates
* `service/macie2`: Updates service documentation
* `service/mediatailor`: Updates service API and documentation
* `service/quicksight`: Updates service API and documentation
  * Support description on columns.

Release v1.35.13 (2020-10-22)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service documentation
* `service/appflow`: Updates service API and documentation
* `service/servicecatalog`: Updates service documentation
  * Documentation updates for servicecatalog
* `service/sns`: Updates service API and documentation
  * SNS now supports a new class of topics: FIFO (First-In-First-Out). FIFO topics provide strictly-ordered, deduplicated, filterable, encryptable, many-to-many messaging at scale.

Release v1.35.12 (2020-10-21)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * CloudFront adds support for managing the public keys for signed URLs and signed cookies directly in CloudFront (it no longer requires the AWS root account).
* `service/ec2`: Updates service API and documentation
  * instance-storage-info nvmeSupport added to DescribeInstanceTypes API
* `service/globalaccelerator`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * AWS Glue crawlers now support incremental crawls for the Amazon Simple Storage Service (Amazon S3) data source.
* `service/kendra`: Updates service API and documentation
  * This release adds custom data sources: a new data source type that gives you full control of the documents added, modified or deleted during a data source sync while providing run history metrics.
* `service/organizations`: Updates service documentation
  * AWS Organizations renamed the 'master account' to 'management account'.

### SDK Bugs
* `aws/credentials`: Fixed a race condition checking if credentials are expired. ([#3448](https://github.com/aws/aws-sdk-go/issues/3448))
  * Fixes [#3524](https://github.com/aws/aws-sdk-go/issues/3524)
* `internal/ini`: Fixes ini file parsing for cases when Right Hand Value is missed in the last statement of the ini file ([#3596](https://github.com/aws/aws-sdk-go/pull/3596)) 
  * related to [#2800](https://github.com/aws/aws-sdk-go/issues/2800)

Release v1.35.11 (2020-10-20)
===

### Service Client Updates
* `service/appsync`: Updates service documentation
* `service/batch`: Updates service API and documentation
  * Adding evaluateOnExit to job retry strategies.
* `service/elasticbeanstalk`: Updates service API
  * EnvironmentStatus enum update to include Aborting, LinkingFrom and LinkingTo

Release v1.35.10 (2020-10-19)
===

### Service Client Updates
* `service/backup`: Updates service documentation
* `service/cloudfront`: Updates service API and documentation
  * Amazon CloudFront adds support for Origin Shield.
* `service/docdb`: Updates service documentation
  * Documentation updates for docdb
* `service/servicecatalog`: Updates service API and documentation
  * An Admin can now update the launch role associated with a Provisioned Product. Admins and End Users can now view the launch role associated with a Provisioned Product.
* `service/ssm`: Updates service API and documentation
  * This Patch Manager release now supports Common Vulnerabilities and Exposure (CVE) Ids for missing packages via the DescribeInstancePatches API.

Release v1.35.9 (2020-10-16)
===

### Service Client Updates
* `service/medialive`: Updates service API, documentation, and paginators
  * The AWS Elemental MediaLive APIs and SDKs now support the ability to transfer the ownership of MediaLive Link devices across AWS accounts.
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations.

### SDK Bugs
* `s3control`: Fixes bug in SDK that caused GetAccessPointPolicy, DeleteAccessPointPolicy, and PutAccessPointPolicy operations to not route properly for S3 on Outposts. ([#3599](https://github.com/aws/aws-sdk-go/pull/3599))
  * Fixes [#3598](https://github.com/aws/aws-sdk-go/issues/3598).

Release v1.35.8 (2020-10-15)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service API and documentation
* `service/budgets`: Updates service API, documentation, and paginators
  * This release introduces AWS Budgets Actions, allowing you to define an explicit response(or set of responses)  to take when your budget exceeds it's action threshold.
* `service/ce`: Updates service API and documentation
* `service/dms`: Updates service API and documentation
  * When creating Endpoints, Replication Instances, and Replication Tasks, the feature provides you the option to specify friendly name to the resources.
* `service/glue`: Updates service documentation
  * API Documentation updates for Glue Get-Plan API
* `service/groundstation`: Updates service API and documentation
* `service/iot`: Updates service API and documentation
  * Add new variable, lastStatusChangeDate, to DescribeDomainConfiguration  API
* `service/macie2`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Return tags for all resources in the output of DescribeDBInstances, DescribeDBSnapshots, DescribeDBClusters, and DescribeDBClusterSnapshots API operations.
* `service/rekognition`: Updates service API and documentation
  * This SDK Release introduces new API (DetectProtectiveEquipment) for Amazon Rekognition. This release also adds ServiceQuotaExceeded exception to Amazon Rekognition IndexFaces API.
* `service/ssm`: Updates service API and documentation
  * This Patch Manager release now supports searching for available packages from Amazon Linux and Amazon Linux 2 via the DescribeAvailablePatches API.
* `service/transfer`: Updates service API and documentation
  * Add support to associate VPC Security Groups at server creation.
* `service/workmail`: Updates service API and documentation
  * Add CreateOrganization and DeleteOrganization API operations.
* `service/workspaces`: Updates service documentation
  * Documentation updates for WorkSpaces
* `service/xray`: Updates service API, documentation, and paginators
  * Enhancing CreateGroup, UpdateGroup, GetGroup and GetGroups APIs to support configuring X-Ray Insights Notifications. Adding TraceLimit information into X-Ray BatchGetTraces API response.

### SDK Bugs
* `s3control`: Fixes bug in SDK that caused input for certain s3control operation to be modified, when using ARNs. ([#3595](https://github.com/aws/aws-sdk-go/pull/3595))
  * Fixes [#3583](https://github.com/aws/aws-sdk-go/issues/3583).

Release v1.35.7 (2020-10-09)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/eks`: Updates service API
* `service/medialive`: Updates service API and documentation
  * WAV audio output. Extracting ancillary captions in MP4 file inputs. Priority on channels feeding a multiplex (higher priority channels will tend to have higher video quality).
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * This new API takes either a ProvisonedProductId or a ProvisionedProductName, along with a list of 1 or more output keys and responds with the (key,value) pairs of those outputs.
* `service/snowball`: Updates service API and documentation
  * We added new APIs to allow customers to better manage their device shipping. You can check if your shipping label expired, generate a new label, and tell us that you received or shipped your job.

Release v1.35.6 (2020-10-08)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * AWS EC2 RevokeSecurityGroupIngress and RevokeSecurityGroupEgress APIs will return IpPermissions which do not match with any existing IpPermissions for security groups in default VPC and EC2-Classic.
* `service/eventbridge`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Amazon EventBridge (formerly called CloudWatch Events) adds support for target Dead-letter Queues and custom retry policies.
* `service/rds`: Updates service API and documentation
  * Supports a new parameter to set the max allocated storage in gigabytes for restore database instance from S3 and restore database instance to a point in time APIs.
* `service/rekognition`: Updates service API and documentation
  * This release provides location information for the manifest validation files.
* `service/sagemaker`: Updates service API and documentation
  * This release enables Sagemaker customers to convert Tensorflow and PyTorch models to CoreML (ML Model) format.
* `service/sns`: Updates service documentation
  * Documentation updates for SNS.

Release v1.35.5 (2020-10-07)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/compute-optimizer`: Updates service API and documentation
* `service/elasticache`: Updates service API, documentation, and paginators
  * This release introduces User and UserGroup to allow customers to have access control list of the Redis resources for AWS ElastiCache. This release also adds support for Outposts  for AWS ElastiCache.
* `service/mediapackage`: Updates service API and documentation
  * AWS Elemental MediaPackage provides access logs that capture detailed information about requests sent to a customer's MediaPackage channel.

### SDK Bugs
* `aws/credentials`: Monotonic clock readings will now be cleared when setting credential expiry time. ([#3573](https://github.com/aws/aws-sdk-go/pull/3573))
  * Prevents potential issues when the host system is hibernated / slept and the monotonic clock readings don't match the wall-clock time.

Release v1.35.4 (2020-10-06)
===

### Service Client Updates
* `service/dms`: Updates service API and documentation
  * Added new S3 endpoint settings to allow partitioning CDC data by date for S3 as target. Exposed some Extra Connection Attributes as endpoint settings for relational databases as target.
* `service/ec2`: Updates service API and documentation
  * This release supports returning additional information about local gateway virtual interfaces, and virtual interface groups.
* `service/kinesisanalyticsv2`: Updates service API and documentation
* `service/marketplace-catalog`: Updates service API and documentation

Release v1.35.3 (2020-10-05)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * This release adds a new ReplicaStatus REGION DISABLED for the Table description. This state indicates that the AWS Region for the replica is inaccessible because the AWS Region is disabled.
* `service/glue`: Updates service API and documentation
  * AWS Glue crawlers now support Amazon DocumentDB (with MongoDB compatibility) and MongoDB collections. You can choose to crawl the entire data set or only a small sample to reduce crawl time.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for AVC-I and VC3 encoding in the MXF OP1a container, Nielsen non-linear watermarking, and InSync FrameFormer frame rate conversion.
* `service/sagemaker`: Updates service API and documentation
  * This release adds support for launching Amazon SageMaker Studio in your VPC. Use AppNetworkAccessType in CreateDomain API to disable access to public internet and restrict the network traffic to VPC.
* `service/streams.dynamodb`: Updates service documentation

Release v1.35.2 (2020-10-02)
===

### Service Client Updates
* `service/batch`: Updates service API, documentation, and examples
  * Support tagging for Batch resources (compute environment, job queue, job definition and job) and tag based access control on Batch APIs
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/personalize-events`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Adds the NCHAR Character Set ID parameter to the CreateDbInstance API for RDS Oracle.
* `service/s3`: Updates service API and documentation
  * Amazon S3 Object Ownership is a new S3 feature that enables bucket owners to automatically assume ownership of objects that are uploaded to their buckets by other AWS Accounts.
* `service/servicediscovery`: Updates service API and documentation

Release v1.35.1 (2020-10-01)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/elasticmapreduce`: Updates service documentation
  * Documentation updates for elasticmapreduce
* `service/glue`: Updates service API and documentation
  * Adding additional optional map parameter to get-plan api
* `service/kafka`: Updates service API and documentation
* `service/quicksight`: Updates service API
  * QuickSight now supports connecting to AWS Timestream data source
* `service/wafv2`: Updates service API and documentation

Release v1.35.0 (2020-09-30)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/datasync`: Updates service API and documentation
* `service/directconnect`: Updates service documentation
  * Documentation updates for AWS Direct Connect.
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR customers can now use EC2 placement group to influence the placement of master nodes in a high-availability (HA) cluster across distinct underlying hardware to improve cluster availability.
* `service/imagebuilder`: Updates service API and documentation
* `service/iot`: Updates service API and documentation
  * AWS IoT Rules Engine adds Timestream action. The Timestream rule action lets you stream time-series data from IoT sensors and applications to Amazon Timestream databases for time series analysis.
* `service/mediaconnect`: Updates service API, documentation, and paginators
* `service/pinpoint`: Updates service API and documentation
  * Amazon Pinpoint - Features - Customers can start a journey based on an event being triggered by an endpoint or user.
* `service/s3`: Updates service API, documentation, and examples
  * Amazon S3 on Outposts expands object storage to on-premises AWS Outposts environments, enabling you to store and retrieve objects using S3 APIs and features.
* `service/s3outposts`: Adds new service
* `service/securityhub`: Updates service API and documentation

### SDK Features
* `service/s3`: Adds support for outposts access point ARNs.
* `service/s3control`: Adds support for S3 on outposts access point and S3 on outposts bucket ARNs.

Release v1.34.34 (2020-09-29)
===

### Service Client Updates
* `service/connect`: Updates service documentation
* `service/ec2`: Updates service documentation
  * This release adds support for Client to Client routing for AWS Client VPN.
* `service/schemas`: Updates service API and documentation
* `service/ssm`: Updates service documentation
  * Simple update to description of ComplianceItemStatus.
* `service/timestream-query`: Adds new service
* `service/timestream-write`: Adds new service

Release v1.34.33 (2020-09-28)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * This release adds the InsufficientAvailableIPsInSubnetFault error for RDS Proxy.

Release v1.34.32 (2020-09-25)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Support custom logging, executionRole, secrets, and linuxParameters (initProcessEnabled, maxSwap, swappiness, sharedMemorySize, and tmpfs). Also, add new context keys for awslogs.
* `service/config`: Updates service API
* `service/docdb`: Updates service documentation
  * Documentation updates for docdb
* `service/ec2`: Updates service API and documentation
  * This release supports returning additional information about local gateway resources, such as the local gateway route table.
* `service/frauddetector`: Updates service API and documentation
* `service/sts`: Updates service API and documentation
  * Documentation update for AssumeRole error

Release v1.34.31 (2020-09-24)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/eks`: Updates service API and documentation
* `service/savingsplans`: Updates service API and documentation
* `service/synthetics`: Updates service API and documentation
* `service/textract`: Updates service API and documentation
* `service/transcribe`: Updates service API and documentation

Release v1.34.30 (2020-09-23)
===

### Service Client Updates
* `service/backup`: Updates service API and documentation
* `service/ce`: Updates service API and documentation
* `service/quicksight`: Updates service API and documentation
  * Added Sheet information to DescribeDashboard, DescribeTemplate and DescribeAnalysis API response.
* `service/translate`: Updates service API and documentation

### SDK Enhancements
* `service/s3/s3manager`:  Prefer using allocated slices from pool over allocating new ones. ([#3534](https://github.com/aws/aws-sdk-go/pull/3534))

Release v1.34.29 (2020-09-22)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `service/lex-models`: Updates service API and documentation
* `service/streams.dynamodb`: Updates service API and documentation
* `service/workmail`: Updates service API, documentation, and paginators
  * Adding support for Mailbox Export APIs

Release v1.34.28 (2020-09-21)
===

### Service Client Updates
* `service/eventbridge`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Add support for Redshift Data API Targets
* `service/glue`: Updates service API and documentation
  * Adding support to update multiple partitions of a table in a single request
* `service/iotsitewise`: Updates service API and documentation
* `service/rds`: Updates service documentation
  * Documentation updates for the RDS DescribeExportTasks API
* `service/resource-groups`: Updates service documentation and paginators
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation updates for the Resource Groups Tagging API.

Release v1.34.27 (2020-09-18)
===

### Service Client Updates
* `service/codestar-connections`: Updates service API
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports batch operations, which allow users to start, stop, and delete multiple MediaLive resources with a single request.
* `service/sso-admin`: Updates service documentation

Release v1.34.26 (2020-09-17)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Adds support for mutual TLS authentication for public regional REST Apis
* `service/apigatewayv2`: Updates service API and documentation
  * Adds support for mutual TLS authentication and disableAPIExecuteEndpoint for public regional HTTP Apis
* `service/cloudfront`: Updates service documentation
  * Documentation updates for CloudFront
* `service/comprehend`: Updates service API and documentation
* `service/es`: Updates service API and documentation
  * Adds support for data plane audit logging in Amazon Elasticsearch Service.
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now supports additional file formats and metadata for FAQs.
* `service/transcribe-streaming`: Updates service API and documentation

Release v1.34.25 (2020-09-16)
===

### Service Client Updates
* `service/connect`: Updates service API, documentation, and paginators
* `service/dlm`: Updates service API and documentation
* `service/greengrass`: Updates service API and documentation
  * This release includes the ability to set run-time configuration for a Greengrass core. The Telemetry feature, also included in this release, can be configured via run-time configuration per core.
* `service/servicecatalog`: Updates service API and documentation
  * Enhance DescribeProvisionedProduct API to allow useProvisionedProduct Name as Input, so customer can provide ProvisionedProduct Name instead of ProvisionedProduct Id to describe a ProvisionedProduct.
* `service/ssm`: Updates service documentation
  * The ComplianceItemEntry Status description was updated to address Windows patches that aren't applicable.

Release v1.34.24 (2020-09-15)
===

### Service Client Updates
* `service/budgets`: Updates service API, documentation, and paginators
  * Documentation updates for Daily Cost and Usage budgets
* `service/ec2`: Updates service API
  * T4g instances are powered by AWS Graviton2 processors
* `service/kafka`: Updates service API, documentation, and paginators
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now returns confidence scores for 'document' query responses.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports CDI (Cloud Digital Interface) inputs which enable uncompressed video from applications on Elastic Cloud Compute (EC2), AWS Media Services, and from AWS partners
* `service/organizations`: Updates service API and documentation
  * AWS Organizations now enables you to add tags to the AWS accounts, organizational units, organization root, and policies in your organization.
* `service/sagemaker`: Updates service API and documentation
  * Sagemaker Ground Truth: Added support for a new Streaming feature which helps to continuously feed data and receive labels in real time. This release adds a new input and output SNS data channel.
* `service/transcribe`: Updates service API and documentation

Release v1.34.23 (2020-09-14)
===

### Service Client Updates
* `service/docdb`: Updates service API, documentation, and paginators
  * Updated API documentation and added paginators for DescribeCertificates, DescribeDBClusterParameterGroups, DescribeDBClusterParameters, DescribeDBClusterSnapshots and DescribePendingMaintenanceActions
* `service/ec2`: Updates service API
  * This release adds support for the T4G instance family to the EC2 ModifyDefaultCreditSpecification and GetDefaultCreditSpecification APIs.
* `service/managedblockchain`: Updates service API and documentation
* `service/states`: Updates service API and documentation
  * This release of the AWS Step Functions SDK introduces support for AWS X-Ray.

Release v1.34.22 (2020-09-11)
===

### Service Client Updates
* `service/workspaces`: Updates service API and documentation
  * Adds API support for WorkSpaces Cross-Region Redirection feature.

Release v1.34.21 (2020-09-10)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * Cloudfront adds support for Brotli. You can enable brotli caching and compression support by enabling it in your Cache Policy.
* `service/ebs`: Updates service documentation
* `service/pinpoint`: Updates service documentation
  * Update SMS message model description to clearly indicate that the MediaUrl field is reserved for future use and is not supported by Pinpoint as of today.
* `service/s3`: Updates service API, documentation, and examples
  * Bucket owner verification feature added. This feature introduces the x-amz-expected-bucket-owner and x-amz-source-expected-bucket-owner headers.
* `service/sso-admin`: Adds new service

Release v1.34.20 (2020-09-09)
===

### Service Client Updates
* `service/glue`: Updates service API, documentation, and paginators
  * Adding support for partitionIndexes to improve GetPartitions performance.
* `service/kinesisanalyticsv2`: Updates service API and documentation
* `service/redshift-data`: Adds new service

Release v1.34.19 (2020-09-08)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * You can now secure HTTP APIs using Lambda authorizers and IAM authorizers. These options enable you to make flexible auth decisions using a Lambda function, or using IAM policies, respectively.
* `service/codebuild`: Updates service API and documentation
  * AWS CodeBuild - Support keyword search for test cases in DecribeTestCases API . Allow deletion of reports in the report group, before deletion of report group using the deleteReports flag.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/lex-models`: Updates service API and documentation
* `service/quicksight`: Updates service API and documentation
  * Adds tagging support for QuickSight customization resources.  A user can now specify a list of tags when creating a customization resource and use a customization ARN in QuickSight's tagging APIs.

Release v1.34.18 (2020-09-04)
===

### Service Client Updates
* `service/ssm`: Updates service documentation
  * Documentation-only updates for AWS Systems Manager
* `service/workspaces`: Updates service API and documentation
  * Adding support for Microsoft Office 2016 and Microsoft Office 2019 in BYOL Images
* `service/xray`: Updates service API and documentation
  * Enhancing CreateGroup, UpdateGroup, GetGroup and GetGroups APIs to support configuring X-Ray Insights

Release v1.34.17 (2020-09-03)
===

### Service Client Updates
* `service/guardduty`: Updates service API and documentation
  * GuardDuty findings triggered by failed events now include the error code name within the AwsApiCallAction section.
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now returns confidence scores for both 'answer' and 'question and answer' query responses.
* `service/mediapackage`: Updates service API and documentation
  * Enables inserting a UTCTiming XML tag in the output manifest of a DASH endpoint which a media player will use to help with time synchronization.
* `service/states`: Updates service API and documentation
  * This release of the AWS Step Functions SDK introduces support for payloads up to 256KB for Standard and Express workflows

Release v1.34.16 (2020-09-02)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds a new transit gateway attachment state and resource type.
* `service/macie2`: Updates service API and documentation

Release v1.34.15 (2020-09-01)
===

### Service Client Updates
* `service/codeguru-reviewer`: Updates service API and documentation
* `service/securityhub`: Updates service API and documentation

Release v1.34.14 (2020-08-31)
===

### Service Client Updates
* `service/backup`: Updates service documentation
* `service/cloudfront`: Updates service API and documentation
  * CloudFront now supports real-time logging for CloudFront distributions. CloudFront real-time logs are more detailed, configurable, and are available in real time.
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 and Spot Fleet now support modification of launch template configs for a running fleet enabling instance type, instance weight, AZ, and AMI updates without losing the current fleet ID.
* `service/sqs`: Updates service documentation
  * Documentation updates for SQS.

### SDK Bugs
* `aws/ec2metadata`: Add support for EC2 IMDS endpoint from environment variable ([#3504](https://github.com/aws/aws-sdk-go/pull/3504))
  * Adds support for specifying a custom EC2 IMDS endpoint from the environment variable, `AWS_EC2_METADATA_SERVICE_ENDPOINT`.
  * The `aws/session#Options` struct also has a new field, `EC2IMDSEndpoint`. This field can be used to configure the custom endpoint of the EC2 IMDS client. The option only applies to EC2 IMDS clients created after the Session with `EC2IMDSEndpoint` is specified.

Release v1.34.13 (2020-08-28)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * You can now manage CloudFront's additional, real-time metrics with the CloudFront API.
* `service/cur`: Updates service API and documentation
  * This release add MONTHLY as the new supported TimeUnit for ReportDefinition.
* `service/elasticmapreduce`: Updates service API, documentation, and paginators
  * Amazon EMR adds support for ICMP, port -1, in Block Public Access Exceptions and API access for EMR Notebooks execution. You can now non-interactively execute EMR Notebooks and pass input parameters.
* `service/route53`: Updates service documentation
  * Documentation updates for Route 53

### SDK Bugs
*  `private/protocol`: Limit iso8601 fractional second precision to milliseconds ([#3507](https://github.com/aws/aws-sdk-go/pull/3507))
  * Fixes [#3498](https://github.com/aws/aws-sdk-go/issues/3498)

Release v1.34.12 (2020-08-27)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Introduces support to initiate Internet Key Exchange (IKE) negotiations for VPN connections from AWS. A user can now send the initial IKE message to their Customer Gateway (CGW) from VPN endpoints.
* `service/gamelift`: Updates service API, documentation, and paginators
  * GameLift FleetIQ as a standalone feature is now generally available. FleetIQ makes low-cost Spot instances viable for game hosting. Use GameLift FleetIQ with your EC2 Auto Scaling groups.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for WebM DASH outputs as well as H.264 4:2:2 10-bit output in MOV and MP4.
* `service/redshift`: Updates service documentation
  * Documentation updates for Amazon Redshift.

Release v1.34.11 (2020-08-26)
===

### Service Client Updates
* `service/appflow`: Adds new service
* `service/route53resolver`: Updates service API, documentation, and paginators

Release v1.34.10 (2020-08-24)
===

### Service Client Updates
* `service/dms`: Updates service API and documentation
  * Added new endpoint settings to include columns with Null and Empty value when using Kinesis and Kafka as target. Added a new endpoint setting to set maximum message size when using Kafka as target.
* `service/ec2`: Updates service API, documentation, and paginators
  * This release enables customers to use VPC prefix lists in their transit gateway route tables, and it adds support for Provisioned IOPS SSD (io2) EBS volumes.
* `service/iotsitewise`: Updates service API and documentation
* `service/kafka`: Updates service API and documentation
* `service/logs`: Updates service documentation
  * Documentation updates for CloudWatch Logs
* `service/ssm`: Updates service API and documentation
  * Add string length constraints to OpsDataAttributeName and OpsFilterValue.
* `service/xray`: Updates service API and documentation
  * AWS X-Ray now supports tagging on sampling rules and groups.

Release v1.34.9 (2020-08-20)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * Customers can now create Amazon API Gateway HTTP APIs that route requests to AWS AppConfig, Amazon EventBridge, Amazon Kinesis Data Streams, Amazon SQS, and AWS Step Functions.
* `service/chime`: Updates service documentation
  * Documentation updates for chime
* `service/fsx`: Updates service documentation

### SDK Enhancements
* `private/protocol`: The SDK now supports the serialization of ISO8601 date-time formats with fractional seconds precision. ([#3489](https://github.com/aws/aws-sdk-go/pull/3489))

Release v1.34.8 (2020-08-19)
===

### Service Client Updates
* `service/ivs`: Updates service API, documentation, and paginators
* `service/lakeformation`: Updates service API and documentation
* `service/organizations`: Updates service documentation
  * Minor documentation updates for AWS Organizations
* `service/servicecatalog`: Updates service API and documentation
  * Enhance SearchProvisionedProducts API to allow queries using productName and provisioningArtifactName. Added lastProvisioningRecordId and lastSuccessfulRecordId to Read ProvisionedProduct APIs
* `service/storagegateway`: Updates service API and documentation
  * Added WORM, tape retention lock, and custom pool features for virtual tapes.
* `service/transcribe-streaming`: Updates service API and documentation

Release v1.34.7 (2020-08-18)
===

### Service Client Updates
* `service/codebuild`: Updates service documentation
  * Documentation updates for codebuild
* `service/cognito-idp`: Updates service API and documentation
* `service/datasync`: Updates service API and documentation
* `service/identitystore`: Adds new service
* `service/securityhub`: Updates service API and documentation
* `service/sesv2`: Updates service API, documentation, and paginators

Release v1.34.6 (2020-08-17)
===

### Service Client Updates
* `service/acm`: Updates service API
  * ACM provides support for the new Private CA feature Cross-account CA sharing. ACM users can issue certificates signed by a private CA belonging to another account where the CA was shared with them.
* `service/acm-pca`: Updates service API and documentation
* `service/ecr`: Updates service API and documentation
  * This feature adds support for pushing and pulling Open Container Initiative (OCI) artifacts.
* `service/elasticloadbalancing`: Updates service documentation
* `service/elasticloadbalancingv2`: Updates service documentation
* `service/kinesis`: Updates service API and documentation
  * Introducing ShardFilter for ListShards API to filter the shards using a position in the stream, and ChildShards support for GetRecords and SubscribeToShard API to discover children shards on shard end
* `service/quicksight`: Updates service API, documentation, and paginators
  * Amazon QuickSight now supports programmatic creation and management of analyses with new APIs.
* `service/robomaker`: Updates service API, documentation, and paginators

Release v1.34.5 (2020-08-14)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Adds support for the Desktop View feature
* `service/braket`: Updates service API
* `service/ec2`: Updates service API
  * New C5ad instances featuring AMD's 2nd Generation EPYC processors, offering up to 96 vCPUs, 192 GiB of instance memory, 3.8 TB of NVMe based SSD instance storage, and 20 Gbps in Network bandwidth
* `service/license-manager`: Updates service documentation
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker now supports 1) creating real-time inference endpoints using model container images from Docker registries in customers' VPC 2) AUC(Area under the curve) as AutoPilot objective metric

Release v1.34.4 (2020-08-13)
===

### Service Client Updates
* `service/appsync`: Updates service documentation
* `service/braket`: Adds new service
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Added MapCustomerOwnedIpOnLaunch and CustomerOwnedIpv4Pool to ModifySubnetAttribute to allow CoIP auto assign. Fields are returned in DescribeSubnets and DescribeNetworkInterfaces responses.
* `service/eks`: Updates service API and documentation
* `service/macie2`: Updates service documentation
* `service/rds`: Updates service API and documentation
  * This release allows customers to specify a replica mode when creating or modifying a Read Replica, for DB engines which support this feature.

Release v1.34.3 (2020-08-12)
===

### Service Client Updates
* `service/cloud9`: Updates service API and documentation
  * Add ConnectionType input parameter to CreateEnvironmentEC2 endpoint. New parameter enables creation of environments with SSM connection.
* `service/comprehend`: Updates service documentation
* `service/ec2`: Updates service API and documentation
  * Introduces support for IPv6-in-IPv4 IPsec tunnels. A user can now send traffic from their on-premise IPv6 network to AWS VPCs that have IPv6 support enabled.
* `service/fsx`: Updates service API and documentation
* `service/iot`: Updates service API, documentation, and paginators
  * Audit finding suppressions: Device Defender enables customers to turn off non-compliant findings for specific resources on a per check basis.
* `service/lambda`: Updates service API and examples
  * Support for creating Lambda Functions using 'java8.al2' and 'provided.al2'
* `service/transfer`: Updates service API, documentation, and paginators
  * Adds security policies to control cryptographic algorithms advertised by your server, additional characters in usernames and length increase, and FIPS compliant endpoints in the US and Canada regions.
* `service/workspaces`: Updates service API and documentation
  * Adds optional EnableWorkDocs property to WorkspaceCreationProperties in the ModifyWorkspaceCreationProperties API

### SDK Enhancements
* `codegen`: Add XXX_Values functions for getting slice of API enums by type.
  * Fixes [#3441](https://github.com/aws/aws-sdk-go/issues/3441) by adding a new XXX_Values function for each API enum type that returns a slice of enum values, e.g `DomainStatus_Values`.
* `aws/request`: Update default retry to retry "use of closed network connection" errors ([#3476](https://github.com/aws/aws-sdk-go/pull/3476))
  * Fixes [#3406](https://github.com/aws/aws-sdk-go/issues/3406)

### SDK Bugs
* `private/protocol/json/jsonutil`: Fixes a bug that truncated millisecond precision time in API response to seconds. ([#3474](https://github.com/aws/aws-sdk-go/pull/3474))
  * Fixes [#3464](https://github.com/aws/aws-sdk-go/issues/3464)
  * Fixes [#3410](https://github.com/aws/aws-sdk-go/issues/3410)
* `codegen`: Export event stream constructor for easier mocking ([#3473](https://github.com/aws/aws-sdk-go/pull/3473))
  * Fixes [#3412](https://github.com/aws/aws-sdk-go/issues/3412) by exporting the operation's EventStream type's constructor function so it can be used to fully initialize fully when mocking out behavior for API operations with event streams.
* `service/ec2`: Fix max retries with client customizations ([#3465](https://github.com/aws/aws-sdk-go/pull/3465))
  * Fixes [#3374](https://github.com/aws/aws-sdk-go/issues/3374) by correcting the EC2 API client's customization for ModifyNetworkInterfaceAttribute and AssignPrivateIpAddresses operations to use the aws.Config.MaxRetries value if set. Previously the API client's customizations would ignore MaxRetries specified in the SDK's aws.Config.MaxRetries field.

Release v1.34.2 (2020-08-11)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * This release rolls back the EC2 On-Demand Capacity Reservations (ODCRs) release 1.11.831 published on 2020-07-30, which was deployed in error.
* `service/lambda`: Updates service API, documentation, and examples
  * Support Managed Streaming for Kafka as an Event Source. Support retry until record expiration for Kinesis and Dynamodb streams event source mappings.
* `service/organizations`: Updates service documentation
  * Minor documentation update for AWS Organizations
* `service/s3`: Updates service API, documentation, and examples
  * Add support for in-region CopyObject and UploadPartCopy through S3 Access Points

Release v1.34.1 (2020-08-10)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Remove CoIP Auto-Assign feature references.
* `service/glue`: Updates service API and documentation
  * Starting today, you can further control orchestration of your ETL workloads in AWS Glue by specifying the maximum number of concurrent runs for a Glue workflow.
* `service/savingsplans`: Updates service API

### SDK Enhancements
* `aws/credentials/stscreds`: Add optional expiry duration to WebIdentityRoleProvider ([#3356](https://github.com/aws/aws-sdk-go/pull/3356))
  * Adds a new optional field to the WebIdentityRoleProvider that allows you to specify the duration the assumed role credentials will be valid for.
* `example/service/s3/putObjectWithProgress`: Fix example for file upload with progress ([#3377](https://github.com/aws/aws-sdk-go/pull/3377))
  * Fixes [#2468](https://github.com/aws/aws-sdk-go/issues/2468) by ignoring the first read of the progress reader wrapper. Since the first read is used for signing the request, not upload progress.
  * Updated the example to write progress inline instead of newlines.
* `service/dynamodb/dynamodbattribute`: Fix typo in package docs ([#3446](https://github.com/aws/aws-sdk-go/pull/3446))
  * Fixes typo in dynamodbattribute package docs.

Release v1.34.0 (2020-08-07)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * AWS Glue now adds support for Network connection type enabling you to access resources inside your VPC using Glue crawlers and Glue ETL jobs.
* `service/organizations`: Updates service API and documentation
  * Documentation updates for some new error reasons.
* `service/s3`: Updates service documentation and examples
  * Updates Amazon S3 API reference documentation.
* `service/sms`: Updates service API and documentation
  * In this release, AWS Server Migration Service (SMS) has added new features: 1. APIs to work with application and instance level validation 2. Import application catalog from AWS Application Discovery Service 3. For an application you can start on-demand replication

### SDK Features
* `service/s3/s3crypto`: Updates to the Amazon S3 Encryption Client - This change includes fixes for issues that were reported by Sophie Schmieg from the Google ISE team, and for issues that were discovered by AWS Cryptography.

Release v1.33.21 (2020-08-06)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and paginators
  * This release supports Wavelength resources, including carrier gateways, and carrier IP addresses.
* `service/lex-models`: Updates service API and documentation
* `service/personalize`: Updates service API and documentation
* `service/personalize-events`: Updates service API and documentation
* `service/personalize-runtime`: Updates service API and documentation
* `service/runtime.lex`: Updates service API and documentation

Release v1.33.20 (2020-08-05)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/fsx`: Updates service documentation
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation updates for the Resource Group Tagging API namespace.
* `service/sns`: Updates service documentation
  * Documentation updates for SNS.
* `service/transcribe`: Updates service API, documentation, and paginators

Release v1.33.19 (2020-08-04)
===

### Service Client Updates
* `service/health`: Updates service documentation
  * Documentation updates for health

Release v1.33.18 (2020-08-03)
===

### Service Client Updates
* `service/ssm`: Updates service waiters and paginators
  * Adds a waiter for CommandExecuted and paginators for various other APIs.

Release v1.33.17 (2020-07-31)
===

### Service Client Updates
* `service/chime`: Updates service API
  * This release increases the CreateMeetingWithAttendee max attendee limit to 10.
* `service/personalize-runtime`: Updates service API and documentation
* `service/resourcegroupstaggingapi`: Updates service API and documentation
  * Updates to the list of services supported by this API.
* `service/storagegateway`: Updates service API and documentation
  * Add support for gateway VM deprecation dates
* `service/wafv2`: Updates service API and documentation

Release v1.33.16 (2020-07-30)
===

### Service Client Updates
* `service/cloudfront`: Updates service documentation
  * Documentation updates for CloudFront
* `service/codebuild`: Updates service API, documentation, and paginators
  * Adding support for BuildBatch, and CodeCoverage APIs. BuildBatch allows you to model your project environment in source, and helps start multiple builds with a single API call. CodeCoverage allows you to track your code coverage using AWS CodeBuild.
* `service/ec2`: Updates service API
  * EC2 On-Demand Capacity Reservations now adds support to bring your own licenses (BYOL) of Windows operating system to launch EC2 instances.
* `service/guardduty`: Updates service API, documentation, and paginators
  * GuardDuty can now provide detailed cost metrics broken down by account, data source, and S3 resources, based on the past 30 days of usage.  This new feature also supports viewing cost metrics for all member accounts as a GuardDuty master.
* `service/kafka`: Updates service API and documentation
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations
* `service/resource-groups`: Updates service documentation
* `service/servicecatalog`: Updates service API and documentation
  * This release adds support for ProvisionProduct, UpdateProvisionedProduct & DescribeProvisioningParameters by product name, provisioning artifact name and path name. In addition DescribeProvisioningParameters now returns a list of provisioning artifact outputs.
* `service/sesv2`: Updates service API, documentation, and paginators

Release v1.33.15 (2020-07-29)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and paginators
  * Adding support to target EC2 On-Demand Capacity Reservations within an AWS Resource Group to launch EC2 instances.
* `service/ecr`: Updates service API and documentation
  * This release adds support for encrypting the contents of your Amazon ECR repository with customer master keys (CMKs) stored in AWS Key Management Service.
* `service/firehose`: Updates service API and documentation
  * This release includes a new Kinesis Data Firehose feature that supports data delivery to Https endpoint and to partners. You can now use Kinesis Data Firehose to ingest real-time data and deliver to Https endpoint and partners in a serverless, reliable, and salable manner.
* `service/guardduty`: Updates service API and documentation
  * GuardDuty now supports S3 Data Events as a configurable data source type. This feature expands GuardDuty's monitoring scope to include S3 data plane operations, such as GetObject and PutObject. This data source is optional and can be enabled or disabled at anytime. Accounts already using GuardDuty must first enable the new feature to use it; new accounts will be enabled by default. GuardDuty masters can configure this data source for individual member accounts and GuardDuty masters associated through AWS Organizations can automatically enable the data source in member accounts.
* `service/resource-groups`: Updates service API and documentation
* `service/servicediscovery`: Updates service documentation

Release v1.33.14 (2020-07-28)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Now you can enable Instance Metadata Service Version 2 (IMDSv2) or disable the instance metadata endpoint with Launch Configurations.
* `service/ec2`: Updates service API and documentation
  * Introduces support for tag-on-create capability for the following APIs: CreateVpnConnection, CreateVpnGateway, and CreateCustomerGateway. A user can now add tags while creating these resources. For further detail, please see AWS Tagging Strategies.
* `service/imagebuilder`: Updates service API and documentation
* `service/ivs`: Updates service API and documentation
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports several new features: EBU-TT-D captions in Microsoft Smooth outputs; interlaced video in HEVC outputs; video noise reduction (using temporal filtering) in HEVC outputs.
* `service/rds`: Updates service documentation
  * Adds reporting of manual cluster snapshot quota to DescribeAccountAttributes API
* `service/securityhub`: Updates service API and documentation

Release v1.33.13 (2020-07-27)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `service/dms`: Updates service API, documentation, and paginators
  * Basic endpoint settings for relational databases, Preflight validation API.
* `service/ec2`: Updates service API
  * m6gd, c6gd, r6gd instances are powered by AWS Graviton2 processors and support local NVMe instance storage
* `service/frauddetector`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * Add ability to manually resume workflows in AWS Glue providing customers further control over the orchestration of ETL workloads.
* `service/ssm`: Updates service documentation
  * Assorted doc ticket-fix updates for Systems Manager.

Release v1.33.12 (2020-07-24)
===

### Service Client Updates
* `service/frauddetector`: Updates service API and documentation
* `service/fsx`: Updates service documentation
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra now supports sorting query results based on document attributes. Amazon Kendra also introduced an option to enclose table and column names with double quotes for database data sources.
* `service/macie2`: Updates service API and documentation
* `service/mediaconnect`: Updates service API and documentation
* `service/mediapackage`: Updates service API and documentation
  * The release adds daterange as a new ad marker option. This option enables MediaPackage to insert EXT-X-DATERANGE tags in HLS and CMAF manifests. The EXT-X-DATERANGE tag is used to signal ad and program transition events.
* `service/monitoring`: Updates service API and documentation
  * AWS CloudWatch ListMetrics now supports an optional parameter (RecentlyActive) to filter results by only metrics that have received new datapoints in the past 3 hours. This enables more targeted metric data retrieval through the Get APIs
* `service/mq`: Updates service API, documentation, and paginators
  * Amazon MQ now supports LDAP (Lightweight Directory Access Protocol), providing authentication and authorization of Amazon MQ users via a customer designated LDAP server.
* `service/sagemaker`: Updates service API, documentation, and paginators
  * Sagemaker Ground Truth:Added support for OIDC (OpenID Connect) to authenticate workers via their own identity provider instead of through Amazon Cognito. This release adds new APIs (CreateWorkforce, DeleteWorkforce, and ListWorkforces) to SageMaker Ground Truth service.  Sagemaker Neo: Added support for detailed target device description by using TargetPlatform fields - OS, architecture, and accelerator. Added support for additional compilation parameters by using JSON field CompilerOptions.  Sagemaker Search: SageMaker Search supports transform job details in trial components.

### SDK Bugs
* `service/s3/s3crypto`: Fix client's temporary file buffer error on retry ([#3344](https://github.com/aws/aws-sdk-go/pull/3344))
  * Fixes the Crypto client's temporary file buffer cleanup returning an error when the request is retried.

Release v1.33.11 (2020-07-23)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/directconnect`: Updates service documentation
  * Documentation updates for AWS Direct Connect
* `service/fsx`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * Added new ConnectionProperties: "KAFKA_SSL_ENABLED" (to toggle SSL connections) and "KAFKA_CUSTOM_CERT" (import CA certificate file)
* `service/lightsail`: Updates service API and documentation
  * This release adds support for Amazon Lightsail content delivery network (CDN) distributions and SSL/TLS certificates.
* `service/workspaces`: Updates service API and documentation
  * Added UpdateWorkspaceImagePermission API to share Amazon WorkSpaces images across AWS accounts.

Release v1.33.10 (2020-07-22)
===

### Service Client Updates
* `service/medialive`: Updates service API and documentation
  * The AWS Elemental MediaLive APIs and SDKs now support the ability to get thumbnails for MediaLive devices that are attached or not attached to a channel. Previously, this thumbnail feature was available only on the console.
* `service/quicksight`: Updates service API, documentation, and paginators
  * New API operations - GetSessionEmbedUrl, CreateNamespace, DescribeNamespace, ListNamespaces, DeleteNamespace, DescribeAccountSettings, UpdateAccountSettings, CreateAccountCustomization, DescribeAccountCustomization, UpdateAccountCustomization, DeleteAccountCustomization. Modified API operations to support custom permissions restrictions - RegisterUser, UpdateUser, UpdateDashboardPermissions

### SDK Enhancements
* `example/aws/request/httptrace`: Update example with more metrics ([#3436](https://github.com/aws/aws-sdk-go/pull/3436))
  * Updates the tracing example to include additional metrics such as SDKs request handlers, and support multiple request attempts.

Release v1.33.9 (2020-07-21)
===

### Service Client Updates
* `service/codeguruprofiler`: Updates service API and documentation

Release v1.33.8 (2020-07-20)
===

### Service Client Updates
* `service/cloudfront`: Adds new service
  * CloudFront adds support for cache policies and origin request policies. With these new policies, you can now more granularly control the query string, header, and cookie values that are included in the cache key and in requests that CloudFront sends to your origin.
* `service/codebuild`: Updates service API and documentation
  * AWS CodeBuild adds support for Session Manager and Windows 2019 Environment type
* `service/ec2`: Updates service API and documentation
  * Added support for tag-on-create for CreateVpcPeeringConnection and CreateRouteTable. You can now specify tags when creating any of these resources. For more information about tagging, see AWS Tagging Strategies. Add poolArn to the response of DescribeCoipPools.
* `service/fms`: Updates service API and documentation
* `service/frauddetector`: Updates service API, documentation, and paginators
* `service/groundstation`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Add a new SupportsParallelQuery output field to DescribeDBEngineVersions. This field shows whether the engine version supports parallelquery. Add a new SupportsGlobalDatabases output field to DescribeDBEngineVersions and DescribeOrderableDBInstanceOptions. This field shows whether global database is supported by engine version or the combination of engine version and instance class.

Release v1.33.7 (2020-07-17)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation
* `service/appsync`: Updates service documentation
* `service/connect`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Documentation updates for EC2
* `service/elasticbeanstalk`: Updates service waiters and paginators
  * Add waiters for `EnvironmentExists`, `EnvironmentUpdated`, and `EnvironmentTerminated`. Add paginators for `DescribeEnvironmentManagedActionHistory` and `ListPlatformVersions`.
* `service/macie2`: Updates service API, documentation, and paginators

### SDK Enhancements
* `service/s3/s3manager`: Clarify documentation and behavior of GetBucketRegion ([#3428](https://github.com/aws/aws-sdk-go/pull/3428))
  * Updates the documentation for GetBucketRegion's behavior with regard to default configuration for path style addressing. Provides examples how to override this behavior.
  * Updates the GetBucketRegion utility to not require a region hint when the session or client was configured with a custom endpoint URL.
  * Related to [#3115](https://github.com/aws/aws-sdk-go/issues/3115)
* `service/s3`: Add failsafe handling for unknown stream messages
  * Adds failsafe handling for receiving unknown stream messages from an API. A `<streamName>UnknownEvent` type will encapsulate the unknown message received from the API. Where `<streamName>` is the name of the API's stream, (e.g. S3's `SelectObjectContentEventStreamUnknownEvent`).

Release v1.33.6 (2020-07-15)
===

### Service Client Updates
* `service/ivs`: Adds new service

### SDK Enhancements
* `service/s3/s3crypto`: Allow envelope unmarshal to accept JSON numbers for tag length [(#3422)](https://github.com/aws/aws-sdk-go/pull/3422)

Release v1.33.5 (2020-07-09)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/amplify`: Updates service documentation
* `service/appmesh`: Updates service API, documentation, and paginators
* `service/cloudhsmv2`: Updates service documentation
  * Documentation updates for cloudhsmv2
* `service/comprehend`: Updates service API and documentation
* `service/ebs`: Updates service API and documentation
* `service/eventbridge`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Amazon CloudWatch Events/EventBridge adds support for API Gateway as a target.
* `service/sagemaker`: Updates service API and documentation
  * This release adds the DeleteHumanTaskUi API to Amazon Augmented AI
* `service/secretsmanager`: Updates service API, documentation, and examples
  * Adds support for filters on the ListSecrets API to allow filtering results by name, tag key, tag value, or description.  Adds support for the BlockPublicPolicy option on the PutResourcePolicy API to block resource policies which grant a wide range of IAM principals access to secrets. Adds support for the ValidateResourcePolicy API to validate resource policies for syntax and prevent lockout error scenarios and wide access to secrets.
* `service/sns`: Updates service documentation
  * This release adds support for SMS origination number as an attribute in the MessageAttributes parameter for the SNS Publish API.
* `service/wafv2`: Updates service API and documentation

Release v1.33.4 (2020-07-08)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * EC2 Spot now enables customers to tag their Spot Instances Requests on creation.
* `service/forecast`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * We have launched a self-service option to make it easier for customers to manage the use of their content by AI services. Certain AI services (Amazon CodeGuru Profiler, Amazon Comprehend, Amazon Lex, Amazon Polly, Amazon Rekognition, Amazon Textract, Amazon Transcribe, and Amazon Translate) may use content to improve the service. Customers have been able to opt out of this use by contacting AWS Support, and now they can opt out on a self-service basis by setting an Organizations policy for all or an individual AI service listed above. Please refer to the technical documentation in the online AWS Organizations User Guide for more details.

Release v1.33.3 (2020-07-07)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * Amazon CloudFront adds support for a new security policy, TLSv1.2_2019.
* `service/ec2`: Updates service API and documentation
  * DescribeAvailabilityZones now returns additional data about Availability Zones and Local Zones.
* `service/elasticfilesystem`: Updates service API, documentation, and examples
  * This release adds support for automatic backups of Amazon EFS file systems to further simplify backup management.
* `service/glue`: Updates service API and documentation
  * AWS Glue Data Catalog supports cross account sharing of tables through AWS Lake Formation
* `service/lakeformation`: Updates service API and documentation
* `service/storagegateway`: Updates service API and documentation
  * Adding support for file-system driven directory refresh, Case Sensitivity toggle for SMB File Shares, and S3 Prefixes and custom File Share names

Release v1.33.2 (2020-07-06)
===

### Service Client Updates
* `service/iotsitewise`: Updates service API
* `service/quicksight`: Updates service API and documentation
  * Add Theme APIs and update Dashboard APIs to support theme overrides.
* `service/rds`: Updates service API and documentation
  * Adds support for Amazon RDS on AWS Outposts.

Release v1.33.1 (2020-07-02)
===

### Service Client Updates
* `service/connect`: Updates service documentation
* `service/elasticache`: Updates service documentation
  * Documentation updates for elasticache

Release v1.33.0 (2020-07-01)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/chime`: Updates service API and documentation
  * This release supports third party emergency call routing configuration for Amazon Chime Voice Connectors.
* `service/codebuild`: Updates service API and documentation
  * Support build status config in project source
* `service/imagebuilder`: Updates service API and documentation
* `service/rds`: Updates service API
  * This release adds the exceptions KMSKeyNotAccessibleFault and InvalidDBClusterStateFault to the Amazon RDS ModifyDBInstance API.
* `service/securityhub`: Updates service API and documentation

### SDK Features
* `service/s3/s3crypto`: Introduces `EncryptionClientV2` and `DecryptionClientV2` encryption and decryption clients which support a new key wrapping algorithm `kms+context`. ([#3403](https://github.com/aws/aws-sdk-go/pull/3403))
  * `DecryptionClientV2` maintains the ability to decrypt objects encrypted using the `EncryptionClient`.
  * Please see `s3crypto` documentation for migration details.

Release v1.32.13 (2020-06-30)
===

### Service Client Updates
* `service/codeguru-reviewer`: Updates service API and documentation
* `service/comprehendmedical`: Updates service API
* `service/ec2`: Updates service API and documentation
  * Added support for tag-on-create for CreateVpc, CreateEgressOnlyInternetGateway, CreateSecurityGroup, CreateSubnet, CreateNetworkInterface, CreateNetworkAcl, CreateDhcpOptions and CreateInternetGateway. You can now specify tags when creating any of these resources. For more information about tagging, see AWS Tagging Strategies.
* `service/ecr`: Updates service API and documentation
  * Add a new parameter (ImageDigest) and a new exception (ImageDigestDoesNotMatchException) to PutImage API to support pushing image by digest.
* `service/rds`: Updates service documentation
  * Documentation updates for rds

Release v1.32.12 (2020-06-29)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation and examples
  * Documentation updates for Amazon EC2 Auto Scaling.
* `service/codeguruprofiler`: Updates service API, documentation, and paginators
* `service/codestar-connections`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API, documentation, and paginators
  * Virtual Private Cloud (VPC) customers can now create and manage their own Prefix Lists to simplify VPC configurations.

Release v1.32.11 (2020-06-26)
===

### Service Client Updates
* `service/cloudformation`: Updates service API and documentation
  * ListStackInstances and DescribeStackInstance now return a new `StackInstanceStatus` object that contains `DetailedStatus` values: a disambiguation of the more generic `Status` value. ListStackInstances output can now be filtered on `DetailedStatus` using the new `Filters` parameter.
* `service/cognito-idp`: Updates service API
* `service/dms`: Updates service documentation
  * This release contains miscellaneous API documentation updates for AWS DMS in response to several customer reported issues.
* `service/quicksight`: Updates service API and documentation
  * Added support for cross-region DataSource credentials copying.
* `service/sagemaker`: Updates service API and documentation
  * The new 'ModelClientConfig' parameter being added for CreateTransformJob and DescribeTransformJob api actions enable customers to configure model invocation related parameters such as timeout and retry.

Release v1.32.10 (2020-06-25)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Added support for tag-on-create for Host Reservations in Dedicated Hosts. You can now specify tags when you create a Host Reservation for a Dedicated Host. For more information about tagging, see AWS Tagging Strategies.
* `service/glue`: Updates service API and documentation
  * This release adds new APIs to support column level statistics in AWS Glue Data Catalog

Release v1.32.9 (2020-06-24)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/autoscaling`: Updates service documentation
  * Documentation updates for Amazon EC2 Auto Scaling.
* `service/backup`: Updates service API and documentation
* `service/codecommit`: Updates service API, documentation, and paginators
  * This release introduces support for reactions to CodeCommit comments. Users will be able to select from a pre-defined list of emojis to express their reaction to any comments.
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR customers can now set allocation strategies for On-Demand and Spot instances in their EMR clusters with instance fleets. These allocation strategies use real-time capacity insights to provision clusters faster and make the most efficient use of available spare capacity to allocate Spot instances to reduce interruptions.
* `service/fsx`: Updates service API and documentation
* `service/honeycode`: Adds new service
* `service/iam`: Updates service documentation
  * Documentation updates for iam
* `service/organizations`: Updates service API and documentation
  * This release adds support for a new backup policy type for AWS Organizations.

Release v1.32.8 (2020-06-23)
===

### Service Client Updates
* `service/mediatailor`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * Added a new error message to support the requirement for a Business License on AWS accounts in China to create an organization.

Release v1.32.7 (2020-06-22)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds Tag On Create feature support for the ImportImage, ImportSnapshot, ExportImage and CreateInstanceExportTask APIs.
* `service/elasticmapreduce`: Updates service API and documentation
  * Adding support for MaximumCoreCapacityUnits parameter for EMR Managed Scaling. It allows users to control how many units/nodes are added to the CORE group/fleet. Remaining units/nodes are added to the TASK groups/fleet in the cluster.
* `service/rds`: Updates service documentation and paginators
  * Added paginators for various APIs.
* `service/rekognition`: Updates service API, documentation, and paginators
  * This update adds the ability to detect black frames, end credits, shots, and color bars in stored videos
* `service/sqs`: Updates service API, documentation, and paginators
  * AWS SQS adds pagination support for ListQueues and ListDeadLetterSourceQueues APIs

Release v1.32.6 (2020-06-19)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Adds support to tag elastic-gpu on the RunInstances api
* `service/elasticache`: Updates service documentation
  * Documentation updates for elasticache
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports Input Prepare schedule actions. This feature improves existing input switching by allowing users to prepare an input prior to switching to it.
* `service/opsworkscm`: Updates service API and documentation
  * Documentation updates for AWS OpsWorks CM.

Release v1.32.5 (2020-06-18)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for NexGuard FileMarker SDK, which allows NexGuard partners to watermark proprietary content in mezzanine and OTT streaming contexts.
* `service/meteringmarketplace`: Updates service documentation
  * Documentation updates for meteringmarketplace
* `service/rds`: Updates service API and documentation
  * Adding support for global write forwarding on secondary clusters in an Aurora global database.
* `service/route53`: Updates service API and documentation
  * Added a new ListHostedZonesByVPC API for customers to list all the private hosted zones that a specified VPC is associated with.
* `service/sesv2`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * Added offset support for specifying the number of days to wait after the date and time specified by a CRON expression before running the maintenance window.
* `service/support`: Updates service documentation
  * Documentation updates for support

Release v1.32.4 (2020-06-17)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * nvmeSupport added to DescribeInstanceTypes API
* `service/macie2`: Updates service documentation
* `service/route53`: Updates service API
  * Add PriorRequestNotComplete exception to AssociateVPCWithHostedZone API
* `service/snowball`: Updates service API and documentation
  * AWS Snowcone is a portable, rugged and secure device for edge computing and data transfer. You can use Snowcone to collect, process, and move data to AWS, either offline by shipping the device to AWS or online by using AWS DataSync. With 2 CPUs and 4 GB RAM of compute and 8 TB of storage, Snowcone can run edge computing workloads and store data securely. Snowcone's small size (8.94" x 5.85" x 3.25" / 227 mm x 148.6 mm x 82.65 mm) allows you to set it next to machinery in a factory. Snowcone weighs about 4.5 lbs. (2 kg), so you can carry one in a backpack, use it with battery-based operation, and use the Wi-Fi interface to gather sensor data. Snowcone supports a file interface with NFS support.

### SDK Enhancements
* `private/protocol`: Adds support for decimal precision UNIX timestamps up to thousandths of a second ([#3376](https://github.com/aws/aws-sdk-go/pull/3376))

Release v1.32.3 (2020-06-16)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Introducing instance refresh, a feature that helps you update all instances in an Auto Scaling group in a rolling fashion (for example, to apply a new AMI or instance type). You can control the pace of the refresh by defining the percentage of the group that must remain running/healthy during the replacement process and the time for new instances to warm up between replacements.
* `service/cloudfront`: Updates service documentation
  * Documentation updates for CloudFront
* `service/dataexchange`: Updates service API
* `service/lambda`: Updates service API, documentation, and examples
  * Adds support for using Amazon Elastic File System (persistent storage) with AWS Lambda. This enables customers to share data across function invocations, read large reference data files, and write function output to a persistent and shared store.
* `service/polly`: Updates service API
  * Amazon Polly adds new US English child voice - Kevin. Kevin is available as Neural voice only.
* `service/qldb`: Updates service documentation

Release v1.32.2 (2020-06-15)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/appconfig`: Updates service API, documentation, and paginators
* `service/chime`: Updates service API and documentation
  * feature: Chime: This release introduces the ability to create an AWS Chime SDK meeting with attendees.
* `service/cognito-idp`: Updates service API and documentation
* `service/iot`: Updates service API and documentation
  * Added support for job executions rollout configuration, job abort configuration, and job executions timeout configuration for AWS IoT Over-the-Air (OTA) Update Feature.

Release v1.32.1 (2020-06-12)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Documentation updates for Amazon API Gateway
* `service/cloudformation`: Updates service documentation
  * The following parameters now return the organization root ID or organizational unit (OU) IDs that you specified for DeploymentTargets: the OrganizationalUnitIds parameter on StackSet and the OrganizationalUnitId parameter on StackInstance, StackInstanceSummary, and StackSetOperationResultSummary
* `service/glue`: Updates service API and documentation
  * You can now choose to crawl the entire table or just a sample of records in DynamoDB when using AWS Glue crawlers. Additionally, you can also specify a scanning rate for crawling DynamoDB tables.
* `service/storagegateway`: Updates service API and documentation
  * Display EndpointType in DescribeGatewayInformation

Release v1.32.0 (2020-06-11)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * This release adds support for deleting capacity providers.
* `service/imagebuilder`: Updates service API and documentation
* `service/lex-models`: Updates service API and documentation

### SDK Features
* `service/iotdataplane`: As part of this release, we are introducing a new feature called named shadow, which extends the capability of AWS IoT Device Shadow to support multiple shadows for a single IoT device. With this release, customers can store different device state data into different shadows, and as a result access only the required state data when needed and reduce individual shadow size.

Release v1.31.15 (2020-06-10)
===

### Service Client Updates
* `service/appconfig`: Updates service API and documentation
* `service/codeartifact`: Adds new service
* `service/compute-optimizer`: Updates service API and documentation
* `service/dlm`: Updates service API
* `service/ec2`: Updates service API
  * New C6g instances powered by AWS Graviton2 processors and ideal for running advanced, compute-intensive workloads; New R6g instances powered by AWS Graviton2 processors and ideal for running memory-intensive workloads.
* `service/lightsail`: Updates service documentation
  * Documentation updates for lightsail
* `service/macie2`: Updates service API and documentation
* `service/servicecatalog`: Updates service documentation
  * Service Catalog Documentation Update for Integration with AWS Organizations Delegated Administrator feature
* `service/shield`: Updates service API and documentation
  * Corrections to the supported format for contact phone numbers and to the description for the create subscription action.

### SDK Enhancements
* `aws/credentials`: Update documentation for shared credentials provider to specify the type of credentials it supports retrieving from shared credentials file.
    * Related to [#3328](https://github.com/aws/aws-sdk-go/issues/3328)

Release v1.31.14 (2020-06-09)
===

### Service Client Updates
* `service/transfer`: Updates service API and documentation
  * This release updates the API so customers can test use of Source IP to allow, deny or limit access to data in their S3 buckets after integrating their identity provider.

Release v1.31.13 (2020-06-08)
===

### Service Client Updates
* `service/servicediscovery`: Updates service API, documentation, and examples
  * Added support for tagging Service and Namespace type resources  in Cloud Map
* `service/shield`: Updates service API, documentation, and paginators
  * This release adds the option for customers to identify a contact name and method that the DDoS Response Team can proactively engage when a Route 53 Health Check that is associated with a Shield protected resource fails.

Release v1.31.12 (2020-06-05)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Amazon API Gateway now allows customers of REST APIs to skip trust chain validation for backend server certificates for HTTP and VPC Link Integration. This feature enables customers to configure their REST APIs to integrate with backends that are secured with certificates vended from private certificate authorities (CA) or certificates that are self-signed.
* `service/cloudfront`: Updates service API and documentation
  * Amazon CloudFront adds support for configurable origin connection attempts and origin connection timeout.
* `service/elasticbeanstalk`: Updates service API and documentation
  * These API changes enable an IAM user to associate an operations role with an Elastic Beanstalk environment, so that the IAM user can call Elastic Beanstalk actions without having access to underlying downstream AWS services that these actions call.
* `service/personalize`: Updates service API and documentation
* `service/personalize-runtime`: Updates service API and documentation
* `service/pinpoint`: Updates service API and documentation
  * This release enables additional functionality for the Amazon Pinpoint journeys feature. With this release, you can send messages through additional channels, including SMS, push notifications, and custom channels.
* `service/runtime.sagemaker`: Updates service API and documentation
* `service/servicecatalog`: Updates service API and documentation
  * This release adds support for DescribeProduct and DescribeProductAsAdmin by product name, DescribeProvisioningArtifact by product name or provisioning artifact name, returning launch paths as part of DescribeProduct output and adds maximum length for provisioning artifact name and provisioning artifact description.

Release v1.31.11 (2020-06-04)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * New C5a instances, the latest generation of EC2's compute-optimized instances featuring AMD's 2nd Generation EPYC processors. C5a instances offer up to 96 vCPUs, 192 GiB of instance memory, 20 Gbps in Network bandwidth; New G4dn.metal bare metal instance with 8 NVIDIA T4 GPUs.
* `service/lightsail`: Updates service API and documentation
  * This release adds the BurstCapacityPercentage and BurstCapacityTime instance metrics, which allow you to track the burst capacity available to your instance.
* `service/mediapackage-vod`: Updates service API and documentation
* `service/meteringmarketplace`: Updates service documentation
  * Documentation updates for meteringmarketplace
* `service/ssm`: Updates service API and documentation
  * SSM State Manager support for executing an association only at specified CRON schedule after creating/updating an association.

### SDK Bugs
* `private/model`: Fixes SDK not enabling endpoint discovery when endpoint is set to empty string ([#3349](https://github.com/aws/aws-sdk-go/pull/3349))

Release v1.31.10 (2020-06-03)
===

### Service Client Updates
* `service/directconnect`: Updates service API and documentation
  * This release supports the virtual interface failover test, which allows you to verify that traffic routes over redundant virtual interfaces when you bring your primary virtual interface out of service.
* `service/elasticache`: Updates service API and documentation
  * This release improves the Multi-AZ feature in ElastiCache by adding a separate flag and proper validations.
* `service/es`: Updates service API, documentation, and paginators
  * Amazon Elasticsearch Service now offers support for cross-cluster search, enabling you to perform searches, aggregations, and visualizations across multiple Amazon Elasticsearch Service domains with a single query or from a single Kibana interface. New feature includes the ability to setup connection, required to perform cross-cluster search, between domains using an approval workflow.
* `service/glue`: Updates service API and documentation
  * Adding databaseName in the response for GetUserDefinedFunctions() API.
* `service/iam`: Updates service API and documentation
  * GenerateServiceLastAccessedDetails will now return ActionLastAccessed details for certain S3 control plane actions
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for the encoding of VP8 or VP9 video in WebM container with Vorbis or Opus audio.

Release v1.31.9 (2020-06-02)
===

### Service Client Updates
* `service/guardduty`: Updates service API and documentation
  * Amazon GuardDuty findings now include S3 bucket details under the resource section if an S3 Bucket was one of the affected resources

Release v1.31.8 (2020-06-01)
===

### Service Client Updates
* `service/athena`: Updates service API, documentation, and paginators
  * This release adds support for connecting Athena to your own Apache Hive Metastores in addition to the AWS Glue Data Catalog. For more information, please see https://docs.aws.amazon.com/athena/latest/ug/connect-to-data-source-hive.html
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR now supports encrypting log files with AWS Key Management Service (KMS) customer managed keys.
* `service/fsx`: Updates service API and documentation
* `service/kms`: Updates service API and documentation
  * AWS Key Management Service (AWS KMS): If the GenerateDataKeyPair or GenerateDataKeyPairWithoutPlaintext APIs are called on a CMK in a custom key store (origin == AWS_CLOUDHSM), they return an UnsupportedOperationException. If a call to UpdateAlias causes a customer to exceed the Alias resource quota, the UpdateAlias API returns a LimitExceededException.
* `service/sagemaker`: Updates service API and documentation
  * We are releasing HumanTaskUiArn as a new parameter in CreateLabelingJob and RenderUiTemplate which can take an ARN for a system managed UI to render a task.
* `service/worklink`: Updates service API and documentation

Release v1.31.7 (2020-05-28)
===

### Service Client Updates
* `service/kafka`: Updates service API and documentation
* `service/marketplace-catalog`: Updates service API and documentation
* `service/qldb-session`: Updates service documentation
* `service/workmail`: Updates service API and documentation
  * This release adds support for Amazon WorkMail organization-level retention policies.

Release v1.31.6 (2020-05-27)
===

### Service Client Updates
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/guardduty`: Updates service documentation
  * Documentation updates for GuardDuty

Release v1.31.5 (2020-05-26)
===

### Service Client Updates
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * ebsOptimizedInfo, efaSupported and supportedVirtualizationTypes added to DescribeInstanceTypes API
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache now allows you to use resource based policies to manage access to operations performed on ElastiCache resources. Also, Amazon ElastiCache now exposes ARN (Amazon Resource Names) for ElastiCache resources such as Cache Clusters and Parameter Groups. ARNs can be used to apply IAM policies to ElastiCache resources.
* `service/macie`: Updates service documentation, paginators, and examples
  * This is a documentation-only update to the Amazon Macie Classic API. This update corrects out-of-date references to the service name.
* `service/quicksight`: Updates service API and documentation
  * Add DataSetArns to QuickSight DescribeDashboard API response.
* `service/ssm`: Updates service API and documentation
  * The AWS Systems Manager GetOpsSummary API action now supports multiple OpsResultAttributes in the request. Currently, this feature only supports OpsResultAttributes with the following TypeNames: [AWS:EC2InstanceComputeOptimizer] or [AWS:EC2InstanceInformation, AWS:EC2InstanceComputeOptimizer]. These TypeNames can be used along with either or both of the following: [AWS:EC2InstanceRecommendation, AWS:RecommendationSource]

Release v1.31.4 (2020-05-22)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation
  * Documentation updates for Amazon EC2 Auto Scaling
* `service/iotsitewise`: Updates service API and documentation

Release v1.31.3 (2020-05-21)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * CodeBuild adds support for tagging with report groups
* `service/ec2`: Updates service API and documentation
  * From this release onwards ProvisionByoipCidr publicly supports IPv6. Updated ProvisionByoipCidr API to support tags for public IPv4 and IPv6 pools. Added NetworkBorderGroup to the DescribePublicIpv4Pools response.
* `service/s3`: Updates service API, documentation, and examples
  * Deprecates unusable input members bound to Content-MD5 header. Updates example and documentation.
* `service/synthetics`: Updates service API and documentation

Release v1.31.2 (2020-05-20)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation
* `service/appmesh`: Updates service API and documentation
* `service/backup`: Updates service API and documentation
* `service/chime`: Updates service API and documentation
  * Amazon Chime enterprise account administrators can now set custom retention policies on chat data in the Amazon Chime application.
* `service/codedeploy`: Updates service API and documentation
  * Amazon ECS customers using application and network load balancers can use CodeDeploy BlueGreen hook to invoke a CloudFormation stack update. With this update you can view CloudFormation deployment and target details via existing APIs and use your stack Id to list or delete all deployments associated with the stack.
* `service/medialive`: Updates service API, documentation, waiters, and paginators
  * AWS Elemental MediaLive now supports the ability to ingest the content that is streaming from an AWS Elemental Link device: https://aws.amazon.com/medialive/features/link/. This release also adds support for SMPTE-2038 and input state waiters.
* `service/securityhub`: Updates service API and documentation
* `service/transcribe-streaming`: Updates service API and documentation

### SDK Bugs
* `service/s3/s3crypto`: Add missing return in encryption client ([#3258](https://github.com/aws/aws-sdk-go/pull/3258))
  * Fixes a missing return in the encryption client that was causing a nil dereference panic.

Release v1.31.1 (2020-05-19)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * You can now receive Voice Connector call events through SNS or SQS.
* `service/ec2`: Updates service API and documentation
  * This release adds support for Federated Authentication via SAML-2.0 in AWS ClientVPN.
* `service/health`: Updates service API, documentation, and paginators
  * Feature: Health: AWS Health added a new field to differentiate Public events from Account-Specific events in the API request and response. Visit https://docs.aws.amazon.com/health/latest/APIReference/API_Event.html to learn more.
* `service/transcribe`: Updates service documentation

Release v1.31.0 (2020-05-18)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * Amazon Chime now supports redacting chat messages.
* `service/dynamodb`: Updates service documentation
  * Documentation updates for dynamodb
* `service/ec2`: Updates service API
  * This release changes the RunInstances CLI and SDK's so that if you do not specify a client token, a randomly generated token is used for the request to ensure idempotency.
* `service/ecs`: Updates service API and documentation
  * This release adds support for specifying environment files to add environment variables to your containers.
* `service/macie2`: Updates service API
* `service/qldb`: Updates service API, documentation, and paginators

### SDK Features
* `service/dynamodb/dynamodbattribute`: Support has been added for empty string and byte values.
  * `Encoder` has added two new configuration options for controlling whether empty string and byte values are sent as null or empty.
    * `NullEmptyString`: Whether string values that are empty will be sent as null (default: `true`).
    * `NullEmptyByteSlice`: Whether byte slice that are empty will be sent as null (default: `true`).
    * The default value for these options retrains the existing behavior of the SDK in prior releases.

Release v1.30.29 (2020-05-15)
===

### Service Client Updates
* `service/cloudformation`: Updates service API, documentation, waiters, and paginators
  * This release adds support for the following features: 1. DescribeType and ListTypeVersions APIs now output a field IsDefaultVersion, indicating if a version is the default version for its type; 2. Add StackRollbackComplete waiter feature to wait until stack status is UPDATE_ROLLBACK_COMPLETE; 3. Add paginators in DescribeAccountLimits, ListChangeSets, ListStackInstances, ListStackSetOperationResults, ListStackSetOperations, ListStackSets APIs.
* `service/ecr`: Updates service API and documentation
  * This release adds support for specifying an image manifest media type when pushing a manifest to Amazon ECR.
* `service/glue`: Updates service API and documentation
  * Starting today, you can stop the execution of Glue workflows that are running. AWS Glue workflows are directed acyclic graphs (DAGs) of Glue triggers, crawlers and jobs. Using a workflow, you can design a complex multi-job extract, transform, and load (ETL) activity that AWS Glue can execute and track as single entity.
* `service/sts`: Updates service API
  * API updates for STS

Release v1.30.28 (2020-05-14)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 now supports adding AWS resource tags for associations between VPCs and local gateways, at creation time.
* `service/imagebuilder`: Updates service API and documentation

Release v1.30.27 (2020-05-13)
===

### Service Client Updates
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache now supports auto-update of ElastiCache clusters after the "recommended apply by date" of  service update has passed. ElastiCache will use your maintenance window to schedule the auto-update of applicable clusters. For more information, see https://docs.aws.amazon.com/AmazonElastiCache/latest/mem-ug/Self-Service-Updates.html and https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/Self-Service-Updates.html
* `service/macie2`: Adds new service

Release v1.30.26 (2020-05-12)
===

### Service Client Updates
* `service/iotsitewise`: Updates service documentation
* `service/workmail`: Updates service API and documentation
  * Minor API fixes and updates to the documentation.

Release v1.30.25 (2020-05-11)
===

### Service Client Updates
* `service/codeguru-reviewer`: Updates service API and documentation
* `service/ec2`: Updates service API
  * M6g instances are our next-generation general purpose instances powered by AWS Graviton2 processors
* `service/kendra`: Updates service API and documentation
  * Amazon Kendra is now generally available. As part of general availability, we are launching * Developer edition * Ability to scale your Amazon Kendra index with capacity units * Support for new connectors * Support for new tagging API's * Support for Deleting data source * Metrics for data source sync operations * Metrics for query & storage utilization

Release v1.30.24 (2020-05-08)
===

### Service Client Updates
* `service/guardduty`: Updates service documentation
  * Documentation updates for GuardDuty
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation updates for resourcegroupstaggingapi
* `service/sagemaker`: Updates service API and documentation
  * This release adds a new parameter (EnableInterContainerTrafficEncryption) to CreateProcessingJob API to allow for enabling inter-container traffic encryption on processing jobs.

### SDK Bugs
* `service/dynamodb/dynamodbattribute`:  Simplified decode logic to decode AttributeValue as it is defined ([#3308](https://github.com/aws/aws-sdk-go/pull/3308))

Release v1.30.23 (2020-05-07)
===

### Service Client Updates
* `service/appconfig`: Updates service documentation
* `service/codebuild`: Updates service API, documentation, and paginators
  * Add COMMIT_MESSAGE enum for webhook filter types
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 now adds warnings to identify issues when creating a launch template or launch template version.
* `service/lightsail`: Updates service API and documentation
  * This release adds support for the following options in instance public ports: Specify source IP addresses, specify ICMP protocol like PING, and enable/disable the Lightsail browser-based SSH and RDP clients' access to your instance.
* `service/logs`: Updates service API and documentation
  * Amazon CloudWatch Logs now offers the ability to interact with Logs Insights queries via the new PutQueryDefinition, DescribeQueryDefinitions, and DeleteQueryDefinition APIs.
* `service/route53`: Updates service API
  * Amazon Route 53 now supports the EU (Milan) Region (eu-south-1) for latency records, geoproximity records, and private DNS for Amazon VPCs in that region.
* `service/ssm`: Updates service API
  * This Patch Manager release supports creating patch baselines for Oracle Linux and Debian

Release v1.30.22 (2020-05-06)
===

### Service Client Updates
* `service/codestar-connections`: Updates service API and documentation
* `service/comprehendmedical`: Updates service API and documentation

Release v1.30.21 (2020-05-05)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * With this release, you can call ModifySubnetAttribute with two new parameters: MapCustomerOwnedIpOnLaunch and CustomerOwnedIpv4Pool, to map a customerOwnedIpv4Pool to a subnet. You will also see these two new fields in the DescribeSubnets response. If your subnet has a customerOwnedIpv4Pool mapped, your network interface will get an auto assigned customerOwnedIpv4 address when placed onto an instance.
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Parameter Store launches new data type to support aliases in EC2 APIs
* `service/support`: Updates service documentation
  * Documentation updates for support

Release v1.30.20 (2020-05-04)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Documentation updates for Amazon API Gateway
* `service/ec2`: Updates service documentation
  * With this release, you can include enriched metadata in Amazon Virtual Private Cloud (Amazon VPC) flow logs published to Amazon CloudWatch Logs or Amazon Simple Storage Service (S3). Prior to this, custom format VPC flow logs enriched with additional metadata could be published only to S3. With this launch, we are also adding additional metadata fields that provide insights about the location such as AWS Region, AWS Availability Zone, AWS Local Zone, AWS Wavelength Zone, or AWS Outpost where the network interface where flow logs are captured exists.
* `service/s3control`: Updates service API and documentation
  * Amazon S3 Batch Operations now supports Object Lock.

Release v1.30.19 (2020-05-01)
===

### Service Client Updates
* `service/elasticfilesystem`: Updates service API
  * Change the TagKeys argument for UntagResource to a URL parameter to address an issue with the Java and .NET SDKs.
* `service/ssm`: Updates service API and documentation
  * Added TimeoutSeconds as part of ListCommands API response.

Release v1.30.18 (2020-04-30)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * AWS IoT Core released Fleet Provisioning for scalable onboarding of IoT devices to the cloud. This release includes support for customer's Lambda functions to validate devices during onboarding. Fleet Provisioning also allows devices to send Certificate Signing Requests (CSR) to AWS IoT Core for signing and getting a unique certificate. Lastly,  AWS IoT Core added a feature to register the same certificate for multiple accounts in the same region without needing to register the certificate authority (CA).
* `service/iotevents`: Updates service API and documentation
* `service/lambda`: Updates service documentation and examples
  * Documentation updates for Lambda
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for including AFD signaling in MXF wrapper.
* `service/schemas`: Updates service API and documentation
* `service/storagegateway`: Updates service API
  * Adding support for S3_INTELLIGENT_TIERING as a storage class option

Release v1.30.17 (2020-04-29)
===

### Service Client Updates
* `service/iotsitewise`: Adds new service
* `service/servicediscovery`: Updates service documentation and examples
  * Documentation updates for servicediscovery
* `service/transcribe`: Updates service API, documentation, and paginators
* `service/waf`: Updates service API and documentation
  * This release add migration API for AWS WAF Classic ("waf" and "waf-regional"). The migration API will parse through your web ACL and generate a CloudFormation template into your S3 bucket. Deploying this template will create equivalent web ACL under new AWS WAF ("wafv2").
* `service/waf-regional`: Updates service API and documentation

Release v1.30.16 (2020-04-28)
===

### Service Client Updates
* `service/ecr`: Updates service API and documentation
  * This release adds support for multi-architecture images also known as a manifest list
* `service/kinesis-video-archived-media`: Updates service API and documentation
* `service/kinesisvideo`: Updates service API and documentation
  * Add "GET_CLIP" to the list of supported API names for the GetDataEndpoint API.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports several new features: enhanced VQ for H.264 (AVC) output encodes; passthrough of timed metadata and of Nielsen ID3 metadata in fMP4 containers in HLS outputs; the ability to generate a SCTE-35 sparse track without additional segmentation, in Microsoft Smooth outputs;  the ability to select the audio from a TS input by specifying the audio track; and conversion of HDR colorspace in the input to an SDR colorspace in the output.
* `service/route53`: Updates service API, documentation, and paginators
  * Amazon Route 53 now supports the Africa (Cape Town) Region (af-south-1) for latency records, geoproximity records, and private DNS for Amazon VPCs in that region.
* `service/ssm`: Updates service API and documentation
  * SSM State Manager support for adding list association filter for Resource Group and manual mode of managing compliance for an association.

### SDK Bugs
* `service/s3`: Fix S3 client behavior wrt 200 OK response with empty payload

Release v1.30.15 (2020-04-27)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service API and documentation
* `service/dataexchange`: Updates service API and documentation
* `service/dms`: Updates service API and documentation
  * Adding minimum replication engine version for describe-endpoint-types api.
* `service/sagemaker`: Updates service API and documentation
  * Change to the input, ResourceSpec, changing EnvironmentArn to SageMakerImageArn. This affects the following preview APIs: CreateDomain, DescribeDomain, UpdateDomain, CreateUserProfile, DescribeUserProfile, UpdateUserProfile, CreateApp and DescribeApp.

Release v1.30.14 (2020-04-24)
===

### Service Client Updates
* `service/dlm`: Updates service documentation
* `service/elastic-inference`: Updates service API, documentation, and paginators
* `service/iot`: Updates service API
  * This release adds a new exception type to the AWS IoT SetV2LoggingLevel API.

Release v1.30.13 (2020-04-23)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API, documentation, and examples
* `service/firehose`: Updates service API and documentation
  * You can now deliver streaming data to an Amazon Elasticsearch Service domain in an Amazon VPC. You can now compress streaming data delivered to S3 using Hadoop-Snappy in addition to Gzip, Zip and Snappy formats.
* `service/mediapackage-vod`: Updates service API and documentation
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API enhances support for sending campaigns through custom channels to locations such as AWS Lambda functions or web applications. Campaigns can now use CustomDeliveryConfiguration and CampaignCustomMessage to configure custom channel settings for a campaign.
* `service/ram`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Adds support for AWS Local Zones, including a new optional parameter AvailabilityZoneGroup for the DescribeOrderableDBInstanceOptions operation.
* `service/storagegateway`: Updates service API and documentation
  * Added AutomaticTapeCreation APIs
* `service/transfer`: Updates service API and documentation
  * This release adds support for transfers over FTPS and FTP in and out of Amazon S3, which makes it easy to migrate File Transfer Protocol over SSL (FTPS) and FTP workloads to AWS, in addition to the existing support for Secure File Transfer Protocol (SFTP).

### SDK Enhancements
* `aws/credentials/stscreds`: Add support for policy ARNs ([#3249](https://github.com/aws/aws-sdk-go/pull/3249))
  * Adds support for passing AWS policy ARNs to the `AssumeRoleProvider` and `WebIdentityRoleProvider` credential providers. This allows you provide policy ARNs when assuming the role that will further limit the permissions of the credentials returned.

Release v1.30.12 (2020-04-22)
===

### Service Client Updates
* `service/codeguru-reviewer`: Updates service API, documentation, and paginators
* `service/es`: Updates service API and documentation
  * This change adds a new field 'OptionalDeployment' to ServiceSoftwareOptions to indicate whether a service software update is optional or mandatory. If True, it indicates that the update is optional, and the service software is not automatically updated. If False, the service software is automatically updated after AutomatedUpdateDate.
* `service/fms`: Updates service API and documentation
* `service/redshift`: Updates service API, documentation, and paginators
  * Amazon Redshift support for usage limits
* `service/transcribe-streaming`: Updates service API and documentation

### SDK Enhancements
* `aws/credentials/stscreds`: Add support for custom web identity TokenFetcher ([#3256](https://github.com/aws/aws-sdk-go/pull/3256))
  * Adds new constructor, `NewWebIdentityRoleProviderWithToken` for `WebIdentityRoleProvider` which takes a `TokenFetcher`. Implement `TokenFetcher` to provide custom sources for web identity tokens. The `TokenFetcher` must be concurrency safe. `TokenFetcher` may return unique value each time it is called.

Release v1.30.11 (2020-04-21)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR adds support for configuring a managed scaling policy for an Amazon EMR cluster. This enables automatic resizing of a cluster to optimize for job execution speed and reduced cluster cost.
* `service/guardduty`: Updates service API, documentation, and paginators
  * AWS GuardDuty now supports using AWS Organizations delegated administrators to create and manage GuardDuty master and member accounts.  The feature also allows GuardDuty to be automatically enabled on associated organization accounts.
* `service/route53domains`: Updates service API and documentation
  * You can now programmatically transfer domains between AWS accounts without having to contact AWS Support

Release v1.30.10 (2020-04-20)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * You can now export an OpenAPI 3.0 compliant API definition file for Amazon API Gateway HTTP APIs using the Export API.
* `service/ce`: Updates service API, documentation, and paginators
* `service/glue`: Updates service API and documentation
  * Added a new ConnectionType "KAFKA" and a ConnectionProperty "KAFKA_BOOTSTRAP_SERVERS" to support Kafka connection.
* `service/iotevents`: Updates service API and documentation
* `service/synthetics`: Adds new service

Release v1.30.9 (2020-04-17)
===

### Service Client Updates
* `service/frauddetector`: Updates service API and documentation
* `service/opsworkscm`: Updates service documentation and paginators
  * Documentation updates for opsworkscm

Release v1.30.8 (2020-04-16)
===

### Service Client Updates
* `service/AWSMigrationHub`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 now supports adding AWS resource tags for placement groups and key pairs, at creation time. The CreatePlacementGroup API will now return placement group information when created successfully. The DeleteKeyPair API now supports deletion by resource ID.
* `service/glue`: Updates service API
  * This release adds support for querying GetUserDefinedFunctions API without databaseName.
* `service/imagebuilder`: Updates service API and documentation
* `service/iotevents`: Updates service API and documentation
* `service/lambda`: Updates service documentation and examples
  * Sample code for AWS Lambda operations
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert now allows you to specify your input captions frame rate for SCC captions sources.
* `service/mediatailor`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * This release adds support for Amazon RDS Proxy with PostgreSQL compatibility.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker now supports running training jobs on ml.g4dn and ml.c5n instance types. Amazon SageMaker supports in "IN" operation for Search now.
* `service/sagemaker-a2i-runtime`: Updates service API and documentation
* `service/securityhub`: Updates service API and documentation
* `service/snowball`: Updates service API
  * An update to the Snowball Edge Storage Optimized device has been launched. Like the previous version, it has 80 TB of capacity for data transfer. Now it has 40 vCPUs, 80 GiB, and a 1 TiB SATA SSD of memory for EC2 compatible compute. The 80 TB of capacity can also be used for EBS-like volumes for AMIs.

Release v1.30.7 (2020-04-08)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * feature: Chime: This release introduces the ability to tag Amazon Chime SDK meeting resources.  You can use tags to organize and identify your resources for cost allocation.
* `service/cloudformation`: Updates service documentation
  * The OrganizationalUnitIds parameter on StackSet and the OrganizationalUnitId parameter on StackInstance, StackInstanceSummary, and StackSetOperationResultSummary are now reserved for internal use. No data is returned for this parameter.
* `service/codeguruprofiler`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API and documentation
  * This release provides the ability to include tags in EC2 event notifications.
* `service/ecs`: Updates service API and documentation
  * This release provides native support for specifying Amazon EFS file systems as volumes in your Amazon ECS task definitions.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK adds support for queue hopping. Jobs can now hop from their original queue to a specified alternate queue, based on the maximum wait time that you specify in the job settings.
* `service/migrationhub-config`: Updates service API and documentation

### SDK Enhancements
* `example/service/ecr`: Add create and delete repository examples ([#3221](https://github.com/aws/aws-sdk-go/pull/3221))
  * Adds examples demonstrating how you can create and delete repositories with the SDK.

Release v1.30.6 (2020-04-07)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Documentation updates for Amazon API Gateway.
* `service/codeguru-reviewer`: Updates service API
* `service/mediaconnect`: Updates service API and documentation

Release v1.30.5 (2020-04-06)
===

### Service Client Updates
* `service/chime`: Updates service API, documentation, and paginators
  * Amazon Chime proxy phone sessions let you provide two users with a shared phone number to communicate via voice or text for up to 12 hours without revealing personal phone numbers. When users call or message the provided phone number, they are connected to the other party and their private phone numbers are replaced with the shared number in Caller ID.
* `service/elasticbeanstalk`: Updates service API, documentation, and paginators
  * This release adds a new action, ListPlatformBranches, and updates two actions, ListPlatformVersions and DescribePlatformVersion, to support the concept of Elastic Beanstalk platform branches.
* `service/iam`: Updates service documentation
  * Documentation updates for AWS Identity and Access Management (IAM).
* `service/transcribe`: Updates service API, documentation, and paginators

Release v1.30.4 (2020-04-03)
===

### Service Client Updates
* `service/personalize-runtime`: Updates service API and documentation
* `service/robomaker`: Updates service API and documentation

Release v1.30.3 (2020-04-02)
===

### Service Client Updates
* `service/gamelift`: Updates service API and documentation
  * Public preview of GameLift FleetIQ as a standalone feature. GameLift FleetIQ makes it possible to use low-cost Spot instances by limiting the chance of interruptions affecting game sessions. FleetIQ is a feature of the managed GameLift service, and can now be used with game hosting in EC2 Auto Scaling groups that you manage in your own account.
* `service/medialive`: Updates service API, documentation, and waiters
  * AWS Elemental MediaLive now supports Automatic Input Failover. This feature provides resiliency upstream of the channel, before ingest starts.
* `service/monitoring`: Updates service API and documentation
  * Amazon CloudWatch Contributor Insights adds support for tags and tagging on resource creation.
* `service/rds`: Updates service documentation
  * Documentation updates for RDS: creating read replicas is now supported for SQL Server DB instances
* `service/redshift`: Updates service documentation
  * Documentation updates for redshift

### SDK Enhancements
* `aws/credentials`: `ProviderWithContext` optional interface has been added to support passing contexts on credential retrieval ([#3223](https://github.com/aws/aws-sdk-go/pull/3223))
  * Credential providers that implement the optional `ProviderWithContext` will have context passed to them
  * `ec2rolecreds.EC2RoleProvider`, `endpointcreds.Provider`, `stscreds.AssumeRoleProvider`, `stscreds.WebIdentityRoleProvider` have been updated to support the `ProviderWithContext` interface
  * Fixes [#3213](https://github.com/aws/aws-sdk-go/issues/3213)
* `aws/ec2metadata`: Context aware operations have been added `EC2Metadata` client ([#3223](https://github.com/aws/aws-sdk-go/pull/3223))

Release v1.30.2 (2020-04-01)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * This release introduces Dimensions for AWS IoT Device Defender. Dimensions can be used in Security Profiles to collect and monitor fine-grained metrics.
* `service/mediaconnect`: Updates service API and documentation

Release v1.30.1 (2020-03-31)
===

### Service Client Updates
* `service/appconfig`: Updates service API and documentation
* `service/detective`: Updates service documentation
* `service/elastic-inference`: Updates service API
* `service/fms`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * Add two enums for MongoDB connection: Added "CONNECTION_URL" to "ConnectionPropertyKey" and added "MONGODB" to "ConnectionType"
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports .NET Core 3.1
* `service/mediastore`: Updates service API and documentation
  * This release adds support for CloudWatch Metrics. You can now set a policy on your container to dictate which metrics MediaStore sends to CloudWatch.
* `service/opsworkscm`: Updates service documentation
  * Documentation updates for OpsWorks-CM CreateServer values.
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces MMS support for SMS messages.
* `service/rekognition`: Updates service API and documentation
  * This release adds DeleteProject and DeleteProjectVersion APIs to Amazon Rekognition Custom Labels.
* `service/storagegateway`: Updates service API and documentation
  * Adding audit logging support for SMB File Shares
* `service/wafv2`: Updates service API and documentation

Release v1.30.0 (2020-03-30)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service API and documentation

### SDK Features
* SDK generated errors are fixed to use pointer receivers preventing confusion, and potential impossible type assertions. The SDK will only return API generated API error types as pointers. This fix ensures Go's type system will catch invalid error type assertions.

### SDK Enhancements
* Update SDK's `go-jmespath` dependency to latest tagged version `0.3.0` ([#3205](https://github.com/aws/aws-sdk-go/pull/3205))

### SDK Bugs
* Fix generated SDK errors to use pointer receivers
  * Fixes the generated SDK API errors to use pointer function receivers instead of value. This fixes potential confusion writing code and not casting to the correct type. The SDK will always return the API error as a pointer, not value.
  * Code that did type assertions from the operation's returned error to the value type would never be satisfied. Leading to errors being missed. Changing the function receiver to a pointer prevents this error. Highlighting it in code bases.

Release v1.29.34 (2020-03-27)
===

### Service Client Updates
* `service/globalaccelerator`: Updates service API and documentation
* `service/kendra`: Updates service API and documentation
  * The Amazon Kendra Microsoft SharePoint data source now supports include and exclude regular expressions and change log features. Include and exclude regular expressions enable you to  provide a list of regular expressions to match the display URL of SharePoint documents to either include or exclude documents respectively. When you enable the changelog feature it enables Amazon Kendra to use the SharePoint change log to determine which documents to update in the index.
* `service/servicecatalog`: Updates service documentation
  * Added "LocalRoleName" as an acceptable Parameter for Launch type in CreateConstraint and UpdateConstraint APIs

Release v1.29.33 (2020-03-26)
===

### Service Client Updates
* `service/fsx`: Updates service API and documentation
* `service/sagemaker`: Updates service API and documentation
  * This release updates Amazon Augmented AI CreateFlowDefinition API and DescribeFlowDefinition response.
* `service/securityhub`: Updates service API and documentation

Release v1.29.32 (2020-03-25)
===

### Service Client Updates
* `service/application-insights`: Updates service API and documentation
* `service/ce`: Updates service API and documentation
* `service/detective`: Updates service API and documentation
* `service/es`: Updates service API, documentation, and paginators
  * Adding support for customer packages (dictionary files) to Amazon Elasticsearch Service
* `service/managedblockchain`: Updates service API and documentation
* `service/xray`: Updates service API and documentation
  * GetTraceSummaries - Now provides additional root cause attribute ClientImpacting which indicates whether root cause impacted trace client.

Release v1.29.31 (2020-03-24)
===

### Service Client Updates
* `service/athena`: Updates service documentation
  * Documentation updates for Athena, including QueryExecutionStatus QUEUED and RUNNING states. QUEUED now indicates that the query has been submitted to the service. RUNNING indicates that the query is in execution phase.
* `service/eks`: Updates service API and documentation
* `service/organizations`: Updates service API, documentation, and paginators
  * Introduces actions for giving a member account administrative Organizations permissions for an AWS service. You can run this action only for AWS services that support this feature.
* `service/rds-data`: Updates service documentation

Release v1.29.30 (2020-03-23)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * Documentation updates to reflect that the default timeout for integrations is now 30 seconds for HTTP APIs.
* `service/eks`: Updates service API and documentation
* `service/route53`: Updates service documentation
  * Documentation updates for Route 53.

Release v1.29.29 (2020-03-20)
===

### Service Client Updates
* `service/servicecatalog`: Updates service API and documentation
  * Added "productId" and "portfolioId" to responses from CreateConstraint, UpdateConstraint, ListConstraintsForPortfolio, and DescribeConstraint APIs

Release v1.29.28 (2020-03-19)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * AWS Certificate Manager documentation updated on API calls ImportCertificate and ListCertificate. Specific updates included input constraints, private key size for import and next token size for list.
* `service/outposts`: Updates service documentation

Release v1.29.27 (2020-03-18)
===

### Service Client Updates
* `service/mediaconnect`: Updates service API and documentation
* `service/personalize`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Updated the MaxRecords type in DescribeExportTasks to Integer.

Release v1.29.26 (2020-03-17)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for: AV1 encoding in File Group MP4, DASH and CMAF DASH outputs; PCM/WAV audio output in MPEG2-TS containers; and Opus audio in Webm inputs.

Release v1.29.25 (2020-03-16)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release adds the ability to update the task placement strategy and constraints for Amazon ECS services.
* `service/elasticache`: Updates service API, documentation, and paginators
  * Amazon ElastiCache now supports Global Datastore for Redis. Global Datastore for Redis offers fully managed, fast, reliable and secure cross-region replication. Using Global Datastore for Redis, you can create cross-region read replica clusters for ElastiCache for Redis to enable low-latency reads and disaster recovery across regions. You can create, modify and describe a Global Datastore, as well as add or remove regions from your Global Datastore and promote a region as primary in Global Datastore.
* `service/s3control`: Updates service API and documentation
  * Amazon S3 now supports Batch Operations job tagging.
* `service/ssm`: Updates service API and documentation
  * Resource data sync for AWS Systems Manager Inventory now includes destination data sharing. This feature enables you to synchronize inventory data from multiple AWS accounts into a central Amazon S3 bucket. To use this feature, all AWS accounts must be listed in AWS Organizations.

Release v1.29.24 (2020-03-13)
===

### Service Client Updates
* `service/appconfig`: Updates service documentation

Release v1.29.23 (2020-03-12)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * Amazon API Gateway HTTP APIs is now generally available. HTTP APIs offer the core functionality of REST API at up to 71% lower price compared to REST API, 60% lower p99 latency, and is significantly easier to use. As part of general availability, we added new features to route requests to private backends such as private ALBs, NLBs, and IP/ports. We also brought over a set of features from REST API such as Stage Variables, and Stage/Route level throttling. Custom domain names can also now be used with both REST And HTTP APIs.
* `service/ec2`: Updates service documentation
  * Documentation updates for EC2
* `service/iot`: Updates service API and documentation
  * As part of this release, we are extending capability of AWS IoT Rules Engine to support IoT Cloudwatch log action. The IoT Cloudwatch log rule action lets you send messages from IoT sensors and applications to Cloudwatch logs for troubleshooting and debugging.
* `service/lex-models`: Updates service API and documentation
* `service/securityhub`: Updates service API and documentation

Release v1.29.22 (2020-03-11)
===

### Service Client Updates
* `service/elasticfilesystem`: Updates service documentation
  * Documentation updates for elasticfilesystem
* `service/redshift`: Updates service API and documentation
  * Amazon Redshift now supports operations to pause and resume a cluster on demand or on a schedule.

Release v1.29.21 (2020-03-10)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Documentation updates for EC2
* `service/iotevents`: Updates service API and documentation
* `service/marketplacecommerceanalytics`: Updates service documentation
  * Change the disbursement data set to look past 31 days instead until the beginning of the month.
* `service/serverlessrepo`: Updates service API and documentation

### SDK Enhancements
* `aws/credentials`: Clarify `token` usage in `NewStaticCredentials` documentation.
  * Related to [#3162](https://github.com/aws/aws-sdk-go/issues/3162).
* `service/s3/s3manager`: Improve memory allocation behavior by replacing sync.Pool with custom pool implementation ([#3183](https://github.com/aws/aws-sdk-go/pull/3183))
  * Improves memory allocations that occur when the provided `io.Reader` to upload does not satisfy both the `io.ReaderAt` and `io.ReadSeeker` interfaces.
  * Fixes [#3075](https://github.com/aws/aws-sdk-go/issues/3075)

Release v1.29.20 (2020-03-09)
===

### Service Client Updates
* `service/dms`: Updates service API and documentation
  * Added new settings for Kinesis target to include detailed transaction info; to capture table DDL details; to use single-line unformatted json, which can be directly queried by AWS Athena if data is streamed into S3 through AWS Kinesis Firehose. Added CdcInsertsAndUpdates in S3 target settings to allow capture ongoing insertions and updates only.
* `service/ec2`: Updates service API and documentation
  * Amazon Virtual Private Cloud (VPC) NAT Gateway adds support for tagging on resource creation.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports the ability to configure the Preferred Channel Pipeline for channels contributing to a Multiplex.

Release v1.29.19 (2020-03-06)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release provides customers with a self-service option to enable Local Zones.
* `service/guardduty`: Updates service API and documentation
  * Amazon GuardDuty findings now include the OutpostArn if the finding is generated for an AWS Outposts EC2 host.
* `service/robomaker`: Updates service API and documentation
* `service/signer`: Updates service API and documentation
  * This release enables signing image format override in PutSigningProfile requests, adding two more enum fields, JSONEmbedded and JSONDetached. This release also extends the length limit of SigningProfile name from 20 to 64.

Release v1.29.18 (2020-03-05)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * You can now create AWS Client VPN Endpoints with a specified VPC and Security Group. Additionally, you can modify these attributes when modifying the endpoint.
* `service/eks`: Updates service API and documentation
* `service/guardduty`: Updates service API and documentation
  * Add a new finding field for EC2 findings indicating the instance's local IP address involved in the threat.
* `service/opsworkscm`: Updates service API
  * Updated the Tag regex pattern to align with AWS tagging APIs.

Release v1.29.17 (2020-03-04)
===

### Service Client Updates
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces support for integrating recommender models with email, push notification, and SMS message templates. You can now use these types of templates to connect to recommender models and add personalized recommendations to messages that you send from campaigns and journeys.

### SDK Bugs
* `service/s3/s3manager`: Fix resource leak on UploadPart failures ([#3144](https://github.com/aws/aws-sdk-go/pull/3144))

Release v1.29.16 (2020-03-03)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and paginators
  * Amazon VPC Flow Logs adds support for tags and tagging on resource creation.

Release v1.29.15 (2020-03-02)
===

### Service Client Updates
* `service/comprehendmedical`: Updates service API and documentation
* `service/monitoring`: Updates service API, documentation, waiters, and paginators
  * Introducing Amazon CloudWatch Composite Alarms

Release v1.29.14 (2020-02-29)
===

### Service Client Updates
* `service/config`: Updates service API and documentation

Release v1.29.13 (2020-02-28)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service paginators
* `service/appmesh`: Updates service API and documentation
* `service/codeguruprofiler`: Updates service documentation
* `service/config`: Updates service API, documentation, and paginators
* `service/elasticloadbalancingv2`: Updates service documentation
* `service/glue`: Updates service API, documentation, and paginators
  * AWS Glue adds resource tagging support for Machine Learning Transforms and adds a new API, ListMLTransforms to support tag filtering.  With this feature, customers can use tags in AWS Glue to organize and control access to Machine Learning Transforms.
* `service/quicksight`: Updates service API, documentation, and paginators
  * Added SearchDashboards API that allows listing of dashboards that a specific user has access to.
* `service/sagemaker-a2i-runtime`: Updates service API and documentation
* `service/workdocs`: Updates service documentation
  * Documentation updates for workdocs

Release v1.29.12 (2020-02-27)
===

### Service Client Updates
* `service/globalaccelerator`: Updates service API and documentation
* `service/lightsail`: Updates service API and documentation
  * Adds support to create notification contacts in Amazon Lightsail, and to create instance, database, and load balancer metric alarms that notify you based on the value of a metric relative to a threshold that you specify.

Release v1.29.11 (2020-02-26)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release changes the RunInstances CLI and SDK's so that if you do not specify a client token, a randomly generated token is used for the request to ensure idempotency.
* `service/sagemaker`: Updates service API and documentation
  * SageMaker UpdateEndpoint API now supports retained variant properties, e.g., instance count, variant weight. SageMaker ListTrials API filter by TrialComponentName. Make ExperimentConfig name length limits consistent with CreateExperiment, CreateTrial, and CreateTrialComponent APIs.
* `service/securityhub`: Updates service API and documentation
* `service/transcribe`: Updates service API and documentation

Release v1.29.10 (2020-02-25)
===

### Service Client Updates
* `service/kafka`: Updates service API and documentation
* `service/outposts`: Updates service API and documentation
* `service/secretsmanager`: Updates service API and documentation
  * This release increases the maximum allowed size of SecretString or SecretBinary from 10KB to 64KB in the CreateSecret, UpdateSecret, PutSecretValue and GetSecretValue APIs.
* `service/states`: Updates service API and documentation
  * This release adds support for CloudWatch Logs for Standard Workflows.

Release v1.29.9 (2020-02-24)
===

### Service Client Updates
* `service/docdb`: Updates service documentation
  * Documentation updates for docdb
* `service/eventbridge`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * This release allows you to create and manage tags for event buses.
* `service/fsx`: Updates service API and documentation
  * Announcing persistent file systems for Amazon FSx for Lustre that are ideal for longer-term storage and workloads, and a new generation of scratch file systems that offer higher burst throughput for spiky workloads.
* `service/iotevents`: Updates service documentation
* `service/snowball`: Updates service API and documentation
  * AWS Snowball adds a field for entering your GSTIN when creating AWS Snowball jobs in the Asia Pacific (Mumbai) region.

Release v1.29.8 (2020-02-21)
===

### Service Client Updates
* `service/imagebuilder`: Updates service API and documentation
* `service/redshift`: Updates service API and documentation
  * Extend elastic resize to support resizing clusters to different instance types.
* `service/wafv2`: Updates service API and documentation

Release v1.29.7 (2020-02-20)
===

### Service Client Updates
* `service/appconfig`: Updates service API and documentation
* `service/pinpoint`: Updates service API
  * As of this release of the Amazon Pinpoint API, the Title property is optional for the CampaignEmailMessage object.
* `service/savingsplans`: Updates service API

Release v1.29.6 (2020-02-19)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation
  * Doc update for EC2 Auto Scaling: Add Enabled parameter for PutScalingPolicy
* `service/lambda`: Updates service API, documentation, and examples
  * AWS Lambda now supports Ruby 2.7
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * "ListPortfolioAccess" API now has a new optional parameter "OrganizationParentId". When it is provided and if the portfolio with the "PortfolioId" given was shared with an organization or organizational unit with "OrganizationParentId", all accounts in the organization sub-tree under parent which inherit an organizational portfolio share will be listed, rather than all accounts with external shares. To accommodate long lists returned from the new option, the API now supports pagination.

Release v1.29.5 (2020-02-18)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Amazon EC2 Auto Scaling now supports the ability to enable/disable target tracking, step scaling, and simple scaling policies.
* `service/chime`: Updates service API and documentation
  * Added AudioFallbackUrl to support Chime SDK client.
* `service/rds`: Updates service API and documentation
  * This release supports Microsoft Active Directory authentication for Amazon Aurora.

Release v1.29.4 (2020-02-17)
===

### Service Client Updates
* `service/cloud9`: Updates service API and documentation
  * AWS Cloud9 now supports the ability to tag Cloud9 development environments.
* `service/ec2`: Updates service API and documentation
  * Documentation updates for EC2
* `service/rekognition`: Updates service API, documentation, and paginators
  * This update adds the ability to detect text in videos and adds filters to image and video text detection.
* ` service/dynamodb`: Add feature update for Amazon DynamoDB
  * Amazon DynamoDB enables you to restore your DynamoDB backup or table data across AWS Regions such that the restored table is created in a different AWS Region from where the source table or backup resides. You can do cross-region restores between AWS commercial Regions, AWS China Regions, and AWS GovCloud (US) Regions.

Release v1.29.3 (2020-02-14)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * You can now enable Multi-Attach on Provisioned IOPS io1 volumes through the create-volume API.
* `service/mediatailor`: Updates service API and documentation
* `service/securityhub`: Updates service API, documentation, and paginators
* `service/shield`: Updates service API and documentation
  * This release adds support for associating Amazon Route 53 health checks to AWS Shield Advanced protected resources.

### SDK Enhancements
* `aws/credentials`: Add support for context when getting credentials.
  * Adds `GetWithContext` to `Credentials` that allows canceling getting the credentials if the context is canceled, or times out. This fixes an issue where API operations would ignore their provide context when waiting for credentials to refresh.
  * Related to [#3127](https://github.com/aws/aws-sdk-go/pull/3127).

Release v1.29.2 (2020-02-13)
===

### Service Client Updates
* `service/mediapackage-vod`: Updates service API and documentation

Release v1.29.1 (2020-02-12)
===

### Service Client Updates
* `service/chime`: Updates service documentation
  * Documentation updates for Amazon Chime
* `service/ds`: Updates service API and documentation
  * Release to add the ExpirationDateTime as an output to ListCertificates so as to ease customers to look into their certificate lifetime and make timely decisions about renewing them.
* `service/ec2`: Updates service API and documentation
  * This release adds support for tagging public IPv4 pools.
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service now offers fine-grained access control, which adds multiple capabilities to give tighter control over data. New features include the ability to use roles to define granular permissions for indices, documents, or fields and to extend Kibana with read-only views and secure multi-tenant support.
* `service/glue`: Updates service API and documentation
  * Adding ability to add arguments that cannot be overridden to AWS Glue jobs
* `service/neptune`: Updates service API and documentation
  * This launch enables Neptune start-db-cluster and stop-db-cluster. Stopping and starting Amazon Neptune clusters helps you manage costs for development and test environments. You can temporarily stop all the DB instances in your cluster, instead of setting up and tearing down all the DB instances each time that you use the cluster.
* `service/workmail`: Updates service API and documentation
  * This release adds support for access control rules management  in Amazon WorkMail.

### SDK Enhancements
* `aws/credentials`: Add grouping of concurrent refresh of credentials ([#3127](https://github.com/aws/aws-sdk-go/pull/3127/)
  * Concurrent calls to `Credentials.Get` are now grouped in order to prevent numerous synchronous calls to refresh the credentials. Replacing the mutex with a singleflight reduces the overall amount of time request signatures need to wait while retrieving credentials. This is improvement becomes pronounced when many requests are being made concurrently.

Release v1.29.0 (2020-02-11)
===

### Service Client Updates
* `service/cloudformation`: Updates service API and documentation
  * This release of AWS CloudFormation StackSets allows you to centrally manage deployments to all the accounts in your organization or specific organizational units (OUs) in AWS Organizations. You will also be able to enable automatic deployments to any new accounts added to your organization or OUs. The permissions needed to deploy across accounts will automatically be taken care of by the StackSets service.
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 Now Supports Tagging Spot Fleet.

### SDK Features
* Remove SDK's `vendor` directory of vendored dependencies
  * Updates the SDK's Go module definition to enumerate all dependencies of the SDK and its components. 
  * SDK's repository root package has been updated to refer to runtime dependencies like `go-jmespath` for `go get` the SDK with Go without modules.
* Deletes the deprecated `awsmigrate` utility from the SDK's repository.
  * This utility is no longer relevant. The utility allowed users the beta pre-release v0 SDK to update to the v1.0 released version of the SDK.

Release v1.28.14 (2020-02-10)
===

### Service Client Updates
* `service/docdb`: Updates service documentation
  * Added clarifying information that Amazon DocumentDB shares operational technology with Amazon RDS and Amazon Neptune.
* `service/kms`: Updates service API and documentation
  * The ConnectCustomKeyStore API now provides a new error code (SUBNET_NOT_FOUND) for customers to better troubleshoot if their "connect-custom-key-store" operation fails.

Release v1.28.13 (2020-02-07)
===

### Service Client Updates
* `service/imagebuilder`: Updates service API and documentation
* `service/rds`: Updates service documentation
  * Documentation updates for RDS: when restoring a DB cluster from a snapshot, must create DB instances
* `service/robomaker`: Updates service API, documentation, and paginators

Release v1.28.12 (2020-02-06)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/codebuild`: Updates service API and documentation
  * AWS CodeBuild adds support for Amazon Elastic File Systems
* `service/ebs`: Updates service documentation
* `service/ec2`: Updates service API and documentation
  * This release adds platform details and billing info to the DescribeImages API.
* `service/ecr`: Updates service documentation
  * This release contains updated text for the GetAuthorizationToken API.
* `service/lex-models`: Updates service API, documentation, and examples

Release v1.28.11 (2020-02-05)
===

### Service Client Updates
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release provides support for tagging when you create a VPC endpoint, or VPC endpoint service.
* `service/forecastquery`: Updates service API and documentation
* `service/groundstation`: Updates service API, documentation, paginators, and examples
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for fine-tuned QVBR quality level.
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation-only update that adds services to the list of supported services.
* `service/securityhub`: Updates service API and documentation

Release v1.28.10 (2020-02-04)
===

### Service Client Updates
* `service/cloudfront`: Updates service documentation
  * Documentation updates for CloudFront
* `service/ec2`: Updates service API and documentation
  * Amazon VPC Flow Logs adds support for 1-minute aggregation intervals.
* `service/iot`: Updates service API
  * Updated ThrottlingException documentation to report that the error code is 400, and not 429, to reflect actual system behaviour.
* `service/kafka`: Updates service API, documentation, and paginators
* `service/ssm`: Updates service API and documentation
  * This feature ensures that an instance is patched up to the available patches on a particular date. It can be enabled by selecting the 'ApproveUntilDate' option as the auto-approval rule while creating the patch baseline. ApproveUntilDate - The cutoff date for auto approval of released patches. Any patches released on or before this date will be installed automatically.
* `service/storagegateway`: Updates service API
  * Adding KVM as a support hypervisor
* `service/workmail`: Updates service API and documentation
  * This release adds support for tagging Amazon WorkMail organizations.

### SDK Enhancements
* `aws/request`: Add support for EC2 specific throttle exception code
  * Adds support for the EC2ThrottledException throttling exception code. The SDK will now treat this error code as throttling.

### SDK Bugs
* `aws/request`: Fixes an issue where the HTTP host header did not reflect changes to the endpoint URL ([#3102](https://github.com/aws/aws-sdk-go/pull/3102))
  * Fixes [#3093](https://github.com/aws/aws-sdk-go/issues/3093)

Release v1.28.9 (2020-01-24)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release provides support for tagging Amazon ECS task sets for services using external deployment controllers.
* `service/eks`: Updates service API
* `service/opsworkscm`: Updates service documentation
  * AWS OpsWorks for Chef Automate now supports in-place upgrade to Chef Automate 2. Eligible servers can be updated through the management console, CLI and APIs.
* `service/workspaces`: Updates service documentation
  * Documentation updates for WorkSpaces

Release v1.28.8 (2020-01-23)
===

### Service Client Updates
* `service/iam`: Updates service API and documentation
  * This release enables the Identity and Access Management policy simulator to simulate permissions boundary policies.
* `service/rds`: Updates service API, documentation, and paginators
  * This SDK release introduces APIs that automate the export of Amazon RDS snapshot data to Amazon S3. The new APIs include: StartExportTask, CancelExportTask, DescribeExportTasks. These APIs automate the extraction of data from an RDS snapshot and export it to an Amazon S3 bucket. The data is stored in a compressed, consistent, and query-able format. After the data is exported, you can query it directly using tools such as Amazon Athena or Redshift Spectrum. You can also consume the data as part of a data lake solution. If you archive the data in S3 Infrequent Access or Glacier, you can reduce long term data storage costs by applying data lifecycle policies.

### SDK Bugs
* Fix generated errors for some JSON APIs not including a message ([#3089](https://github.com/aws/aws-sdk-go/issues/3089))
  * Fixes the SDK's generated errors to all include the `Message` member regardless if it was modeled on the error shape. This fixes the bug identified in #3088 where some JSON errors were not modeled with the Message member.

Release v1.28.7 (2020-01-21)
===

### Service Client Updates
* `service/codepipeline`: Updates service API and documentation
  * AWS CodePipeline enables an ability to stop pipeline executions.
* `service/discovery`: Updates service documentation
  * Documentation updates for the AWS Application Discovery Service.
* `service/ec2`: Updates service API
  * Add an enum value to the result of DescribeByoipCidrs to support CIDRs that are not publicly advertisable.
* `service/iotevents`: Updates service documentation
* `service/marketplacecommerceanalytics`: Updates service documentation
  * Remove 4 deprecated data sets, change some data sets available dates to 2017-09-15

Release v1.28.6 (2020-01-20)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/application-insights`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API, documentation, and paginators
  * This release provides support for a preview of bringing your own IPv6 addresses (BYOIP for IPv6) for use in AWS.
* `service/kms`: Updates service API and documentation
  * The ConnectCustomKeyStore operation now provides new error codes (USER_LOGGED_IN and USER_NOT_FOUND) for customers to better troubleshoot if their connect custom key store operation fails. Password length validation during CreateCustomKeyStore now also occurs on the client side.
* `service/lambda`: Updates service API and documentation
  * Added reason codes to StateReasonCode (InvalidSubnet, InvalidSecurityGroup) and LastUpdateStatusReasonCode (SubnetOutOfIPAddresses, InvalidSubnet, InvalidSecurityGroup) for functions that connect to a VPC.
* `service/monitoring`: Updates service API and documentation
  * Updating DescribeAnomalyDetectors API to return AnomalyDetector Status value in response.

### SDK Bugs
* `service/dynamodb/expression`: Allow AttributeValue as a value to BuildOperand. ([#3057](https://github.com/aws/aws-sdk-go/pull/3057))
  * This change fixes the SDK's behavior with DynamoDB Expression builder to not double marshal AttributeValues when used as BuildOperands, `Value` type. The AttributeValue will be used in the expression as the specific value set in the AttributeValue, instead of encoded as another AttributeValue.

Release v1.28.5 (2020-01-17)
===

### Service Client Updates
* `service/batch`: Updates service documentation
  * This release ensures INACTIVE job definitions are permanently deleted after 180 days.
* `service/cloudhsmv2`: Updates service API and documentation
  * This release introduces resource-level and tag-based access control for AWS CloudHSM resources. You can now tag CloudHSM backups, tag CloudHSM clusters on creation, and tag a backup as you copy it to another region.
* `service/ecs`: Updates service API, documentation, and paginators
  * This release provides a public preview for specifying Amazon EFS file systems as volumes in your Amazon ECS task definitions.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for MP3 audio only outputs.
* `service/neptune`: Updates service API and documentation
  * This release includes Deletion Protection for Amazon Neptune databases.
* `service/redshift`: Updates service documentation
  * Documentation updates for redshift

Release v1.28.4 (2020-01-16)
===

### Service Client Updates
* `service/ds`: Updates service API
  * To reduce the number of errors our customers are facing, we have modified the requirements of input parameters for two of Directory Service APIs.
* `service/ec2`: Updates service API and documentation
  * Client VPN now supports Port Configuration for VPN Endpoints, allowing usage of either port 443 or port 1194.
* `service/sagemaker`: Updates service API and documentation
  * This release adds two new APIs (UpdateWorkforce and DescribeWorkforce) to SageMaker Ground Truth service for workforce IP whitelisting.

Release v1.28.3 (2020-01-15)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * General Update to EC2 Docs and SDKs
* `service/organizations`: Updates service documentation
  * Updated description for PolicyID parameter and ConstraintViolationException.
* `service/securityhub`: Updates service API and documentation
* `service/ssm`: Updates service documentation
  * Document updates for Patch Manager 'NoReboot' feature.

### SDK Enhancements
* `service/s3/s3crypto`: Added X-Ray support to encrypt/decrypt clients ([#2912](https://github.com/aws/aws-sdk-go/pull/2912))
  * Adds support for passing Context down to the crypto client's KMS client enabling tracing for tools like X-Ray, and metrics.

Release v1.28.2 (2020-01-14)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds support for partition placement groups and instance metadata option in Launch Templates

Release v1.28.1 (2020-01-13)
===

### Service Client Updates
* `service/backup`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service documentation
  * Documentation updates for the StopInstances API. You can now stop and start an Amazon EBS-backed Spot Instance at will, instead of relying on the Stop interruption behavior to stop your Spot Instances when interrupted.
* `service/elasticfilesystem`: Updates service API, documentation, and paginators
  * This release adds support for managing EFS file system policies and EFS Access Points.

Release v1.28.0 (2020-01-10)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * Add shared profile support to new and existing users
* `service/ec2`: Updates service API and documentation
  * This release introduces the ability to tag egress only internet gateways, local gateways, local gateway route tables, local gateway virtual interfaces, local gateway virtual interface groups, local gateway route table VPC association and local gateway route table virtual interface group association. You can use tags to organize and identify your resources for cost allocation.
* `service/rds`: Updates service API and documentation
  * This release adds an operation that enables users to override the system-default SSL/TLS certificate for new Amazon RDS DB instances temporarily, or remove the customer override.
* `service/sagemaker`: Updates service API and documentation
  * SageMaker ListTrialComponents API filter by TrialName and ExperimentName.
* `service/transfer`: Updates service API and documentation
  * This release introduces a new endpoint type that allows you to attach Elastic IP addresses from your AWS account with your server's endpoint directly and whitelist access to your server by client's internet IP address(es) using VPC Security Groups.
* `service/workspaces`: Updates service API and documentation
  * Added the migrate feature to Amazon WorkSpaces.

### SDK Features
* Add generated error types for JSONRPC and RESTJSON APIs
  * Adds generated error types for APIs using JSONRPC and RESTJSON protocols. This allows you to retrieve additional error metadata within an error message that was previously unavailable. For example, Amazon DynamoDB's TransactWriteItems operation can return a `TransactionCanceledException` continuing detailed `CancellationReasons` member. This data is now available by type asserting the error returned from the operation call to `TransactionCanceledException` type.
* `service/dynamodb/dynamodbattribute`: Go 1.9+, Add caching of struct serialization ([#3070](https://github.com/aws/aws-sdk-go/pull/3070))
  * For Go 1.9 and above, adds struct field caching to the SDK's DynamoDB AttributeValue marshalers and unmarshalers. This significantly reduces time, and overall allocations of the (un)marshalers by caching the reflected structure's fields. This should improve the performance of applications using DynamoDB AttributeValue (un)marshalers.

### SDK Bugs
* `service/s3/s3manager`: Fix resource leak on failed CreateMultipartUpload calls ([#3069](https://github.com/aws/aws-sdk-go/pull/3069))
  * Fixes [#3000](https://github.com/aws/aws-sdk-go/issues/3000), [#3035](https://github.com/aws/aws-sdk-go/issues/3035)

Release v1.27.4 (2020-01-09)
===

### Service Client Updates
* `service/logs`: Updates service documentation
  * Documentation updates for logs
* `service/sts`: Updates service examples
  * Documentation updates for sts

Release v1.27.3 (2020-01-08)
===

### Service Client Updates
* `service/ce`: Updates service documentation
* `service/fms`: Updates service API and documentation
* `service/translate`: Updates service API, documentation, and paginators

Release v1.27.2 (2020-01-07)
===

### Service Client Updates
* `service/AWSMigrationHub`: Updates service API, documentation, and paginators
* `service/codebuild`: Updates service API and documentation
  * Add encryption key override to StartBuild API in AWS CodeBuild.
* `service/xray`: Updates service documentation
  * Documentation updates for xray

### SDK Enhancements
* `aws`: Add configuration option enable the SDK to unmarshal API response header maps to normalized lower case map keys. ([#3033](https://github.com/aws/aws-sdk-go/pull/3033))
  * Setting `aws.Config.LowerCaseHeaderMaps` to `true` will result in S3's X-Amz-Meta prefixed header to be unmarshaled to lower case Metadata member's map keys.

### SDK Bugs
* `aws/ec2metadata` : Reduces request timeout for EC2Metadata client along with maximum number of retries ([#3066](https://github.com/aws/aws-sdk-go/pull/3066))
  * Reduces latency while fetching response from EC2Metadata client running in a container to around 3 seconds
  * Fixes [#2972](https://github.com/aws/aws-sdk-go/issues/2972)

Release v1.27.1 (2020-01-06)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release supports service providers configuring a private DNS name for services other than AWS services and services available in the AWS marketplace. This feature allows consumers to access the service using an existing DNS name without making changes to their applications.
* `service/mediapackage`: Updates service API and documentation
  * You can now restrict direct access to AWS Elemental MediaPackage by securing requests for live content using CDN authorization. With CDN authorization, content requests require a specific HTTP header and authorization code.

### SDK Bugs
* `aws/session`: Fix client init not exposing endpoint resolve error ([#3059](https://github.com/aws/aws-sdk-go/pull/3059))
  * Fixes the SDK API clients not surfacing endpoint resolution errors, when the EndpointResolver is unable to resolve an endpoint for the client and region.

Release v1.27.0 (2020-01-02)
===

### Service Client Updates
* `service/ce`: Updates service documentation
* `service/ecr`: Updates service waiters
  * Adds waiters for ImageScanComplete and LifecyclePolicyPreviewComplete
* `service/lex-models`: Updates service documentation
* `service/lightsail`: Updates service API and documentation
  * This release adds support for Certificate Authority (CA) certificate identifier to managed databases in Amazon Lightsail.

### SDK Features
* `services/transcribestreamingservice`: Support for Amazon Transcribe Streaming ([#3048](https://github.com/aws/aws-sdk-go/pull/3048))
  * The SDK now supports the Amazon Transcribe Streaming APIs by utilizing event stream encoding over HTTP/2
  * See [Amazon Transcribe Developer Guide](https://docs.aws.amazon.com/transcribe/latest/dg)
  * Fixes [#2487](https://github.com/aws/aws-sdk-go/issues/2487)

Release v1.26.8 (2019-12-23)
===

### Service Client Updates
* `service/detective`: Updates service documentation
* `service/fsx`: Updates service API, documentation, and paginators
* `service/health`: Updates service API, documentation, and paginators
  * With this release, you can now centrally aggregate AWS Health events from all accounts in your AWS organization. Visit AWS Health documentation to learn more about enabling and using this feature: https://docs.aws.amazon.com/health/latest/ug/organizational-view-health.html.

Release v1.26.7 (2019-12-20)
===

### Service Client Updates
* `service/devicefarm`: Updates service API, documentation, and paginators
  * Introduced browser testing support through AWS Device Farm
* `service/ec2`: Updates service API and documentation
  * This release introduces the ability to tag key pairs, placement groups, export tasks, import image tasks, import snapshot tasks and export image tasks. You can use tags to organize and identify your resources for cost allocation.
* `service/eks`: Updates service API and documentation
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces versioning support for message templates.
* `service/rds`: Updates service API and documentation
  * This release adds an operation that enables users to specify whether a database is restarted when its SSL/TLS certificate is rotated. Only customers who do not use SSL/TLS should use this operation.
* `service/redshift`: Updates service documentation
  * Documentation updates for Amazon Redshift RA3 node types.
* `service/securityhub`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * This release updates the attachments support to include AttachmentReference source for Automation documents.
* `service/transcribe`: Updates service API, documentation, and paginators

Release v1.26.6 (2019-12-19)
===

### Service Client Updates
* `service/codestar-connections`: Adds new service
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * We are updating the supportedRootDevices field to supportedRootDeviceTypes for DescribeInstanceTypes API to ensure that the actual value is returned, correcting a previous error in the model.
* `service/gamelift`: Updates service API and documentation
  * Amazon GameLift now supports ARNs for all key GameLift resources, tagging for GameLift resource authorization management, and updated documentation that articulates GameLift's resource authorization strategy.
* `service/lex-models`: Updates service API and documentation
* `service/personalize-runtime`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * This release allows customers to add tags to Automation execution, enabling them to sort and filter executions in different ways, such as by resource, purpose, owner, or environment.
* `service/transcribe`: Updates service API and documentation

Release v1.26.5 (2019-12-18)
===

### Service Client Updates
* `service/cloudfront`: Updates service documentation
  * Documentation updates for CloudFront
* `service/ec2`: Updates service API and documentation
  * This release introduces the ability to tag Elastic Graphics accelerators. You can use tags to organize and identify your accelerators for cost allocation.
* `service/opsworkscm`: Updates service API and documentation
  * AWS OpsWorks CM now supports tagging, and tag-based access control, of servers and backups.
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation updates for resourcegroupstaggingapi
* `service/s3`: Updates service documentation
  * Updates Amazon S3 endpoints allowing you to configure your client to opt-in to using S3 with the us-east-1 regional endpoint, instead of global.

### SDK Bugs
* `aws/request`: Fix shouldRetry behavior for nested errors ([#3017](https://github.com/aws/aws-sdk-go/pull/3017))

Release v1.26.4 (2019-12-17)
===

### Service Client Updates
* `service/ec2`: Updates service documentation
  * Documentation updates for Amazon EC2
* `service/ecs`: Updates service documentation
  * Documentation updates for Amazon ECS.
* `service/iot`: Updates service API and documentation
  * Added a new Over-the-Air (OTA) Update feature that allows you to use different, or multiple, protocols to transfer an image from the AWS cloud to IoT devices.
* `service/kinesisanalyticsv2`: Updates service API
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports HLS ID3 segment tagging, HLS redundant manifests for CDNs that support different publishing/viewing endpoints, fragmented MP4 (fMP4), and frame capture intervals specified in milliseconds.
* `service/ssm`: Updates service API and documentation
  * Added support for Cloud Watch Output and Document Version to the Run Command tasks in Maintenance Windows.

Release v1.26.3 (2019-12-16)
===

### Service Client Updates
* `service/comprehendmedical`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * You can now configure your EC2 Fleet to preferentially use EC2 Capacity Reservations for launching On-Demand instances, enabling you to fully utilize the available (and unused) Capacity Reservations before launching On-Demand instances on net new capacity.
* `service/mq`: Updates service API and documentation
  * Amazon MQ now supports throughput-optimized message brokers, backed by Amazon EBS.

Release v1.26.2 (2019-12-13)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * CodeBuild adds support for cross account
* `service/detective`: Adds new service
* `service/sesv2`: Updates service API and documentation

Release v1.26.1 (2019-12-12)
===

### Service Client Updates
* `service/accessanalyzer`: Updates service API and documentation

### SDK Bugs
* `service/s3/s3crypto`: Fixes a bug where `gcmEncryptReader` and `gcmDecryptReader` would return an invalid number of bytes as having been read. ([#3005](https://github.com/aws/aws-sdk-go/pull/3005))
  * Fixes [#2999](https://github.com/aws/aws-sdk-go/issues/2999)

Release v1.26.0 (2019-12-11)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release allows customers to attach multiple Elastic Inference Accelerators to a single EC2 instance. It adds support for a Count parameter for each Elastic Inference Accelerator type you specify on the RunInstances and LaunchTemplate APIs.

### SDK Features

* `aws/credentials/stscreds`: Add support for session tags to `AssumeRoleProvider` ([#2993](https://github.com/aws/aws-sdk-go/pull/2993))
  * Adds support for session tags to the AssumeRoleProvider. This feature is used to enable modeling Attribute Based Access Control (ABAC) on top of AWS IAM Policies, User and Roles.
  * https://docs.aws.amazon.com/IAM/latest/UserGuide/id_session-tags.html

### SDK Enhancements
* `aws/request`: Adds `ThrottledException` to the list of retryable request exceptions ([#3006](https://github.com/aws/aws-sdk-go/pull/3006))

Release v1.25.50 (2019-12-10)
===

### Service Client Updates
* `service/kendra`: Updates service API and documentation
  * 1. Adding DocumentTitleFieldName as an optional configuration for SharePoint. 2. updating s3 object pattern to  support all s3 keys.

Release v1.25.49 (2019-12-09)
===

### Service Client Updates
* `service/kafka`: Updates service API and documentation
* `service/kms`: Updates service API and documentation
  * The Verify operation now returns KMSInvalidSignatureException on invalid signatures. The Sign and Verify operations now return KMSInvalidStateException when a request is made against a CMK pending deletion.
* `service/quicksight`: Updates service documentation
  * Documentation updates for QuickSight
* `service/ssm`: Updates service API and documentation
  * Adds the SSM GetCalendarState API and ChangeCalendar SSM Document type. These features enable the forthcoming Systems Manager Change Calendar feature, which will allow you to schedule events during which actions should (or should not) be performed.

### SDK Bugs
* `service/s3`: Fix SDK support for Accesspoint ARNs with slash in resource ([#3001](https://github.com/aws/aws-sdk-go/pull/3001))
  * Fixes the SDK's handling of S3 Accesspoint ARNs to correctly parse ARNs with slashes in the resource component as valid. Previously the SDK's ARN parsing incorrectly identify ARN resources with slash delimiters as invalid ARNs.

Release v1.25.48 (2019-12-05)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
  * Amazon API Gateway now supports HTTP APIs (beta), enabling customers to quickly build high performance RESTful APIs that are up to 71% cheaper than REST APIs also available from API Gateway. HTTP APIs are optimized for building APIs that proxy to AWS Lambda functions or HTTP backends, making them ideal for serverless workloads. Using HTTP APIs, you can secure your APIs using OIDC and OAuth 2 out of box, quickly build web applications using a simple CORS experience, and get started immediately with automatic deployment and simple create workflows.
* `service/kinesis-video-signaling`: Adds new service
* `service/kinesisvideo`: Updates service API, documentation, and paginators
  * Introduces management of signaling channels for Kinesis Video Streams.

Release v1.25.47 (2019-12-04)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/ebs`: Adds new service
* `service/lambda`: Updates service API, documentation, and paginators
  * - Added the ProvisionedConcurrency type and operations. Allocate provisioned concurrency to enable your function to scale up without fluctuations in latency. Use PutProvisionedConcurrencyConfig to configure provisioned concurrency on a version of a function, or on an alias.
* `service/rds`: Updates service API, documentation, and paginators
  * This release adds support for the Amazon RDS Proxy
* `service/rekognition`: Updates service API, documentation, waiters, and paginators
  * This SDK Release introduces APIs for Amazon Rekognition Custom Labels feature (CreateProjects, CreateProjectVersion,DescribeProjects, DescribeProjectVersions, StartProjectVersion, StopProjectVersion and DetectCustomLabels).  Also new is  AugmentedAI (Human In The Loop) Support for DetectModerationLabels in Amazon Rekognition.
* `service/sagemaker`: Updates service API, documentation, waiters, and paginators
  * You can now use SageMaker Autopilot for automatically training and tuning candidate models using a combination of various feature engineering, ML algorithms, and hyperparameters determined from the user's input data. SageMaker Automatic Model Tuning now supports tuning across multiple algorithms. With Amazon SageMaker Experiments users can create Experiments, ExperimentTrials, and ExperimentTrialComponents to track, organize, and evaluate their ML training jobs. With Amazon SageMaker Debugger, users can easily debug training jobs using a number of pre-built rules provided by Amazon SageMaker, or build custom rules. With Amazon SageMaker Processing, users can run on-demand, distributed, and fully managed jobs for data pre- or post- processing or model evaluation. With Amazon SageMaker Model Monitor, a user can create MonitoringSchedules to automatically monitor endpoints to detect data drift and other issues and get alerted on them. This release also includes the preview version of Amazon SageMaker Studio with Domains, UserProfiles, and Apps. This release also includes the preview version of Amazon Augmented AI to easily implement human review of machine learning predictions by creating FlowDefinitions, HumanTaskUis, and HumanLoops.
* `service/states`: Updates service API and documentation
  * This release of the AWS Step Functions SDK introduces support for Express Workflows.

Release v1.25.46 (2019-12-03)
===

### Service Client Updates
* `service/codeguru-reviewer`: Adds new service
* `service/codeguruprofiler`: Adds new service
* `service/compute-optimizer`: Adds new service
* `service/ec2`: Updates service API and documentation
  * This release adds support for the following features: 1. An option to enable acceleration for Site-to-Site VPN connections, to improve connection performance by leveraging AWS Global Accelerator; 2. Inf1 instances featuring up to 16 AWS Inferentia chips, custom-built for ML inference applications to deliver low latency and high throughput performance. Use Inf1 instances to run high scale ML inference applications such as image recognition, speech recognition, natural language processing, personalization, and fraud detection at the lowest cost in the cloud. Inf1 instances will soon be available for use with Amazon SageMaker, Amazon EKS and Amazon ECS. To get started, see https://aws.amazon.com/ec2/instance-types/Inf1; 3. The ability to associate route tables with internet gateways and virtual private gateways, and define routes to insert network and security virtual appliances in the path of inbound and outbound traffic. For more information on Amazon VPC Ingress Routing, see https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Route_Tables.html#gateway-route-table; 4. AWS Local Zones that place compute, storage, database, and other select services closer to you for applications that require very low latency to your end-users. AWS Local Zones also allow you to seamlessly connect to the full range of services in the AWS Region through the same APIs and tool sets; 5. Launching and viewing EC2 instances and EBS volumes running locally in Outposts. This release also introduces a new local gateway (LGW) with Outposts to enable connectivity between Outposts and local on-premises networks as well as the internet; 6. Peering Transit Gateways between regions simplifying creation of secure and private global networks on AWS; 7. Transit Gateway Multicast, enabling multicast routing within and between VPCs using Transit Gateway as a multicast router.
* `service/ecs`: Updates service API, documentation, and paginators
  * This release supports ECS Capacity Providers, Fargate Spot, and ECS Cluster Auto Scaling.  These features enable new ways for ECS to manage compute capacity used by tasks.
* `service/eks`: Updates service API, documentation, and paginators
* `service/es`: Updates service API and documentation
  * UltraWarm storage provides a cost-effective way to store large amounts of read-only data on Amazon Elasticsearch Service. Rather than attached storage, UltraWarm nodes use Amazon S3 and a sophisticated caching solution to improve performance. For indices that you are not actively writing to and query less frequently, UltraWarm storage offers significantly lower costs per GiB. In Elasticsearch, these warm indices behave just like any other index. You can query them using the same APIs or use them to create dashboards in Kibana.
* `service/frauddetector`: Adds new service
* `service/kendra`: Adds new service
  * It is a preview launch of Amazon Kendra. Amazon Kendra is a managed, highly accurate and easy to use enterprise search service that is powered by machine learning.
* `service/networkmanager`: Adds new service
* `service/outposts`: Adds new service
* `service/s3`: Updates service documentation and examples
  * Amazon S3 Access Points is a new S3 feature that simplifies managing data access at scale for shared data sets on Amazon S3. Access Points provide a customizable way to access the objects in a bucket, with a unique hostname and access policy that enforces the specific permissions and network controls for any request made through the access point. This represents a new way of provisioning access to shared data sets.
* `service/s3control`: Updates service documentation
  * Amazon S3 Access Points is a new S3 feature that simplifies managing data access at scale for shared data sets on Amazon S3. Access Points provide a customizable way to access the objects in a bucket, with a unique hostname and access policy that enforces the specific permissions and network controls for any request made through the access point. This represents a new way of provisioning access to shared data sets.
* `service/sagemaker-a2i-runtime`: Adds new service
* `service/textract`: Updates service API and documentation

### SDK Enhancements
* `service/s3`: Add support for Access Point resources
  * Adds support for using Access Point resource with Amazon S3 API operation calls. The Access Point resource are identified by an Amazon Resource Name (ARN).
  * To make operation calls to an S3 Access Point instead of a S3 Bucket, provide the Access Point ARN string as the value of the Bucket parameter. You can create an Access Point for your bucket with the Amazon S3 Control API. The Access Point ARN can be obtained from the S3 Control API. You should avoid building the ARN directly.

Release v1.25.45 (2019-12-02)
===

### Service Client Updates
* `service/accessanalyzer`: Adds new service

Release v1.25.44 (2019-12-02)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * AWS now provides a new BYOL experience for software licenses, such as Windows and SQL Server, that require a dedicated physical server. You can now enjoy the flexibility and cost effectiveness of using your own licenses on Amazon EC2 Dedicated Hosts, but with the simplicity, resiliency, and elasticity of AWS. You can specify your Dedicated Host management preferences, such as host allocation, host capacity utilization, and instance placement in AWS License Manager.  Once set up, AWS takes care of these administrative tasks on your behalf, so that you can seamlessly launch virtual machines (instances) on Dedicated Hosts just like you would launch an EC2 instance with AWS provided licenses.
* `service/imagebuilder`: Adds new service
* `service/license-manager`: Updates service API and documentation
* `service/schemas`: Adds new service

Release v1.25.43 (2019-11-26)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/ds`: Updates service API and documentation
  * This release will introduce optional encryption over LDAP network traffic using SSL certificates between customer's self-managed AD and AWS Directory Services instances. The release also provides APIs for Certificate management.
* `service/dynamodb`: Updates service API, documentation, and paginators
  * 1) Amazon Contributor Insights for Amazon DynamoDB is a diagnostic tool for identifying frequently accessed keys and understanding database traffic trends. 2) Support for displaying new fields when a table's encryption state is Inaccessible or the table have been Archived.
* `service/elastic-inference`: Adds new service
* `service/mediatailor`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * Introduces the DescribeEffectivePolicy action, which returns the contents of the policy that's in effect for the account.
* `service/quicksight`: Updates service documentation
  * Documentation updates for QuickSight
* `service/rds-data`: Updates service API and documentation
* `service/resourcegroupstaggingapi`: Updates service API, documentation, and paginators
  * You can use tag policies to help standardize on tags across your organization's resources.
* `service/serverlessrepo`: Updates service API and documentation
* `service/workspaces`: Updates service API and documentation
  * For the WorkspaceBundle API, added the image identifier and the time of the last update.

Release v1.25.42 (2019-11-25)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/appconfig`: Adds new service
* `service/application-autoscaling`: Updates service API and documentation
* `service/application-insights`: Updates service API, documentation, and paginators
* `service/athena`: Updates service API and documentation
  * This release adds additional query lifecycle metrics to the QueryExecutionStatistics object in GetQueryExecution response.
* `service/ce`: Updates service API and documentation
* `service/codebuild`: Updates service API and documentation
  * CodeBuild adds support for test reporting
* `service/cognito-idp`: Updates service API
* `service/comprehend`: Updates service API and documentation
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release adds two new APIs: 1. ModifyDefaultCreditSpecification, which allows you to set default credit specification at the account level per AWS Region, per burstable performance instance family, so that all new burstable performance instances in the account launch using the new default credit specification. 2. GetDefaultCreditSpecification, which allows you to get current default credit specification per AWS Region, per burstable performance instance family. This release also adds new client exceptions for StartInstances and StopInstances.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/greengrass`: Updates service API and documentation
  * IoT Greengrass supports machine learning resources in 'No container' mode.
* `service/iot`: Updates service API and documentation
  * This release adds: 1) APIs for fleet provisioning claim and template, 2) endpoint configuration and custom domains, 3) support for enhanced custom authentication, d) support for 4 additional audit checks: Device and CA certificate key quality checks, IoT role alias over-permissive check and IoT role alias access to unused services check, 5) extended capability of AWS IoT Rules Engine to support IoT SiteWise rule action. The IoT SiteWise rule action lets you send messages from IoT sensors and applications to IoT SiteWise asset properties
* `service/iotsecuretunneling`: Adds new service
* `service/kinesisanalyticsv2`: Updates service API and documentation
* `service/kms`: Updates service API and documentation
  * AWS Key Management Service (KMS) now enables creation and use of asymmetric Customer Master Keys (CMKs) and the generation of asymmetric data key pairs.
* `service/lambda`: Updates service API, documentation, waiters, and paginators
  * Added the function state and update status to the output of GetFunctionConfiguration and other actions. Check the state information to ensure that a function is ready before you perform operations on it. Functions take time to become ready when you connect them to a VPC.Added the EventInvokeConfig type and operations to configure error handling options for asynchronous invocation. Use PutFunctionEventInvokeConfig to configure the number of retries and the maximum age of events when you invoke the function asynchronously.Added on-failure and on-success destination settings for asynchronous invocation. Configure destinations to send an invocation record to an SNS topic, an SQS queue, an EventBridge event bus, or a Lambda function.Added error handling options to event source mappings. This enables you to configure the number of retries, configure the maximum age of records, or retry with smaller batches when an error occurs when a function processes a Kinesis or DynamoDB stream.Added the on-failure destination setting to event source mappings. This enables you to send discarded events to an SNS topic or SQS queue when all retries fail or when the maximum record age is exceeded when a function processes a Kinesis or DynamoDB stream.Added the ParallelizationFactor option to event source mappings to increase concurrency per shard when a function processes a Kinesis or DynamoDB stream.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for 8K outputs and support for QuickTime Animation Codec (RLE) inputs.
* `service/medialive`: Updates service API, documentation, waiters, and paginators
  * AWS Elemental MediaLive now supports the ability to create a multiple program transport stream (MPTS).
* `service/mediapackage-vod`: Updates service API and documentation
* `service/monitoring`: Updates service API, documentation, and paginators
  * This release adds a new feature called "Contributor Insights". "Contributor Insights" supports the following 6 new APIs (PutInsightRule, DeleteInsightRules, EnableInsightRules, DisableInsightRules, DescribeInsightRules and GetInsightRuleReport).
* `service/ram`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Cluster Endpoints can now be tagged by using --tags in the create-db-cluster-endpoint API
* `service/redshift`: Updates service API, documentation, and paginators
  * This release contains changes for 1. Redshift Scheduler 2. Update to the DescribeNodeConfigurationOptions to include a new action type recommend-node-config
* `service/runtime.lex`: Updates service API and documentation
* `service/sesv2`: Updates service API, documentation, and paginators
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Documents now supports more Document Types: ApplicationConfiguration, ApplicationConfigurationSchema and DeploymentStrategy. This release also extends Document Permissions capabilities and introduces a new Force flag for DeleteDocument API.

### SDK Enhancements
* `aws/credentials/processcreds`: Increase the default max buffer size ([#2957](https://github.com/aws/aws-sdk-go/pull/2957))
  * Fixes [#2875](https://github.com/aws/aws-sdk-go/issues/2875)

Release v1.25.41 (2019-11-22)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * This release adds support for Tag-Based IAM for AWS Certificate Manager and adding tags to certificates upon creation.
* `service/application-autoscaling`: Updates service API
* `service/autoscaling-plans`: Updates service API
* `service/codebuild`: Updates service API and documentation
  * Add Canonical ARN to LogsLocation.
* `service/ec2`: Updates service API and documentation
  * This release adds two new APIs (DescribeInstanceTypes and DescribeInstanceTypeOfferings) that give customers access to instance type attributes and regional and zonal offerings.
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR adds support for concurrent step execution and cancelling running steps. Amazon EMR has added a new Outpost ARN field in the ListCluster and DescribeCluster API responses that is populated for clusters launched in an AWS Outpost subnet.
* `service/forecast`: Updates service API and documentation
* `service/mediapackage-vod`: Updates service API and documentation
* `service/rekognition`: Updates service API and documentation
  * This release adds enhanced face filtering support to the IndexFaces API operation, and introduces face filtering for CompareFaces and SearchFacesByImage API operations.
* `service/sns`: Updates service documentation
  * Added documentation for the dead-letter queue feature.
* `service/ssm`: Updates service API and documentation
  * Add RebootOption and LastNoRebootInstallOperationTime for DescribeInstancePatchStates and DescribeInstancePatchStatesForPatchGroup API
* `service/sts`: Updates service API, documentation, and examples
  * Support tagging for STS sessions and tag based access control for the STS APIs

### SDK Bugs
* `aws/ec2metadata`: Fix failing concurrency test for ec2metadata client ([#2960](https://github.com/aws/aws-sdk-go/pull/2960))
  * Fixes a resource leak  in ec2metadata client, where response body was not closed after reading

Release v1.25.40 (2019-11-21)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/appsync`: Updates service API and documentation
* `service/config`: Updates service API and documentation
* `service/connect`: Updates service API and documentation
* `service/connectparticipant`: Adds new service
* `service/dynamodb`: Updates service API and documentation
  * With this release, you can convert an existing Amazon DynamoDB table to a global table by adding replicas in other AWS Regions.
* `service/ec2`: Updates service API and documentation
  * This release adds support for attaching AWS License Manager Configurations to Amazon Machine Image (AMI) using ImportImage API; and adds support for running different instance sizes on EC2 Dedicated Hosts
* `service/glue`: Updates service API and documentation
  * This release adds support for Glue 1.0 compatible ML Transforms.
* `service/lex-models`: Updates service API and documentation
* `service/meteringmarketplace`: Updates service documentation
  * Documentation updates for the AWS Marketplace Metering Service.
* `service/runtime.lex`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * The release contains new API and API changes for AWS Systems Manager Explorer product.
* `service/transcribe`: Updates service API

Release v1.25.39 (2019-11-20)
===

### Service Client Updates
* `service/AWSMigrationHub`: Updates service API, documentation, and paginators
* `service/chime`: Updates service API, documentation, and paginators
  * Adds APIs to create and manage meeting session resources for the Amazon Chime SDK
* `service/cloudtrail`: Updates service API and documentation
  * 1. This release adds two new APIs, GetInsightSelectors and PutInsightSelectors, which let you configure CloudTrail Insights event delivery on a trail. An Insights event is a new type of event that is generated when CloudTrail detects unusual activity in your AWS account. In this release, only "ApiCallRateInsight" is a supported Insights event type. 2. This release also adds the new "ExcludeManagementEventSource" option to the existing PutEventSelectors API. This field currently supports only AWS Key Management Services.
* `service/codecommit`: Updates service API, documentation, and paginators
  * This release adds support for creating pull request approval rules and pull request approval rule templates in AWS CodeCommit. This allows developers to block merges of pull requests, contingent on the approval rules being satisfiied.
* `service/datasync`: Updates service API and documentation
* `service/discovery`: Updates service API and documentation
  * New exception type for use with Migration Hub home region
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API, documentation, waiters, and paginators
  * This release of Amazon Elastic Compute Cloud (Amazon EC2) introduces support for Amazon Elastic Block Store (Amazon EBS) fast snapshot restores.
* `service/ecs`: Updates service API and documentation
  * Added support for CPU and memory task-level overrides on the RunTask and StartTask APIs.  Added location information to Tasks.
* `service/firehose`: Updates service API and documentation
  * With this release, Amazon Kinesis Data Firehose allows server side encryption with customer managed CMKs. Customer managed CMKs ( "Customer Master Keys") are AWS Key Management Service (KMS) keys that are fully managed by the customer. With customer managed CMKs, customers can establish and maintain their key policies, IAM policies, rotating policies and add tags. For more information about AWS KMS and CMKs, please refer to:  https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html. Please refer to the following link to create CMKs: https://docs.aws.amazon.com/kms/latest/developerguide/importing-keys-create-cmk.html
* `service/fsx`: Updates service API and documentation
* `service/mediastore`: Updates service API and documentation
  * This release fixes a broken link in the SDK documentation.
* `service/migrationhub-config`: Adds new service
* `service/quicksight`: Updates service API, documentation, and paginators
  * Amazon QuickSight now supports programmatic creation and management of data sources, data sets, dashboards and templates with new APIs. Templates hold dashboard metadata, and can be used to create copies connected to the same or different dataset as required. Also included in this release are APIs for SPICE ingestions, fine-grained access control over AWS resources using AWS Identity and Access Management (IAM) policies, as well AWS tagging. APIs are supported for both Standard and Enterprise Edition, with edition-specific support for specific functionality.
* `service/s3`: Updates service API and documentation
  * This release introduces support for Amazon S3 Replication Time Control, a new feature of S3 Replication that provides a predictable replication time backed by a Service Level Agreement. S3 Replication Time Control helps customers meet compliance or business requirements for data replication, and provides visibility into the replication process with new Amazon CloudWatch Metrics.
* `service/storagegateway`: Updates service API and documentation
  * The new DescribeAvailabilityMonitorTest API provides the results of the most recent High Availability monitoring test. The new StartAvailabilityMonitorTest API verifies the storage gateway is configured for High Availability monitoring. The new ActiveDirectoryStatus response element has been added to the DescribeSMBSettings and JoinDomain APIs to indicate the status of the gateway after the most recent JoinDomain operation. The new TimeoutInSeconds parameter of the JoinDomain API allows for the configuration of the timeout in which the JoinDomain operation must complete.
* `service/transcribe`: Updates service API and documentation

Release v1.25.38 (2019-11-19)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Amazon EC2 Auto Scaling now supports Instance Weighting and Max Instance Lifetime. Instance Weighting allows specifying the capacity units for each instance type included in the MixedInstancesPolicy and how they would contribute to your application's performance. Max Instance Lifetime allows specifying the maximum length of time that an instance can be in service. If any instances are approaching this limit, Amazon EC2 Auto Scaling gradually replaces them.
* `service/cloudformation`: Updates service API and documentation
  * This release of AWS CloudFormation StackSets enables users to detect drift on a stack set and the stack instances that belong to that stack set.
* `service/codebuild`: Updates service API and documentation
  * Add support for ARM and GPU-enhanced build environments and a new SSD-backed Linux compute type with additional CPU and memory in CodeBuild
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release adds support for RunInstances to specify the metadata options for new instances; adds a new API, ModifyInstanceMetadataOptions, which lets you modify the metadata options for a running or stopped instance; and adds support for CreateCustomerGateway to specify a device name.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/iam`: Updates service API, documentation, and examples
  * IAM reports the timestamp when a role's credentials were last used to make an AWS request. This helps you identify unused roles and remove them confidently from your AWS accounts.
* `service/iot`: Updates service API and documentation
  * As part of this release, we are extending the capability of AWS IoT Rules Engine to send messages directly to customer's own web services/applications. Customers can now create topic rules with HTTP actions to route messages from IoT Core directly to URL's that they own. Ownership is proved by creating and confirming topic rule destinations.
* `service/lambda`: Updates service API
  * This release provides three new runtimes to support Node.js 12 (initially 12.13.0), Python 3.8 and Java 11.

### SDK Enhancements
* `aws/ec2metadata`: Adds support for EC2Metadata client to use secure tokens provided by the IMDS ([#2958](https://github.com/aws/aws-sdk-go/pull/2958))
  * Modifies and adds tests to verify the behavior of the EC2Metadata client.

Release v1.25.37 (2019-11-18)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/cloudformation`: Updates service API, documentation, waiters, and paginators
  * This release introduces APIs for the CloudFormation Registry, a new service to submit and discover resource providers with which you can manage third-party resources natively in CloudFormation.
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces support for using and managing message templates for messages that are sent through the voice channel. It also introduces support for specifying default values for message variables in message templates.
* `service/rds`: Updates service documentation
  * Documentation updates for rds
* `service/runtime.sagemaker`: Updates service API and documentation
* `service/s3`: Updates service API, documentation, and examples
  * Added support for S3 Replication for existing objects. This release allows customers who have requested and been granted access to replicate existing S3 objects across buckets.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker now supports multi-model endpoints to host multiple models on an endpoint using a single inference container.
* `service/ssm`: Updates service API and documentation
  * The release contains new API and API changes for AWS Systems Manager Explorer product.

Release v1.25.36 (2019-11-15)
===

### Service Client Updates
* `service/chime`: Updates service API, documentation, and paginators
  * This release adds support for Chime Room Management APIs
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * You can now add tags while copying snapshots. Previously, a user had to first copy the snapshot and then add tags to the copied snapshot manually. Moving forward, you can specify the list of tags you wish to be applied to the copied snapshot as a parameter on the Copy Snapshot API.
* `service/eks`: Updates service API, documentation, waiters, paginators, and examples
* `service/elasticloadbalancingv2`: Updates service documentation
* `service/elasticmapreduce`: Updates service API and documentation
  * Access to the cluster ARN makes it easier for you to author resource-level permissions policies in AWS Identity and Access Management. To simplify the process of obtaining the cluster ARN, Amazon EMR has added a new field containing the cluster ARN to all API responses that include the cluster ID.
* `service/guardduty`: Updates service API, documentation, and paginators
  * This release includes new operations related to findings export, including: CreatePublishingDestination, UpdatePublishingDestination, DescribePublishingDestination, DeletePublishingDestination and ListPublishingDestinations.
* `service/logs`: Updates service documentation
  * Documentation updates for logs
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for DolbyVision encoding, and SCTE35 & ESAM insertion to DASH ISO EMSG.
* `service/ssm`: Updates service documentation
  * This release updates AWS Systems Manager Parameter Store documentation for the enhanced search capability.
* `service/workspaces`: Updates service API and documentation
  * Added APIs to register your directories with Amazon WorkSpaces and to modify directory details.

Release v1.25.35 (2019-11-14)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/connect`: Updates service API and documentation
* `service/meteringmarketplace`: Updates service API and documentation
  * Added CustomerNotEntitledException in MeterUsage API for Container use case.
* `service/personalize`: Updates service API, documentation, and paginators
* `service/ssm`: Updates service API and documentation
  * Updates support for adding attachments to Systems Manager Automation documents

### SDK Enhancements
* `aws/endpoints`: Add support for regional S3 us-east-1 endpoint
  * Adds support for S3 configuring an SDK Amazon S3 client for the regional us-east-1 endpoint instead of the default global S3 endpoint.
  * Adds a new configuration option, `S3UsEast1RegionalEndpoint` which when set to `RegionalS3UsEast1Endpoint`, and region is `us-east-1` the S3 client will resolve the `us-east-1` regional endpoint, `s3.us-east-1.amazonaws.com` instead of the global S3 endpoint, `s3.amazonaws.com`. The SDK defaults to the current global S3 endpoint resolution for backwards compatibility.
  * Opt-in to the `us-east-1` regional endpoint via the SDK's Config, environment variable, `AWS_S3_US_EAST_1_REGIONAL_ENDPOINT=regional`, or shared config option, `s3_us_east_1_regional_endpoint=regional`.
  * Note the SDK does not support the shared configuration file by default.  You must opt-in to that behavior via Session Option `SharedConfigState`, or `AWS_SDK_LOAD_CONFIG=true` environment variable.

Release v1.25.34 (2019-11-13)
===

### Service Client Updates
* `service/cloudsearch`: Updates service API, documentation, paginators, and examples
  * Amazon CloudSearch domains let you require that all traffic to the domain arrive over HTTPS. This security feature helps you block clients that send unencrypted requests to the domain.
* `service/dataexchange`: Adds new service
* `service/dlm`: Updates service API and documentation
* `service/iot`: Updates service API and documentation
  * This release adds the custom fields definition support in the index definition for AWS IoT Fleet Indexing Service. Custom fields can be used as an aggregation field to run aggregations with both existing GetStatistics API and newly added GetCardinality, GetPercentiles APIs. GetStatistics will return all statistics (min/max/sum/avg/count...) with this release. For more information, please refer to our latest documentation: https://docs.aws.amazon.com/iot/latest/developerguide/iot-indexing.html
* `service/sesv2`: Adds new service

### SDK Enhancements
* Replaced case-insensitive string comparisons with `strings.EqualFold(...)` ([#2922](https://github.com/aws/aws-sdk-go/pull/2922))

Release v1.25.33 (2019-11-12)
===

### Service Client Updates
* `service/codepipeline`: Updates service API and documentation
  * AWS CodePipeline now supports the use of variables in action configuration.
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB enables you to restore your data to a new DynamoDB table using a point-in-time or on-demand backup. You now can modify the settings on the new restored table. Specifically, you can exclude some or all of the local and global secondary indexes from being created with the restored table. In addition, you can change the billing mode and provisioned capacity settings.
* `service/elasticloadbalancingv2`: Updates service documentation
* `service/marketplace-catalog`: Adds new service
* `service/transcribe`: Updates service API and documentation

Release v1.25.32 (2019-11-11)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/cloudformation`: Updates service API, documentation, and waiters
  * The Resource Import feature enables customers to import existing AWS resources into new or existing CloudFormation Stacks.

Release v1.25.31 (2019-11-08)
===

### Service Client Updates
* `service/cognito-identity`: Updates service API and documentation
* `service/ecr`: Updates service documentation
  * This release contains ticket fixes for Amazon ECR.

### SDK Bugs
* `aws/request`: Ensure New request handles nil retryer ([#2934](https://github.com/aws/aws-sdk-go/pull/2934))
  * Adds additional default behavior to the SDK's New request constructor, to handle the case where a nil Retryer was passed in. This error could occur when the SDK's Request type was being used to create requests directly, not through one of the SDK's client.
  * Fixes [#2889](https://github.com/aws/aws-sdk-go/issues/2889)

Release v1.25.30 (2019-11-07)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `service/ssm`: Updates service API
  * AWS Systems Manager Session Manager target length increased to 400.
* `service/sso`: Adds new service
  * This is an initial release of AWS Single Sign-On (SSO) end-user access. This release adds support for accessing AWS accounts assigned in AWS SSO using short term credentials.
* `service/sso-oidc`: Adds new service

Release v1.25.29 (2019-11-06)
===

### Service Client Updates
* `service/savingsplans`: Updates service documentation

Release v1.25.28 (2019-11-06)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Documentation updates for budgets to track Savings Plans utilization and coverage
* `service/ce`: Updates service API, documentation, and paginators
* `service/codebuild`: Updates service API and documentation
  * Add support for Build Number, Secrets Manager and Exported Environment Variables.
* `service/elasticfilesystem`: Updates service API
  * EFS customers can select a lifecycle policy that automatically moves files that have not been accessed for 7 days into the EFS Infrequent Access (EFS IA) storage class. EFS IA provides price/performance that is cost-optimized for files that are not accessed every day.
* `service/savingsplans`: Adds new service
* `service/signer`: Updates service API and documentation
  * This release adds support for tagging code-signing profiles in AWS Signer.

Release v1.25.27 (2019-11-05)
===

### Service Client Updates
* `service/codestar-notifications`: Adds new service
* `service/rds`: Updates service documentation
  * Documentation updates for Amazon RDS

Release v1.25.26 (2019-11-04)
===

### Service Client Updates
* `service/dax`: Updates service documentation
  * Documentation updates for dax
* `service/ec2`: Updates service API and documentation
  * Documentation updates for ec2
* `service/robomaker`: Updates service API and documentation

Release v1.25.25 (2019-11-01)
===

### Service Client Updates
* `service/cloudtrail`: Updates service API, documentation, and paginators
  * This release adds two new APIs, GetTrail and ListTrails, and support for adding tags when you create a trail by using a new TagsList parameter on CreateTrail operations.
* `service/dms`: Updates service API and documentation
  * This release contains task timeline attributes in replication task statistics. This release also adds a note to the documentation for the CdcStartPosition task request parameter. This note describes how to enable the use of native CDC start points for a PostgreSQL source by setting the new slotName extra connection attribute on the source endpoint to the name of an existing logical replication slot.
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces support for using and managing journeys, and querying analytics data for journeys.

Release v1.25.24 (2019-10-31)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/s3`: Updates service API and examples
  * S3 Inventory now supports a new field 'IntelligentTieringAccessTier' that reports the access tier (frequent or infrequent) of objects stored in Intelligent-Tiering storage class.
* `service/support`: Updates service API, documentation, and paginators
  * The status descriptions for TrustedAdvisorCheckRefreshStatus have been updated

Release v1.25.23 (2019-10-30)
===

### Service Client Updates
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache for Redis 5.0.5 now allows you to modify authentication tokens by setting and rotating new tokens. You can now modify active tokens while in use, or add brand-new tokens to existing encryption-in-transit enabled clusters that were previously setup without authentication tokens. This is a two-step process that allows you to set and rotate the token without interrupting client requests.

Release v1.25.22 (2019-10-29)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Adds support for providing domain names that can embed streaming sessions
* `service/cloud9`: Updates service API, documentation, and examples
  * Added CREATING and CREATE_FAILED environment lifecycle statuses.

Release v1.25.21 (2019-10-28)
===

### Service Client Updates
* `service/s3`: Updates service API, documentation, and examples
  * Adding support in SelectObjectContent for scanning a portion of an object specified by a scan range.

Release v1.25.20 (2019-10-28)
===

### Service Client Updates
* `service/ecr`: Updates service API, documentation, and paginators
  * This release of Amazon Elastic Container Registry Service (Amazon ECR) introduces support for image scanning. This identifies the software vulnerabilities in the container image based on the Common Vulnerabilities and Exposures (CVE) database.
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache adds support for migrating Redis workloads hosted on Amazon EC2 into ElastiCache by syncing the data between the source Redis cluster and target ElastiCache for Redis cluster in real time. For more information, see https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/migrate-to-elasticache.html.
* `service/transfer`: Updates service API and documentation
  * This release adds logical directories support to your AWS SFTP server endpoint, so you can now create logical directory structures mapped to Amazon Simple Storage Service (Amazon S3) bucket paths for users created and stored within the service. Amazon S3 bucket names and paths can now be hidden from AWS SFTP users, providing an additional level of privacy to meet security requirements. You can lock down your SFTP users' access to designated folders (commonly referred to as 'chroot'), and simplify complex folder structures for data distribution through SFTP without replicating files across multiple users.

### SDK Enhancements
* `aws/client`: Add PartitionID to Config ([#2902](https://github.com/aws/aws-sdk-go/pull/2902))
* `aws/client/metadata`: Add PartitionID to ClientInfo ([#2902](https://github.com/aws/aws-sdk-go/pull/2902))
* `aws/endpoints`: Add PartitionID to ResolvedEndpoint ([#2902](https://github.com/aws/aws-sdk-go/pull/2902))

### SDK Bugs
* `aws/endpoints`: Fix resolve endpoint with empty region ([#2911](https://github.com/aws/aws-sdk-go/pull/2911))
  * Fixes the SDK's behavior when attempting to resolve a service's endpoint when no region was provided. Adds legacy support for services that were able to resolve a valid endpoint. No new service will support resolving an endpoint without an region.
  * Fixes [#2909](https://github.com/aws/aws-sdk-go/issues/2909)

Release v1.25.19 (2019-10-24)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/chime`: Updates service API, documentation, and paginators
  * * This release introduces Voice Connector PDX region and defaults previously created Voice Connectors to IAD. You can create Voice Connector Groups and add region specific Voice Connectors to direct telephony traffic across AWS regions in case of regional failures. With this release you can add phone numbers to Voice Connector Groups and can bulk move phone numbers between Voice Connectors, between Voice Connector and Voice Connector Groups and between Voice Connector Groups. Voice Connector now supports additional settings to enable SIP Log capture. This is in addition to the launch of Voice Connector Cloud Watch metrics in this release. This release also supports assigning outbound calling name (CNAM) to AWS account and individual phone numbers assigned to Voice Connectors. * Voice Connector now supports a setting to enable real time audio streaming delivered via Kinesis Audio streams. Please note that recording Amazon Chime Voice Connector calls with this feature maybe be subject to laws or regulations regarding the recording of telephone calls and other electronic communications. AWS Customer and their end users' have the responsibility to comply with all applicable laws regarding the recording, including properly notifying all participants in a recorded session or to a recorded communication that the session or communication is being recorded and obtain their consent.
* `service/ec2`: Updates service API and documentation
  * This release updates CreateFpgaImage to support tagging FPGA images on creation
* `service/gamelift`: Updates service API
  * Amazon GameLift offers expanded hardware options for game hosting: Custom game builds can use the Amazon Linux 2 operating system, and fleets for both custom builds and Realtime servers can now use C5, M5, and R5 instance types.
* `service/sagemaker`: Updates service API
  * Adds support for the new family of Elastic Inference Accelerators (eia2) for SageMaker Hosting and Notebook Services

Release v1.25.18 (2019-10-23)
===

### Service Client Updates
* `service/connect`: Updates service API, documentation, and paginators
* `service/polly`: Updates service API
  * Amazon Polly adds new female voices: US Spanish - Lupe and Brazilian Portuguese - Camila; both voices are available in Standard and Neural engine.
* `service/sts`: Updates service documentation
  * AWS Security Token Service (STS) now supports a regional configuration flag to make the client respect the region without the need for the endpoint parameter.

### SDK Enhancements
* `aws/endpoints`: Adds support for STS Regional Flags ([#2779](https://github.com/aws/aws-sdk-go/pull/2779))
  * Implements STS regional flag, with support for `legacy` and `regional` options. Defaults to `legacy`. Legacy, will force all regions specified in aws/endpoints/sts_legacy_regions.go to resolve to the STS global endpoint, sts.amazonaws.com. This is the SDK's current behavior.
  * When the flag's value is `regional` the SDK will resolve the endpoint based on the endpoints.json model. This allows STS to update their service's modeled endpoints to be regionalized for all regions. When `regional` turned on use `aws-global` as the region to use the global endpoint.
  * `AWS_STS_REGIONAL_ENDPOINTS=regional` for environment, or `sts_regional_endpoints=regional` in shared config file.
  * The regions the SDK defaults to the STS global endpoint in `legacy` mode are: 
    * ap-northeast-1
    * ap-south-1
    * ap-southeast-1
    * ap-southeast-2
    * aws-global
    * ca-central-1
    * eu-central-1
    * eu-north-1
    * eu-west-1
    * eu-west-2
    * eu-west-3
    * sa-east-1
    * us-east-1
    * us-east-2
    * us-west-1
    * us-west-2

Release v1.25.17 (2019-10-22)
===

### Service Client Updates
* `service/iotevents`: Updates service API and documentation
* `service/opsworkscm`: Updates service API and documentation
  * AWS OpsWorks for Chef Automate (OWCA) now allows customers to use a custom domain and respective certificate, for their AWS OpsWorks For Chef Automate servers. Customers can now provide a CustomDomain, CustomCertificate and CustomPrivateKey in CreateServer API to configure their Chef Automate servers with a custom domain and certificate.

### SDK Bugs
* `service/s3`,`service/kinesis`: Fix streaming APIs' Err method closing stream ([#2882](https://github.com/aws/aws-sdk-go/pull/2882))
  * Fixes calling the Err method on SDK's Amazon Kinesis's SubscribeToShared and Amazon S3's SelectObjectContent response EventStream members closing the stream. This would cause unexpected read errors, or early termination of the streams. Only the Close method of the streaming members will close the streams.
  * Related to [#2769](https://github.com/aws/aws-sdk-go/issues/2769)

Release v1.25.16 (2019-10-18)
===

### Service Client Updates
* `service/monitoring`: Updates service API and documentation
  * New Period parameter added to MetricDataQuery structure.

Release v1.25.15 (2019-10-17)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Adding support for Compute Environment Allocation Strategies
* `service/rds`: Updates service API, documentation, and paginators
  * Amazon RDS now supports Amazon RDS on VMware with the introduction of APIs related to Custom Availability Zones and Media installation.

Release v1.25.14 (2019-10-16)
===

### Service Client Updates
* `service/kafka`: Updates service API and documentation
* `service/marketplacecommerceanalytics`: Updates service API and documentation
  * add 2 more values for the supporting sections - age of past due funds + uncollected funds breakdown
* `service/robomaker`: Updates service API

Release v1.25.13 (2019-10-15)
===

### Service Client Updates
* `service/kinesis-video-archived-media`: Updates service API and documentation

Release v1.25.12 (2019-10-14)
===

### Service Client Updates
* `service/personalize`: Updates service API and documentation
* `service/workspaces`: Updates service documentation
  * Documentation updates for WorkSpaces

Release v1.25.11 (2019-10-11)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/greengrass`: Updates service API
  * Greengrass OTA service supports Raspbian/Armv6l platforms.

Release v1.25.10 (2019-10-10)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * New EC2 M5n, M5dn, R5n, R5dn instances with 100 Gbps network performance and Elastic Fabric Adapter (EFA) for ultra low latency; New A1.metal bare metal instance powered by AWS Graviton Processors
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/fms`: Updates service API and documentation
* `service/iotanalytics`: Updates service API and documentation
* `service/runtime.lex`: Updates service API and documentation

Release v1.25.9 (2019-10-09)
===

### Service Client Updates
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache now allows you to apply available service updates on demand to your Memcached and Redis Cache Clusters. Features included: (1) Access to the list of applicable service updates and their priorities. (2) Service update monitoring and regular status updates. (3) Recommended apply-by-dates for scheduling the service updates. (4) Ability to stop and later re-apply updates. For more information, see https://docs.aws.amazon.com/AmazonElastiCache/latest/mem-ug/Self-Service-Updates.html and https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/Self-Service-Updates.html
* `service/kafka`: Updates service documentation
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for Dolby Atmos encoding, up to 36 outputs, accelerated transcoding with frame capture and preferred acceleration feature.

Release v1.25.8 (2019-10-08)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/eventbridge`: Updates service documentation
* `service/firehose`: Updates service API and documentation
  * With this release, you can use Amazon Kinesis Firehose delivery streams to deliver streaming data to Amazon Elasticsearch Service version 7.x clusters. For technical documentation, look for CreateDeliveryStream operation in Amazon Kinesis Firehose API reference.
* `service/organizations`: Updates service documentation
  * Documentation updates for organizations

Release v1.25.7 (2019-10-07)
===

### Service Client Updates
* `service/directconnect`: Updates service API and documentation
  * This release adds a service provider field for physical connection creation and provides a list of available partner providers for each Direct Connect location.
* `service/firehose`: Updates service API and documentation
  * Amazon Kinesis Data Firehose now allows delivering data to Elasticsearch clusters set up in a different AWS account than the Firehose AWS account. For technical documentation, look for ElasticsearchDestinationConfiguration in the Amazon Kinesis Firehose API reference.
* `service/glue`: Updates service API and documentation
  * AWS Glue now provides ability to use custom certificates for JDBC Connections.
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint API introduces support for using and managing message templates.
* `service/pinpoint-email`: Updates service API and documentation
* `service/snowball`: Updates service API and documentation
  * AWS Snowball Edge now allows you to perform an offline update to the software of your Snowball Edge device when your device is not connected to the internet. Previously, updating your Snowball Edge's software required that the device be connected to the internet or be sent back to AWS. Now, you can keep your Snowball Edge software up to date even if your device(s) cannot connect to the internet, or are required to run in an air-gapped environment. To complete offline updates, download the software update from a client machine with connection to the internet using the AWS Command Line Interface (CLI). Then, have the Snowball Edge device download and install the software update using the Snowball Edge device API. For more information about offline updates, visit the Snowball Edge documentation page.

Release v1.25.6 (2019-10-04)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API, documentation, and paginators
* `service/mediapackage`: Updates service API, documentation, and paginators
  * New Harvest Job APIs to export segment-accurate content windows from MediaPackage Origin Endpoints to S3. See https://docs.aws.amazon.com/mediapackage/latest/ug/harvest-jobs.html for more info
* `service/ssm`: Updates service documentation
  * Documentation updates for Systems Manager / StartSession.

Release v1.25.5 (2019-10-03)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation and examples
* `service/devicefarm`: Updates service documentation and examples
  * Documentation updates for devicefarm
* `service/ec2`: Updates service API and documentation
  * This release allows customers to purchase regional EC2 RIs on a future date.
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service now supports configuring additional options for domain endpoint, such as whether to require HTTPS for all traffic.

### SDK Bugs
* `service/dynamodb/dynamodbattribute`: Fixes a panic when decoding into a map with a key string type alias. ([#2870](https://github.com/aws/aws-sdk-go/pull/2870))

Release v1.25.4 (2019-10-02)
===

### Service Client Updates
* `service/lightsail`: Updates service API and documentation
  * This release adds support for the automatic snapshots add-on for instances and block storage disks.

### SDK Enhancements
* `service/s3/s3manager`: Allow reuse of Uploader buffer `sync.Pool` amongst multiple Upload calls ([#2863](https://github.com/aws/aws-sdk-go/pull/2863)) 
  * The `sync.Pool` used for the reuse of `[]byte` slices when handling streaming payloads will now be shared across multiple Upload calls when the upload part size remains constant. 

### SDK Bugs
* `internal/ini`: Fix ini parser to handle empty values [#2860](https://github.com/aws/aws-sdk-go/pull/2860)
  * Fixes incorrect modifications to the previous token value of the skipper. Adds checks for cases where a skipped statement should be marked as complete and not be ignored.
  * Adds tests for nested and empty field value parsing, along with tests suggested in [#2801](https://github.com/aws/aws-sdk-go/pull/2801)
  * Fixes [#2800](https://github.com/aws/aws-sdk-go/issues/2800)

Release v1.25.3 (2019-10-01)
===

### Service Client Updates
* `service/docdb`: Updates service API and documentation
  * This release provides support for describe and modify CA certificates.

### SDK Bugs
* `private/model/api` : Fixes broken test for code generation example ([#2855](https://github.com/aws/aws-sdk-go/pull/2855))

Release v1.25.2 (2019-09-30)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mq`: Updates service API and documentation
  * Amazon MQ now includes the ability to scale your brokers by changing the host instance type. See the hostInstanceType property of UpdateBrokerInput (https://docs.aws.amazon.com/amazon-mq/latest/api-reference/brokers-broker-id.html#brokers-broker-id-model-updatebrokerinput), and pendingHostInstanceType property of DescribeBrokerOutput (https://docs.aws.amazon.com/amazon-mq/latest/api-reference/brokers-broker-id.html#brokers-broker-id-model-describebrokeroutput).
* `service/rds`: Updates service API, documentation, and waiters
  * This release adds support for creating a Read Replica with Active Directory domain information. This release updates RDS API to indicate whether an OrderableDBInstanceOption supports Kerberos Authentication.
* `service/waf`: Updates service API and documentation
  * Lowering the threshold for Rate Based rule from 2000 to 100.

Release v1.25.1 (2019-09-27)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) removes FirelensConfiguration from the DescribeTask output during the FireLens public preview.

### SDK Enhancements
* `private/protocol/xml/xmlutil`: Support for sorting xml attributes ([#2854](https://github.com/aws/aws-sdk-go/pull/2854))

### SDK Bugs
* `private/model/api`: Write locationName for top-level shapes for rest-xml and ec2 protocols ([#2854](https://github.com/aws/aws-sdk-go/pull/2854))
* `private/mode/api`: Colliding fields should serialize with original name ([#2854](https://github.com/aws/aws-sdk-go/pull/2854))
  * Fixes [#2806](https://github.com/aws/aws-sdk-go/issues/2806) 
Release v1.25.0 (2019-09-26)
===

### Service Client Updates
* `service/codepipeline`: Updates service documentation
  * Documentation updates for CodePipeline
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/ssm`: Updates service API and documentation
  * This release updates the AWS Systems Manager Parameter Store PutParameter and LabelParameterVersion APIs to return the "Tier" of parameter created/updated and the "parameter version" labeled respectively.

### SDK Features
* `service/dynamodb/dynamodbattribute`: Add EnableEmptyCollections flag to Encoder and Decoder ([#2834](https://github.com/aws/aws-sdk-go/pull/2834))
  * The `Encoder` and `Decoder` types have been enhanced to allow support for specifying the SDK's behavior when marshaling structures, maps, and slices to DynamoDB.
  * When `EnableEmptyCollections` is set to `True` the SDK will preserve the empty of these types in DynamoDB rather then encoding a NULL AttributeValue.
  * Fixes [#682](https://github.com/aws/aws-sdk-go/issues/682)
  * Fixes [#1890](https://github.com/aws/aws-sdk-go/issues/1890)
  * Fixes [#2746](https://github.com/aws/aws-sdk-go/issues/2746)
* `service/s3/s3manager`: Add Download Buffer Provider ([#2823](https://github.com/aws/aws-sdk-go/pull/2823))
  * Adds a new `BufferProvider` member for specifying how part data can be buffered in memory when copying from the http response body.
  * Windows platforms will now default to buffering 1MB per part to reduce contention when downloading files.
  * Non-Windows platforms will continue to employ a non-buffering behavior.
  * Fixes [#2180](https://github.com/aws/aws-sdk-go/issues/2180)
  * Fixes [#2662](https://github.com/aws/aws-sdk-go/issues/2662)

Release v1.24.6 (2019-09-25)
===

### Service Client Updates
* `service/dms`: Updates service API, documentation, and examples
  * This release adds a new DeleteConnection API to delete the connection between a replication instance and an endpoint. It also adds an optional S3 setting to specify the precision of any TIMESTAMP column values written to an S3 object file in .parquet format.
* `service/globalaccelerator`: Updates service API and documentation
* `service/sagemaker`: Updates service API and documentation
  * Enable G4D and R5 instances in SageMaker Hosting Services

Release v1.24.5 (2019-09-24)
===

### Service Client Updates
* `service/comprehendmedical`: Updates service API and documentation
* `service/datasync`: Updates service API and documentation
* `service/transcribe`: Updates service API and documentation

### SDK Enhancements
* `private/model/api`: Skip unsupported API models during code generation ([#2849](https://github.com/aws/aws-sdk-go/pull/2849))
  * Adds support for removing API modeled operations that use unsupported features. If a API model results in having no operations it will be skipped.

Release v1.24.4 (2019-09-23)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds-data`: Updates service API, documentation, paginators, and examples
* `service/redshift`: Updates service API, documentation, and paginators
  * Adds API operation DescribeNodeConfigurationOptions and associated data structures.

Release v1.24.3 (2019-09-20)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * G4 instances are Amazon EC2 instances based on NVIDIA T4 GPUs and are designed to provide cost-effective machine learning inference for applications, like image classification, object detection, recommender systems, automated speech recognition, and language translation. G4 instances are also a cost-effective platform for building and running graphics-intensive applications, such as remote graphics workstations, video transcoding, photo-realistic design, and game streaming in the cloud. To get started with G4 instances visit https://aws.amazon.com/ec2/instance-types/g4.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/greengrass`: Updates service API and documentation
  * Greengrass OTA service now returns the updated software version in the PlatformSoftwareVersion parameter of a CreateSoftwareUpdateJob response
* `service/rds`: Updates service API and documentation
  * Add a new LeaseID output field to DescribeReservedDBInstances, which shows the unique identifier for the lease associated with the reserved DB instance. AWS Support might request the lease ID for an issue related to a reserved DB instance.
* `service/workspaces`: Updates service API and documentation
  * Adds the WorkSpaces restore feature

Release v1.24.2 (2019-09-19)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for container image manifest digests. This enables you to identify all tasks launched using a container image pulled from ECR in order to correlate what was built with where it is running.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * AWS Glue DevEndpoints now supports GlueVersion, enabling you to choose Apache Spark 2.4.3 (in addition to Apache Spark 2.2.1). In addition to supporting the latest version of Spark, you will also have the ability to choose between Python 2 and Python 3.
* `service/mediaconnect`: Updates service API and documentation

Release v1.24.1 (2019-09-18)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Amazon API Gateway simplifies accessing PRIVATE APIs by allowing you to associate one or more Amazon Virtual Private Cloud (VPC) Endpoints to a private API. API Gateway will create and manage DNS alias records necessary for easily invoking the private APIs. With this feature, you can leverage private APIs in web applications hosted within your VPCs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/ram`: Updates service API, documentation, and paginators
* `service/waf-regional`: Updates service API and documentation

Release v1.24.0 (2019-09-17)
===

### Service Client Updates
* `service/athena`: Updates service API and documentation
  * This release adds DataManifestLocation field indicating the location and file name of the data manifest file. Users can get a list of files that the Athena query wrote or intended to write from the manifest file.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iam`: Updates service documentation and examples
  * Documentation updates for iam
* `service/personalize`: Updates service API and documentation

### SDK Features
* `service/s3/s3manager`: Add Upload Buffer Provider ([#2792](https://github.com/aws/aws-sdk-go/pull/2792))
  * Adds a new `BufferProvider` member for specifying how part data can be buffered in memory.
  * Windows platforms will now default to buffering 1MB per part to reduce contention when uploading files.
  * Non-Windows platforms will continue to employ a non-buffering behavior.

### SDK Enhancements
* `awstesting/integration/performance/s3UploadManager`: Extended to support benchmarking and usage of new `BufferProvider` ([#2792](https://github.com/aws/aws-sdk-go/pull/2792))

Release v1.23.22 (2019-09-16)
===

### Service Client Updates
* `service/eks`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added support for multi-DRM SPEKE with CMAF outputs, MP3 ingest, and options for improved video quality.

### SDK Enhancements
* `aws/client`: Adds configurations to the default retryer ([#2830](https://github.com/aws/aws-sdk-go/pull/2830))
  * Exposes members of the default retryer. Adds NoOpRetryer to support no retry behavior. 
  * Updates the underlying logic used by the default retryer to calculate jittered delay for retry. 
  * Fixes [#2829](https://github.com/aws/aws-sdk-go/issues/2829)

Release v1.23.21 (2019-09-12)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Fix for FleetActivityStatus and FleetStateCode enum
* `service/elasticloadbalancingv2`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive now supports High Efficiency Video Coding (HEVC) for standard-definition (SD), high-definition (HD), and ultra-high-definition (UHD) encoding with HDR support.Encoding with HEVC offers a number of advantages. While UHD video requires an advanced codec beyond H.264 (AVC), high frame rate (HFR) or High Dynamic Range (HDR) content in HD also benefit from HEVC's advancements. In addition, benefits can be achieved with HD and SD content even if HDR and HFR are not needed.
* `service/workmailmessageflow`: Adds new service

### SDK Enhancements
* `aws`: Add value/pointer conversion functions for all basic number types ([#2740](https://github.com/aws/aws-sdk-go/pull/2740))
  * Adds value and pointer conversion utilities for the remaining set of integer and float number types.

Release v1.23.20 (2019-09-11)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release adds support for new data fields and log format in VPC flow logs.
* `service/email`: Updates service documentation
  * Updated API documentation to correct broken links, and to update content based on customer feedback.
* `service/mediaconnect`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * This release allows customers to specify a custom parameter group when creating a Read Replica, for DB engines which support this feature.
* `service/states`: Updates service API and documentation
  * Fixing letter case in Map history event details to be small case

### SDK Enhancements
* Enabled verbose logging for continuous integration tests ([#2815](https://github.com/aws/aws-sdk-go/pull/2815))

Release v1.23.19 (2019-09-10)
===

### Service Client Updates
* `service/storagegateway`: Updates service API and documentation
  * The CloudWatchLogGroupARN parameter of the UpdateGatewayInformation API allows for configuring the gateway to use a CloudWatch log-group where Storage Gateway health events will be logged.

Release v1.23.18 (2019-09-09)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/appstream`: Updates service API and documentation
  * IamRoleArn support in CreateFleet, UpdateFleet, CreateImageBuilder APIs
* `service/ec2`: Updates service API and documentation
  * This release expands Site-to-Site VPN tunnel options to allow customers to restrict security algorithms and configure timer settings for VPN connections. Customers can specify these new options while creating new VPN connections, or they can modify the tunnel options on existing connections using a new API.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/marketplacecommerceanalytics`: Updates service API and documentation
  * Add FDP+FPS (monthly_revenue_field_demonstration_usage + monthly_revenue_flexible_payment_schedule)  to Marketplace Commerce Analytics Service
* `service/qldb`: Adds new service
* `service/qldb-session`: Adds new service
* `service/robomaker`: Updates service API and documentation

### SDK Bugs
* Fixed failing tests when executed as a module dependency ([#2817](https://github.com/aws/aws-sdk-go/pull/2817))
Release v1.23.17 (2019-09-06)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kinesisanalytics`: Updates service documentation
  * Documentation updates for kinesisanalytics

### SDK Bugs
* `awstesting`: Fixes AssertXML to correctly assert on differences ([#2804](https://github.com/aws/aws-sdk-go/pull/2804))
Release v1.23.16 (2019-09-05)
===

### Service Client Updates
* `service/config`: Updates service API, documentation, and paginators
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.23.15 (2019-09-04)
===

### Service Client Updates
* `service/eks`: Updates service API, documentation, and paginators
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/states`: Updates service API and documentation
  * Added support for new history events
* `service/transcribe`: Updates service API and documentation

Release v1.23.14 (2019-09-03)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for attaching Amazon Elastic Inference accelerators to your containers. This enables you to run deep learning inference workloads with hardware acceleration in a more efficient way.
* `service/gamelift`: Updates service API and documentation
  * You can now make use of PKI resources to provide more secure connections between your game clients and servers.  To learn more, please refer to the public Amazon GameLift documentation.
* `service/resourcegroupstaggingapi`: Updates service documentation
  * Documentation updates for resourcegroupstaggingapi

Release v1.23.13 (2019-08-30)
===

### Service Client Updates
* `service/apigatewaymanagementapi`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for modifying the cluster settings for existing clusters, which enables you to toggle whether Container Insights is enabled or not. Support is also introduced for custom log routing using the ECS FireLens integration.
* `service/mq`: Updates service API and documentation
  * Adds support for updating security groups selection of an Amazon MQ broker.

### SDK Bugs
* `aws/csm`: Fix metricChan's unsafe atomic operations ([#2785](https://github.com/aws/aws-sdk-go/pull/2785))
  * Fixes [#2784](https://github.com/aws/aws-sdk-go/issues/2784) test failure caused by the metricChan.paused member being a value instead of a pointer. If the metricChan value was ever copied the atomic operations performed on paused would be invalid.
* `aws/client`: Updates logic for request retry delay calculation ([#2796](https://github.com/aws/aws-sdk-go/pull/2796))
  * Updates logic for calculating the delay after which a request can be retried. Retry delay now includes the Retry-After duration specified in a request. Fixes broken test for retry delays for throttled exceptions.
  * Fixes [#2795](https://github.com/aws/aws-sdk-go/issues/2795)
Release v1.23.12 (2019-08-29)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API, documentation, and paginators
* `service/codepipeline`: Updates service API and documentation
  * Introducing pipeline execution trigger details in ListPipelineExecutions API.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for including Docker container IDs in the API response when describing and stopping tasks. This enables customers to easily map containers to the tasks they are associated with.
* `service/elasticache`: Updates service API and documentation
  * Amazon ElastiCache for Redis now supports encryption at rest using customer managed customer master keys (CMKs) in AWS Key Management Service (KMS). Amazon ElastiCache now supports cluster names upto 40 characters for replicationGoups and upto 50 characters for cacheClusters.
* `service/lambda`: Updates service API, documentation, and paginators
  * Adds a "MaximumBatchingWindowInSeconds" parameter to event source mapping api's. Usable by Dynamodb and Kinesis event sources.

### SDK Enhancements
* `aws/ec2metadata`: Add marketplaceProductCodes to EC2 Instance Identity Document
  * Adds `MarketplaceProductCodes` to the EC2 Instance Metadata's Identity Document. The ec2metadata client will now retrieve these values if they are available.
  * Fixes [#2781](https://github.com/aws/aws-sdk-go/issues/2781)
* `private/protocol`: Add support for parsing fractional time ([#2760](https://github.com/aws/aws-sdk-go/pull/2760))
  * Fixes the SDK's ability to parse fractional unix timestamp values and added tests.
  * Fixes [#1448](https://github.com/aws/aws-sdk-go/pull/1448)

Release v1.23.11 (2019-08-28)
===

### Service Client Updates
* `service/globalaccelerator`: Updates service API and documentation
* `service/mediaconvert`: Updates service API and documentation
  * This release adds the ability to send a job to an on-demand queue while simulating the performance of a job sent to a reserved queue. Use this setting to estimate the number of reserved transcoding slots (RTS) you need for a reserved queue.
* `service/sqs`: Updates service API and documentation
  * Added support for message system attributes, which currently lets you send AWS X-Ray trace IDs through Amazon SQS.

Release v1.23.10 (2019-08-27)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/organizations`: Updates service documentation
  * Documentation updates for organizations

### SDK Bugs
* `service/ec2`: Fix int overflow in minTime on 386 and arm ([#2787](https://github.com/aws/aws-sdk-go/pull/2787))
  * Fixes [2786](https://github.com/aws/aws-sdk-go/issues/2786) int overflow issue on 32-bit platforms like 386 and arm.
Release v1.23.9 (2019-08-26)
===

### Service Client Updates
* `service/securityhub`: Updates service API
* `service/ssm`: Updates service API and documentation
  * This feature adds "default tier" to the AWS Systems Manager Parameter Store for parameter creation and update. AWS customers can now set the "default tier" to one of the following values: Standard (default), Advanced or Intelligent-Tiering.  This allows customers to create advanced parameters or parameters in corresponding tiers with one setting rather than code change to specify parameter tiers.

Release v1.23.8 (2019-08-23)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release of EC2 VM Import Export adds support for exporting Amazon Machine Image(AMI)s to a VM file
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediapackage-vod`: Updates service API and documentation
* `service/transcribe`: Updates service API and documentation

### SDK Enhancements
* `aws/session`: Add support for CSM options from shared config file ([#2768](https://github.com/aws/aws-sdk-go/pull/2768))
  * Adds support for enabling and controlling the Client Side Metrics (CSM) reporting from the shared configuration files in addition to the environment variables.

### SDK Bugs
* `service/s3/s3crypto`: Fix tmp file not being deleted after upload ([#2776](https://github.com/aws/aws-sdk-go/pull/2776))
  * Fixes the s3crypto's getWriterStore utiliy's send handler not cleaning up the temporary file after Send completes.
* `private/protocol`: Add protocol tests for blob types and headers ([#2770](https://github.com/aws/aws-sdk-go/pull/2770))
  * Adds RESTJSON and RESTXML protocol tests for blob headers.
  * Related to [#750](https://github.com/aws/aws-sdk-go/issues/750)
* `service/dynamodb/expression`: Improved reporting of bad key conditions ([#2775](https://github.com/aws/aws-sdk-go/pull/2775))
  * Improved error reporting when invalid key conditions are constructed using KeyConditionBuilder
Release v1.23.7 (2019-08-22)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * This release allows users to enable RDS Data API while creating Aurora Serverless databases.

### SDK Bugs
* `aws/request`: Fix IsErrorRetryable returning true for nil error ([#2774](https://github.com/aws/aws-sdk-go/pull/2774))
  * Fixes [#2773](https://github.com/aws/aws-sdk-go/pull/2773) where the IsErrorRetryable helper was incorrectly returning true for nil errors passed in. IsErrorRetryable will now correctly return false when the passed in error is nil like documented.

Release v1.23.6 (2019-08-21)
===

### Service Client Updates
* `service/elasticache`: Updates service API and documentation
  * ElastiCache extends support for Scale down for Redis Cluster-mode enabled and disabled replication groups
* `service/forecast`: Adds new service
* `service/forecastquery`: Adds new service
* `service/personalize-runtime`: Updates service API
* `service/rekognition`: Updates service documentation
  * Documentation updates for Amazon Rekognition.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker now supports Amazon EFS and Amazon FSx for Lustre file systems as data sources for training machine learning models. Amazon SageMaker now supports running training jobs on ml.p3dn.24xlarge instance type. This instance type is offered as a limited private preview for certain SageMaker customers. If you are interested in joining the private preview, please reach out to the SageMaker Product Management team via AWS Support."
* `service/sqs`: Updates service API and documentation
  * This release provides a way to add metadata tags to a queue when it is created. You can use tags to organize and identify your Amazon SQS queues for cost allocation.

### SDK Enhancements
* `aws/session`: Ignore invalid shared config file when not used ([#2731](https://github.com/aws/aws-sdk-go/pull/2731))
  * Updates the Session to not fail to load when credentials are provided via the environment variable, the AWS_PROFILE/Option.Profile have not been specified, and the shared config has not been enabled.
  * Fixes [#2455](https://github.com/aws/aws-sdk-go/issues/2455)

Release v1.23.5 (2019-08-20)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/appstream`: Updates service API and documentation
  * Includes API updates to support streaming through VPC endpoints for image builders and stacks.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker introduces Managed Spot Training. Increases the maximum number of metric definitions to 40 for SageMaker Training and Hyperparameter Tuning Jobs. SageMaker Neo adds support for Acer aiSage and Qualcomm QCS605 and QCS603.
* `service/transfer`: Updates service API and documentation
  * New field in response of TestIdentityProvider

Release v1.23.4 (2019-08-19)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/cur`: Updates service API and documentation
  * New IAM permission required for editing AWS Cost and Usage Reports - Starting today, you can allow or deny IAM users permission to edit Cost & Usage Reports through the API and the Billing and Cost Management console. To allow users to edit Cost & Usage Reports, ensure that they have 'cur: ModifyReportDefinition' permission. Refer to the technical documentation (https://docs.aws.amazon.com/aws-cost-management/latest/APIReference/API_cur_ModifyReportDefinition.html) for additional details.

Release v1.23.3 (2019-08-16)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for controlling the usage of swap space on a per-container basis for Linux containers.
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR  has introduced an account level configuration called Block Public Access that allows you to block clusters with ports open to traffic from public IP sources (i.e. 0.0.0.0/0 for IPv4 and ::/0 for IPv6) from launching.  Individual ports or port ranges can be added as exceptions to allow public access.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/robomaker`: Updates service API and documentation

Release v1.23.2 (2019-08-15)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/athena`: Updates service API and documentation
  * This release adds support for querying S3 Requester Pays buckets. Users can enable this feature through their Workgroup settings.
* `service/codecommit`: Updates service API and documentation
  * This release adds an API, BatchGetCommits, that allows retrieval of metadata for multiple commits in an AWS CodeCommit repository.
* `service/ec2`: Updates service API and documentation
  * This release adds an option to use private certificates from AWS Certificate Manager (ACM) to authenticate a Site-to-Site VPN connection's tunnel endpoints and customer gateway device.
* `service/glue`: Updates service API, documentation, and paginators
  * GetJobBookmarks API is withdrawn.
* `service/storagegateway`: Updates service API and documentation
  * CreateSnapshotFromVolumeRecoveryPoint API supports new parameter: Tags (to be attached to the created resource)

### SDK Enhancements
* `service/kinesis`: Add support for retrying service specific API errors ([#2751](https://github.com/aws/aws-sdk-go/pull/2751)
  * Adds support for retrying the Kinesis API error, LimitExceededException.
  * Fixes [#1376](https://github.com/aws/aws-sdk-go/issues/1376)
* `aws/credentials/stscreds`: Add STS and Assume Role specific retries ([#2752](https://github.com/aws/aws-sdk-go/pull/2752))
  * Adds retries to specific STS API errors to the STS AssumeRoleWithWebIdentity credential provider, and STS API operations in general.

Release v1.23.1 (2019-08-14)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds a new API called SendDiagnosticInterrupt, which allows you to send diagnostic interrupts to your EC2 instance.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.23.0 (2019-08-13)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

### SDK Features
* SDK code generation will no longer remove stutter from operations and type names in service API packages. New API operations and types will not have the service's name removed from them. The SDK was previously squashing API types if removing stutter from a type resulted in a name of a type that already existed. The existing type would be deleted. Only the renamed type remained. This has been fixed, and previously deleted types are now available.
  * `AWS Glue`'s `GlueTable` with `Table`.  The API's previously deleted `Table` is available as `TableData`.
  * `AWS IoT Events`'s `IotEventsAction` with `Action`. The previously deleted `Action` is available as `ActionData`.

### SDK Bugs
* `private/model/api`: Fix broken shape stutter rename during generation ([#2747](https://github.com/aws/aws-sdk-go/pull/2747))
  * Fixes the SDK's code generation incorrectly renaming types and operations. The code generation would incorrectly rename an API type by removing the service's name from the type's name. This was done without checking for if a type with the new name already existed. Causing the SDK to replace the existing type with the renamed one.
  * Fixes [#2741](https://github.com/aws/aws-sdk-go/issues/2741)
* `private/model/api`: Fix API doc being generated with wrong value ([#2748](https://github.com/aws/aws-sdk-go/pull/2748))
  * Fixes the SDK's generated API documentation for structure member being generated with the wrong documentation value when the member was included multiple times in the model doc-2.json file, but under different types.
Release v1.22.4 (2019-08-12)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation
* `service/autoscaling`: Updates service documentation
  * Amazon EC2 Auto Scaling now supports a new Spot allocation strategy "capacity-optimized" that fulfills your request using Spot Instance pools that are optimally chosen based on the available Spot capacity.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/monitoring`: Updates service documentation
  * Documentation updates for monitoring
* `service/rekognition`: Updates service API
  * Adding new Emotion, Fear

Release v1.22.3 (2019-08-09)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/guardduty`: Updates service API and documentation
  * New "evidence" field in the finding model to provide evidence information explaining why the finding has been triggered. Currently only threat-intelligence findings have this field. Some documentation updates.
* `service/iot`: Updates service API and documentation
  * This release adds Quality of Service (QoS) support for AWS IoT rules engine republish action.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert has added support for multi-DRM SPEKE with CMAF outputs, MP3 ingest, and options for improved video quality.
* `service/redshift`: Updates service API and documentation
  * Add expectedNextSnapshotScheduleTime and expectedNextSnapshotScheduleTimeStatus to redshift cluster object.
* `service/runtime.lex`: Updates service API and documentation

Release v1.22.2 (2019-08-08)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * CodeBuild adds CloudFormation support for SourceCredential
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API, documentation, and paginators
  * You can now use AWS Glue to find matching records across dataset even without identifiers to join on by using the new FindMatches ML Transform. Find related products, places, suppliers, customers, and more by teaching a custom machine learning transformation that you can use to identify matching matching records as part of your analysis, data cleaning, or master data management project by adding the FindMatches transformation to your Glue ETL Jobs. If your problem is more along the lines of deduplication, you can use the FindMatches in much the same way to identify customers who have signed up more than ones, products that have accidentally been added to your product catalog more than once, and so forth. Using the FindMatches MLTransform, you can teach a Transform your definition of a duplicate through examples, and it will use machine learning to identify other potential duplicates in your dataset. As with data integration, you can then use your new Transform in your deduplication projects by adding the FindMatches transformation to your Glue ETL Jobs. This release also contains additional APIs that support AWS Lake Formation.
* `service/lakeformation`: Adds new service
* `service/opsworkscm`: Updates service API
  * This release adds support for Chef Automate 2 specific engine attributes.

Release v1.22.1 (2019-08-07)
===

### Service Client Updates
* `service/application-insights`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.22.0 (2019-08-06)
===

### Service Client Updates
* `service/batch`: Updates service documentation
  * Documentation updates for AWS Batch

### SDK Features
* `aws/session`: Corrected order of SDK environment and shared config loading.
  * Environment credentials have precedence over shared config credentials even if the AWS_PROFILE environment credentials are present. The session.Options.Profile value needs to be used to specify a profile for shared config to have precedence over environment credentials. #2694 incorrectly gave AWS_PROFILE for shared config precedence over environment credentials as well.

### SDK Bugs
* `aws/session`: Fix credential loading order for env and shared config ([#2729](https://github.com/aws/aws-sdk-go/pull/2729))
  * Fixes the credential loading order for environment credentials, when the presence of an AWS_PROFILE value is also provided. The environment credentials have precedence over the AWS_PROFILE.
  * Fixes [#2727](https://github.com/aws/aws-sdk-go/issues/2727)
Release v1.21.10 (2019-08-05)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `service/ec2`: Updates service API
  * Amazon EC2 now supports a new Spot allocation strategy "Capacity-optimized" that fulfills your request using Spot Instance pools that are optimally chosen based on the available Spot capacity.
* `service/iot`: Updates service API and documentation
  * In this release, AWS IoT Device Defender introduces audit mitigation actions that can be applied to audit findings to help mitigate security issues.

Release v1.21.9 (2019-08-02)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/sts`: Updates service documentation
  * Documentation updates for sts

### SDK Enhancements
* `aws/endpoints`: Expose DNSSuffix for partitions ([#2711](https://github.com/aws/aws-sdk-go/pull/2711))
  * Exposes the underlying partition metadata's DNSSuffix value via the `DNSSuffix` method on the endpoint's `Partition` type. This allows access to the partition's DNS suffix, e.g. "amazon.com".
  * Fixes [#2710](https://github.com/aws/aws-sdk-go/issues/2710)

Release v1.21.8 (2019-07-30)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * MediaConvert adds support for specifying priority (-50 to 50) on jobs submitted to on demand or reserved queues
* `service/polly`: Updates service API and documentation
  * Amazon Polly adds support for Neural text-to-speech engine.
* `service/route53`: Updates service API and documentation
  * Amazon Route 53 now supports the Middle East (Bahrain) Region (me-south-1) for latency records, geoproximity records, and private DNS for Amazon VPCs in that region.

Release v1.21.7 (2019-07-29)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * This release supports better exception handling for merges.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.21.6 (2019-07-26)
===

### Service Client Updates
* `service/batch`: Updates service API, documentation, and paginators
  * AWS Batch now supports SDK auto-pagination and Job-level docker devices.
* `service/ce`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * You can now create EC2 Capacity Reservations using Availability Zone ID or Availability Zone name. You can view usage of Amazon EC2 Capacity Reservations per AWS account.
* `service/glue`: Updates service API, documentation, and paginators
  * This release provides GetJobBookmark and GetJobBookmarks APIs. These APIs enable users to look at specific versions or all versions of the JobBookmark for a specific job. This release also enables resetting the job bookmark to a specific run via an enhancement of the ResetJobBookmark API.
* `service/greengrass`: Updates service API and documentation
  * Greengrass OTA service supports openwrt/aarch64 and openwrt/armv7l platforms.
* `service/logs`: Updates service API and documentation
  * Allow for specifying multiple log groups in an Insights query, and deprecate storedByte field for LogStreams and interleaved field for FilterLogEventsRequest.
* `service/mediaconnect`: Updates service API and documentation

Release v1.21.5 (2019-07-25)
===

### Service Client Updates
* `service/ecr`: Updates service API and documentation
  * This release adds support for immutable image tags.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert has added several features including support for: audio normalization using ITU BS.1770-3, 1770-4 algorithms, extension of job progress indicators, input cropping rectangle & output position rectangle filters per input, and dual SCC caption mapping to additional codecs and containers.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive is adding Input Clipping, Immediate Mode Input Switching, and Dynamic Inputs.

Release v1.21.4 (2019-07-24)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release introduces support for split tunnel with AWS Client VPN, and also adds support for opt-in Regions in DescribeRegions API. In addition, customers can now also tag Launch Templates on creation.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * This release provides GlueVersion option for Job APIs and WorkerType option for DevEndpoint APIs. Job APIs enable users to pick specific GlueVersion for a specific job and pin the job to a specific runtime environment. DevEndpoint APIs enable users to pick different WorkerType for memory intensive workload.
* `service/pinpoint`: Updates service API and documentation
  * This release adds support for programmatic access to many of the same campaign metrics that are displayed on the Amazon Pinpoint console. You can now use the Amazon Pinpoint API to monitor and assess performance data for campaigns, and integrate metrics data with other reporting tools. We update the metrics data continuously, resulting in a data latency timeframe that is limited to approximately two hours.
* `service/sts`: Updates service API and documentation
  * New STS GetAccessKeyInfo API operation that returns the account identifier for the specified access key ID.

Release v1.21.3 (2019-07-23)
===

### Service Client Updates
* `service/secretsmanager`: Updates service API and documentation
  * This release increases the maximum allowed size of SecretString or SecretBinary from 7KB to 10KB in the CreateSecret, UpdateSecret, PutSecretValue and GetSecretValue APIs. This release also increases the maximum allowed size of ResourcePolicy from 4KB to 20KB in the GetResourcePolicy and PutResourcePolicy APIs.
* `service/ssm`: Updates service API and documentation
  * You can now use Maintenance Windows to select a resource group as the target. By selecting a resource group as the target of a Maintenance Window, customers can perform routine tasks across different resources such as Amazon Elastic Compute Cloud (AmazonEC2) instances, Amazon Elastic Block Store (Amazon EBS) volumes, and Amazon Simple Storage Service(Amazon S3) buckets within the same recurring time window.

Release v1.21.2 (2019-07-22)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mq`: Updates service API and documentation
  * Adds support for AWS Key Management Service (KMS) to offer server-side encryption. You can now select your own customer managed CMK, or use an AWS managed CMK in your KMS  account.
* `service/shield`: Updates service API and documentation
  * Adding new VectorType (HTTP_Reflection) and related top contributor types to describe WordPress Pingback DDoS attacks.

### SDK Enhancements
* Fixup SDK source formating, test error checking, and simplify type conervsions
  * [#2703](https://github.com/aws/aws-sdk-go/pull/2703), [#2704](https://github.com/aws/aws-sdk-go/pull/2704), [#2705](https://github.com/aws/aws-sdk-go/pull/2705), [#2706](https://github.com/aws/aws-sdk-go/pull/2706), [#2707](https://github.com/aws/aws-sdk-go/pull/2707), [#2708](https://github.com/aws/aws-sdk-go/pull/2708)

### SDK Bugs
* `aws/request`: Fix SDK error checking when seeking readers ([#2696](https://github.com/aws/aws-sdk-go/pull/2696))
  * Fixes the SDK handling of seeking a reader to ensure errors are not lost, and are bubbled up.
  * In several places the SDK ignored Seek errors when attempting to determine a reader's length, or rewinding the reader for retry attempts.
  * Related to [#2525](https://github.com/aws/aws-sdk-go/issues/2525)
Release v1.21.1 (2019-07-19)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iotevents`: Updates service API and documentation
* `service/sqs`: Updates service documentation
  * This release updates the information about the availability of FIFO queues and includes miscellaneous fixes.

Release v1.21.0 (2019-07-18)
===

### Service Client Updates
* `service/codedeploy`: Updates service documentation
  * Documentation updates for codedeploy
* `service/comprehend`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for cluster settings. Cluster settings specify whether CloudWatch Container Insights is enabled or disabled for the cluster.
* `service/elasticache`: Updates service documentation
  * Updates for Elasticache

### SDK Features
* `aws/session`: Add support for assuming role via Web Identity Token ([#2667](https://github.com/aws/aws-sdk-go/pull/2667))
  * Adds support for assuming an role via the Web Identity Token. Allows for OIDC token files to be used by specifying the token path through the AWS_WEB_IDENTITY_TOKEN_FILE, and AWS_ROLE_ARN environment variables.

### SDK Bugs
* `aws/session`: Fix SDK AWS_PROFILE and static environment credential behavior ()
  * Fixes the SDK's behavior when determining the source of credentials to load. Previously the SDK would ignore the AWS_PROFILE environment, if static environment credentials were also specified.
  * If both AWS_PROFILE and static environment credentials are defined, the SDK will load any credentials from the shared config/credentials file for the AWS_PROFILE first. Only if there are no credentials defined in the shared config/credentials file will the SDK use the static environment credentials instead.
Release v1.20.21 (2019-07-17)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation
  * Documentation updates for autoscaling
* `service/config`: Updates service API
* `service/dms`: Updates service API and documentation
  * S3 endpoint settings update: 1) Option to append operation column to full-load files. 2) Option to add a commit timestamp column to full-load and cdc files. Updated DescribeAccountAttributes to include UniqueAccountIdentifier.

Release v1.20.20 (2019-07-12)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/es`: Updates service API
  * Amazon Elasticsearch Service now supports M5, C5, and R5 instance types.
* `service/iam`: Updates service API
  * Removed exception that was indicated but never thrown for IAM GetAccessKeyLastUsed API
* `service/robomaker`: Updates service API and documentation

Release v1.20.19 (2019-07-11)
===

### Service Client Updates
* `service/eventbridge`: Adds new service
* `service/events`: Updates service API and documentation
  * Adds APIs for partner event sources, partner event buses, and custom event buses. These new features are managed in the EventBridge service.

### SDK Enhancements
* `aws/session`: Add Assume role for credential process from aws shared config ([#2674](https://github.com/aws/aws-sdk-go/pull/2674))
  * Adds support for assuming role using credential process from the shared config file. Also updated SDK's environment testing and added SDK's CI testing with Windows.
* `aws/csm`: Add support for AWS_CSM_HOST env option ([#2677](https://github.com/aws/aws-sdk-go/pull/2677))
  * Adds support for a host to be configured for the SDK's metric reporting Client Side Metrics (CSM) client via the AWS_CSM_HOST environment variable.

Release v1.20.18 (2019-07-10)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glacier`: Updates service documentation
  * Documentation updates for glacier
* `service/quicksight`: Updates service API and documentation
  * Amazon QuickSight now supports embedding dashboards for all non-federated QuickSight users. This includes IAM users, AD users and users from the QuickSight user pool. The get-dashboard-embed-url API accepts QUICKSIGHT as identity type with a user ARN to authenticate the embeddable dashboard viewer as a non-federated user.
* `service/servicecatalog`: Updates service API and documentation
  * This release adds support for Parameters in ExecuteProvisionedProductServiceAction and adds functionality to get the default parameter values for a Self-Service Action execution against a Provisioned Product via DescribeServiceActionExecutionParameters

Release v1.20.17 (2019-07-09)
===

### Service Client Updates
* `service/amplify`: Updates service API and documentation
* `service/config`: Updates service API and documentation
* `service/elasticfilesystem`: Updates service API and documentation
  * EFS customers can now enable Lifecycle Management for all file systems. You can also now select from one of four Lifecycle Management policies (14, 30, 60 and 90 days), to automatically move files that have not been accessed for the period of time defined by the policy, from the EFS Standard storage class to the EFS Infrequent Access (IA) storage class. EFS IA provides price/performance that is cost-optimized for files that are not accessed every day.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/gamelift`: Updates service API and documentation
  * GameLift FlexMatch now supports matchmaking of up to 200 players per game session, and FlexMatch can now automatically backfill your game sessions whenever there is an open slot.
* `service/kinesis-video-archived-media`: Updates service API, documentation, and paginators
* `service/kinesisvideo`: Updates service API and paginators
  * Add "GET_DASH_STREAMING_SESSION_URL" as an API name to the GetDataEndpoint API.
* `service/monitoring`: Updates service API and documentation
  * This release adds three new APIs (PutAnomalyDetector, DeleteAnomalyDetector, and DescribeAnomalyDetectors) to support the new feature, CloudWatch Anomaly Detection. In addition, PutMetricAlarm and DescribeAlarms APIs are updated to support management of Anomaly Detection based alarms.
* `service/waf`: Updates service API and documentation
  * Updated SDK APIs to add tags to WAF Resources: WebACL, Rule, Rulegroup and RateBasedRule. Tags can also be added during creation of these resources.
* `service/waf-regional`: Updates service API and documentation

Release v1.20.16 (2019-07-08)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.20.15 (2019-07-03)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * AssignPrivateIpAddresses response includes two new fields: AssignedPrivateIpAddresses, NetworkInterfaceId
* `service/rds`: Updates service API and documentation
  * This release supports Cross-Account Cloning for Amazon Aurora clusters.
* `service/s3`: Updates service API, documentation, and examples
  * Add S3 x-amz-server-side-encryption-context support.
* `service/swf`: Updates service API and documentation
  * This release adds APIs that allow adding and removing tags to a SWF domain, and viewing tags for a domain. It also enables adding tags when creating a domain.

Release v1.20.14 (2019-07-02)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Adding ImageBuilderName in Fleet API and Documentation updates for AppStream.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediastore`: Updates service API, documentation, and paginators
  * This release adds support for tagging, untagging, and listing tags for AWS Elemental MediaStore containers.

Release v1.20.13 (2019-07-01)
===

### Service Client Updates
* `service/docdb`: Updates service API and documentation
  * This release provides support for cluster delete protection and the ability to stop and start clusters.
* `service/ec2`: Updates service API and documentation
  * This release adds support for specifying a maximum hourly price for all On-Demand and Spot instances in both Spot Fleet and EC2 Fleet.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/organizations`: Updates service API and documentation
  * Specifying the tag key and tag value is required for tagging requests.
* `service/rds`: Updates service API and documentation
  * This release adds support for RDS DB Cluster major version upgrade

Release v1.20.12 (2019-06-28)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * You can now launch 8xlarge and 16xlarge instance sizes on the general purpose M5 and memory optimized R5 instance types.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service API and documentation
  * ClusterAvailabilityStatus: The availability status of the cluster for queries. Possible values are the following: Available, Unavailable, Maintenance, Modifying, Failed.
* `service/workspaces`: Updates service API and documentation
  * Minor API fixes for WorkSpaces.

Release v1.20.11 (2019-06-27)
===

### Service Client Updates
* `service/directconnect`: Updates service API and documentation
  * Tags will now be included in the API responses of all supported resources (Virtual interfaces, Connections, Interconnects and LAGs). You can also add tags while creating these resources.
* `service/ec2-instance-connect`: Adds new service
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/pinpoint`: Updates service API and documentation
  * This release includes editorial updates for the Amazon Pinpoint API documentation.
* `service/workspaces`: Updates service API and documentation
  * Added support for the WorkSpaces restore feature and copying WorkSpaces Images across AWS Regions.

Release v1.20.10 (2019-06-27)
===

### Service Client Updates
* `service/dynamodb`: Updates service documentation and examples
  * Documentation updates for dynamodb
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.20.9 (2019-06-26)
===

### Service Client Updates
* `service/apigatewayv2`: Updates service API and documentation
* `service/codecommit`: Updates service API and documentation
  * This release supports better exception handling for merges.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.20.8 (2019-06-25)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and paginators
  * Starting today, you can use Traffic Mirroring  to copy network traffic from an elastic network interface of Amazon EC2 instances and then send it to out-of-band security and monitoring appliances for content inspection, threat monitoring, and troubleshooting. These appliances can be deployed as individual instances, or as a fleet of instances behind a Network Load Balancer with a User Datagram Protocol (UDP) listener. Traffic Mirroring supports filters and packet truncation, so that you only extract the traffic of interest to monitor by using monitoring tools of your choice.
* `service/eks`: Updates service API

Release v1.20.7 (2019-06-24)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Customers can pick different security policies (TLS version + cipher suite) for custom domains in API Gateway
* `service/apigatewayv2`: Updates service API and documentation
  * Customers can get information about security policies set on custom domain resources in API Gateway
* `service/application-insights`: Adds new service
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/fsx`: Updates service API and documentation
* `service/resourcegroupstaggingapi`: Updates service API, documentation, and paginators
  * Updated service APIs and documentation.
* `service/securityhub`: Updates service API, documentation, and paginators
* `service/service-quotas`: Adds new service
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager now supports deleting a specific version of a SSM Document.

Release v1.20.6 (2019-06-21)
===

### Service Client Updates
* `service/devicefarm`: Updates service documentation
  * This release includes updated documentation about the default timeout value for test runs and remote access sessions. This release also includes miscellaneous bug fixes for the documentation.
* `service/iam`: Updates service API, documentation, and examples
  * We are making it easier for you to manage your permission guardrails i.e. service control policies by enabling you to retrieve the last timestamp when an AWS service was accessed within an account or AWS Organizations entity.
* `service/kinesis-video-media`: Updates service documentation
* `service/mediapackage`: Updates service API and documentation
  * Added two new origin endpoint fields for configuring which SCTE-35 messages are treated as advertisements.

Release v1.20.5 (2019-06-20)
===

### Service Client Updates
* `service/acm-pca`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API, documentation, and paginators
  * Starting today, you can now use workflows in AWS Glue to author directed acyclic graphs (DAGs) of Glue triggers, crawlers and jobs. Workflows enable orchestration of your ETL workloads by building dependencies between Glue entities (triggers, crawlers and jobs).  You can visually track status of the different nodes in the workflows on the console making it easier to monitor progress and troubleshoot issues. Also, you can share parameters across entities in the workflow.
* `service/health`: Updates service API and documentation
  * API improvements for the AWS Health service.
* `service/iotevents-data`: Updates service API and documentation
* `service/opsworks`: Updates service documentation
  * Documentation updates for OpsWorks Stacks.
* `service/rds`: Updates service API and documentation
  * This release adds support for RDS storage autoscaling

Release v1.20.4 (2019-06-19)
===

### Service Client Updates
* `service/eks`: Updates service documentation

Release v1.20.3 (2019-06-18)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * You can now launch new 12xlarge, 24xlarge, and metal instance sizes on the Amazon EC2 compute optimized C5 instance types featuring 2nd Gen Intel Xeon Scalable Processors.
* `service/resourcegroupstaggingapi`: Updates service API, documentation, and paginators
  * You can use tag policies to help standardize on tags across your organization's resources.

Release v1.20.2 (2019-06-17)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/neptune`: Updates service API and documentation
  * This release adds a feature to configure Amazon Neptune to publish audit logs to Amazon CloudWatch Logs.
* `service/robomaker`: Updates service API and documentation
* `service/servicecatalog`: Updates service API
  * Restrict concurrent calls by a single customer account for CreatePortfolioShare and DeletePortfolioShare when sharing/unsharing to an Organization.

Release v1.20.1 (2019-06-14)
===

### Service Client Updates
* `service/appstream`: Updates service API
  * Added 2 new values(WINDOWS_SERVER_2016, WINDOWS_SERVER_2019) for PlatformType enum.
* `service/cloudfront`: Adds new service
  * A new datatype in the CloudFront API, AliasICPRecordal, provides the ICP recordal status for CNAMEs associated with distributions. AWS services in China customers must file for an Internet Content Provider (ICP) recordal if they want to serve content publicly on an alternate domain name, also known as a CNAME, that they have added to CloudFront. The status value is returned in the CloudFront response; you cannot configure it yourself. The status is set to APPROVED for all CNAMEs (aliases) in regions outside of China.
* `service/ec2`: Updates service API
  * Correction to enumerations in EC2 client.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/personalize`: Updates service documentation

Release v1.20.0 (2019-06-13)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `service/ec2`: Updates service API
  * G4 instances are Amazon EC2 instances based on NVIDIA T4 GPUs and are designed to provide cost-effective machine learning inference for applications, like image classification, object detection, recommender systems, automated speech recognition, and language translation. G4 instances are also a cost-effective platform for building and running graphics-intensive applications, such as remote graphics workstations, video transcoding, photo-realistic design, and game streaming in the cloud. To get started with G4 instances visit https://aws.amazon.com/ec2/instance-types/g4.
* `service/elasticache`: Updates service API and documentation
  * This release is to add support for reader endpoint for cluster-mode disabled Amazon ElastiCache for Redis clusters.
* `service/guardduty`: Updates service API, documentation, and paginators
  * Support for tagging functionality in Create and Get operations for Detector, IP Set, Threat Intel Set, and Finding Filter resources and 3 new tagging APIs: ListTagsForResource, TagResource, and UntagResource.

### SDK Features

* `aws/session`: Add support for chaining assume IAM role from shared config ([#2579](https://github.com/aws/aws-sdk-go/pull/2579))
  * Adds support chaining assume role credentials from the shared config/credentials files. This change allows you to create an assume role chain of multiple levels of assumed IAM roles. The config profile the deepest in the chain must use static credentials, or `credential_source`. If the deepest profile doesn't have either of these the session will fail to load.
  * Fixes the SDK's shared config credential source not assuming a role with environment and ECS credentials. EC2 credentials were already supported.
  * Fix [#2528](https://github.com/aws/aws-sdk-go/issue/2528)
  * Fix [#2385](https://github.com/aws/aws-sdk-go/issue/2385)

### SDK Enhancements
* `service/s3/s3manager/s3manageriface`: Add missing methods ([#2612](https://github.com/aws/aws-sdk-go/pull/2612))
  * Adds the missing interface and methods from the `s3manager` Uploader, Downloader, and Batch Delete utilities.

Release v1.19.49 (2019-06-12)
===

### Service Client Updates
* `service/servicecatalog`: Updates service API and documentation
  * This release adds a new field named Guidance to update provisioning artifact, this field can be set by the administrator to provide guidance to end users about which provisioning artifacts to use.

Release v1.19.48 (2019-06-11)
===

### Service Client Updates
* `service/sagemaker`: Updates service API and documentation
  * The default TaskTimeLimitInSeconds of labeling job is increased to 8 hours. Batch Transform introduces a new DataProcessing field which supports input and output filtering and data joining. Training job increases the max allowed input channels from 8 to 20.

Release v1.19.47 (2019-06-10)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * AWS CodeBuild adds support for source version on project level.
* `service/codecommit`: Updates service API, documentation, and paginators
  * This release adds two merge strategies for merging pull requests: squash and three-way. It also adds functionality for resolving merge conflicts, testing merge outcomes, and for merging branches using one of the three supported merge strategies.
* `service/personalize`: Adds new service
* `service/personalize-events`: Adds new service
* `service/personalize-runtime`: Adds new service

Release v1.19.46 (2019-06-07)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Adds DNS entries and NLB ARNs to describe-vpc-endpoint-connections API response. Adds owner ID to describe-vpc-endpoints and create-vpc-endpoint API responses.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.19.45 (2019-06-06)
===

### Service Client Updates
* `service/dynamodb`: Updates service documentation
  * Documentation updates for dynamodb
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for launching container instances using supported Amazon EC2 instance types that have increased elastic network interface density. Using these instance types and opting in to the awsvpcTrunking account setting provides increased elastic network interface (ENI) density on newly launched container instances which allows you to place more tasks on each container instance.
* `service/email`: Updates service API and documentation
  * You can now specify whether the Amazon Simple Email Service must deliver email over a connection that is encrypted using Transport Layer Security (TLS).
* `service/guardduty`: Updates service API, documentation, paginators, and examples
  * Improve FindingCriteria Condition field names, support long-typed conditions and deprecate old Condition field names.
* `service/logs`: Updates service documentation
  * Documentation updates for logs
* `service/mediaconnect`: Updates service API and documentation
* `service/organizations`: Updates service API, documentation, and paginators
  * You can tag and untag accounts in your organization and view tags on an account in your organization.
* `service/ssm`: Updates service API and documentation
  * OpsCenter is a new Systems Manager capability that allows you to view, diagnose, and remediate, operational issues, aka OpsItems, related to various AWS resources by bringing together contextually relevant investigation information. New APIs to create, update, describe, and get OpsItems as well as OpsItems summary API.

Release v1.19.44 (2019-06-05)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * Support specifying python version for Python shell jobs. A new parameter PythonVersion is added to the JobCommand data type.

Release v1.19.43 (2019-06-04)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds support for Host Recovery feature which automatically restarts instances on to a new replacement host if failures are detected on Dedicated Host.
* `service/elasticache`: Updates service API, documentation, and paginators
  * Amazon ElastiCache now allows you to apply available service updates on demand. Features included: (1) Access to the list of applicable service updates and their priorities. (2) Service update monitoring and regular status updates. (3) Recommended apply-by-dates for scheduling the service updates, which is critical if your cluster is in ElastiCache-supported compliance programs. (4) Ability to stop and later re-apply updates. For more information, see https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/Self-Service-Updates.html
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iam`: Updates service API, documentation, and examples
  * This release adds validation for policy path field. This field is now restricted to be max 512 characters.
* `service/s3`: Updates service documentation
  * Documentation updates for s3
* `service/storagegateway`: Updates service API and documentation
  * AWS Storage Gateway now supports AWS PrivateLink, enabling you to administer and use gateways without needing to use public IP addresses or a NAT/Internet Gateway, while avoiding traffic from going over the internet.

Release v1.19.42 (2019-06-03)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Amazon EC2 I3en instances are the new storage-optimized instances offering up to 60 TB NVMe SSD instance storage and up to 100 Gbps of network bandwidth.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service documentation
  * Amazon RDS Data API is generally available. Removing beta notes in the documentation.

Release v1.19.41 (2019-05-30)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * This release adds APIs that allow adding and removing tags to a repository, and viewing tags for a repository. It also enables adding tags when creating a repository.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iotanalytics`: Updates service API and documentation
* `service/iotevents`: Adds new service
* `service/iotevents-data`: Adds new service
* `service/kafka`: Updates service API, documentation, and paginators
* `service/pinpoint-email`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * This release adds support for Activity Streams for database clusters.
* `service/rds-data`: Updates service API, documentation, and examples
* `service/servicecatalog`: Updates service API and documentation
  * Service Catalog ListStackInstancesForProvisionedProduct API enables customers to get details of a provisioned product with type "CFN_STACKSET". By passing the provisioned product id, the API will list account, region and status of each stack instances that are associated with this provisioned product.

Release v1.19.40 (2019-05-29)
===

### Service Client Updates
* `service/dlm`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Customers can now simultaneously take snapshots of multiple EBS volumes attached to an EC2 instance. With this new capability, snapshots guarantee crash-consistency across multiple volumes by preserving the order of IO operations. This new feature is fully integrated with Amazon Data Lifecycle Manager (DLM) allowing customers to automatically manage snapshots by creating lifecycle policies.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iotthingsgraph`: Adds new service
* `service/rds`: Updates service documentation
  * Documentation updates for rds
* `service/securityhub`: Updates service API, documentation, and paginators
* `service/ssm`: Updates service documentation
  * Systems Manager - Documentation updates

### SDK Enhancements
* `service/mediastoredata`: Add support for nonseekable io.Reader ([#2622](https://github.com/aws/aws-sdk-go/pull/2622))
  * Updates the SDK's documentation to clarify how you can use the SDK's `aws.ReadSeekCloser` utility function to wrap an io.Reader to be used with an API operation that allows streaming unsigned payload in the operation's request.
  * Adds example using ReadSeekCloser with AWS Elemental MediaStore Data's PutObject API operation.
* Update CI validation testing for Go module files ([#2626](https://github.com/aws/aws-sdk-go/pull/2626))
  * Suppress changes to the Go module definition files during CI code generation validation testing.

### SDK Bugs
* `service/pinpointemail`: Fix client unable to make API requests ([#2625](https://github.com/aws/aws-sdk-go/pull/2625))
  * Fixes the API client's code generation to ignore the `targetPrefix` modeled value. This value is not valid for the REST-JSON protocol.
  * Updates the SDK's code generation to ignore the `targetPrefix` for all protocols other than RPCJSON.

Release v1.19.39 (2019-05-28)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * This release adds the ability to search and order toll free phone numbers for Voice Connectors.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/groundstation`: Adds new service
* `service/pinpoint-email`: Updates service API, documentation, and paginators
* `service/rds`: Updates service API and documentation
  * Add a new output field Status to DBEngineVersion which shows the status of the engine version (either available or deprecated). Add a new parameter IncludeAll to DescribeDBEngineVersions to make it possible to return both available and deprecated engine versions. These changes enable a user to create a Read Replica of an DB instance on a deprecated engine version.
* `service/robomaker`: Updates service API and documentation
* `service/storagegateway`: Updates service API and documentation
  * Introduce AssignTapePool operation to allow customers to migrate tapes between pools.
* `service/sts`: Updates service documentation
* `service/transcribe`: Updates service API
* `service/waf`: Updates service documentation
  * Documentation updates for waf

Release v1.19.38 (2019-05-24)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * AWS CodeDeploy now supports tagging for the application and deployment group resources.
* `service/mediastore-data`: Updates service API, documentation, and paginators
* `service/opsworkscm`: Updates service documentation
  * Documentation updates for OpsWorks for Chef Automate; attribute values updated for Chef Automate 2.0 release.

### SDK Bugs
* `service/dynamodb/expression`: Fix Builder with KeyCondition example ([#2618](https://github.com/aws/aws-sdk-go/pull/2618))
  * Fixes the ExampleBuilder_WithKeyCondition example to include the ExpressionAttributeNames member being set.
  * Related to [aws/aws-sdk-go-v2#285](https://github.com/aws/aws-sdk-go-v2/issues/285)
* `private/model/api`: Improve SDK API reference doc generation ([#2617](https://github.com/aws/aws-sdk-go/pull/2617))
  * Improves the SDK's generated documentation for API client, operation, and types. This fixes several bugs in the doc generation causing poor formatting, an difficult to read reference documentation.
  * Fixes [#2572](https://github.com/aws/aws-sdk-go/pull/2572)
  * Fixes [#2374](https://github.com/aws/aws-sdk-go/pull/2374)

Release v1.19.37 (2019-05-23)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * New APIs to enable EBS encryption by default feature. Once EBS encryption by default is enabled in a region within the account, all new EBS volumes and snapshot copies are always encrypted
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/waf-regional`: Updates service documentation

Release v1.19.36 (2019-05-22)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * This release adds support for tagging of Amazon API Gateway resources.
* `service/budgets`: Updates service API and documentation
  * Added new datatype PlannedBudgetLimits to Budget model, and updated examples for AWS Budgets API for UpdateBudget, CreateBudget, DescribeBudget, and DescribeBudgets
* `service/devicefarm`: Updates service API and documentation
  * This release introduces support for tagging, tag-based access control, and resource-based access control.
* `service/ec2`: Updates service API and documentation
  * This release adds idempotency support for associate, create route and authorization APIs for AWS Client VPN Endpoints.
* `service/elasticfilesystem`: Updates service API and documentation
  * AWS EFS documentation updated to reflect the minimum required value for ProvisionedThroughputInMibps is 1 from the previously documented 0. The service has always required a minimum value of 1, therefor service behavior is not changed.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service documentation
  * Documentation updates for rds
* `service/servicecatalog`: Updates service API and documentation
  * Service Catalog UpdateProvisionedProductProperties API enables customers to manage provisioned product ownership. Administrators can now update the user associated to a provisioned product to another user within the same account allowing the new user to describe, update, terminate and execute service actions in that Service Catalog resource. New owner will also be able to list and describe all past records executed for that provisioned product.
* `service/worklink`: Updates service API, documentation, and paginators

Release v1.19.35 (2019-05-21)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/datasync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.19.34 (2019-05-20)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kafka`: Updates service API, documentation, and paginators
* `service/mediapackage-vod`: Adds new service
* `service/meteringmarketplace`: Updates service documentation
  * Documentation updates for meteringmarketplace

### SDK Enhancements
* Add raw error message bytes to SerializationError errors ([#2600](https://github.com/aws/aws-sdk-go/pull/2600))
  * Updates the SDK's API error message SerializationError handling to capture the original error message byte, and include it in the SerializationError error value.
  * Fixes [#2562](https://github.com/aws/aws-sdk-go/issues/2562), [#2411](https://github.com/aws/aws-sdk-go/issues/2411), [#2315](https://github.com/aws/aws-sdk-go/issues/2315)

### SDK Bugs
* `service/s3/s3manager`: Fix uploader to check for empty part before max parts check (#2556)
  * Fixes the S3 Upload manager's behavior for uploading exactly MaxUploadParts * PartSize to S3. The uploader would previously return an error after the full content was uploaded, because the assert on max upload parts was occurring before the check if there were any more parts to upload.
  * Fixes [#2557](https://github.com/aws/aws-sdk-go/issues/2557)

Release v1.19.33 (2019-05-17)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Includes APIs for managing subscriptions to AppStream 2.0 usage reports and configuring idle disconnect timeouts on AppStream 2.0 fleets.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.19.32 (2019-05-16)
===

### Service Client Updates
* `service/medialive`: Updates service waiters and paginators
  * Added channel state waiters to MediaLive.
* `service/s3`: Updates service API, documentation, and examples
  * This release updates the Amazon S3 PUT Bucket replication API to include a new optional field named token, which allows you to add a replication configuration to an S3 bucket that has Object Lock enabled.

Release v1.19.31 (2019-05-15)
===

### Service Client Updates
* `service/codepipeline`: Updates service API, documentation, and paginators
  * This feature includes new APIs to add, edit, remove and view tags for pipeline, custom action type and webhook resources. You can also add tags while creating these resources.
* `service/ec2`: Updates service API and documentation
  * Adding tagging support for VPC Endpoints and VPC Endpoint Services.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediapackage`: Updates service API and documentation
  * Adds optional configuration for DASH SegmentTemplateFormat to refer to segments by Number with Duration, rather than Number or Time with SegmentTimeline.
* `service/rds`: Updates service documentation
  * In the RDS API and CLI documentation, corrections to the descriptions for Boolean parameters to avoid references to TRUE and FALSE. The RDS CLI does not allow TRUE and FALSE values values for Boolean parameters.
* `service/transcribe`: Updates service API

Release v1.19.30 (2019-05-14)
===

### Service Client Updates
* `service/chime`: Updates service API and documentation
  * Amazon Chime private bots GA release.
* `service/comprehend`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API, documentation, and paginators
  * Pagination support for ec2.DescribeSubnets, ec2.DescribeDhcpOptions
* `service/storagegateway`: Updates service API and documentation
  * Add Tags parameter to CreateSnapshot and UpdateSnapshotSchedule APIs, used for creating tags on create for one off snapshots and scheduled snapshots.

Release v1.19.29 (2019-05-13)
===

### Service Client Updates
* `service/datasync`: Updates service API and documentation
* `service/iotanalytics`: Updates service API and documentation
* `service/lambda`: Updates service API and waiters
  * AWS Lambda now supports Node.js v10

Release v1.19.28 (2019-05-10)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * AWS Glue now supports specifying existing catalog tables for a crawler to examine as a data source. A new parameter CatalogTargets is added to the CrawlerTargets data type.
* `service/sts`: Updates service API, documentation, and examples
  * AWS Security Token Service (STS) now supports passing IAM Managed Policy ARNs as session policies when you programmatically create temporary sessions for a role or federated user. The Managed Policy ARNs can be passed via the PolicyArns parameter, which is now available in the AssumeRole, AssumeRoleWithWebIdentity, AssumeRoleWithSAML, and GetFederationToken APIs. The session policies referenced by the PolicyArn parameter will only further restrict the existing permissions of an IAM User or Role for individual sessions.

Release v1.19.27 (2019-05-08)
===

### Service Client Updates
* `service/eks`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot1click-projects`: Updates service paginators
* `service/kinesisanalytics`: Updates service API and documentation
  * Kinesis Data Analytics APIs now support tagging on applications.
* `service/kinesisanalyticsv2`: Updates service API and documentation
* `service/sagemaker`: Updates service API and documentation
  * Workteams now supports notification configurations. Neo now supports Jetson Nano as a target device and NumberOfHumanWorkersPerDataObject is now included in the ListLabelingJobsForWorkteam response.
* `service/servicecatalog`: Updates service API and documentation
  * Adds "Parameters" field in UpdateConstraint API, which will allow Admin user to update "Parameters" in created Constraints.

Release v1.19.26 (2019-05-07)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/appsync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/ssm`: Updates service API and documentation
  * Patch Manager adds support for Microsoft Application Patching.
* `service/storagegateway`: Updates service API and documentation
  * Add optional field AdminUserList to CreateSMBFileShare and UpdateSMBFileShare APIs.

Release v1.19.25 (2019-05-06)
===

### Service Client Updates
* `service/codepipeline`: Updates service documentation
  * Documentation updates for codepipeline
* `service/config`: Updates service API and documentation
* `service/iam`: Updates service documentation
  * Documentation updates for iam
* `service/sts`: Updates service documentation
  * Documentation updates for sts

Release v1.19.24 (2019-05-03)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * DASH output groups using DRM encryption can now enable a playback device compatibility mode to correct problems with playback on older devices.
* `service/medialive`: Updates service API and documentation
  * You can now switch the channel mode of your channels from standard to single pipeline and from single pipeline to standard. In order to switch a channel from single pipeline to standard all inputs attached to the channel must support two encoder pipelines.
* `service/workmail`: Updates service API, documentation, and paginators
  * Amazon WorkMail is releasing two new actions: 'GetMailboxDetails' and 'UpdateMailboxQuota'. They add insight into how much space is used by a given mailbox (size) and what its limit is (quota). A mailbox quota can be updated, but lowering the value will not influence WorkMail per user charges. For a closer look at the actions please visit https://docs.aws.amazon.com/workmail/latest/APIReference/API_Operations.html

Release v1.19.23 (2019-05-02)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kms`: Updates service API and documentation
  * AWS Key Management Service (KMS) can return an INTERNAL_ERROR connection error code if it cannot connect a custom key store to its AWS CloudHSM cluster. INTERNAL_ERROR is one of several connection error codes that help you to diagnose and fix a problem with your custom key store.

Release v1.19.22 (2019-05-01)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds an API for the modification of a VPN Connection, enabling migration from a Virtual Private Gateway (VGW) to a Transit Gateway (TGW), while preserving the VPN endpoint IP addresses on the AWS side as well as the tunnel options.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces additional task definition parameters that enable you to define secret options for Docker log configuration, a per-container list contains secrets stored in AWS Systems Manager Parameter Store or AWS Secrets Manager.
* `service/xray`: Updates service API, documentation, and paginators
  * AWS X-Ray now includes Analytics, an interactive approach to analyzing user request paths (i.e., traces). Analytics will allow you to easily understand how your application and its underlying services are performing. With X-Ray Analytics, you can quickly detect application issues, pinpoint the root cause of the issue, determine the severity of the issues, and identify which end users were impacted. With AWS X-Ray Analytics you can explore, analyze, and visualize traces, allowing you to find increases in response time to user requests or increases in error rates. Metadata around peak periods, including frequency and actual times of occurrence, can be investigated by applying filters with a few clicks. You can then drill down on specific errors, faults, and response time root causes and view the associated traces.

Release v1.19.21 (2019-04-30)
===

### Service Client Updates
* `service/codepipeline`: Updates service API and documentation
  * This release contains an update to the PipelineContext object that includes the Pipeline ARN, and the Pipeline Execution Id. The ActionContext object is also updated to include the Action Execution Id.
* `service/directconnect`: Updates service API and documentation
  * This release adds support for AWS Direct Connect customers to use AWS Transit Gateway with AWS Direct Connect gateway to route traffic between on-premise networks and their VPCs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/managedblockchain`: Adds new service
* `service/neptune`: Updates service API, documentation, and examples
  * Adds a feature to allow customers to specify a custom parameter group when restoring a database cluster.
* `service/s3control`: Updates service API, documentation, and paginators
  * Add support for Amazon S3 Batch Operations.
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * Admin users can now associate/disassociate aws budgets with a portfolio or product in Service Catalog. End users can see the association by listing it or as part of the describe portfolio/product output. A new optional boolean parameter, "DisableTemplateValidation", is added to ProvisioningArtifactProperties data type. The purpose of the parameter is to enable or disable the CloudFormation template validtion when creating a product or a provisioning artifact.

Release v1.19.20 (2019-04-29)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Adds support for Elastic Fabric Adapter (EFA) ENIs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/transfer`: Updates service API, documentation, and paginators
  * This release adds support for per-server host-key management. You can now specify the SSH RSA private key used by your SFTP server.

Release v1.19.19 (2019-04-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iam`: Updates service API, documentation, waiters, and examples
  * AWS Security Token Service (STS) enables you to request session tokens from the global STS endpoint that work in all AWS Regions. You can configure the global STS endpoint to vend session tokens that are compatible with all AWS Regions using the new IAM SetSecurityTokenServicePreferences API.
* `service/sns`: Updates service API and documentation
  * With this release AWS SNS adds tagging support for Topics.

Release v1.19.18 (2019-04-25)
===

### Service Client Updates
* `service/batch`: Updates service documentation
  * Documentation updates for AWS Batch.
* `service/dynamodb`: Updates service API and documentation
  * This update allows you to tag Amazon DynamoDB tables when you create them. Tags are labels you can attach to AWS resources to make them easier to manage, search, and filter.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/gamelift`: Updates service API and documentation
  * This release introduces the new Realtime Servers feature, giving game developers a lightweight yet flexible solution that eliminates the need to build a fully custom game server. The AWS SDK updates provide support for scripts, which are used to configure and customize Realtime Servers.
* `service/inspector`: Updates service API and documentation
  * AWS Inspector - Improve the ListFindings API response time and decreases the maximum number of agentIDs from 500 to 99.
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports the GetLayerVersionByArn API.
* `service/workspaces`: Updates service documentation
  * Documentation updates for workspaces

Release v1.19.17 (2019-04-24)
===

### Bug Fixes
* `aws/endpoints`: Fix incorrect AWS Organizations global endpoint
  * Fixes the endpoint metadata for the AWS Organization in [Release v1.19.16](https://github.com/aws/aws-sdk-go/releases/tag/v1.19.16)

Release v1.19.16 (2019-04-24)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/cloudformation`: Updates service documentation
  * Documentation updates for cloudformation
* `service/ec2`: Updates service API
  * You can now launch the new Amazon EC2 general purpose burstable instance types T3a that feature AMD EPYC processors.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconnect`: Updates service API, documentation, and paginators
* `service/mediatailor`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * A new parameter "feature-name" is added to the add-role and remove-role db cluster APIs. The value for the parameter is optional for Aurora MySQL compatible database clusters, but mandatory for Aurora PostgresQL. You can find the valid list of values using describe db engine versions API.
* `service/route53`: Updates service API and documentation
  * Amazon Route 53 now supports the Asia Pacific (Hong Kong) Region (ap-east-1) for latency records, geoproximity records, and private DNS for Amazon VPCs in that region.
* `service/ssm`: Updates service API and documentation
  * This release updates AWS Systems Manager APIs to allow customers to configure parameters to use either the standard-parameter tier (the default tier) or the advanced-parameter tier. It allows customers to create parameters with larger values and attach parameter policies to an Advanced Parameter.
* `service/storagegateway`: Updates service API, documentation, and paginators
  * AWS Storage Gateway now supports Access Control Lists (ACLs) on File Gateway SMB shares, enabling you to apply fine grained access controls for Active Directory users and groups.
* `service/textract`: Updates service API and documentation

Release v1.19.15 (2019-04-19)
===

### Service Client Updates
* `service/resource-groups`: Updates service API and documentation
* `service/transcribe`: Updates service API
* `service/workspaces`: Updates service API and documentation
  * Added a new reserved field.

Release v1.19.14 (2019-04-18)
===

### Service Client Updates
* `service/cognito-idp`: Updates service documentation
* `service/discovery`: Updates service API
  * The Application Discovery Service's DescribeImportTasks and BatchDeleteImportData APIs now return additional statuses for error reporting.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kafka`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * AWS Organizations is now available in the AWS GovCloud (US) Regions, and we added a new API action for creating accounts in those Regions. For more information, see CreateGovCloudAccount in the AWS Organizations API Reference.
* `service/rds`: Updates service API and documentation
  * This release adds the TimeoutAction parameter to the ScalingConfiguration of an Aurora Serverless DB cluster. You can now configure the behavior when an auto-scaling capacity change can't find a scaling point.
* `service/worklink`: Updates service API, documentation, and paginators
* `service/workspaces`: Updates service documentation
  * Documentation updates for workspaces

Release v1.19.13 (2019-04-17)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds support for requester-managed Interface VPC Endpoints (powered by AWS PrivateLink). The feature prevents VPC endpoint owners from accidentally deleting or otherwise mismanaging the VPC endpoints of some AWS VPC endpoint services.
* `service/polly`: Updates service API
  * Amazon Polly adds Arabic language support with new female voice - "Zeina"

Release v1.19.12 (2019-04-16)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API, documentation, and paginators
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/monitoring`: Updates service documentation
  * Documentation updates for monitoring
* `service/mq`: Updates service API and documentation
  * This release adds the ability to retrieve information about broker engines and broker instance options. See Broker Engine Types and Broker Instance Options in the Amazon MQ REST API Reference.
* `service/organizations`: Updates service documentation
  * Documentation updates for organizations
* `service/redshift`: Updates service API and documentation
  * DescribeResize can now return percent of data transferred from source cluster to target cluster for a classic resize.
* `service/storagegateway`: Updates service API and documentation
  * This change allows you to select either a weekly or monthly maintenance window for your volume or tape gateway. It also allows you to tag your tape and volume resources on creation by adding a Tag value on calls to the respective api endpoints.

### SDK Enhancements
* `example/service/dynamodb`: Add custom unmarshaller error example for TransactWriteItems ([#2548](https://github.com/aws/aws-sdk-go/pull/2548))
  * Adds an example for building and using a custom unmarshaller to unmarshal TransactionCancelledExceptions from the error response of TransactWriteItems operation.

Release v1.19.11 (2019-04-05)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * AWS Glue now supports workerType choices in the CreateJob, UpdateJob, and StartJobRun APIs, to be used for memory-intensive jobs.
* `service/iot1click-devices`: Updates service API and documentation
* `service/mediaconvert`: Updates service API
  * Rectify incorrect modelling of DisassociateCertificate method
* `service/medialive`: Updates service API, documentation, and paginators
  * Today AWS Elemental MediaLive (https://aws.amazon.com/medialive/) adds the option to create "Single Pipeline" channels, which offers a lower-cost option compared to Standard channels. MediaLive Single Pipeline channels have a single encoding pipeline rather than the redundant dual Availability Zone (AZ) pipelines that MediaLive provides with a "Standard" channel.

Release v1.19.10 (2019-04-04)
===

### Service Client Updates
* `service/eks`: Updates service API and documentation
* `service/iam`: Updates service documentation
  * Documentation updates for iam

Release v1.19.9 (2019-04-03)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Support for GPU resource requirement in RegisterJobDefinition and SubmitJob
* `service/comprehend`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.19.8 (2019-04-02)
===

### Service Client Updates
* `service/acm`: Updates service documentation
  * Documentation updates for acm
* `service/ec2`: Updates service paginators
  * Add paginators.
* `service/securityhub`: Updates service API and documentation

Release v1.19.7 (2019-04-01)
===

### Service Client Updates
* `service/elasticmapreduce`: Updates service API, documentation, and paginators
  * Amazon EMR adds the ability to modify instance group configurations on a running cluster through the new "configurations" field in the ModifyInstanceGroups API.
* `service/ssm`: Updates service documentation
  * March 2019 documentation updates for Systems Manager.

Release v1.19.6 (2019-03-29)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/greengrass`: Updates service API and documentation
  * Greengrass APIs now support tagging operations on resources
* `service/monitoring`: Updates service API and documentation
  * Added 3 new APIs, and one additional parameter to PutMetricAlarm API, to support tagging of CloudWatch Alarms.

Release v1.19.5 (2019-03-28)
===

### Service Client Updates
* `service/medialive`: Updates service API and documentation
  * This release adds a new output locking mode synchronized to the Unix epoch.
* `service/pinpoint-email`: Updates service API and documentation
* `service/servicecatalog`: Updates service API and documentation
  * Adds "Tags" field in UpdateProvisionedProduct API. The product should have a new RESOURCE_UPDATE Constraint with TagUpdateOnProvisionedProduct field set to ALLOWED for it to work. See API docs for CreateConstraint for more information
* `service/workspaces`: Updates service API and documentation
  * Amazon WorkSpaces adds tagging support for WorkSpaces Images, WorkSpaces directories, WorkSpaces bundles and IP Access control groups.

Release v1.19.4 (2019-03-27)
===

### Service Client Updates
* `service/directconnect`: Updates service API and documentation
  * Direct Connect gateway enables you to establish connectivity between your on-premise networks and Amazon Virtual Private Clouds (VPCs) in any commercial AWS Region (except in China) using AWS Direct Connect connections at any AWS Direct Connect location. This release enables multi-account support for Direct Connect gateway, with multi-account support for Direct Connect gateway, you can associate up to ten VPCs from any AWS account with a Direct Connect gateway. The AWS accounts owning VPCs and the Direct Connect gateway must belong to the same AWS payer account ID. This release also enables Direct Connect Gateway owners to allocate allowed prefixes from each associated VPCs.
* `service/fms`: Updates service API, documentation, and paginators
* `service/iotanalytics`: Updates service API and documentation
* `service/mediaconvert`: Updates service API and documentation
  * This release adds support for detailed job progress status and S3 server-side output encryption. In addition, the anti-alias filter will now be automatically applied to all outputs
* `service/robomaker`: Updates service API, documentation, and paginators
* `service/transcribe`: Updates service API and documentation

Release v1.19.3 (2019-03-27)
===

### Service Client Updates
* `service/appmesh`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API
  * You can now launch the new Amazon EC2 R5ad and M5ad instances that feature local NVMe attached SSD instance storage (up to 3600 GB). M5ad and R5ad feature AMD EPYC processors that offer a 10% cost savings over the M5d and R5d EC2 instances.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for external deployment controllers for ECS services with the launch of task set management APIs. Task sets are a new primitive for controlled management of application deployments within a single ECS service.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/s3`: Updates service API, documentation, and examples
  * S3 Glacier Deep Archive provides secure, durable object storage class for long term data archival. This SDK release provides API support for this new storage class.
* `service/storagegateway`: Updates service API and documentation
  * This change allows you to select a pool for archiving virtual tapes. Pools are associated with S3 storage classes. You can now choose to archive virtual tapes in either S3 Glacier or S3 Glacier Deep Archive storage class. CreateTapes API now takes a new PoolId parameter which can either be GLACIER or DEEP_ARCHIVE. Tapes created with this parameter will be archived in the corresponding storage class.
* `service/transfer`: Updates service API and documentation
  * This release adds PrivateLink support to your AWS SFTP server endpoint, enabling the customer to access their SFTP server within a VPC, without having to traverse the internet. Customers can now can create a server and specify an option whether they want the endpoint to be hosted as public or in their VPC, and with the in VPC option, SFTP clients and users can access the server only from the customer's VPC or from their on-premises environments using DX or VPN. This release also relaxes the SFTP user name requirements to allow underscores and hyphens.

Release v1.19.2 (2019-03-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * This new feature will now allow customers to add a customized csv classifier with classifier API. They can specify a custom delimiter, quote symbol and control other behavior they'd like crawlers to have while recognizing csv files
* `service/workmail`: Updates service API and documentation
  * Documentation updates for Amazon WorkMail.

Release v1.19.1 (2019-03-22)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot1click-projects`: Updates service API and documentation
* `service/transcribe`: Updates service API and documentation

Release v1.19.0 (2019-03-21)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation
  * Documentation updates for Amazon EC2 Auto Scaling
* `service/cognito-idp`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Added 3 new APIs, and one additional parameter to the PutRule API, to support tagging of CloudWatch Events rules.
* `service/iot`: Updates service API and documentation
  * This release adds the GetStatistics API for the AWS IoT Fleet Indexing Service, which allows customers to query for statistics about registered devices that match a search query. This release only supports the count statistics. For more information about this API, see https://docs.aws.amazon.com/iot/latest/apireference/API_GetStatistics.html
* `service/lightsail`: Updates service API and documentation
  * This release adds the DeleteKnownHostKeys API, which enables Lightsail's browser-based SSH or RDP clients to connect to the instance after a host key mismatch.

### SDK Features
* `aws/credentials/stscreds`: Update StdinTokenProvider to prompt on stder ([#2481](https://github.com/aws/aws-sdk-go/pull/2481))
  * Updates the `stscreds` package default MFA token provider, `StdinTokenProvider`, to prompt on `stderr` instead of `stdout`. This is to make it possible to redirect/pipe output when using `StdinTokenProvider` and still seeing the prompt text.

Release v1.18.6 (2019-03-20)
===

### Service Client Updates
* `service/codepipeline`: Updates service API and documentation
  * Add support for viewing details of each action execution belonging to past and latest pipeline executions that have occurred in customer's pipeline. The details include start/updated times, action execution results, input/output artifacts information, etc. Customers also have the option to add pipelineExecutionId in the input to filter the results down to a single pipeline execution.
* `service/cognito-identity`: Updates service API and documentation
* `service/meteringmarketplace`: Updates service API and documentation
  * This release increases AWS Marketplace Metering Service maximum usage quantity to 2147483647 and makes parameters usage quantity and dryrun optional.

### SDK Bugs
* `private/protocol`: Use correct Content-Type for rest json protocol ([#2497](https://github.com/aws/aws-sdk-go/pull/2497))
  * Updates the SDK to use the correct `application/json` content type for all rest json protocol based AWS services. This fixes the bug where the jsonrpc protocol's `application/x-amz-json-X.Y` content type would be used for services like Pinpoint SMS.

Release v1.18.5 (2019-03-19)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/eks`: Updates service API and documentation

Release v1.18.4 (2019-03-18)
===

### Service Client Updates
* `service/chime`: Updates service API, documentation, and paginators
  * This release adds support for the Amazon Chime Business Calling and Voice Connector features.
* `service/dms`: Updates service API, documentation, and paginators
  * S3 Endpoint Settings added support for 1) Migrating to Amazon S3 as a target in Parquet format 2) Encrypting S3 objects after migration with custom KMS Server-Side encryption. Redshift Endpoint Settings added support for encrypting intermediate S3 objects during migration with custom KMS Server-Side encryption.
* `service/ec2`: Updates service API and documentation
  * DescribeFpgaImages API now returns a new DataRetentionSupport attribute to indicate if the AFI meets the requirements to support DRAM data retention. DataRetentionSupport is a read-only attribute.

Release v1.18.3 (2019-03-14)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * AWS Certificate Manager has added a new API action, RenewCertificate. RenewCertificate causes ACM to force the renewal of any private certificate which has been exported.
* `service/acm-pca`: Updates service API, documentation, and paginators
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * This release adds tagging support for Dedicated Host Reservations.
* `service/iot`: Updates service API and documentation
  * In this release, AWS IoT introduces support for tagging OTA Update and Stream resources. For more information about tagging, see the AWS IoT Developer Guide.
* `service/monitoring`: Updates service API, documentation, and paginators
  * New Messages parameter for the output of GetMetricData, to support new metric search functionality.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker Automatic Model Tuning now supports random search and hyperparameter scaling.

Release v1.18.2 (2019-03-13)
===

### Service Client Updates
* `service/config`: Updates service API, documentation, and paginators
* `service/logs`: Updates service documentation
  * Documentation updates for logs

Release v1.18.1 (2019-03-12)
===

### Service Client Updates
* `service/serverlessrepo`: Updates service API and documentation

Release v1.18.0 (2019-03-11)
===

### Service Client Updates
* `service/ce`: Updates service API
* `service/elasticbeanstalk`: Updates service API and documentation
  * Elastic Beanstalk added support for tagging, and tag-based access control, of all Elastic Beanstalk resources.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * CreateDevEndpoint and UpdateDevEndpoint now support Arguments to configure the DevEndpoint.
* `service/iot`: Updates service documentation
  * Documentation updates for iot
* `service/quicksight`: Updates service API and documentation
  * Amazon QuickSight user and group operation results now include group principal IDs and user principal IDs. This release also adds "DeleteUserByPrincipalId", which deletes users given their principal ID. The update also improves role session name validation.
* `service/rekognition`: Updates service documentation
  * Documentation updates for Amazon Rekognition

### SDK Features
* `service/kinesis`: Enable support for SubscribeToStream API operation ([#2402](https://github.com/aws/aws-sdk-go/pull/2402))
  * Adds support for Kinesis's SubscribeToStream API operation. The API operation response type, `SubscribeToStreamOutput` member, EventStream has a method `Events` which returns a channel to read Kinesis record events from.

Release v1.17.14 (2019-03-08)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * CodeBuild also now supports Git Submodules.  CodeBuild now supports opting out of Encryption for S3 Build Logs.  By default these logs are encrypted.
* `service/s3`: Updates service documentation and examples
  * Documentation updates for s3
* `service/sagemaker`: Updates service API and documentation
  * SageMaker notebook instances now support enabling or disabling root access for notebook users. SageMaker Neo now supports rk3399 and rk3288 as compilation target devices.

Release v1.17.13 (2019-03-07)
===

### Service Client Updates
* `service/appmesh`: Adds new service
* `service/autoscaling`: Updates service documentation
  * Documentation updates for autoscaling
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces additional task definition parameters that enable you to define dependencies for container startup and shutdown, a per-container start and stop timeout value, as well as an AWS App Mesh proxy configuration which eases the integration between Amazon ECS and AWS App Mesh.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/gamelift`: Updates service API and documentation
  * Amazon GameLift-hosted instances can now securely access resources on other AWS services using IAM roles. See more details at https://aws.amazon.com/releasenotes/amazon-gamelift/.
* `service/greengrass`: Updates service API and documentation
  * Greengrass group UID and GID settings can now be configured to use a provided default via FunctionDefaultConfig. If configured, all Lambda processes in your deployed Greengrass group will by default start with the provided UID and/or GID, rather than by default starting with UID "ggc_user" and GID "ggc_group" as they would if not configured. Individual Lambdas can also be configured to override the defaults if desired via each object in the Functions list of your FunctionDefinitionVersion.
* `service/medialive`: Updates service API and documentation
  * This release adds a MediaPackage output group, simplifying configuration of outputs to AWS Elemental MediaPackage.
* `service/rds`: Updates service API and documentation
  * You can configure your Aurora database cluster to automatically copy tags on the cluster to any automated or manual database cluster snapshots that are created from the cluster. This allows you to easily set metadata on your snapshots to match the parent cluster, including access policies. You may enable or disable this functionality while creating a new cluster, or by modifying an existing database cluster.

Release v1.17.12 (2019-03-06)
===

### Service Client Updates
* `service/directconnect`: Updates service API and documentation
  * Exposed a new available port speeds field in the DescribeLocation api call.
* `service/ec2`: Updates service API, documentation, and paginators
  * This release adds pagination support for ec2.DescribeVpcs, ec2.DescribeInternetGateways and ec2.DescribeNetworkAcls APIs
* `service/elasticfilesystem`: Updates service examples
  * Documentation updates for elasticfilesystem adding new examples for EFS Lifecycle Management feature.

Release v1.17.11 (2019-03-05)
===

### Service Client Updates
* `service/codedeploy`: Updates service documentation
  * Documentation updates for codedeploy
* `service/medialive`: Updates service API and documentation
  * This release adds support for pausing and unpausing one or both pipelines at scheduled times.
* `service/storagegateway`: Updates service API and documentation
  * ActivateGateway, CreateNFSFileShare and CreateSMBFileShare APIs support a new parameter: Tags (to be attached to the created resource). Output for DescribeNFSFileShare, DescribeSMBFileShare and DescribeGatewayInformation APIs now also list the Tags associated with the resource. Minimum length of a KMSKey is now 7 characters.
* `service/textract`: Adds new service

Release v1.17.10 (2019-03-04)
===

### Service Client Updates
* `service/mediapackage`: Updates service API and documentation
  * This release adds support for user-defined tagging of MediaPackage resources. Users may now call operations to list, add and remove tags from channels and origin-endpoints. Users can also specify tags to be attached to these resources during their creation. Describe and list operations on these resources will now additionally return any tags associated with them.
* `service/ssm`: Updates service API and documentation
  * This release updates AWS Systems Manager APIs to support service settings for AWS customers.  A service setting is a key-value pair that defines how a user interacts with or uses an AWS service, and is typically created and consumed by the AWS service team. AWS customers can read a service setting via GetServiceSetting API and update the setting via UpdateServiceSetting API or ResetServiceSetting API, which are introduced in this release. For example, if an AWS service charges money to the account based on a feature or service usage, then the AWS service team might create a setting with the default value of "false".   This means the user can't use this feature unless they update the setting to "true" and  intentionally opt in for a paid feature.

Release v1.17.9 (2019-03-01)
===

### Service Client Updates
* `service/autoscaling-plans`: Updates service documentation
* `service/ec2`: Updates service API and documentation
  * This release adds support for modifying instance event start time which allows users to reschedule EC2 events.

### SDK Enhancements
* `example/service/s3`: Add example of S3 download with progress ([#2456](https://github.com/aws/aws-sdk-go/pull/2456))
  * Adds a new example to the S3 service's examples. This example shows how you could use the S3's GetObject API call in conjunction with a custom writer keeping track of progress.
  * Related to [#1868](https://github.com/aws/aws-sdk-go/pull/1868), [#2468](https://github.com/aws/aws-sdk-go/pull/2468)

### SDK Bugs
* `aws/session`: Allow HTTP Proxy with custom CA bundle ([#2343](https://github.com/aws/aws-sdk-go/pull/2343))
  * Ensures Go HTTP Client's  `ProxyFromEnvironment` functionality is still enabled when  custom CA bundles are used with the SDK.
  * Fix [#2287](https://github.com/aws/aws-sdk-go/pull/2287)

Release v1.17.8 (2019-02-28)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/apigatewayv2`: Updates service API and documentation
* `service/application-autoscaling`: Updates service documentation
* `service/ssm`: Updates service API and documentation

Release v1.17.7 (2019-02-28)
===

### Service Client Updates
* `service/waf`: Updates service documentation
  * Documentation updates for waf
* `service/waf-regional`: Updates service documentation

### SDK Bugs
* `aws/request`: Fix RequestUserAgent tests to be stable ([#2462](https://github.com/aws/aws-sdk-go/pull/2462))
  * Fixes the request User-Agent unit tests to be stable across all platforms and environments.
  * Fixes [#2366](https://github.com/aws/aws-sdk-go/issues/2366)
* `aws/ec2metadata`: Fix EC2 Metadata client panic with debug logging ([#2461](https://github.com/aws/aws-sdk-go/pull/2461))
  * Fixes a panic that could occur witihin the EC2 Metadata client when both `AWS_EC2_METADATA_DISABLED` env var is set and log level is LogDebugWithHTTPBody.
* `private/protocol/rest`: Trim space in header key and value ([#2460](https://github.com/aws/aws-sdk-go/pull/2460))
  * Updates the REST protocol marshaler to trip leading and trailing space from header keys and values before setting the HTTP request header. Fixes a bug when using S3 metadata where metadata values with leading spaces would trigger request signature validation errors when the request is received by the service.
  * Fixes [#2448](https://github.com/aws/aws-sdk-go/issues/2448)

Release v1.17.6 (2019-02-26)
===

### Service Client Updates
* `service/cur`: Updates service API, documentation, and examples
  * Adding support for Athena and new report preferences to the Cost and Usage Report API.
* `service/discovery`: Updates service documentation
  * Documentation updates for discovery
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added several features including support for: auto-rotation or user-specified rotation of 0, 90, 180, or 270 degrees; multiple output groups with DRM; ESAM XML documents to specify ad insertion points; Offline Apple HLS FairPlay content protection.
* `service/opsworkscm`: Updates service documentation
  * Documentation updates for opsworkscm
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations
* `service/pinpoint`: Updates service API and documentation
  * This release adds support for the Amazon Resource Groups Tagging API to Amazon Pinpoint, which means that you can now add and manage tags for Amazon Pinpoint projects (apps), campaigns, and segments. A tag is a label that you optionally define and associate with Amazon Pinpoint resource. Tags can help you categorize and manage these types of resources in different ways, such as by purpose, owner, environment, or other criteria. For example, you can use tags to apply policies or automation, or to identify resources that are subject to certain compliance requirements. A project, campaign, or segment can have as many as 50 tags. For more information about using and managing tags in Amazon Pinpoint, see the Amazon Pinpoint Developer Guide at https://docs.aws.amazon.com/pinpoint/latest/developerguide/welcome.html. For more information about the Amazon Resource Group Tagging API, see the Amazon Resource Group Tagging API Reference at https://docs.aws.amazon.com/resourcegroupstagging/latest/APIReference/Welcome.html.
* `service/resource-groups`: Updates service documentation

Release v1.17.5 (2019-02-25)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Added support for passing an empty SpotMaxPrice parameter to remove a value previously set when updating an Amazon EC2 Auto Scaling group.
* `service/ce`: Updates service documentation
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/mediastore`: Updates service API and documentation
  * This release adds support for access logging, which provides detailed records for the requests that are made to objects in a container.

Release v1.17.4 (2019-02-22)
===

### Service Client Updates
* `service/athena`: Updates service API and documentation
  * This release adds tagging support for Workgroups to Amazon Athena. Use these APIs to add, remove, or list tags on Workgroups, and leverage the tags for various authorization and billing scenarios.
* `service/cloud9`: Updates service API and documentation
  * Adding EnvironmentLifecycle to the Environment data type.
* `service/glue`: Updates service API, documentation, and paginators
  * AWS Glue adds support for assigning AWS resource tags to jobs, triggers, development endpoints, and crawlers. Each tag consists of a key and an optional value, both of which you define. With this capacity, customers can use tags in AWS Glue to easily organize and identify your resources, create cost allocation reports, and control access to resources.
* `service/states`: Updates service API and documentation
  * This release adds support for tag-on-create. You can now add tags when you create AWS Step Functions activity and state machine resources. For more information about tagging, see AWS Tagging Strategies.

Release v1.17.3 (2019-02-21)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Add support for CodeBuild local caching feature
* `service/kinesis-video-archived-media`: Updates service API and documentation
* `service/kinesis-video-media`: Updates service documentation
* `service/kinesisvideo`: Updates service documentation
  * Documentation updates for Kinesis Video Streams
* `service/monitoring`: Updates service documentation
  * Documentation updates for monitoring
* `service/organizations`: Updates service documentation
  * Documentation updates for organizations
* `service/transfer`: Updates service API and documentation
  * Bug fix: increased the max length allowed for request parameter NextToken when paginating List operations
* `service/workdocs`: Updates service documentation
  * Documentation updates for workdocs

Release v1.17.2 (2019-02-20)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * This release adds an API for adding / updating / deleting / copying / moving / setting file modes for one or more files directly to an AWS CodeCommit repository without requiring a Git client.
* `service/directconnect`: Updates service API and documentation
  * Documentation updates for AWS Direct Connect
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/medialive`: Updates service API and documentation
  * This release adds support for VPC inputs, allowing you to push content from your Amazon VPC directly to MediaLive.

Release v1.17.1 (2019-02-19)
===

### Service Client Updates
* `service/ds`: Updates service API and documentation
  * This release adds support for tags during directory creation (CreateDirectory, CreateMicrosoftAd, ConnectDirectory).
* `service/elasticfilesystem`: Updates service API, documentation, and examples
  * Amazon EFS now supports adding tags to file system resources as part of the CreateFileSystem API . Using this capability, customers can now more easily enforce tag-based authorization for EFS file system resources.
* `service/iot`: Updates service API and documentation
  * AWS IoT - AWS IoT Device Defender adds support for configuring behaviors in a security profile with statistical thresholds. Device Defender also adds support for configuring multiple data-point evaluations before a violation is either created or cleared.
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager now supports adding tags when creating Activations, Patch Baselines, Documents, Parameters, and Maintenance Windows

Release v1.17.0 (2019-02-18)
===

### Service Client Updates
* `service/athena`: Updates service API, documentation, and paginators
  * This release adds support for Workgroups to Amazon Athena. Use Workgroups to isolate users, teams, applications or workloads in the same account, control costs by setting up query limits and creating Amazon SNS alarms, and publish query-related metrics to Amazon CloudWatch.
* `service/secretsmanager`: Updates service API and documentation
  * This release increases the maximum allowed size of SecretString or SecretBinary from 4KB to 7KB in the CreateSecret, UpdateSecret, PutSecretValue and GetSecretValue APIs.

### SDK Features
* `service/s3/s3manager`: Update S3 Upload Multipart location ([#2453](https://github.com/aws/aws-sdk-go/pull/2453))
  * Updates the Location returned value of S3 Upload's Multipart UploadOutput type to be consistent with single part upload URL. This update also brings the multipart upload Location inline with the S3 object URLs created by the SDK
  * Fix [#1385](https://github.com/aws/aws-sdk-go/issues/1385)

### SDK Enhancements
* `service/s3`: Update BucketRegionError message to include more information ([#2451](https://github.com/aws/aws-sdk-go/pull/2451))
  * Updates the BucketRegionError error message to include information about the endpoint and actual region the bucket is in if known. This error message is created by the SDK, but could produce a confusing error message if the user provided a region that doesn't match the endpoint.
  * Fix [#2426](https://github.com/aws/aws-sdk-go/pull/2451)

Release v1.16.36 (2019-02-15)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/chime`: Updates service documentation
  * Documentation updates for Amazon Chime
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API and documentation
  * In this release, IoT Device Defender introduces support for tagging Scheduled Audit resources.

Release v1.16.35 (2019-02-14)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds tagging and ARN support for AWS Client VPN Endpoints.You can now run bare metal workloads on EC2 M5 and M5d instances. m5.metal and m5d.metal instances are powered by custom Intel Xeon Scalable Processors with a sustained all core frequency of up to 3.1 GHz. m5.metal and m5d.metal offer 96 vCPUs and 384 GiB of memory. With m5d.metal, you also have access to 3.6 TB of NVMe SSD-backed instance storage. m5.metal and m5d.metal instances deliver 25 Gbps of aggregate network bandwidth using Elastic Network Adapter (ENA)-based Enhanced Networking, as well as 14 Gbps of bandwidth to EBS.You can now run bare metal workloads on EC2 z1d instances. z1d.metal instances are powered by custom Intel Xeon Scalable Processors with a sustained all core frequency of up to 4.0 GHz. z1d.metal offers 48 vCPUs, 384 GiB of memory, and 1.8 TB of NVMe SSD-backed instance storage. z1d.metal instances deliver 25 Gbps of aggregate network bandwidth using Elastic Network Adapter (ENA)-based Enhanced Networking, as well as 14 Gbps of bandwidth to EBS.
* `service/kinesisvideo`: Updates service API and documentation
  * Adds support for Tag-On-Create for Kinesis Video Streams. A list of tags associated with the stream can be created at the same time as the stream creation.

Release v1.16.34 (2019-02-13)
===

### Service Client Updates
* `service/elasticfilesystem`: Updates service API and documentation
  * Customers can now use the EFS Infrequent Access (IA) storage class to more cost-effectively store larger amounts of data in their file systems. EFS IA is cost-optimized storage for files that are not accessed every day. You can create a new file system and enable Lifecycle Management to automatically move files that have not been accessed for 30 days from the Standard storage class to the IA storage class.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediatailor`: Updates service API and documentation
* `service/rekognition`: Updates service API and documentation
  * GetContentModeration now returns the version of the moderation detection model used to detect unsafe content.

Release v1.16.33 (2019-02-12)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lambda`: Updates service documentation
  * Documentation updates for AWS Lambda

Release v1.16.32 (2019-02-11)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This update enables customers to find the start time, max expiration time, and connection status associated with AppStream streaming session.
* `service/codebuild`: Updates service API and documentation
  * Add customized webhook filter support
* `service/mediapackage`: Updates service API and documentation
  * Adds optional configuration for DASH to compact the manifest by combining duplicate SegmentTemplate tags. Adds optional configuration for DASH SegmentTemplate format to refer to segments by "Number" (default) or by "Time".

Release v1.16.31 (2019-02-08)
===

### Service Client Updates
* `service/discovery`: Updates service documentation
  * Documentation updates for the AWS Application Discovery Service.
* `service/dlm`: Updates service API and documentation
* `service/ecs`: Updates service API, documentation, and examples
  * Amazon ECS introduces the PutAccountSettingDefault API, an API that allows a user to set the default ARN/ID format opt-in status for all the roles and users in the account. Previously, setting the account's default opt-in status required the use of the root user with the PutAccountSetting API.

Release v1.16.30 (2019-02-07)
===

### Service Client Updates
* `service/es`: Updates service API and documentation
  * Feature: Support for three Availability Zone deployments
* `service/gamelift`: Updates service API and documentation
  * This release delivers a new API action for deleting unused matchmaking rule sets. More details are available at https://aws.amazon.com/releasenotes/?tag=releasenotes%23keywords%23amazon-gamelift.
* `service/medialive`: Updates service API and documentation
  * This release adds tagging of channels, inputs, and input security groups.
* `service/robomaker`: Updates service API and documentation

Release v1.16.29 (2019-02-06)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Add Linux with SQL Server Standard, Linux with SQL Server Web, and Linux with SQL Server Enterprise to the list of allowed instance platforms for On-Demand Capacity Reservations.
* `service/fsx`: Updates service API and documentation

Release v1.16.28 (2019-02-05)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * ec2.DescribeVpcPeeringConnections pagination support
* `service/servicecatalog`: Updates service documentation
  * Service Catalog Documentation Update for ProvisionedProductDetail
* `service/shield`: Updates service API and documentation
  * The DescribeProtection request now accepts resource ARN as valid parameter.

Release v1.16.27 (2019-02-04)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation
* `service/codecommit`: Updates service API
  * This release supports a more graceful handling of the error case when a repository is not associated with a pull request ID in a merge request in AWS CodeCommit.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for GPU workloads by enabling you to create clusters with GPU-enabled container instances.
* `service/workspaces`: Updates service API
  * This release sets ClientProperties as a required parameter.

Release v1.16.26 (2019-01-25)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * The PutFile API will now throw new exception FilePathConflictsWithSubmodulePathException when a submodule exists at the input file path; PutFile API will also throw FolderContentSizeLimitExceededException when the total size of any folder on the path exceeds the limit as a result of the operation.
* `service/devicefarm`: Updates service API and documentation
  * Introduces a new rule in Device Pools - "Availability". Customers can now ensure they pick devices that are available (i.e., not being used by other customers).
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconnect`: Updates service API and documentation
* `service/medialive`: Updates service API and documentation
  * This release adds support for Frame Capture output groups and for I-frame only manifests (playlists) in HLS output groups.

Release v1.16.25 (2019-01-24)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * This release adds support for cross-account ECR images and private registry authentication.
* `service/ecr`: Updates service API
  * Amazon ECR updated the default endpoint URL to support AWS Private Link.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/logs`: Updates service documentation
  * Documentation updates for CloudWatch Logs
* `service/rds`: Updates service API and documentation
  * The Amazon RDS API allows you to add or remove Identity and Access Management (IAM) role associated with a specific feature name with an RDS database instance. This helps with capabilities such as invoking Lambda functions from within a trigger in the database, load data from Amazon S3 and so on
* `service/sms-voice`: Updates service API and documentation

Release v1.16.24 (2019-01-23)
===

### Service Client Updates
* `service/acm-pca`: Updates service API, documentation, and waiters
* `service/apigatewaymanagementapi`: Updates service API
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/worklink`: Adds new service

### SDK Enhancements
* `aws`: Update Context to be an alias of context.Context for Go 1.9 ([#2412](https://github.com/aws/aws-sdk-go/pull/2412))
  * Updates aws.Context interface to be an alias of the standard libraries context.Context type instead of redefining the interface. This will allow IDEs and utilities to interpret the aws.Context as the exactly same type as the standard libraries context.Context.

Release v1.16.23 (2019-01-21)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update includes support for tagging Stack, Fleet, and ImageBuilder resources at creation time.
* `service/discovery`: Updates service API, documentation, and paginators
  * The Application Discovery Service's import APIs allow you to import information about your on-premises servers and applications into ADS so that you can track the status of your migrations through the Migration Hub console.
* `service/dms`: Updates service waiters
  * Update for DMS TestConnectionSucceeds waiter
* `service/fms`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager State Manager now supports configuration management of all AWS resources through integration with Automation.

Release v1.16.22 (2019-01-18)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Adjust EC2's available instance types.
* `service/glue`: Updates service API and documentation
  * AllocatedCapacity field is being deprecated and replaced with MaxCapacity field

Release v1.16.21 (2019-01-17)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lambda`: Updates service documentation and examples
  * Documentation updates for AWS Lambda
* `service/lightsail`: Updates service API and documentation
  * This release adds functionality to the CreateDiskSnapshot API that allows users to snapshot instance root volumes. It also adds various documentation updates.
* `service/pinpoint`: Updates service API and documentation
  * This release updates the PutEvents operation. AppPackageName, AppTitle, AppVersionCode, SdkName fields will now be accepted as a part of the event when submitting events.
* `service/rekognition`: Updates service API and documentation
  * GetLabelDetection now returns bounding box information for common objects and a hierarchical taxonomy of detected labels. The version of the model used for video label detection is also returned. DetectModerationLabels now returns the version of the model used for detecting unsafe content.

### SDK Enhancements
* `aws/request: Improve error handling in shouldRetryCancel ([#2298](https://github.com/aws/aws-sdk-go/pull/2298))
  * Simplifies and improves SDK's detection of HTTP request errors that should be retried. Previously the SDK would incorrectly attempt to retry `EHOSTDOWN` connection errors. This change fixes this, by using the `Temporary` interface when available.

Release v1.16.20 (2019-01-16)
===

### Service Client Updates
* `service/backup`: Adds new service
* `service/ce`: Updates service documentation
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB now integrates with AWS Backup, a centralized backup service that makes it easy for customers to configure and audit the AWS resources they want to backup, automate backup scheduling, set retention policies, and monitor all recent backup and restore activity. AWS Backup provides a fully managed, policy-based backup solution, simplifying your backup management, and helping you meet your business and regulatory backup compliance requirements. For more information, see the Amazon DynamoDB Developer Guide.

Release v1.16.19 (2019-01-14)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * IMF decode from a Composition Playlist for IMF specializations App #2 and App #2e; up to 99 input clippings; caption channel selection for MXF; and updated rate control for CBR jobs. Added support for acceleration in preview
* `service/storagegateway`: Updates service API and documentation
  * JoinDomain API supports two more  parameters: organizational unit(OU) and domain controllers.  Two new APIs are introduced: DetachVolume and AttachVolume.

### SDK Enhancements
* `aws/endpoints`: Add customization for AWS GovCloud (US) Application Autoscalling ([#2395](https://github.com/aws/aws-sdk-go/pull/2395))
  * Adds workaround to correct the endpoint for Application Autoscaling running in AWS GovCloud (US).
  * Fixes [#2391](https://github.com/aws/aws-sdk-go/issues/2391)

Release v1.16.18 (2019-01-11)
===

### Service Client Updates
* `service/elasticmapreduce`: Updates service API and documentation
  * Documentation updates for Amazon EMR
* `service/rds-data`: Updates service API, documentation, paginators, and examples

Release v1.16.17 (2019-01-10)
===

### Service Client Updates
* `service/codedeploy`: Updates service documentation
  * Documentation updates for codedeploy
* `service/ec2`: Updates service API and documentation
  * EC2 Spot: a) CreateFleet support for Single AvailabilityZone requests and b) support for paginated DescribeSpotInstanceRequests.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API and documentation
  * This release adds tagging support for rules of AWS IoT Rules Engine. Tags enable you to categorize your rules in different ways, for example, by purpose, owner, or environment. For more information about tagging, see AWS Tagging Strategies (https://aws.amazon.com/answers/account-management/aws-tagging-strategies/). For technical documentation, look for the tagging operations in the AWS IoT Core API reference or User Guide (https://docs.aws.amazon.com/iot/latest/developerguide/tagging-iot.html).
* `service/sagemaker`: Updates service API and documentation
  * SageMaker Training Jobs now support Inter-Container traffic encryption.

Release v1.16.16 (2019-01-09)
===

### Service Client Updates
* `service/docdb`: Adds new service
  * Amazon DocumentDB (with MongoDB compatibility) is a fast, reliable, and fully-managed database service. Amazon DocumentDB makes it easy for developers to set up, run, and scale MongoDB-compatible databases in the cloud.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service API and documentation
  * DescribeSnapshotSchedules returns a list of snapshot schedules. With this release, this API will have a list of clusters and number of clusters associated with the schedule.

Release v1.16.15 (2019-01-07)
===

### Service Client Updates
* `service/appmesh`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.16.14 (2019-01-04)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * "This release provides support for running Appium Node.js and Appium Ruby tests on AWS Device Farm.
* `service/ecs`: Updates service documentation
  * Documentation updates for Amazon ECS tagging feature.

Release v1.16.13 (2019-01-03)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iotanalytics`: Updates service API and documentation

### SDK Enhancements
* `aws/credentials`: Add support for getting credential's ExpiresAt. ([#2375](https://github.com/aws/aws-sdk-go/pull/2375))
  * Adds an Expirer interface that Providers can implement, and add a suitable implementation to Expiry class used by most Providers. Add a method on Credentials to get the expiration time of the underlying Provider, if Expirer is supported, without exposing Provider to callers.
  * Fix [#1329](https://github.com/aws/aws-sdk-go/pull/1329)

### SDK Bugs
* `aws/ec2metadata`: bounds check region identifier before split ([#2380](https://github.com/aws/aws-sdk-go/pull/2380))
  * Adds empty response checking to ec2metadata's Region request to prevent a out of bounds panic if empty response received.
* Fix SDK's generated API reference doc page's constants section links ([#2373](https://github.com/aws/aws-sdk-go/pull/2373))
  * Fixes the SDK's generated API reference documentation page's constants section links to to be clickable.

Release v1.16.12 (2019-01-03)
===

### Service Client Updates
* `service/opsworkscm`: Updates service documentation
  * Documentation updates for opsworkscm

Release v1.16.11 (2018-12-21)
===

### Service Client Updates
* `service/acm-pca`: Updates service documentation, waiters, paginators, and examples
* `service/dynamodb`: Updates service API and documentation
  * Added provisionedThroughPut exception on the request level for transaction APIs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/sms-voice`: Updates service API and documentation
* `service/states`: Updates service API and documentation
  * This release adds support for cost allocation tagging. You can now create, delete, and list tags for AWS Step Functions activity and state machine resources. For more information about tagging, see AWS Tagging Strategies.

Release v1.16.10 (2018-12-20)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/comprehend`: Updates service API and documentation
* `service/firehose`: Updates service API and documentation
  * Support for specifying customized s3 keys and supplying a separate prefix for failed-records
* `service/medialive`: Updates service API and documentation
  * This release provides support for ID3 tags and video quality setting for subgop_length.
* `service/transcribe`: Updates service API and documentation

### SDK Enhancements
* `service/dynamodb/expression`: Clarify expression examples ([#2367](https://github.com/aws/aws-sdk-go/pull/2367))
  * Clarifies the expression package's examples to distinguish the pkg expression from a expr value.

Release v1.16.9 (2018-12-19)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release adds support for specifying partition as a strategy for EC2 Placement Groups. This new strategy allows one to launch instances into partitions that do not share certain underlying hardware between partitions, to assist with building and deploying highly available replicated applications.
* `service/sagemaker`: Updates service API and documentation
  * Batch Transform Jobs now supports TFRecord as a Split Type. ListCompilationJobs API action now supports SortOrder and SortBy inputs.
* `service/waf`: Updates service API and documentation
  * This release adds rule-level control for rule group. If a rule group contains a rule that blocks legitimate traffic, previously you had to override the entire rule group to COUNT in order to allow the traffic. You can now use the UpdateWebACL API to exclude specific rules within a rule group. Excluding rules changes the action for the individual rules to COUNT. Excluded rules will be recorded in the new "excludedRules" attribute of the WAF logs.
* `service/waf-regional`: Updates service API and documentation

Release v1.16.8 (2018-12-18)
===

### Service Client Updates
* `service/apigatewaymanagementapi`: Adds new service
* `service/apigatewayv2`: Adds new service
  * This is the initial SDK release for the Amazon API Gateway v2 APIs. This SDK will allow you to manage and configure APIs in Amazon API Gateway; this first release provides the capabilities that allow you to programmatically setup and manage WebSocket APIs end to end.
* `service/ec2`: Updates service API and documentation
  * Client VPN, is a client-based VPN service. With Client VPN, you can securely access resources in AWS as well as access resources in on-premises from any location using OpenVPN based devices. With Client VPN, you can set network based firewall rules that can restrict access to networks based on Active Directory groups.
* `service/elasticbeanstalk`: Updates service API and documentation
  * This release adds a new resource that Elastic Beanstalk will soon support, EC2 launch template, to environment resource descriptions.
* `service/globalaccelerator`: Updates service documentation

Release v1.16.7 (2018-12-17)
===

### Service Client Updates
* `service/ecr`: Updates service API and documentation
  * This release adds support for ECR repository tagging.
* `service/quicksight`: Updates service API and documentation
  * Amazon QuickSight's RegisterUser API now generates a user invitation URL when registering a user with the QuickSight identity type. This URL can then be used by the registered QuickSight user to complete the user registration process. This release also corrects some HTTP return status codes.

Release v1.16.6 (2018-12-14)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/cloudformation`: Updates service documentation
  * Documentation updates for cloudformation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service documentation
  * Documentation updates for Amazon Redshift

### SDK Bugs
* `private/mode/api`: Fix idempotency members not to require validation [#2353](https://github.com/aws/aws-sdk-go/pull/2353)
  * Fixes the SDK's usage of API operation request members marked as idempotency tokens to not require validation. These fields will be auto populated by the SDK if the user does not provide a value. The SDK was requiring the user to provide a value or disable validation to use these APIs.
* deps: Update Go Deps lock file to correct tracking hash [#2354](https://github.com/aws/aws-sdk-go/pull/2354)

Release v1.16.5 (2018-12-13)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations
* `service/pinpoint-email`: Updates service API, documentation, and paginators

Release v1.16.4 (2018-12-12)
===

### Service Client Updates
* `service/eks`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * API Update for Glue: this update enables encryption of password inside connection objects stored in AWS Glue Data Catalog using DataCatalogEncryptionSettings.  In addition, a new "HidePassword" flag is added to GetConnection and GetConnections to return connections without passwords.
* `service/route53`: Updates service API and documentation
  * You can now specify a new region, eu-north-1 (in Stockholm, Sweden), as a region for latency-based or geoproximity routing.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker Automatic Model Tuning now supports early stopping of training jobs. With early stopping, training jobs that are unlikely to generate good models will be automatically stopped during a Hyperparameter Tuning Job.

Release v1.16.3 (2018-12-11)
===

### Service Client Updates
* `service/connect`: Updates service API and documentation
* `service/ecs`: Updates service documentation
  * Documentation updates for Amazon ECS.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediastore`: Updates service API and documentation
  * This release adds Delete Object Lifecycling to AWS MediaStore Containers.

### SDK Bugs
* `private/model/api`: Fix SDK's unmarshaling of unmodeled response payload ([#2340](https://github.com/aws/aws-sdk-go/pull/2340))
  * Fixes the SDK's unmarshaling of API operation response payloads for operations that are unmodeled. Prevents the SDK due to unexpected response payloads causing errors in the API protocol unmarshaler.
  * Fixes [#2332](https://github.com/aws/aws-sdk-go/issues/2332)

Release v1.16.2 (2018-12-07)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/ec2`: Updates service API
  * You can now launch the larger-sized P3dn.24xlarge instance that features NVIDIA Tesla V100s with double the GPU memory, 100Gbps networking and local NVMe storage.
* `service/iam`: Updates service API, documentation, and examples
  * We are making it easier for you to manage your AWS Identity and Access Management (IAM) policy permissions by enabling you to retrieve the last timestamp when an IAM entity (e.g., user, role, or a group) accessed an AWS service. This feature also allows you to audit service access for your entities.
* `service/servicecatalog`: Updates service documentation
  * Documentation updates for servicecatalog.

### SDK Enhancements
* `aws/signer/v4`: Always sign a request with the current time. ([#2336](https://github.com/aws/aws-sdk-go/pull/2336))
  * Updates the SDK's v4 request signer to always sign requests with the current time. For the first request attempt, the request's creation time was used in the request's signature. In edge cases this allowed the signature to expire before the request was sent if there was significant delay between creating the request and sending it, (e.g. rate limiting).
* `aws/endpoints`: Deprecate endpoint service ID generation. ([#2338](https://github.com/aws/aws-sdk-go/pull/2338))
  * Deprecates the service ID generation. The list of service IDs do not directly 1:1 relate to a AWS service. The set of ServiceIDs is confusing, and inaccurate. Instead users should use the EndpointID value defined in each service client's package

Release v1.16.1 (2018-12-06)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Support personal access tokens for GitHub source and app passwords for Bitbucket source
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/medialive`: Updates service API and documentation
  * This release enables the AWS Elemental MediaConnect input type in AWS Elemental MediaLive. This can then be used to automatically create and manage AWS Elemental MediaConnect Flow Outputs when you create a channel using those inputs.
* `service/rds`: Updates service documentation
  * Documentation updates for Amazon RDS

Release v1.16.0 (2018-12-05)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/mediatailor`: Updates service API and documentation
* `service/mq`: Updates service API and documentation
  * This release adds support for cost allocation tagging. You can now create, delete, and list tags for AmazonMQ resources. For more information about tagging, see AWS Tagging Strategies.

### SDK Features
* `aws/credential`: Add credential_process provider ([#2217](https://github.com/aws/aws-sdk-go/pull/2217))
  * Adds support for the shared configuration file's `credential_process` property. This property allows the application to execute a command in order to retrieve AWS credentials for AWS service API request.  In order to use this feature your application must enable the SDK's support of the shared configuration file. See, https://docs.aws.amazon.com/sdk-for-go/api/aws/session/#hdr-Sessions_from_Shared_Config for more information on enabling shared config support.

### SDK Enhancements
* `service/sqs`: Add batch checksum validation test ([#2307](https://github.com/aws/aws-sdk-go/pull/2307))
  * Adds additional test of the SQS batch checksum validation.
* `aws/awsutils`: Update not to retrun sensitive fields for StringValue ([#2310](https://github.com/aws/aws-sdk-go/pull/2310))
* Update SDK client integration tests to be code generated. ([#2308](https://github.com/aws/aws-sdk-go/pull/2308))
* private/mode/api: Update SDK to require URI path members not be empty ([#2323](https://github.com/aws/aws-sdk-go/pull/2323))
  * Updates the SDK's validation to require that members serialized to URI path must not have empty (zero length) values. Generally these fields are modeled as required, but not always. Fixing this will prevent bugs with REST URI paths requests made for unexpected resources.

### SDK Bugs
* aws/session: Fix formatting bug in doc. ([#2294](https://github.com/aws/aws-sdk-go/pull/2294))
  * Fixes a minor issue in aws/session/doc.go where mistakenly used format specifiers in logger.Println.
* Fix SDK model cleanup to remove old model folder ([#2324](https://github.com/aws/aws-sdk-go/pull/2324))
  * Fixes the SDK's model cleanup to remove the entire old model folder not just the api-2.json file.
* Fix SDK's vet usage to use go vet with build tags ([#2300](https://github.com/aws/aws-sdk-go/pull/2300))
  * Updates the SDK's usage of vet to use go vet instead of go tool vet. This allows the SDK to pass build tags and packages instead of just folder paths to the tool.

Release v1.15.90 (2018-12-04)
===

### Service Client Updates
* `service/health`: Updates service API and documentation
  * AWS Health API DescribeAffectedEntities operation now includes a field that returns the URL of the affected entity.
* `service/s3`: Updates service API
  * S3 Inventory reports can now be generated in Parquet format by setting the Destination Format to be 'Parquet'.

Release v1.15.89 (2018-12-03)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * Customers can now schedule runs without a need to create a Device Pool. They also get realtime information on public device availability.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service documentation
  * Documentation updates for mediaconvert
* `service/servicecatalog`: Updates service documentation
  * Documentation updates for servicecatalog
* `service/storagegateway`: Updates service API and documentation
  * API list-local-disks returns a list of the gateway's local disks. This release adds a field DiskAttributeList to these disks.

Release v1.15.88 (2018-11-29)
===

### Service Client Updates
* `service/s3`: Updates service documentation
  * Fixed issue with Content-MD5 for S3 PutObjectLegalHold, PutObjectRetention and PutObjectLockConfiguration.

Release v1.15.87 (2018-11-29)
===

### Service Client Updates
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Support for Managed Rules (rules that are created and maintained by the AWS services in your account) is added.
* `service/kafka`: Adds new service
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports Lambda Layers and Ruby as a runtime. Lambda Layers are a new type of artifact that contains arbitrary code and data, and may be referenced by zero, one, or more functions at the same time.  You can also now develop your AWS Lambda function code using the Ruby programming language.
* `service/s3`: Updates service API and examples
  * Fixed issue with ObjectLockRetainUntilDate in S3 PutObject
* `service/serverlessrepo`: Updates service API, documentation, and paginators
* `service/states`: Updates service API and documentation
  * AWS Step Functions is now integrated with eight additional AWS services: Amazon ECS, AWS Fargate, Amazon DynamoDB, Amazon SNS, Amazon SQS, AWS Batch, AWS Glue, and Amazon SageMaker. To learn more, please see https://docs.aws.amazon.com/step-functions/index.html
* `service/xray`: Updates service API and documentation
  * GetTraceSummaries - Now provides additional information regarding your application traces such as Availability Zone, Instance ID, Resource ARN details, Revision, Entry Point, Root Cause Exceptions and Root Causes for Fault, Error and Response Time.

Release v1.15.86 (2018-11-29)
===

### Service Client Updates
* `service/appmesh`: Adds new service
* `service/ec2`: Updates service API and documentation
  * Adds the following updates: 1. You can now hibernate and resume Amazon-EBS backed instances using the StopInstances and StartInstances APIs. For more information about using this feature and supported instance types and operating systems, visit the user guide. 2. Amazon Elastic Inference accelerators are resources that you can attach to current generation EC2 instances to accelerate your deep learning inference workloads. With Amazon Elastic Inference, you can configure the right amount of inference acceleration to your deep learning application without being constrained by fixed hardware configurations and limited GPU selection. 3. AWS License Manager makes it easier to manage licenses in AWS and on premises when customers run applications using existing licenses from a variety of software vendors including Microsoft, SAP, Oracle, and IBM.
* `service/license-manager`: Adds new service
* `service/lightsail`: Updates service API and documentation
  * This update adds the following features: 1. Copy instance and disk snapshots within the same AWS Region or from one region to another in Amazon Lightsail. 2. Export Lightsail instance and disk snapshots to Amazon Elastic Compute Cloud (Amazon EC2). 3. Create an Amazon EC2 instance from an exported Lightsail instance snapshot using AWS CloudFormation stacks. 4. Apply tags to filter your Lightsail resources, or organize your costs, or control access.
* `service/sagemaker`: Updates service API, documentation, and paginators
  * Amazon SageMaker now has Algorithm and Model Package entities that can be used to create Training Jobs, Hyperparameter Tuning Jobs and hosted Models. Subscribed Marketplace products can be used on SageMaker to create Training Jobs, Hyperparameter Tuning Jobs and Models. Notebook Instances and Endpoints can leverage Elastic Inference accelerator types for on-demand GPU computing. Model optimizations can be performed with Compilation Jobs. Labeling Jobs can be created and supported by a Workforce. Models can now contain up to 5 containers allowing for inference pipelines within Endpoints. Code Repositories (such as Git) can be linked with SageMaker and loaded into Notebook Instances. Network isolation is now possible on Models, Training Jobs, and Hyperparameter Tuning Jobs, which restricts inbound/outbound network calls for the container. However, containers can talk to their peers in distributed training mode within the same security group. A Public Beta Search API was added that currently supports Training Jobs.
* `service/servicediscovery`: Updates service API and documentation
  * AWS Cloud Map lets you define friendly names for your cloud resources so that your applications can quickly and dynamically discover them. When a resource becomes available (for example, an Amazon EC2 instance running a web server), you can register a Cloud Map service instance. Then your application can discover service instances by submitting DNS queries or API calls.

Release v1.15.85 (2018-11-28)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB now supports the following features: DynamoDB on-demand and transactions. DynamoDB on-demand is a flexible new billing option for DynamoDB capable of serving thousands of requests per second without capacity planning. DynamoDB on-demand offers simple pay-per-request pricing for read and write requests so that you only pay for what you use, making it easy to balance costs and performance. Transactions simplify the developer experience of making coordinated, all-or-nothing changes to multiple items both within and across tables. The new transactional APIs provide atomicity, consistency, isolation, and durability (ACID) in DynamoDB, helping developers support sophisticated workflows and business logic that requires adding, updating, or deleting multiple items using native, server-side transactions. For more information, see the Amazon DynamoDB Developer Guide.
* `service/fsx`: Adds new service
* `service/rds`: Updates service API, documentation, and paginators
  * Amazon Aurora Global Database. This release introduces support for Global Database, a feature that allows a single Amazon Aurora database to span multiple AWS regions. Customers can use the feature to replicate data with no impact on database performance, enable fast local reads with low latency in each region, and improve disaster recovery from region-wide outages. You can create, modify and describe an Aurora Global Database, as well as add or remove regions from your Global Database.
* `service/securityhub`: Adds new service

Release v1.15.84 (2018-11-28)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * Support for Amazon ECS service deployment - AWS CodeDeploy now supports the deployment of Amazon ECS services. An Amazon ECS deployment uses an Elastic Load Balancer, two Amazon ECS target groups, and a listener to reroute production traffic from your Amazon ECS service's original task set to a new replacement task set. The original task set is terminated when the deployment is complete. Success of a deployment can be validated using Lambda functions that are referenced by the deployment. This provides the opportunity to rollback if necessary. You can use the new ECSService, ECSTarget, and ECSTaskSet data types in the updated SDK to create or retrieve an Amazon ECS deployment.
* `service/comprehendmedical`: Adds new service
* `service/ec2`: Updates service API and documentation
  * With VPC sharing, you can now allow multiple accounts in the same AWS Organization to launch their application resources, like EC2 instances, RDS databases, and Redshift clusters into shared, centrally managed VPCs.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for blue/green deployment feature. Customers can now update their ECS services in a blue/green deployment pattern via using AWS CodeDeploy.
* `service/kinesisanalytics`: Updates service API and documentation
  * Improvements to error messages, validations, and more to the Kinesis Data Analytics APIs.
* `service/kinesisanalyticsv2`: Adds new service
* `service/logs`: Updates service API and documentation
  * Six new APIs added to support CloudWatch Logs Insights. The APIs are StartQuery, StopQuery, GetQueryResults, GetLogRecord, GetLogGroupFields, and DescribeQueries.
* `service/mediaconnect`: Adds new service
* `service/meteringmarketplace`: Updates service API, documentation, and paginators
  * RegisterUsage operation added to AWS Marketplace Metering Service, allowing sellers to meter and entitle Docker container software use with AWS Marketplace. For details on integrating Docker containers with RegisterUsage see: https://docs.aws.amazon.com/marketplace/latest/userguide/entitlement-and-metering-for-paid-products.html
* `service/translate`: Updates service API and documentation

Release v1.15.83 (2018-11-27)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Adds the following updates: 1. Transit Gateway helps easily scale connectivity across thousands of Amazon VPCs, AWS accounts, and on-premises networks. 2. Amazon EC2 A1 instance is a new Arm architecture based general purpose instance. 3. You can now launch the new Amazon EC2 compute optimized C5n instances that can utilize up to 100 Gbps of network bandwidth.
* `service/globalaccelerator`: Adds new service
* `service/greengrass`: Updates service API and documentation
  * Support Greengrass Connectors and allow Lambda functions to run without Greengrass containers.
* `service/iot`: Updates service API and documentation
  * As part of this release, we are extending capability of AWS IoT Rules Engine to support IoT Events rule action. The IoT Events rule action lets you send messages from IoT sensors and applications to IoT Events for pattern recognition and event detection.
* `service/iotanalytics`: Updates service API and documentation
* `service/kms`: Updates service API and documentation
  * AWS Key Management Service (KMS) now enables customers to create and manage dedicated, single-tenant key stores in addition to the default KMS key store. These are known as custom key stores and are deployed using AWS CloudHSM clusters. Keys that are created in a KMS custom key store can be used like any other customer master key in KMS.
* `service/s3`: Updates service API and documentation
  * Four new Amazon S3 Glacier features help you reduce your storage costs by making it even easier to build archival applications using the Amazon S3 Glacier storage class. S3 Object Lock enables customers to apply Write Once Read Many (WORM) protection to objects in S3 in order to prevent object deletion for a customer-defined retention period. S3 Inventory now supports fields for reporting on S3 Object Lock. "ObjectLockRetainUntilDate", "ObjectLockMode", and "ObjectLockLegalHoldStatus" are now available as valid optional fields.
* `service/sms`: Updates service API, documentation, and paginators
  * In this release, AWS Server Migration Service (SMS) has added multi-server migration support to simplify the application migration process. Customers can migrate all their application-specific servers together as a single unit as opposed to moving individual server one at a time. The new functionality includes - 1. Ability to group on-premises servers into applications and application tiers. 2. Auto-generated CloudFormation Template and Stacks for launching migrated servers into EC2. 3. Ability to run post-launch configuration scripts to configure servers and applications in EC2. In order for SMS to launch servers into your AWS account using CloudFormation Templates, we have also updated the ServerMigrationServiceRole IAM policy to include appropriate permissions. Refer to Server Migration Service documentation for more details.

### SDK Enhancements
* `service/s3/s3manager`: Generate Upload Manager's UploadInput structure ([#2296](https://github.com/aws/aws-sdk-go/pull/2296))
  * Updates the SDK's code generation to also generate the S3 Upload Manager's UploadInput structure type based on the modeled S3 PutObjectInput. This ensures parity between the two types, and the S3 manager does not fall behind the capabilities of PutObject.

### SDK Bugs
* `private/model/api`: Fix model loading to not require docs model. ([#2303](https://github.com/aws/aws-sdk-go/pull/2303))
  * Fixes the SDK's model loading to not require that the docs model be present. This model isn't explicitly required.
* Fixup endpoint discovery unit test to be stable ([#2305](https://github.com/aws/aws-sdk-go/pull/2305))
  * Fixes the SDK's endpoint discovery async unit test to be stable, and produce consistent unit test results.

Release v1.15.82 (2018-11-26)
===

### Service Client Updates
* `service/amplify`: Adds new service
* `service/datasync`: Adds new service
* `service/robomaker`: Adds new service
* `service/s3`: Updates service API, documentation, and examples
  * The INTELLIGENT_TIERING storage class is designed to optimize storage costs by automatically moving data to the most cost effective storage access tier, without performance impact or operational overhead. This SDK release provides API support for this new storage class.
* `service/snowball`: Updates service API and documentation
  * AWS announces the availability of AWS Snowball Edge Compute Optimized to run compute-intensive applications is disconnected and physically harsh environments. It comes with 52 vCPUs, 208GB memory, 8TB NVMe SSD, and 42TB S3-compatible storage to accelerate local processing and is well suited for use cases such as full motion video processing, deep IoT analytics, and continuous machine learning in bandwidth-constrained locations. It features new instances types called SBE-C instances that are available in eight sizes and multiple instances can be run on the device at the same time. Optionally, developers can choose the compute optimized device to include a GPU and use SBE-G instances for accelerating their application performance.
* `service/transfer`: Adds new service
  * AWS Transfer for SFTP is a fully managed service that enables transfer of secure data over the internet into and out of Amazon S3. SFTP is deeply embedded in data exchange workflows across different industries such as financial services, healthcare, advertising, and retail, among others.

Release v1.15.81 (2018-11-21)
===

### Service Client Updates
* `service/rekognition`: Updates service API and documentation
  * This release updates the DetectFaces and IndexFaces operation. When the Attributes input parameter is set to ALL, the face location landmarks includes 5 new landmarks: upperJawlineLeft, midJawlineLeft, chinBottom, midJawlineRight, upperJawlineRight.

Release v1.15.80 (2018-11-20)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/autoscaling-plans`: Updates service API and documentation
* `service/cloudfront`: Adds new service
  * With Origin Failover capability in CloudFront, you can setup two origins for your distributions - primary and secondary, such that your content is served from your secondary origin if CloudFront detects that your primary origin is unavailable. These origins can be any combination of AWS origins or non-AWS custom HTTP origins. For example, you can have two Amazon S3 buckets that serve as your origin that you independently upload your content to. If an object that CloudFront requests from your primary bucket is not present or if connection to your primary bucket times-out, CloudFront will request the object from your secondary bucket. So, you can configure CloudFront to trigger a failover in response to either HTTP 4xx or 5xx status codes.
* `service/devicefarm`: Updates service API and documentation
  * Disabling device filters
* `service/medialive`: Updates service API and documentation
  * You can now include the media playlist(s) from both pipelines in the HLS master manifest for seamless failover.
* `service/monitoring`: Updates service API and documentation
  * Amazon CloudWatch now supports alarms on metric math expressions.
* `service/quicksight`: Adds new service
  * Amazon QuickSight is a fully managed, serverless, cloud business intelligence system that allows you to extend data and insights to every user in your organization. The first release of APIs for Amazon QuickSight introduces embedding and user/group management capabilities. The get-dashboard-embed-url API allows you to obtain an authenticated dashboard URL that can be embedded in application domains whitelisted for QuickSight dashboard embedding. User APIs allow you to programmatically expand and manage your QuickSight deployments while group APIs allow easier permissions management for resources within QuickSight.
* `service/rds-data`: Adds new service
* `service/redshift`: Updates service documentation
  * Documentation updates for redshift
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Distributor helps you securely distribute and install software packages.
* `service/xray`: Updates service API and documentation
  * Groups build upon X-Ray filter expressions to allow for fine tuning trace summaries and service graph results. You can configure groups by using the AWS X-Ray console or by using the CreateGroup API. The addition of groups has extended the available request fields to the GetServiceGraph API. You can now specify a group name or group ARN to retrieve its service graph.

Release v1.15.79 (2018-11-20)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Adding multinode parallel jobs, placement group support for compute environments.
* `service/cloudformation`: Updates service API and documentation
  * Use the CAPABILITY_AUTO_EXPAND capability to create or update a stack directly from a stack template that contains macros, without first reviewing the resulting changes in a change set first.
* `service/cloudtrail`: Updates service API and documentation
  * This release supports creating a trail in CloudTrail that logs events for all AWS accounts in an organization in AWS Organizations. This helps enable you to define a uniform event logging strategy for your organization. An organization trail is applied automatically to each account in the organization and cannot be modified by member accounts. To learn more, please see the AWS CloudTrail User Guide https://docs.aws.amazon.com/awscloudtrail/latest/userguide/cloudtrail-user-guide.html
* `service/config`: Updates service API and documentation
* `service/devicefarm`: Updates service API and documentation
  * Customers can now schedule runs without a need to create a Device Pool. They also get realtime information on public device availability.
* `service/ec2`: Updates service API and documentation
  * Adding AvailabilityZoneId to DescribeAvailabilityZones
* `service/iot`: Updates service API and documentation
  * IoT now supports resource tagging and tag based access control for Billing Groups, Thing Groups, Thing Types, Jobs, and Security Profiles. IoT Billing Groups help you group devices to categorize and track your costs. AWS IoT Device Management also introduces three new features: 1. Dynamic thing groups. 2. Jobs dynamic rollouts. 3. Device connectivity indexing. Dynamic thing groups lets you to create a group of devices using a Fleet Indexing query. The devices in your group will be automatically added or removed when they match your specified query criteria. Jobs dynamic rollout allows you to configure an exponentially increasing rate of deployment for device updates and define failure criteria to cancel your job. Device connectivity indexing allows you to index your devices' lifecycle events to discover whether devices are connected or disconnected to AWS IoT.
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports python3.7 and  the Kinesis Data Streams (KDS) enhanced fan-out and HTTP/2 data retrieval features for Kinesis event sources.
* `service/lightsail`: Updates service API
  * Add Managed Database operations to OperationType enum.
* `service/mediaconvert`: Updates service API and documentation
  * AWS Elemental MediaConvert SDK has added several features including support for: SPEKE full document encryption, up to 150 elements for input stitching, input and motion image insertion, AWS CLI path arguments in S3 links including special characters, AFD signaling, additional caption types, and client-side encrypted input files.
* `service/rds`: Updates service API and documentation
  * This release adds a new parameter to specify VPC security groups for restore from DB snapshot, restore to point int time and create read replica operations. For more information, see Amazon RDS Documentation.
* `service/workdocs`: Updates service API and documentation
  * With this release, clients can now use the GetResources API to fetch files and folders from the user's SharedWithMe collection. And also through this release, the existing DescribeActivities API has been enhanced to support additional filters such as the ActivityType and the ResourceId.
* `service/workspaces`: Updates service API and documentation
  * Added new APIs to Modify and Describe WorkSpaces client properties for users in a directory. With the new APIs, you can enable/disable remember me option in WorkSpaces client for users in a directory.

### SDK Bugs
* `internal/ini`: trimSpaces not trimming rhs properly (#2282)
  * Fixes trimSpaces to behave properly by removing the necessary rhs spaces of a literal.

Release v1.15.78 (2018-11-16)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/comprehend`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for additional Docker flags as Task Definition parameters. Customers can now configure their ECS Tasks to use pidMode (pid) and ipcMode (ipc) Docker flags.
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Automation now allows you to execute and manage Automation workflows across multiple accounts and regions.
* `service/workspaces`: Updates service API and documentation
  * Added new Bring Your Own License (BYOL) automation APIs. With the new APIs, you can list available management CIDR ranges for dedicated tenancy, enable your account for BYOL, describe BYOL status of your account, and import BYOL images. Added new APIs to also describe and delete WorkSpaces images.

Release v1.15.77 (2018-11-16)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Adding queue phase and configurable queue timeout to CodeBuild.
* `service/comprehend`: Updates service API and documentation
* `service/directconnect`: Updates service API and documentation
  * This release enables DirectConnect customers to have logical redundancy on virtual interfaces within supported DirectConnect locations.
* `service/dms`: Updates service API, documentation, and waiters
  * Settings structures have been added to our DMS endpoint APIs to support Kinesis and Elasticsearch as targets. We are introducing the ability to configure custom DNS name servers on a replication instance as a beta feature.
* `service/ecs`: Updates service API, documentation, and examples
  * In this release, Amazon ECS introduces multiple features. First, ECS now supports integration with Systems Manager Parameter Store for injecting runtime secrets. Second, ECS introduces support for resources tagging. Finally, ECS introduces a new ARN and ID Format for its resources, and provides new APIs for opt-in to the new formats.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iam`: Updates service API, documentation, and examples
  * We are making it easier for you to manage your AWS Identity and Access Management (IAM) resources by enabling you to add tags to your IAM principals (users and roles). Adding tags on IAM principals will enable you to write fewer policies for permissions management and make policies easier to comprehend.  Additionally, tags will also make it easier for you to grant access to AWS resources.
* `service/pinpoint`: Updates service API and documentation
  * 1. With Amazon Pinpoint Voice, you can use text-to-speech technology to deliver personalized voice messages to your customers. Amazon Pinpoint Voice is a great way to deliver transactional messages -- such as one-time passwords and identity confirmations -- to customers. 2. Adding support for Campaign Event Triggers. With Campaign Event Triggers you can now schedule campaigns to execute based on incoming event data and target just the source of the event.
* `service/ram`: Adds new service
* `service/rds`: Updates service API, documentation, and paginators
  * Introduces DB Instance Automated Backups for the MySQL, MariaDB, PostgreSQL, Oracle and Microsoft SQL Server database engines. You can now retain Amazon RDS automated backups (system snapshots and transaction logs) when you delete a database instance. This allows you to restore a deleted database instance to a specified point in time within the backup retention period even after it has been deleted, protecting you against accidental deletion of data. For more information, see Amazon RDS Documentation.
* `service/redshift`: Updates service API and documentation
  * With this release, Redshift is providing API's for better snapshot management by supporting user defined automated snapshot schedules, retention periods for manual snapshots, and aggregate snapshot actions including batch deleting user snapshots, viewing account level snapshot storage metrics, and better filtering and sorting on the describe-cluster-snapshots API. Automated snapshots can be scheduled to be taken at a custom interval and the schedule created can be reused across clusters. Manual snapshot retention periods can be set at the cluster, snapshot, and cross-region-copy level. The retention period set on a manual snapshot indicates how many days the snapshot will be retained before being automatically deleted.
* `service/route53resolver`: Adds new service
* `service/s3`: Updates service API, documentation, and examples
  * Add support for new S3 Block Public Access bucket-level APIs. The new Block Public Access settings allow bucket owners to prevent public access to S3 data via bucket/object ACLs or bucket policies.
* `service/s3control`: Adds new service
  * Add support for new S3 Block Public Access account-level APIs. The Block Public Access settings allow account owners to prevent public access to S3 data via bucket/object ACLs or bucket policies.
* `service/sms-voice`: Adds new service
* `service/transcribe`: Updates service API and documentation

Release v1.15.76 (2018-11-14)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * EC2 Auto Scaling now allows users to provision and automatically scale instances across purchase options (Spot, On-Demand, and RIs) and instance types in a single Auto Scaling group (ASG).
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 Fleet now supports a new request type "Instant" that you can use to provision capacity synchronously across instance types & purchase models and CreateFleet will return the instances launched in the API response.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediatailor`: Updates service API and documentation
* `service/resource-groups`: Updates service API and documentation
* `service/sagemaker`: Updates service API and documentation
  * SageMaker now makes the final set of metrics published from training jobs available in the DescribeTrainingJob results.  Automatic Model Tuning now supports warm start of hyperparameter tuning jobs.  Notebook instances now support a larger number of instance types to include instances from the ml.t3, ml.m5, ml.c4, ml.c5 families.
* `service/servicecatalog`: Updates service API and documentation
  * Adds support for Cloudformation StackSets in Service Catalog
* `service/sns`: Updates service API and documentation
  * Added an optional request parameter, named Attributes, to the Amazon SNS CreateTopic API action. For more information, see the Amazon SNS API Reference (https://docs.aws.amazon.com/sns/latest/api/API_CreateTopic.html).

Release v1.15.75 (2018-11-13)
===

### Service Client Updates
* `service/budgets`: Updates service documentation
  * Doc Update: 1. Available monthly-budgets maximal history data points from 12 to 13.  2. Added 'Amazon Elasticsearch' costfilters support.
* `service/chime`: Updates service API and documentation
  * This release adds support in ListUsers API to filter the list by an email address.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service API and documentation
  * Amazon Redshift provides the option to defer non-mandatory maintenance updates to a later date.

Release v1.15.74 (2018-11-12)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Adding EC2 Launch Template support in AWS Batch Compute Environments.
* `service/budgets`: Updates service API and documentation
  * 1. Added budget performance history, enabling you to see how well your budgets matched your actual costs and usage.                                                                                             2. Added budget performance history, notification state, and last updated time, enabling you to see how well your budgets matched your actual costs and usage, how often your budget alerts triggered, and when your budget was last updated.
* `service/cloudformation`: Updates service API, documentation, and paginators
  * The Drift Detection feature enables customers to detect whether a stack's actual configuration differs, or has drifted, from its expected configuration as defined within AWS CloudFormation.
* `service/codepipeline`: Updates service API and documentation
  * Add support for cross-region pipeline with accompanying definitions as needed in the AWS CodePipeline API Guide.
* `service/firehose`: Updates service API and documentation
  * With this release, Amazon Kinesis Data Firehose allows you to enable/disable server-side encryption(SSE) for your delivery streams ensuring encryption of data at rest. For technical documentation, look at https://docs.aws.amazon.com/firehose/latest/dev/encryption.html
* `service/polly`: Updates service API
  * Amazon Polly adds new female voices: Italian - Bianca, Castilian Spanish - Lucia and new language: Mexican Spanish with new female voice - Mia.
* `service/rds`: Updates service API and documentation
  * API Update for RDS: this update enables Custom Endpoints, a new feature compatible with Aurora Mysql, Aurora PostgreSQL and Neptune that allows users to configure a customizable endpoint that will provide access to their instances in a cluster.

### SDK Bugs
* `internal/ini`: allowing LHS of equal expression to contain spaces (#2265)
  * Fixes a backward compatibility issue where LHS of equal expr could contain spaces

Release v1.15.73 (2018-11-09)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediapackage`: Updates service API and documentation
  * As a part of SPEKE DRM encryption, MediaPackage now supports encrypted content keys. You can enable this enhanced content protection in an OriginEndpoint's encryption settings. When this is enabled, MediaPackage indicates to the key server that it requires an encrypted response. To use this, your DRM key provider must support content key encryption. For details on this feature, see the AWS MediaPackage User Guide at https://docs.aws.amazon.com/mediapackage/latest/ug/what-is.html.

Release v1.15.72 (2018-11-08)
===

### Service Client Updates
* `service/dlm`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/events`: Updates service documentation
  * Documentation updates for events
* `service/medialive`: Updates service API and documentation
  * You can now switch a live channel between preconfigured inputs. This means assigned inputs for a running channel can be changed according to a defined schedule. You can also use MP4 files as inputs.

Release v1.15.71 (2018-11-07)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/dms`: Updates service waiters
  * Update the DMS TestConnectionSucceeds waiter.
* `service/ec2`: Updates service API and documentation
  * VM Import/Export now supports generating encrypted EBS snapshots, as well as AMIs backed by encrypted EBS snapshots during the import process.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.15.70 (2018-11-06)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * AWS WAF integration with APIGW. Changes for adding webAclArn as a part of  Stage output. When the user calls a get-stage or get-stages, webAclArn will also be returned as a part of the output.
* `service/codebuild`: Updates service documentation
  * Documentation updates for codebuild
* `service/ec2`: Updates service API and paginators
  * You can now launch the new Amazon EC2 memory optimized R5a and general purpose M5a instances families that feature AMD EPYC processors.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/pinpoint`: Updates service API and documentation
  * This update adds the ability to send transactional email by using the SendMessage API. Transactional emails are emails that you send directly to specific email addresses. Unlike campaign-based email that you send from Amazon Pinpoint, you don't have to create segments and campaigns in order to send transactional email.
* `service/pinpoint-email`: Adds new service
* `service/waf-regional`: Updates service API and documentation

Release v1.15.69 (2018-11-05)
===

### Service Client Updates
* `service/eks`: Updates service waiters
* `service/serverlessrepo`: Updates service API and documentation

Release v1.15.68 (2018-11-02)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * ListObjectParents API now supports a bool parameter IncludeAllLinksToEachParent, which if set to true, will return a ParentLinks list instead of a Parents map; BatchRead API now supports ListObjectParents operation.
* `service/rekognition`: Updates service API and documentation
  * This release updates the DetectLabels operation. Bounding boxes are now returned for certain objects, a hierarchical taxonomy is now available for labels, and you can now get the version of the detection model used for detection.

### SDK Bugs
* `internal/ini`: profile names did not allow for ':' character (#2247)
  * Fixes an issue where profile names would return an error if the name contained a ':'

Release v1.15.67 (2018-11-01)
===

### Service Client Updates
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * Service Catalog integration with AWS Organizations, enables customers to more easily create and manage a portfolio of IT services across an organization. Administrators can now take advantage of the AWS account structure and account groupings configured in AWS Organizations to share Service Catalog Portfolios increasing agility and reducing risk. With this integration the admin user will leverage the trust relationship that exists within the accounts of the Organization to share portfolios to the entire Organization, a specific Organizational Unit or a specific Account.

### SDK Bugs
* `internal/ini`: removing // comments (#2240)
  * removes // comments since that was never supported previously.

Release v1.15.66 (2018-10-31)
===

### Service Client Updates
* `service/config`: Updates service API
* `service/greengrass`: Updates service API and documentation
  * Greengrass APIs now support bulk deployment operations, and APIs that list definition versions now support pagination.
* `service/mediastore-data`: Updates service API and documentation
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for AWS Secrets Manager.

Release v1.15.65 (2018-10-30)
===

### Service Client Updates
* `service/chime`: Adds new service
  * This is the initial release for the Amazon Chime AWS SDK. In this release, Amazon Chime adds support for administrative actions on users and accounts. API Documentation is also updated on https://docs.aws.amazon.com/chime/index.html
* `service/dms`: Updates service waiters
  * Add waiters for TestConnectionSucceeds, EndpointDeleted, ReplicationInstanceAvailable, ReplicationInstanceDeleted, ReplicationTaskReady, ReplicationTaskStopped, ReplicationTaskRunning and ReplicationTaskDeleted.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * This release adds the listener connection endpoint for SQL Server Always On to the list of fields returned when performing a describe-db-instances operation.

Release v1.15.64 (2018-10-26)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service documentation
* `service/sagemaker`: Updates service API and documentation
  * SageMaker notebook instances can now have a volume size configured.
* `service/ssm`: Updates service API and documentation
  * Compliance Severity feature release for State Manager. Users now have the ability to select compliance severity to their association in state manager console or CLI.

Release v1.15.63 (2018-10-25)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * As part of this release we are introducing EC2 On-Demand Capacity Reservations. With On-Demand Capacity Reservations, customers can reserve the exact EC2 capacity they need, and can keep it only for as long as they need it.

Release v1.15.62 (2018-10-24)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/codestar`: Updates service API and documentation
  * This release lets you create projects from source code and a toolchain definition that you provide.

Release v1.15.61 (2018-10-23)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and examples
  * Provides customers the ability to Bring Your Own IP (BYOIP) prefix.  You can bring part or all of your public IPv4 address range from your on-premises network to your AWS account. You continue to own the address range, but AWS advertises it on the internet.

Release v1.15.60 (2018-10-22)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/inspector`: Updates service API and documentation
  * Finding will be decorated with ec2 related metadata
* `service/shield`: Updates service API and documentation
  * AWS Shield Advanced API introduced a new service-specific AccessDeniedException which will be thrown when accessing individual attack information without sufficient permission.

Release v1.15.59 (2018-10-19)
===

### Service Client Updates
* `service/ssm`: Updates service API and documentation
  * Rate Control feature release for State Manager. Users now have the ability to apply rate control parameters similar to run command to their association in state manager console or CLI.
* `service/workspaces`: Updates service API
  * Added support for PowerPro and GraphicsPro WorkSpaces bundles.

### SDK Enhancements
* `aws/request`: Add private ini package (#2210)
  * Get rids of go-ini dependency in favor of `internal/ini` package.

Release v1.15.58 (2018-10-18)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update adds support for creating, managing, and deleting users in the AppStream 2.0 user pool.
* `service/medialive`: Updates service API and documentation
  * This release allows you to now turn on Quality-Defined Variable Bitrate (QVBR) encoding for your AWS Elemental MediaLive channels. You can now deliver a consistently high-quality video viewing experience while reducing overall distribution bitrates by using Quality-Defined Variable Bitrate (QVBR) encoding with AWS Elemental MediaLive. QVBR is a video compression technique that automatically adjusts output bitrates to the complexity of source content and only use the bits required to maintain a defined level of quality. This means using QVBR encoding, you can save on distribution cost, while maintaining, or increasing video quality for your viewers.
* `service/route53`: Updates service API and documentation
  * This change allows customers to disable health checks.

Release v1.15.57 (2018-10-17)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Documentation updates for API Gateway
* `service/events`: Updates service API and documentation
  * AWS Events - AWS Organizations Support in Event-Bus Policies. This release introduces a new parameter in the PutPermission API named Condition. Using the Condition parameter, customers can allow one or more AWS Organizations to access their CloudWatch Events Event-Bus resource.

Release v1.15.56 (2018-10-16)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * New Glue APIs for creating, updating, reading and deleting Data Catalog resource-based policies.
* `service/lightsail`: Updates service API and documentation
  * Adds support for Lightsail managed databases.
* `service/resource-groups`: Updates service API and documentation

Release v1.15.55 (2018-10-15)
===

### Service Client Updates
* `service/lambda`: Updates service API and documentation
  * Documentation updates for lambda
* `service/rds`: Updates service API and documentation
  * This release adds a new parameter to specify the DB instance or cluster parameter group for restore from DB snapshot and restore to point int time operations. For more information, see Amazon RDS Documentation.
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * AWS Service Catalog enables you to reduce administrative maintenance and end-user training while adhering to compliance and security measures. With service actions, you as the administrator can enable end users to perform operational tasks, troubleshoot issues, run approved commands, or request permissions within Service Catalog. Service actions are defined using AWS Systems Manager documents, where you have access to pre-defined actions that implement AWS best practices, such asEC2 stop and reboot, as well as the ability to define custom actions.

Release v1.15.54 (2018-10-12)
===

### Service Client Updates
* `service/cloudtrail`: Updates service API and documentation
  * The LookupEvents API now supports two new attribute keys: ReadOnly and AccessKeyId

### SDK Enhancements
* `aws/session`: Add support for credential source(#2201)
  * Allows for shared config file to contain `credential_source` with any of the given values `EcsContainer`, `Environment` or `Ec2InstanceMetadata`

Release v1.15.53 (2018-10-11)
===

### Service Client Updates
* `service/athena`: Updates service API and documentation
  * 1. GetQueryExecution API changes to return statementType of a submitted Athena query.  2. GetQueryResults API changes to return the number of rows added to a table when a CTAS query is executed.
* `service/directconnect`: Updates service API and documentation
  * This release adds support for Jumbo Frames over AWS Direct Connect. You can now set MTU value when creating new virtual interfaces. This release also includes a new API to modify MTU value of existing virtual interfaces.
* `service/ec2`: Updates service API
  * You can now launch the smaller-sized G3 instance called g3s.xlarge. G3s.xlarge provides 4 vCPU, 30.5 GB RAM and a NVIDIA Tesla M60 GPU. It is ideal for remote workstations, engineering and architectural applications, and 3D visualizations and rendering for visual effects.
* `service/mediaconvert`: Updates service paginators
  * Added Paginators for all the MediaConvert list operations
* `service/transcribe`: Updates service API and documentation

Release v1.15.52 (2018-10-10)
===

### Service Client Updates
* `service/comprehend`: Updates service API
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service now supports customer-scheduled service software updates. When new service software becomes available, you can request an update to your domain and benefit from new features more quickly. If you take no action, we update the service software automatically after a certain time frame.
* `service/transcribe`: Updates service API and documentation

Release v1.15.51 (2018-10-09)
===

### Service Client Updates
* `service/ssm`: Updates service API and documentation
  * Adds StartDate, EndDate, and ScheduleTimezone to CreateMaintenanceWindow and UpdateMaintenanceWindow; Adds NextExecutionTime to GetMaintenanceWindow and DescribeMaintenanceWindows; Adds CancelMaintenanceWindowExecution, DescribeMaintenanceWindowSchedule and DescribeMaintenanceWindowsForTarget APIs.

Release v1.15.50 (2018-10-08)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * We are releasing job execution timeout functionalities to customers. Customer now can set job execution timeout on the job level when creating a job.
* `service/iot-jobs-data`: Updates service API and documentation

Release v1.15.49 (2018-10-05)
===

### Service Client Updates
* `service/ds`: Updates service API and documentation
  * SDK changes to create a new type of trust for active directory

Release v1.15.48 (2018-10-04)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Adding support for multi-value parameters in TestInvokeMethod and TestInvokeAuthorizer.
* `service/codebuild`: Updates service API and documentation
  * Add resolved source version field in build output
* `service/ssm`: Updates service API and documentation
  * Adds RejectedPatchesAction to baseline to enable stricted validation of the rejected Patches List ; Add InstalledRejected and InstallOverrideList to compliance reporting
* `service/storagegateway`: Updates service API and documentation
  * AWS Storage Gateway now enables you to specify folders and subfolders when you update your file gateway's view of your S3 objects using the Refresh Cache API.

Release v1.15.47 (2018-10-02)
===

### Service Client Updates
* `service/sagemaker`: Updates service waiters
  * Waiter for SageMaker Batch Transform Jobs.
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

### SDK Enhancements
* `aws/config`: fix typo in Config struct documentation (#2169)
  * fix typo in Config struct documentation in aws-sdk-go/aws/config.go
* `internal/csm`: Add region to api call metrics (#2175)
* `private/model/api`: Use modeled service signing version in code generation (#2162)
  * Updates the SDK's code generate to make use of the model's service signature version when generating the client for the service. This allows the SDK to generate a client using the correct signature version, e.g v4 vs s3v4 without the need for additional customizations.

### SDK Bugs
* `service/cloudfront/sign`: Do not Escape HTML when encode the cloudfront sign policy (#2164)
  * Fixes the signer escaping HTML elements `<`, `>`, and `&` in the signature policy incorrectly. Allows use of multiple query parameters in the URL to be signed.
  * Fixes #2163

Release v1.15.46 (2018-10-01)
===

### Service Client Updates
* `service/guardduty`: Updates service API and documentation
  * Support optional FindingPublishingFrequency parameter in CreateDetector and UpdateDetector operations, and ClientToken on Create* operations
* `service/rekognition`: Updates service documentation
  * Documentation updates for Amazon Rekognition

Release v1.15.45 (2018-09-28)
===

### Service Client Updates
* `service/codestar`: Updates service API and documentation
  * This release enables tagging CodeStar Projects at creation. The CreateProject API now includes optional tags parameter.
* `service/ec2`: Updates service API
  * You can now use EC2 High Memory instances with 6 TiB memory (u-6tb1.metal), 9 TiB memory (u-9tb1.metal), and 12 TiB memory (u-12tb1.metal), which are ideal for running large in-memory databases, including production deployments of SAP HANA. These instances offer 448 logical processors, where each logical processor is a hyperthread on 224 cores. These instance deliver high networking throughput and lower latency with up to 25 Gbps of aggregate network bandwidth using Elastic Network Adapter (ENA)-based Enhanced Networking. These instances are EBS-Optimized by default, and support encrypted and unencrypted EBS volumes. This instance is only available in host-tenancy. You will need an EC2 Dedicated Host for this instance type to launch an instance.

Release v1.15.44 (2018-09-27)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Adding support for OpenAPI 3.0 import and export.
* `service/codecommit`: Updates service API and documentation
  * This release adds API support for getting the contents of a file, getting the contents of a folder, and for deleting a file in an AWS CodeCommit repository.
* `service/mq`: Updates service API and documentation
  * Amazon MQ supports ActiveMQ 5.15.6, in addition to 5.15.0. Automatic minor version upgrades can be toggled. Updated the documentation.

Release v1.15.43 (2018-09-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * AWS Glue now supports data encryption at rest for ETL jobs and development endpoints. With encryption enabled, when you run ETL jobs, or development endpoints, Glue will use AWS KMS keys to write encrypted data at rest. You can also encrypt the metadata stored in the Glue Data Catalog using keys that you manage with AWS KMS. Additionally, you can use AWS KMS keys to encrypt the logs generated by crawlers and ETL jobs as well as encrypt ETL job bookmarks. Encryption settings for Glue crawlers, ETL jobs, and development endpoints can be configured using the security configurations in Glue. Glue Data Catalog encryption can be enabled via the settings for the Glue Data Catalog.
* `service/opsworkscm`: Updates service API and documentation
  * This release introduces a new API called ExportServerEngineAttribute to Opsworks-CM. You can use this API call to export engine specific attributes like the UserData script used for unattended bootstrapping of new nodes that connect to the server.
* `service/rds`: Updates service API and documentation
  * This release includes Deletion Protection for RDS databases.
* `service/sqs`: Updates service API and documentation
  * Documentation updates for Amazon SQS.

### SDK Enhancements
* `private/protocol/restjson/restjson`: Use json.Decoder to decrease memory allocation (#2141)
  * Update RESTJSON protocol unmarshaler to use json.Decoder instead of ioutil.ReadAll to reduce allocations.
* `private/protocol/jsonrpc/jsonrpc`: Use json.Decoder to decrease memory allocation (#2142)
  * Update JSONPRC protocol unmarshaler to use json.Decoder instead of ioutil.ReadAll to reduce allocations.

Release v1.15.42 (2018-09-25)
===

### Service Client Updates
* `service/cloudfront`: Updates service documentation
  * Documentation updates for cloudfront
* `service/ds`: Updates service API and documentation
  * API changes related to launch of cross account for Directory Service.
* `service/ec2`: Updates service API and documentation
  * Add pagination support for ec2.describe-route-tables API.

Release v1.15.41 (2018-09-24)
===

### Service Client Updates
* `service/connect`: Updates service API, documentation, and paginators
* `service/rds`: Updates service API and documentation
  * Adds DB engine version requirements for option group option settings, and specifies if an option setting requires a value.

Release v1.15.40 (2018-09-21)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * To offer lower prices for predictable, non-urgent workloads, we propose the concept of Reserved Transcode pricing. Reserved Transcode pricing Reserved Transcoding pricing would offer the customer access to a fixed parallel processing capacity for a fixed monthly rate. This capacity would be stated in terms of number of Reserved Transcode Slots (RTSs). One RTS would be able to process one job at a time for a fixed monthly fee.

Release v1.15.39 (2018-09-20)
===

### Service Client Updates
* `service/ds`: Updates service API and documentation
  * Added CreateLogSubscription, DeleteLogSubscription, and ListLogSubscriptions APIs for Microsoft AD. Customers can now opt in to have Windows security event logs from the domain controllers forwarded to a log group in their account.
* `service/ec2`: Updates service API
  * You can now launch f1.4xlarge, a new instance size within the existing f1 family which provides two Xilinx Virtex Field Programmable Arrays (FPGAs) for acceleration. FPGA acceleration provide additional performance and time sensitivity for specialized accelerated workloads such as clinical genomics and real-time video processing. F1.4xlarge instances are available in the US East (N. Virginia), US West (Oregon), GovCloud (US), and EU West (Dublin) AWS Regions.
* `service/rds`: Updates service API and documentation
  * This launch enables RDS start-db-cluster and stop-db-cluster. Stopping and starting Amazon Aurora clusters helps you manage costs for development and test environments. You can temporarily stop all the DB instances in your cluster, instead of setting up and tearing down all the DB instances each time that you use the cluster.

Release v1.15.38 (2018-09-19)
===

### Service Client Updates
* `service/monitoring`: Updates service API and documentation
  * Amazon CloudWatch adds the ability to request png image snapshots of metric widgets using the GetMetricWidgetImage API.
* `service/organizations`: Updates service API and documentation
  * Introducing a new exception - AccountOwnerNotVerifiedException which will be returned for InviteAccountToOrganization call for unverified accounts.
* `service/s3`: Updates service API and documentation
  * S3 Cross Region Replication now allows customers to use S3 object tags to filter the scope of replication. By using S3 object tags, customers can identify individual objects for replication across AWS Regions for compliance and data protection. Cross Region Replication for S3 enables automatic and asynchronous replication of objects to another AWS Region, and with this release customers can replicate at a bucket level, prefix level or by using object tags.

Release v1.15.37 (2018-09-18)
===

### Service Client Updates
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service adds support for node-to-node encryption for new domains running Elasticsearch version 6.0 and above
* `service/rekognition`: Updates service API and documentation
  * This release updates the Amazon Rekognition IndexFaces API operation. It introduces a QualityFilter parameter that allows you to automatically filter out detected faces that are deemed to be of low quality by Amazon Rekognition. The quality bar is based on a variety of common use cases.  You can filter low-quality detected faces by setting QualityFilter to AUTO, which is also the default setting. To index all detected faces regardless of quality, you can specify NONE.  This release also provides a MaxFaces parameter that is useful when you want to only index the most prominent and largest faces in an image and don't want to index other faces detected in the image, such as smaller faces belonging to people standing in the background.

Release v1.15.36 (2018-09-17)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Support build logs configuration.
* `service/ec2`: Updates service API and documentation
  * Added support for customers to tag EC2 Dedicated Hosts on creation.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for additional Docker flags as Task Definition parameters. Customers can now configure their ECS Tasks to use systemControls (sysctl), pseudoTerminal (tty), and interactive (i) Docker flags.
* `service/elasticache`: Updates service API and documentation
  * ElastiCache for Redis added support for adding and removing read-replicas from any cluster with no cluster downtime, Shard naming: ElastiCache for Redis customers have the option of allowing ElastiCache to create names for their node groups (shards) or generating their own node group names. For more information, see https:// docs.aws.amazon.com/AmazonElastiCache/latest/APIReference/API_NodeGroupConfiguration.html, ShardsToRetain: When reducing the number of node groups (shards) in an ElastiCache for Redis (cluster mode enabled) you have the option of specifying which node groups to retain or which node groups to remove. For more information, see https:// docs.aws.amazon.com/AmazonElastiCache/latest/APIReference/API_ModifyReplicationGroupShardConfiguration.html, ReservationARN: ReservedNode includes an ARN, ReservationARN, member which identifies the reserved node. For more information, see https:// docs.aws.amazon.com/AmazonElastiCache/latest/APIReference/API_ReservedCacheNode.html
* `service/elastictranscoder`: Updates service API, documentation, and paginators
  * Added support for MP2 container
* `service/monitoring`: Updates service API and documentation
  * Amazon CloudWatch adds the ability to publish values and counts using PutMetricData
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

Release v1.15.35 (2018-09-13)
===

### Service Client Updates
* `service/polly`: Updates service API and documentation
  * Amazon Polly adds Mandarin Chinese language support with new female voice - "Zhiyu"

Release v1.15.34 (2018-09-12)
===

### Service Client Updates
* `service/connect`: Updates service API and documentation
* `service/ec2`: Updates service API, documentation, and paginators
  * Pagination Support for DescribeNetworkInterfaces API
* `service/email`: Updates service documentation
  * Documentation updates for Amazon Simple Email Service
* `service/fms`: Updates service API and documentation

Release v1.15.33 (2018-09-11)
===

### Service Client Updates
* `service/opsworkscm`: Updates service documentation
  * Documentation updates for opsworkscm
* `service/ssm`: Updates service API and documentation
  * Session Manager is a fully managed AWS Systems Manager capability that provides interactive one-click access to Amazon EC2 Linux and Windows instances.

Release v1.15.32 (2018-09-10)
===

### Service Client Updates
* `service/cloudhsmv2`: Updates service API and documentation
  * With this release, we are adding 2 new APIs. DeleteBackup deletes a specified AWS CloudHSM backup. A backup can be restored up to 7 days after the DeleteBackup request. During this 7-day period, the backup will be in state PENDING_DELETION. Backups can be restored using the RestoreBackup API, which will move the backup from state PENDING_DELETION back to ACTIVE.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service API and documentation
  * Adding support to Redshift to change the encryption type after cluster creation completes.

Release v1.15.31 (2018-09-07)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/logs`: Updates service API and documentation
  * * Adding a log prefix parameter for filter log events API and minor updates to the documentation

### SDK Enhancements
* `private/protocol/json/jsonutil`: Use json.Decoder to decrease memory allocation ([#2115](https://github.com/aws/aws-sdk-go/pull/2115))
  * Updates the SDK's JSON protocol marshaler to use `json.Decoder` instead of `ioutil.ReadAll`. This reduces the memory unmarshaling JSON payloads by about 50%.
  * Fix [#2114](https://github.com/aws/aws-sdk-go/pull/2114)

Release v1.15.29 (2018-09-06)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Add support for Active X-Ray with API Gateway
* `service/codecommit`: Updates service API and documentation
  * This release adds additional optional fields to the pull request APIs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediaconvert`: Updates service API and documentation
  * This release adds support for Cost Allocation through tagging and also enables adding, editing, and removal of tags from the MediaConvert console.

### SDK Enhancements
* `private/protocol`: Serialization errors will now be wrapped in `awserr.RequestFailure` types ([#2135](https://github.com/aws/aws-sdk-go/pull/2135))
  * Updates the SDK protocol unmarshaling to handle the `SerializationError` as a request failure allowing for inspection of `requestID`s and status codes.

Release v1.15.28 (2018-09-05)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Added support for enabling persistent application settings for a stack. When these settings are enabled, changes that users make to applications and Windows settings are automatically saved after each session and applied to the next session.
* `service/dynamodb`: Updates service API and documentation
  * New feature for Amazon DynamoDB.
* `service/elasticloadbalancing`: Updates service API and documentation
* `service/rds`: Updates service documentation
  * Fix broken links in the RDS CLI Reference to the Aurora User Guide
* `service/s3`: Updates service API, documentation, and examples
  * Parquet input format support added for the SelectObjectContent API

### SDK Enhancements
* `private/model/api`: Add "Deprecated" to deprecated API operation and type doc strings ([#2129](https://github.com/aws/aws-sdk-go/pull/2129))
  * Updates the SDK's code generation to include `Deprecated` in the documentation string for API operations and types that are depercated by a service.
  * Related to [golang/go#10909](https://github.com/golang/go/issues/10909)
  * https://blog.golang.org/godoc-documenting-go-code

### SDK Bugs
* `service/s3/s3manager`: Fix Download Manager with iterator docs ([#2131](https://github.com/aws/aws-sdk-go/pull/2131))
  * Fixes the S3 Download manager's DownloadWithIterator documentation example.
  * Fixes [#1824](https://github.com/aws/aws-sdk-go/issues/1824)

Release v1.15.27 (2018-09-04)
===

### Service Client Updates
* `service/rds`: Updates service documentation
  * Updating cross references for the new Aurora User Guide.
* `service/rekognition`: Updates service API and documentation
  * This release introduces a new API called DescribeCollection to Amazon Rekognition. You can use DescribeCollection to get information about an existing face collection. Given the ID for a face collection, DescribeCollection returns the following information: the number of faces indexed into the collection, the version of the face detection model used by the collection, the Amazon Resource Name (ARN) of the collection and the creation date/time of the collection.

Release v1.15.26 (2018-08-31)
===

### Service Client Updates
* `service/eks`: Updates service API and documentation
* `service/waf`: Updates service API and documentation
  * This change includes support for the WAF FullLogging feature through which Customers will have access to all the logs of requests that are inspected by a WAF WebACL. The new APIs allow Customers to manage association of a WebACL with one or more supported "LogDestination" and redact any request fields from the logs.
* `service/waf-regional`: Updates service API and documentation

Release v1.15.25 (2018-08-30)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Support multiple sources and artifacts for CodeBuild projects.
* `service/sagemaker`: Updates service API and documentation
  * VolumeKmsKeyId now available in Batch Transform Job

Release v1.15.24 (2018-08-29)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * AWS Glue now supports data encryption at rest for ETL jobs and development endpoints. With encryption enabled, when you run ETL jobs, or development endpoints, Glue will use AWS KMS keys to write encrypted data at rest. You can also encrypt the metadata stored in the Glue Data Catalog using keys that you manage with AWS KMS. Additionally, you can use AWS KMS keys to encrypt the logs generated by crawlers and ETL jobs as well as encrypt ETL job bookmarks. Encryption settings for Glue crawlers, ETL jobs, and development endpoints can be configured using the security configurations in Glue. Glue Data Catalog encryption can be enabled via the settings for the Glue Data Catalog.
* `service/mediapackage`: Updates service API and documentation
  * MediaPackage now provides input redundancy. Channels have two ingest endpoints that can receive input from encoders. OriginEndpoints pick one of the inputs receiving content for playback and automatically switch to the other input if the active input stops receiving content. Refer to the User Guide (https://docs.aws.amazon.com/mediapackage/latest/ug/what-is.html) for more details on this feature.
* `service/runtime.sagemaker`: Updates service API and documentation

Release v1.15.23 (2018-08-28)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * New Glue APIs for creating, updating, reading and deleting Data Catalog resource-based policies.
* `service/xray`: Updates service API and documentation
  * Support for new APIs that enable management of sampling rules.

Release v1.15.22 (2018-08-27)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API and documentation
  * This release adds support to create a Stream and Code signing for Amazon FreeRTOS job along with Over-the-air updates.
* `service/iotanalytics`: Updates service API, documentation, and paginators
* `service/redshift`: Updates service documentation
  * Documentation updates for redshift
* `service/signer`: Adds new service
  * AWS Signer is a new feature that allows Amazon FreeRTOS (AFR) Over The Air (OTA) customers to cryptographically sign code using code-signing certificates managed by AWS Certificate Manager.

Release v1.15.21 (2018-08-25)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * AWS Glue now supports data encryption at rest for ETL jobs and development endpoints. With encryption enabled, when you run ETL jobs, or development endpoints, Glue will use AWS KMS keys to write encrypted data at rest. You can also encrypt the metadata stored in the Glue Data Catalog using keys that you manage with AWS KMS. Additionally, you can use AWS KMS keys to encrypt the logs generated by crawlers and ETL jobs as well as encrypt ETL job bookmarks. Encryption settings for Glue crawlers, ETL jobs, and development endpoints can be configured using the security configurations in Glue. Glue Data Catalog encryption can be enabled via the settings for the Glue Data Catalog.

Release v1.15.20 (2018-08-24)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/events`: Updates service API and documentation
  * Added Fargate and NetworkConfiguration support to EcsParameters.

Release v1.15.19 (2018-08-23)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * This release adds support for IoT Thing Group Indexing and Searching functionality.
* `service/iotanalytics`: Updates service API and documentation
* `service/lex-models`: Updates service API
* `service/medialive`: Updates service API, documentation, and paginators
  * Adds two APIs for working with Channel Schedules: BatchUpdateSchedule and DescribeSchedule. These APIs allow scheduling actions for SCTE-35 message insertion and for static image overlays.
* `service/rekognition`: Updates service API, documentation, and examples
  * This release introduces a new API called DescribeCollection to Amazon Rekognition.  You can use DescribeCollection to get information about an existing face collection. Given the ID for a face collection, DescribeCollection returns the following information: the number of faces indexed into the collection, the version of the face detection model used by the collection, the Amazon Resource Name (ARN) of the collection and the creation date/time of the collection.

Release v1.15.18 (2018-08-22)
===

### Service Client Updates
* `service/snowball`: Updates service API
  * Snowball job states allow customers to track the status of the Snowball job. We are launching a new Snowball job state "WithSortingFacility"!  When customer returns the Snowball to AWS, the device first goes to a sorting facility before it reaches an AWS data center.  Many customers have requested us to add a new state to reflect the presence of the device at the sorting facility for better tracking. Today when a customer returns  the Snowball, the state first changes from "InTransitToAWS" to "WithAWS". With the addition of new state, the device will move from "InTransitToAWS" to "WithAWSSortingFacility", and then to "WithAWS".  There are no other changes to the API at this time besides adding this new state.

Release v1.15.17 (2018-08-21)
===

### Service Client Updates
* `service/dlm`: Updates service documentation
* `service/ec2`: Updates service API
  * Added support for T3 Instance type in EC2. To learn more about T3 instances, please see https://aws.amazon.com/ec2/instance-types/t3/
* `service/elasticbeanstalk`: Updates service API, documentation, and examples
  * Elastic Beanstalk adds the "Privileged" field to the "CPUUtilization" type, to support enhanced health reporting in Windows environments.
* `service/rds`: Updates service paginators
  * Adds a paginator for the DescribeDBClusters operation.

Release v1.15.16 (2018-08-20)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * Added SSESpecification block to update-table command which allows users to modify table Server-Side Encryption. Added two new fields (SSEType and KMSMasterKeyId) to SSESpecification block used by create-table and update-table commands. Added new SSEDescription Status value UPDATING.
* `service/mediaconvert`: Updates service API
  * This release fixes backward-incompatible changes from a previous release. That previous release changed non-required job settings to required, which prevented jobs and job templates from merging correctly. The current change removes validation of required settings from the SDK and instead centralizes the validation in the service API. For information on required settings, see the Resources chapter of the AWS Elemental MediaConvert API Reference https://docs.aws.amazon.com/mediaconvert/latest/apireference/resources.html

Release v1.15.15 (2018-08-17)
===

### Service Client Updates
* `service/dax`: Updates service API
  * DAX CreateClusterRequest is updated to include IamRoleArn as a required request parameter.
* `service/sagemaker`: Updates service API and documentation
  * Added an optional boolean parameter, 'DisassociateLifecycleConfig', to the UpdateNotebookInstance operation. When set to true, the lifecycle configuration associated with the notebook instance will be removed, allowing a new one to be set via a new 'LifecycleConfigName' parameter.
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for Secrets Manager

Release v1.15.14 (2018-08-16)
===

### Service Client Updates
* `service/discovery`: Updates service API, documentation, and paginators
  * The Application Discovery Service's Continuous Export APIs allow you to analyze your on-premises server inventory data, including system performance and network dependencies, in Amazon Athena.
* `service/ec2`: Updates service API
  * The 'Attribute' parameter DescribeVolumeAttribute request has been marked as required - the API has always required this parameter, but up until now this wasn't reflected appropriately in the SDK.
* `service/mediaconvert`: Updates service API and documentation
  * Added WriteSegmentTimelineInRepresentation option for Dash Outputs
* `service/redshift`: Updates service API and documentation
  * You can now resize your Amazon Redshift cluster quickly. With the new ResizeCluster action, your cluster is available for read and write operations within minutes
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Inventory now supports groups to quickly see a count of which managed instances are and arent configured to collect one or more Inventory types

Release v1.15.13 (2018-08-15)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * Support for running tests in a custom environment with live logs/video streaming, full test features parity and reduction in overall test execution time.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.15.12 (2018-08-14)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Add batch operations for creating/updating and deleting scheduled scaling actions.
* `service/cloudfront`: Adds new service
  * Lambda@Edge Now Provides You Access to the Request Body for HTTP POST/PUT Processing. With this feature, you can now offload more origin logic to the edge and improve end-user latency. Developers typically use Web/HTML forms or Web Beacons/Bugs as a mechanism to collect data from the end users and then process that data at their origins servers. For example, if you are collecting end user behavior data through a web beacon on your website, you can use this feature to access the user behavior data and directly log it to an Amazon Kinesis Firehose endpoint from the Lambda function, thereby simplifying your origin infrastructure.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/es`: Updates service API, documentation, and paginators
  * Amazon Elasticsearch Service adds support for no downtime, in-place upgrade for Elasticsearch version 5.1 and above.

Release v1.15.11 (2018-08-13)
===

### Service Client Updates
* `service/sagemaker`: Updates service API and documentation
  * SageMaker updated the default endpoint URL to support Private Link via the CLI/SDK.

Release v1.15.10 (2018-08-10)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * This release adds support for a new rate control mode, Quality-Defined Variable Bitrate (QVBR) encoding, includes updates to optimize transcoding performance, and resolves previously reported bugs.
* `service/rds`: Updates service documentation
  * Documentation updates for rds

Release v1.15.9 (2018-08-09)
===

### Service Client Updates
* `service/dax`: Updates service API and documentation
  * Add the SSESpecification field to CreateCluster to allow creation of clusters with server-side encryption, and add the SSEDescription field to DescribeClusters to display the status of server-side encryption for a cluster.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for Docker volumes and Docker volume drivers. Customers can now configure their ECS Tasks to use Docker volumes, enabling stateful and storage-intensive applications to be deployed on ECS.
* `service/rds`: Updates service API, documentation, and examples
  * Launch RDS Aurora Serverless

Release v1.15.8 (2018-08-08)
===

### Service Client Updates
* `service/secretsmanager`: Updates service API and documentation
  * This release introduces a ForceDeleteWithoutRecovery parameter to the DeleteSecret API enabling customers to force the deletion of a secret without any recovery window
* `service/ssm`: Updates service API and documentation
  * AWS Systems Manager Automation is launching two new features for Automation Execution Rate Control based on tags and customized parameter maps. With the first feature, customer can target their resources by specifying a Tag with Key/Value. With the second feature, Parameter maps rate control, customers can benefit from customization of input parameters.

Release v1.15.7 (2018-08-07)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Release semantic versioning feature for CodeBuild
* `service/ec2`: Updates service API and documentation
  * Amazon VPC Flow Logs adds support for delivering flow logs directly to S3
* `service/logs`: Updates service API and documentation
  * Documentation Update
* `service/pinpoint`: Updates service API and documentation
  * This release includes a new batch API call for Amazon Pinpoint which can be used to update endpoints and submit events. This call will accept events from clients such as mobile devices and AWS SDKs. This call will accept requests which has multiple endpoints and multiple events attached to those endpoints in a single call. This call will update the endpoints attached and will ingest events for those endpoints. The response from this call will be a multipart response per endpoint/per event submitted.
* `service/ssm`: Updates service API and documentation
  * Two new filters ExecutionStage and DocumentName will be added to ListCommands so that customers will have more approaches to query their commands.

Release v1.15.6 (2018-08-06)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB Point-in-time recovery (PITR) provides continuous backups of your table data. DynamoDB now supports the ability to self-restore a deleted PITR enabled table. Now, when a table with PITR enabled is deleted, a system backup is automatically created and retained for 35 days (at no additional cost). System backups allow you to restore the deleted PITR enabled table to the state it was just before the point of deletion. For more information, see the Amazon DynamoDB Developer Guide.
* `service/health`: Updates service API, documentation, and paginators
  * Updates the ARN structure vended by AWS Health API. All ARNs will now include the service and type code of the associated event, as vended by DescribeEventTypes.

Release v1.15.5 (2018-08-03)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation

Release v1.15.4 (2018-08-02)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kinesis`: Updates service API, documentation, and paginators
  * This update introduces SubscribeToShard and RegisterStreamConsumer APIs which allows for retrieving records on a data stream over HTTP2 with enhanced fan-out capabilities. With this new feature the Java SDK now supports event streaming natively which will allow you to define payload and exception structures on the client over a persistent connection. For more information, see Developing Consumers with Enhanced Fan-Out in the Kinesis Developer Guide.
* `service/polly`: Updates service API and documentation
  * Amazon Polly enables female voice Aditi to speak Hindi language
* `service/resource-groups`: Updates service API and documentation
* `service/ssm`: Updates service API and documentation
  * This release updates AWS Systems Manager APIs to let customers create and use service-linked roles to register and edit Maintenance Window tasks.

Release v1.15.3 (2018-08-01)
===

### Service Client Updates
* `service/storagegateway`: Updates service API, documentation, and examples
  * AWS Storage Gateway now enables you to create stored volumes with AWS KMS support.
* `service/transcribe`: Updates service API and documentation

Release v1.15.2 (2018-07-31)
===

### Service Client Updates
* `service/connect`: Updates service API and documentation
* `service/es`: Updates service API and documentation
  * Amazon Elasticsearch Service adds support for enabling Elasticsearch error logs, providing you valuable information for troubleshooting your Elasticsearch domains quickly and easily. These logs are published to the Amazon CloudWatch Logs service and can be turned on or off at will.
* `service/iot`: Updates service API and documentation
  * As part of this release we are introducing a new IoT security service, AWS IoT Device Defender, and extending capability of AWS IoT to support Step Functions rule action. The AWS IoT Device Defender is a fully managed service that helps you secure your fleet of IoT devices. For more details on this new service, go to https://aws.amazon.com/iot-device-defender. The Step Functions rule action lets you start an execution of AWS Step Functions state machine from a rule.
* `service/kms`: Updates service API and documentation
  * Added a KeyID parameter to the ListAliases operation. This parameter allows users to list only the aliases that refer to a particular AWS KMS customer master key. All other functionality remains intact.
* `service/mediaconvert`: Updates service API and documentation
  * Fixes an issue with modeled timestamps being labeled with the incorrect format.

### SDK Enhancements
* `service/dynamodb/dynamodbattribute`: Add support for custom struct tag keys([#2054](https://github.com/aws/aws-sdk-go/pull/2054))
  * Adds support for (un)marshaling Go types using custom struct tag keys. The new `MarshalOptions.TagKey` allows the user to specify the tag key to use when (un)marshaling struct fields.  Adds support for struct tags such as `yaml`, `toml`, etc. Support for these keys are in name only, and require the tag value format and values to be supported by the package's Marshalers.

### SDK Bugs
* `aws/endpoints`: Add workaround for AWS China Application Autoscaling ([#2080](https://github.com/aws/aws-sdk-go/pull/2080))
  * Adds workaround to correct the endpoint for Application Autoscaling running in AWS China. This will allow your application to make API calls to Application Autoscaling service in AWS China.
  * Fixes [#2079](https://github.com/aws/aws-sdk-go/issues/2079)
  * Fixes [#1957](https://github.com/aws/aws-sdk-go/issues/1957)
* `private/protocol/xml/xmlutil`: Fix SDK marshaling of empty types ([#2081](https://github.com/aws/aws-sdk-go/pull/2081))
  * Fixes the SDK's marshaling of types without members. This corrects the issue where the SDK would not marshal an XML tag for a type, if that type did not have any exported members.
  * Fixes [#2015](https://github.com/aws/aws-sdk-go/issues/2015)

Release v1.15.1 (2018-07-30)
===

### Service Client Updates
* `service/cloudhsmv2`: Updates service API and documentation
  * This update  to the AWS CloudHSM API adds copy-backup-to-region, which allows you to copy a backup of a cluster from one region to another. The copied backup can be used in the destination region to create a new AWS CloudHSM cluster as a clone of the original cluster.
* `service/directconnect`: Updates service API and documentation
  * 1. awsDeviceV2 field is introduced for Connection/Lag/Interconnect/VirtualInterface/Bgp Objects, while deprecating the awsDevice field for Connection/Lag/Interconnect Objects. 2. region field is introduced for VirtualInterface/Location objects
* `service/glacier`: Updates service API and documentation
  * Documentation updates for glacier
* `service/glue`: Updates service API and documentation
  * Glue Development Endpoints now support association of multiple SSH public keys with a development endpoint.
* `service/iot`: Updates service API and documentation
  * get rid of documentParameters field from CreateJob API
* `service/mq`: Updates service API, documentation, and paginators
  * Modified the CreateBroker, UpdateBroker, and DescribeBroker operations to support integration with Amazon CloudWatch Logs. Added a field to indicate the IP address(es) that correspond to wire-level endpoints of broker instances. While a single-instance broker has one IP address, an active/standby broker for high availability has 2 IP addresses. Added fields to indicate the time when resources were created. Updated documentation for Amazon MQ.
* `service/sagemaker`: Updates service API and documentation
  * Added SecondaryStatusTransitions to DescribeTrainingJob to provide more visibility into SageMaker training job progress and lifecycle.

Release v1.15.0 (2018-07-26)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Add artifacts encryptionDisabled and build encryptionKey.
* `service/ec2`: Updates service API and documentation
  * This change provides the EC2/Spot customers with two new allocation strategies -- LowestN for Spot instances, and OD priority for on-demand instances.
* `service/greengrass`: Updates service documentation
  * Documentation updates for Greengrass Local Resource Access feature
* `service/inspector`: Updates service API and documentation
  * inspector will return ServiceTemporarilyUnavailableException when service is under stress
* `service/redshift`: Updates service API and documentation
  * When we make a new version of Amazon Redshift available, we update your cluster during its maintenance window. By selecting a maintenance track, you control whether we update your cluster with the most recent approved release, or with the previous release. The two values for maintenance track are current and trailing. If you choose the current track, your cluster is updated with the latest approved release. If you choose the trailing track, your cluster is updated with the release that was approved previously.The new API operation for managing maintenance tracks for a cluster is DescribeClusterTracks. In addition, the following API operations have new MaintenanceTrackName parameters:  Cluster,  PendingModifiedValues,  ModifyCluster,  RestoreFromClusterSnapshot,  CreateCluster,  Snapshot
* `service/ssm`: Updates service API and documentation
  * This release updates AWS Systems Manager APIs to allow customers to attach labels to history parameter records and reference history parameter records via labels.  It also adds Parameter Store integration with AWS Secrets Manager to allow referencing and retrieving AWS Secrets Manager's secrets from Parameter Store.

### SDK Features
* `private/model/api`: SDK APIs input/output are not consistently generated ([#2073](https://github.com/aws/aws-sdk-go/pull/2073))
  * Updates the SDK's API code generation to generate the API input and output types consistently. This ensures that the SDK will no longer rename input/output types unexpectedly as in [#2070](https://github.com/aws/aws-sdk-go/issues/2070). SDK API input and output parameter types will always be the API name with a suffix of Input and Output.
  * Existing service APIs which were incorrectly modeled have been preserved to ensure they do not break.
  * Fixes [#2070](https://github.com/aws/aws-sdk-go/issues/2070)

### SDK Enhancements
* `service/s3/s3manager`: Document default behavior for Upload's MaxNumParts ([#2077](https://github.com/aws/aws-sdk-go/issues/2077))
  * Updates the S3 Upload Manager's default behavior for MaxNumParts, and ensures that the Uploader.MaxNumPart's member value is initialized properly if the type was created via struct initialization instead of using the NewUploader function.
  * Fixes [#2015](https://github.com/aws/aws-sdk-go/issues/2015)

### SDK Bugs
* `private/model/api`: SDK APIs input/output are not consistently generated ([#2073](https://github.com/aws/aws-sdk-go/pull/2073))
  * Fixes EFS service breaking change in v1.14.26 where `FileSystemDescription` was incorrectly renamed to `UpdateFileSystemOutput.
  * Fixes [#2070](https://github.com/aws/aws-sdk-go/issues/2070)

Release v1.14.33 (2018-07-25)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * R5 is the successor to R4 in EC2's memory-optimized instance family. R5d is a variant of R5 that has local NVMe SSD. Z1d instances deliver both high compute and high memory. Z1d instances use custom Intel Xeon Scalable Processors running at up to 4.0 GHz, powered by sustained all-core Turbo Boost. They are available in 6 sizes, with up to 48 vCPUs, 384 GiB of memory, and 1.8 TB of local NVMe storage.
* `service/ecs`: Updates service API and documentation
  * This release of Amazon Elastic Container Service (Amazon ECS) introduces support for private registry authentication using AWS Secrets Manager. With private registry authentication, private Docker images can be used in a task definition.
* `service/elasticloadbalancingv2`: Updates service API and documentation

Release v1.14.32 (2018-07-24)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * With this SDK update, APIs UpdateGlobalTableSettings and DescribeGlobalTableSettings now allow consistently configuring AutoScaling settings for a DynamoDB global table. Previously, they would only allow consistently setting IOPS. Now new APIs are being released, existing APIs are being extended.

Release v1.14.31 (2018-07-20)
===

### Service Client Updates
* `service/config`: Updates service API
* `service/dlm`: Updates service documentation

### SDK Enhancements
* `service/s3/s3manager`: Add documentation for sequential download [#2065](https://github.com/aws/aws-sdk-go/pull/2065)
  * Adds documentation for downloading object sequentially with the S3 download manager.

Release v1.14.30 (2018-07-19)
===

### Service Client Updates
* `service/mediapackage`: Updates service API and documentation
  * Adds support for DASH OriginEnpoints with multiple media presentation description periods triggered by presence of SCTE-35 ad markers in Channel input streams.

### SDK Enhancements
* `aws/default`: Add helper to get default provider chain list of credential providers ([#2059](https://github.com/aws/aws-sdk-go/issues/2051))
  * Exports the default provider chain list of providers so it can be used to compose custom chains of credential providers.
  * Fixes [#2051](https://github.com/aws/aws-sdk-go/issues/2051)

Release v1.14.29 (2018-07-18)
===

### Service Client Updates
* `service/iotanalytics`: Updates service API and documentation

Release v1.14.28 (2018-07-17)
===

### Service Client Updates
* `service/comprehend`: Updates service API and documentation
* `service/polly`: Updates service API, documentation, and paginators
  * Amazon Polly adds new API for asynchronous synthesis to S3
* `service/sagemaker`: Updates service API, documentation, and paginators
  * Amazon SageMaker has added the capability for customers to run fully-managed, high-throughput batch transform machine learning models with a simple API call. Batch Transform is ideal for high-throughput workloads and predictions in non-real-time scenarios where data is accumulated over a period of time for offline processing.
* `service/snowball`: Updates service API and documentation
  * AWS Snowball Edge announces the availability of Amazon EC2 compute instances that run on the device. AWS Snowball Edge is a 100-TB ruggedized device built to transfer data into and out of AWS with optional support for local Lambda-based compute functions. With this feature, developers and administrators can run their EC2-based applications on the device providing them with an end to end vertically integrated AWS experience. Designed for data pre-processing, compression, machine learning, and data collection applications, these new instances, called SBE1 instances, feature 1.8 GHz Intel Xeon D processors up to 16 vCPUs, and 32 GB of memory. The SBE1 instance type is available in four sizes and multiple instances can be run on the device at the same time. Customers can now run compute instances using the same Amazon Machine Images (AMIs) that are used in Amazon EC2.

Release v1.14.27 (2018-07-13)
===

### Service Client Updates
* `service/appstream`: Updates service API, documentation, and paginators
  * This API update adds support for sharing AppStream images across AWS accounts within the same region.
* `service/kinesis-video-archived-media`: Updates service API and documentation
* `service/kinesisvideo`: Updates service API and documentation
  * Adds support for HLS video playback of Kinesis Video streams using the KinesisVideo client by including "GET_HLS_STREAMING_SESSION_URL" as an additional APIName parameter in the GetDataEndpoint input.

Release v1.14.26 (2018-07-12)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/codebuild`: Updates service API
  * Update CodeBuild CreateProject API - serviceRole is a required input
* `service/dlm`: Adds new service
* `service/elasticfilesystem`: Updates service API and documentation
  * Amazon EFS now allows you to instantly provision the throughput required for your applications independent of the amount of data stored in your file system, allowing you to optimize throughput for your applications performance needs. Starting today, you can provision the throughput your applications require quickly with a few simple steps using AWS Console, AWS CLI or AWS API to achieve consistent performance.
* `service/elasticmapreduce`: Updates service API and documentation
  * Documentation updates for EMR.
* `service/iam`: Updates service API and documentation
  * SDK release to support IAM delegated administrator feature. The feature lets customers attach permissions boundary to IAM principals. The IAM principals cannot operate exceeding the permission specified in permissions boundary.

### SDK Enhancements
* `aws/credentials/ec2rolecreds`: Avoid unnecessary redirect [#2037](https://github.com/aws/aws-sdk-go/pull/2037)
  * This removes the unnecessary redirect for /latest/meta-data/iam/security-credentials/

Release v1.14.25 (2018-07-11)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Support for fine grain throttling for API gateway.
* `service/ce`: Updates service API and documentation
* `service/s3`: Updates service API and documentation
  * S3 Select support for BZIP2 compressed input files
* `service/ssm`: Updates service API and documentation
  * Support Conditional Branching OnFailure for SSM Automation

Release v1.14.24 (2018-07-10)
===

### Service Client Updates
* `service/appstream`: Updates service API, documentation, paginators, and examples
  * This API update adds pagination to the DescribeImages API to support future features and enhancements.
* `service/codebuild`: Updates service API and documentation
  * API changes to CodeBuild service, support report build status for Github sources
* `service/ec2`: Updates service API and documentation
  * Support CpuOptions field in Launch Template data and allow Launch Template name to contain hyphen.
* `service/glue`: Updates service API and documentation
  * AWS Glue adds the ability to crawl DynamoDB tables.
* `service/opsworks`: Updates service documentation
  * Documentation updates for AWS OpsWorks Stacks.

Release v1.14.23 (2018-07-10)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation

Release v1.14.22 (2018-07-09)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API
* `service/ce`: Updates service API and documentation
* `service/dms`: Updates service API and documentation
  * Added support for DmsTransfer endpoint type and support for re-validate option in table reload API.
* `service/lambda`: Updates service API
  * Add support for .NET Core 2.1 to Lambda.
* `service/transcribe`: Updates service API and documentation

Release v1.14.21 (2018-07-06)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * This release adds support for the following 1) users can specify tags to be attached to queues, presets, and templates during creation of those resources on MediaConvert. 2) users can now view the count of jobs in submitted state and in progressing state on a per queue basis.
* `service/serverlessrepo`: Updates service API and documentation

Release v1.14.20 (2018-07-05)
===

### Service Client Updates
* `service/pinpoint`: Updates service API and documentation
  * This release of the Amazon Pinpoint SDK adds the ability to create complex segments and validate phone numbers for SMS messages. It also adds the ability to get or delete endpoints based on user IDs, remove attributes from endpoints, and list the defined channels for an app.
* `service/sagemaker`: Updates service API and documentation
  * Amazon SageMaker NotebookInstances supports 'Updating' as a NotebookInstanceStatus.  In addition, DescribeEndpointOutput now includes Docker repository digest of deployed Model images.

Release v1.14.19 (2018-07-03)
===

### Service Client Updates
* `service/acm`: Updates service waiters
  * Adds a "CertificateValidated" waiter to AWS Certificate Manager clients, which polls on a new certificate's validation state.
* `service/ec2`: Updates service API, documentation, and examples
  * Added support for customers to tag EC2 Dedicated Hosts
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/redshift`: Updates service API and documentation
  * Feature 1 - On-demand cluster release version - When Amazon Redshift releases a new cluster version, you can choose to upgrade to that version immediately instead of waiting until your next maintenance window. You can also choose to roll back to a previous version. The two new APIs added for managing cluster release version are - ModifyClusterDbRevision, DescribeClusterDbRevisions. Feature 2 - Upgradeable reserved instance - You can now exchange one Reserved Instance for a new Reserved Instance with no changes to the terms of your existing Reserved Instance (term, payment type, or number of nodes). The two new APIs added for managing these upgrades are - AcceptReservedNodeExchange, GetReservedNodeExchangeOfferings.

### SDK Enhancements
* `private/model/api`: Add EventStream support over RPC protocol ([#1998](https://github.com/aws/aws-sdk-go/pull/1998))
  * Adds support for EventStream over JSON PRC protocol. This adds support for the EventStream's initial-response event, EventStream headers, and EventStream modeled exceptions. Also replaces the hand written tests with generated tests for EventStream usage.

Release v1.14.18 (2018-07-02)
===

### Service Client Updates
* `service/ssm`: Updates service API, documentation, and examples
  * Execution History and StartAssociationOnce release for State Manager. Users now have the ability to view association execution history with DescribeAssociationExecutions and DescribeAssociationExecutionTargets. Users can also execute an association by calling StartAssociationOnce.

Release v1.14.17 (2018-06-29)
===

### Service Client Updates
* `service/secretsmanager`: Updates service examples
  * New SDK code snippet examples for the new APIs released for the Resource-based Policy support in Secrets Manager

Release v1.14.16 (2018-06-28)
===

### Service Client Updates
* `service/elasticbeanstalk`: Updates service API, documentation, and examples
  * Elastic Beanstalk adds "Suspended" health status to the EnvironmentHealthStatus enum type and updates document.
* `service/lambda`: Updates service API and documentation
  * Support for SQS as an event source.
* `service/storagegateway`: Updates service API, documentation, and examples
  * AWS Storage Gateway now enables you to use Server Message Block (SMB) protocol  to store and access objects in Amazon Simple Storage Service (S3).

Release v1.14.15 (2018-06-27)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * Unpublish delete-service-linked-role API.
* `service/codepipeline`: Updates service API
  * UpdatePipeline may now throw a LimitExceededException when adding or updating Source Actions that use periodic checks for change detection
* `service/comprehend`: Updates service API, documentation, and paginators
* `service/secretsmanager`: Updates service documentation, paginators, and examples
  * Documentation updates for secretsmanager

### SDK Bugs
* `aws/csm`: Final API Call Attempt events were not being called [#2008](https://github.com/aws/aws-sdk-go/pull/2008)

Release v1.14.14 (2018-06-26)
===

### Service Client Updates
* `service/inspector`: Updates service API, documentation, and paginators
  * Introduce four new APIs to view and preview Exclusions.  Exclusions show which intended security checks are excluded from an assessment, along with reasons and recommendations to fix.  The APIs are CreateExclusionsPreview, GetExclusionsPreview, ListExclusions, and DescribeExclusions.
* `service/s3`: Updates service API and documentation
  * Add AllowQuotedRecordDelimiter to Amazon S3 Select API. Please refer to https://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectSELECTContent.html for usage details.
* `service/secretsmanager`: Updates service API, documentation, paginators, and examples
  * This release adds support for resource-based policies that attach directly to your secrets. These policies provide an additional way to control who can access your secrets and what they can do with them. For more information, see https://docs.aws.amazon.com/secretsmanager/latest/userguide/auth-and-access_resource-based-policies.html in the Secrets Manager User Guide.

Release v1.14.13 (2018-06-22)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/appstream`: Updates service API, documentation, paginators, and examples
  * This API update enables customers to find their VPC private IP address and ENI ID associated with AppStream streaming sessions.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.14.12 (2018-06-21)
===

### Service Client Updates
* `service/clouddirectory`: Adds new service
  * SDK release to support Flexible Schema initiative being carried out by Amazon Cloud Directory. This feature lets customers using new capabilities like: variant typed attributes, dynamic facets and AWS managed Cloud Directory schemas.

Release v1.14.11 (2018-06-21)
===

### Service Client Updates
* `service/macie`: Adds new service
  * Amazon Macie is a security service that uses machine learning to automatically discover, classify, and protect sensitive data in AWS. With this release, we are launching the following Macie HTTPS API operations: AssociateMemberAccount, AssociateS3Resources, DisassociateMemberAccount, DisassociateS3Resources, ListMemberAccounts, ListS3Resources, and UpdateS3Resources. With these API operations you can issue HTTPS requests directly to the service.
* `service/neptune`: Updates service API, documentation, and examples
  * Deprecates the PubliclyAccessible parameter that is not supported by Amazon Neptune.
* `service/ssm`: Updates service API, documentation, and examples
  * Adds Amazon Linux 2 support to Patch Manager

Release v1.14.10 (2018-06-20)
===

### Service Client Updates
* `service/acm-pca`: Updates service API, documentation, paginators, and examples
* `service/medialive`: Updates service API, documentation, and paginators
  * AWS Elemental MediaLive now makes Reserved Outputs and Inputs available through the AWS Management Console and API. You can reserve outputs and inputs with a 12 month commitment in exchange for discounted hourly rates. Pricing is available at https://aws.amazon.com/medialive/pricing/
* `service/rds`: Updates service API, documentation, and examples
  * This release adds a new parameter to specify the retention period for Performance Insights data for RDS instances. You can either choose 7 days (default) or 731 days. For more information, see Amazon RDS Documentation.

### SDK Enhancements
* `service/s3`: Update SelectObjectContent doc example to be on the API not nested type. ([#1991](https://github.com/aws/aws-sdk-go/pull/1991))

### SDK Bugs
* `aws/client`: Fix HTTP debug log EventStream payloads ([#2000](https://github.com/aws/aws-sdk-go/pull/2000))
  * Fixes the SDK's HTTP client debug logging to not log the HTTP response body for EventStreams. This prevents the SDK from buffering a very large amount of data to be logged at once. The aws.LogDebugWithEventStreamBody should be used to log the event stream events.
  * Fixes a bug in the SDK's response logger which will buffer the response body's content if LogDebug is enabled but LogDebugWithHTTPBody is not.

Release v1.14.9 (2018-06-19)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rekognition`: Updates service documentation and examples
  * Documentation updates for rekognition

### SDK Bugs
* `private/model/api`: Update client ServiceName to be based on name of service for new services. ([#1997](https://github.com/aws/aws-sdk-go/pull/1997))
    * Fixes the SDK's `ServiceName` AWS service client package value to be unique based on the service name for new AWS services. Does not change exiting client packages.

Release v1.14.8 (2018-06-15)
===

### Service Client Updates
* `service/mediaconvert`: Updates service API and documentation
  * This release adds language code support according to the ISO-639-3 standard. Custom 3-character language codes are now supported on input and output for both audio and captions.

Release v1.14.7 (2018-06-14)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Support for PRIVATE endpoint configuration type
* `service/dynamodb`: Updates service API and documentation
  * Added two new fields SSEType and KMSMasterKeyArn to SSEDescription block in describe-table output.
* `service/iotanalytics`: Updates service API and documentation

Release v1.14.6 (2018-06-13)
===

### Service Client Updates
* `service/servicecatalog`: Updates service API
  * Introduced new length limitations for few of the product fields.
* `service/ssm`: Updates service API and documentation
  * Added support for new parameter, CloudWatchOutputConfig, for SendCommand API. Users can now have RunCommand output sent to CloudWatchLogs.

Release v1.14.5 (2018-06-12)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * Adding VPCEndpoint support for Remote access. Allows customers to be able to access their private endpoints/services running in their VPC during remote access.
* `service/ecs`: Updates service API and documentation
  * Introduces daemon scheduling capability to deploy one task per instance on selected instances in a cluster.  Adds a "force" flag to the DeleteService API to delete a service without requiring to scale down the number of tasks to zero.

### SDK Enhancements
* `service/rds/rdsutils`: Clean up the rdsutils package and adds a new builder to construct connection strings ([#1985](https://github.com/aws/aws-sdk-go/pull/1985))
    * Rewords documentation to be more useful and provides links to prior setup needed to support authentication tokens. Introduces a builder that allows for building connection strings

### SDK Bugs
* `aws/signer/v4`: Fix X-Amz-Content-Sha256 being in to query for presign ([#1976](https://github.com/aws/aws-sdk-go/pull/1976))
    * Fixes the bug which would allow the X-Amz-Content-Sha256 header to be promoted to the query string when presigning a S3 request. This bug also was preventing users from setting their own sha256 value for a presigned URL. Presigned requests generated with the custom sha256 would of always failed with invalid signature.
    * Fixes [#1974](https://github.com/aws/aws-sdk-go/pull/1974)

Release v1.14.4 (2018-06-11)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * Amazon Cloud Directory now supports optional attributes on Typed Links, giving users the ability to associate and manage data on Typed Links.
* `service/rds`: Updates service documentation
  * Changed lists of valid EngineVersion values to links to the RDS User Guide.
* `service/storagegateway`: Updates service API and documentation
  * AWS Storage Gateway now enables you to create cached volumes and tapes with AWS KMS support.

Release v1.14.3 (2018-06-08)
===

### Service Client Updates
* `service/mediatailor`: Updates service API

Release v1.14.2 (2018-06-07)
===

### Service Client Updates
* `service/medialive`: Updates service API, documentation, and paginators
  * AWS Elemental MediaLive now makes channel log information available through Amazon CloudWatch Logs. You can set up each MediaLive channel with a logging level; when the channel is run, logs will automatically be published to your account on Amazon CloudWatch Logs

Release v1.14.1 (2018-06-05)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/polly`: Updates service API and documentation
  * Amazon Polly adds new French voice - "Lea"
* `service/rds`: Updates service API and documentation
  * This release adds customizable processor features for RDS instances.
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager
* `service/shield`: Updates service API and documentation
  * DDoS Response Team access management for AWS Shield

Release v1.14.0 (2018-06-04)
===

### Service Client Updates
* `service/AWSMigrationHub`: Updates service documentation
* `service/appstream`: Updates service API and documentation
  * Amazon AppStream 2.0 adds support for Google Drive for G Suite. With this feature, customers will be able to connect their G Suite accounts with AppStream 2.0 and enable Google Drive access for an AppStream 2.0 stack. Users of the stack can then link their Google Drive using their G Suite login credentials and use their existing files stored in Drive with their AppStream 2.0 applications. File changes will be synced automatically to Google cloud.
* `service/ec2`: Updates service API and documentation
  * You are now able to use instance storage (up to 3600 GB of NVMe based SSD) on M5 instances, the next generation of EC2's General Purpose instances in us-east-1, us-west-2, us-east-2, eu-west-1 and ca-central-1. M5 instances offer up to 96 vCPUs, 384 GiB of DDR4 instance memory, 25 Gbps in Network bandwidth and improved EBS and Networking bandwidth on smaller instance sizes and provide a balance of compute, memory and network resources for many applications.
* `service/eks`: Adds new service
* `service/mediaconvert`: Updates service API and documentation
  * This release adds the support for Common Media Application Format (CMAF) fragmented outputs, RF64 WAV audio output format, and HEV1 or HEVC1 MP4 packaging types when using HEVC in DASH or CMAF outputs.
* `service/sagemaker`: Updates service API, documentation, and paginators
  * Amazon SageMaker has added the ability to run hyperparameter tuning jobs. A hyperparameter tuning job will create and evaluate multiple training jobs while tuning algorithm hyperparameters, to optimize a customer specified objective metric.

### SDK Features
* Add support for EventStream based APIs (S3 SelectObjectContent) ([#1941](https://github.com/aws/aws-sdk-go/pull/1941))
  * Adds support for EventStream asynchronous APIs such as S3 SelectObjectContents API. This API allows your application to receiving multiple events asynchronously from the API response. Your application receives these events from a channel on the API response.
  * See PR [#1941](https://github.com/aws/aws-sdk-go/pull/1941) for example.
  * Fixes [#1895](https://github.com/aws/aws-sdk-go/issues/1895)

Release v1.13.60 (2018-06-01)
===

### Service Client Updates
* `service/ds`: Updates service API and documentation
  * Added ResetUserPassword API. Customers can now reset their users' passwords without providing the old passwords in Simple AD and Microsoft AD.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API and documentation
  * We are releasing force CancelJob and CancelJobExecution functionalities to customers.
* `service/mediatailor`: Adds new service
* `service/redshift`: Updates service documentation
  * Documentation updates for redshift
* `service/sns`: Updates service API, documentation, and paginators
  * The SNS Subscribe API has been updated with two new optional parameters: Attributes and ReturnSubscriptionArn. Attributes is a map of subscription attributes which can be one or more of: FilterPolicy, DeliveryPolicy, and RawMessageDelivery. ReturnSubscriptionArn is a boolean parameter that overrides the default behavior of returning "pending confirmation" for subscriptions that require confirmation instead of returning the subscription ARN.

### SDK Bugs
* `private/mode/api`: Fix error code constants being generated incorrectly.([#1958](https://github.com/aws/aws-sdk-go/issues/1958))
    * Fixes the SDK's code generation to not modify the error code text value when generating error code constants. This prevents generating error code values which are invalid and will never be sent by the service. This change does not change the error code constant variable name generated by the SDK, only the value of the error code.
    * Fixes [#1856](https://github.com/aws/aws-sdk-go/issues/1856)

Release v1.13.59 (2018-05-31)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.58 (2018-05-30)
===

### Service Client Updates
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/neptune`: Adds new service
  * Amazon Neptune is a fast, reliable graph database service that makes it easy to build and run applications that work with highly connected datasets. Neptune supports popular graph models Property Graph and W3C's Resource Description Frame (RDF), and their respective query languages Apache TinkerPop Gremlin 3.3.2 and SPARQL 1.1.

Release v1.13.57 (2018-05-29)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/pi`: Adds new service

Release v1.13.56 (2018-05-25)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update enables customers to control whether users can transfer data between their local devices and their streaming applications through file uploads and downloads, clipboard operations, or printing to local devices
* `service/config`: Updates service API and documentation
* `service/glue`: Updates service API and documentation
  * AWS Glue now sends a delay notification to Amazon CloudWatch Events when an ETL job runs longer than the specified delay notification threshold.
* `service/iot`: Updates service API
  * We are exposing DELETION_IN_PROGRESS as a new job status in regards to the release of DeleteJob API.

Release v1.13.55 (2018-05-24)
===

### Service Client Updates
* `service/codebuild`: Updates service API
  * AWS CodeBuild Adds Support for Windows Builds.
* `service/elasticloadbalancingv2`: Updates service documentation
* `service/rds`: Updates service API and documentation
  * This release adds CloudWatch Logs integration capabilities to RDS Aurora MySQL clusters
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

### SDK Bugs
* `service/cloudwatchlogs`: Fix pagination with cloudwatchlogs ([#1945](https://github.com/aws/aws-sdk-go/pull/1945))
  * Fixes the SDK's behavior with CloudWatchLogs APIs which return duplicate `NextToken` values to signal end of pagination.
  * Fixes [#1908](https://github.com/aws/aws-sdk-go/pull/1908)

Release v1.13.54 (2018-05-22)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * Amazon Elastic Container Service (ECS) adds service discovery for services that use host or bridged network mode. ECS can now also register instance IPs for active tasks using bridged and host networking with Route 53, making them available via DNS.
* `service/inspector`: Updates service API
  * We are launching the ability to target all EC2 instances. With this launch, resourceGroupArn is now optional for CreateAssessmentTarget and UpdateAssessmentTarget. If resourceGroupArn is not specified, all EC2 instances in the account in the AWS region are included in the assessment target.

Release v1.13.53 (2018-05-21)
===

### Service Client Updates
* `service/cloudformation`: Updates service API and documentation
  * 1) Filtered Update for StackSet based on Accounts and Regions: This feature will allow flexibility for the customers to roll out updates on a StackSet based on specific Accounts and Regions.   2) Support for customized ExecutionRoleName: This feature will allow customers to attach ExecutionRoleName to the StackSet thus ensuring more security and controlling the behavior of any AWS resources in the target accounts.

Release v1.13.52 (2018-05-18)
===

### Service Client Updates
* `service/email`: Updates service documentation
  * Fixed a broken link in the documentation for S3Action.
* `service/iot`: Updates service API and documentation
  * We are releasing DeleteJob and DeleteJobExecution APIs to allow customer to delete resources created using AWS IoT Jobs.

Release v1.13.51 (2018-05-17)
===

### Service Client Updates
* `service/codedeploy`: Updates service documentation
  * Documentation updates for codedeploy
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * You are now able to use instance storage (up to 1800 GB of NVMe based SSD) on C5 instances, the next generation of EC2's compute optimized instances in us-east-1, us-west-2, us-east-2, eu-west-1 and ca-central-1. C5 instances offer up to 72 vCPUs, 144 GiB of DDR4 instance memory, 25 Gbps in Network bandwidth and improved EBS and Networking bandwidth on smaller instance sizes to deliver improved performance for compute-intensive workloads.You can now run bare metal workloads on EC2 with i3.metal instances. As a new instance size belonging to the I3 instance family, i3.metal instances have the same characteristics as other instances in the family, including NVMe SSD-backed instance storage optimized for low latency, very high random I/O performance, and high sequential read throughput. I3.metal instances are powered by 2.3 GHz Intel Xeon processors, offering 36 hyper-threaded cores (72 logical processors), 512 GiB of memory, and 15.2 TB of NVMe SSD-backed instance storage. These instances deliver high networking throughput and lower latency with up to 25 Gbps of aggregate network bandwidth using Elastic Network Adapter (ENA)-based Enhanced Networking.

Release v1.13.50 (2018-05-16)
===

### Service Client Updates
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager
* `service/servicecatalog`: Updates service API and documentation
  * Users can now pass a new option to ListAcceptedPortfolioShares called portfolio-share-type with a value of AWS_SERVICECATALOG in order to access Getting Started Portfolios that contain selected products representing common customer use cases.

Release v1.13.49 (2018-05-15)
===

### Service Client Updates
* `service/config`: Updates service API

Release v1.13.48 (2018-05-14)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Adding support for more override fields for StartBuild API, add support for idempotency token field  for StartBuild API in AWS CodeBuild.
* `service/iot1click-devices`: Adds new service
* `service/iot1click-projects`: Adds new service
* `service/organizations`: Updates service documentation
  * Documentation updates for organizations

Release v1.13.47 (2018-05-10)
===

### Service Client Updates
* `service/firehose`: Updates service API and documentation
  * With this release, Amazon Kinesis Data Firehose can convert the format of your input data from JSON to Apache Parquet or Apache ORC before storing the data in Amazon S3. Parquet and ORC are columnar data formats that save space and enable faster queries compared to row-oriented formats like JSON.

Release v1.13.46 (2018-05-10)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/gamelift`: Updates service API and documentation
  * AutoScaling Target Tracking scaling simplification along with StartFleetActions and StopFleetActions APIs to suspend and resume automatic scaling at will.

Release v1.13.45 (2018-05-10)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Updating the regex for the NumericValue fields.
* `service/ec2`: Updates service API and documentation
  * Enable support for latest flag with Get Console Output
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * Changes to support the Aurora MySQL Backtrack feature.

Release v1.13.44 (2018-05-08)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Enable support for specifying CPU options during instance launch.
* `service/rds`: Updates service documentation
  * Correction to the documentation about copying unencrypted snapshots.

Release v1.13.43 (2018-05-07)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API
* `service/budgets`: Updates service API and documentation
  * "With this release, customers can use AWS Budgets to monitor how much of their Amazon EC2, Amazon RDS, Amazon Redshift, and Amazon ElastiCache instance usage is covered by reservations, and receive alerts when their coverage falls below the threshold they define."
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/es`: Updates service API, documentation, and paginators
  * This change brings support for Reserved Instances to AWS Elasticsearch.
* `service/s3`: Updates service API and documentation
  * Added BytesReturned details for Progress and Stats Events for Amazon S3 Select .

Release v1.13.42 (2018-05-04)
===

### Service Client Updates
* `service/guardduty`: Updates service API, documentation, and paginators
  * Amazon GuardDuty is adding five new API operations for creating and managing filters. For each filter, you can specify a criteria and an action. The action you specify is applied to findings that match the specified criteria.

Release v1.13.41 (2018-05-03)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `service/config`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

Release v1.13.40 (2018-05-02)
===

### Service Client Updates
* `service/acm`: Updates service documentation
  * Documentation updates for acm
* `service/codepipeline`: Updates service API and documentation
  * Added support for webhooks with accompanying definitions as needed in the AWS CodePipeline API Guide.
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 Fleet is a new feature that simplifies the provisioning of Amazon EC2 capacity across different EC2 instance types, Availability Zones, and the On-Demand, Reserved Instance, and Spot Instance purchase models. With a single API call, you can now provision capacity to achieve desired scale, performance, and cost.
* `service/ssm`: Updates service API and documentation
  * Added support for new parameter, DocumentVersion, for SendCommand API. Users can now specify version of SSM document to be executed on the target(s).

Release v1.13.39 (2018-04-30)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/dynamodb`: Updates service API and documentation
  * Adds two new APIs UpdateGlobalTableSettings and DescribeGlobalTableSettings. This update introduces new constraints in the CreateGlobalTable and UpdateGlobalTable APIs . Tables must have the same write capacity units. If Global Secondary Indexes exist then they must have the same write capacity units and key schema.
* `service/guardduty`: Updates service API and documentation
  * You can disable the email notification when inviting GuardDuty members using the disableEmailNotification parameter in the InviteMembers operation.
* `service/route53domains`: Updates service API and documentation
  * This release adds a SubmittedSince attribute to the ListOperations API, so you can list operations that were submitted after a specified date and time.
* `service/sagemaker`: Updates service API and documentation
  * SageMaker has added support for VPC configuration for both Endpoints and Training Jobs. This allows you to connect from the instances running the Endpoint or Training Job to your VPC and any resources reachable in the VPC rather than being restricted to resources that were internet accessible.
* `service/workspaces`: Updates service API and documentation
  * Added new IP Access Control APIs, an API to change the state of a Workspace, and the ADMIN_MAINTENANCE WorkSpace state. With the new IP Access Control APIs, you can now create/delete IP Access Control Groups, add/delete/update rules for IP Access Control Groups, Associate/Disassociate IP Access Control Groups to/from a WorkSpaces Directory, and Describe IP Based Access Control Groups.

Release v1.13.38 (2018-04-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glacier`: Updates service documentation
  * Documentation updates for Glacier to fix a broken link
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

Release v1.13.37 (2018-04-25)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * AWS CodeDeploy has a new exception that indicates when a GitHub token is not valid.
* `service/rekognition`: Updates service documentation
  * Documentation updates for Amazon Rekognition.
* `service/xray`: Updates service API and documentation
  * Added PutEncryptionConfig and GetEncryptionConfig APIs for managing data encryption settings. Use PutEncryptionConfig to configure X-Ray to use an AWS Key Management Service customer master key to encrypt trace data at rest.

Release v1.13.36 (2018-04-24)
===

### Service Client Updates
* `service/elasticbeanstalk`: Updates service API and documentation
  * Support tracking Elastic Beanstalk resources in AWS Config.
* `service/secretsmanager`: Updates service documentation
  * Documentation updates for secretsmanager

Release v1.13.35 (2018-04-23)
===

### Service Client Updates
* `service/autoscaling-plans`: Updates service API and documentation
* `service/iot`: Updates service API and documentation
  * Add IotAnalyticsAction which sends message data to an AWS IoT Analytics channel
* `service/iotanalytics`: Adds new service

### SDK Enhancements
* `aws/endpoints`: Add Get Region description to endpoints package ([#1909](https://github.com/aws/aws-sdk-go/pull/1909))
  * Adds exposing the description field of the endpoints Region struct.
  * Fixes [#1194](https://github.com/aws/aws-sdk-go/issues/1194)

### SDK Bugs
* Fix XML unmarshaler not correctly unmarshaling list of timestamp values ([#1894](https://github.com/aws/aws-sdk-go/pull/1894))
  * Fixes a bug in the XML unmarshaler that would incorrectly try to unmarshal "time.Time" parameters that did not have the struct tag type on them. This would occur for nested lists like CloudWatch's GetMetricDataResponse MetricDataResults timestamp parameters.
  * Fixes [#1892](https://github.com/aws/aws-sdk-go/issues/1892)

Release v1.13.34 (2018-04-20)
===

### Service Client Updates
* `service/firehose`: Updates service API and documentation
  * With this release, Amazon Kinesis Data Firehose allows you to tag your delivery streams. Tags are metadata that you can create and use to manage your delivery streams. For more information about tagging, see AWS Tagging Strategies. For technical documentation, look for the tagging operations in the Amazon Kinesis Firehose API reference.
* `service/medialive`: Updates service API and documentation
  * With AWS Elemental MediaLive you can now output live channels as RTMP (Real-Time Messaging Protocol) and RTMPS as the encrypted version of the protocol (Secure, over SSL/TLS). RTMP is the preferred protocol for sending live streams to popular social platforms which  means you can send live channel content to social and sharing platforms in a secure and reliable way while continuing to stream to your own website, app or network.

Release v1.13.33 (2018-04-19)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/codepipeline`: Updates service API and documentation
  * Added new SourceRevision structure to Execution Summary with accompanying definitions as needed in the AWS CodePipeline API Guide.
* `service/devicefarm`: Updates service API and documentation
  * Adding support for VPCEndpoint feature. Allows customers to be able to access their private endpoints/services running in their VPC during test automation.
* `service/ec2`: Updates service API and documentation
  * Added support for customers to see the time at which a Dedicated Host was allocated or released.
* `service/rds`: Updates service API and documentation
  * The ModifyDBCluster operation now includes an EngineVersion parameter. You can use this to upgrade the engine for a clustered database.
* `service/secretsmanager`: Updates service documentation and examples
  * Documentation updates
* `service/ssm`: Updates service API and documentation
  * Added new APIs DeleteInventory and DescribeInventoryDeletions, for customers to delete their custom inventory data.

Release v1.13.32 (2018-04-10)
===

### Service Client Updates
* `service/dms`: Updates service API and documentation
  * Native Change Data Capture start point and task recovery support in Database Migration Service.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * "AWS Glue now supports timeout values for ETL jobs. With this release, all new ETL jobs have a default timeout value of 48 hours. AWS Glue also now supports the ability to start a schedule or job events trigger when it is created."
* `service/mediapackage`: Updates service API and documentation
  * Adds a new OriginEndpoint package type CmafPackage in MediaPackage. Origin endpoints can now be configured to use the Common Media Application Format (CMAF) media streaming format. This version of CmafPackage only supports HTTP Live Streaming (HLS) manifests with fragmented MP4.
* `service/ssm`: Updates service API and documentation
  * Added TooManyUpdates exception for AddTagsToResource and RemoveTagsFromResource API
* `service/workmail`: Updates service API, documentation, and paginators
  * Amazon WorkMail adds the ability to grant users and groups with "Full Access", "Send As" and "Send on Behalf" permissions on a given mailbox.

Release v1.13.31 (2018-04-09)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * Cloud Directory customers can fetch attributes within a facet on an object with the new GetObjectAttributes API and can fetch attributes from multiple facets or objects with the BatchGetObjectAttributes operation.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.30 (2018-04-06)
===

### Service Client Updates
* `service/batch`: Updates service API and documentation
  * Support for Timeout in SubmitJob and RegisterJobDefinition

Release v1.13.29 (2018-04-05)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/ssm`: Updates service documentation

Release v1.13.28 (2018-04-04)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * AWS Certificate Manager has added support for AWS Certificate Manager Private Certificate Authority (CA). Customers can now request private certificates with the RequestCertificate API, and also export private certificates with the ExportCertificate API.
* `service/acm-pca`: Adds new service
* `service/config`: Updates service API and documentation
* `service/fms`: Adds new service
* `service/monitoring`: Updates service API and documentation
  * The new GetMetricData API enables you to collect batch amounts of metric data and optionally perform math expressions on the data. With one GetMetricData call you can retrieve as many as 100 different metrics and a total of 100,800 data points.
* `service/s3`: Updates service API and documentation
  * ONEZONE_IA storage class stores object data in only one Availability Zone at a lower price than STANDARD_IA. This SDK release provides API support for this new storage class.
* `service/sagemaker`: Updates service API and documentation
  * SageMaker is now supporting many additional instance types in previously supported families for Notebooks, Training Jobs, and Endpoints. Training Jobs and Endpoints now support instances in the m5 family in addition to the previously supported instance families. For specific instance types supported please see the documentation for the SageMaker API.
* `service/secretsmanager`: Adds new service
  * AWS Secrets Manager enables you to easily create and manage the secrets that you use in your customer-facing apps.  Instead of embedding credentials into your source code, you can dynamically query Secrets Manager from your app whenever you need credentials.  You can automatically and frequently rotate your secrets without having to deploy updates to your apps.  All secret values are encrypted when they're at rest with AWS KMS, and while they're in transit with HTTPS and TLS.
* `service/transcribe`: Updates service API, documentation, and paginators

Release v1.13.27 (2018-04-03)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * Added Private Device Management feature. Customers can now manage their private devices efficiently - view their status, set labels and apply profiles on them. Customers can also schedule automated tests and remote access sessions on individual instances in their private device fleet.
* `service/lambda`: Updates service API and documentation
  * added nodejs8.10 as a valid runtime
* `service/translate`: Updates service API and documentation

Release v1.13.26 (2018-04-02)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Amazon API Gateway now supports resource policies for APIs making it easier to set access controls for invoking APIs.
* `service/cloudfront`: Adds new service
  * You can now use a new Amazon CloudFront capability called Field-Level Encryption to further enhance the security of sensitive data, such as credit card numbers or personally identifiable information (PII) like social security numbers. CloudFront's field-level encryption further encrypts sensitive data in an HTTPS form using field-specific encryption keys (which you supply) before a POST request is forwarded to your origin. This ensures that sensitive data can only be decrypted and viewed by certain components or services in your application stack. Field-level encryption is easy to setup. Simply configure the fields that have to be further encrypted by CloudFront using the public keys you specify and you can reduce attack surface for your sensitive data.
* `service/es`: Updates service API and documentation
  * This adds Amazon Cognito authentication support to Kibana.

Release v1.13.25 (2018-03-30)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * Documentation updates for acm
* `service/connect`: Adds new service
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.24 (2018-03-29)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API, documentation, and paginators
* `service/cloudformation`: Updates service API and documentation
  * Enabling resource level permission control for StackSets APIs. Adding support for customers to use customized AdministrationRole to create security boundaries between different users.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/greengrass`: Updates service API and documentation
  * Greengrass APIs now support creating Machine Learning resource types and configuring binary data as the input payload for Greengrass Lambda functions.
* `service/ssm`: Updates service API
  * This Patch Manager release supports creating patch baselines for CentOS.

Release v1.13.23 (2018-03-28)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iam`: Updates service API and documentation
  * Add support for Longer Role Sessions. Four APIs manage max session duration: GetRole, ListRoles, CreateRole, and the new API UpdateRole. The max session duration integer attribute is measured in seconds.
* `service/mturk-requester`: Updates service API and documentation
* `service/sts`: Updates service API and documentation
  * Change utilizes the Max Session Duration attribute introduced for IAM Roles and allows STS customers to request session duration up to the Max Session Duration of 12 hours from AssumeRole based APIs.

Release v1.13.22 (2018-03-27)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * AWS Certificate Manager has added support for customers to disable Certificate Transparency logging on a per-certificate basis.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.21 (2018-03-26)
===

### Service Client Updates
* `service/dynamodb`: Updates service API and documentation
  * Point-in-time recovery (PITR) provides continuous backups of your DynamoDB table data. With PITR, you do not have to worry about creating, maintaining, or scheduling backups. You enable PITR on your table and your backup is available for restore at any point in time from the moment you enable it, up to a maximum of the 35 preceding days. PITR provides continuous backups until you explicitly disable it. For more information, see the Amazon DynamoDB Developer Guide.

Release v1.13.20 (2018-03-23)
===

### Service Client Updates
* `service/rds`: Updates service documentation
  * Documentation updates for RDS

Release v1.13.19 (2018-03-22)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Feedback URL allows admins to provide a feedback link or a survey link for collecting user feedback while streaming sessions. When a feedback link is provided, streaming users will see a "Send Feedback" choice in their streaming session toolbar. On selecting this choice, user will be redirected to the link provided in a new browser tab. If a feedback link is not provided, users will not see the "Send Feedback" option.
* `service/codebuild`: Updates service API and documentation
  * Adding support for branch filtering when using webhooks with AWS CodeBuild.
* `service/ecs`: Updates service API and documentation
  * Amazon Elastic Container Service (ECS) now includes integrated Service Discovery using Route 53 Auto Naming. Customers can now specify a Route 53 Auto Naming service as part of an ECS service. ECS will register task IPs with Route 53, making them available via DNS in your VPC.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

### SDK Bugs
* `aws/endpoints`: Use service metadata for fallback signing name ([#1854](https://github.com/aws/aws-sdk-go/pull/1854))
  * Updates the SDK's endpoint resolution to fallback deriving the service's signing name from the service's modeled metadata in addition the endpoints modeled data.
  * Fixes [#1850](https://github.com/aws/aws-sdk-go/issues/1850)

Release v1.13.18 (2018-03-21)
===

### Service Client Updates
* `service/serverlessrepo`: Updates service documentation

Release v1.13.17 (2018-03-20)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/config`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * Amazon ECS users can now mount a temporary volume in memory in containers and specify the shared memory that a container can use through the use of docker's 'tmpfs' and 'shm-size' features respectively. These fields can be specified under linuxParameters in ContainerDefinition in the Task Definition Template.
* `service/elasticbeanstalk`: Updates service documentation
  * Documentation updates for the new Elastic Beanstalk API DescribeAccountAttributes.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/events`: Updates service API and documentation
  * Added SQS FIFO queue target support
* `service/glue`: Updates service API and documentation
  * API Updates for DevEndpoint: PublicKey is now optional for CreateDevEndpoint. The new DevEndpoint field PrivateAddress will be populated for DevEndpoints associated with a VPC.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive has added support for updating Inputs and Input Security Groups. You can update Input Security Groups at any time and it will update all channels using that Input Security Group. Inputs can be updated as long as they are not attached to a currently running channel.

Release v1.13.16 (2018-03-16)
===

### Service Client Updates
* `service/elasticbeanstalk`: Updates service API and documentation
  * AWS Elastic Beanstalk is launching a new public API named DescribeAccountAttributes which allows customers to access account level attributes. In this release, the API will support quotas for resources such as applications, application versions, and environments.

Release v1.13.15 (2018-03-15)
===

### Service Client Updates
* `service/organizations`: Updates service API and documentation
  * This release adds additional reason codes to improve clarity to exceptions that can occur.
* `service/pinpoint`: Updates service API and documentation
  * With this release, you can delete endpoints from your Amazon Pinpoint projects. Customers can now specify one of their leased dedicated long or short codes to send text messages.
* `service/sagemaker`: Updates service API, documentation, and paginators
  * This release provides support for ml.p3.xlarge instance types for notebook instances.  Lifecycle configuration is now available to customize your notebook instances on start; the configuration can be reused between multiple notebooks.  If a notebook instance is attached to a VPC you can now opt out of internet access that by default is provided by SageMaker.

Release v1.13.14 (2018-03-14)
===

### Service Client Updates
* `service/lightsail`: Updates service API and documentation
  * Updates to existing Lightsail documentation

Release v1.13.13 (2018-03-13)
===

### Service Client Updates
* `service/servicediscovery`: Updates service API and documentation
  * This release adds support for custom health checks, which let you check the health of resources that aren't accessible over the internet. For example, you can use a custom health check when the instance is in an Amazon VPC.

Release v1.13.12 (2018-03-12)
===

### Service Client Updates
* `service/cloudhsmv2`: Updates service API
  * CreateCluster can now take both 8 and 17 character Subnet IDs. DeleteHsm can now take both 8 and 17 character ENI IDs.
* `service/discovery`: Updates service API and documentation
  * Documentation updates for discovery
* `service/iot`: Updates service API and documentation
  * We added new fields to the response of the following APIs. (1) describe-certificate: added new generationId, customerVersion fields (2) describe-ca-certificate: added new generationId, customerVersion and lastModifiedDate fields (3) get-policy: added generationId, creationDate and lastModifiedDate fields
* `service/redshift`: Updates service API and documentation
  * DescribeClusterSnapshotsMessage with ClusterExists flag returns snapshots of existing clusters. Else both existing and deleted cluster snapshots are returned

Release v1.13.11 (2018-03-08)
===

### Service Client Updates
* `service/AWSMigrationHub`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * Amazon Elastic Container Service (ECS) now supports container health checks. Customers can now specify a docker container health check command and parameters in their task definition. ECS will monitor, report and take scheduling action based on the health status.
* `service/pinpoint`: Updates service API and documentation
  * With this release, you can export endpoints from your Amazon Pinpoint projects. You can export a) all of the endpoints assigned to a project or b) the subset of endpoints assigned to a segment.
* `service/rds`: Updates service documentation
  * Documentation updates for RDS

Release v1.13.10 (2018-03-07)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/medialive`: Updates service API and documentation
  * Updates API to model required traits and minimum/maximum constraints.

Release v1.13.9 (2018-03-06)
===

### Service Client Updates
* `service/ecs`: Updates service documentation
  * Documentation updates for Amazon ECS
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.8 (2018-03-01)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Added support for modifying Placement Group association of instances via ModifyInstancePlacement API.
* `service/events`: Updates service API and documentation
  * Added BatchParameters to the PutTargets API
* `service/servicecatalog`: Updates service API and documentation
  * This release of ServiceCatalog adds the DeleteTagOption API.
* `service/ssm`: Updates service API and documentation
  * This Inventory release supports the status message details reported by the last sync for the resource data sync API.
* `service/storagegateway`: Updates service API and documentation
  * AWS Storage Gateway (File) support for two new file share attributes are added.           1. Users can specify the S3 Canned ACL to use for new objects created in the file share.         2. Users can create file shares for requester-pays buckets.

Release v1.13.7 (2018-02-28)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.6 (2018-02-27)
===

### Service Client Updates
* `service/ecr`: Updates service documentation
  * Documentation updates for Amazon ECR.

Release v1.13.5 (2018-02-26)
===

### Service Client Updates
* `service/route53`: Updates service API
  * Added support for creating LBR rules using ap-northeast-3 region.
* `service/sts`: Updates service API and documentation
  * Increased SAMLAssertion parameter size from 50000 to 100000 for AWS Security Token Service AssumeRoleWithSAML API to allow customers to pass bigger SAML assertions

Release v1.13.4 (2018-02-23)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update is to enable customers to copy their Amazon AppStream 2.0 images within and between AWS Regions
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.3 (2018-02-22)
===

### Service Client Updates
* `service/ce`: Updates service API and documentation
* `service/elasticloadbalancingv2`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.13.2 (2018-02-21)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * This release adds an API for adding a file directly to an AWS CodeCommit repository without requiring a Git client.
* `service/ec2`: Updates service API and documentation
  * Adds support for tagging an EBS snapshot as part of the API call that creates the EBS snapshot
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/serverlessrepo`: Updates service API, documentation, and paginators

Release v1.13.1 (2018-02-20)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Amazon EC2 Auto Scaling support for service-linked roles
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/waf`: Updates service API and documentation
  * The new PermissionPolicy APIs in AWS WAF Regional allow customers to attach resource-based policies to their entities.
* `service/waf-regional`: Updates service API and documentation

Release v1.13.0 (2018-02-19)
===

### Service Client Updates
* `service/config`: Updates service API
  * With this release, AWS Config updated the ConfigurationItemStatus enum values. The values prior to this update did not represent appropriate values returned by GetResourceConfigHistory. You must update your code to enumerate the new enum values so this is a breaking change. To map old properties to new properties, use the following descriptions: New discovered resource - Old property: Discovered, New property: ResourceDiscovered. Updated resource - Old property: Ok, New property: OK. Deleted resource - Old property: Deleted, New property: ResourceDeleted or ResourceDeletedNotRecorded. Not-recorded resource - Old property: N/A, New property: ResourceNotRecorded or ResourceDeletedNotRecorded.

Release v1.12.79 (2018-02-16)
===

### Service Client Updates
* `service/rds`: Updates service API and documentation
  * Updates RDS API to indicate whether a DBEngine supports read replicas.

Release v1.12.78 (2018-02-15)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/gamelift`: Updates service API and documentation
  * Updates to allow Fleets to run on On-Demand or Spot instances.
* `service/mediaconvert`: Updates service API and documentation
  * Nielsen ID3 tags can now be inserted into transport stream (TS) and HLS outputs. For more information on Nielsen configuration you can go to https://docs.aws.amazon.com/mediaconvert/latest/apireference/jobs.html#jobs-nielsenconfiguration

Release v1.12.77 (2018-02-14)
===

### Service Client Updates
* `service/appsync`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lex-models`: Updates service API and documentation

### Bug Fixes
* `aws/request`: Fix support for streamed payloads for unsigned body request ([#1778](https://github.com/aws/aws-sdk-go/pull/1778))
  * Fixes the SDK's handling of the SDK's `ReaderSeekerCloser` helper type to not allow erroneous request retries, and request signature generation. This Fix allows you to use the `aws.ReaderSeekerCloser` to wrap an arbitrary `io.Reader` for request `io.ReadSeeker` input parameters. APIs such as lex-runtime's PostContent can now make use of the
ReaderSeekerCloser type without causing unexpected failures.
  * Fixes [#1776](https://github.com/aws/aws-sdk-go/issues/1776)

Release v1.12.76 (2018-02-13)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glacier`: Updates service documentation
  * Documentation updates for glacier
* `service/route53`: Updates service API
  * Added support for creating Private Hosted Zones and metric-based healthchecks in the ap-northeast-3 region for whitelisted customers.

Release v1.12.75 (2018-02-12)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Network interfaces now supply the following additional status of "associated" to better distinguish the current status.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/guardduty`: Updates service API and documentation
  * Added PortProbeAction information to the Action section of the port probe-type finding.
* `service/kms`: Updates service API
  * This release of AWS Key Management Service includes support for InvalidArnException in the RetireGrant API.
* `service/rds`: Updates service documentation
  * Aurora MySQL now supports MySQL 5.7.

Release v1.12.74 (2018-02-09)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Users can now better understand the longer ID opt-in status of their account using the two new APIs DescribeAggregateIdFormat and DescribePrincipalIdFormat
* `service/lex-models`: Updates service API and documentation
* `service/runtime.lex`: Updates service API and documentation

Release v1.12.73 (2018-02-08)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Adds support for allowing customers to provide a redirect URL for a stack. Users will be redirected to the link provided by the admin at the end of their streaming session.
* `service/budgets`: Updates service API and documentation
  * Making budgetLimit and timePeriod optional, and updating budgets docs.
* `service/dms`: Updates service API, documentation, and paginators
  * This release includes the addition of two new APIs: describe replication instance task logs and reboot instance. The first allows user to see how much storage each log for a task on a given instance is occupying. The second gives users the option to reboot the application software on the instance and force a fail over for MAZ instances to test robustness of their integration with our service.
* `service/ds`: Updates service API
  * Updated the regex of some input parameters to support longer EC2 identifiers.
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB now supports server-side encryption using a default service key (alias/aws/dynamodb) from the AWS Key Management Service (KMS). AWS KMS is a service that combines secure, highly available hardware and software to provide a key management system scaled for the cloud. AWS KMS is used via the AWS Management Console or APIs to centrally create encryption keys, define the policies that control how keys can be used, and audit key usage to prove they are being used correctly. For more information, see the Amazon DynamoDB Developer Guide.
* `service/gamelift`: Updates service API and documentation
  * Amazon GameLift FlexMatch added the StartMatchBackfill API.  This API allows developers to add new players to an existing game session using the same matchmaking rules and player data that were used to initially create the session.
* `service/medialive`: Updates service API and documentation
  * AWS Elemental MediaLive has added support for updating channel settings for idle channels. You can now update channel name, channel outputs and output destinations, encoder settings, user role ARN, and input specifications. Channel settings can be updated in the console or with API calls. Please note that running channels need to be stopped before they can be updated. We've also deprecated the 'Reserved' field.
* `service/mediastore`: Updates service API and documentation
  * AWS Elemental MediaStore now supports per-container CORS configuration.

Release v1.12.72 (2018-02-07)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * This new feature will now allow customers to add a customized json classifier. They can specify a json path to indicate the object, array or field of the json documents they'd like crawlers to inspect when they crawl json files.
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * This release of Service Catalog adds SearchProvisionedProducts API and ProvisionedProductPlan APIs.
* `service/servicediscovery`: Updates service API and documentation
  * This release adds support for registering CNAME record types and creating Route 53 alias records that route traffic to Amazon Elastic Load Balancers using Amazon Route 53 Auto Naming APIs.
* `service/ssm`: Updates service API and documentation
  * This Patch Manager release supports configuring Linux repos as part of patch baselines, controlling updates of non-OS security packages and also creating patch baselines for SUSE12

### SDK Enhancements
* `private/model/api`: Add validation to ensure there is no duplication of services in models/apis ([#1758](https://github.com/aws/aws-sdk-go/pull/1758))
    * Prevents the SDK from mistakenly generating code a single service multiple times with different model versions.
* `example/service/ec2/instancesbyRegion`: Fix typos in example ([#1762](https://github.com/aws/aws-sdk-go/pull/1762))
* `private/model/api`: removing SDK API reference crosslinks from input/output shapes. (#1765)

### SDK Bugs
* `aws/session`: Fix bug in session.New not supporting AWS_SDK_LOAD_CONFIG ([#1770](https://github.com/aws/aws-sdk-go/pull/1770))
    * Fixes a bug in the session.New function that was not correctly sourcing the shared configuration files' path.
    * Fixes [#1771](https://github.com/aws/aws-sdk-go/pull/1771)

Release v1.12.71 (2018-02-05)
===

### Service Client Updates
* `service/acm`: Updates service documentation
  * Documentation updates for acm
* `service/cloud9`: Updates service documentation and examples
  * API usage examples for AWS Cloud9.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kinesis`: Updates service API and documentation
  * Using ListShards a Kinesis Data Streams customer or client can get information about shards in a data stream (including meta-data for each shard) without obtaining data stream level information.
* `service/opsworks`: Updates service API, documentation, and waiters
  * AWS OpsWorks Stacks supports EBS encryption and HDD volume types. Also, a new DescribeOperatingSystems API is available, which lists all operating systems supported by OpsWorks Stacks.

Release v1.12.70 (2018-01-26)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * Add InteractionMode in CreateRemoteAccessSession for DirectDeviceAccess feature.
* `service/medialive`: Updates service API and documentation
  * Add InputSpecification to CreateChannel (specification of input attributes is used for channel sizing and affects pricing);  add NotFoundException to DeleteInputSecurityGroups.
* `service/mturk-requester`: Updates service documentation

Release v1.12.69 (2018-01-26)
===

### SDK Bugs
* `models/api`: Fix colliding names [#1754](https://github.com/aws/aws-sdk-go/pull/1754) [#1756](https://github.com/aws/aws-sdk-go/pull/1756)
    * SDK had duplicate folders that were causing errors in some builds.
    * Fixes [#1753](https://github.com/aws/aws-sdk-go/issues/1753)

Release v1.12.68 (2018-01-25)
===

### Service Client Updates
* `service/alexaforbusiness`: Updates service API and documentation
* `service/codebuild`: Updates service API and documentation
  * Adding support for Shallow Clone and GitHub Enterprise in AWS CodeBuild.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/guardduty`: Adds new service
  * Added the missing AccessKeyDetails object to the resource shape.
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports Revision ID on your function versions and aliases, to track and apply conditional updates when you are updating your function version or alias resources.

### SDK Bugs
* `service/s3/s3manager`: Fix check for nil OrigErr in Error() [#1749](https://github.com/aws/aws-sdk-go/issues/1749)
    * S3 Manager's `Error` type did not check for nil of `OrigErr` when calling `Error()`
    * Fixes [#1748](https://github.com/aws/aws-sdk-go/issues/1748)

Release v1.12.67 (2018-01-22)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Add additional costTypes: IncludeDiscount, UseAmortized,  to support finer control for different charges included in a cost budget.

Release v1.12.66 (2018-01-19)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glue`: Updates service API and documentation
  * New AWS Glue DataCatalog APIs to manage table versions and a new feature to skip archiving of the old table version when updating table.
* `service/transcribe`: Adds new service

Release v1.12.65 (2018-01-18)
===

### Service Client Updates
* `service/sagemaker`: Updates service API and documentation
  * CreateTrainingJob and CreateEndpointConfig now supports KMS Key for volume encryption.

Release v1.12.64 (2018-01-17)
===

### Service Client Updates
* `service/autoscaling-plans`: Updates service documentation
* `service/ec2`: Updates service documentation
  * Documentation updates for EC2

Release v1.12.63 (2018-01-17)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/autoscaling-plans`: Adds new service
* `service/rds`: Updates service API and documentation
  * With this release you can now integrate RDS DB instances with CloudWatch Logs. We have added parameters to the operations for creating and modifying DB instances (for example CreateDBInstance) to allow you to take advantage of this capability through the CLI and API. Once you enable this feature, a stream of log events will publish to CloudWatch Logs for each log type you enable.

Release v1.12.62 (2018-01-15)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lambda`: Updates service API and documentation
  * Support for creating Lambda Functions using 'dotnetcore2.0' and 'go1.x'.

Release v1.12.61 (2018-01-12)
===

### Service Client Updates
* `service/glue`: Updates service API and documentation
  * Support is added to generate ETL scripts in Scala which can now be run by  AWS Glue ETL jobs. In addition, the trigger API now supports firing when any conditions are met (in addition to all conditions). Also, jobs can be triggered based on a "failed" or "stopped" job run (in addition to a "succeeded" job run).

Release v1.12.60 (2018-01-11)
===

### Service Client Updates
* `service/elasticloadbalancing`: Updates service API and documentation
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/rds`: Updates service API and documentation
  * Read Replicas for Amazon RDS for MySQL, MariaDB, and PostgreSQL now support Multi-AZ deployments.Amazon RDS Read Replicas enable you to create one or more read-only copies of your database instance within the same AWS Region or in a different AWS Region. Updates made to the source database are asynchronously copied to the Read Replicas. In addition to providing scalability for read-heavy workloads, you can choose to promote a Read Replica to become standalone a DB instance when needed.Amazon RDS Multi-AZ Deployments provide enhanced availability for database instances within a single AWS Region. With Multi-AZ, your data is synchronously replicated to a standby in a different Availability Zone (AZ). In case of an infrastructure failure, Amazon RDS performs an automatic failover to the standby, minimizing disruption to your applications.You can now combine Read Replicas with Multi-AZ as part of a disaster recovery strategy for your production databases. A well-designed and tested plan is critical for maintaining business continuity after a disaster. Since Read Replicas can also be created in different regions than the source database, your Read Replica can be promoted to become the new production database in case of a regional disruption.You can also combine Read Replicas with Multi-AZ for your database engine upgrade process. You can create a Read Replica of your production database instance and upgrade it to a new database engine version. When the upgrade is complete, you can stop applications, promote the Read Replica to a standalone database instance and switch over your applications. Since the database instance is already a Multi-AZ deployment, no additional steps are needed.For more information, see the Amazon RDS User Guide.
* `service/ssm`: Updates service documentation
  * Updates documentation for the HierarchyLevelLimitExceededException error.

Release v1.12.59 (2018-01-09)
===

### Service Client Updates
* `service/kms`: Updates service documentation
  * Documentation updates for AWS KMS

Release v1.12.58 (2018-01-09)
===

### Service Client Updates
* `service/ds`: Updates service API and documentation
  * On October 24 we introduced AWS Directory Service for Microsoft Active Directory (Standard Edition), also known as AWS Microsoft AD (Standard Edition), which is a managed Microsoft Active Directory (AD) that is optimized for small and midsize businesses (SMBs). With this SDK release, you can now create an AWS Microsoft AD directory using API. This enables you to run typical SMB workloads using a cost-effective, highly available, and managed Microsoft AD in the AWS Cloud.

Release v1.12.57 (2018-01-08)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * The AWS CodeDeploy API was updated to support DeleteGitHubAccountToken, a new method that deletes a GitHub account connection.
* `service/discovery`: Updates service API and documentation
  * Documentation updates for AWS Application Discovery Service.
* `service/route53`: Updates service API and documentation
  * This release adds an exception to the CreateTrafficPolicyVersion API operation.

Release v1.12.56 (2018-01-05)
===

### Service Client Updates
* `service/inspector`: Updates service API, documentation, and examples
  * Added 2 new attributes to the DescribeAssessmentTemplate response, indicating the total number of assessment runs and last assessment run ARN (if present.)
* `service/snowball`: Updates service documentation
  * Documentation updates for snowball
* `service/ssm`: Updates service documentation
  * Documentation updates for ssm

Release v1.12.55 (2018-01-02)
===

### Service Client Updates
* `service/rds`: Updates service documentation
  * Documentation updates for rds

Release v1.12.54 (2017-12-29)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/workspaces`: Updates service API and documentation
  * Modify WorkSpaces have been updated with flexible storage and switching of hardware bundles feature. The following configurations have been added to ModifyWorkSpacesProperties: storage and compute. This update provides the capability to configure the storage of a WorkSpace. It also adds the capability of switching hardware bundle of a WorkSpace by specifying an eligible compute (Value, Standard, Performance, Power).

Release v1.12.53 (2017-12-22)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * This release fixes an issue with tags not showing in DescribeAddresses responses.
* `service/ecs`: Updates service API and documentation
  * Amazon ECS users can now set a health check initialization wait period of their ECS services, the services that are associated with an Elastic Load Balancer (ELB) will wait for a period of time before the ELB become healthy. You can now configure this in Create and Update Service.
* `service/inspector`: Updates service API and documentation
  * PreviewAgents API now returns additional fields within the AgentPreview data type. The API now shows the agent health and availability status for all instances included in the assessment target. This allows users to check the health status of Inspector Agents before running an assessment. In addition, it shows the instance ID, hostname, and IP address of the targeted instances.
* `service/sagemaker`: Updates service API and documentation
  * SageMaker Models no longer support SupplementalContainers.  API's that have been affected are CreateModel and DescribeModel.

Release v1.12.52 (2017-12-21)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Adding support allowing AWS CodeBuild customers to select specific curated image versions.
* `service/ec2`: Updates service API and documentation
  * Elastic IP tagging enables you to add key and value metadata to your Elastic IPs so that you can search, filter, and organize them according to your organization's needs.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kinesisanalytics`: Updates service API and documentation
  * Kinesis Analytics now supports AWS Lambda functions as output.

Release v1.12.51 (2017-12-21)
===

### Service Client Updates
* `service/config`: Updates service API
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API and documentation
  * This release adds support for code signed Over-the-air update functionality for Amazon FreeRTOS. Users can now create and schedule Over-the-air updates to their Amazon FreeRTOS devices using these new APIs.

Release v1.12.50 (2017-12-19)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * API Gateway now adds support for calling API with compressed payloads using one of the supported content codings, tagging an API stage for cost allocation, and returning API keys from a custom authorizer for use with a usage plan.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mediastore-data`: Updates service documentation
* `service/route53`: Updates service API and documentation
  * Route 53 added support for a new China (Ningxia) region, cn-northwest-1. You can now specify cn-northwest-1 as the region for latency-based or geoproximity routing. Route 53 also added support for a new EU (Paris) region, eu-west-3. You can now associate VPCs in eu-west-3 with private hosted zones and create alias records that route traffic to resources in eu-west-3.

Release v1.12.49 (2017-12-19)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/monitoring`: Updates service documentation
  * Documentation updates for monitoring

Release v1.12.48 (2017-12-15)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update is to enable customers to add tags to their Amazon AppStream 2.0 resources

Release v1.12.47 (2017-12-14)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Adds support for Cognito Authorizer scopes at the API method level.
* `service/email`: Updates service documentation
  * Added information about the maximum number of transactions per second for the SendCustomVerificationEmail operation.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.46 (2017-12-12)
===

### Service Client Updates
* `service/workmail`: Adds new service
  * Today, Amazon WorkMail released an administrative SDK and enabled AWS CloudTrail integration. With the administrative SDK, you can natively integrate WorkMail with your existing services. The SDK enables programmatic user, resource, and group management through API calls. This means your existing IT tools and workflows can now automate WorkMail management, and third party applications can streamline WorkMail migrations and account actions.

Release v1.12.45 (2017-12-11)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lex-models`: Updates service API and documentation
* `service/sagemaker`: Updates service API
  * CreateModel API Update:  The request parameter 'ExecutionRoleArn' has changed from optional to required.

Release v1.12.44 (2017-12-08)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * This API update is to support the feature that allows customers to automatically consume the latest Amazon AppStream 2.0 agent as and when published by AWS.
* `service/ecs`: Updates service documentation
  * Documentation updates for Windows containers.
* `service/monitoring`: Updates service API and documentation
  * With this launch, you can now create a CloudWatch alarm that alerts you when M out of N datapoints of a metric are breaching your predefined threshold, such as three out of five times in any given five minutes interval or two out of six times in a thirty minutes interval. When M out of N datapoints are not breaching your threshold in an interval, the alarm will be in OK state. Please note that the M datapoints out of N datapoints in an interval can be of any order and does not need to be consecutive. Consequently, you can now get alerted even when the spikes in your metrics are intermittent over an interval.

Release v1.12.43 (2017-12-07)
===

### Service Client Updates
* `service/email`: Updates service API, documentation, and paginators
  * Customers can customize the emails that Amazon SES sends when verifying new identities. This feature is helpful for developers whose applications send email through Amazon SES on behalf of their customers.
* `service/es`: Updates service API and documentation
  * Added support for encryption of data at rest on Amazon Elasticsearch Service using AWS KMS

### SDK Bugs
* `models/apis` Fixes removes colliding sagemaker models folders ([#1686](https://github.com/aws/aws-sdk-go/pull/1686))
  * Fixes Release v1.12.42's SageMaker vs sagemaker model folders.

Release v1.12.42 (2017-12-06)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * Amazon Cloud Directory makes it easier for you to apply schema changes across your directories with in-place schema upgrades. Your directories now remain available while backward-compatible schema changes are being applied, such as the addition of new fields. You also can view the history of your schema changes in Cloud Directory by using both major and minor version identifiers, which can help you track and audit schema versions across directories.
* `service/elasticbeanstalk`: Updates service documentation
  * Documentation updates for AWS Elastic Beanstalk.
* `service/sagemaker`: Adds new service
  * Initial waiters for common SageMaker workflows.

Release v1.12.41 (2017-12-05)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * Add error action API for RulesEngine.
* `service/servicecatalog`: Updates service API and documentation
  * ServiceCatalog has two distinct personas for its use, an "admin" persona (who creates sets of products with different versions and prescribes who has access to them) and an "end-user" persona (who can launch cloud resources based on the configuration data their admins have given them access to).  This API update will allow admin users to deactivate/activate product versions, end-user will only be able to access and launch active product versions.
* `service/servicediscovery`: Adds new service
  * Amazon Route 53 Auto Naming lets you configure public or private namespaces that your microservice applications run in. When instances of the service become available, you can call the Auto Naming API to register the instance, and Amazon Route 53 automatically creates up to five DNS records and an optional health check. Clients that submit DNS queries for the service receive an answer that contains up to eight healthy records.

Release v1.12.40 (2017-12-04)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Add additional costTypes to support finer control for different charges included in a cost budget.
* `service/ecs`: Updates service documentation
  * Documentation updates for ecs

Release v1.12.39 (2017-12-01)
===

### Service Client Updates
* `service/SageMaker`: Updates service waiters

Release v1.12.38 (2017-11-30)
===

### Service Client Updates
* `service/AWSMoneypenny`: Adds new service
* `service/Cloud9`: Adds new service
* `service/Serverless Registry`: Adds new service
* `service/apigateway`: Updates service API, documentation, and paginators
  * Added support Private Integration and VPC Link features in API Gateway. This allows to create an API with the API Gateway private integration, thus providing clients access to HTTP/HTTPS resources in an Amazon VPC from outside of the VPC through a VpcLink resource.
* `service/ec2`: Updates service API and documentation
  * Adds the following updates: 1. Spread Placement ensures that instances are placed on distinct hardware in order to reduce correlated failures. 2. Inter-region VPC Peering allows customers to peer VPCs across different AWS regions without requiring additional gateways, VPN connections or physical hardware
* `service/lambda`: Updates service API and documentation
  * AWS Lambda now supports the ability to set the concurrency limits for individual functions, and increasing memory to 3008 MB.

Release v1.12.37 (2017-11-30)
===

### Service Client Updates
* `service/Ardi`: Adds new service
* `service/autoscaling`: Updates service API and documentation
  * You can now use Auto Scaling with EC2 Launch Templates via the CreateAutoScalingGroup and UpdateAutoScalingGroup APIs.
* `service/ec2`: Updates service API and documentation
  * Adds the following updates: 1. T2 Unlimited enables high CPU performance for any period of time whenever required 2. You are now able to create and launch EC2 m5 and h1 instances
* `service/lightsail`: Updates service API and documentation
  * This release adds support for load balancer and TLS/SSL certificate management. This set of APIs allows customers to create, manage, and scale secure load balanced applications on Lightsail infrastructure. To provide support for customers who manage their DNS on Lightsail, we've added the ability create an Alias A type record which can point to a load balancer DNS name via the CreateDomainEntry API http://docs.aws.amazon.com/lightsail/2016-11-28/api-reference/API_CreateDomainEntry.html.
* `service/ssm`: Updates service API and documentation
  * This release updates AWS Systems Manager APIs to enable executing automations at controlled rate, target resources in a resource groups and execute entire automation at once or single step at a time. It is now also possible to use YAML, in addition to JSON, when creating Systems Manager documents.
* `service/waf`: Updates service API and documentation
  * This release adds support for rule group and managed rule group. Rule group is a container of rules that customers can create, put rules in it and associate the rule group to a WebACL. All rules in a rule group will function identically as they would if each rule was individually associated to the WebACL. Managed rule group is a pre-configured rule group composed by our security partners and made available via the AWS Marketplace. Customers can subscribe to these managed rule groups, associate the managed rule group to their WebACL and start using them immediately to protect their resources.
* `service/waf-regional`: Updates service API and documentation

Release v1.12.36 (2017-11-29)
===

### Service Client Updates
* `service/DeepInsight`: Adds new service
* `service/IronmanRuntime`: Adds new service
* `service/Orchestra - Laser`: Adds new service
* `service/SageMaker`: Adds new service
* `service/Shine`: Adds new service
* `service/archived.kinesisvideo`: Adds new service
* `service/data.kinesisvideo`: Adds new service
* `service/dynamodb`: Updates service API and documentation
  * Amazon DynamoDB now supports the following features: Global Table and On-Demand Backup. Global Table is a fully-managed, multi-region, multi-master database. DynamoDB customers can now write anywhere and read anywhere with single-digit millisecond latency by performing database operations closest to where end users reside. Global Table also enables customers to disaster-proof their applications, keeping them running and data accessible even in the face of natural disasters or region disruptions. Customers can set up Global Table with just a few clicks in the AWS Management Console-no application rewrites required. On-Demand Backup capability is to protect data from loss due to application errors, and meet customers' archival needs for compliance and regulatory reasons. Customers can backup and restore their DynamoDB table data anytime, with a single-click in the AWS management console or a single API call. Backup and restore actions execute with zero impact on table performance or availability. For more information, see the Amazon DynamoDB Developer Guide.
* `service/ecs`: Updates service API and documentation
  * Amazon Elastic Container Service (Amazon ECS) released a new launch type for running containers on a serverless infrastructure. The Fargate launch type allows you to run your containerized applications without the need to provision and manage the backend infrastructure. Just register your task definition and Fargate launches the container for you.
* `service/glacier`: Updates service API and documentation
  * This release includes support for Glacier Select, a new feature that allows you to filter and analyze your Glacier archives and store the results in a user-specified S3 location.
* `service/greengrass`: Updates service API and documentation
  * Greengrass OTA feature allows updating Greengrass Core and Greengrass OTA Agent. Local Resource Access feature allows Greengrass Lambdas to access local resources such as peripheral devices and volumes.
* `service/iot`: Updates service API and documentation
  * This release adds support for a number of new IoT features, including AWS IoT Device Management (Jobs, Fleet Index and Thing Registration), Thing Groups, Policies on Thing Groups, Registry & Job Events, JSON Logs, Fine-Grained Logging Controls, Custom Authorization and AWS Service Authentication Using X.509 Certificates.
* `service/kinesisvideo`: Adds new service
  * Announcing Amazon Kinesis Video Streams, a fully managed video ingestion and storage service. Kinesis Video Streams makes it easy to securely stream video from connected devices to AWS for machine learning, analytics, and processing. You can also stream other time-encoded data like RADAR and LIDAR signals using Kinesis Video Streams.
* `service/rekognition`: Updates service API, documentation, and paginators
  * This release introduces Amazon Rekognition support for video analysis.
* `service/s3`: Updates service API and documentation
  * This release includes support for Glacier Select, a new feature that allows you to filter and analyze your Glacier storage class objects and store the results in a user-specified S3 location.

Release v1.12.35 (2017-11-29)
===

### Service Client Updates
* `service/AmazonMQ`: Adds new service
* `service/GuardDuty`: Adds new service
* `service/apigateway`: Updates service API and documentation
  * Changes related to CanaryReleaseDeployment feature. Enables API developer to create a deployment as canary deployment and test API changes with percentage of customers before promoting changes to all customers.
* `service/batch`: Updates service API and documentation
  * Add support for Array Jobs which allow users to easily submit many copies of a job with a single API call. This change also enhances the job dependency model to support N_TO_N and sequential dependency chains. The ListJobs and DescribeJobs APIs now have the ability to list or describe the status of entire Array Jobs or individual elements within the array.
* `service/cognito-idp`: Updates service API and documentation
* `service/deepdish`: Adds new service
  * AWS AppSync is an enterprise-level, fully managed GraphQL service with real-time data synchronization and offline programming features.
* `service/ec2`: Updates service API and documentation
  * Adds the following updates: 1. You are now able to host a service powered by AWS PrivateLink to provide private connectivity to other VPCs. You are now also able to create endpoints to other services powered by PrivateLink including AWS services, Marketplace Seller services or custom services created by yourself or other AWS VPC customers. 2. You are now able to save launch parameters in a single template that can be used with Auto Scaling, Spot Fleet, Spot, and On Demand instances. 3. You are now able to launch Spot instances via the RunInstances API, using a single additional parameter. RunInstances will response synchronously with an instance ID should capacity be available for your Spot request. 4. A simplified Spot pricing model which delivers low, predictable prices that adjust gradually, based on long-term trends in supply and demand. 5. Amazon EC2 Spot can now hibernate Amazon EBS-backed instances in the event of an interruption, so your workloads pick up from where they left off. Spot can fulfill your request by resuming instances from a hibernated state when capacity is available.
* `service/lambda`: Updates service API and documentation
  * Lambda aliases can now shift traffic between two function versions, based on preassigned weights.

Release v1.12.34 (2017-11-27)
===

### Service Client Updates
* `service/data.mediastore`: Adds new service
* `service/mediaconvert`: Adds new service
  * AWS Elemental MediaConvert is a file-based video conversion service that transforms media into formats required for traditional broadcast and for internet streaming to multi-screen devices.
* `service/medialive`: Adds new service
  * AWS Elemental MediaLive is a video service that lets you easily create live outputs for broadcast and streaming delivery.
* `service/mediapackage`: Adds new service
  * AWS Elemental MediaPackage is a just-in-time video packaging and origination service that lets you format highly secure and reliable live outputs for a variety of devices.
* `service/mediastore`: Adds new service
  * AWS Elemental MediaStore is an AWS storage service optimized for media. It gives you the performance, consistency, and low latency required to deliver live and on-demand video content. AWS Elemental MediaStore acts as the origin store in your video workflow.

Release v1.12.33 (2017-11-22)
===

### Service Client Updates
* `service/acm`: Updates service API and documentation
  * AWS Certificate Manager now supports the ability to import domainless certs and additional Key Types as well as an additional validation method for DNS.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.32 (2017-11-22)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Add support for Access logs and customizable integration timeouts
* `service/cloudformation`: Updates service API and documentation
  * 1) Instance-level parameter overrides (CloudFormation-StackSet feature): This feature will allow the customers to override the template parameters on specific stackInstances. Customers will also have ability to update their existing instances with/without parameter-overrides using a new API "UpdateStackInstances"                                                                                                                                                                                                                                                         2) Add support for SSM parameters in CloudFormation - This feature will allow the customers to use Systems Manager parameters in CloudFormation templates. They will be able to see values for these parameters in Describe APIs.
* `service/codebuild`: Updates service API and documentation
  * Adding support for accessing Amazon VPC resources from AWS CodeBuild, dependency caching and build badges.
* `service/elasticmapreduce`: Updates service API and documentation
  * Enable Kerberos on Amazon EMR.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rekognition`: Updates service API and documentation
  * This release includes updates to Amazon Rekognition for the following APIs. The new DetectText API allows you to recognize and extract textual content from images. Face Model Versioning has been added to operations that deal with face detection.
* `service/shield`: Updates service API, documentation, and paginators
  * The AWS Shield SDK has been updated in order to support Elastic IP address protections, the addition of AttackProperties objects in DescribeAttack responses, and a new GetSubscriptionState operation.
* `service/storagegateway`: Updates service API and documentation
  * AWS Storage Gateway now enables you to get notification when all your files written to your NFS file share have been uploaded to Amazon S3. Storage Gateway also enables guessing of the MIME type for uploaded objects based on file extensions.
* `service/xray`: Updates service API, documentation, and paginators
  * Added automatic pagination support for AWS X-Ray APIs in the SDKs that support this feature.

Release v1.12.31 (2017-11-20)
===

### Service Client Updates
* `service/apigateway`: Updates service documentation
  * Documentation updates for Apigateway
* `service/codecommit`: Updates service API, documentation, and paginators
  * AWS CodeCommit now supports pull requests. You can use pull requests to collaboratively review code changes for minor changes or fixes, major feature additions, or new versions of your released software.
* `service/firehose`: Updates service API and documentation
  * This release includes a new Kinesis Firehose feature that supports Splunk as Kinesis Firehose delivery destination. You can now use Kinesis Firehose to ingest real-time data to Splunk in a serverless, reliable, and salable manner. This release also includes a new feature that allows you to configure Lambda buffer size in Kinesis Firehose data transformation feature. You can now customize the data buffer size before invoking Lambda function in Kinesis Firehose for data transformation. This feature allows you to flexibly trade-off processing and delivery latency with cost and efficiency based on your specific use cases and requirements.
* `service/iis`: Adds new service
  * The AWS Cost Explorer API gives customers programmatic access to AWS cost and usage information, allowing them to perform adhoc queries and build interactive cost management applications that leverage this dataset.
* `service/kinesis`: Updates service API and documentation
  * Customers can now obtain the important characteristics of their stream with DescribeStreamSummary. The response will not include the shard list for the stream but will have the number of open shards, and all the other fields included in the DescribeStream response.
* `service/workdocs`: Updates service API and documentation
  * DescribeGroups API and miscellaneous enhancements

### SDK Bugs
* `aws/client`: Retry delays for throttled exception were not limited to 5 minutes [#1654](https://github.com/aws/aws-sdk-go/pull/1654)
  * Fixes [#1653](https://github.com/aws/aws-sdk-go/issues/1653)

Release v1.12.30 (2017-11-17)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/dms`: Updates service API, documentation, and paginators
  * Support for migration task assessment. Support for data validation after the migration.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * Amazon RDS now supports importing MySQL databases by using backup files from Amazon S3.
* `service/s3`: Updates service API
  * Added ORC to the supported S3 Inventory formats.

### SDK Bugs
* `private/protocol/restjson`: Define JSONValue marshaling for body and querystring ([#1640](https://github.com/aws/aws-sdk-go/pull/1640))
  * Adds support for APIs which use JSONValue for body and querystring targets.
  * Fixes [#1636](https://github.com/aws/aws-sdk-go/issues/1636)

Release v1.12.29 (2017-11-16)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/ec2`: Updates service API
  * You are now able to create and launch EC2 x1e smaller instance sizes
* `service/glue`: Updates service API and documentation
  * API update for AWS Glue. New crawler configuration attribute enables customers to specify crawler behavior. New XML classifier enables classification of XML data.
* `service/opsworkscm`: Updates service API, documentation, and waiters
  * Documentation updates for OpsWorks-cm: a new feature, OpsWorks for Puppet Enterprise, that allows users to create and manage OpsWorks-hosted Puppet Enterprise servers.
* `service/organizations`: Updates service API, documentation, and paginators
  * This release adds APIs that you can use to enable and disable integration with AWS services designed to work with AWS Organizations. This integration allows the AWS service to perform operations on your behalf on all of the accounts in your organization. Although you can use these APIs yourself, we recommend that you instead use the commands provided in the other AWS service to enable integration with AWS Organizations.
* `service/route53`: Updates service API and documentation
  * You can use Route 53's GetAccountLimit/GetHostedZoneLimit/GetReusableDelegationSetLimit APIs to view your current limits (including custom set limits) on Route 53 resources such as hosted zones and health checks. These APIs also return the number of each resource you're currently using to enable comparison against your current limits.

Release v1.12.28 (2017-11-15)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * 1. Extended GetDocumentationParts operation to support retrieving documentation parts resources without contents.  2. Added hosted zone ID in the custom domain response.
* `service/email`: Updates service API, documentation, and examples
  * SES launches Configuration Set Reputation Metrics and Email Pausing Today, two features that build upon the capabilities of the reputation dashboard. The first is the ability to export reputation metrics for individual configuration sets. The second is the ability to temporarily pause email sending, either at the configuration set level, or across your entire Amazon SES account.
* `service/polly`: Updates service API
  * Amazon Polly adds Korean language support with new female voice - "Seoyeon" and new Indian English female voice - "Aditi"
* `service/states`: Updates service API and documentation
  * You can now use the UpdateStateMachine API to update your state machine definition and role ARN. Existing executions will continue to use the previous definition and role ARN. You can use the DescribeStateMachineForExecution API to determine which state machine definition and role ARN is associated with an execution

Release v1.12.27 (2017-11-14)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * Added new mode for Task Networking in ECS, called awsvpc mode. Mode configuration parameters to be passed in via awsvpcConfiguration. Updated APIs now use/show this new mode - RegisterTaskDefinition, CreateService, UpdateService, RunTask, StartTask.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lightsail`: Updates service API and documentation
  * Lightsail now supports attached block storage, which allows you to scale your applications and protect application data with additional SSD-backed storage disks. This feature allows Lightsail customers to attach secure storage disks to their Lightsail instances and manage their attached disks, including creating and deleting disks, attaching and detaching disks from instances, and backing up disks via snapshot.
* `service/route53`: Updates service API and documentation
  * When a Route 53 health check or hosted zone is created by a linked AWS service, the object now includes information about the service that created it. Hosted zones or health checks that are created by a linked service can't be updated or deleted using Route 53.
* `service/ssm`: Updates service API and documentation
  * EC2 Systems Manager GetInventory API adds support for aggregation.

### SDK Enhancements
* `aws/request`: Remove default port from HTTP host header ([#1618](https://github.com/aws/aws-sdk-go/pull/1618))
  * Updates the SDK to automatically remove default ports based on the URL's scheme when setting the HTTP Host header's value.
  * Fixes [#1537](https://github.com/aws/aws-sdk-go/issues/1537)

Release v1.12.26 (2017-11-09)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Introduces the following features: 1. Create a default subnet in an Availability Zone if no default subnet exists. 2. Spot Fleet integrates with Elastic Load Balancing to enable you to attach one or more load balancers to a Spot Fleet request. When you attach the load balancer, it automatically registers the instance in the Spot Fleet to the load balancers which distributes incoming traffic across the instances.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.25 (2017-11-08)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/batch`: Updates service documentation
  * Documentation updates for AWS Batch.
* `service/ec2`: Updates service API and documentation
  * AWS PrivateLink for Amazon Services - Customers can now privately access Amazon services from their Amazon Virtual Private Cloud (VPC), without using public IPs, and without requiring the traffic to traverse across the Internet.
* `service/elasticache`: Updates service API and documentation
  * This release adds online resharding for ElastiCache for Redis offering, providing the ability to add and remove shards from a running cluster. Developers can now dynamically scale-out or scale-in their Redis cluster workloads to adapt to changes in demand. ElastiCache will resize the cluster by adding or removing shards and redistribute hash slots uniformly across the new shard configuration, all while the cluster continues to stay online and serves requests.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.24 (2017-11-07)
===

### Service Client Updates
* `service/elasticloadbalancingv2`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * DescribeOrderableDBInstanceOptions now returns the minimum and maximum allowed values for storage size, total provisioned IOPS, and provisioned IOPS per GiB for a DB instance.
* `service/s3`: Updates service API, documentation, and examples
  * This releases adds support for 4 features: 1. Default encryption for S3 Bucket, 2. Encryption status in inventory and Encryption support for inventory.  3. Cross region replication of KMS-encrypted objects, and 4. ownership overwrite for CRR.

Release v1.12.23 (2017-11-07)
===

### Service Client Updates
* `service/api.pricing`: Adds new service
* `service/ec2`: Updates service API
  * You are now able to create and launch EC2 C5 instances, the next generation of EC2's compute-optimized instances, in us-east-1, us-west-2 and eu-west-1. C5 instances offer up to 72 vCPUs, 144 GiB of DDR4 instance memory, 25 Gbps in Network bandwidth and improved EBS and Networking bandwidth on smaller instance sizes to deliver improved performance for compute-intensive workloads.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kms`: Updates service API, documentation, and examples
  * Documentation updates for AWS KMS.
* `service/organizations`: Updates service documentation
  * This release updates permission statements for several API operations, and corrects some other minor errors.
* `service/states`: Updates service API, documentation, and paginators
  * Documentation update.

Release v1.12.22 (2017-11-03)
===

### Service Client Updates
* `service/ecs`: Updates service API and documentation
  * Amazon ECS users can now add devices to their containers and enable init process in containers through the use of docker's 'devices' and 'init' features. These fields can be specified under linuxParameters in ContainerDefinition in the Task Definition Template.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.21 (2017-11-02)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * This release supports creating and managing Regional and Edge-Optimized API endpoints.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

### SDK Bugs
* `aws/request`: Fix bug in request presign creating invalid URL ([#1624](https://github.com/aws/aws-sdk-go/pull/1624))
  * Fixes a bug the Request Presign and PresignRequest methods that would allow a invalid expire duration as input. A expire time of 0 would be interpreted by the SDK to generate a normal request signature, not a presigned URL. This caused the returned URL unusable.
  * Fixes [#1617](https://github.com/aws/aws-sdk-go/issues/1617)

Release v1.12.20 (2017-11-01)
===

### Service Client Updates
* `service/acm`: Updates service documentation
  * Documentation updates for ACM
* `service/cloudhsmv2`: Updates service documentation
  * Minor documentation update for AWS CloudHSM (cloudhsmv2).
* `service/directconnect`: Updates service API and documentation
  * AWS DirectConnect now provides support for Global Access for Virtual Private Cloud (VPC) via a new feature called Direct Connect Gateway. A Direct Connect Gateway will allow you to group multiple Direct Connect Private Virtual Interfaces (DX-VIF) and Private Virtual Gateways (VGW) from different AWS regions (but belonging to the same AWS Account) and pass traffic from any DX-VIF to any VPC in the grouping.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

### SDK Enhancements
* `aws/client`: Adding status code 429 to throttlable status codes in default retryer (#1621)

Release v1.12.19 (2017-10-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.18 (2017-10-26)
===

### Service Client Updates
* `service/cloudfront`: Updates service API and documentation
  * You can now specify additional options for MinimumProtocolVersion, which controls the SSL/TLS protocol that CloudFront uses to communicate with viewers. The minimum protocol version that you choose also determines the ciphers that CloudFront uses to encrypt the content that it returns to viewers.
* `service/ec2`: Updates service API
  * You are now able to create and launch EC2 P3 instance, next generation GPU instances, optimized for machine learning and high performance computing applications. With up to eight NVIDIA Tesla V100 GPUs, P3 instances provide up to one petaflop of mixed-precision, 125 teraflops of single-precision, and 62 teraflops of double-precision floating point performance, as well as a 300 GB/s second-generation NVLink interconnect that enables high-speed, low-latency GPU-to-GPU communication. P3 instances also feature up to 64 vCPUs based on custom Intel Xeon E5 (Broadwell) processors, 488 GB of DRAM, and 25 Gbps of dedicated aggregate network bandwidth using the Elastic Network Adapter (ENA).
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.17 (2017-10-24)
===

### Service Client Updates
* `service/config`: Updates service API
* `service/elasticache`: Updates service API, documentation, and examples
  * Amazon ElastiCache for Redis today announced support for data encryption both for data in-transit and data at-rest. The new encryption in-transit functionality enables ElastiCache for Redis customers to encrypt data for all communication between clients and Redis engine, and all intra-cluster Redis communication. The encryption at-rest functionality allows customers to encrypt their S3 based backups. Customers can begin using the new functionality by simply enabling this functionality via AWS console, and a small configuration change in their Redis clients. The ElastiCache for Redis service automatically manages life cycle of the certificates required for encryption, including the issuance, renewal and expiration of certificates. Additionally, as part of this launch, customers will gain the ability to start using the Redis AUTH command that provides an added level of authentication.
* `service/glue`: Adds new service
  * AWS Glue: Adding a new API, BatchStopJobRun, to stop one or more job runs for a specified Job.
* `service/pinpoint`: Updates service API and documentation
  * Added support for APNs VoIP messages. Added support for collapsible IDs, message priority, and TTL for APNs and FCM/GCM.

Release v1.12.16 (2017-10-23)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/organizations`: Updates service API and documentation
  * This release supports integrating other AWS services with AWS Organizations through the use of an IAM service-linked role called AWSServiceRoleForOrganizations. Certain operations automatically create that role if it does not already exist.

Release v1.12.15 (2017-10-20)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Adding pagination support for DescribeSecurityGroups for EC2 Classic and VPC Security Groups

Release v1.12.14 (2017-10-19)
===

### Service Client Updates
* `service/sqs`: Updates service API and documentation
  * Added support for tracking cost allocation by adding, updating, removing, and listing the metadata tags of Amazon SQS queues.
* `service/ssm`: Updates service API and documentation
  * EC2 Systems Manager versioning support for Parameter Store. Also support for referencing parameter versions in SSM Documents.

Release v1.12.13 (2017-10-18)
===

### Service Client Updates
* `service/lightsail`: Updates service API and documentation
  * This release adds support for Windows Server-based Lightsail instances. The GetInstanceAccessDetails API now returns the password of your Windows Server-based instance when using the default key pair. GetInstanceAccessDetails also returns a PasswordData object for Windows Server instances containing the ciphertext and keyPairName. The Blueprint data type now includes a list of platform values (LINUX_UNIX or WINDOWS). The Bundle data type now includes a list of SupportedPlatforms values (LINUX_UNIX or WINDOWS).

Release v1.12.12 (2017-10-17)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/es`: Updates service API and documentation
  * This release adds support for VPC access to Amazon Elasticsearch Service.
  * This release adds support for VPC access to Amazon Elasticsearch Service.

Release v1.12.11 (2017-10-16)
===

### Service Client Updates
* `service/cloudhsm`: Updates service API and documentation
  * Documentation updates for AWS CloudHSM Classic.
* `service/ec2`: Updates service API and documentation
  * You can now change the tenancy of your VPC from dedicated to default with a single API operation. For more details refer to the documentation for changing VPC tenancy.
* `service/es`: Updates service API and documentation
  * AWS Elasticsearch adds support for enabling slow log publishing. Using slow log publishing options customers can configure and enable index/query slow log publishing of their domain to preferred AWS Cloudwatch log group.
* `service/rds`: Updates service API and waiters
  * Adds waiters for DBSnapshotAvailable and DBSnapshotDeleted.
* `service/waf`: Updates service API and documentation
  * This release adds support for regular expressions as match conditions in rules, and support for geographical location by country of request IP address as a match condition in rules.
* `service/waf-regional`: Updates service API and documentation

Release v1.12.10 (2017-10-12)
===

### Service Client Updates
* `service/codecommit`: Updates service API and documentation
  * This release includes the DeleteBranch API and a change to the contents of a Commit object.
* `service/dms`: Updates service API and documentation
  * This change includes addition of new optional parameter to an existing API
* `service/elasticbeanstalk`: Updates service API and documentation
  * Added the ability to add, delete or update Tags
* `service/polly`: Updates service API
  * Amazon Polly exposes two new voices: "Matthew" (US English) and "Takumi" (Japanese)
* `service/rds`: Updates service API and documentation
  * You can now call DescribeValidDBInstanceModifications to learn what modifications you can make to your DB instance. You can use this information when you call ModifyDBInstance.

Release v1.12.9 (2017-10-11)
===

### Service Client Updates
* `service/ecr`: Updates service API, documentation, and paginators
  * Adds support for new API set used to manage Amazon ECR repository lifecycle policies. Amazon ECR lifecycle policies enable you to specify the lifecycle management of images in a repository. The configuration is a set of one or more rules, where each rule defines an action for Amazon ECR to apply to an image. This allows the automation of cleaning up unused images, for example expiring images based on age or status. A lifecycle policy preview API is provided as well, which allows you to see the impact of a lifecycle policy on an image repository before you execute it
* `service/email`: Updates service API and documentation
  * Added content related to email template management and templated email sending operations.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.12.8 (2017-10-10)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * This release includes updates to AWS Virtual Private Gateway.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `service/opsworkscm`: Updates service API and documentation
  * Provide engine specific information for node associations.

Release v1.12.7 (2017-10-06)
===

### Service Client Updates
* `service/sqs`: Updates service documentation
  * Documentation updates regarding availability of FIFO queues and miscellaneous corrections.

Release v1.12.6 (2017-10-05)
===

### Service Client Updates
* `service/redshift`: Updates service API and documentation
  * DescribeEventSubscriptions API supports tag keys and tag values as request parameters.

Release v1.12.5 (2017-10-04)
===

### Service Client Updates
* `service/kinesisanalytics`: Updates service API and documentation
  * Kinesis Analytics now supports schema discovery on objects in S3. Additionally, Kinesis Analytics now supports input data preprocessing through Lambda.
* `service/route53domains`: Updates service API and documentation
  * Added a new API that checks whether a domain name can be transferred to Amazon Route 53.

### SDK Bugs
* `service/s3/s3crypto`: Correct PutObjectRequest documentation ([#1568](https://github.com/aws/aws-sdk-go/pull/1568))
  * s3Crypto's PutObjectRequest docstring example was using an incorrect value. Corrected the type used in the example.

Release v1.12.4 (2017-10-03)
===

### Service Client Updates
* `service/ec2`: Updates service API, documentation, and waiters
  * This release includes service updates to AWS VPN.
* `service/ssm`: Updates service API and documentation
  * EC2 Systems Manager support for tagging SSM Documents. Also support for tag-based permissions to restrict access to SSM Documents based on these tags.

Release v1.12.3 (2017-10-02)
===

### Service Client Updates
* `service/cloudhsm`: Updates service documentation and paginators
  * Documentation updates for CloudHSM

Release v1.12.2 (2017-09-29)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Includes APIs for managing and accessing image builders, and deleting images.
* `service/codebuild`: Updates service API and documentation
  * Adding support for Building GitHub Pull Requests in AWS CodeBuild
* `service/mturk-requester`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * This release flags the HandshakeParty structure's Type and Id fields as 'required'. They effectively were required in the past, as you received an error if you did not include them. This is now reflected at the API definition level.
* `service/route53`: Updates service API and documentation
  * This change allows customers to reset elements of health check.

### SDK Bugs
* `private/protocol/query`: Fix query protocol handling of nested byte slices ([#1557](https://github.com/aws/aws-sdk-go/issues/1557))
  * Fixes the query protocol to correctly marshal nested []byte values of API operations.
* `service/s3`: Fix PutObject and UploadPart API to include ContentMD5 field ([#1559](https://github.com/aws/aws-sdk-go/pull/1559))
  * Fixes the SDK's S3 PutObject and UploadPart API code generation to correctly render the ContentMD5 field into the associated input types for these two API operations.
  * Fixes [#1553](https://github.com/aws/aws-sdk-go/pull/1553)

Release v1.12.1 (2017-09-27)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/pinpoint`: Updates service API and documentation
  * Added two new push notification channels: Amazon Device Messaging (ADM) and, for push notification support in China, Baidu Cloud Push. Added support for APNs auth via .p8 key file. Added operation for direct message deliveries to user IDs, enabling you to message an individual user on multiple endpoints.

Release v1.12.0 (2017-09-26)
===

### SDK Bugs
* `API Marshaler`: Revert REST JSON and XML protocol marshaler improvements
  * Bug [#1550](https://github.com/aws/aws-sdk-go/issues/1550) identified a missed condition in the Amazon Route 53 RESTXML protocol marshaling causing requests to that service to fail. Reverting the marshaler improvements until the bug can be fixed.

Release v1.11.0 (2017-09-26)
===

### Service Client Updates
* `service/cloudformation`: Updates service API and documentation
  * You can now prevent a stack from being accidentally deleted by enabling termination protection on the stack. If you attempt to delete a stack with termination protection enabled, the deletion fails and the stack, including its status, remains unchanged. You can enable termination protection on a stack when you create it. Termination protection on stacks is disabled by default. After creation, you can set termination protection on a stack whose status is CREATE_COMPLETE, UPDATE_COMPLETE, or UPDATE_ROLLBACK_COMPLETE.

### SDK Features
* Add dep Go dependency management metadata files (#1544)
  * Adds the Go `dep` dependency management metadata files to the SDK.
  * Fixes [#1451](https://github.com/aws/aws-sdk-go/issues/1451)
  * Fixes [#634](https://github.com/aws/aws-sdk-go/issues/634)
* `service/dynamodb/expression`: Add expression building utility for DynamoDB ([#1527](https://github.com/aws/aws-sdk-go/pull/1527))
  * Adds a new package, expression, to the SDK providing builder utilities to create DynamoDB expressions safely taking advantage of type safety.
* `API Marshaler`: Add generated marshalers for RESTXML protocol ([#1409](https://github.com/aws/aws-sdk-go/pull/1409))
  * Updates the RESTXML protocol marshaler to use generated code instead of reflection for REST XML based services.
* `API Marshaler`: Add generated marshalers for RESTJSON protocol ([#1547](https://github.com/aws/aws-sdk-go/pull/1547))
  * Updates the RESTJSON protocol marshaler to use generated code instead of reflection for REST JSON based services.

### SDK Enhancements
* `private/protocol`: Update format of REST JSON and XMl benchmarks ([#1546](https://github.com/aws/aws-sdk-go/pull/1546))
  * Updates the format of the REST JSON and XML benchmarks to be readable. RESTJSON benchmarks were updated to more accurately bench building of the protocol.

Release v1.10.51 (2017-09-22)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation
  * Amazon ECS users can now add and drop Linux capabilities to their containers through the use of docker's cap-add and cap-drop features. Customers can specify the capabilities they wish to add or drop for each container in their task definition.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service documentation
  * Documentation updates for rds

Release v1.10.50 (2017-09-21)
===

### Service Client Updates
* `service/budgets`: Updates service API
  * Including "DuplicateRecordException" in UpdateNotification and UpdateSubscriber.
* `service/ec2`: Updates service API and documentation
  * Add EC2 APIs to copy Amazon FPGA Images (AFIs) within the same region and across multiple regions, delete AFIs, and modify AFI attributes. AFI attributes include name, description and granting/denying other AWS accounts to load the AFI.
* `service/logs`: Updates service API and documentation
  * Adds support for associating LogGroups with KMS Keys.

### SDK Bugs
* Fix greengrass service model being duplicated with different casing. ([#1541](https://github.com/aws/aws-sdk-go/pull/1541))
  * Fixes [#1540](https://github.com/aws/aws-sdk-go/issues/1540)
  * Fixes [#1539](https://github.com/aws/aws-sdk-go/issues/1539)

Release v1.10.49 (2017-09-20)
===

### Service Client Updates
* `service/Greengrass`: Adds new service
* `service/appstream`: Updates service API and documentation
  * API updates for supporting On-Demand fleets.
* `service/codepipeline`: Updates service API and documentation
  * This change includes a PipelineMetadata object that is part of the output from the GetPipeline API that includes the Pipeline ARN, created, and updated timestamp.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * Introduces the --option-group-name parameter to the ModifyDBSnapshot CLI command. You can specify this parameter when you upgrade an Oracle DB snapshot. The same option group considerations apply when upgrading a DB snapshot as when upgrading a DB instance.  For more information, see http://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_UpgradeDBInstance.Oracle.html#USER_UpgradeDBInstance.Oracle.OGPG.OG
* `service/runtime.lex`: Updates service API and documentation

Release v1.10.48 (2017-09-19)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Fixed bug in EC2 clients preventing ElasticGpuSet from being set.

### SDK Enhancements
* `aws/credentials`: Add EnvProviderName constant. ([#1531](https://github.com/aws/aws-sdk-go/issues/1531))
  * Adds the "EnvConfigCredentials" string literal as EnvProviderName constant.
  * Fixes [#1444](https://github.com/aws/aws-sdk-go/issues/1444)

Release v1.10.47 (2017-09-18)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 now lets you opt for Spot instances to be stopped in the event of an interruption instead of being terminated.  Your Spot request can be fulfilled again by restarting instances from a previously stopped state, subject to availability of capacity at or below your preferred price.  When you submit a persistent Spot request, you can choose from "terminate" or "stop" as the instance interruption behavior.  Choosing "stop" will shutdown your Spot instances so you can continue from this stopped state later on.  This feature is only available for instances with Amazon EBS volume as their root device.
* `service/email`: Updates service API and documentation
  * Amazon Simple Email Service (Amazon SES) now lets you customize the domains used for tracking open and click events. Previously, open and click tracking links referred to destinations hosted on domains operated by Amazon SES. With this feature, you can use your own branded domains for capturing open and click events.
* `service/iam`: Updates service API and documentation
  * A new API, DeleteServiceLinkedRole, submits a service-linked role deletion request and returns a DeletionTaskId, which you can use to check the status of the deletion.

Release v1.10.46 (2017-09-15)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Add a new enum "REQUEST" to '--type <value>' field in the current create-authorizer API, and make "identitySource" optional.

Release v1.10.45 (2017-09-14)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Supporting Parameter Store in environment variables for AWS CodeBuild
* `service/organizations`: Updates service documentation
  * Documentation updates for AWS Organizations
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * This release of Service Catalog adds API support to copy products.

Release v1.10.44 (2017-09-13)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Customers can create Life Cycle Hooks at the time of creating Auto Scaling Groups through the CreateAutoScalingGroup API
* `service/batch`: Updates service documentation and examples
  * Documentation updates for batch
* `service/ec2`: Updates service API
  * You are now able to create and launch EC2 x1e.32xlarge instance, a new EC2 instance in the X1 family, in us-east-1, us-west-2, eu-west-1, and ap-northeast-1. x1e.32xlarge offers 128 vCPUs, 3,904 GiB of DDR4 instance memory, high memory bandwidth, large L3 caches, and leading reliability capabilities to boost the performance and reliability of in-memory applications.
* `service/events`: Updates service API and documentation
  * Exposes ConcurrentModificationException as one of the valid exceptions for PutPermission and RemovePermission operation.

### SDK Enhancements
* `service/autoscaling`: Fix documentation for PutScalingPolicy.AutoScalingGroupName [#1522](https://github.com/aws/aws-sdk-go/pull/1522)
* `service/s3/s3manager`: Clarify S3 Upload manager Concurrency config [#1521](https://github.com/aws/aws-sdk-go/pull/1521)
  * Fixes [#1458](https://github.com/aws/aws-sdk-go/issues/1458)
* `service/dynamodb/dynamodbattribute`: Add support for time alias. [#1520](https://github.com/aws/aws-sdk-go/pull/1520)
  * Related to [#1505](https://github.com/aws/aws-sdk-go/pull/1505)

Release v1.10.43 (2017-09-12)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Fixed bug in EC2 clients preventing HostOfferingSet from being set
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.10.42 (2017-09-12)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * DeviceFarm has added support for two features - RemoteDebugging and Customer Artifacts. Customers  can now do remote Debugging on their Private Devices and can now retrieve custom files generated by their tests on the device and the device host (execution environment) on both public and private devices.

Release v1.10.41 (2017-09-08)
===

### Service Client Updates
* `service/logs`: Updates service API and documentation
  * Adds support for the PutResourcePolicy, DescribeResourcePolicy and DeleteResourcePolicy APIs.

Release v1.10.40 (2017-09-07)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service documentation
* `service/ec2`: Updates service API and documentation
  * With Tagging support, you can add Key and Value metadata to search, filter and organize your NAT Gateways according to your organization's needs.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lex-models`: Updates service API and documentation
* `service/route53`: Updates service API and documentation
  * You can configure Amazon Route 53 to log information about the DNS queries that Amazon Route 53 receives for your domains and subdomains. When you configure query logging, Amazon Route 53 starts to send logs to CloudWatch Logs. You can use various tools, including the AWS console, to access the query logs.

Release v1.10.39 (2017-09-06)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Add an optional "thresholdType" to notifications to support percentage or absolute value thresholds.

Release v1.10.38 (2017-09-05)
===

### Service Client Updates
* `service/codestar`: Updates service API and documentation
  * Added support to tag CodeStar projects. Tags can be used to organize and find CodeStar projects on key-value pairs that you can choose. For example, you could add a tag with a key of "Release" and a value of "Beta" to projects your organization is working on for an upcoming beta release.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.10.37 (2017-09-01)
===

### Service Client Updates
* `service/MobileHub`: Adds new service
* `service/gamelift`: Updates service API and documentation
  * GameLift VPC resources can be peered with any other AWS VPC. R4 memory-optimized instances now available to deploy.
* `service/ssm`: Updates service API and documentation
  * Adding KMS encryption support to SSM Inventory Resource Data Sync. Exposes the ClientToken parameter on SSM StartAutomationExecution to provide idempotent execution requests.

Release v1.10.36 (2017-08-31)
===

### Service Client Updates
* `service/codebuild`: Updates service API, documentation, and examples
  * The AWS CodeBuild HTTP API now provides the BatchDeleteBuilds operation, which enables you to delete existing builds.
* `service/ec2`: Updates service API and documentation
  * Descriptions for Security Group Rules enables customers to be able to define a description for ingress and egress security group rules . The Descriptions for Security Group Rules feature supports one description field per Security Group rule for both ingress and egress rules . Descriptions for Security Group Rules provides a simple way to describe the purpose or function of a Security Group Rule allowing for easier customer identification of configuration elements .      Prior to the release of Descriptions for Security Group Rules , customers had to maintain a separate system outside of AWS if they wanted to track Security Group Rule mapping and their purpose for being implemented. If a security group rule has already been created and you would like to update or change your description for that security group rule you can use the UpdateSecurityGroupRuleDescription API.
* `service/elasticloadbalancingv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lex-models`: Updates service API and documentation

### SDK Bugs
* `aws/signer/v4`: Revert [#1491](https://github.com/aws/aws-sdk-go/issues/1491) as change conflicts with an undocumented AWS v4 signature test case.
  * Related to: [#1495](https://github.com/aws/aws-sdk-go/issues/1495).

Release v1.10.35 (2017-08-30)
===

### Service Client Updates
* `service/application-autoscaling`: Updates service API and documentation
* `service/organizations`: Updates service API and documentation
  * The exception ConstraintViolationException now contains a new reason subcode MASTERACCOUNT_MISSING_CONTACT_INFO to make it easier to understand why attempting to remove an account from an Organization can fail. We also improved several other of the text descriptions and examples.

Release v1.10.34 (2017-08-29)
===

### Service Client Updates
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Provides capability to add secondary CIDR blocks to a VPC.

### SDK Bugs
* `aws/signer/v4`: Fix Signing Unordered Multi Value Query Parameters ([#1491](https://github.com/aws/aws-sdk-go/pull/1491))
  * Removes sorting of query string values when calculating v4 signing as this is not part of the spec. The spec only requires the keys, not values, to be sorted which is achieved by Query.Encode().

Release v1.10.33 (2017-08-25)
===

### Service Client Updates
* `service/cloudformation`: Updates service API and documentation
  * Rollback triggers enable you to have AWS CloudFormation monitor the state of your application during stack creation and updating, and to roll back that operation if the application breaches the threshold of any of the alarms you've specified.
* `service/gamelift`: Updates service API
  * Update spelling of MatchmakingTicket status values for internal consistency.
* `service/rds`: Updates service API and documentation
  * Option group options now contain additional properties that identify requirements for certain options. Check these properties to determine if your DB instance must be in a VPC or have auto minor upgrade turned on before you can use an option. Check to see if you can downgrade the version of an option after you have installed it.

### SDK Enhancements
* `example/service/ec2`: Add EC2 list instances example ([#1492](https://github.com/aws/aws-sdk-go/pull/1492))

Release v1.10.32 (2017-08-25)
===

### Service Client Updates
* `service/rekognition`: Updates service API, documentation, and examples
  * Update the enum value of LandmarkType and GenderType to be consistent with service response

Release v1.10.31 (2017-08-23)
===

### Service Client Updates
* `service/appstream`: Updates service documentation
  * Documentation updates for appstream
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.10.30 (2017-08-22)
===

### Service Client Updates
* `service/ssm`: Updates service API and documentation
  * Changes to associations in Systems Manager State Manager can now be recorded. Previously, when you edited associations, you could not go back and review older association settings. Now, associations are versioned, and can be named using human-readable strings, allowing you to see a trail of association changes. You can also perform rate-based scheduling, which allows you to schedule associations more granularly.

Release v1.10.29 (2017-08-21)
===

### Service Client Updates
* `service/firehose`: Updates service API, documentation, and paginators
  * This change will allow customers to attach a Firehose delivery stream to an existing Kinesis stream directly. You no longer need a forwarder to move data from a Kinesis stream to a Firehose delivery stream. You can now run your streaming applications on your Kinesis stream and easily attach a Firehose delivery stream to it for data delivery to S3, Redshift, or Elasticsearch concurrently.
* `service/route53`: Updates service API and documentation
  * Amazon Route 53 now supports CAA resource record type. A CAA record controls which certificate authorities are allowed to issue certificates for the domain or subdomain.

Release v1.10.28 (2017-08-18)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.10.27 (2017-08-16)
===

### Service Client Updates
* `service/gamelift`: Updates service API and documentation
  * The Matchmaking Grouping Service is a new feature that groups player match requests for a given game together into game sessions based on developer configured rules.

### SDK Enhancements
* `aws/arn`: aws/arn: Package for parsing and producing ARNs ([#1463](https://github.com/aws/aws-sdk-go/pull/1463))
  * Adds the `arn` package for AWS ARN parsing and building. Use this package to build AWS ARNs for services such as outlined in the [documentation](http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html).

### SDK Bugs
* `aws/signer/v4`: Correct V4 presign signature to include content sha25 in URL ([#1469](https://github.com/aws/aws-sdk-go/pull/1469))
  * Updates the V4 signer so that when a Presign is generated the `X-Amz-Content-Sha256` header is added to the query string instead of being required to be in the header. This allows you to generate presigned URLs for GET requests, e.g S3.GetObject that do not require additional headers to be set by the downstream users of the presigned URL.
  * Related To: [#1467](https://github.com/aws/aws-sdk-go/issues/1467)

Release v1.10.26 (2017-08-15)
===

### Service Client Updates
* `service/ec2`: Updates service API
  * Fixed bug in EC2 clients preventing HostReservation from being set
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.10.25 (2017-08-14)
===

### Service Client Updates
* `service/AWS Glue`: Adds new service
* `service/batch`: Updates service API and documentation
  * This release enhances the DescribeJobs API to include the CloudWatch logStreamName attribute in ContainerDetail and ContainerDetailAttempt
* `service/cloudhsmv2`: Adds new service
  * CloudHSM provides hardware security modules for protecting sensitive data and cryptographic keys within an EC2 VPC, and enable the customer to maintain control over key access and use. This is a second-generation of the service that will improve security, lower cost and provide better customer usability.
* `service/elasticfilesystem`: Updates service API, documentation, and paginators
  * Customers can create encrypted EFS file systems and specify a KMS master key to encrypt it with.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/mgh`: Adds new service
  * AWS Migration Hub provides a single location to track migrations across multiple AWS and partner solutions. Using Migration Hub allows you to choose the AWS and partner migration tools that best fit your needs, while providing visibility into the status of your entire migration portfolio. Migration Hub also provides key metrics and progress for individual applications, regardless of which tools are being used to migrate them. For example, you might use AWS Database Migration Service, AWS Server Migration Service, and partner migration tools to migrate an application comprised of a database, virtualized web servers, and a bare metal server. Using Migration Hub will provide you with a single screen that shows the migration progress of all the resources in the application. This allows you to quickly get progress updates across all of your migrations, easily identify and troubleshoot any issues, and reduce the overall time and effort spent on your migration projects. Migration Hub is available to all AWS customers at no additional charge. You only pay for the cost of the migration tools you use, and any resources being consumed on AWS.
* `service/ssm`: Updates service API and documentation
  * Systems Manager Maintenance Windows include the following changes or enhancements: New task options using Systems Manager Automation, AWS Lambda, and AWS Step Functions; enhanced ability to edit the targets of a Maintenance Window, including specifying a target name and description, and ability to edit the owner field; enhanced ability to edits tasks; enhanced support for Run Command parameters; and you can now use a --safe flag when attempting to deregister a target. If this flag is enabled when you attempt to deregister a target, the system returns an error if the target is referenced by any task. Also, Systems Manager now includes Configuration Compliance to scan your fleet of managed instances for patch compliance and configuration inconsistencies. You can collect and aggregate data from multiple AWS accounts and Regions, and then drill down into specific resources that aren't compliant.
* `service/storagegateway`: Updates service API and documentation
  * Add optional field ForceDelete to DeleteFileShare api.

Release v1.10.24 (2017-08-11)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * Adds support for specifying Application Load Balancers in deployment groups, for both in-place and blue/green deployments.
* `service/cognito-idp`: Updates service API and documentation
* `service/ec2`: Updates service API and documentation
  * Provides customers an opportunity to recover an EIP that was released

Release v1.10.23 (2017-08-10)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * Enable BatchDetachPolicy
* `service/codebuild`: Updates service API
  * Supporting Bitbucket as source type in AWS CodeBuild.

Release v1.10.22 (2017-08-09)
===

### Service Client Updates
* `service/rds`: Updates service documentation
  * Documentation updates for RDS.

Release v1.10.21 (2017-08-09)
===

### Service Client Updates
* `service/elasticbeanstalk`: Updates service API and documentation
  * Add support for paginating the result of DescribeEnvironments     Include the ARN of described environments in DescribeEnvironments output

### SDK Enhancements
* `aws`: Add pointer conversion utilities to transform int64 to time.Time [#1433](https://github.com/aws/aws-sdk-go/pull/1433)
  * Adds `SecondsTimeValue` and `MillisecondsTimeValue` utilities.

Release v1.10.20 (2017-08-01)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * AWS CodeDeploy now supports the use of multiple tag groups in a single deployment group (an intersection of tags) to identify the instances for a deployment. When you create or update a deployment group, use the new ec2TagSet and onPremisesTagSet structures to specify up to three groups of tags. Only instances that are identified by at least one tag in each of the tag groups are included in the deployment group.
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service waiters
  * Ec2 SpotInstanceRequestFulfilled waiter update
* `service/elasticloadbalancingv2`: Updates service waiters
* `service/email`: Updates service API, documentation, paginators, and examples
  * This update adds information about publishing email open and click events. This update also adds information about publishing email events to Amazon Simple Notification Service (Amazon SNS).
* `service/pinpoint`: Updates service API and documentation
  * This release of the Pinpoint SDK enables App management - create, delete, update operations, Raw Content delivery for APNs and GCM campaign messages and From Address override.

Release v1.10.19 (2017-08-01)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/inspector`: Updates service API, documentation, and paginators
  * Inspector's StopAssessmentRun API has been updated with a new input option - stopAction. This request parameter can be set to either START_EVALUATION or SKIP_EVALUATION. START_EVALUATION (the default value, and the previous behavior) stops the AWS agent data collection and begins the results evaluation for findings generation based on the data collected so far. SKIP_EVALUATION cancels the assessment run immediately, after which no findings are generated.
* `service/ssm`: Updates service API and documentation
  * Adds a SendAutomationSignal API to SSM Service. This API is used to send a signal to an automation execution to change the current behavior or status of the execution.

Release v1.10.18 (2017-07-27)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * The CreateDefaultVPC API enables you to create a new default VPC . You no longer need to contact AWS support, if your default VPC has been deleted.
* `service/kinesisanalytics`: Updates service API and documentation
  * Added additional exception types and clarified documentation.

Release v1.10.17 (2017-07-27)
===

### Service Client Updates
* `service/dynamodb`: Updates service documentation and examples
  * Corrected a typo.
* `service/ec2`: Updates service API and documentation
  * Amazon EC2 Elastic GPUs allow you to easily attach low-cost graphics acceleration to current generation EC2 instances. With Amazon EC2 Elastic GPUs, you can configure the right amount of graphics acceleration to your particular workload without being constrained by fixed hardware configurations and limited GPU selection.
* `service/monitoring`: Updates service documentation
  * This release adds high resolution features to CloudWatch, with support for Custom Metrics down to 1 second and Alarms down to 10 seconds.

Release v1.10.16 (2017-07-26)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * Cloud Directory adds support for additional batch operations.
* `service/cloudformation`: Updates service API and documentation
  * AWS CloudFormation StackSets enables you to manage stacks across multiple accounts and regions.

### SDK Enhancements
* `aws/signer/v4`: Optimize V4 signer's header duplicate space stripping. [#1417](https://github.com/aws/aws-sdk-go/pull/1417)

Release v1.10.15 (2017-07-24)
===

### Service Client Updates
* `service/appstream`: Updates service API, documentation, and waiters
  * Amazon AppStream 2.0 image builders and fleets can now access applications and network resources that rely on Microsoft Active Directory (AD) for authentication and permissions. This new feature allows you to join your streaming instances to your AD, so you can use your existing AD user management tools.
* `service/ec2`: Updates service API and documentation
  * Spot Fleet tagging capability allows customers to automatically tag instances launched by Spot Fleet. You can use this feature to label or distinguish instances created by distinct Spot Fleets. Tagging your EC2 instances also enables you to see instance cost allocation by tag in your AWS bill.

### SDK Bugs
* `aws/signer/v4`: Fix out of bounds panic in stripExcessSpaces [#1412](https://github.com/aws/aws-sdk-go/pull/1412)
  * Fixes the out of bands panic in stripExcessSpaces caused by an incorrect calculation of the stripToIdx value. Simplified to code also.
  * Fixes [#1411](https://github.com/aws/aws-sdk-go/issues/1411)

Release v1.10.14 (2017-07-20)
===

### Service Client Updates
* `service/elasticmapreduce`: Updates service API and documentation
  * Amazon EMR now includes the ability to use a custom Amazon Linux AMI and adjustable root volume size when launching a cluster.

Release v1.10.13 (2017-07-19)
===

### Service Client Updates
* `service/budgets`: Updates service API and documentation
  * Update budget Management API's to list/create/update RI_UTILIZATION type budget. Update budget Management API's to support DAILY timeUnit for RI_UTILIZATION type budget.

### SDK Enhancements
* `service/s3`:  Use interfaces assertions instead of ValuesAtPath for S3 field lookups. [#1401](https://github.com/aws/aws-sdk-go/pull/1401)
  * Improves the performance across the board for all S3 API calls by removing the usage of `ValuesAtPath` being used for every S3 API call.

### SDK Bugs
* `aws/request`: waiter test bug
  * waiters_test.go file would sometimes fail due to travis hiccups. This occurs because a test would sometimes fail the cancel check and succeed the timeout. However, the timeout check should never occur in that test. This fix introduces a new field that dictates how waiters will sleep.

Release v1.10.12 (2017-07-17)
===

### Service Client Updates
* `service/cognito-idp`: Updates service API and documentation
* `service/lambda`: Updates service API and documentation
  * Lambda@Edge lets you run code closer to your end users without provisioning or managing servers. With Lambda@Edge, your code runs in AWS edge locations, allowing you to respond to your end users at the lowest latency. Your code is triggered by Amazon CloudFront events, such as requests to and from origin servers and viewers, and it is ready to execute at every AWS edge location whenever a request for content is received. You just upload your Node.js code to AWS Lambda and Lambda takes care of everything required to run and scale your code with high availability. You only pay for the compute time you consume - there is no charge when your code is not running.

Release v1.10.11 (2017-07-14)
===

### Service Client Updates
* `service/discovery`: Updates service API and documentation
  * Adding feature to the Export API for Discovery Service to allow filters for the export task to allow export based on per agent id.
* `service/ec2`: Updates service API
  * New EC2 GPU Graphics instance
* `service/marketplacecommerceanalytics`: Updates service documentation
  * Update to Documentation Model For New Report Cadence / Reformat of Docs

Release v1.10.10 (2017-07-13)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Adds support for management of gateway responses.
* `service/ec2`: Updates service API and documentation
  * X-ENI (or Cross-Account ENI) is a new feature that allows the attachment or association of Elastic Network Interfaces (ENI) between VPCs in different AWS accounts located in the same availability zone. With this new capability, service providers and partners can deliver managed solutions in a variety of new architectural patterns where the provider and consumer of the service are in different AWS accounts.
* `service/lex-models`: Updates service documentation

Release v1.10.9 (2017-07-12)
===

### Service Client Updates
* `service/autoscaling`: Updates service API and documentation
  * Auto Scaling now supports a new type of scaling policy called target tracking scaling policies that you can use to set up dynamic scaling for your application.
* `service/swf`: Updates service API, documentation, paginators, and examples
  * Added support for attaching control data to Lambda tasks. Control data lets you attach arbitrary strings to your decisions and history events.

Release v1.10.8 (2017-07-06)
===

### Service Client Updates
* `service/ds`: Updates service API, documentation, and paginators
  * You can now improve the resilience and performance of your Microsoft AD directory by deploying additional domain controllers. Added UpdateNumberofDomainControllers API that allows you to update the number of domain controllers you want for your directory, and DescribeDomainControllers API that allows you to describe the detailed information of each domain controller of your directory. Also added the 'DesiredNumberOfDomainControllers' field to the DescribeDirectories API output for Microsoft AD.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/kinesis`: Updates service API and documentation
  * You can now encrypt your data at rest within an Amazon Kinesis Stream using server-side encryption. Server-side encryption via AWS KMS makes it easy for customers to meet strict data management requirements by encrypting their data at rest within the Amazon Kinesis Streams, a fully managed real-time data processing service.
* `service/kms`: Updates service API and documentation
  * This release of AWS Key Management Service introduces the ability to determine whether a key is AWS managed or customer managed.
* `service/ssm`: Updates service API and documentation
  * Amazon EC2 Systems Manager now expands Patching support to Amazon Linux, Red Hat and Ubuntu in addition to the already supported Windows Server.

Release v1.10.7 (2017-07-05)
===

### Service Client Updates
* `service/monitoring`: Updates service API and documentation
  * We are excited to announce the availability of APIs and CloudFormation support for CloudWatch Dashboards. You can use the new dashboard APIs or CloudFormation templates to dynamically build and maintain dashboards to monitor your infrastructure and applications. There are four new dashboard APIs - PutDashboard, GetDashboard, DeleteDashboards, and ListDashboards APIs. PutDashboard is used to create a new dashboard or modify an existing one whereas GetDashboard is the API to get the details of a specific dashboard. ListDashboards and DeleteDashboards are used to get the names or delete multiple dashboards respectively. Getting started with dashboard APIs is similar to any other AWS APIs. The APIs can be accessed through AWS SDK or through CLI tools.
* `service/route53`: Updates service API and documentation
  * Bug fix for InvalidChangeBatch exception.

### SDK Enhancements
* `service/s3/s3manager`: adding cleanup function to batch objects [#1375](https://github.com/aws/aws-sdk-go/issues/1375)
  * This enhancement will add an After field that will be called after each iteration of the batch operation.

Release v1.10.6 (2017-06-30)
===

### Service Client Updates
* `service/marketplacecommerceanalytics`: Updates service documentation
  * Documentation updates for AWS Marketplace Commerce Analytics.
* `service/s3`: Updates service API and documentation
  * API Update for S3: Adding Object Tagging Header to MultipartUpload Initialization

Release v1.10.5 (2017-06-29)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/events`: Updates service API and documentation
  * CloudWatch Events now allows different AWS accounts to share events with each other through a new resource called event bus. Event buses accept events from AWS services, other AWS accounts and PutEvents API calls. Currently all AWS accounts have one default event bus. To send events to another account, customers simply write rules to match the events of interest and attach an event bus in the receiving account as the target to the rule. The PutTargets API has been updated to allow adding cross account event buses as targets. In addition, we have released two new APIs - PutPermission and RemovePermission - that enables customers to add/remove permissions to their default event bus.
* `service/gamelift`: Updates service API and documentation
  * Allow developers to download GameLift fleet creation logs to assist with debugging.
* `service/ssm`: Updates service API and documentation
  * Adding Resource Data Sync support to SSM Inventory.  New APIs:  * CreateResourceDataSync - creates a new resource data sync configuration,  * ListResourceDataSync - lists existing resource data sync configurations,  * DeleteResourceDataSync - deletes an existing resource data sync configuration.

Release v1.10.4 (2017-06-27)
===

### Service Client Updates
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * Proper tagging of resources is critical to post-launch operations such as billing, cost allocation, and resource management. By using Service Catalog's TagOption Library, administrators can define a library of re-usable TagOptions that conform to company standards, and associate these with Service Catalog portfolios and products. Learn how to move your current tags to the new library, create new TagOptions, and view and associate your library items with portfolios and products. Understand how to ensure that the right tags are created on products launched through Service Catalog and how to provide users with defined selectable tags.

### SDK Bugs
* `aws/signer/v4`: checking length on `stripExcessSpaces` [#1372](https://github.com/aws/aws-sdk-go/issues/1372)
  * Fixes a bug where `stripExcessSpaces` did not check length against the slice.
  * Fixes: [#1371](https://github.com/aws/aws-sdk-go/issues/1371)

Release v1.10.3 (2017-06-23)
===

### Service Client Updates
* `service/lambda`: Updates service API and documentation
  * The Lambda Invoke API will now throw new exception InvalidRuntimeException (status code 502) for invokes with deprecated runtimes.

Release v1.10.2 (2017-06-22)
===

### Service Client Updates
* `service/codepipeline`: Updates service API, documentation, and paginators
  * A new API, ListPipelineExecutions, enables you to retrieve summary information about the most recent executions in a pipeline, including pipeline execution ID, status, start time, and last updated time. You can request information for a maximum of 100 executions. Pipeline execution data is available for the most recent 12 months of activity.
* `service/dms`: Updates service API and documentation
  * Added tagging for DMS certificates.
* `service/elasticloadbalancing`: Updates service waiters
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lightsail`: Updates service API and documentation
  * This release adds a new nextPageToken property to the result of the GetOperationsForResource API. Developers can now get the next set of items in a list by making subsequent calls to GetOperationsForResource API with the token from the previous call. This release also deprecates the nextPageCount property, which previously returned null (use the nextPageToken property instead). This release also deprecates the customImageName property on the CreateInstancesRequest class, which was previously ignored by the API.
* `service/route53`: Updates service API and documentation
  * This release reintroduces the HealthCheckInUse exception.

Release v1.10.1 (2017-06-21)
===

### Service Client Updates
* `service/dax`: Adds new service
  * Amazon DynamoDB Accelerator (DAX) is a fully managed, highly available, in-memory cache for DynamoDB that delivers up to a 10x performance improvement - from milliseconds to microseconds - even at millions of requests per second. DAX does all the heavy lifting required to add in-memory acceleration to your DynamoDB tables, without requiring developers to manage cache invalidation, data population, or cluster management.
* `service/route53`: Updates service API and documentation
  * Amazon Route 53 now supports multivalue answers in response to DNS queries, which lets you route traffic approximately randomly to multiple resources, such as web servers. Create one multivalue answer record for each resource and, optionally, associate an Amazon Route 53 health check with each record, and Amazon Route 53 responds to DNS queries with up to eight healthy records.
* `service/ssm`: Updates service API, documentation, and paginators
  * Adding hierarchy support to the SSM Parameter Store API. Added support tor tagging. New APIs: GetParameter - retrieves one parameter, DeleteParameters - deletes multiple parameters (max number 10), GetParametersByPath - retrieves parameters located in the hierarchy. Updated APIs: PutParameter - added ability to enforce parameter value by applying regex (AllowedPattern), DescribeParameters - modified to support Tag filtering.
* `service/waf`: Updates service API and documentation
  * You can now create, edit, update, and delete a new type of WAF rule with a rate tracking component.
* `service/waf-regional`: Updates service API and documentation

Release v1.10.0 (2017-06-20)
===

### Service Client Updates
* `service/workdocs`: Updates service API and documentation
  * This release provides a new API to retrieve the activities performed by WorkDocs users.

### SDK Features
* `aws/credentials/plugincreds`: Add support for Go plugin for credentials [#1320](https://github.com/aws/aws-sdk-go/pull/1320)
  * Adds support for using plugins to retrieve credentials for API requests. This change adds a new package plugincreds under aws/credentials. See the `example/aws/credentials/plugincreds` folder in the SDK for example usage.

Release v1.9.00 (2017-06-19)
===

### Service Client Updates
* `service/organizations`: Updates service API and documentation
  * Improvements to Exception Modeling

### SDK Features
* `service/s3/s3manager`: Adds batch operations to s3manager [#1333](https://github.com/aws/aws-sdk-go/pull/1333)
  * Allows for batch upload, download, and delete of objects. Also adds the interface pattern to allow for easy traversal of objects. E.G `DownloadWithIterator`, `UploadWithIterator`, and `BatchDelete`. `BatchDelete` also contains a utility iterator using the `ListObjects` API to easily delete a list of objects.

Release v1.8.44 (2017-06-16)
===

### Service Client Updates
* `service/xray`: Updates service API, documentation, and paginators
  * Add a response time histogram to the services in response of GetServiceGraph API.

Release v1.8.43 (2017-06-15)
===

### Service Client Updates
* `service/ec2`: Updates service API and documentation
  * Adds API to describe Amazon FPGA Images (AFIs) available to customers, which includes public AFIs, private AFIs that you own, and AFIs owned by other AWS accounts for which you have load permissions.
* `service/ecs`: Updates service API and documentation
  * Added support for cpu, memory, and memory reservation container overrides on the RunTask and StartTask APIs.
* `service/iot`: Updates service API and documentation
  * Revert the last release: remove CertificatePem from DescribeCertificate API.
* `service/servicecatalog`: Updates service API, documentation, and paginators
  * Added ProvisioningArtifactSummaries to DescribeProductAsAdmin's output to show the provisioning artifacts belong to the product. Allow filtering by SourceProductId in SearchProductsAsAdmin for AWS Marketplace products. Added a verbose option to DescribeProvisioningArtifact to display the CloudFormation template used to create the provisioning artifact.Added DescribeProvisionedProduct API. Changed the type of ProvisionedProduct's Status to be distinct from Record's Status. New ProvisionedProduct's Status are AVAILABLE, UNDER_CHANGE, TAINTED, ERROR. Changed Record's Status set of values to CREATED, IN_PROGRESS, IN_PROGRESS_IN_ERROR, SUCCEEDED, FAILED.

### SDK Bugs
* `private/model/api`: Fix RESTXML support for XML Namespace [#1343](https://github.com/aws/aws-sdk-go/pull/1343)
  * Fixes a bug with the SDK's generation of services using the REST XML protocol not annotating shape references with the XML Namespace attribute.
  * Fixes [#1334](https://github.com/aws/aws-sdk-go/pull/1334)

Release v1.8.42 (2017-06-14)
===

### Service Client Updates
* `service/applicationautoscaling`: Updates service API and documentation
* `service/clouddirectory`: Updates service documentation
  * Documentation update for Cloud Directory

Release v1.8.41 (2017-06-13)
===

### Service Client Updates
* `service/configservice`: Updates service API

Release v1.8.40 (2017-06-13)
===

### Service Client Updates
* `service/rds`: Updates service API and documentation
  * API Update for RDS: this update enables copy-on-write, a new Aurora MySQL Compatible Edition feature that allows users to restore their database, and support copy of TDE enabled snapshot cross region.

### SDK Bugs
* `aws/request`: Fix NewErrParamMinLen to use correct ParamMinLenErrCode [#1336](https://github.com/aws/aws-sdk-go/issues/1336)
  * Fixes the `NewErrParamMinLen` function returning the wrong error code. `ParamMinLenErrCode` should be returned not `ParamMinValueErrCode`.
  * Fixes [#1335](https://github.com/aws/aws-sdk-go/issues/1335)

Release v1.8.39 (2017-06-09)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/opsworks`: Updates service API and documentation
  * Tagging Support for AWS OpsWorks Stacks

Release v1.8.38 (2017-06-08)
===

### Service Client Updates
* `service/iot`: Updates service API and documentation
  * In addition to using certificate ID, AWS IoT customers can now obtain the description of a certificate with the certificate PEM.
* `service/pinpoint`: Updates service API and documentation
  * Starting today Amazon Pinpoint adds SMS Text and Email Messaging support in addition to Mobile Push Notifications, providing developers, product managers and marketers with multi-channel messaging capabilities to drive user engagement in their applications. Pinpoint also enables backend services and applications to message users directly and provides advanced user and app analytics to understand user behavior and messaging performance.
* `service/rekognition`: Updates service API and documentation
  * API Update for AmazonRekognition: Adding RecognizeCelebrities API

Release v1.8.37 (2017-06-07)
===

### Service Client Updates
* `service/codebuild`: Updates service API and documentation
  * Add support to APIs for privileged containers. This change would allow performing privileged operations like starting the Docker daemon inside builds possible in custom docker images.
* `service/greengrass`: Adds new service
  * AWS Greengrass is software that lets you run local compute, messaging, and device state synchronization for connected devices in a secure way. With AWS Greengrass, connected devices can run AWS Lambda functions, keep device data in sync, and communicate with other devices securely even when not connected to the Internet. Using AWS Lambda, Greengrass ensures your IoT devices can respond quickly to local events, operate with intermittent connections, and minimize the cost of transmitting IoT data to the cloud.

Release v1.8.36 (2017-06-06)
===

### Service Client Updates
* `service/acm`: Updates service documentation
  * Documentation update for AWS Certificate Manager.
* `service/cloudfront`: Updates service documentation
  * Doc update to fix incorrect prefix in S3OriginConfig
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/iot`: Updates service API
  * Update client side validation for SalesForce action.

Release v1.8.35 (2017-06-05)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * AppStream 2.0 Custom Security Groups allows you to easily control what network resources your streaming instances and images have access to. You can assign up to 5 security groups per Fleet to control the inbound and outbound network access to your streaming instances to specific IP ranges, network protocols, or ports.
* `service/iot`: Updates service API, documentation, paginators, and examples
  * Added Salesforce action to IoT Rules Engine.

Release v1.8.34 (2017-06-02)
===

### Service Client Updates
* `service/kinesisanalytics`: Updates service API, documentation, and paginators
  * Kinesis Analytics publishes error messages CloudWatch logs in case of application misconfigurations
* `service/workdocs`: Updates service API and documentation
  * This release includes new APIs to manage tags and custom metadata on resources and also new APIs to add and retrieve comments at the document level.

Release v1.8.33 (2017-06-01)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * AWS CodeDeploy has improved how it manages connections to GitHub accounts and repositories. You can now create and store up to 25 connections to GitHub accounts in order to associate AWS CodeDeploy applications with GitHub repositories. Each connection can support multiple repositories. You can create connections to up to 25 different GitHub accounts, or create more than one connection to a single account. The ListGitHubAccountTokenNames command has been introduced to retrieve the names of stored connections to GitHub accounts that you have created. The name of the connection to GitHub used for an AWS CodeDeploy application is also included in the ApplicationInfo structure.  Two new fields, lastAttemptedDeployment and lastSuccessfulDeployment, have been added to DeploymentGroupInfo to improve the handling of deployment group information in the AWS CodeDeploy console. Information about these latest deployments can also be retrieved using the GetDeploymentGroup and BatchGetDeployment group requests. Also includes a region update  (us-gov-west-1).
* `service/cognitoidentityprovider`: Updates service API, documentation, and paginators
* `service/elbv2`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lexmodelbuildingservice`: Updates service documentation and examples

### SDK Enhancements
* `aws/defaults`: Exports shared credentials and config default filenames used by the SDK. [#1308](https://github.com/aws/aws-sdk-go/pull/1308)
  * Adds SharedCredentialsFilename and SharedConfigFilename functions to defaults package.

### SDK Bugs
* `aws/credentials`: Fixes shared credential provider's default filename on Windows. [#1308](https://github.com/aws/aws-sdk-go/pull/1308)
  * The shared credentials provider would attempt to use the wrong filename on Windows if the `HOME` environment variable was defined.
* `service/s3/s3manager`: service/s3/s3manager: Fix Downloader ignoring Range get parameter [#1311](https://github.com/aws/aws-sdk-go/pull/1311)
  * Fixes the S3 Download Manager ignoring the GetObjectInput's Range parameter. If this parameter is provided it will force the downloader to fallback to a single GetObject request disabling concurrency and automatic part size gets.
  * Fixes [#1296](https://github.com/aws/aws-sdk-go/issues/1296)

Release v1.8.32 (2017-05-31)
===

### Service Client Updates
* `service/rds`: Updates service API and documentation
  * Amazon RDS customers can now easily and quickly stop and start their DB instances.

Release v1.8.31 (2017-05-30)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API, documentation, and paginators
  * Cloud Directory has launched support for Typed Links, enabling customers to create object-to-object relationships that are not hierarchical in nature. Typed Links enable customers to quickly query for data along these relationships. Customers can also enforce referential integrity using Typed Links, ensuring data in use is not inadvertently deleted.
* `service/s3`: Updates service paginators and examples
  * New example snippets for Amazon S3.

Release v1.8.30 (2017-05-25)
===

### Service Client Updates
* `service/appstream`: Updates service API and documentation
  * Support added for persistent user storage, backed by S3.
* `service/rekognition`: Updates service API and documentation
  * Updated the CompareFaces API response to include orientation information, unmatched faces, landmarks, pose, and quality of the compared faces.

Release v1.8.29 (2017-05-24)
===

### Service Client Updates
* `service/iam`: Updates service API
  * The unique ID and access key lengths were extended from 32 to 128
* `service/storagegateway`: Updates service API and documentation
  * Two Storage Gateway data types, Tape and TapeArchive, each have a new response element, TapeUsedInBytes. This element helps you manage your virtual tapes. By using TapeUsedInBytes, you can see the amount of data written to each virtual tape.
* `service/sts`: Updates service API, documentation, and paginators
  * The unique ID and access key lengths were extended from 32 to 128.

Release v1.8.28 (2017-05-23)
===

### Service Client Updates
* `service/databasemigrationservice`: Updates service API, documentation, paginators, and examples
  * This release adds support for using Amazon S3 and Amazon DynamoDB as targets for database migration, and using MongoDB as a source for database migration. For more information, see the AWS Database Migration Service documentation.

Release v1.8.27 (2017-05-22)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/resourcegroupstaggingapi`: Updates service API, documentation, and paginators
  * You can now specify the number of resources returned per page in GetResources operation, as an optional parameter, to easily manage the list of resources returned by your queries.

### SDK Bugs
* `aws/request`: Add support for PUT temporary redirects (307) [#1283](https://github.com/aws/aws-sdk-go/issues/1283)
  * Adds support for Go 1.8's GetBody function allowing the SDK's http request using PUT and POST methods to be redirected with temporary redirects with 307 status code.
  * Fixes: [#1267](https://github.com/aws/aws-sdk-go/issues/1267)
* `aws/request`: Add handling for retrying temporary errors during unmarshal [#1289](https://github.com/aws/aws-sdk-go/issues/1289)
  * Adds support for retrying temporary errors that occur during unmarshaling of a request's response body.
  * Fixes: [#1275](https://github.com/aws/aws-sdk-go/issues/1275)

Release v1.8.26 (2017-05-18)
===

### Service Client Updates
* `service/athena`: Adds new service
  * This release adds support for Amazon Athena. Amazon Athena is an interactive query service that makes it easy to analyze data in Amazon S3 using standard SQL. Athena is serverless, so there is no infrastructure to manage, and you pay only for the queries that you run.
* `service/lightsail`: Updates service API, documentation, and paginators
  * This release adds new APIs that make it easier to set network port configurations on Lightsail instances. Developers can now make a single request to both open and close public ports on an instance using the PutInstancePublicPorts operation.

### SDK Bugs
* `aws/request`: Fix logging from reporting wrong retry request errors #1281
  * Fixes the SDK's retry request logging to report the the actual error that occurred, not a stubbed Unknown error message.
  * Fixes the SDK's response logger to not output the response log multiple times per retry.

Release v1.8.25 (2017-05-17)
===

### Service Client Updates
* `service/autoscaling`: Updates service documentation, paginators, and examples
  * Various Auto Scaling documentation updates
* `service/cloudwatchevents`: Updates service documentation
  * Various CloudWatch Events documentation updates.
* `service/cloudwatchlogs`: Updates service documentation and paginators
  * Various CloudWatch Logs documentation updates.
* `service/polly`: Updates service API
  * Amazon Polly adds new German voice "Vicki"

Release v1.8.24 (2017-05-16)
===

### Service Client Updates
* `service/codedeploy`: Updates service API and documentation
  * This release introduces the previousRevision field in the responses to the GetDeployment and BatchGetDeployments actions. previousRevision provides information about the application revision that was deployed to the deployment group before the most recent successful deployment.  Also, the fileExistsBehavior parameter has been added for CreateDeployment action requests. In the past, if the AWS CodeDeploy agent detected files in a target location that weren't part of the application revision from the most recent successful deployment, it would fail the current deployment by default. This new parameter provides options for how the agent handles these files: fail the deployment, retain the content, or overwrite the content.
* `service/gamelift`: Updates service API and documentation
  * Allow developers to specify how metrics are grouped in CloudWatch for their GameLift fleets. Developers can also specify how many concurrent game sessions activate on a per-instance basis.
* `service/inspector`: Updates service API, documentation, paginators, and examples
  * Adds ability to produce an assessment report that includes detailed and comprehensive results of a specified assessment run.
* `service/kms`: Updates service documentation
  * Update documentation for KMS.

Release v1.8.23 (2017-05-15)
===

### Service Client Updates
* `service/ssm`: Updates service API and documentation
  * UpdateAssociation API now supports updating document name and targets of an association. GetAutomationExecution API can return FailureDetails as an optional field to the StepExecution Object, which contains failure type, failure stage as well as other failure related information for a failed step.

### SDK Enhancements
* `aws/session`: SDK should be able to load multiple custom shared config files. [#1258](https://github.com/aws/aws-sdk-go/issues/1258)
  * This change adds a `SharedConfigFiles` field to the `session.Options` type that allows you to specify the files, and their order, the SDK will use for loading shared configuration and credentials from when the `Session` is created. Use the `NewSessionWithOptions` Session constructor to specify these options. You'll also most likely want to enable support for the shared configuration file's additional attributes by setting `session.Option`'s `SharedConfigState` to `session.SharedConfigEnabled`.

Release v1.8.22 (2017-05-11)
===

### Service Client Updates
* `service/elb`: Updates service API, documentation, and paginators
* `service/elbv2`: Updates service API and documentation
* `service/lexmodelbuildingservice`: Updates service API and documentation
* `service/organizations`: Updates service API, documentation, paginators, and examples
  * AWS Organizations APIs that return an Account object now include the email address associated with the account’s root user.

Release v1.8.21 (2017-05-09)
===

### Service Client Updates
* `service/codestar`: Updates service documentation
  * Updated documentation for AWS CodeStar.
* `service/workspaces`: Updates service API, documentation, and paginators
  * Doc-only Update for WorkSpaces

Release v1.8.20 (2017-05-04)
===

### Service Client Updates
* `service/ecs`: Updates service API, documentation, and paginators
  * Exposes container instance registration time in ECS:DescribeContainerInstances.
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/marketplaceentitlementservice`: Adds new service
* `service/lambda`: Updates service API and documentation
  * Support for UpdateFunctionCode DryRun option

Release v1.8.19 (2017-04-28)
===

### Service Client Updates
* `service/cloudformation`: Updates service waiters and paginators
  * Adding back the removed waiters and paginators.

Release v1.8.18 (2017-04-28)
===

### Service Client Updates
* `service/cloudformation`: Updates service API, documentation, waiters, paginators, and examples
  * API update for CloudFormation: New optional parameter ClientRequestToken which can be used as an idempotency token to safely retry certain operations as well as tagging StackEvents.
* `service/rds`: Updates service API, documentation, and examples
  * The DescribeDBClusterSnapshots API now returns a SourceDBClusterSnapshotArn field which identifies the source DB cluster snapshot of a copied snapshot.
* `service/rekognition`: Updates service API
  * Fix for missing file type check
* `service/snowball`: Updates service API, documentation, and paginators
  * The Snowball API has a new exception that can be thrown for list operation requests.
* `service/sqs`: Updates service API, documentation, and paginators
  * Adding server-side encryption (SSE) support to SQS by integrating with AWS KMS; adding new queue attributes to SQS CreateQueue, SetQueueAttributes and GetQueueAttributes APIs to support SSE.

Release v1.8.17 (2017-04-26)
===

### Service Client Updates
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * With Amazon Relational Database Service (Amazon RDS) running MySQL or Amazon Aurora, you can now authenticate to your DB instance using IAM database authentication.

Release v1.8.16 (2017-04-21)
===

### Service Client Updates
* `service/appstream`: Updates service API, documentation, and paginators
  * The new feature named "Default Internet Access" will enable Internet access from AppStream 2.0 instances - image builders and fleet instances. Admins will check a flag either through AWS management console for AppStream 2.0 or through API while creating an image builder or while creating/updating a fleet.
* `service/kinesis`: Updates service API, documentation, waiters, and paginators
  * Adds a new waiter, StreamNotExists, to Kinesis.

### SDK Enhancements
* `aws/endpoints`: Add utilities improving endpoints lookup (#1218)
  * Adds several utilities to the endpoints packages to make looking up partitions, regions, and services easier.
  * Fixes #994

### SDK Bugs
* `private/protocol/xml/xmlutil`: Fix unmarshaling dropping errors (#1219)
  * The XML unmarshaler would drop any serialization or body read error that occurred on the floor effectively hiding any errors that would occur.
  * Fixes #1205

Release v1.8.15 (2017-04-20)
===

### Service Client Updates
* `service/devicefarm`: Updates service API and documentation
  * API Update for AWS Device Farm: Support for Deals and Promotions
* `service/directconnect`: Updates service documentation
  * Documentation updates for AWS Direct Connect.
* `service/elbv2`: Updates service waiters
* `service/kms`: Updates service documentation and examples
  * Doc-only update for Key Management Service (KMS): Update docs for GrantConstraints and GenerateRandom
* `service/route53`: Updates service documentation
  * Release notes: SDK documentation now includes examples for ChangeResourceRecordSets for all types of resource record set, such as weighted, alias, and failover.
* `service/route53domains`: Updates service API, documentation, and paginators
  * Adding examples and other documentation updates.

### SDK Enhancements
* `service/s3`: Add utilities to make getting a bucket's region easier (#1207)
  * Adds two features which make it easier to get a bucket's region, `s3.NormalizeBucketLocation` and `s3manager.GetBucketRegion`.

### SDK Bugs
* `service/s3`: Fix HeadObject's incorrect documented error codes (#1213)
  * The HeadObject's model incorrectly states that the operation can return the NoSuchKey error code.
  * Fixes #1208

Release v1.8.14 (2017-04-19)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * Add support for "embed" property.
* `service/codestar`: Adds new service
  * AWS CodeStar is a cloud-based service for creating, managing, and working with software development projects on AWS. An AWS CodeStar project creates and integrates AWS services for your project development toolchain. AWS CodeStar also manages the permissions required for project users.
* `service/ec2`: Updates service API and documentation
  * Adds support for creating an Amazon FPGA Image (AFI) from a specified design checkpoint (DCP).
* `service/iam`: Updates service API and documentation
  * This changes introduces a new IAM role type, Service Linked Role, which works like a normal role but must be managed via services' control.
* `service/lambda`: Updates service API and documentation
  * Lambda integration with CloudDebugger service to enable customers to enable tracing for the Lambda functions and send trace information to the CloudDebugger service.
* `service/lexmodelbuildingservice`: Adds new service
* `service/polly`: Updates service API, documentation, and paginators
  * API Update for Amazon Polly: Add support for speech marks
* `service/rekognition`: Updates service API and documentation
  * Given an image, the API detects explicit or suggestive adult content in the image and returns a list of corresponding labels with confidence scores, as well as a taxonomy (parent-child relation) for each label.

Release v1.8.13 (2017-04-18)
===

### Service Client Updates
* `service/lambda`: Updates service API and documentation
  * You can use tags to group and filter your Lambda functions, making it easier to analyze them for billing allocation purposes. For more information, see Tagging Lambda Functions.  You can now write or upgrade your Lambda functions using Python version 3.6. For more information, see Programming Model for Authoring Lambda Functions in Python. Note: Features will be rolled out in the US regions on 4/19.

### SDK Enhancements
* `aws/request`: add support for appengine's custom standard library (#1190)
  * Remove syscall error checking on appengine platforms.

Release v1.8.12 (2017-04-11)
===

### Service Client Updates
* `service/apigateway`: Updates service API and documentation
  * API Gateway request validators
* `service/batch`: Updates service API and documentation
  * API Update for AWS Batch: Customer provided AMI for MANAGED Compute Environment
* `service/gamelift`: Updates service API and documentation
  * Allows developers to utilize an improved workflow when calling our Queues API and introduces a new feature that allows developers to specify a maximum allowable latency per Queue.
* `service/opsworks`: Updates service API, documentation, and paginators
  * Cloudwatch Logs agent configuration can now be attached to OpsWorks Layers using CreateLayer and UpdateLayer. OpsWorks will then automatically install and manage the CloudWatch Logs agent on the instances part of the OpsWorks Layer.

### SDK Bugs
* `aws/client`: Fix clients polluting handler list (#1197)
  * Fixes the clients potentially polluting the passed in handler list with the client's customizations. This change ensures every client always works with a clean copy of the request handlers and it cannot pollute the handlers back upstream.
  * Fixes #1184
* `aws/request`: Fix waiter error match condition (#1195)
  * Fixes the waiters's matching overwriting the request's err, effectively ignoring the error condition. This broke waiters with the FailureWaiterState matcher state.

Release v1.8.11 (2017-04-07)
===

### Service Client Updates
* `service/redshift`: Updates service API, documentation, and paginators
  * This update adds the GetClusterCredentials API which is used to get temporary login credentials to the cluster. AccountWithRestoreAccess now has a new member AccountAlias, this is the identifier of the AWS support account authorized to restore the specified snapshot. This is added to support the feature where the customer can share their snapshot with the Amazon Redshift Support Account without having to manually specify the AWS Redshift Service account ID on the AWS Console/API.

Release v1.8.10 (2017-04-06)
===

### Service Client Updates
* `service/elbv2`: Updates service documentation

Release v1.8.9 (2017-04-05)
===

### Service Client Updates
* `service/elasticache`: Updates service API, documentation, paginators, and examples
  * ElastiCache added support for testing the Elasticache Multi-AZ feature with Automatic Failover.

Release v1.8.8 (2017-04-04)
===

### Service Client Updates
* `service/cloudwatch`: Updates service API, documentation, and paginators
  * Amazon Web Services announced the immediate availability of two additional alarm configuration rules for Amazon CloudWatch Alarms. The first rule is for configuring missing data treatment. Customers have the options to treat missing data as alarm threshold breached, alarm threshold not breached, maintain alarm state and the current default treatment. The second rule is for alarms based on percentiles metrics that can trigger unnecessarily if the percentile is calculated from a small number of samples. The new rule can treat percentiles with low sample counts as same as missing data. If the first rule is enabled, the same treatment will be applied when an alarm encounters a percentile with low sample counts.

Release v1.8.7 (2017-04-03)
===

### Service Client Updates
* `service/lexruntimeservice`: Updates service API and documentation
  * Adds support to PostContent for speech input

### SDK Enhancements
* `aws/request`: Improve handler copy, push back, push front performance (#1171)
  * Minor optimization to the handler list's handling of copying and pushing request handlers to the handler list.
* Update codegen header to use Go std wording (#1172)
  * Go recently accepted the proposal for standard generated file header wording in, https://golang.org/s/generatedcode.

### SDK Bugs
* `service/dynamodb`: Fix DynamoDB using custom retryer (#1170)
  * Fixes (#1139) the DynamoDB service client clobbering any custom retryer that was passed into the service client or Session's config.

Release v1.8.6 (2017-04-01)
===

### Service Client Updates
* `service/clouddirectory`: Updates service API and documentation
  * ListObjectAttributes now supports filtering by facet.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.8.5 (2017-03-30)
===

### Service Client Updates
* `service/cloudformation`: Updates service waiters and paginators
  * Adding paginators for ListExports and ListImports
* `service/cloudfront`: Adds new service
  * Amazon CloudFront now supports user configurable HTTP Read and Keep-Alive Idle Timeouts for your Custom Origin Servers
* `service/configservice`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/resourcegroupstaggingapi`: Adds new service
* `service/storagegateway`: Updates service API and documentation
  * File gateway mode in AWS Storage gateway provides access to objects in S3 as files on a Network File System (NFS) mount point. Once a file share is created, any changes made externally to the S3 bucket will not be reflected by the gateway. Using the cache refresh feature in this update, the customer can trigger an on-demand scan of the keys in their S3 bucket and refresh the file namespace cached on the gateway. It takes as an input the fileShare ARN and refreshes the cache for only that file share. Additionally there is new functionality on file gateway that allows you configure what squash options they would like on their file share, this allows a customer to configure their gateway to not squash root permissions. This can be done by setting options in NfsOptions for CreateNfsFileShare and UpdateNfsFileShare APIs.

Release v1.8.4 (2017-03-28)
===

### Service Client Updates
* `service/batch`: Updates service API, documentation, and paginators
  * Customers can now provide a retryStrategy as part of the RegisterJobDefinition and SubmitJob API calls. The retryStrategy object has a number value for attempts. This is the number of non successful executions before a job is considered FAILED. In addition, the JobDetail object now has an attempts field and shows all execution attempts.
* `service/ec2`: Updates service API and documentation
	* Customers can now tag their Amazon EC2 Instances and Amazon EBS Volumes at
	the time of their creation. You can do this from the EC2 Instance launch
	wizard or through the RunInstances or CreateVolume APIs. By tagging
	resources at the time of creation, you can eliminate the need to run custom
	tagging scripts after resource creation. In addition, you can now set
	resource-level permissions on the CreateVolume, CreateTags, DeleteTags, and
	the RunInstances APIs. This allows you to implement stronger security
	policies by giving you more granular control over which users and groups
	have access to these APIs. You can also enforce the use of tagging and
	control what tag keys and values are set on your resources. When you combine
	tag usage and resource-level IAM policies together, you can ensure your
	instances and volumes are properly secured upon creation and achieve more
	accurate cost allocation reporting. These new features are provided at no
	additional cost.

### SDK Enhancements
* `aws/request`: Add retry support for RequestTimeoutException (#1158)
  * Adds support for retrying RequestTimeoutException error code that is returned by some services.

### SDK Bugs
* `private/model/api`: Fix Waiter and Paginators panic on nil param inputs (#1157)
  * Corrects the code generation for Paginators and waiters that caused a panic if nil input parameters were used with the operations.

Release v1.8.3 (2017-03-27)
===

## Service Client Updates
* `service/ssm`: Updates service API, documentation, and paginators
  * Updated validation rules for SendCommand and RegisterTaskWithMaintenanceWindow APIs.
Release v1.8.2 (2017-03-24)
===

Service Client Updates
---
* `service/applicationautoscaling`: Updates service API, documentation, and paginators
  * Application AutoScaling is launching support for a new target resource (AppStream 2.0 Fleets) as a scalable target.
* `service/cloudtrail`: Updates service API and documentation
  * Doc-only Update for CloudTrail: Add required parameters for GetEventSelectors and PutEventSelectors

Release v1.8.1 (2017-03-23)
===

Service Client Updates
---
* `service/applicationdiscoveryservice`: Updates service API, documentation, and paginators
  * Adds export configuration options to the AWS Discovery Service API.
* `service/elbv2`: Updates waiters
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/lambda`: Updates service API and paginators
  * Adds support for new runtime Node.js v6.10 for AWS Lambda service

Release v1.8.0 (2017-03-22)
===

Service Client Updates
---
* `service/codebuild`: Updates service documentation
* `service/directconnect`: Updates service API
  * Deprecated DescribeConnectionLoa, DescribeInterconnectLoa, AllocateConnectionOnInterconnect and DescribeConnectionsOnInterconnect operations in favor of DescribeLoa, DescribeLoa, AllocateHostedConnection and DescribeHostedConnections respectively.
* `service/marketplacecommerceanalytics`: Updates service API, documentation, and paginators
  * This update adds a new data set, us_sales_and_use_tax_records, which enables AWS Marketplace sellers to programmatically access to their U.S. Sales and Use Tax report data.
* `service/pinpoint`: Updates service API and documentation
  * Amazon Pinpoint User Segmentation
  * Added ability to segment endpoints by user attributes in addition to endpoint attributes. Amazon Pinpoint Event Stream Preview
  * Added functionality to publish raw app analytics and campaign events data as events streams to Kinesis and Kinesis Firehose
  * The feature provides developers with increased flexibility of exporting raw events to S3, Redshift, Elasticsearch using a Kinesis Firehose stream or enable real time event processing use cases using a Kinesis stream
* `service/rekognition`: Updates service documentation.

SDK Features
---
* `aws/request`: Add support for context.Context to SDK API operation requests (#1132)
  * Adds support for context.Context to the SDK by adding `WithContext` methods for each API operation, Paginators and Waiters. e.g `PutObjectWithContext`. This change also adds the ability to provide request functional options to the method calls instead of requiring you to use the `Request` API operation method (e.g `PutObjectRequest`).
  * Adds a `Complete` Request handler list that will be called ever time a request is completed. This includes both success and failure. Complete will only be called once per API operation request.
  * `private/waiter` package moved from the private group to `aws/request/waiter` and made publicly available.
  * Adds Context support to all API operations, Waiters(WaitUntil) and Paginators(Pages) methods.
  * Adds Context support for s3manager and s3crypto clients.

SDK Enhancements
---
* `aws/signer/v4`: Adds support for unsigned payload signer config (#1130)
  * Adds configuration option to the v4.Signer to specify the request's body should not be signed. This will only correctly function on services that support unsigned payload. e.g. S3, Glacier.

SDK Bug Fixes
---
* `service/s3`: Fix S3 HostID to be available in S3 request error message (#1131)
  * Adds a new type s3.RequestFailure which exposes the S3 HostID value from a S3 API operation response. This is helpful when you have an error with S3, and need to contact support. Both RequestID and HostID are needed.
* `private/model/api`: Do not return a link if uid is empty (#1133)
  * Fixes SDK's doc generation to not generate API reference doc links if the SDK us unable to create a valid link.
* `aws/request`: Optimization to handler list copy to prevent multiple alloc calls. (#1134)
Release v1.7.9 (2017-03-13)
===

Service Client Updates
---
* `service/devicefarm`: Updates service API, documentation, paginators, and examples
  * Network shaping allows users to simulate network connections and conditions while testing their Android, iOS, and web apps with AWS Device Farm.
* `service/cloudwatchevents`: Updates service API, documentation, and examples

SDK Enhancement
===
* `aws/session`: Add support for side loaded CA bundles (#1117)
  * Adds supports for side loading Certificate Authority bundle files to the SDK using AWS_CA_BUNDLE environment variable or CustomCABundle session option.
* `service/s3/s3crypto`: Add support for AES/CBC/PKCS5Padding (#1124)

SDK Bug
===
* `service/rds`: Fixing issue when not providing `SourceRegion` on cross
region operations (#1127)
* `service/rds`: Enables cross region for `CopyDBClusterSnapshot` and
`CreateDBCluster` (#1128)

Release v1.7.8 (2017-03-10)
===

Service Client Updates
---
* `service/codedeploy`: Updates service paginators
  * Add paginators for Codedeploy
* `service/emr`: Updates service API, documentation, and paginators
  * This release includes support for instance fleets in Amazon EMR.

Release v1.7.7 (2017-03-09)
===

Service Client Updates
---
* `service/apigateway`: Updates service API, documentation, and paginators
  * API Gateway has added support for ACM certificates on custom domain names. Both Amazon-issued certificates and uploaded third-part certificates are supported.
* `service/clouddirectory`: Updates service API, documentation, and paginators
  * Introduces a new Cloud Directory API that enables you to retrieve all available parent paths for any type of object (a node, leaf node, policy node, and index node) in a hierarchy.

Release v1.7.6 (2017-03-09)
===

Service Client Updates
---
* `service/organizations`: Updates service documentation and examples
  * Doc-only Update for Organizations: Add SDK Code Snippets
* `service/workdocs`: Adds new service
  * The Administrative SDKs for Amazon WorkDocs provides full administrator level access to WorkDocs site resources, allowing developers to integrate their applications to manage WorkDocs users, content and permissions programmatically

Release v1.7.5 (2017-03-08)
===

Service Client Updates
---
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * Add support to using encrypted clusters as cross-region replication masters. Update CopyDBClusterSnapshot API to support encrypted cross region copy of Aurora cluster snapshots.

Release v1.7.4 (2017-03-06)
===

Service Client Updates
---
* `service/budgets`: Updates service API and paginators
  * When creating or editing a budget via the AWS Budgets API you can define notifications that are sent to subscribers when the actual or forecasted value for cost or usage exceeds the notificationThreshold associated with the budget notification object. Starting today, the maximum allowed value for the notificationThreshold was raised from 100 to 300. This change was made to give you more flexibility when setting budget notifications.
* `service/cloudtrail`: Updates service documentation and paginators
  * Doc-only update for AWSCloudTrail: Updated links/descriptions
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/opsworkscm`: Updates service API, documentation, and paginators
  * OpsWorks for Chef Automate has added a new field "AssociatePublicIpAddress" to the CreateServer request, "CloudFormationStackArn" to the Server model and "TERMINATED" server state.


Release v1.7.3 (2017-02-28)
===

Service Client Updates
---
* `service/mturk`: Renaming service
  * service/mechanicalturkrequesterservice was renamed to service/mturk. Be sure to change any references of the old client to the new.

Release v1.7.2 (2017-02-28)
===

Service Client Updates
---
* `service/dynamodb`: Updates service API and documentation
  * Release notes: Time to Live (TTL) is a feature that allows you to define when items in a table expire and can be purged from the database, so that you don't have to track expired data and delete it manually. With TTL enabled on a DynamoDB table, you can set a timestamp for deletion on a per-item basis, allowing you to limit storage usage to only those records that are relevant.
* `service/iam`: Updates service API, documentation, and paginators
  * This release adds support for AWS Organizations service control policies (SCPs) to SimulatePrincipalPolicy operation. If there are SCPs associated with the simulated user's account, their effect on the result is captured in the OrganizationDecisionDetail element in the EvaluationResult.
* `service/mechanicalturkrequesterservice`: Adds new service
  * Amazon Mechanical Turk is a web service that provides an on-demand, scalable, human workforce to complete jobs that humans can do better than computers, for example, recognizing objects in photos.
* `service/organizations`: Adds new service
  * AWS Organizations is a web service that enables you to consolidate your multiple AWS accounts into an organization and centrally manage your accounts and their resources.
* `service/dynamodbstreams`: Updates service API, documentation, and paginators
* `service/waf`: Updates service API, documentation, and paginators
  * Aws WAF - For GetSampledRequests action, changed max number of samples from 100 to 500.
* `service/wafregional`: Updates service API, documentation, and paginators

Release v1.7.1 (2017-02-24)
===

Service Client Updates
---
* `service/elasticsearchservice`: Updates service API, documentation, paginators, and examples
  * Added three new API calls to existing Amazon Elasticsearch service to expose Amazon Elasticsearch imposed limits to customers.

Release v1.7.0 (2017-02-23)
===

Service Client Updates
---
* `service/ec2`: Updates service API
  * New EC2 I3 instance type

SDK Bug
---
* `service/s3/s3manager`: Adding support for SSE (#1097)
  * Fixes SSE fields not being applied to a part during multi part upload.

SDK Feature
---
* `aws/session`: Add support for AssumeRoles with MFA (#1088)
  * Adds support for assuming IAM roles with MFA enabled. A TokenProvider func was added to stscreds.AssumeRoleProvider that will be called each time the role's credentials need to be refreshed. A basic token provider that sources the MFA token from stdin as stscreds.StdinTokenProvider.
* `aws/session`: Update SDK examples and docs to use session.Must (#1099)
  * Updates the SDK's example and docs to use session.Must where possible to highlight its usage as apposed to session error checking that is most cases errors will be terminal to the application anyways.
Release v1.6.27 (2017-02-22)
===

Service Client Updates
---
* `service/clouddirectory`: Updates service documentation
  * ListObjectAttributes documentation updated based on forum feedback
* `service/elasticbeanstalk`: Updates service API, documentation, and paginators
  * Elastic Beanstalk adds support for creating and managing custom platform.
* `service/gamelift`: Updates service API, documentation, and paginators
  * Allow developers to configure global queues for creating GameSessions. Allow PlayerData on PlayerSessions to store player-specific data.
* `service/route53`: Updates service API, documentation, and examples
  * Added support for operations CreateVPCAssociationAuthorization and DeleteVPCAssociationAuthorization to throw a ConcurrentModification error when a conflicting modification occurs in parallel to the authorizations in place for a given hosted zone.

Release v1.6.26 (2017-02-21)
===

Service Client Updates
---
* `service/ec2`: Updates service API and documentation
  * Added the billingProduct parameter to the RegisterImage API.

Release v1.6.25 (2017-02-17)
===

Service Client Updates
---
* `service/directconnect`: Updates service API, documentation, and paginators
  * This update will introduce the ability for Direct Connect customers to take advantage of Link Aggregation (LAG).     This allows you to bundle many individual physical interfaces into a single logical interface, referred to as a LAG.     This makes administration much simpler as the majority of configuration is done on the LAG while you are free     to add or remove physical interfaces from the bundle as bandwidth demand increases or decreases. A concrete example     of the simplification added by LAG is that customers need only a single BGP session as opposed to one session per     physical connection.

Release v1.6.24 (2017-02-16)
===

Service Client Updates
---
* `service/cognitoidentity`: Updates service API, documentation, and paginators
  * Allow createIdentityPool and updateIdentityPool API to set server side token check value on identity pool
* `service/configservice`: Updates service API and documentation
  * AWS Config now supports a new test mode for the PutEvaluations API. Set the TestMode parameter to true in your custom rule to verify whether your AWS Lambda function will deliver evaluation results to AWS Config. No updates occur to your existing evaluations, and evaluation results are not sent to AWS Config.

Release v1.6.23 (2017-02-15)
===

Service Client Updates
---
* `service/kms`: Updates service API, documentation, paginators, and examples
  * his release of AWS Key Management Service introduces the ability to tag keys. Tagging keys can help you organize your keys and track your KMS costs in the cost allocation report. This release also increases the maximum length of a key ID to accommodate ARNs that include a long key alias.

Release v1.6.22 (2017-02-14)
===

Service Client Updates
---
* `service/ec2`: Updates service API, documentation, and paginators
  * Adds support for the new Modify Volumes apis.

Release v1.6.21 (2017-02-11)
===

Service Client Updates
---
* `service/storagegateway`: Updates service API, documentation, and paginators
  * File gateway mode in AWS Storage gateway provides access to objects in S3 as files on a Network File System (NFS) mount point. This is done by creating Nfs file shares using existing APIs CreateNfsFileShare. Using the feature in this update, the customer can restrict the clients that have read/write access to the gateway by specifying the list of clients as a list of IP addresses or CIDR blocks. This list can be specified using the API CreateNfsFileShare while creating new file shares, or UpdateNfsFileShare while update existing file shares. To find out the list of clients that have access, the existing API DescribeNfsFileShare will now output the list of clients that have access.

Release v1.6.20 (2017-02-09)
===

Service Client Updates
---
* `service/ec2`: Updates service API and documentation
  * This feature allows customers to associate an IAM profile to running instances that do not have any.
* `service/rekognition`: Updates service API and documentation
  * DetectFaces and IndexFaces operations now return an estimate of the age of the face as an age range.

SDK Features
---
* `aws/endpoints`: Add option to resolve unknown endpoints (#1074)
Release v1.6.19 (2017-02-08)
===

Service Client Updates
---
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glacier`: Updates service examples
	* Doc Update
* `service/lexruntimeservice`: Adds new service
	* Preview release

SDK Bug Fixes
---
* `private/protocol/json`: Fixes json to throw an error if a float number is (+/-)Inf and NaN (#1068)
* `private/model/api`: Fix documentation error listing (#1067)

SDK Features
---
* `private/model`: Add service response error code generation (#1061)

Release v1.6.18 (2017-01-27)
===

Service Client Updates
---
* `service/clouddirectory`: Adds new service
  * Amazon Cloud Directory is a highly scalable, high performance, multi-tenant directory service in the cloud. Its web-based directories make it easy for you to organize and manage application resources such as users, groups, locations, devices, policies, and the rich relationships between them.
* `service/codedeploy`: Updates service API, documentation, and paginators
  * This release of AWS CodeDeploy introduces support for blue/green deployments. In a blue/green deployment, the current set of instances in a deployment group is replaced by new instances that have the latest application revision installed on them. After traffic is rerouted behind a load balancer to the replacement instances, the original instances can be terminated automatically or kept running for other uses.
* `service/ec2`: Updates service API and documentation
  * Adds instance health check functionality to replace unhealthy EC2 Spot fleet instances with fresh ones.
* `service/rds`: Updates service API and documentation
  * Snapshot Engine Version Upgrade

Release v1.6.17 (2017-01-25)
===

Service Client Updates
---
* `service/elbv2`: Updates service API, documentation, and paginators
  * Application Load Balancers now support native Internet Protocol version 6 (IPv6) in an Amazon Virtual Private Cloud (VPC). With this ability, clients can now connect to the Application Load Balancer in a dual-stack mode via either IPv4 or IPv6.
* `service/rds`: Updates service API and documentation
  * Cross Region Read Replica Copying (CreateDBInstanceReadReplica)

Release v1.6.16 (2017-01-24)
===

Service Client Updates
---
* `service/codebuild`: Updates service documentation and paginators
  * Documentation updates
* `service/codecommit`: Updates service API, documentation, and paginators
  * AWS CodeCommit now includes the option to view the differences between a commit and its parent commit from within the console. You can view the differences inline (Unified view) or side by side (Split view). To view information about the differences between a commit and something other than its parent, you can use the AWS CLI and the get-differences and get-blob commands, or you can use the GetDifferences and GetBlob APIs.
* `service/ecs`: Updates service API and documentation
  * Amazon ECS now supports a state for container instances that can be used to drain a container instance in preparation for maintenance or cluster scale down.

Release v1.6.15 (2017-01-20)
===

Service Client Updates
---
* `service/acm`: Updates service API, documentation, and paginators
  * Update for AWS Certificate Manager: Updated response elements for DescribeCertificate API in support of managed renewal
* `service/health`: Updates service documentation

Release v1.6.14 (2017-01-19)
===

Service Client Updates
---
* `service/ec2`: Updates service API, documentation, and paginators
  * Amazon EC2 Spot instances now support dedicated tenancy, providing the ability to run Spot instances single-tenant manner on physically isolated hardware within a VPC to satisfy security, privacy, or other compliance requirements. Dedicated Spot instances can be requested using RequestSpotInstances and RequestSpotFleet.

Release v1.6.13 (2017-01-18)
===

Service Client Updates
---
* `service/rds`: Updates service API, documentation, and paginators

Release v1.6.12 (2017-01-17)
===

Service Client Updates
---
* `service/dynamodb`: Updates service API, documentation, and paginators
  * Tagging Support for Amazon DynamoDB Tables and Indexes
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/glacier`: Updates service API, paginators, and examples
  * Doc-only Update for Glacier: Added code snippets
* `service/polly`: Updates service documentation and examples
  * Doc-only update for Amazon Polly -- added snippets
* `service/rekognition`: Updates service documentation and paginators
  * Added code samples to Rekognition reference topics.
* `service/route53`: Updates service API and paginators
  * Add ca-central-1 and eu-west-2 enum values to CloudWatchRegion enum

Release v1.6.11 (2017-01-16)
===

Service Client Updates
---
* `service/configservice`: Updates service API, documentation, and paginators
* `service/costandusagereportservice`: Adds new service
  * The AWS Cost and Usage Report Service API allows you to enable and disable the Cost & Usage report, as well as modify the report name, the data granularity, and the delivery preferences.
* `service/dynamodb`: Updates service API, documentation, and examples
  * Snippets for the DynamoDB API.
* `service/elasticache`: Updates service API, documentation, and examples
  * Adds new code examples.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.6.10 (2017-01-04)
===

Service Client Updates
---
* `service/configservice`: Updates service API and documentation
  * AWSConfig is planning to add support for OversizedConfigurationItemChangeNotification message type in putConfigRule. After this release customers can use/write rules based on OversizedConfigurationItemChangeNotification mesage type.
* `service/efs`: Updates service API, documentation, and examples
  * Doc-only Update for EFS: Added code snippets
* `service/iam`: Updates service documentation and examples
* `service/lambda`: Updates service documentation and examples
  * Doc only updates for Lambda: Added code snippets
* `service/marketplacecommerceanalytics`: Updates service API and documentation
  * Added support for data set disbursed_amount_by_instance_hours, with historical data available starting 2012-09-04. New data is published to this data set every 30 days.
* `service/rds`: Updates service documentation
  * Updated documentation for CopyDBSnapshot.
* `service/rekognition`: Updates service documentation and examples
  * Doc-only Update for Rekognition: Added code snippets
* `service/snowball`: Updates service examples
* `service/dynamodbstreams`: Updates service API and examples
  * Doc-only Update for DynamoDB Streams:  Added code snippets

SDK Feature
---
* `private/model/api`: Increasing the readability of code generated files. (#1024)
Release v1.6.9 (2016-12-30)
===

Service Client Updates
---
* `service/codedeploy`: Updates service API and documentation
  * CodeDeploy will support Iam Session Arns in addition to Iam User Arns for on premise host authentication.
* `service/ecs`: Updates service API and documentation
  * Amazon EC2 Container Service (ECS) now supports the ability to customize the placement of tasks on container instances.
* `aws/endpoints`: Updated Regions and Endpoints metadata.

Release v1.6.8 (2016-12-22)
===

Service Client Updates
---
* `service/apigateway`: Updates service API and documentation
  * Amazon API Gateway is adding support for generating SDKs in more languages. This update introduces two new operations used to dynamically discover these SDK types and what configuration each type accepts.
* `service/directoryservice`: Updates service documentation
  * Added code snippets for the DS SDKs
* `service/elasticbeanstalk`: Updates service API and documentation
* `service/iam`: Updates service API and documentation
  * Adds service-specific credentials to IAM service to make it easier to onboard CodeCommit customers.  These are username/password credentials that work with a single service.
* `service/kms`: Updates service API, documentation, and examples
  * Update docs and add SDK examples

Release v1.6.7 (2016-12-22)
===

Service Client Updates
---
* `service/ecr`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/rds`: Updates service API and documentation
  * Cross Region Encrypted Snapshot Copying (CopyDBSnapshot)

Release v1.6.6 (2016-12-20)
===

Service Client Updates
---
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/firehose`: Updates service API, documentation, and examples
  * Processing feature enables users to process and modify records before Amazon Firehose delivers them to destinations.
* `service/route53`: Updates service API and documentation
  * Enum updates for eu-west-2 and ca-central-1
* `service/storagegateway`: Updates service API, documentation, and examples
  * File gateway is a new mode in the AWS Storage Gateway that support a file interface into S3, alongside the current block-based volume and VTL storage. File gateway combines a service and virtual software appliance, enabling you to store and retrieve objects in Amazon S3 using industry standard file protocols such as NFS. The software appliance, or gateway, is deployed into your on-premises environment as a virtual machine (VM) running on VMware ESXi. The gateway provides access to objects in S3 as files on a Network File System (NFS) mount point.

Release v1.6.5 (2016-12-19)
===

Service Client Updates
---
* `service/cloudformation`: Updates service documentation
  * Minor doc update for CloudFormation.
* `service/cloudtrail`: Updates service paginators
* `service/cognitoidentity`: Updates service API and documentation
  * We are adding Groups to Cognito user pools. Developers can perform CRUD operations on groups, add and remove users from groups, list users in groups, etc. We are adding fine-grained role-based access control for Cognito identity pools. Developers can configure an identity pool to get the IAM role from an authenticated user's token, or they can configure rules that will map a user to a different role
* `service/applicationdiscoveryservice`: Updates service API and documentation
  * Adds new APIs to group discovered servers into Applications with get summary and neighbors. Includes additional filters for ListConfigurations and DescribeAgents API.
* `service/inspector`: Updates service API, documentation, and examples
  * Doc-only Update for Inspector: Adding SDK code snippets for Inspector
* `service/sqs`: Updates service documentation

SDK Bug Fixes
---
* `aws/request`: Add PriorRequestNotComplete to throttle retry codes (#1011)
  * Fixes: Not retrying when PriorRequestNotComplete #1009

SDK Feature
---
* `private/model/api`: Adds crosslinking to service documentation (#1010)

Release v1.6.4 (2016-12-15)
===

Service Client Updates
---
* `service/cognitoidentityprovider`: Updates service API and documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/ssm`: Updates service API and documentation
  * This will provide customers with access to the Patch Baseline and Patch Compliance APIs.

SDK Bug Fixes
---
* `service/route53`: Fix URL path cleaning for Route53 API requests (#1006)
  * Fixes: SerializationError when using Route53 ChangeResourceRecordSets #1005
* `aws/request`: Add PriorRequestNotComplete to throttle retry codes (#1002)
  * Fixes: Not retrying when PriorRequestNotComplete #1001

Release v1.6.3 (2016-12-14)
===

Service Client Updates
---
* `service/batch`: Adds new service
  * AWS Batch is a batch computing service that lets customers define queues and compute environments and then submit work as batch jobs.
* `service/databasemigrationservice`: Updates service API and documentation
  * Adds support for SSL enabled Oracle endpoints and task modification.
* `service/elasticbeanstalk`: Updates service documentation
* `aws/endpoints`: Updated Regions and Endpoints metadata.
* `service/cloudwatchlogs`: Updates service API and documentation
  * Add support for associating LogGroups with AWSTagris tags
* `service/marketplacecommerceanalytics`: Updates service API and documentation
  * Add new enum to DataSetType: sales_compensation_billed_revenue
* `service/rds`: Updates service documentation
  * Doc-only Update for RDS: New versions available in CreateDBInstance
* `service/sts`: Updates service documentation
  * Adding Code Snippet Examples for SDKs for STS

SDK Bug Fixes
---
* `aws/request`: Fix retrying timeout requests (#981)
  * Fixes: Requests Retrying is broken if the error was caused due to a client timeout #947
* `aws/request`: Fix for Go 1.8 request incorrectly sent with body (#991)
  * Fixes: service/route53: ListHostedZones hangs and then fails with go1.8 #984
* private/protocol/rest: Use RawPath instead of Opaque (#993)
  * Fixes: HTTP2 request failing with REST protocol services, e.g AWS X-Ray
* private/model/api: Generate REST-JSON JSONVersion correctly (#998)
  * Fixes: REST-JSON protocol service code missing JSONVersion metadata.

Release v1.6.2 (2016-12-08)
===

Service Client Updates
---
* `service/cloudfront`: Add lambda function associations to cache behaviors
* `service/codepipeline`: This is a doc-only update request to incorporate some recent minor revisions to the doc content.
* `service/rds`: Updates service API and documentation
* `service/wafregional`: With this new feature, customers can use AWS WAF directly on Application Load Balancers in a VPC within available regions to protect their websites and web services from malicious attacks such as SQL injection, Cross Site Scripting, bad bots, etc.

Release v1.6.1 (2016-12-07)
===

Service Client Updates
---
* `service/config`: Updates service API
* `service/s3`: Updates service API
* `service/sqs`: Updates service API and documentation

Release v1.6.0 (2016-12-06)
===

Service Client Updates
---
* `service/config`: Updates service API and documentation
* `service/ec2`: Updates service API
* `service/sts`: Updates service API, documentation, and examples

SDK Bug Fixes
---
* private/protocol/xml/xmlutil: Fix SDK XML unmarshaler #975
  * Fixes GetBucketACL Grantee required type always nil. #916

SDK Feature
---
* aws/endpoints: Add endpoint metadata to SDK #961
  * Adds Region and Endpoint metadata to the SDK. This allows you to enumerate regions and endpoint metadata based on a defined model embedded in the SDK.

Release v1.5.13 (2016-12-01)
===

Service Client Updates
---
* `service/apigateway`: Updates service API and documentation
* `service/appstream`: Adds new service
* `service/codebuild`: Adds new service
* `service/directconnect`: Updates service API and documentation
* `service/ec2`: Adds new service
* `service/elasticbeanstalk`: Updates service API and documentation
* `service/health`: Adds new service
* `service/lambda`: Updates service API and documentation
* `service/opsworkscm`: Adds new service
* `service/pinpoint`: Adds new service
* `service/shield`: Adds new service
* `service/ssm`: Updates service API and documentation
* `service/states`: Adds new service
* `service/xray`: Adds new service

Release v1.5.12 (2016-11-30)
===

Service Client Updates
---
* `service/lightsail`: Adds new service
* `service/polly`: Adds new service
* `service/rekognition`: Adds new service
* `service/snowball`: Updates service API and documentation

Release v1.5.11 (2016-11-29)
===

Service Client Updates
---
`service/s3`: Updates service API and documentation

Release v1.5.10 (2016-11-22)
===

Service Client Updates
---
* `service/cloudformation`: Updates service API and documentation
* `service/glacier`: Updates service API, documentation, and examples
* `service/route53`: Updates service API and documentation
* `service/s3`: Updates service API and documentation

SDK Bug Fixes
---
* `private/protocol/xml/xmlutil`: Fixes xml marshaler to unmarshal properly
into tagged fields
[#916](https://github.com/aws/aws-sdk-go/issues/916)

Release v1.5.9 (2016-11-22)
===

Service Client Updates
---
* `service/cloudtrail`: Updates service API and documentation
* `service/ecs`: Updates service API and documentation

Release v1.5.8 (2016-11-18)
===

Service Client Updates
---
* `service/application-autoscaling`: Updates service API and documentation
* `service/elasticmapreduce`: Updates service API and documentation
* `service/elastictranscoder`: Updates service API, documentation, and examples
* `service/gamelift`: Updates service API and documentation
* `service/lambda`: Updates service API and documentation

Release v1.5.7 (2016-11-18)
===

Service Client Updates
---
* `service/apigateway`: Updates service API and documentation
* `service/meteringmarketplace`: Updates service API and documentation
* `service/monitoring`: Updates service API and documentation
* `service/sqs`: Updates service API, documentation, and examples

Release v1.5.6 (2016-11-16)
===

Service Client Updates
---
`service/route53`: Updates service API and documentation
`service/servicecatalog`: Updates service API and documentation

Release v1.5.5 (2016-11-15)
===

Service Client Updates
---
* `service/ds`: Updates service API and documentation
* `service/elasticache`: Updates service API and documentation
* `service/kinesis`: Updates service API and documentation

Release v1.5.4 (2016-11-15)
===

Service Client Updates
---
* `service/cognito-idp`: Updates service API and documentation

Release v1.5.3 (2016-11-11)
===

Service Client Updates
---
* `service/cloudformation`: Updates service documentation and examples
* `service/logs`: Updates service API and documentation

Release v1.5.2 (2016-11-03)
===

Service Client Updates
---
* `service/directconnect`: Updates service API and documentation

Release v1.5.1 (2016-11-02)
===

Service Client Updates
---
* `service/email`: Updates service API and documentation

Release v1.5.0 (2016-11-01)
===

Service Client Updates
---
* `service/cloudformation`: Updates service API and documentation
* `service/ecr`: Updates service paginators

SDK Feature Updates
---
* `private/model/api`: Add generated setters for API parameters (#918)
  * Adds setters to the SDK's API parameter types, and are a convenience method that reduce the need to use `aws.String` and like utility.

Release v1.4.22 (2016-10-25)
===

Service Client Updates
---
* `service/elasticloadbalancingv2`: Updates service documentation.
* `service/autoscaling`: Updates service documentation.

Release v1.4.21 (2016-10-24)
===

Service Client Updates
---
* `service/sms`: AWS Server Migration Service (SMS) is an agentless service which makes it easier and faster for you to migrate thousands of on-premises workloads to AWS. AWS SMS allows you to automate, schedule, and track incremental replications of live server volumes, making it easier for you to coordinate large-scale server migrations.
* `service/ecs`: Updates documentation.

SDK Feature Updates
---
* `private/models/api`: Improve code generation of documentation.

Release v1.4.20 (2016-10-20)
===

Service Client Updates
---
* `service/budgets`: Adds new service, AWS Budgets.
* `service/waf`: Updates service documentation.

Release v1.4.19 (2016-10-18)
===

Service Client Updates
---
* `service/cloudfront`: Updates service API and documentation.
  * Ability to use Amazon CloudFront to deliver your content both via IPv6 and IPv4 using HTTP/HTTPS.
* `service/configservice`: Update service API and documentation.
* `service/iot`: Updates service API and documentation.
* `service/kinesisanalytics`: Updates service API and documentation.
  * Whenever Amazon Kinesis Analytics is not able to detect schema for the given streaming source on DiscoverInputSchema API, we would return the raw records that was sampled to detect the schema.
* `service/rds`: Updates service API and documentation.
  * Amazon Aurora integrates with other AWS services to allow you to extend your Aurora DB cluster to utilize other capabilities in the AWS cloud. Permission to access other AWS services is granted by creating an IAM role with the necessary permissions, and then associating the role with your DB cluster.

SDK Feature Updates
---
* `service/dynamodb/dynamodbattribute`: Add UnmarshalListOfMaps #897
  * Adds support for unmarshaling a list of maps. This is useful for unmarshaling the DynamoDB AttributeValue list of maps returned by APIs like Query and Scan.

Release v1.4.18 (2016-10-17)
===

Service Model Updates
---
* `service/route53`: Updates service API and documentation.

Release v1.4.17
===

Service Model Updates
---
* `service/acm`: Update service API, and documentation.
  * This change allows users to import third-party SSL/TLS certificates into ACM.
* `service/elasticbeanstalk`: Update service API, documentation, and pagination.
  * Elastic Beanstalk DescribeApplicationVersions API is being updated to support pagination.
* `service/gamelift`: Update service API, and documentation.
  * New APIs to protect game developer resource (builds, alias, fleets, instances, game sessions and player sessions) against abuse.

SDK Features
---
* `service/s3`: Add support for accelerate with dualstack [#887](https://github.com/aws/aws-sdk-go/issues/887)

Release v1.4.16 (2016-10-13)
===

Service Model Updates
---
* `service/ecr`: Update Amazon EC2 Container Registry service model
  * DescribeImages is a new api used to expose image metadata which today includes image size and image creation timestamp.
* `service/elasticache`: Update Amazon ElastiCache service model
  * Elasticache is launching a new major engine release of Redis, 3.2 (providing stability updates and new command sets over 2.8), as well as ElasticSupport for enabling Redis Cluster in 3.2, which provides support for multiple node groups to horizontally scale data, as well as superior engine failover capabilities

SDK Bug Fixes
---
* `aws/session`: Skip shared config on read errors [#883](https://github.com/aws/aws-sdk-go/issues/883)
* `aws/signer/v4`: Add support for URL.EscapedPath to signer [#885](https://github.com/aws/aws-sdk-go/issues/885)

SDK Features
---
* `private/model/api`: Add docs for errors to API operations [#881](https://github.com/aws/aws-sdk-go/issues/881)
* `private/model/api`: Improve field and waiter doc strings [#879](https://github.com/aws/aws-sdk-go/issues/879)
* `service/dynamodb/dynamodbattribute`: Allow multiple struct tag elements [#886](https://github.com/aws/aws-sdk-go/issues/886)
* Add build tags to internal SDK tools [#880](https://github.com/aws/aws-sdk-go/issues/880)

Release v1.4.15 (2016-10-06)
===

Service Model Updates
---
* `service/cognitoidentityprovider`: Update Amazon Cognito Identity Provider service model
* `service/devicefarm`: Update AWS Device Farm documentation
* `service/opsworks`: Update AWS OpsWorks service model
* `service/s3`: Update Amazon Simple Storage Service model
* `service/waf`: Update AWS WAF service model

SDK Bug Fixes
---
* `aws/request`: Fix HTTP Request Body race condition [#874](https://github.com/aws/aws-sdk-go/issues/874)

SDK Feature Updates
---
* `aws/ec2metadata`: Add support for EC2 User Data [#872](https://github.com/aws/aws-sdk-go/issues/872)
* `aws/signer/v4`: Remove logic determining if request needs to be resigned [#876](https://github.com/aws/aws-sdk-go/issues/876)

Release v1.4.14 (2016-09-29)
===
* `service/ec2`:  api, documentation, and paginators updates.
* `service/s3`:  api and documentation updates.

Release v1.4.13 (2016-09-27)
===
* `service/codepipeline`:  documentation updates.
* `service/cloudformation`:  api and documentation updates.
* `service/kms`:  documentation updates.
* `service/elasticfilesystem`:  documentation updates.
* `service/snowball`:  documentation updates.
