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
  * Amazon Web Services announced the immediate availability of two additional alarm configuration rules for Amazon CloudWatch Alarms. The first rule is for configuring missing data treatment. Customers have the options to treat missing data as alarm threshold breached, alarm threshold not breached, maintain alarm state and the current default treatment. The second rule is for alarms based on percentiles metrics that can trigger unnecassarily if the percentile is calculated from a small number of samples. The new rule can treat percentiles with low sample counts as same as missing data. If the first rule is enabled, the same treatment will be applied when an alarm encounters a percentile with low sample counts.

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
  * Adds configuration option to the v4.Signer to specify the request's body should not be signed. This will only correclty function on services that support unsigned payload. e.g. S3, Glacier. 

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
