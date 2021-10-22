# Change History

## Breaking Changes

### Removed Constants

1. AuthenticationType.AuthenticationTypeAuthenticationTypeAnonymous
1. AuthenticationType.AuthenticationTypeAuthenticationTypeBasic
1. AuthenticationType.AuthenticationTypeAuthenticationTypeClientCertificate
1. AuthenticationType.AuthenticationTypeAuthenticationTypeWebLinkedServiceTypeProperties
1. AuthorizationType.AuthorizationTypeAuthorizationTypeKey
1. AuthorizationType.AuthorizationTypeAuthorizationTypeLinkedIntegrationRuntimeType
1. AuthorizationType.AuthorizationTypeAuthorizationTypeRBAC
1. Type.TypeTypeAzureKeyVaultSecret
1. Type.TypeTypeSecretBase
1. Type.TypeTypeSecureString

### Signature Changes

#### Struct Fields

1. CommonDataServiceForAppsLinkedServiceTypeProperties.AuthenticationType changed type from DynamicsAuthenticationType to interface{}
1. CommonDataServiceForAppsLinkedServiceTypeProperties.DeploymentType changed type from DynamicsDeploymentType to interface{}
1. DynamicsCrmLinkedServiceTypeProperties.AuthenticationType changed type from DynamicsAuthenticationType to interface{}
1. DynamicsCrmLinkedServiceTypeProperties.DeploymentType changed type from DynamicsDeploymentType to interface{}
1. JSONWriteSettings.FilePattern changed type from JSONWriteFilePattern to interface{}

## Additive Changes

### New Constants

1. AuthenticationType.AuthenticationTypeAnonymous
1. AuthenticationType.AuthenticationTypeBasic
1. AuthenticationType.AuthenticationTypeClientCertificate
1. AuthenticationType.AuthenticationTypeWebLinkedServiceTypeProperties
1. AuthorizationType.AuthorizationTypeKey
1. AuthorizationType.AuthorizationTypeLinkedIntegrationRuntimeType
1. AuthorizationType.AuthorizationTypeRBAC
1. CompressionCodec.CompressionCodecBzip2
1. CompressionCodec.CompressionCodecDeflate
1. CompressionCodec.CompressionCodecGzip
1. CompressionCodec.CompressionCodecLz4
1. CompressionCodec.CompressionCodecLzo
1. CompressionCodec.CompressionCodecNone
1. CompressionCodec.CompressionCodecSnappy
1. CompressionCodec.CompressionCodecTar
1. CompressionCodec.CompressionCodecTarGZip
1. CompressionCodec.CompressionCodecZipDeflate
1. DatasetCompressionLevel.DatasetCompressionLevelFastest
1. DatasetCompressionLevel.DatasetCompressionLevelOptimal
1. HdiNodeTypes.HdiNodeTypesHeadnode
1. HdiNodeTypes.HdiNodeTypesWorkernode
1. HdiNodeTypes.HdiNodeTypesZookeeper
1. IntegrationRuntimeEntityReferenceType.IntegrationRuntimeEntityReferenceTypeCredentialReference
1. JSONFormatFilePattern.JSONFormatFilePatternArrayOfObjects
1. JSONFormatFilePattern.JSONFormatFilePatternSetOfObjects
1. ServicePrincipalCredentialType.ServicePrincipalCredentialTypeServicePrincipalCert
1. ServicePrincipalCredentialType.ServicePrincipalCredentialTypeServicePrincipalKey
1. Type.TypeAzureKeyVaultSecret
1. Type.TypeSecretBase
1. Type.TypeSecureString

### New Funcs

1. ArmIDWrapper.MarshalJSON() ([]byte, error)
1. ConnectionStateProperties.MarshalJSON() ([]byte, error)
1. ExposureControlResponse.MarshalJSON() ([]byte, error)
1. IntegrationRuntimeNodeIPAddress.MarshalJSON() ([]byte, error)
1. LinkedIntegrationRuntime.MarshalJSON() ([]byte, error)
1. ManagedIntegrationRuntimeError.MarshalJSON() ([]byte, error)
1. ManagedIntegrationRuntimeOperationResult.MarshalJSON() ([]byte, error)
1. ManagedIntegrationRuntimeStatusTypeProperties.MarshalJSON() ([]byte, error)
1. PipelineRunInvokedBy.MarshalJSON() ([]byte, error)
1. PossibleCompressionCodecValues() []CompressionCodec
1. PossibleDatasetCompressionLevelValues() []DatasetCompressionLevel
1. PossibleHdiNodeTypesValues() []HdiNodeTypes
1. PossibleJSONFormatFilePatternValues() []JSONFormatFilePattern
1. PossibleServicePrincipalCredentialTypeValues() []ServicePrincipalCredentialType
1. PrivateLinkResourceProperties.MarshalJSON() ([]byte, error)
1. SubResource.MarshalJSON() ([]byte, error)
1. TriggerSubscriptionOperationStatus.MarshalJSON() ([]byte, error)

### Struct Changes

#### New Structs

1. MetadataItem

#### New Struct Fields

1. AmazonMWSSource.DisableMetricsCollection
1. AmazonRedshiftSource.DisableMetricsCollection
1. AmazonS3CompatibleReadSettings.DisableMetricsCollection
1. AmazonS3ReadSettings.DisableMetricsCollection
1. AvroSink.DisableMetricsCollection
1. AvroSource.DisableMetricsCollection
1. AzureBlobFSReadSettings.DisableMetricsCollection
1. AzureBlobFSSink.DisableMetricsCollection
1. AzureBlobFSSink.Metadata
1. AzureBlobFSSource.DisableMetricsCollection
1. AzureBlobFSWriteSettings.DisableMetricsCollection
1. AzureBlobStorageReadSettings.DisableMetricsCollection
1. AzureBlobStorageWriteSettings.DisableMetricsCollection
1. AzureDataExplorerSink.DisableMetricsCollection
1. AzureDataExplorerSource.DisableMetricsCollection
1. AzureDataLakeStoreReadSettings.DisableMetricsCollection
1. AzureDataLakeStoreSink.DisableMetricsCollection
1. AzureDataLakeStoreSource.DisableMetricsCollection
1. AzureDataLakeStoreWriteSettings.DisableMetricsCollection
1. AzureDatabricksDeltaLakeSink.DisableMetricsCollection
1. AzureDatabricksDeltaLakeSource.DisableMetricsCollection
1. AzureFileStorageReadSettings.DisableMetricsCollection
1. AzureFileStorageWriteSettings.DisableMetricsCollection
1. AzureMariaDBSource.DisableMetricsCollection
1. AzureMySQLSink.DisableMetricsCollection
1. AzureMySQLSource.DisableMetricsCollection
1. AzurePostgreSQLSink.DisableMetricsCollection
1. AzurePostgreSQLSource.DisableMetricsCollection
1. AzureQueueSink.DisableMetricsCollection
1. AzureSQLSink.DisableMetricsCollection
1. AzureSQLSource.DisableMetricsCollection
1. AzureSearchIndexSink.DisableMetricsCollection
1. AzureTableSink.DisableMetricsCollection
1. AzureTableSource.DisableMetricsCollection
1. BinarySink.DisableMetricsCollection
1. BinarySource.DisableMetricsCollection
1. BlobSink.DisableMetricsCollection
1. BlobSink.Metadata
1. BlobSource.DisableMetricsCollection
1. CassandraSource.DisableMetricsCollection
1. CommonDataServiceForAppsSink.DisableMetricsCollection
1. CommonDataServiceForAppsSource.DisableMetricsCollection
1. ConcurSource.DisableMetricsCollection
1. CopySink.DisableMetricsCollection
1. CopySource.DisableMetricsCollection
1. CosmosDbMongoDbAPISink.DisableMetricsCollection
1. CosmosDbMongoDbAPISource.DisableMetricsCollection
1. CosmosDbSQLAPISink.DisableMetricsCollection
1. CosmosDbSQLAPISource.DisableMetricsCollection
1. CouchbaseSource.DisableMetricsCollection
1. Db2Source.DisableMetricsCollection
1. DelimitedTextSink.DisableMetricsCollection
1. DelimitedTextSource.DisableMetricsCollection
1. DocumentDbCollectionSink.DisableMetricsCollection
1. DocumentDbCollectionSource.DisableMetricsCollection
1. DrillSource.DisableMetricsCollection
1. DynamicsAXSource.DisableMetricsCollection
1. DynamicsCrmSink.DisableMetricsCollection
1. DynamicsCrmSource.DisableMetricsCollection
1. DynamicsSink.DisableMetricsCollection
1. DynamicsSource.DisableMetricsCollection
1. EloquaSource.DisableMetricsCollection
1. ExcelDatasetTypeProperties.SheetIndex
1. ExcelSource.DisableMetricsCollection
1. FileServerReadSettings.DisableMetricsCollection
1. FileServerWriteSettings.DisableMetricsCollection
1. FileSystemSink.DisableMetricsCollection
1. FileSystemSource.DisableMetricsCollection
1. FtpReadSettings.DisableMetricsCollection
1. GoogleAdWordsSource.DisableMetricsCollection
1. GoogleBigQuerySource.DisableMetricsCollection
1. GoogleCloudStorageReadSettings.DisableMetricsCollection
1. GreenplumSource.DisableMetricsCollection
1. HBaseSource.DisableMetricsCollection
1. HTTPReadSettings.DisableMetricsCollection
1. HTTPSource.DisableMetricsCollection
1. HdfsReadSettings.DisableMetricsCollection
1. HdfsSource.DisableMetricsCollection
1. HiveSource.DisableMetricsCollection
1. HubspotSource.DisableMetricsCollection
1. ImpalaSource.DisableMetricsCollection
1. InformixSink.DisableMetricsCollection
1. InformixSource.DisableMetricsCollection
1. IntegrationRuntimeSsisProperties.ManagedCredential
1. JSONSink.DisableMetricsCollection
1. JSONSource.DisableMetricsCollection
1. JiraSource.DisableMetricsCollection
1. MagentoSource.DisableMetricsCollection
1. MariaDBSource.DisableMetricsCollection
1. MarketoSource.DisableMetricsCollection
1. MicrosoftAccessSink.DisableMetricsCollection
1. MicrosoftAccessSource.DisableMetricsCollection
1. MongoDbAtlasSink.DisableMetricsCollection
1. MongoDbAtlasSource.DisableMetricsCollection
1. MongoDbSource.DisableMetricsCollection
1. MongoDbV2Sink.DisableMetricsCollection
1. MongoDbV2Source.DisableMetricsCollection
1. MySQLSource.DisableMetricsCollection
1. NetezzaSource.DisableMetricsCollection
1. ODataSource.DisableMetricsCollection
1. OdbcSink.DisableMetricsCollection
1. OdbcSource.DisableMetricsCollection
1. Office365Source.DisableMetricsCollection
1. OracleCloudStorageReadSettings.DisableMetricsCollection
1. OracleServiceCloudSource.DisableMetricsCollection
1. OracleSink.DisableMetricsCollection
1. OracleSource.DisableMetricsCollection
1. OrcSink.DisableMetricsCollection
1. OrcSource.DisableMetricsCollection
1. ParquetSink.DisableMetricsCollection
1. ParquetSource.DisableMetricsCollection
1. PaypalSource.DisableMetricsCollection
1. PhoenixSource.DisableMetricsCollection
1. PostgreSQLSource.DisableMetricsCollection
1. PrestoSource.DisableMetricsCollection
1. QuickBooksSource.DisableMetricsCollection
1. RelationalSource.DisableMetricsCollection
1. ResponsysSource.DisableMetricsCollection
1. RestSink.DisableMetricsCollection
1. RestSource.DisableMetricsCollection
1. SQLDWSink.DisableMetricsCollection
1. SQLDWSource.DisableMetricsCollection
1. SQLMISink.DisableMetricsCollection
1. SQLMISource.DisableMetricsCollection
1. SQLServerSink.DisableMetricsCollection
1. SQLServerSource.DisableMetricsCollection
1. SQLSink.DisableMetricsCollection
1. SQLSource.DisableMetricsCollection
1. SalesforceMarketingCloudSource.DisableMetricsCollection
1. SalesforceServiceCloudSink.DisableMetricsCollection
1. SalesforceServiceCloudSource.DisableMetricsCollection
1. SalesforceSink.DisableMetricsCollection
1. SalesforceSource.DisableMetricsCollection
1. SapBwSource.DisableMetricsCollection
1. SapCloudForCustomerSink.DisableMetricsCollection
1. SapCloudForCustomerSource.DisableMetricsCollection
1. SapEccSource.DisableMetricsCollection
1. SapHanaSource.DisableMetricsCollection
1. SapOpenHubSource.DisableMetricsCollection
1. SapTableSource.DisableMetricsCollection
1. ServiceNowSource.DisableMetricsCollection
1. SftpReadSettings.DisableMetricsCollection
1. SftpWriteSettings.DisableMetricsCollection
1. SharePointOnlineListSource.DisableMetricsCollection
1. ShopifySource.DisableMetricsCollection
1. SnowflakeSink.DisableMetricsCollection
1. SnowflakeSource.DisableMetricsCollection
1. SparkSource.DisableMetricsCollection
1. SquareSource.DisableMetricsCollection
1. StoreReadSettings.DisableMetricsCollection
1. StoreWriteSettings.DisableMetricsCollection
1. SybaseSource.DisableMetricsCollection
1. TabularSource.DisableMetricsCollection
1. TeradataSource.DisableMetricsCollection
1. VerticaSource.DisableMetricsCollection
1. WebSource.DisableMetricsCollection
1. XMLSource.DisableMetricsCollection
1. XeroSource.DisableMetricsCollection
1. ZohoSource.DisableMetricsCollection
