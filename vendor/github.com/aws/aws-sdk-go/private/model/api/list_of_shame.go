package api

// shamelist is used to not rename certain operation's input and output shapes.
// We need to maintain backwards compatibility with pre-existing services. Since
// not generating unique input/output shapes is not desired, we will generate
// unique input/output shapes for new operations.
var shamelist = map[string]map[string]struct {
	input  bool
	output bool
}{
	"APIGateway": {
		"CreateApiKey": {
			output: true,
		},
		"CreateAuthorizer": {
			output: true,
		},
		"CreateBasePathMapping": {
			output: true,
		},
		"CreateDeployment": {
			output: true,
		},
		"CreateDocumentationPart": {
			output: true,
		},
		"CreateDocumentationVersion": {
			output: true,
		},
		"CreateDomainName": {
			output: true,
		},
		"CreateModel": {
			output: true,
		},
		"CreateResource": {
			output: true,
		},
		"CreateRestApi": {
			output: true,
		},
		"CreateStage": {
			output: true,
		},
		"CreateUsagePlan": {
			output: true,
		},
		"CreateUsagePlanKey": {
			output: true,
		},
		"GenerateClientCertificate": {
			output: true,
		},
		"GetAccount": {
			output: true,
		},
		"GetApiKey": {
			output: true,
		},
		"GetAuthorizer": {
			output: true,
		},
		"GetBasePathMapping": {
			output: true,
		},
		"GetClientCertificate": {
			output: true,
		},
		"GetDeployment": {
			output: true,
		},
		"GetDocumentationPart": {
			output: true,
		},
		"GetDocumentationVersion": {
			output: true,
		},
		"GetDomainName": {
			output: true,
		},
		"GetIntegration": {
			output: true,
		},
		"GetIntegrationResponse": {
			output: true,
		},
		"GetMethod": {
			output: true,
		},
		"GetMethodResponse": {
			output: true,
		},
		"GetModel": {
			output: true,
		},
		"GetResource": {
			output: true,
		},
		"GetRestApi": {
			output: true,
		},
		"GetSdkType": {
			output: true,
		},
		"GetStage": {
			output: true,
		},
		"GetUsage": {
			output: true,
		},
		"GetUsagePlan": {
			output: true,
		},
		"GetUsagePlanKey": {
			output: true,
		},
		"ImportRestApi": {
			output: true,
		},
		"PutIntegration": {
			output: true,
		},
		"PutIntegrationResponse": {
			output: true,
		},
		"PutMethod": {
			output: true,
		},
		"PutMethodResponse": {
			output: true,
		},
		"PutRestApi": {
			output: true,
		},
		"UpdateAccount": {
			output: true,
		},
		"UpdateApiKey": {
			output: true,
		},
		"UpdateAuthorizer": {
			output: true,
		},
		"UpdateBasePathMapping": {
			output: true,
		},
		"UpdateClientCertificate": {
			output: true,
		},
		"UpdateDeployment": {
			output: true,
		},
		"UpdateDocumentationPart": {
			output: true,
		},
		"UpdateDocumentationVersion": {
			output: true,
		},
		"UpdateDomainName": {
			output: true,
		},
		"UpdateIntegration": {
			output: true,
		},
		"UpdateIntegrationResponse": {
			output: true,
		},
		"UpdateMethod": {
			output: true,
		},
		"UpdateMethodResponse": {
			output: true,
		},
		"UpdateModel": {
			output: true,
		},
		"UpdateResource": {
			output: true,
		},
		"UpdateRestApi": {
			output: true,
		},
		"UpdateStage": {
			output: true,
		},
		"UpdateUsage": {
			output: true,
		},
		"UpdateUsagePlan": {
			output: true,
		},
	},
	"AutoScaling": {
		"ResumeProcesses": {
			input: true,
		},
		"SuspendProcesses": {
			input: true,
		},
	},
	"CognitoIdentity": {
		"CreateIdentityPool": {
			output: true,
		},
		"DescribeIdentity": {
			output: true,
		},
		"DescribeIdentityPool": {
			output: true,
		},
		"UpdateIdentityPool": {
			input:  true,
			output: true,
		},
	},
	"DirectConnect": {
		"AllocateConnectionOnInterconnect": {
			output: true,
		},
		"AllocateHostedConnection": {
			output: true,
		},
		"AllocatePrivateVirtualInterface": {
			output: true,
		},
		"AllocatePublicVirtualInterface": {
			output: true,
		},
		"AssociateConnectionWithLag": {
			output: true,
		},
		"AssociateHostedConnection": {
			output: true,
		},
		"AssociateVirtualInterface": {
			output: true,
		},
		"CreateConnection": {
			output: true,
		},
		"CreateInterconnect": {
			output: true,
		},
		"CreateLag": {
			output: true,
		},
		"CreatePrivateVirtualInterface": {
			output: true,
		},
		"CreatePublicVirtualInterface": {
			output: true,
		},
		"DeleteConnection": {
			output: true,
		},
		"DeleteLag": {
			output: true,
		},
		"DescribeConnections": {
			output: true,
		},
		"DescribeConnectionsOnInterconnect": {
			output: true,
		},
		"DescribeHostedConnections": {
			output: true,
		},
		"DescribeLoa": {
			output: true,
		},
		"DisassociateConnectionFromLag": {
			output: true,
		},
		"UpdateLag": {
			output: true,
		},
	},
	"EC2": {
		"AttachVolume": {
			output: true,
		},
		"CreateSnapshot": {
			output: true,
		},
		"CreateVolume": {
			output: true,
		},
		"DetachVolume": {
			output: true,
		},
		"RunInstances": {
			output: true,
		},
	},
	"EFS": {
		"CreateFileSystem": {
			output: true,
		},
		"CreateMountTarget": {
			output: true,
		},
	},
	"ElastiCache": {
		"AddTagsToResource": {
			output: true,
		},
		"ListTagsForResource": {
			output: true,
		},
		"ModifyCacheParameterGroup": {
			output: true,
		},
		"RemoveTagsFromResource": {
			output: true,
		},
		"ResetCacheParameterGroup": {
			output: true,
		},
	},
	"ElasticBeanstalk": {
		"ComposeEnvironments": {
			output: true,
		},
		"CreateApplication": {
			output: true,
		},
		"CreateApplicationVersion": {
			output: true,
		},
		"CreateConfigurationTemplate": {
			output: true,
		},
		"CreateEnvironment": {
			output: true,
		},
		"DescribeEnvironments": {
			output: true,
		},
		"TerminateEnvironment": {
			output: true,
		},
		"UpdateApplication": {
			output: true,
		},
		"UpdateApplicationVersion": {
			output: true,
		},
		"UpdateConfigurationTemplate": {
			output: true,
		},
		"UpdateEnvironment": {
			output: true,
		},
	},
	"Glacier": {
		"DescribeJob": {
			output: true,
		},
		"UploadArchive": {
			output: true,
		},
		"CompleteMultipartUpload": {
			output: true,
		},
	},
	"IAM": {
		"GetContextKeysForCustomPolicy": {
			output: true,
		},
		"GetContextKeysForPrincipalPolicy": {
			output: true,
		},
		"SimulateCustomPolicy": {
			output: true,
		},
		"SimulatePrincipalPolicy": {
			output: true,
		},
	},
	"Kinesis": {
		"DisableEnhancedMonitoring": {
			output: true,
		},
		"EnableEnhancedMonitoring": {
			output: true,
		},
	},
	"KMS": {
		"ListGrants": {
			output: true,
		},
		"ListRetirableGrants": {
			output: true,
		},
	},
	"Lambda": {
		"CreateAlias": {
			output: true,
		},
		"CreateEventSourceMapping": {
			output: true,
		},
		"CreateFunction": {
			output: true,
		},
		"DeleteEventSourceMapping": {
			output: true,
		},
		"GetAlias": {
			output: true,
		},
		"GetEventSourceMapping": {
			output: true,
		},
		"GetFunctionConfiguration": {
			output: true,
		},
		"PublishVersion": {
			output: true,
		},
		"UpdateAlias": {
			output: true,
		},
		"UpdateEventSourceMapping": {
			output: true,
		},
		"UpdateFunctionCode": {
			output: true,
		},
		"UpdateFunctionConfiguration": {
			output: true,
		},
	},
	"RDS": {
		"ModifyDBClusterParameterGroup": {
			output: true,
		},
		"ModifyDBParameterGroup": {
			output: true,
		},
		"ResetDBClusterParameterGroup": {
			output: true,
		},
		"ResetDBParameterGroup": {
			output: true,
		},
	},
	"Redshift": {
		"DescribeLoggingStatus": {
			output: true,
		},
		"DisableLogging": {
			output: true,
		},
		"EnableLogging": {
			output: true,
		},
		"ModifyClusterParameterGroup": {
			output: true,
		},
		"ResetClusterParameterGroup": {
			output: true,
		},
	},
	"S3": {
		"GetBucketNotification": {
			input:  true,
			output: true,
		},
		"GetBucketNotificationConfiguration": {
			input:  true,
			output: true,
		},
	},
	"SWF": {
		"CountClosedWorkflowExecutions": {
			output: true,
		},
		"CountOpenWorkflowExecutions": {
			output: true,
		},
		"CountPendingActivityTasks": {
			output: true,
		},
		"CountPendingDecisionTasks": {
			output: true,
		},
		"ListClosedWorkflowExecutions": {
			output: true,
		},
		"ListOpenWorkflowExecutions": {
			output: true,
		},
	},
}
