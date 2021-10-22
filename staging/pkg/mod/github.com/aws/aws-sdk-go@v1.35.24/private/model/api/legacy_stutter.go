package api

type stutterNames struct {
	Operations map[string]string
	Shapes     map[string]string
	ShapeOrder []string
}

var legacyStutterNames = map[string]stutterNames{
	"WorkSpaces": {
		Shapes: map[string]string{
			"WorkspacesIpGroup":      "IpGroup",
			"WorkspacesIpGroupsList": "IpGroupsList",
		},
	},
	"WorkMail": {
		Shapes: map[string]string{
			"WorkMailIdentifier": "Identifier",
		},
	},
	"WAF": {
		Shapes: map[string]string{
			"WAFInvalidPermissionPolicyException":   "InvalidPermissionPolicyException",
			"WAFInvalidOperationException":          "InvalidOperationException",
			"WAFInternalErrorException":             "InternalErrorException",
			"WAFDisallowedNameException":            "DisallowedNameException",
			"WAFReferencedItemException":            "ReferencedItemException",
			"WAFInvalidParameterException":          "InvalidParameterException",
			"WAFLimitsExceededException":            "LimitsExceededException",
			"WAFNonexistentContainerException":      "NonexistentContainerException",
			"WAFInvalidAccountException":            "InvalidAccountException",
			"WAFSubscriptionNotFoundException":      "SubscriptionNotFoundException",
			"WAFBadRequestException":                "BadRequestException",
			"WAFNonexistentItemException":           "NonexistentItemException",
			"WAFServiceLinkedRoleErrorException":    "ServiceLinkedRoleErrorException",
			"WAFNonEmptyEntityException":            "NonEmptyEntityException",
			"WAFTagOperationInternalErrorException": "TagOperationInternalErrorException",
			"WAFStaleDataException":                 "StaleDataException",
			"WAFTagOperationException":              "TagOperationException",
			"WAFInvalidRegexPatternException":       "InvalidRegexPatternException",
		},
	},
	"Translate": {
		Operations: map[string]string{
			"TranslateText": "Text",
		},
		Shapes: map[string]string{
			"TranslateTextRequest":  "TextRequest",
			"TranslateTextResponse": "TextResponse",
		},
	},
	"Storage Gateway": {
		Shapes: map[string]string{
			"StorageGatewayError": "Error",
		},
	},
	"Snowball": {
		Shapes: map[string]string{
			"SnowballType":     "Type",
			"SnowballCapacity": "Capacity",
		},
	},
	"S3": {
		Shapes: map[string]string{
			"S3KeyFilter": "KeyFilter",
			"S3Location":  "Location",
		},
	},
	"Rekognition": {
		Shapes: map[string]string{
			"RekognitionUniqueId": "UniqueId",
		},
	},
	"QuickSight": {
		Shapes: map[string]string{
			"QuickSightUserNotFoundException": "UserNotFoundException",
		},
	},
	"Marketplace Commerce Analytics": {
		Shapes: map[string]string{
			"MarketplaceCommerceAnalyticsException": "Exception",
		},
	},
	"KMS": {
		Shapes: map[string]string{
			"KMSInternalException":     "InternalException",
			"KMSInvalidStateException": "InvalidStateException",
		},
	},
	"Kinesis Analytics": {
		Shapes: map[string]string{
			"KinesisAnalyticsARN": "ARN",
		},
	},
	"IoT Events": {
		ShapeOrder: []string{
			"Action",
			"IotEventsAction",
		},
		Shapes: map[string]string{
			"Action":          "ActionData",
			"IotEventsAction": "Action",
		},
	},
	"Inspector": {
		Shapes: map[string]string{
			"InspectorServiceAttributes": "ServiceAttributes",
			"InspectorEvent":             "Event",
		},
	},
	"GuardDuty": {
		Shapes: map[string]string{
			"GuardDutyArn": "Arn",
		},
	},
	"GroundStation": {
		Shapes: map[string]string{
			"GroundStationList": "List",
			"GroundStationData": "Data",
		},
	},
	"Glue": {
		ShapeOrder: []string{
			"Table",
			"GlueTable",
			"GlueTables",
			"GlueResourceArn",
			"GlueEncryptionException",
		},
		Shapes: map[string]string{
			"Table":                   "TableData",
			"GlueTable":               "Table",
			"GlueTables":              "Tables",
			"GlueResourceArn":         "ResourceArn",
			"GlueEncryptionException": "EncryptionException",
		},
	},
	"Glacier": {
		Shapes: map[string]string{
			"GlacierJobDescription": "JobDescription",
		},
	},
	"Elastic Beanstalk": {
		Shapes: map[string]string{
			"ElasticBeanstalkServiceException": "ServiceException",
		},
	},
	"Direct Connect": {
		Shapes: map[string]string{
			"DirectConnectClientException":                 "ClientException",
			"DirectConnectGatewayAssociationProposalId":    "GatewayAssociationProposalId",
			"DirectConnectGatewayAssociationProposalState": "GatewayAssociationProposalState",
			"DirectConnectGatewayAttachmentType":           "GatewayAttachmentType",
			"DirectConnectGatewayAttachmentList":           "GatewayAttachmentList",
			"DirectConnectGatewayAssociationProposalList":  "GatewayAssociationProposalList",
			"DirectConnectGatewayAssociationId":            "GatewayAssociationId",
			"DirectConnectGatewayList":                     "GatewayList",
			"DirectConnectGatewayName":                     "GatewayName",
			"DirectConnectGatewayAttachment":               "GatewayAttachment",
			"DirectConnectServerException":                 "ServerException",
			"DirectConnectGatewayState":                    "GatewayState",
			"DirectConnectGateway":                         "Gateway",
			"DirectConnectGatewayId":                       "GatewayId",
			"DirectConnectGatewayAttachmentState":          "GatewayAttachmentState",
			"DirectConnectGatewayAssociation":              "GatewayAssociation",
			"DirectConnectGatewayAssociationProposal":      "GatewayAssociationProposal",
			"DirectConnectGatewayAssociationState":         "GatewayAssociationState",
			"DirectConnectGatewayAssociationList":          "GatewayAssociationList",
		},
	},
	"Comprehend": {
		Shapes: map[string]string{
			"ComprehendArnName": "ArnName",
			"ComprehendArn":     "Arn",
		},
	},
	"Cognito Identity": {
		Shapes: map[string]string{
			"CognitoIdentityProviderList":       "ProviderList",
			"CognitoIdentityProviderName":       "ProviderName",
			"CognitoIdentityProviderClientId":   "ProviderClientId",
			"CognitoIdentityProviderTokenCheck": "ProviderTokenCheck",
			"CognitoIdentityProvider":           "Provider",
		},
	},
	"CloudTrail": {
		Shapes: map[string]string{
			"CloudTrailAccessNotEnabledException": "AccessNotEnabledException",
			"CloudTrailARNInvalidException":       "ARNInvalidException",
		},
	},
	"CloudFront": {
		Shapes: map[string]string{
			"CloudFrontOriginAccessIdentitySummaryList":   "OriginAccessIdentitySummaryList",
			"CloudFrontOriginAccessIdentity":              "OriginAccessIdentity",
			"CloudFrontOriginAccessIdentityAlreadyExists": "OriginAccessIdentityAlreadyExists",
			"CloudFrontOriginAccessIdentityConfig":        "OriginAccessIdentityConfig",
			"CloudFrontOriginAccessIdentitySummary":       "OriginAccessIdentitySummary",
			"CloudFrontOriginAccessIdentityList":          "OriginAccessIdentityList",
			"CloudFrontOriginAccessIdentityInUse":         "OriginAccessIdentityInUse",
		},
	},
	"Backup": {
		Shapes: map[string]string{
			"BackupPlan":                    "Plan",
			"BackupRule":                    "Rule",
			"BackupSelectionName":           "SelectionName",
			"BackupSelectionsList":          "SelectionsList",
			"BackupVaultEvents":             "VaultEvents",
			"BackupRuleName":                "RuleName",
			"BackupVaultName":               "VaultName",
			"BackupJob":                     "Job",
			"BackupJobState":                "JobState",
			"BackupJobsList":                "JobsList",
			"BackupVaultEvent":              "VaultEvent",
			"BackupPlanVersionsList":        "PlanVersionsList",
			"BackupPlansListMember":         "PlansListMember",
			"BackupSelection":               "Selection",
			"BackupVaultList":               "VaultList",
			"BackupVaultListMember":         "VaultListMember",
			"BackupPlanInput":               "PlanInput",
			"BackupRules":                   "Rules",
			"BackupPlansList":               "PlansList",
			"BackupPlanTemplatesList":       "PlanTemplatesList",
			"BackupRuleInput":               "RuleInput",
			"BackupPlanTemplatesListMember": "PlanTemplatesListMember",
			"BackupRulesInput":              "RulesInput",
			"BackupSelectionsListMember":    "SelectionsListMember",
			"BackupPlanName":                "PlanName",
		},
	},
	"Auto Scaling": {
		Shapes: map[string]string{
			"AutoScalingGroupDesiredCapacity": "GroupDesiredCapacity",
			"AutoScalingGroupNames":           "GroupNames",
			"AutoScalingGroupsType":           "GroupsType",
			"AutoScalingNotificationTypes":    "NotificationTypes",
			"AutoScalingGroupNamesType":       "GroupNamesType",
			"AutoScalingInstancesType":        "InstancesType",
			"AutoScalingInstanceDetails":      "InstanceDetails",
			"AutoScalingGroupMaxSize":         "GroupMaxSize",
			"AutoScalingGroups":               "Groups",
			"AutoScalingGroupMinSize":         "GroupMinSize",
			"AutoScalingGroup":                "Group",
		},
	},
	"AppStream": {
		Shapes: map[string]string{
			"AppstreamAgentVersion": "AgentVersion",
		},
	},
}
