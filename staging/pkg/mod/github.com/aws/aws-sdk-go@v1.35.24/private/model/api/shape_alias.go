package api

var shapeNameAliases = map[string]map[string]string{
	"APIGateway": map[string]string{
		"RequestValidator": "UpdateRequestValidatorOutput",
		"VpcLink":          "UpdateVpcLinkOutput",
		"GatewayResponse":  "UpdateGatewayResponseOutput",
	},
	"Lambda": map[string]string{
		"Concurrency": "PutFunctionConcurrencyOutput",
	},
	"Neptune": map[string]string{
		"DBClusterParameterGroupNameMessage": "ResetDBClusterParameterGroupOutput",
		"DBParameterGroupNameMessage":        "ResetDBParameterGroupOutput",
	},
	"RDS": map[string]string{
		"DBClusterBacktrack": "BacktrackDBClusterOutput",
	},
}
