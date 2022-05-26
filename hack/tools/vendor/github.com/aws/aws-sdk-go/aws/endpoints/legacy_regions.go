package endpoints

var legacyGlobalRegions = map[string]map[string]struct{}{
	"sts": {
		"ap-northeast-1": {},
		"ap-south-1":     {},
		"ap-southeast-1": {},
		"ap-southeast-2": {},
		"ca-central-1":   {},
		"eu-central-1":   {},
		"eu-north-1":     {},
		"eu-west-1":      {},
		"eu-west-2":      {},
		"eu-west-3":      {},
		"sa-east-1":      {},
		"us-east-1":      {},
		"us-east-2":      {},
		"us-west-1":      {},
		"us-west-2":      {},
	},
	"s3": {
		"us-east-1": {},
	},
}
