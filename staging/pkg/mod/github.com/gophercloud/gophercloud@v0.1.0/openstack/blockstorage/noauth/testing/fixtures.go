package testing

// NoAuthResult is the expected result of the noauth Service Client
type NoAuthResult struct {
	TokenID  string
	Endpoint string
}

var naTestResult = NoAuthResult{
	TokenID:  "user:test",
	Endpoint: "http://cinder:8776/v2/test/",
}

var naResult = NoAuthResult{
	TokenID:  "admin:admin",
	Endpoint: "http://cinder:8776/v2/admin/",
}

var errorResult = "CinderEndpoint is required"
