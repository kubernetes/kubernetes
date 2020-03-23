package shareddefaults

const (
	// ECSCredsProviderEnvVar is an environmental variable key used to
	// determine which path needs to be hit.
	ECSCredsProviderEnvVar = "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
)

// ECSContainerCredentialsURI is the endpoint to retrieve container
// credentials. This can be overridden to test to ensure the credential process
// is behaving correctly.
var ECSContainerCredentialsURI = "http://169.254.170.2"
