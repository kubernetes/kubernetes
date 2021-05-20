package session

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/endpoints"
)

// EnvProviderName provides a name of the provider when config is loaded from environment.
const EnvProviderName = "EnvConfigCredentials"

// envConfig is a collection of environment values the SDK will read
// setup config from. All environment values are optional. But some values
// such as credentials require multiple values to be complete or the values
// will be ignored.
type envConfig struct {
	// Environment configuration values. If set both Access Key ID and Secret Access
	// Key must be provided. Session Token and optionally also be provided, but is
	// not required.
	//
	//	# Access Key ID
	//	AWS_ACCESS_KEY_ID=AKID
	//	AWS_ACCESS_KEY=AKID # only read if AWS_ACCESS_KEY_ID is not set.
	//
	//	# Secret Access Key
	//	AWS_SECRET_ACCESS_KEY=SECRET
	//	AWS_SECRET_KEY=SECRET=SECRET # only read if AWS_SECRET_ACCESS_KEY is not set.
	//
	//	# Session Token
	//	AWS_SESSION_TOKEN=TOKEN
	Creds credentials.Value

	// Region value will instruct the SDK where to make service API requests to. If is
	// not provided in the environment the region must be provided before a service
	// client request is made.
	//
	//	AWS_REGION=us-east-1
	//
	//	# AWS_DEFAULT_REGION is only read if AWS_SDK_LOAD_CONFIG is also set,
	//	# and AWS_REGION is not also set.
	//	AWS_DEFAULT_REGION=us-east-1
	Region string

	// Profile name the SDK should load use when loading shared configuration from the
	// shared configuration files. If not provided "default" will be used as the
	// profile name.
	//
	//	AWS_PROFILE=my_profile
	//
	//	# AWS_DEFAULT_PROFILE is only read if AWS_SDK_LOAD_CONFIG is also set,
	//	# and AWS_PROFILE is not also set.
	//	AWS_DEFAULT_PROFILE=my_profile
	Profile string

	// SDK load config instructs the SDK to load the shared config in addition to
	// shared credentials. This also expands the configuration loaded from the shared
	// credentials to have parity with the shared config file. This also enables
	// Region and Profile support for the AWS_DEFAULT_REGION and AWS_DEFAULT_PROFILE
	// env values as well.
	//
	//	AWS_SDK_LOAD_CONFIG=1
	EnableSharedConfig bool

	// Shared credentials file path can be set to instruct the SDK to use an alternate
	// file for the shared credentials. If not set the file will be loaded from
	// $HOME/.aws/credentials on Linux/Unix based systems, and
	// %USERPROFILE%\.aws\credentials on Windows.
	//
	//	AWS_SHARED_CREDENTIALS_FILE=$HOME/my_shared_credentials
	SharedCredentialsFile string

	// Shared config file path can be set to instruct the SDK to use an alternate
	// file for the shared config. If not set the file will be loaded from
	// $HOME/.aws/config on Linux/Unix based systems, and
	// %USERPROFILE%\.aws\config on Windows.
	//
	//	AWS_CONFIG_FILE=$HOME/my_shared_config
	SharedConfigFile string

	// Sets the path to a custom Credentials Authority (CA) Bundle PEM file
	// that the SDK will use instead of the system's root CA bundle.
	// Only use this if you want to configure the SDK to use a custom set
	// of CAs.
	//
	// Enabling this option will attempt to merge the Transport
	// into the SDK's HTTP client. If the client's Transport is
	// not a http.Transport an error will be returned. If the
	// Transport's TLS config is set this option will cause the
	// SDK to overwrite the Transport's TLS config's  RootCAs value.
	//
	// Setting a custom HTTPClient in the aws.Config options will override this setting.
	// To use this option and custom HTTP client, the HTTP client needs to be provided
	// when creating the session. Not the service client.
	//
	//  AWS_CA_BUNDLE=$HOME/my_custom_ca_bundle
	CustomCABundle string

	// Sets the TLC client certificate that should be used by the SDK's HTTP transport
	// when making requests. The certificate must be paired with a TLS client key file.
	//
	//  AWS_SDK_GO_CLIENT_TLS_CERT=$HOME/my_client_cert
	ClientTLSCert string

	// Sets the TLC client key that should be used by the SDK's HTTP transport
	// when making requests. The key must be paired with a TLS client certificate file.
	//
	//  AWS_SDK_GO_CLIENT_TLS_KEY=$HOME/my_client_key
	ClientTLSKey string

	csmEnabled  string
	CSMEnabled  *bool
	CSMPort     string
	CSMHost     string
	CSMClientID string

	// Enables endpoint discovery via environment variables.
	//
	//	AWS_ENABLE_ENDPOINT_DISCOVERY=true
	EnableEndpointDiscovery *bool
	enableEndpointDiscovery string

	// Specifies the WebIdentity token the SDK should use to assume a role
	// with.
	//
	//  AWS_WEB_IDENTITY_TOKEN_FILE=file_path
	WebIdentityTokenFilePath string

	// Specifies the IAM role arn to use when assuming an role.
	//
	//  AWS_ROLE_ARN=role_arn
	RoleARN string

	// Specifies the IAM role session name to use when assuming a role.
	//
	//  AWS_ROLE_SESSION_NAME=session_name
	RoleSessionName string

	// Specifies the STS Regional Endpoint flag for the SDK to resolve the endpoint
	// for a service.
	//
	// AWS_STS_REGIONAL_ENDPOINTS=regional
	// This can take value as `regional` or `legacy`
	STSRegionalEndpoint endpoints.STSRegionalEndpoint

	// Specifies the S3 Regional Endpoint flag for the SDK to resolve the
	// endpoint for a service.
	//
	// AWS_S3_US_EAST_1_REGIONAL_ENDPOINT=regional
	// This can take value as `regional` or `legacy`
	S3UsEast1RegionalEndpoint endpoints.S3UsEast1RegionalEndpoint

	// Specifies if the S3 service should allow ARNs to direct the region
	// the client's requests are sent to.
	//
	// AWS_S3_USE_ARN_REGION=true
	S3UseARNRegion bool

	// Specifies the alternative endpoint to use for EC2 IMDS.
	//
	// AWS_EC2_METADATA_SERVICE_ENDPOINT=http://[::1]
	EC2IMDSEndpoint string
}

var (
	csmEnabledEnvKey = []string{
		"AWS_CSM_ENABLED",
	}
	csmHostEnvKey = []string{
		"AWS_CSM_HOST",
	}
	csmPortEnvKey = []string{
		"AWS_CSM_PORT",
	}
	csmClientIDEnvKey = []string{
		"AWS_CSM_CLIENT_ID",
	}
	credAccessEnvKey = []string{
		"AWS_ACCESS_KEY_ID",
		"AWS_ACCESS_KEY",
	}
	credSecretEnvKey = []string{
		"AWS_SECRET_ACCESS_KEY",
		"AWS_SECRET_KEY",
	}
	credSessionEnvKey = []string{
		"AWS_SESSION_TOKEN",
	}

	enableEndpointDiscoveryEnvKey = []string{
		"AWS_ENABLE_ENDPOINT_DISCOVERY",
	}

	regionEnvKeys = []string{
		"AWS_REGION",
		"AWS_DEFAULT_REGION", // Only read if AWS_SDK_LOAD_CONFIG is also set
	}
	profileEnvKeys = []string{
		"AWS_PROFILE",
		"AWS_DEFAULT_PROFILE", // Only read if AWS_SDK_LOAD_CONFIG is also set
	}
	sharedCredsFileEnvKey = []string{
		"AWS_SHARED_CREDENTIALS_FILE",
	}
	sharedConfigFileEnvKey = []string{
		"AWS_CONFIG_FILE",
	}
	webIdentityTokenFilePathEnvKey = []string{
		"AWS_WEB_IDENTITY_TOKEN_FILE",
	}
	roleARNEnvKey = []string{
		"AWS_ROLE_ARN",
	}
	roleSessionNameEnvKey = []string{
		"AWS_ROLE_SESSION_NAME",
	}
	stsRegionalEndpointKey = []string{
		"AWS_STS_REGIONAL_ENDPOINTS",
	}
	s3UsEast1RegionalEndpoint = []string{
		"AWS_S3_US_EAST_1_REGIONAL_ENDPOINT",
	}
	s3UseARNRegionEnvKey = []string{
		"AWS_S3_USE_ARN_REGION",
	}
	ec2IMDSEndpointEnvKey = []string{
		"AWS_EC2_METADATA_SERVICE_ENDPOINT",
	}
	useCABundleKey = []string{
		"AWS_CA_BUNDLE",
	}
	useClientTLSCert = []string{
		"AWS_SDK_GO_CLIENT_TLS_CERT",
	}
	useClientTLSKey = []string{
		"AWS_SDK_GO_CLIENT_TLS_KEY",
	}
)

// loadEnvConfig retrieves the SDK's environment configuration.
// See `envConfig` for the values that will be retrieved.
//
// If the environment variable `AWS_SDK_LOAD_CONFIG` is set to a truthy value
// the shared SDK config will be loaded in addition to the SDK's specific
// configuration values.
func loadEnvConfig() (envConfig, error) {
	enableSharedConfig, _ := strconv.ParseBool(os.Getenv("AWS_SDK_LOAD_CONFIG"))
	return envConfigLoad(enableSharedConfig)
}

// loadEnvSharedConfig retrieves the SDK's environment configuration, and the
// SDK shared config. See `envConfig` for the values that will be retrieved.
//
// Loads the shared configuration in addition to the SDK's specific configuration.
// This will load the same values as `loadEnvConfig` if the `AWS_SDK_LOAD_CONFIG`
// environment variable is set.
func loadSharedEnvConfig() (envConfig, error) {
	return envConfigLoad(true)
}

func envConfigLoad(enableSharedConfig bool) (envConfig, error) {
	cfg := envConfig{}

	cfg.EnableSharedConfig = enableSharedConfig

	// Static environment credentials
	var creds credentials.Value
	setFromEnvVal(&creds.AccessKeyID, credAccessEnvKey)
	setFromEnvVal(&creds.SecretAccessKey, credSecretEnvKey)
	setFromEnvVal(&creds.SessionToken, credSessionEnvKey)
	if creds.HasKeys() {
		// Require logical grouping of credentials
		creds.ProviderName = EnvProviderName
		cfg.Creds = creds
	}

	// Role Metadata
	setFromEnvVal(&cfg.RoleARN, roleARNEnvKey)
	setFromEnvVal(&cfg.RoleSessionName, roleSessionNameEnvKey)

	// Web identity environment variables
	setFromEnvVal(&cfg.WebIdentityTokenFilePath, webIdentityTokenFilePathEnvKey)

	// CSM environment variables
	setFromEnvVal(&cfg.csmEnabled, csmEnabledEnvKey)
	setFromEnvVal(&cfg.CSMHost, csmHostEnvKey)
	setFromEnvVal(&cfg.CSMPort, csmPortEnvKey)
	setFromEnvVal(&cfg.CSMClientID, csmClientIDEnvKey)

	if len(cfg.csmEnabled) != 0 {
		v, _ := strconv.ParseBool(cfg.csmEnabled)
		cfg.CSMEnabled = &v
	}

	regionKeys := regionEnvKeys
	profileKeys := profileEnvKeys
	if !cfg.EnableSharedConfig {
		regionKeys = regionKeys[:1]
		profileKeys = profileKeys[:1]
	}

	setFromEnvVal(&cfg.Region, regionKeys)
	setFromEnvVal(&cfg.Profile, profileKeys)

	// endpoint discovery is in reference to it being enabled.
	setFromEnvVal(&cfg.enableEndpointDiscovery, enableEndpointDiscoveryEnvKey)
	if len(cfg.enableEndpointDiscovery) > 0 {
		cfg.EnableEndpointDiscovery = aws.Bool(cfg.enableEndpointDiscovery != "false")
	}

	setFromEnvVal(&cfg.SharedCredentialsFile, sharedCredsFileEnvKey)
	setFromEnvVal(&cfg.SharedConfigFile, sharedConfigFileEnvKey)

	if len(cfg.SharedCredentialsFile) == 0 {
		cfg.SharedCredentialsFile = defaults.SharedCredentialsFilename()
	}
	if len(cfg.SharedConfigFile) == 0 {
		cfg.SharedConfigFile = defaults.SharedConfigFilename()
	}

	setFromEnvVal(&cfg.CustomCABundle, useCABundleKey)
	setFromEnvVal(&cfg.ClientTLSCert, useClientTLSCert)
	setFromEnvVal(&cfg.ClientTLSKey, useClientTLSKey)

	var err error
	// STS Regional Endpoint variable
	for _, k := range stsRegionalEndpointKey {
		if v := os.Getenv(k); len(v) != 0 {
			cfg.STSRegionalEndpoint, err = endpoints.GetSTSRegionalEndpoint(v)
			if err != nil {
				return cfg, fmt.Errorf("failed to load, %v from env config, %v", k, err)
			}
		}
	}

	// S3 Regional Endpoint variable
	for _, k := range s3UsEast1RegionalEndpoint {
		if v := os.Getenv(k); len(v) != 0 {
			cfg.S3UsEast1RegionalEndpoint, err = endpoints.GetS3UsEast1RegionalEndpoint(v)
			if err != nil {
				return cfg, fmt.Errorf("failed to load, %v from env config, %v", k, err)
			}
		}
	}

	var s3UseARNRegion string
	setFromEnvVal(&s3UseARNRegion, s3UseARNRegionEnvKey)
	if len(s3UseARNRegion) != 0 {
		switch {
		case strings.EqualFold(s3UseARNRegion, "false"):
			cfg.S3UseARNRegion = false
		case strings.EqualFold(s3UseARNRegion, "true"):
			cfg.S3UseARNRegion = true
		default:
			return envConfig{}, fmt.Errorf(
				"invalid value for environment variable, %s=%s, need true or false",
				s3UseARNRegionEnvKey[0], s3UseARNRegion)
		}
	}

	setFromEnvVal(&cfg.EC2IMDSEndpoint, ec2IMDSEndpointEnvKey)

	return cfg, nil
}

func setFromEnvVal(dst *string, keys []string) {
	for _, k := range keys {
		if v := os.Getenv(k); len(v) != 0 {
			*dst = v
			break
		}
	}
}
