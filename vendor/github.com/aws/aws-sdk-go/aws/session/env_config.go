package session

import (
	"os"
	"strconv"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/defaults"
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

	// Sets the path to a custom Credentials Authroity (CA) Bundle PEM file
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

	csmEnabled  string
	CSMEnabled  bool
	CSMPort     string
	CSMClientID string
}

var (
	csmEnabledEnvKey = []string{
		"AWS_CSM_ENABLED",
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
)

// loadEnvConfig retrieves the SDK's environment configuration.
// See `envConfig` for the values that will be retrieved.
//
// If the environment variable `AWS_SDK_LOAD_CONFIG` is set to a truthy value
// the shared SDK config will be loaded in addition to the SDK's specific
// configuration values.
func loadEnvConfig() envConfig {
	enableSharedConfig, _ := strconv.ParseBool(os.Getenv("AWS_SDK_LOAD_CONFIG"))
	return envConfigLoad(enableSharedConfig)
}

// loadEnvSharedConfig retrieves the SDK's environment configuration, and the
// SDK shared config. See `envConfig` for the values that will be retrieved.
//
// Loads the shared configuration in addition to the SDK's specific configuration.
// This will load the same values as `loadEnvConfig` if the `AWS_SDK_LOAD_CONFIG`
// environment variable is set.
func loadSharedEnvConfig() envConfig {
	return envConfigLoad(true)
}

func envConfigLoad(enableSharedConfig bool) envConfig {
	cfg := envConfig{}

	cfg.EnableSharedConfig = enableSharedConfig

	setFromEnvVal(&cfg.Creds.AccessKeyID, credAccessEnvKey)
	setFromEnvVal(&cfg.Creds.SecretAccessKey, credSecretEnvKey)
	setFromEnvVal(&cfg.Creds.SessionToken, credSessionEnvKey)

	// CSM environment variables
	setFromEnvVal(&cfg.csmEnabled, csmEnabledEnvKey)
	setFromEnvVal(&cfg.CSMPort, csmPortEnvKey)
	setFromEnvVal(&cfg.CSMClientID, csmClientIDEnvKey)
	cfg.CSMEnabled = len(cfg.csmEnabled) > 0

	// Require logical grouping of credentials
	if len(cfg.Creds.AccessKeyID) == 0 || len(cfg.Creds.SecretAccessKey) == 0 {
		cfg.Creds = credentials.Value{}
	} else {
		cfg.Creds.ProviderName = EnvProviderName
	}

	regionKeys := regionEnvKeys
	profileKeys := profileEnvKeys
	if !cfg.EnableSharedConfig {
		regionKeys = regionKeys[:1]
		profileKeys = profileKeys[:1]
	}

	setFromEnvVal(&cfg.Region, regionKeys)
	setFromEnvVal(&cfg.Profile, profileKeys)

	setFromEnvVal(&cfg.SharedCredentialsFile, sharedCredsFileEnvKey)
	setFromEnvVal(&cfg.SharedConfigFile, sharedConfigFileEnvKey)

	if len(cfg.SharedCredentialsFile) == 0 {
		cfg.SharedCredentialsFile = defaults.SharedCredentialsFilename()
	}
	if len(cfg.SharedConfigFile) == 0 {
		cfg.SharedConfigFile = defaults.SharedConfigFilename()
	}

	cfg.CustomCABundle = os.Getenv("AWS_CA_BUNDLE")

	return cfg
}

func setFromEnvVal(dst *string, keys []string) {
	for _, k := range keys {
		if v := os.Getenv(k); len(v) > 0 {
			*dst = v
			break
		}
	}
}
