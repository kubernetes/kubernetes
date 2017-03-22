package session

import (
	"os"
	"path/filepath"
	"strconv"

	"github.com/aws/aws-sdk-go/aws/credentials"
)

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
}

var (
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

	// Require logical grouping of credentials
	if len(cfg.Creds.AccessKeyID) == 0 || len(cfg.Creds.SecretAccessKey) == 0 {
		cfg.Creds = credentials.Value{}
	} else {
		cfg.Creds.ProviderName = "EnvConfigCredentials"
	}

	regionKeys := regionEnvKeys
	profileKeys := profileEnvKeys
	if !cfg.EnableSharedConfig {
		regionKeys = regionKeys[:1]
		profileKeys = profileKeys[:1]
	}

	setFromEnvVal(&cfg.Region, regionKeys)
	setFromEnvVal(&cfg.Profile, profileKeys)

	cfg.SharedCredentialsFile = sharedCredentialsFilename()
	cfg.SharedConfigFile = sharedConfigFilename()

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

func sharedCredentialsFilename() string {
	if name := os.Getenv("AWS_SHARED_CREDENTIALS_FILE"); len(name) > 0 {
		return name
	}

	return filepath.Join(userHomeDir(), ".aws", "credentials")
}

func sharedConfigFilename() string {
	if name := os.Getenv("AWS_CONFIG_FILE"); len(name) > 0 {
		return name
	}

	return filepath.Join(userHomeDir(), ".aws", "config")
}

func userHomeDir() string {
	homeDir := os.Getenv("HOME") // *nix
	if len(homeDir) == 0 {       // windows
		homeDir = os.Getenv("USERPROFILE")
	}

	return homeDir
}
