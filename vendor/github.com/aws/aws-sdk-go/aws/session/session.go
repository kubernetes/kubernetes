package session

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/endpoints"
)

// A Session provides a central location to create service clients from and
// store configurations and request handlers for those services.
//
// Sessions are safe to create service clients concurrently, but it is not safe
// to mutate the Session concurrently.
//
// The Session satisfies the service client's client.ClientConfigProvider.
type Session struct {
	Config   *aws.Config
	Handlers request.Handlers
}

// New creates a new instance of the handlers merging in the provided configs
// on top of the SDK's default configurations. Once the Session is created it
// can be mutated to modify the Config or Handlers. The Session is safe to be
// read concurrently, but it should not be written to concurrently.
//
// If the AWS_SDK_LOAD_CONFIG environment is set to a truthy value, the New
// method could now encounter an error when loading the configuration. When
// The environment variable is set, and an error occurs, New will return a
// session that will fail all requests reporting the error that occured while
// loading the session. Use NewSession to get the error when creating the
// session.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded, in addition to
// the shared credentials file (~/.aws/config). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file.
//
// Deprecated: Use NewSession functiions to create sessions instead. NewSession
// has the same functionality as New except an error can be returned when the
// func is called instead of waiting to receive an error until a request is made.
func New(cfgs ...*aws.Config) *Session {
	// load initial config from environment
	envCfg := loadEnvConfig()

	if envCfg.EnableSharedConfig {
		s, err := newSession(envCfg, cfgs...)
		if err != nil {
			// Old session.New expected all errors to be discovered when
			// a request is made, and would report the errors then. This
			// needs to be replicated if an error occurs while creating
			// the session.
			msg := "failed to create session with AWS_SDK_LOAD_CONFIG enabled. " +
				"Use session.NewSession to handle errors occuring during session creation."

			// Session creation failed, need to report the error and prevent
			// any requests from succeeding.
			s = &Session{Config: defaults.Config()}
			s.Config.MergeIn(cfgs...)
			s.Config.Logger.Log("ERROR:", msg, "Error:", err)
			s.Handlers.Validate.PushBack(func(r *request.Request) {
				r.Error = err
			})
		}
		return s
	}

	return oldNewSession(cfgs...)
}

// NewSession returns a new Session created from SDK defaults, config files,
// environment, and user provided config files. Once the Session is created
// it can be mutated to modify the Config or Handlers. The Session is safe to
// be read concurrently, but it should not be written to concurrently.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded in addition to
// the shared credentials file (~/.aws/config). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file. Enabling the Shared Config will also allow the Session
// to be built with retrieving credentials with AssumeRole set in the config.
//
// See the NewSessionWithOptions func for information on how to override or
// control through code how the Session will be created. Such as specifing the
// config profile, and controlling if shared config is enabled or not.
func NewSession(cfgs ...*aws.Config) (*Session, error) {
	envCfg := loadEnvConfig()

	return newSession(envCfg, cfgs...)
}

// SharedConfigState provides the ability to optionally override the state
// of the session's creation based on the shared config being enabled or
// disabled.
type SharedConfigState int

const (
	// SharedConfigStateFromEnv does not override any state of the
	// AWS_SDK_LOAD_CONFIG env var. It is the default value of the
	// SharedConfigState type.
	SharedConfigStateFromEnv SharedConfigState = iota

	// SharedConfigDisable overrides the AWS_SDK_LOAD_CONFIG env var value
	// and disables the shared config functionality.
	SharedConfigDisable

	// SharedConfigEnable overrides the AWS_SDK_LOAD_CONFIG env var value
	// and enables the shared config functionality.
	SharedConfigEnable
)

// Options provides the means to control how a Session is created and what
// configuration values will be loaded.
//
type Options struct {
	// Provides config values for the SDK to use when creating service clients
	// and making API requests to services. Any value set in with this field
	// will override the associated value provided by the SDK defaults,
	// environment or config files where relevent.
	//
	// If not set, configuration values from from SDK defaults, environment,
	// config will be used.
	Config aws.Config

	// Overrides the config profile the Session should be created from. If not
	// set the value of the environment variable will be loaded (AWS_PROFILE,
	// or AWS_DEFAULT_PROFILE if the Shared Config is enabled).
	//
	// If not set and environment variables are not set the "default"
	// (DefaultSharedConfigProfile) will be used as the profile to load the
	// session config from.
	Profile string

	// Instructs how the Session will be created based on the AWS_SDK_LOAD_CONFIG
	// environment variable. By default a Session will be created using the
	// value provided by the AWS_SDK_LOAD_CONFIG environment variable.
	//
	// Setting this value to SharedConfigEnable or SharedConfigDisable
	// will allow you to override the AWS_SDK_LOAD_CONFIG environment variable
	// and enable or disable the shared config functionality.
	SharedConfigState SharedConfigState
}

// NewSessionWithOptions returns a new Session created from SDK defaults, config files,
// environment, and user provided config files. This func uses the Options
// values to configure how the Session is created.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded in addition to
// the shared credentials file (~/.aws/config). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file. Enabling the Shared Config will also allow the Session
// to be built with retrieving credentials with AssumeRole set in the config.
//
//     // Equivalent to session.New
//     sess, err := session.NewSessionWithOptions(session.Options{})
//
//     // Specify profile to load for the session's config
//     sess, err := session.NewSessionWithOptions(session.Options{
//          Profile: "profile_name",
//     })
//
//     // Specify profile for config and region for requests
//     sess, err := session.NewSessionWithOptions(session.Options{
//          Config: aws.Config{Region: aws.String("us-east-1")},
//          Profile: "profile_name",
//     })
//
//     // Force enable Shared Config support
//     sess, err := session.NewSessionWithOptions(session.Options{
//         SharedConfigState: SharedConfigEnable,
//     })
func NewSessionWithOptions(opts Options) (*Session, error) {
	var envCfg envConfig
	if opts.SharedConfigState == SharedConfigEnable {
		envCfg = loadSharedEnvConfig()
	} else {
		envCfg = loadEnvConfig()
	}

	if len(opts.Profile) > 0 {
		envCfg.Profile = opts.Profile
	}

	switch opts.SharedConfigState {
	case SharedConfigDisable:
		envCfg.EnableSharedConfig = false
	case SharedConfigEnable:
		envCfg.EnableSharedConfig = true
	}

	return newSession(envCfg, &opts.Config)
}

// Must is a helper function to ensure the Session is valid and there was no
// error when calling a NewSession function.
//
// This helper is intended to be used in variable initialization to load the
// Session and configuration at startup. Such as:
//
//     var sess = session.Must(session.NewSession())
func Must(sess *Session, err error) *Session {
	if err != nil {
		panic(err)
	}

	return sess
}

func oldNewSession(cfgs ...*aws.Config) *Session {
	cfg := defaults.Config()
	handlers := defaults.Handlers()

	// Apply the passed in configs so the configuration can be applied to the
	// default credential chain
	cfg.MergeIn(cfgs...)
	cfg.Credentials = defaults.CredChain(cfg, handlers)

	// Reapply any passed in configs to override credentials if set
	cfg.MergeIn(cfgs...)

	s := &Session{
		Config:   cfg,
		Handlers: handlers,
	}

	initHandlers(s)

	return s
}

func newSession(envCfg envConfig, cfgs ...*aws.Config) (*Session, error) {
	cfg := defaults.Config()
	handlers := defaults.Handlers()

	// Get a merged version of the user provided config to determine if
	// credentials were.
	userCfg := &aws.Config{}
	userCfg.MergeIn(cfgs...)

	// Order config files will be loaded in with later files overwriting
	// previous config file values.
	cfgFiles := []string{envCfg.SharedConfigFile, envCfg.SharedCredentialsFile}
	if !envCfg.EnableSharedConfig {
		// The shared config file (~/.aws/config) is only loaded if instructed
		// to load via the envConfig.EnableSharedConfig (AWS_SDK_LOAD_CONFIG).
		cfgFiles = cfgFiles[1:]
	}

	// Load additional config from file(s)
	sharedCfg, err := loadSharedConfig(envCfg.Profile, cfgFiles)
	if err != nil {
		return nil, err
	}

	mergeConfigSrcs(cfg, userCfg, envCfg, sharedCfg, handlers)

	s := &Session{
		Config:   cfg,
		Handlers: handlers,
	}

	initHandlers(s)

	return s, nil
}

func mergeConfigSrcs(cfg, userCfg *aws.Config, envCfg envConfig, sharedCfg sharedConfig, handlers request.Handlers) {
	// Merge in user provided configuration
	cfg.MergeIn(userCfg)

	// Region if not already set by user
	if len(aws.StringValue(cfg.Region)) == 0 {
		if len(envCfg.Region) > 0 {
			cfg.WithRegion(envCfg.Region)
		} else if envCfg.EnableSharedConfig && len(sharedCfg.Region) > 0 {
			cfg.WithRegion(sharedCfg.Region)
		}
	}

	// Configure credentials if not already set
	if cfg.Credentials == credentials.AnonymousCredentials && userCfg.Credentials == nil {
		if len(envCfg.Creds.AccessKeyID) > 0 {
			cfg.Credentials = credentials.NewStaticCredentialsFromCreds(
				envCfg.Creds,
			)
		} else if envCfg.EnableSharedConfig && len(sharedCfg.AssumeRole.RoleARN) > 0 && sharedCfg.AssumeRoleSource != nil {
			cfgCp := *cfg
			cfgCp.Credentials = credentials.NewStaticCredentialsFromCreds(
				sharedCfg.AssumeRoleSource.Creds,
			)
			cfg.Credentials = stscreds.NewCredentials(
				&Session{
					Config:   &cfgCp,
					Handlers: handlers.Copy(),
				},
				sharedCfg.AssumeRole.RoleARN,
				func(opt *stscreds.AssumeRoleProvider) {
					opt.RoleSessionName = sharedCfg.AssumeRole.RoleSessionName

					if len(sharedCfg.AssumeRole.ExternalID) > 0 {
						opt.ExternalID = aws.String(sharedCfg.AssumeRole.ExternalID)
					}

					// MFA not supported
				},
			)
		} else if len(sharedCfg.Creds.AccessKeyID) > 0 {
			cfg.Credentials = credentials.NewStaticCredentialsFromCreds(
				sharedCfg.Creds,
			)
		} else {
			// Fallback to default credentials provider, include mock errors
			// for the credential chain so user can identify why credentials
			// failed to be retrieved.
			cfg.Credentials = credentials.NewCredentials(&credentials.ChainProvider{
				VerboseErrors: aws.BoolValue(cfg.CredentialsChainVerboseErrors),
				Providers: []credentials.Provider{
					&credProviderError{Err: awserr.New("EnvAccessKeyNotFound", "failed to find credentials in the environment.", nil)},
					&credProviderError{Err: awserr.New("SharedCredsLoad", fmt.Sprintf("failed to load profile, %s.", envCfg.Profile), nil)},
					defaults.RemoteCredProvider(*cfg, handlers),
				},
			})
		}
	}
}

type credProviderError struct {
	Err error
}

var emptyCreds = credentials.Value{}

func (c credProviderError) Retrieve() (credentials.Value, error) {
	return credentials.Value{}, c.Err
}
func (c credProviderError) IsExpired() bool {
	return true
}

func initHandlers(s *Session) {
	// Add the Validate parameter handler if it is not disabled.
	s.Handlers.Validate.Remove(corehandlers.ValidateParametersHandler)
	if !aws.BoolValue(s.Config.DisableParamValidation) {
		s.Handlers.Validate.PushBackNamed(corehandlers.ValidateParametersHandler)
	}
}

// Copy creates and returns a copy of the current Session, coping the config
// and handlers. If any additional configs are provided they will be merged
// on top of the Session's copied config.
//
//     // Create a copy of the current Session, configured for the us-west-2 region.
//     sess.Copy(&aws.Config{Region: aws.String("us-west-2")})
func (s *Session) Copy(cfgs ...*aws.Config) *Session {
	newSession := &Session{
		Config:   s.Config.Copy(cfgs...),
		Handlers: s.Handlers.Copy(),
	}

	initHandlers(newSession)

	return newSession
}

// ClientConfig satisfies the client.ConfigProvider interface and is used to
// configure the service client instances. Passing the Session to the service
// client's constructor (New) will use this method to configure the client.
func (s *Session) ClientConfig(serviceName string, cfgs ...*aws.Config) client.Config {
	s = s.Copy(cfgs...)
	endpoint, signingRegion := endpoints.NormalizeEndpoint(
		aws.StringValue(s.Config.Endpoint),
		serviceName,
		aws.StringValue(s.Config.Region),
		aws.BoolValue(s.Config.DisableSSL),
		aws.BoolValue(s.Config.UseDualStack),
	)

	return client.Config{
		Config:        s.Config,
		Handlers:      s.Handlers,
		Endpoint:      endpoint,
		SigningRegion: signingRegion,
	}
}
