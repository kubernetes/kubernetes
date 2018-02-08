package session

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
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
// session that will fail all requests reporting the error that occurred while
// loading the session. Use NewSession to get the error when creating the
// session.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded, in addition to
// the shared credentials file (~/.aws/credentials). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file.
//
// Deprecated: Use NewSession functions to create sessions instead. NewSession
// has the same functionality as New except an error can be returned when the
// func is called instead of waiting to receive an error until a request is made.
func New(cfgs ...*aws.Config) *Session {
	// load initial config from environment
	envCfg := loadEnvConfig()

	if envCfg.EnableSharedConfig {
		s, err := newSession(Options{}, envCfg, cfgs...)
		if err != nil {
			// Old session.New expected all errors to be discovered when
			// a request is made, and would report the errors then. This
			// needs to be replicated if an error occurs while creating
			// the session.
			msg := "failed to create session with AWS_SDK_LOAD_CONFIG enabled. " +
				"Use session.NewSession to handle errors occurring during session creation."

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

	return deprecatedNewSession(cfgs...)
}

// NewSession returns a new Session created from SDK defaults, config files,
// environment, and user provided config files. Once the Session is created
// it can be mutated to modify the Config or Handlers. The Session is safe to
// be read concurrently, but it should not be written to concurrently.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded in addition to
// the shared credentials file (~/.aws/credentials). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file. Enabling the Shared Config will also allow the Session
// to be built with retrieving credentials with AssumeRole set in the config.
//
// See the NewSessionWithOptions func for information on how to override or
// control through code how the Session will be created. Such as specifying the
// config profile, and controlling if shared config is enabled or not.
func NewSession(cfgs ...*aws.Config) (*Session, error) {
	opts := Options{}
	opts.Config.MergeIn(cfgs...)

	return NewSessionWithOptions(opts)
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
	// environment or config files where relevant.
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

	// Ordered list of files the session will load configuration from.
	// It will override environment variable AWS_SHARED_CREDENTIALS_FILE, AWS_CONFIG_FILE.
	SharedConfigFiles []string

	// When the SDK's shared config is configured to assume a role with MFA
	// this option is required in order to provide the mechanism that will
	// retrieve the MFA token. There is no default value for this field. If
	// it is not set an error will be returned when creating the session.
	//
	// This token provider will be called when ever the assumed role's
	// credentials need to be refreshed. Within the context of service clients
	// all sharing the same session the SDK will ensure calls to the token
	// provider are atomic. When sharing a token provider across multiple
	// sessions additional synchronization logic is needed to ensure the
	// token providers do not introduce race conditions. It is recommend to
	// share the session where possible.
	//
	// stscreds.StdinTokenProvider is a basic implementation that will prompt
	// from stdin for the MFA token code.
	//
	// This field is only used if the shared configuration is enabled, and
	// the config enables assume role wit MFA via the mfa_serial field.
	AssumeRoleTokenProvider func() (string, error)

	// Reader for a custom Credentials Authority (CA) bundle in PEM format that
	// the SDK will use instead of the default system's root CA bundle. Use this
	// only if you want to replace the CA bundle the SDK uses for TLS requests.
	//
	// Enabling this option will attempt to merge the Transport into the SDK's HTTP
	// client. If the client's Transport is not a http.Transport an error will be
	// returned. If the Transport's TLS config is set this option will cause the SDK
	// to overwrite the Transport's TLS config's  RootCAs value. If the CA
	// bundle reader contains multiple certificates all of them will be loaded.
	//
	// The Session option CustomCABundle is also available when creating sessions
	// to also enable this feature. CustomCABundle session option field has priority
	// over the AWS_CA_BUNDLE environment variable, and will be used if both are set.
	CustomCABundle io.Reader
}

// NewSessionWithOptions returns a new Session created from SDK defaults, config files,
// environment, and user provided config files. This func uses the Options
// values to configure how the Session is created.
//
// If the AWS_SDK_LOAD_CONFIG environment variable is set to a truthy value
// the shared config file (~/.aws/config) will also be loaded in addition to
// the shared credentials file (~/.aws/credentials). Values set in both the
// shared config, and shared credentials will be taken from the shared
// credentials file. Enabling the Shared Config will also allow the Session
// to be built with retrieving credentials with AssumeRole set in the config.
//
//     // Equivalent to session.New
//     sess := session.Must(session.NewSessionWithOptions(session.Options{}))
//
//     // Specify profile to load for the session's config
//     sess := session.Must(session.NewSessionWithOptions(session.Options{
//          Profile: "profile_name",
//     }))
//
//     // Specify profile for config and region for requests
//     sess := session.Must(session.NewSessionWithOptions(session.Options{
//          Config: aws.Config{Region: aws.String("us-east-1")},
//          Profile: "profile_name",
//     }))
//
//     // Force enable Shared Config support
//     sess := session.Must(session.NewSessionWithOptions(session.Options{
//         SharedConfigState: session.SharedConfigEnable,
//     }))
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

	if len(envCfg.SharedCredentialsFile) == 0 {
		envCfg.SharedCredentialsFile = defaults.SharedCredentialsFilename()
	}
	if len(envCfg.SharedConfigFile) == 0 {
		envCfg.SharedConfigFile = defaults.SharedConfigFilename()
	}

	// Only use AWS_CA_BUNDLE if session option is not provided.
	if len(envCfg.CustomCABundle) != 0 && opts.CustomCABundle == nil {
		f, err := os.Open(envCfg.CustomCABundle)
		if err != nil {
			return nil, awserr.New("LoadCustomCABundleError",
				"failed to open custom CA bundle PEM file", err)
		}
		defer f.Close()
		opts.CustomCABundle = f
	}

	return newSession(opts, envCfg, &opts.Config)
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

func deprecatedNewSession(cfgs ...*aws.Config) *Session {
	cfg := defaults.Config()
	handlers := defaults.Handlers()

	// Apply the passed in configs so the configuration can be applied to the
	// default credential chain
	cfg.MergeIn(cfgs...)
	if cfg.EndpointResolver == nil {
		// An endpoint resolver is required for a session to be able to provide
		// endpoints for service client configurations.
		cfg.EndpointResolver = endpoints.DefaultResolver()
	}
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

func newSession(opts Options, envCfg envConfig, cfgs ...*aws.Config) (*Session, error) {
	cfg := defaults.Config()
	handlers := defaults.Handlers()

	// Get a merged version of the user provided config to determine if
	// credentials were.
	userCfg := &aws.Config{}
	userCfg.MergeIn(cfgs...)

	// Ordered config files will be loaded in with later files overwriting
	// previous config file values.
	var cfgFiles []string
	if opts.SharedConfigFiles != nil {
		cfgFiles = opts.SharedConfigFiles
	} else {
		cfgFiles = []string{envCfg.SharedConfigFile, envCfg.SharedCredentialsFile}
		if !envCfg.EnableSharedConfig {
			// The shared config file (~/.aws/config) is only loaded if instructed
			// to load via the envConfig.EnableSharedConfig (AWS_SDK_LOAD_CONFIG).
			cfgFiles = cfgFiles[1:]
		}
	}

	// Load additional config from file(s)
	sharedCfg, err := loadSharedConfig(envCfg.Profile, cfgFiles)
	if err != nil {
		return nil, err
	}

	if err := mergeConfigSrcs(cfg, userCfg, envCfg, sharedCfg, handlers, opts); err != nil {
		return nil, err
	}

	s := &Session{
		Config:   cfg,
		Handlers: handlers,
	}

	initHandlers(s)

	// Setup HTTP client with custom cert bundle if enabled
	if opts.CustomCABundle != nil {
		if err := loadCustomCABundle(s, opts.CustomCABundle); err != nil {
			return nil, err
		}
	}

	return s, nil
}

func loadCustomCABundle(s *Session, bundle io.Reader) error {
	var t *http.Transport
	switch v := s.Config.HTTPClient.Transport.(type) {
	case *http.Transport:
		t = v
	default:
		if s.Config.HTTPClient.Transport != nil {
			return awserr.New("LoadCustomCABundleError",
				"unable to load custom CA bundle, HTTPClient's transport unsupported type", nil)
		}
	}
	if t == nil {
		t = &http.Transport{}
	}

	p, err := loadCertPool(bundle)
	if err != nil {
		return err
	}
	if t.TLSClientConfig == nil {
		t.TLSClientConfig = &tls.Config{}
	}
	t.TLSClientConfig.RootCAs = p

	s.Config.HTTPClient.Transport = t

	return nil
}

func loadCertPool(r io.Reader) (*x509.CertPool, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, awserr.New("LoadCustomCABundleError",
			"failed to read custom CA bundle PEM file", err)
	}

	p := x509.NewCertPool()
	if !p.AppendCertsFromPEM(b) {
		return nil, awserr.New("LoadCustomCABundleError",
			"failed to load custom CA bundle PEM file", err)
	}

	return p, nil
}

func mergeConfigSrcs(cfg, userCfg *aws.Config, envCfg envConfig, sharedCfg sharedConfig, handlers request.Handlers, sessOpts Options) error {
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
			if len(sharedCfg.AssumeRole.MFASerial) > 0 && sessOpts.AssumeRoleTokenProvider == nil {
				// AssumeRole Token provider is required if doing Assume Role
				// with MFA.
				return AssumeRoleTokenProviderNotSetError{}
			}
			cfg.Credentials = stscreds.NewCredentials(
				&Session{
					Config:   &cfgCp,
					Handlers: handlers.Copy(),
				},
				sharedCfg.AssumeRole.RoleARN,
				func(opt *stscreds.AssumeRoleProvider) {
					opt.RoleSessionName = sharedCfg.AssumeRole.RoleSessionName

					// Assume role with external ID
					if len(sharedCfg.AssumeRole.ExternalID) > 0 {
						opt.ExternalID = aws.String(sharedCfg.AssumeRole.ExternalID)
					}

					// Assume role with MFA
					if len(sharedCfg.AssumeRole.MFASerial) > 0 {
						opt.SerialNumber = aws.String(sharedCfg.AssumeRole.MFASerial)
						opt.TokenProvider = sessOpts.AssumeRoleTokenProvider
					}
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

	return nil
}

// AssumeRoleTokenProviderNotSetError is an error returned when creating a session when the
// MFAToken option is not set when shared config is configured load assume a
// role with an MFA token.
type AssumeRoleTokenProviderNotSetError struct{}

// Code is the short id of the error.
func (e AssumeRoleTokenProviderNotSetError) Code() string {
	return "AssumeRoleTokenProviderNotSetError"
}

// Message is the description of the error
func (e AssumeRoleTokenProviderNotSetError) Message() string {
	return fmt.Sprintf("assume role with MFA enabled, but AssumeRoleTokenProvider session option not set.")
}

// OrigErr is the underlying error that caused the failure.
func (e AssumeRoleTokenProviderNotSetError) OrigErr() error {
	return nil
}

// Error satisfies the error interface.
func (e AssumeRoleTokenProviderNotSetError) Error() string {
	return awserr.SprintError(e.Code(), e.Message(), "", nil)
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
	// Backwards compatibility, the error will be eaten if user calls ClientConfig
	// directly. All SDK services will use ClientconfigWithError.
	cfg, _ := s.clientConfigWithErr(serviceName, cfgs...)

	return cfg
}

func (s *Session) clientConfigWithErr(serviceName string, cfgs ...*aws.Config) (client.Config, error) {
	s = s.Copy(cfgs...)

	var resolved endpoints.ResolvedEndpoint
	var err error

	region := aws.StringValue(s.Config.Region)

	if endpoint := aws.StringValue(s.Config.Endpoint); len(endpoint) != 0 {
		resolved.URL = endpoints.AddScheme(endpoint, aws.BoolValue(s.Config.DisableSSL))
		resolved.SigningRegion = region
	} else {
		resolved, err = s.Config.EndpointResolver.EndpointFor(
			serviceName, region,
			func(opt *endpoints.Options) {
				opt.DisableSSL = aws.BoolValue(s.Config.DisableSSL)
				opt.UseDualStack = aws.BoolValue(s.Config.UseDualStack)

				// Support the condition where the service is modeled but its
				// endpoint metadata is not available.
				opt.ResolveUnknownService = true
			},
		)
	}

	return client.Config{
		Config:        s.Config,
		Handlers:      s.Handlers,
		Endpoint:      resolved.URL,
		SigningRegion: resolved.SigningRegion,
		SigningName:   resolved.SigningName,
	}, err
}

// ClientConfigNoResolveEndpoint is the same as ClientConfig with the exception
// that the EndpointResolver will not be used to resolve the endpoint. The only
// endpoint set must come from the aws.Config.Endpoint field.
func (s *Session) ClientConfigNoResolveEndpoint(cfgs ...*aws.Config) client.Config {
	s = s.Copy(cfgs...)

	var resolved endpoints.ResolvedEndpoint

	region := aws.StringValue(s.Config.Region)

	if ep := aws.StringValue(s.Config.Endpoint); len(ep) > 0 {
		resolved.URL = endpoints.AddScheme(ep, aws.BoolValue(s.Config.DisableSSL))
		resolved.SigningRegion = region
	}

	return client.Config{
		Config:        s.Config,
		Handlers:      s.Handlers,
		Endpoint:      resolved.URL,
		SigningRegion: resolved.SigningRegion,
		SigningName:   resolved.SigningName,
	}
}
