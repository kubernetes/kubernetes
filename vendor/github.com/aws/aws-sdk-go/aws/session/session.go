package session

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/csm"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
)

const (
	// ErrCodeSharedConfig represents an error that occurs in the shared
	// configuration logic
	ErrCodeSharedConfig = "SharedConfigErr"

	// ErrCodeLoadCustomCABundle error code for unable to load custom CA bundle.
	ErrCodeLoadCustomCABundle = "LoadCustomCABundleError"

	// ErrCodeLoadClientTLSCert error code for unable to load client TLS
	// certificate or key
	ErrCodeLoadClientTLSCert = "LoadClientTLSCertError"
)

// ErrSharedConfigSourceCollision will be returned if a section contains both
// source_profile and credential_source
var ErrSharedConfigSourceCollision = awserr.New(ErrCodeSharedConfig, "only one credential type may be specified per profile: source profile, credential source, credential process, web identity token, or sso", nil)

// ErrSharedConfigECSContainerEnvVarEmpty will be returned if the environment
// variables are empty and Environment was set as the credential source
var ErrSharedConfigECSContainerEnvVarEmpty = awserr.New(ErrCodeSharedConfig, "EcsContainer was specified as the credential_source, but 'AWS_CONTAINER_CREDENTIALS_RELATIVE_URI' was not set", nil)

// ErrSharedConfigInvalidCredSource will be returned if an invalid credential source was provided
var ErrSharedConfigInvalidCredSource = awserr.New(ErrCodeSharedConfig, "credential source values must be EcsContainer, Ec2InstanceMetadata, or Environment", nil)

// A Session provides a central location to create service clients from and
// store configurations and request handlers for those services.
//
// Sessions are safe to create service clients concurrently, but it is not safe
// to mutate the Session concurrently.
//
// The Session satisfies the service client's client.ConfigProvider.
type Session struct {
	Config   *aws.Config
	Handlers request.Handlers

	options Options
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
	envCfg, envErr := loadEnvConfig()

	if envCfg.EnableSharedConfig {
		var cfg aws.Config
		cfg.MergeIn(cfgs...)
		s, err := NewSessionWithOptions(Options{
			Config:            cfg,
			SharedConfigState: SharedConfigEnable,
		})
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
			s.logDeprecatedNewSessionError(msg, err, cfgs)
		}

		return s
	}

	s := deprecatedNewSession(envCfg, cfgs...)
	if envErr != nil {
		msg := "failed to load env config"
		s.logDeprecatedNewSessionError(msg, envErr, cfgs)
	}

	if csmCfg, err := loadCSMConfig(envCfg, []string{}); err != nil {
		if l := s.Config.Logger; l != nil {
			l.Log(fmt.Sprintf("ERROR: failed to load CSM configuration, %v", err))
		}
	} else if csmCfg.Enabled {
		err := enableCSM(&s.Handlers, csmCfg, s.Config.Logger)
		if err != nil {
			msg := "failed to enable CSM"
			s.logDeprecatedNewSessionError(msg, err, cfgs)
		}
	}

	return s
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
// control through code how the Session will be created, such as specifying the
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

	// When the SDK's shared config is configured to assume a role this option
	// may be provided to set the expiry duration of the STS credentials.
	// Defaults to 15 minutes if not set as documented in the
	// stscreds.AssumeRoleProvider.
	AssumeRoleDuration time.Duration

	// Reader for a custom Credentials Authority (CA) bundle in PEM format that
	// the SDK will use instead of the default system's root CA bundle. Use this
	// only if you want to replace the CA bundle the SDK uses for TLS requests.
	//
	// HTTP Client's Transport concrete implementation must be a http.Transport
	// or creating the session will fail.
	//
	// If the Transport's TLS config is set this option will cause the SDK
	// to overwrite the Transport's TLS config's  RootCAs value. If the CA
	// bundle reader contains multiple certificates all of them will be loaded.
	//
	// Can also be specified via the environment variable:
	//
	//  AWS_CA_BUNDLE=$HOME/ca_bundle
	//
	// Can also be specified via the shared config field:
	//
	//  ca_bundle = $HOME/ca_bundle
	CustomCABundle io.Reader

	// Reader for the TLC client certificate that should be used by the SDK's
	// HTTP transport when making requests. The certificate must be paired with
	// a TLS client key file. Will be ignored if both are not provided.
	//
	// HTTP Client's Transport concrete implementation must be a http.Transport
	// or creating the session will fail.
	//
	// Can also be specified via the environment variable:
	//
	//  AWS_SDK_GO_CLIENT_TLS_CERT=$HOME/my_client_cert
	ClientTLSCert io.Reader

	// Reader for the TLC client key that should be used by the SDK's HTTP
	// transport when making requests. The key must be paired with a TLS client
	// certificate file. Will be ignored if both are not provided.
	//
	// HTTP Client's Transport concrete implementation must be a http.Transport
	// or creating the session will fail.
	//
	// Can also be specified via the environment variable:
	//
	//  AWS_SDK_GO_CLIENT_TLS_KEY=$HOME/my_client_key
	ClientTLSKey io.Reader

	// The handlers that the session and all API clients will be created with.
	// This must be a complete set of handlers. Use the defaults.Handlers()
	// function to initialize this value before changing the handlers to be
	// used by the SDK.
	Handlers request.Handlers

	// Allows specifying a custom endpoint to be used by the EC2 IMDS client
	// when making requests to the EC2 IMDS API. The endpoint value should
	// include the URI scheme. If the scheme is not present it will be defaulted to http.
	//
	// If unset, will the EC2 IMDS client will use its default endpoint.
	//
	// Can also be specified via the environment variable,
	// AWS_EC2_METADATA_SERVICE_ENDPOINT.
	//
	//   AWS_EC2_METADATA_SERVICE_ENDPOINT=http://169.254.169.254
	//
	// If using an URL with an IPv6 address literal, the IPv6 address
	// component must be enclosed in square brackets.
	//
	//   AWS_EC2_METADATA_SERVICE_ENDPOINT=http://[::1]
	EC2IMDSEndpoint string

	// Specifies the EC2 Instance Metadata Service default endpoint selection mode (IPv4 or IPv6)
	//
	// AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE=IPv6
	EC2IMDSEndpointMode endpoints.EC2IMDSEndpointModeState

	// Specifies options for creating credential providers.
	// These are only used if the aws.Config does not already
	// include credentials.
	CredentialsProviderOptions *CredentialsProviderOptions
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
	var err error
	if opts.SharedConfigState == SharedConfigEnable {
		envCfg, err = loadSharedEnvConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to load shared config, %v", err)
		}
	} else {
		envCfg, err = loadEnvConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to load environment config, %v", err)
		}
	}

	if len(opts.Profile) != 0 {
		envCfg.Profile = opts.Profile
	}

	switch opts.SharedConfigState {
	case SharedConfigDisable:
		envCfg.EnableSharedConfig = false
	case SharedConfigEnable:
		envCfg.EnableSharedConfig = true
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

// Wraps the endpoint resolver with a resolver that will return a custom
// endpoint for EC2 IMDS.
func wrapEC2IMDSEndpoint(resolver endpoints.Resolver, endpoint string, mode endpoints.EC2IMDSEndpointModeState) endpoints.Resolver {
	return endpoints.ResolverFunc(
		func(service, region string, opts ...func(*endpoints.Options)) (
			endpoints.ResolvedEndpoint, error,
		) {
			if service == ec2MetadataServiceID && len(endpoint) > 0 {
				return endpoints.ResolvedEndpoint{
					URL:           endpoint,
					SigningName:   ec2MetadataServiceID,
					SigningRegion: region,
				}, nil
			} else if service == ec2MetadataServiceID {
				opts = append(opts, func(o *endpoints.Options) {
					o.EC2MetadataEndpointMode = mode
				})
			}
			return resolver.EndpointFor(service, region, opts...)
		})
}

func deprecatedNewSession(envCfg envConfig, cfgs ...*aws.Config) *Session {
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

	if !(len(envCfg.EC2IMDSEndpoint) == 0 && envCfg.EC2IMDSEndpointMode == endpoints.EC2IMDSEndpointModeStateUnset) {
		cfg.EndpointResolver = wrapEC2IMDSEndpoint(cfg.EndpointResolver, envCfg.EC2IMDSEndpoint, envCfg.EC2IMDSEndpointMode)
	}

	cfg.Credentials = defaults.CredChain(cfg, handlers)

	// Reapply any passed in configs to override credentials if set
	cfg.MergeIn(cfgs...)

	s := &Session{
		Config:   cfg,
		Handlers: handlers,
		options: Options{
			EC2IMDSEndpoint: envCfg.EC2IMDSEndpoint,
		},
	}

	initHandlers(s)
	return s
}

func enableCSM(handlers *request.Handlers, cfg csmConfig, logger aws.Logger) error {
	if logger != nil {
		logger.Log("Enabling CSM")
	}

	r, err := csm.Start(cfg.ClientID, csm.AddressWithDefaults(cfg.Host, cfg.Port))
	if err != nil {
		return err
	}
	r.InjectHandlers(handlers)

	return nil
}

func newSession(opts Options, envCfg envConfig, cfgs ...*aws.Config) (*Session, error) {
	cfg := defaults.Config()

	handlers := opts.Handlers
	if handlers.IsEmpty() {
		handlers = defaults.Handlers()
	}

	// Get a merged version of the user provided config to determine if
	// credentials were.
	userCfg := &aws.Config{}
	userCfg.MergeIn(cfgs...)
	cfg.MergeIn(userCfg)

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
	sharedCfg, err := loadSharedConfig(envCfg.Profile, cfgFiles, envCfg.EnableSharedConfig)
	if err != nil {
		if len(envCfg.Profile) == 0 && !envCfg.EnableSharedConfig && (envCfg.Creds.HasKeys() || userCfg.Credentials != nil) {
			// Special case where the user has not explicitly specified an AWS_PROFILE,
			// or session.Options.profile, shared config is not enabled, and the
			// environment has credentials, allow the shared config file to fail to
			// load since the user has already provided credentials, and nothing else
			// is required to be read file. Github(aws/aws-sdk-go#2455)
		} else if _, ok := err.(SharedConfigProfileNotExistsError); !ok {
			return nil, err
		}
	}

	if err := mergeConfigSrcs(cfg, userCfg, envCfg, sharedCfg, handlers, opts); err != nil {
		return nil, err
	}

	if err := setTLSOptions(&opts, cfg, envCfg, sharedCfg); err != nil {
		return nil, err
	}

	s := &Session{
		Config:   cfg,
		Handlers: handlers,
		options:  opts,
	}

	initHandlers(s)

	if csmCfg, err := loadCSMConfig(envCfg, cfgFiles); err != nil {
		if l := s.Config.Logger; l != nil {
			l.Log(fmt.Sprintf("ERROR: failed to load CSM configuration, %v", err))
		}
	} else if csmCfg.Enabled {
		err = enableCSM(&s.Handlers, csmCfg, s.Config.Logger)
		if err != nil {
			return nil, err
		}
	}

	return s, nil
}

type csmConfig struct {
	Enabled  bool
	Host     string
	Port     string
	ClientID string
}

var csmProfileName = "aws_csm"

func loadCSMConfig(envCfg envConfig, cfgFiles []string) (csmConfig, error) {
	if envCfg.CSMEnabled != nil {
		if *envCfg.CSMEnabled {
			return csmConfig{
				Enabled:  true,
				ClientID: envCfg.CSMClientID,
				Host:     envCfg.CSMHost,
				Port:     envCfg.CSMPort,
			}, nil
		}
		return csmConfig{}, nil
	}

	sharedCfg, err := loadSharedConfig(csmProfileName, cfgFiles, false)
	if err != nil {
		if _, ok := err.(SharedConfigProfileNotExistsError); !ok {
			return csmConfig{}, err
		}
	}
	if sharedCfg.CSMEnabled != nil && *sharedCfg.CSMEnabled == true {
		return csmConfig{
			Enabled:  true,
			ClientID: sharedCfg.CSMClientID,
			Host:     sharedCfg.CSMHost,
			Port:     sharedCfg.CSMPort,
		}, nil
	}

	return csmConfig{}, nil
}

func setTLSOptions(opts *Options, cfg *aws.Config, envCfg envConfig, sharedCfg sharedConfig) error {
	// CA Bundle can be specified in both environment variable shared config file.
	var caBundleFilename = envCfg.CustomCABundle
	if len(caBundleFilename) == 0 {
		caBundleFilename = sharedCfg.CustomCABundle
	}

	// Only use environment value if session option is not provided.
	customTLSOptions := map[string]struct {
		filename string
		field    *io.Reader
		errCode  string
	}{
		"custom CA bundle PEM":   {filename: caBundleFilename, field: &opts.CustomCABundle, errCode: ErrCodeLoadCustomCABundle},
		"custom client TLS cert": {filename: envCfg.ClientTLSCert, field: &opts.ClientTLSCert, errCode: ErrCodeLoadClientTLSCert},
		"custom client TLS key":  {filename: envCfg.ClientTLSKey, field: &opts.ClientTLSKey, errCode: ErrCodeLoadClientTLSCert},
	}
	for name, v := range customTLSOptions {
		if len(v.filename) != 0 && *v.field == nil {
			f, err := os.Open(v.filename)
			if err != nil {
				return awserr.New(v.errCode, fmt.Sprintf("failed to open %s file", name), err)
			}
			defer f.Close()
			*v.field = f
		}
	}

	// Setup HTTP client with custom cert bundle if enabled
	if opts.CustomCABundle != nil {
		if err := loadCustomCABundle(cfg.HTTPClient, opts.CustomCABundle); err != nil {
			return err
		}
	}

	// Setup HTTP client TLS certificate and key for client TLS authentication.
	if opts.ClientTLSCert != nil && opts.ClientTLSKey != nil {
		if err := loadClientTLSCert(cfg.HTTPClient, opts.ClientTLSCert, opts.ClientTLSKey); err != nil {
			return err
		}
	} else if opts.ClientTLSCert == nil && opts.ClientTLSKey == nil {
		// Do nothing if neither values are available.

	} else {
		return awserr.New(ErrCodeLoadClientTLSCert,
			fmt.Sprintf("client TLS cert(%t) and key(%t) must both be provided",
				opts.ClientTLSCert != nil, opts.ClientTLSKey != nil), nil)
	}

	return nil
}

func getHTTPTransport(client *http.Client) (*http.Transport, error) {
	var t *http.Transport
	switch v := client.Transport.(type) {
	case *http.Transport:
		t = v
	default:
		if client.Transport != nil {
			return nil, fmt.Errorf("unsupported transport, %T", client.Transport)
		}
	}
	if t == nil {
		// Nil transport implies `http.DefaultTransport` should be used. Since
		// the SDK cannot modify, nor copy the `DefaultTransport` specifying
		// the values the next closest behavior.
		t = getCustomTransport()
	}

	return t, nil
}

func loadCustomCABundle(client *http.Client, bundle io.Reader) error {
	t, err := getHTTPTransport(client)
	if err != nil {
		return awserr.New(ErrCodeLoadCustomCABundle,
			"unable to load custom CA bundle, HTTPClient's transport unsupported type", err)
	}

	p, err := loadCertPool(bundle)
	if err != nil {
		return err
	}
	if t.TLSClientConfig == nil {
		t.TLSClientConfig = &tls.Config{}
	}
	t.TLSClientConfig.RootCAs = p

	client.Transport = t

	return nil
}

func loadCertPool(r io.Reader) (*x509.CertPool, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, awserr.New(ErrCodeLoadCustomCABundle,
			"failed to read custom CA bundle PEM file", err)
	}

	p := x509.NewCertPool()
	if !p.AppendCertsFromPEM(b) {
		return nil, awserr.New(ErrCodeLoadCustomCABundle,
			"failed to load custom CA bundle PEM file", err)
	}

	return p, nil
}

func loadClientTLSCert(client *http.Client, certFile, keyFile io.Reader) error {
	t, err := getHTTPTransport(client)
	if err != nil {
		return awserr.New(ErrCodeLoadClientTLSCert,
			"unable to get usable HTTP transport from client", err)
	}

	cert, err := ioutil.ReadAll(certFile)
	if err != nil {
		return awserr.New(ErrCodeLoadClientTLSCert,
			"unable to get read client TLS cert file", err)
	}

	key, err := ioutil.ReadAll(keyFile)
	if err != nil {
		return awserr.New(ErrCodeLoadClientTLSCert,
			"unable to get read client TLS key file", err)
	}

	clientCert, err := tls.X509KeyPair(cert, key)
	if err != nil {
		return awserr.New(ErrCodeLoadClientTLSCert,
			"unable to load x509 key pair from client cert", err)
	}

	tlsCfg := t.TLSClientConfig
	if tlsCfg == nil {
		tlsCfg = &tls.Config{}
	}

	tlsCfg.Certificates = append(tlsCfg.Certificates, clientCert)

	t.TLSClientConfig = tlsCfg
	client.Transport = t

	return nil
}

func mergeConfigSrcs(cfg, userCfg *aws.Config,
	envCfg envConfig, sharedCfg sharedConfig,
	handlers request.Handlers,
	sessOpts Options,
) error {

	// Region if not already set by user
	if len(aws.StringValue(cfg.Region)) == 0 {
		if len(envCfg.Region) > 0 {
			cfg.WithRegion(envCfg.Region)
		} else if envCfg.EnableSharedConfig && len(sharedCfg.Region) > 0 {
			cfg.WithRegion(sharedCfg.Region)
		}
	}

	if cfg.EnableEndpointDiscovery == nil {
		if envCfg.EnableEndpointDiscovery != nil {
			cfg.WithEndpointDiscovery(*envCfg.EnableEndpointDiscovery)
		} else if envCfg.EnableSharedConfig && sharedCfg.EnableEndpointDiscovery != nil {
			cfg.WithEndpointDiscovery(*sharedCfg.EnableEndpointDiscovery)
		}
	}

	// Regional Endpoint flag for STS endpoint resolving
	mergeSTSRegionalEndpointConfig(cfg, []endpoints.STSRegionalEndpoint{
		userCfg.STSRegionalEndpoint,
		envCfg.STSRegionalEndpoint,
		sharedCfg.STSRegionalEndpoint,
		endpoints.LegacySTSEndpoint,
	})

	// Regional Endpoint flag for S3 endpoint resolving
	mergeS3UsEast1RegionalEndpointConfig(cfg, []endpoints.S3UsEast1RegionalEndpoint{
		userCfg.S3UsEast1RegionalEndpoint,
		envCfg.S3UsEast1RegionalEndpoint,
		sharedCfg.S3UsEast1RegionalEndpoint,
		endpoints.LegacyS3UsEast1Endpoint,
	})

	var ec2IMDSEndpoint string
	for _, v := range []string{
		sessOpts.EC2IMDSEndpoint,
		envCfg.EC2IMDSEndpoint,
		sharedCfg.EC2IMDSEndpoint,
	} {
		if len(v) != 0 {
			ec2IMDSEndpoint = v
			break
		}
	}

	var endpointMode endpoints.EC2IMDSEndpointModeState
	for _, v := range []endpoints.EC2IMDSEndpointModeState{
		sessOpts.EC2IMDSEndpointMode,
		envCfg.EC2IMDSEndpointMode,
		sharedCfg.EC2IMDSEndpointMode,
	} {
		if v != endpoints.EC2IMDSEndpointModeStateUnset {
			endpointMode = v
			break
		}
	}

	if len(ec2IMDSEndpoint) != 0 || endpointMode != endpoints.EC2IMDSEndpointModeStateUnset {
		cfg.EndpointResolver = wrapEC2IMDSEndpoint(cfg.EndpointResolver, ec2IMDSEndpoint, endpointMode)
	}

	// Configure credentials if not already set by the user when creating the
	// Session.
	if cfg.Credentials == credentials.AnonymousCredentials && userCfg.Credentials == nil {
		creds, err := resolveCredentials(cfg, envCfg, sharedCfg, handlers, sessOpts)
		if err != nil {
			return err
		}
		cfg.Credentials = creds
	}

	cfg.S3UseARNRegion = userCfg.S3UseARNRegion
	if cfg.S3UseARNRegion == nil {
		cfg.S3UseARNRegion = &envCfg.S3UseARNRegion
	}
	if cfg.S3UseARNRegion == nil {
		cfg.S3UseARNRegion = &sharedCfg.S3UseARNRegion
	}

	for _, v := range []endpoints.DualStackEndpointState{userCfg.UseDualStackEndpoint, envCfg.UseDualStackEndpoint, sharedCfg.UseDualStackEndpoint} {
		if v != endpoints.DualStackEndpointStateUnset {
			cfg.UseDualStackEndpoint = v
			break
		}
	}

	for _, v := range []endpoints.FIPSEndpointState{userCfg.UseFIPSEndpoint, envCfg.UseFIPSEndpoint, sharedCfg.UseFIPSEndpoint} {
		if v != endpoints.FIPSEndpointStateUnset {
			cfg.UseFIPSEndpoint = v
			break
		}
	}

	return nil
}

func mergeSTSRegionalEndpointConfig(cfg *aws.Config, values []endpoints.STSRegionalEndpoint) {
	for _, v := range values {
		if v != endpoints.UnsetSTSEndpoint {
			cfg.STSRegionalEndpoint = v
			break
		}
	}
}

func mergeS3UsEast1RegionalEndpointConfig(cfg *aws.Config, values []endpoints.S3UsEast1RegionalEndpoint) {
	for _, v := range values {
		if v != endpoints.UnsetS3UsEast1Endpoint {
			cfg.S3UsEast1RegionalEndpoint = v
			break
		}
	}
}

func initHandlers(s *Session) {
	// Add the Validate parameter handler if it is not disabled.
	s.Handlers.Validate.Remove(corehandlers.ValidateParametersHandler)
	if !aws.BoolValue(s.Config.DisableParamValidation) {
		s.Handlers.Validate.PushBackNamed(corehandlers.ValidateParametersHandler)
	}
}

// Copy creates and returns a copy of the current Session, copying the config
// and handlers. If any additional configs are provided they will be merged
// on top of the Session's copied config.
//
//     // Create a copy of the current Session, configured for the us-west-2 region.
//     sess.Copy(&aws.Config{Region: aws.String("us-west-2")})
func (s *Session) Copy(cfgs ...*aws.Config) *Session {
	newSession := &Session{
		Config:   s.Config.Copy(cfgs...),
		Handlers: s.Handlers.Copy(),
		options:  s.options,
	}

	initHandlers(newSession)

	return newSession
}

// ClientConfig satisfies the client.ConfigProvider interface and is used to
// configure the service client instances. Passing the Session to the service
// client's constructor (New) will use this method to configure the client.
func (s *Session) ClientConfig(service string, cfgs ...*aws.Config) client.Config {
	s = s.Copy(cfgs...)

	resolvedRegion := normalizeRegion(s.Config)

	region := aws.StringValue(s.Config.Region)
	resolved, err := s.resolveEndpoint(service, region, resolvedRegion, s.Config)
	if err != nil {
		s.Handlers.Validate.PushBack(func(r *request.Request) {
			if len(r.ClientInfo.Endpoint) != 0 {
				// Error occurred while resolving endpoint, but the request
				// being invoked has had an endpoint specified after the client
				// was created.
				return
			}
			r.Error = err
		})
	}

	return client.Config{
		Config:             s.Config,
		Handlers:           s.Handlers,
		PartitionID:        resolved.PartitionID,
		Endpoint:           resolved.URL,
		SigningRegion:      resolved.SigningRegion,
		SigningNameDerived: resolved.SigningNameDerived,
		SigningName:        resolved.SigningName,
		ResolvedRegion:     resolvedRegion,
	}
}

const ec2MetadataServiceID = "ec2metadata"

func (s *Session) resolveEndpoint(service, region, resolvedRegion string, cfg *aws.Config) (endpoints.ResolvedEndpoint, error) {

	if ep := aws.StringValue(cfg.Endpoint); len(ep) != 0 {
		return endpoints.ResolvedEndpoint{
			URL:           endpoints.AddScheme(ep, aws.BoolValue(cfg.DisableSSL)),
			SigningRegion: region,
		}, nil
	}

	resolved, err := cfg.EndpointResolver.EndpointFor(service, region,
		func(opt *endpoints.Options) {
			opt.DisableSSL = aws.BoolValue(cfg.DisableSSL)

			opt.UseDualStack = aws.BoolValue(cfg.UseDualStack)
			opt.UseDualStackEndpoint = cfg.UseDualStackEndpoint

			opt.UseFIPSEndpoint = cfg.UseFIPSEndpoint

			// Support for STSRegionalEndpoint where the STSRegionalEndpoint is
			// provided in envConfig or sharedConfig with envConfig getting
			// precedence.
			opt.STSRegionalEndpoint = cfg.STSRegionalEndpoint

			// Support for S3UsEast1RegionalEndpoint where the S3UsEast1RegionalEndpoint is
			// provided in envConfig or sharedConfig with envConfig getting
			// precedence.
			opt.S3UsEast1RegionalEndpoint = cfg.S3UsEast1RegionalEndpoint

			// Support the condition where the service is modeled but its
			// endpoint metadata is not available.
			opt.ResolveUnknownService = true

			opt.ResolvedRegion = resolvedRegion

			opt.Logger = cfg.Logger
			opt.LogDeprecated = cfg.LogLevel.Matches(aws.LogDebugWithDeprecated)
		},
	)
	if err != nil {
		return endpoints.ResolvedEndpoint{}, err
	}

	return resolved, nil
}

// ClientConfigNoResolveEndpoint is the same as ClientConfig with the exception
// that the EndpointResolver will not be used to resolve the endpoint. The only
// endpoint set must come from the aws.Config.Endpoint field.
func (s *Session) ClientConfigNoResolveEndpoint(cfgs ...*aws.Config) client.Config {
	s = s.Copy(cfgs...)

	resolvedRegion := normalizeRegion(s.Config)

	var resolved endpoints.ResolvedEndpoint
	if ep := aws.StringValue(s.Config.Endpoint); len(ep) > 0 {
		resolved.URL = endpoints.AddScheme(ep, aws.BoolValue(s.Config.DisableSSL))
		resolved.SigningRegion = aws.StringValue(s.Config.Region)
	}

	return client.Config{
		Config:             s.Config,
		Handlers:           s.Handlers,
		Endpoint:           resolved.URL,
		SigningRegion:      resolved.SigningRegion,
		SigningNameDerived: resolved.SigningNameDerived,
		SigningName:        resolved.SigningName,
		ResolvedRegion:     resolvedRegion,
	}
}

// logDeprecatedNewSessionError function enables error handling for session
func (s *Session) logDeprecatedNewSessionError(msg string, err error, cfgs []*aws.Config) {
	// Session creation failed, need to report the error and prevent
	// any requests from succeeding.
	s.Config.MergeIn(cfgs...)
	s.Config.Logger.Log("ERROR:", msg, "Error:", err)
	s.Handlers.Validate.PushBack(func(r *request.Request) {
		r.Error = err
	})
}

// normalizeRegion resolves / normalizes the configured region (converts pseudo fips regions), and modifies the provided
// config to have the equivalent options for resolution and returns the resolved region name.
func normalizeRegion(cfg *aws.Config) (resolved string) {
	const fipsInfix = "-fips-"
	const fipsPrefix = "-fips"
	const fipsSuffix = "fips-"

	region := aws.StringValue(cfg.Region)

	if strings.Contains(region, fipsInfix) ||
		strings.Contains(region, fipsPrefix) ||
		strings.Contains(region, fipsSuffix) {
		resolved = strings.Replace(strings.Replace(strings.Replace(
			region, fipsInfix, "-", -1), fipsPrefix, "", -1), fipsSuffix, "", -1)
		cfg.UseFIPSEndpoint = endpoints.FIPSEndpointStateEnabled
	}

	return resolved
}
