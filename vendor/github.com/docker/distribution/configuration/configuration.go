package configuration

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"time"
)

// Configuration is a versioned registry configuration, intended to be provided by a yaml file, and
// optionally modified by environment variables.
//
// Note that yaml field names should never include _ characters, since this is the separator used
// in environment variable names.
type Configuration struct {
	// Version is the version which defines the format of the rest of the configuration
	Version Version `yaml:"version"`

	// Log supports setting various parameters related to the logging
	// subsystem.
	Log struct {
		// AccessLog configures access logging.
		AccessLog struct {
			// Disabled disables access logging.
			Disabled bool `yaml:"disabled,omitempty"`
		} `yaml:"accesslog,omitempty"`

		// Level is the granularity at which registry operations are logged.
		Level Loglevel `yaml:"level"`

		// Formatter overrides the default formatter with another. Options
		// include "text", "json" and "logstash".
		Formatter string `yaml:"formatter,omitempty"`

		// Fields allows users to specify static string fields to include in
		// the logger context.
		Fields map[string]interface{} `yaml:"fields,omitempty"`

		// Hooks allows users to configure the log hooks, to enabling the
		// sequent handling behavior, when defined levels of log message emit.
		Hooks []LogHook `yaml:"hooks,omitempty"`
	}

	// Loglevel is the level at which registry operations are logged. This is
	// deprecated. Please use Log.Level in the future.
	Loglevel Loglevel `yaml:"loglevel,omitempty"`

	// Storage is the configuration for the registry's storage driver
	Storage Storage `yaml:"storage"`

	// Auth allows configuration of various authorization methods that may be
	// used to gate requests.
	Auth Auth `yaml:"auth,omitempty"`

	// Middleware lists all middlewares to be used by the registry.
	Middleware map[string][]Middleware `yaml:"middleware,omitempty"`

	// Reporting is the configuration for error reporting
	Reporting Reporting `yaml:"reporting,omitempty"`

	// HTTP contains configuration parameters for the registry's http
	// interface.
	HTTP struct {
		// Addr specifies the bind address for the registry instance.
		Addr string `yaml:"addr,omitempty"`

		// Net specifies the net portion of the bind address. A default empty value means tcp.
		Net string `yaml:"net,omitempty"`

		// Host specifies an externally-reachable address for the registry, as a fully
		// qualified URL.
		Host string `yaml:"host,omitempty"`

		Prefix string `yaml:"prefix,omitempty"`

		// Secret specifies the secret key which HMAC tokens are created with.
		Secret string `yaml:"secret,omitempty"`

		// RelativeURLs specifies that relative URLs should be returned in
		// Location headers
		RelativeURLs bool `yaml:"relativeurls,omitempty"`

		// TLS instructs the http server to listen with a TLS configuration.
		// This only support simple tls configuration with a cert and key.
		// Mostly, this is useful for testing situations or simple deployments
		// that require tls. If more complex configurations are required, use
		// a proxy or make a proposal to add support here.
		TLS struct {
			// Certificate specifies the path to an x509 certificate file to
			// be used for TLS.
			Certificate string `yaml:"certificate,omitempty"`

			// Key specifies the path to the x509 key file, which should
			// contain the private portion for the file specified in
			// Certificate.
			Key string `yaml:"key,omitempty"`

			// Specifies the CA certs for client authentication
			// A file may contain multiple CA certificates encoded as PEM
			ClientCAs []string `yaml:"clientcas,omitempty"`

			// LetsEncrypt is used to configuration setting up TLS through
			// Let's Encrypt instead of manually specifying certificate and
			// key. If a TLS certificate is specified, the Let's Encrypt
			// section will not be used.
			LetsEncrypt struct {
				// CacheFile specifies cache file to use for lets encrypt
				// certificates and keys.
				CacheFile string `yaml:"cachefile,omitempty"`

				// Email is the email to use during Let's Encrypt registration
				Email string `yaml:"email,omitempty"`
			} `yaml:"letsencrypt,omitempty"`
		} `yaml:"tls,omitempty"`

		// Headers is a set of headers to include in HTTP responses. A common
		// use case for this would be security headers such as
		// Strict-Transport-Security. The map keys are the header names, and
		// the values are the associated header payloads.
		Headers http.Header `yaml:"headers,omitempty"`

		// Debug configures the http debug interface, if specified. This can
		// include services such as pprof, expvar and other data that should
		// not be exposed externally. Left disabled by default.
		Debug struct {
			// Addr specifies the bind address for the debug server.
			Addr string `yaml:"addr,omitempty"`
		} `yaml:"debug,omitempty"`

		// HTTP2 configuration options
		HTTP2 struct {
			// Specifies whether the registry should disallow clients attempting
			// to connect via http2. If set to true, only http/1.1 is supported.
			Disabled bool `yaml:"disabled,omitempty"`
		} `yaml:"http2,omitempty"`
	} `yaml:"http,omitempty"`

	// Notifications specifies configuration about various endpoint to which
	// registry events are dispatched.
	Notifications Notifications `yaml:"notifications,omitempty"`

	// Redis configures the redis pool available to the registry webapp.
	Redis struct {
		// Addr specifies the the redis instance available to the application.
		Addr string `yaml:"addr,omitempty"`

		// Password string to use when making a connection.
		Password string `yaml:"password,omitempty"`

		// DB specifies the database to connect to on the redis instance.
		DB int `yaml:"db,omitempty"`

		DialTimeout  time.Duration `yaml:"dialtimeout,omitempty"`  // timeout for connect
		ReadTimeout  time.Duration `yaml:"readtimeout,omitempty"`  // timeout for reads of data
		WriteTimeout time.Duration `yaml:"writetimeout,omitempty"` // timeout for writes of data

		// Pool configures the behavior of the redis connection pool.
		Pool struct {
			// MaxIdle sets the maximum number of idle connections.
			MaxIdle int `yaml:"maxidle,omitempty"`

			// MaxActive sets the maximum number of connections that should be
			// opened before blocking a connection request.
			MaxActive int `yaml:"maxactive,omitempty"`

			// IdleTimeout sets the amount time to wait before closing
			// inactive connections.
			IdleTimeout time.Duration `yaml:"idletimeout,omitempty"`
		} `yaml:"pool,omitempty"`
	} `yaml:"redis,omitempty"`

	Health Health `yaml:"health,omitempty"`

	Proxy Proxy `yaml:"proxy,omitempty"`

	// Compatibility is used for configurations of working with older or deprecated features.
	Compatibility struct {
		// Schema1 configures how schema1 manifests will be handled
		Schema1 struct {
			// TrustKey is the signing key to use for adding the signature to
			// schema1 manifests.
			TrustKey string `yaml:"signingkeyfile,omitempty"`
		} `yaml:"schema1,omitempty"`
	} `yaml:"compatibility,omitempty"`

	// Validation configures validation options for the registry.
	Validation struct {
		// Enabled enables the other options in this section. This field is
		// deprecated in favor of Disabled.
		Enabled bool `yaml:"enabled,omitempty"`
		// Disabled disables the other options in this section.
		Disabled bool `yaml:"disabled,omitempty"`
		// Manifests configures manifest validation.
		Manifests struct {
			// URLs configures validation for URLs in pushed manifests.
			URLs struct {
				// Allow specifies regular expressions (https://godoc.org/regexp/syntax)
				// that URLs in pushed manifests must match.
				Allow []string `yaml:"allow,omitempty"`
				// Deny specifies regular expressions (https://godoc.org/regexp/syntax)
				// that URLs in pushed manifests must not match.
				Deny []string `yaml:"deny,omitempty"`
			} `yaml:"urls,omitempty"`
		} `yaml:"manifests,omitempty"`
	} `yaml:"validation,omitempty"`

	// Policy configures registry policy options.
	Policy struct {
		// Repository configures policies for repositories
		Repository struct {
			// Classes is a list of repository classes which the
			// registry allows content for. This class is matched
			// against the configuration media type inside uploaded
			// manifests. When non-empty, the registry will enforce
			// the class in authorized resources.
			Classes []string `yaml:"classes"`
		} `yaml:"repository,omitempty"`
	} `yaml:"policy,omitempty"`
}

// LogHook is composed of hook Level and Type.
// After hooks configuration, it can execute the next handling automatically,
// when defined levels of log message emitted.
// Example: hook can sending an email notification when error log happens in app.
type LogHook struct {
	// Disable lets user select to enable hook or not.
	Disabled bool `yaml:"disabled,omitempty"`

	// Type allows user to select which type of hook handler they want.
	Type string `yaml:"type,omitempty"`

	// Levels set which levels of log message will let hook executed.
	Levels []string `yaml:"levels,omitempty"`

	// MailOptions allows user to configure email parameters.
	MailOptions MailOptions `yaml:"options,omitempty"`
}

// MailOptions provides the configuration sections to user, for specific handler.
type MailOptions struct {
	SMTP struct {
		// Addr defines smtp host address
		Addr string `yaml:"addr,omitempty"`

		// Username defines user name to smtp host
		Username string `yaml:"username,omitempty"`

		// Password defines password of login user
		Password string `yaml:"password,omitempty"`

		// Insecure defines if smtp login skips the secure certification.
		Insecure bool `yaml:"insecure,omitempty"`
	} `yaml:"smtp,omitempty"`

	// From defines mail sending address
	From string `yaml:"from,omitempty"`

	// To defines mail receiving address
	To []string `yaml:"to,omitempty"`
}

// FileChecker is a type of entry in the health section for checking files.
type FileChecker struct {
	// Interval is the duration in between checks
	Interval time.Duration `yaml:"interval,omitempty"`
	// File is the path to check
	File string `yaml:"file,omitempty"`
	// Threshold is the number of times a check must fail to trigger an
	// unhealthy state
	Threshold int `yaml:"threshold,omitempty"`
}

// HTTPChecker is a type of entry in the health section for checking HTTP URIs.
type HTTPChecker struct {
	// Timeout is the duration to wait before timing out the HTTP request
	Timeout time.Duration `yaml:"timeout,omitempty"`
	// StatusCode is the expected status code
	StatusCode int
	// Interval is the duration in between checks
	Interval time.Duration `yaml:"interval,omitempty"`
	// URI is the HTTP URI to check
	URI string `yaml:"uri,omitempty"`
	// Headers lists static headers that should be added to all requests
	Headers http.Header `yaml:"headers"`
	// Threshold is the number of times a check must fail to trigger an
	// unhealthy state
	Threshold int `yaml:"threshold,omitempty"`
}

// TCPChecker is a type of entry in the health section for checking TCP servers.
type TCPChecker struct {
	// Timeout is the duration to wait before timing out the TCP connection
	Timeout time.Duration `yaml:"timeout,omitempty"`
	// Interval is the duration in between checks
	Interval time.Duration `yaml:"interval,omitempty"`
	// Addr is the TCP address to check
	Addr string `yaml:"addr,omitempty"`
	// Threshold is the number of times a check must fail to trigger an
	// unhealthy state
	Threshold int `yaml:"threshold,omitempty"`
}

// Health provides the configuration section for health checks.
type Health struct {
	// FileCheckers is a list of paths to check
	FileCheckers []FileChecker `yaml:"file,omitempty"`
	// HTTPCheckers is a list of URIs to check
	HTTPCheckers []HTTPChecker `yaml:"http,omitempty"`
	// TCPCheckers is a list of URIs to check
	TCPCheckers []TCPChecker `yaml:"tcp,omitempty"`
	// StorageDriver configures a health check on the configured storage
	// driver
	StorageDriver struct {
		// Enabled turns on the health check for the storage driver
		Enabled bool `yaml:"enabled,omitempty"`
		// Interval is the duration in between checks
		Interval time.Duration `yaml:"interval,omitempty"`
		// Threshold is the number of times a check must fail to trigger an
		// unhealthy state
		Threshold int `yaml:"threshold,omitempty"`
	} `yaml:"storagedriver,omitempty"`
}

// v0_1Configuration is a Version 0.1 Configuration struct
// This is currently aliased to Configuration, as it is the current version
type v0_1Configuration Configuration

// UnmarshalYAML implements the yaml.Unmarshaler interface
// Unmarshals a string of the form X.Y into a Version, validating that X and Y can represent unsigned integers
func (version *Version) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var versionString string
	err := unmarshal(&versionString)
	if err != nil {
		return err
	}

	newVersion := Version(versionString)
	if _, err := newVersion.major(); err != nil {
		return err
	}

	if _, err := newVersion.minor(); err != nil {
		return err
	}

	*version = newVersion
	return nil
}

// CurrentVersion is the most recent Version that can be parsed
var CurrentVersion = MajorMinorVersion(0, 1)

// Loglevel is the level at which operations are logged
// This can be error, warn, info, or debug
type Loglevel string

// UnmarshalYAML implements the yaml.Umarshaler interface
// Unmarshals a string into a Loglevel, lowercasing the string and validating that it represents a
// valid loglevel
func (loglevel *Loglevel) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var loglevelString string
	err := unmarshal(&loglevelString)
	if err != nil {
		return err
	}

	loglevelString = strings.ToLower(loglevelString)
	switch loglevelString {
	case "error", "warn", "info", "debug":
	default:
		return fmt.Errorf("Invalid loglevel %s Must be one of [error, warn, info, debug]", loglevelString)
	}

	*loglevel = Loglevel(loglevelString)
	return nil
}

// Parameters defines a key-value parameters mapping
type Parameters map[string]interface{}

// Storage defines the configuration for registry object storage
type Storage map[string]Parameters

// Type returns the storage driver type, such as filesystem or s3
func (storage Storage) Type() string {
	var storageType []string

	// Return only key in this map
	for k := range storage {
		switch k {
		case "maintenance":
			// allow configuration of maintenance
		case "cache":
			// allow configuration of caching
		case "delete":
			// allow configuration of delete
		case "redirect":
			// allow configuration of redirect
		default:
			storageType = append(storageType, k)
		}
	}
	if len(storageType) > 1 {
		panic("multiple storage drivers specified in configuration or environment: " + strings.Join(storageType, ", "))
	}
	if len(storageType) == 1 {
		return storageType[0]
	}
	return ""
}

// Parameters returns the Parameters map for a Storage configuration
func (storage Storage) Parameters() Parameters {
	return storage[storage.Type()]
}

// setParameter changes the parameter at the provided key to the new value
func (storage Storage) setParameter(key string, value interface{}) {
	storage[storage.Type()][key] = value
}

// UnmarshalYAML implements the yaml.Unmarshaler interface
// Unmarshals a single item map into a Storage or a string into a Storage type with no parameters
func (storage *Storage) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var storageMap map[string]Parameters
	err := unmarshal(&storageMap)
	if err == nil {
		if len(storageMap) > 1 {
			types := make([]string, 0, len(storageMap))
			for k := range storageMap {
				switch k {
				case "maintenance":
					// allow for configuration of maintenance
				case "cache":
					// allow configuration of caching
				case "delete":
					// allow configuration of delete
				case "redirect":
					// allow configuration of redirect
				default:
					types = append(types, k)
				}
			}

			if len(types) > 1 {
				return fmt.Errorf("Must provide exactly one storage type. Provided: %v", types)
			}
		}
		*storage = storageMap
		return nil
	}

	var storageType string
	err = unmarshal(&storageType)
	if err == nil {
		*storage = Storage{storageType: Parameters{}}
		return nil
	}

	return err
}

// MarshalYAML implements the yaml.Marshaler interface
func (storage Storage) MarshalYAML() (interface{}, error) {
	if storage.Parameters() == nil {
		return storage.Type(), nil
	}
	return map[string]Parameters(storage), nil
}

// Auth defines the configuration for registry authorization.
type Auth map[string]Parameters

// Type returns the auth type, such as htpasswd or token
func (auth Auth) Type() string {
	// Return only key in this map
	for k := range auth {
		return k
	}
	return ""
}

// Parameters returns the Parameters map for an Auth configuration
func (auth Auth) Parameters() Parameters {
	return auth[auth.Type()]
}

// setParameter changes the parameter at the provided key to the new value
func (auth Auth) setParameter(key string, value interface{}) {
	auth[auth.Type()][key] = value
}

// UnmarshalYAML implements the yaml.Unmarshaler interface
// Unmarshals a single item map into a Storage or a string into a Storage type with no parameters
func (auth *Auth) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var m map[string]Parameters
	err := unmarshal(&m)
	if err == nil {
		if len(m) > 1 {
			types := make([]string, 0, len(m))
			for k := range m {
				types = append(types, k)
			}

			// TODO(stevvooe): May want to change this slightly for
			// authorization to allow multiple challenges.
			return fmt.Errorf("must provide exactly one type. Provided: %v", types)

		}
		*auth = m
		return nil
	}

	var authType string
	err = unmarshal(&authType)
	if err == nil {
		*auth = Auth{authType: Parameters{}}
		return nil
	}

	return err
}

// MarshalYAML implements the yaml.Marshaler interface
func (auth Auth) MarshalYAML() (interface{}, error) {
	if auth.Parameters() == nil {
		return auth.Type(), nil
	}
	return map[string]Parameters(auth), nil
}

// Notifications configures multiple http endpoints.
type Notifications struct {
	// Endpoints is a list of http configurations for endpoints that
	// respond to webhook notifications. In the future, we may allow other
	// kinds of endpoints, such as external queues.
	Endpoints []Endpoint `yaml:"endpoints,omitempty"`
}

// Endpoint describes the configuration of an http webhook notification
// endpoint.
type Endpoint struct {
	Name              string        `yaml:"name"`              // identifies the endpoint in the registry instance.
	Disabled          bool          `yaml:"disabled"`          // disables the endpoint
	URL               string        `yaml:"url"`               // post url for the endpoint.
	Headers           http.Header   `yaml:"headers"`           // static headers that should be added to all requests
	Timeout           time.Duration `yaml:"timeout"`           // HTTP timeout
	Threshold         int           `yaml:"threshold"`         // circuit breaker threshold before backing off on failure
	Backoff           time.Duration `yaml:"backoff"`           // backoff duration
	IgnoredMediaTypes []string      `yaml:"ignoredmediatypes"` // target media types to ignore
}

// Reporting defines error reporting methods.
type Reporting struct {
	// Bugsnag configures error reporting for Bugsnag (bugsnag.com).
	Bugsnag BugsnagReporting `yaml:"bugsnag,omitempty"`
	// NewRelic configures error reporting for NewRelic (newrelic.com)
	NewRelic NewRelicReporting `yaml:"newrelic,omitempty"`
}

// BugsnagReporting configures error reporting for Bugsnag (bugsnag.com).
type BugsnagReporting struct {
	// APIKey is the Bugsnag api key.
	APIKey string `yaml:"apikey,omitempty"`
	// ReleaseStage tracks where the registry is deployed.
	// Examples: production, staging, development
	ReleaseStage string `yaml:"releasestage,omitempty"`
	// Endpoint is used for specifying an enterprise Bugsnag endpoint.
	Endpoint string `yaml:"endpoint,omitempty"`
}

// NewRelicReporting configures error reporting for NewRelic (newrelic.com)
type NewRelicReporting struct {
	// LicenseKey is the NewRelic user license key
	LicenseKey string `yaml:"licensekey,omitempty"`
	// Name is the component name of the registry in NewRelic
	Name string `yaml:"name,omitempty"`
	// Verbose configures debug output to STDOUT
	Verbose bool `yaml:"verbose,omitempty"`
}

// Middleware configures named middlewares to be applied at injection points.
type Middleware struct {
	// Name the middleware registers itself as
	Name string `yaml:"name"`
	// Flag to disable middleware easily
	Disabled bool `yaml:"disabled,omitempty"`
	// Map of parameters that will be passed to the middleware's initialization function
	Options Parameters `yaml:"options"`
}

// Proxy configures the registry as a pull through cache
type Proxy struct {
	// RemoteURL is the URL of the remote registry
	RemoteURL string `yaml:"remoteurl"`

	// Username of the hub user
	Username string `yaml:"username"`

	// Password of the hub user
	Password string `yaml:"password"`
}

// Parse parses an input configuration yaml document into a Configuration struct
// This should generally be capable of handling old configuration format versions
//
// Environment variables may be used to override configuration parameters other than version,
// following the scheme below:
// Configuration.Abc may be replaced by the value of REGISTRY_ABC,
// Configuration.Abc.Xyz may be replaced by the value of REGISTRY_ABC_XYZ, and so forth
func Parse(rd io.Reader) (*Configuration, error) {
	in, err := ioutil.ReadAll(rd)
	if err != nil {
		return nil, err
	}

	p := NewParser("registry", []VersionedParseInfo{
		{
			Version: MajorMinorVersion(0, 1),
			ParseAs: reflect.TypeOf(v0_1Configuration{}),
			ConversionFunc: func(c interface{}) (interface{}, error) {
				if v0_1, ok := c.(*v0_1Configuration); ok {
					if v0_1.Loglevel == Loglevel("") {
						v0_1.Loglevel = Loglevel("info")
					}
					if v0_1.Storage.Type() == "" {
						return nil, errors.New("No storage configuration provided")
					}
					return (*Configuration)(v0_1), nil
				}
				return nil, fmt.Errorf("Expected *v0_1Configuration, received %#v", c)
			},
		},
	})

	config := new(Configuration)
	err = p.Parse(in, config)
	if err != nil {
		return nil, err
	}

	return config, nil
}
