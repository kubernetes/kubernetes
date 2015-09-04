package aws

import (
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws/credentials"
)

// The default number of retries for a service. The value of -1 indicates that
// the service specific retry default will be used.
const DefaultRetries = -1

// A Config provides service configuration for service clients. By default,
// all clients will use the {defaults.DefaultConfig} structure.
type Config struct {
	// The credentials object to use when signing requests. Defaults to
	// {defaults.DefaultChainCredentials}.
	Credentials *credentials.Credentials

	// An optional endpoint URL (hostname only or fully qualified URI)
	// that overrides the default generated endpoint for a client. Set this
	// to `""` to use the default generated endpoint.
	//
	// @note You must still provide a `Region` value when specifying an
	//   endpoint for a client.
	Endpoint *string

	// The region to send requests to. This parameter is required and must
	// be configured globally or on a per-client basis unless otherwise
	// noted. A full list of regions is found in the "Regions and Endpoints"
	// document.
	//
	// @see http://docs.aws.amazon.com/general/latest/gr/rande.html
	//   AWS Regions and Endpoints
	Region *string

	// Set this to `true` to disable SSL when sending requests. Defaults
	// to `false`.
	DisableSSL *bool

	// The HTTP client to use when sending requests. Defaults to
	// `http.DefaultClient`.
	HTTPClient *http.Client

	// An integer value representing the logging level. The default log level
	// is zero (LogOff), which represents no logging. To enable logging set
	// to a LogLevel Value.
	LogLevel *LogLevelType

	// The logger writer interface to write logging messages to. Defaults to
	// standard out.
	Logger Logger

	// The maximum number of times that a request will be retried for failures.
	// Defaults to -1, which defers the max retry setting to the service specific
	// configuration.
	MaxRetries *int

	// Disables semantic parameter validation, which validates input for missing
	// required fields and/or other semantic request input errors.
	DisableParamValidation *bool

	// Disables the computation of request and response checksums, e.g.,
	// CRC32 checksums in Amazon DynamoDB.
	DisableComputeChecksums *bool

	// Set this to `true` to force the request to use path-style addressing,
	// i.e., `http://s3.amazonaws.com/BUCKET/KEY`. By default, the S3 client will
	// use virtual hosted bucket addressing when possible
	// (`http://BUCKET.s3.amazonaws.com/KEY`).
	//
	// @note This configuration option is specific to the Amazon S3 service.
	// @see http://docs.aws.amazon.com/AmazonS3/latest/dev/VirtualHosting.html
	//   Amazon S3: Virtual Hosting of Buckets
	S3ForcePathStyle *bool

	SleepDelay func(time.Duration)
}

// NewConfig returns a new Config pointer that can be chained with builder methods to
// set multiple configuration values inline without using pointers.
//
//     svc := s3.New(aws.NewConfig().WithRegion("us-west-2").WithMaxRetries(10))
//
func NewConfig() *Config {
	return &Config{}
}

// WithCredentials sets a config Credentials value returning a Config pointer
// for chaining.
func (c *Config) WithCredentials(creds *credentials.Credentials) *Config {
	c.Credentials = creds
	return c
}

// WithEndpoint sets a config Endpoint value returning a Config pointer for
// chaining.
func (c *Config) WithEndpoint(endpoint string) *Config {
	c.Endpoint = &endpoint
	return c
}

// WithRegion sets a config Region value returning a Config pointer for
// chaining.
func (c *Config) WithRegion(region string) *Config {
	c.Region = &region
	return c
}

// WithDisableSSL sets a config DisableSSL value returning a Config pointer
// for chaining.
func (c *Config) WithDisableSSL(disable bool) *Config {
	c.DisableSSL = &disable
	return c
}

// WithHTTPClient sets a config HTTPClient value returning a Config pointer
// for chaining.
func (c *Config) WithHTTPClient(client *http.Client) *Config {
	c.HTTPClient = client
	return c
}

// WithMaxRetries sets a config MaxRetries value returning a Config pointer
// for chaining.
func (c *Config) WithMaxRetries(max int) *Config {
	c.MaxRetries = &max
	return c
}

// WithDisableParamValidation sets a config DisableParamValidation value
// returning a Config pointer for chaining.
func (c *Config) WithDisableParamValidation(disable bool) *Config {
	c.DisableParamValidation = &disable
	return c
}

// WithDisableComputeChecksums sets a config DisableComputeChecksums value
// returning a Config pointer for chaining.
func (c *Config) WithDisableComputeChecksums(disable bool) *Config {
	c.DisableComputeChecksums = &disable
	return c
}

// WithLogLevel sets a config LogLevel value returning a Config pointer for
// chaining.
func (c *Config) WithLogLevel(level LogLevelType) *Config {
	c.LogLevel = &level
	return c
}

// WithLogger sets a config Logger value returning a Config pointer for
// chaining.
func (c *Config) WithLogger(logger Logger) *Config {
	c.Logger = logger
	return c
}

// WithS3ForcePathStyle sets a config S3ForcePathStyle value returning a Config
// pointer for chaining.
func (c *Config) WithS3ForcePathStyle(force bool) *Config {
	c.S3ForcePathStyle = &force
	return c
}

// WithSleepDelay overrides the function used to sleep while waiting for the
// next retry. Defaults to time.Sleep.
func (c *Config) WithSleepDelay(fn func(time.Duration)) *Config {
	c.SleepDelay = fn
	return c
}

// Merge returns a new Config with the other Config's attribute values merged into
// this Config. If the other Config's attribute is nil it will not be merged into
// the new Config to be returned.
func (c Config) Merge(other *Config) *Config {
	if other == nil {
		return &c
	}

	dst := c

	if other.Credentials != nil {
		dst.Credentials = other.Credentials
	}

	if other.Endpoint != nil {
		dst.Endpoint = other.Endpoint
	}

	if other.Region != nil {
		dst.Region = other.Region
	}

	if other.DisableSSL != nil {
		dst.DisableSSL = other.DisableSSL
	}

	if other.HTTPClient != nil {
		dst.HTTPClient = other.HTTPClient
	}

	if other.LogLevel != nil {
		dst.LogLevel = other.LogLevel
	}

	if other.Logger != nil {
		dst.Logger = other.Logger
	}

	if other.MaxRetries != nil {
		dst.MaxRetries = other.MaxRetries
	}

	if other.DisableParamValidation != nil {
		dst.DisableParamValidation = other.DisableParamValidation
	}

	if other.DisableComputeChecksums != nil {
		dst.DisableComputeChecksums = other.DisableComputeChecksums
	}

	if other.S3ForcePathStyle != nil {
		dst.S3ForcePathStyle = other.S3ForcePathStyle
	}

	if other.SleepDelay != nil {
		dst.SleepDelay = other.SleepDelay
	}

	return &dst
}

// Copy will return a shallow copy of the Config object.
func (c Config) Copy() *Config {
	dst := c
	return &dst
}
