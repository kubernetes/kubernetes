package aws

import (
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/endpoints"
)

// UseServiceDefaultRetries instructs the config to use the service's own
// default number of retries. This will be the default action if
// Config.MaxRetries is nil also.
const UseServiceDefaultRetries = -1

// RequestRetryer is an alias for a type that implements the request.Retryer
// interface.
type RequestRetryer interface{}

// A Config provides service configuration for service clients. By default,
// all clients will use the defaults.DefaultConfig structure.
//
//     // Create Session with MaxRetries configuration to be shared by multiple
//     // service clients.
//     sess := session.Must(session.NewSession(&aws.Config{
//         MaxRetries: aws.Int(3),
//     }))
//
//     // Create S3 service client with a specific Region.
//     svc := s3.New(sess, &aws.Config{
//         Region: aws.String("us-west-2"),
//     })
type Config struct {
	// Enables verbose error printing of all credential chain errors.
	// Should be used when wanting to see all errors while attempting to
	// retrieve credentials.
	CredentialsChainVerboseErrors *bool

	// The credentials object to use when signing requests. Defaults to a
	// chain of credential providers to search for credentials in environment
	// variables, shared credential file, and EC2 Instance Roles.
	Credentials *credentials.Credentials

	// An optional endpoint URL (hostname only or fully qualified URI)
	// that overrides the default generated endpoint for a client. Set this
	// to `nil` or the value to `""` to use the default generated endpoint.
	//
	// Note: You must still provide a `Region` value when specifying an
	// endpoint for a client.
	Endpoint *string

	// The resolver to use for looking up endpoints for AWS service clients
	// to use based on region.
	EndpointResolver endpoints.Resolver

	// EnforceShouldRetryCheck is used in the AfterRetryHandler to always call
	// ShouldRetry regardless of whether or not if request.Retryable is set.
	// This will utilize ShouldRetry method of custom retryers. If EnforceShouldRetryCheck
	// is not set, then ShouldRetry will only be called if request.Retryable is nil.
	// Proper handling of the request.Retryable field is important when setting this field.
	EnforceShouldRetryCheck *bool

	// The region to send requests to. This parameter is required and must
	// be configured globally or on a per-client basis unless otherwise
	// noted. A full list of regions is found in the "Regions and Endpoints"
	// document.
	//
	// See http://docs.aws.amazon.com/general/latest/gr/rande.html for AWS
	// Regions and Endpoints.
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
	// Defaults to -1, which defers the max retry setting to the service
	// specific configuration.
	MaxRetries *int

	// Retryer guides how HTTP requests should be retried in case of
	// recoverable failures.
	//
	// When nil or the value does not implement the request.Retryer interface,
	// the client.DefaultRetryer will be used.
	//
	// When both Retryer and MaxRetries are non-nil, the former is used and
	// the latter ignored.
	//
	// To set the Retryer field in a type-safe manner and with chaining, use
	// the request.WithRetryer helper function:
	//
	//   cfg := request.WithRetryer(aws.NewConfig(), myRetryer)
	//
	Retryer RequestRetryer

	// Disables semantic parameter validation, which validates input for
	// missing required fields and/or other semantic request input errors.
	DisableParamValidation *bool

	// Disables the computation of request and response checksums, e.g.,
	// CRC32 checksums in Amazon DynamoDB.
	DisableComputeChecksums *bool

	// Set this to `true` to force the request to use path-style addressing,
	// i.e., `http://s3.amazonaws.com/BUCKET/KEY`. By default, the S3 client
	// will use virtual hosted bucket addressing when possible
	// (`http://BUCKET.s3.amazonaws.com/KEY`).
	//
	// Note: This configuration option is specific to the Amazon S3 service.
	//
	// See http://docs.aws.amazon.com/AmazonS3/latest/dev/VirtualHosting.html
	// for Amazon S3: Virtual Hosting of Buckets
	S3ForcePathStyle *bool

	// Set this to `true` to disable the SDK adding the `Expect: 100-Continue`
	// header to PUT requests over 2MB of content. 100-Continue instructs the
	// HTTP client not to send the body until the service responds with a
	// `continue` status. This is useful to prevent sending the request body
	// until after the request is authenticated, and validated.
	//
	// http://docs.aws.amazon.com/AmazonS3/latest/API/RESTObjectPUT.html
	//
	// 100-Continue is only enabled for Go 1.6 and above. See `http.Transport`'s
	// `ExpectContinueTimeout` for information on adjusting the continue wait
	// timeout. https://golang.org/pkg/net/http/#Transport
	//
	// You should use this flag to disable 100-Continue if you experience issues
	// with proxies or third party S3 compatible services.
	S3Disable100Continue *bool

	// Set this to `true` to enable S3 Accelerate feature. For all operations
	// compatible with S3 Accelerate will use the accelerate endpoint for
	// requests. Requests not compatible will fall back to normal S3 requests.
	//
	// The bucket must be enable for accelerate to be used with S3 client with
	// accelerate enabled. If the bucket is not enabled for accelerate an error
	// will be returned. The bucket name must be DNS compatible to also work
	// with accelerate.
	S3UseAccelerate *bool

	// S3DisableContentMD5Validation config option is temporarily disabled,
	// For S3 GetObject API calls, #1837.
	//
	// Set this to `true` to disable the S3 service client from automatically
	// adding the ContentMD5 to S3 Object Put and Upload API calls. This option
	// will also disable the SDK from performing object ContentMD5 validation
	// on GetObject API calls.
	S3DisableContentMD5Validation *bool

	// Set this to `true` to have the S3 service client to use the region specified
	// in the ARN, when an ARN is provided as an argument to a bucket parameter.
	S3UseARNRegion *bool

	// Set this to `true` to enable the SDK to unmarshal API response header maps to
	// normalized lower case map keys.
	//
	// For example S3's X-Amz-Meta prefixed header will be unmarshaled to lower case
	// Metadata member's map keys. The value of the header in the map is unaffected.
	LowerCaseHeaderMaps *bool

	// Set this to `true` to disable the EC2Metadata client from overriding the
	// default http.Client's Timeout. This is helpful if you do not want the
	// EC2Metadata client to create a new http.Client. This options is only
	// meaningful if you're not already using a custom HTTP client with the
	// SDK. Enabled by default.
	//
	// Must be set and provided to the session.NewSession() in order to disable
	// the EC2Metadata overriding the timeout for default credentials chain.
	//
	// Example:
	//    sess := session.Must(session.NewSession(aws.NewConfig()
	//       .WithEC2MetadataDisableTimeoutOverride(true)))
	//
	//    svc := s3.New(sess)
	//
	EC2MetadataDisableTimeoutOverride *bool

	// Instructs the endpoint to be generated for a service client to
	// be the dual stack endpoint. The dual stack endpoint will support
	// both IPv4 and IPv6 addressing.
	//
	// Setting this for a service which does not support dual stack will fail
	// to make requests. It is not recommended to set this value on the session
	// as it will apply to all service clients created with the session. Even
	// services which don't support dual stack endpoints.
	//
	// If the Endpoint config value is also provided the UseDualStack flag
	// will be ignored.
	//
	// Only supported with.
	//
	//     sess := session.Must(session.NewSession())
	//
	//     svc := s3.New(sess, &aws.Config{
	//         UseDualStack: aws.Bool(true),
	//     })
	UseDualStack *bool

	// SleepDelay is an override for the func the SDK will call when sleeping
	// during the lifecycle of a request. Specifically this will be used for
	// request delays. This value should only be used for testing. To adjust
	// the delay of a request see the aws/client.DefaultRetryer and
	// aws/request.Retryer.
	//
	// SleepDelay will prevent any Context from being used for canceling retry
	// delay of an API operation. It is recommended to not use SleepDelay at all
	// and specify a Retryer instead.
	SleepDelay func(time.Duration)

	// DisableRestProtocolURICleaning will not clean the URL path when making rest protocol requests.
	// Will default to false. This would only be used for empty directory names in s3 requests.
	//
	// Example:
	//    sess := session.Must(session.NewSession(&aws.Config{
	//         DisableRestProtocolURICleaning: aws.Bool(true),
	//    }))
	//
	//    svc := s3.New(sess)
	//    out, err := svc.GetObject(&s3.GetObjectInput {
	//    	Bucket: aws.String("bucketname"),
	//    	Key: aws.String("//foo//bar//moo"),
	//    })
	DisableRestProtocolURICleaning *bool

	// EnableEndpointDiscovery will allow for endpoint discovery on operations that
	// have the definition in its model. By default, endpoint discovery is off.
	// To use EndpointDiscovery, Endpoint should be unset or set to an empty string.
	//
	// Example:
	//    sess := session.Must(session.NewSession(&aws.Config{
	//         EnableEndpointDiscovery: aws.Bool(true),
	//    }))
	//
	//    svc := s3.New(sess)
	//    out, err := svc.GetObject(&s3.GetObjectInput {
	//    	Bucket: aws.String("bucketname"),
	//    	Key: aws.String("/foo/bar/moo"),
	//    })
	EnableEndpointDiscovery *bool

	// DisableEndpointHostPrefix will disable the SDK's behavior of prefixing
	// request endpoint hosts with modeled information.
	//
	// Disabling this feature is useful when you want to use local endpoints
	// for testing that do not support the modeled host prefix pattern.
	DisableEndpointHostPrefix *bool

	// STSRegionalEndpoint will enable regional or legacy endpoint resolving
	STSRegionalEndpoint endpoints.STSRegionalEndpoint

	// S3UsEast1RegionalEndpoint will enable regional or legacy endpoint resolving
	S3UsEast1RegionalEndpoint endpoints.S3UsEast1RegionalEndpoint
}

// NewConfig returns a new Config pointer that can be chained with builder
// methods to set multiple configuration values inline without using pointers.
//
//     // Create Session with MaxRetries configuration to be shared by multiple
//     // service clients.
//     sess := session.Must(session.NewSession(aws.NewConfig().
//         WithMaxRetries(3),
//     ))
//
//     // Create S3 service client with a specific Region.
//     svc := s3.New(sess, aws.NewConfig().
//         WithRegion("us-west-2"),
//     )
func NewConfig() *Config {
	return &Config{}
}

// WithCredentialsChainVerboseErrors sets a config verbose errors boolean and returning
// a Config pointer.
func (c *Config) WithCredentialsChainVerboseErrors(verboseErrs bool) *Config {
	c.CredentialsChainVerboseErrors = &verboseErrs
	return c
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

// WithEndpointResolver sets a config EndpointResolver value returning a
// Config pointer for chaining.
func (c *Config) WithEndpointResolver(resolver endpoints.Resolver) *Config {
	c.EndpointResolver = resolver
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

// WithS3Disable100Continue sets a config S3Disable100Continue value returning
// a Config pointer for chaining.
func (c *Config) WithS3Disable100Continue(disable bool) *Config {
	c.S3Disable100Continue = &disable
	return c
}

// WithS3UseAccelerate sets a config S3UseAccelerate value returning a Config
// pointer for chaining.
func (c *Config) WithS3UseAccelerate(enable bool) *Config {
	c.S3UseAccelerate = &enable
	return c

}

// WithS3DisableContentMD5Validation sets a config
// S3DisableContentMD5Validation value returning a Config pointer for chaining.
func (c *Config) WithS3DisableContentMD5Validation(enable bool) *Config {
	c.S3DisableContentMD5Validation = &enable
	return c

}

// WithS3UseARNRegion sets a config S3UseARNRegion value and
// returning a Config pointer for chaining
func (c *Config) WithS3UseARNRegion(enable bool) *Config {
	c.S3UseARNRegion = &enable
	return c
}

// WithUseDualStack sets a config UseDualStack value returning a Config
// pointer for chaining.
func (c *Config) WithUseDualStack(enable bool) *Config {
	c.UseDualStack = &enable
	return c
}

// WithEC2MetadataDisableTimeoutOverride sets a config EC2MetadataDisableTimeoutOverride value
// returning a Config pointer for chaining.
func (c *Config) WithEC2MetadataDisableTimeoutOverride(enable bool) *Config {
	c.EC2MetadataDisableTimeoutOverride = &enable
	return c
}

// WithSleepDelay overrides the function used to sleep while waiting for the
// next retry. Defaults to time.Sleep.
func (c *Config) WithSleepDelay(fn func(time.Duration)) *Config {
	c.SleepDelay = fn
	return c
}

// WithEndpointDiscovery will set whether or not to use endpoint discovery.
func (c *Config) WithEndpointDiscovery(t bool) *Config {
	c.EnableEndpointDiscovery = &t
	return c
}

// WithDisableEndpointHostPrefix will set whether or not to use modeled host prefix
// when making requests.
func (c *Config) WithDisableEndpointHostPrefix(t bool) *Config {
	c.DisableEndpointHostPrefix = &t
	return c
}

// WithSTSRegionalEndpoint will set whether or not to use regional endpoint flag
// when resolving the endpoint for a service
func (c *Config) WithSTSRegionalEndpoint(sre endpoints.STSRegionalEndpoint) *Config {
	c.STSRegionalEndpoint = sre
	return c
}

// WithS3UsEast1RegionalEndpoint will set whether or not to use regional endpoint flag
// when resolving the endpoint for a service
func (c *Config) WithS3UsEast1RegionalEndpoint(sre endpoints.S3UsEast1RegionalEndpoint) *Config {
	c.S3UsEast1RegionalEndpoint = sre
	return c
}

// WithLowerCaseHeaderMaps sets a config LowerCaseHeaderMaps value
// returning a Config pointer for chaining.
func (c *Config) WithLowerCaseHeaderMaps(t bool) *Config {
	c.LowerCaseHeaderMaps = &t
	return c
}

// WithDisableRestProtocolURICleaning sets a config DisableRestProtocolURICleaning value
// returning a Config pointer for chaining.
func (c *Config) WithDisableRestProtocolURICleaning(t bool) *Config {
	c.DisableRestProtocolURICleaning = &t
	return c
}

// MergeIn merges the passed in configs into the existing config object.
func (c *Config) MergeIn(cfgs ...*Config) {
	for _, other := range cfgs {
		mergeInConfig(c, other)
	}
}

func mergeInConfig(dst *Config, other *Config) {
	if other == nil {
		return
	}

	if other.CredentialsChainVerboseErrors != nil {
		dst.CredentialsChainVerboseErrors = other.CredentialsChainVerboseErrors
	}

	if other.Credentials != nil {
		dst.Credentials = other.Credentials
	}

	if other.Endpoint != nil {
		dst.Endpoint = other.Endpoint
	}

	if other.EndpointResolver != nil {
		dst.EndpointResolver = other.EndpointResolver
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

	if other.Retryer != nil {
		dst.Retryer = other.Retryer
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

	if other.S3Disable100Continue != nil {
		dst.S3Disable100Continue = other.S3Disable100Continue
	}

	if other.S3UseAccelerate != nil {
		dst.S3UseAccelerate = other.S3UseAccelerate
	}

	if other.S3DisableContentMD5Validation != nil {
		dst.S3DisableContentMD5Validation = other.S3DisableContentMD5Validation
	}

	if other.S3UseARNRegion != nil {
		dst.S3UseARNRegion = other.S3UseARNRegion
	}

	if other.UseDualStack != nil {
		dst.UseDualStack = other.UseDualStack
	}

	if other.EC2MetadataDisableTimeoutOverride != nil {
		dst.EC2MetadataDisableTimeoutOverride = other.EC2MetadataDisableTimeoutOverride
	}

	if other.SleepDelay != nil {
		dst.SleepDelay = other.SleepDelay
	}

	if other.DisableRestProtocolURICleaning != nil {
		dst.DisableRestProtocolURICleaning = other.DisableRestProtocolURICleaning
	}

	if other.EnforceShouldRetryCheck != nil {
		dst.EnforceShouldRetryCheck = other.EnforceShouldRetryCheck
	}

	if other.EnableEndpointDiscovery != nil {
		dst.EnableEndpointDiscovery = other.EnableEndpointDiscovery
	}

	if other.DisableEndpointHostPrefix != nil {
		dst.DisableEndpointHostPrefix = other.DisableEndpointHostPrefix
	}

	if other.STSRegionalEndpoint != endpoints.UnsetSTSEndpoint {
		dst.STSRegionalEndpoint = other.STSRegionalEndpoint
	}

	if other.S3UsEast1RegionalEndpoint != endpoints.UnsetS3UsEast1Endpoint {
		dst.S3UsEast1RegionalEndpoint = other.S3UsEast1RegionalEndpoint
	}

	if other.LowerCaseHeaderMaps != nil {
		dst.LowerCaseHeaderMaps = other.LowerCaseHeaderMaps
	}
}

// Copy will return a shallow copy of the Config object. If any additional
// configurations are provided they will be merged into the new config returned.
func (c *Config) Copy(cfgs ...*Config) *Config {
	dst := &Config{}
	dst.MergeIn(c)

	for _, cfg := range cfgs {
		dst.MergeIn(cfg)
	}

	return dst
}
