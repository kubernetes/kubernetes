package ec2metadata

import (
	"io/ioutil"
	"net/http"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/service"
	"github.com/aws/aws-sdk-go/aws/service/serviceinfo"
)

// DefaultRetries states the default number of times the service client will
// attempt to retry a failed request before failing.
const DefaultRetries = 3

// A Config provides the configuration for the EC2 Metadata service.
type Config struct {
	// An optional endpoint URL (hostname only or fully qualified URI)
	// that overrides the default service endpoint for a client. Set this
	// to nil, or `""` to use the default service endpoint.
	Endpoint *string

	// The HTTP client to use when sending requests. Defaults to
	// `http.DefaultClient`.
	HTTPClient *http.Client

	// An integer value representing the logging level. The default log level
	// is zero (LogOff), which represents no logging. To enable logging set
	// to a LogLevel Value.
	Logger aws.Logger

	// The logger writer interface to write logging messages to. Defaults to
	// standard out.
	LogLevel *aws.LogLevelType

	// The maximum number of times that a request will be retried for failures.
	// Defaults to DefaultRetries for the number of retries to be performed
	// per request.
	MaxRetries *int
}

// A Client is an EC2 Metadata service Client.
type Client struct {
	*service.Service
}

// New creates a new instance of the EC2 Metadata service client.
//
// In the general use case the configuration for this service client should not
// be needed and `nil` can be provided. Configuration is only needed if the
// `ec2metadata.Config` defaults need to be overridden. Eg. Setting LogLevel.
//
// @note This configuration will NOT be merged with the default AWS service
// client configuration `defaults.DefaultConfig`. Due to circular dependencies
// with the defaults package and credentials EC2 Role Provider.
func New(config *Config) *Client {
	service := &service.Service{
		ServiceInfo: serviceinfo.ServiceInfo{
			Config:      copyConfig(config),
			ServiceName: "Client",
			Endpoint:    "http://169.254.169.254/latest",
			APIVersion:  "latest",
		},
	}
	service.Initialize()
	service.Handlers.Unmarshal.PushBack(unmarshalHandler)
	service.Handlers.UnmarshalError.PushBack(unmarshalError)
	service.Handlers.Validate.Clear()
	service.Handlers.Validate.PushBack(validateEndpointHandler)

	return &Client{service}
}

func copyConfig(config *Config) *aws.Config {
	if config == nil {
		config = &Config{}
	}
	c := &aws.Config{
		Credentials: credentials.AnonymousCredentials,
		Endpoint:    config.Endpoint,
		HTTPClient:  config.HTTPClient,
		Logger:      config.Logger,
		LogLevel:    config.LogLevel,
		MaxRetries:  config.MaxRetries,
	}

	if c.HTTPClient == nil {
		c.HTTPClient = http.DefaultClient
	}
	if c.Logger == nil {
		c.Logger = aws.NewDefaultLogger()
	}
	if c.LogLevel == nil {
		c.LogLevel = aws.LogLevel(aws.LogOff)
	}
	if c.MaxRetries == nil {
		c.MaxRetries = aws.Int(DefaultRetries)
	}

	return c
}

type metadataOutput struct {
	Content string
}

func unmarshalHandler(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	b, err := ioutil.ReadAll(r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.New("SerializationError", "unable to unmarshal EC2 metadata respose", err)
	}

	data := r.Data.(*metadataOutput)
	data.Content = string(b)
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	_, err := ioutil.ReadAll(r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.New("SerializationError", "unable to unmarshal EC2 metadata error respose", err)
	}

	// TODO extract the error...
}

func validateEndpointHandler(r *request.Request) {
	if r.Service.Endpoint == "" {
		r.Error = aws.ErrMissingEndpoint
	}
}
