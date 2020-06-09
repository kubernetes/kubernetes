// Package ec2metadata provides the client for making API calls to the
// EC2 Metadata service.
//
// This package's client can be disabled completely by setting the environment
// variable "AWS_EC2_METADATA_DISABLED=true". This environment variable set to
// true instructs the SDK to disable the EC2 Metadata client. The client cannot
// be used while the environment variable is set to true, (case insensitive).
package ec2metadata

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
)

const (
	// ServiceName is the name of the service.
	ServiceName          = "ec2metadata"
	disableServiceEnvVar = "AWS_EC2_METADATA_DISABLED"

	// Headers for Token and TTL
	ttlHeader   = "x-aws-ec2-metadata-token-ttl-seconds"
	tokenHeader = "x-aws-ec2-metadata-token"

	// Named Handler constants
	fetchTokenHandlerName          = "FetchTokenHandler"
	unmarshalMetadataHandlerName   = "unmarshalMetadataHandler"
	unmarshalTokenHandlerName      = "unmarshalTokenHandler"
	enableTokenProviderHandlerName = "enableTokenProviderHandler"

	// TTL constants
	defaultTTL          = 21600 * time.Second
	ttlExpirationWindow = 30 * time.Second
)

// A EC2Metadata is an EC2 Metadata service Client.
type EC2Metadata struct {
	*client.Client
}

// New creates a new instance of the EC2Metadata client with a session.
// This client is safe to use across multiple goroutines.
//
//
// Example:
//     // Create a EC2Metadata client from just a session.
//     svc := ec2metadata.New(mySession)
//
//     // Create a EC2Metadata client with additional configuration
//     svc := ec2metadata.New(mySession, aws.NewConfig().WithLogLevel(aws.LogDebugHTTPBody))
func New(p client.ConfigProvider, cfgs ...*aws.Config) *EC2Metadata {
	c := p.ClientConfig(ServiceName, cfgs...)
	return NewClient(*c.Config, c.Handlers, c.Endpoint, c.SigningRegion)
}

// NewClient returns a new EC2Metadata client. Should be used to create
// a client when not using a session. Generally using just New with a session
// is preferred.
//
// If an unmodified HTTP client is provided from the stdlib default, or no client
// the EC2RoleProvider's EC2Metadata HTTP client's timeout will be shortened.
// To disable this set Config.EC2MetadataDisableTimeoutOverride to false. Enabled by default.
func NewClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string, opts ...func(*client.Client)) *EC2Metadata {
	if !aws.BoolValue(cfg.EC2MetadataDisableTimeoutOverride) && httpClientZero(cfg.HTTPClient) {
		// If the http client is unmodified and this feature is not disabled
		// set custom timeouts for EC2Metadata requests.
		cfg.HTTPClient = &http.Client{
			// use a shorter timeout than default because the metadata
			// service is local if it is running, and to fail faster
			// if not running on an ec2 instance.
			Timeout: 1 * time.Second,
		}
		// max number of retries on the client operation
		cfg.MaxRetries = aws.Int(2)
	}

	svc := &EC2Metadata{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName: ServiceName,
				ServiceID:   ServiceName,
				Endpoint:    endpoint,
				APIVersion:  "latest",
			},
			handlers,
		),
	}

	// token provider instance
	tp := newTokenProvider(svc, defaultTTL)

	// NamedHandler for fetching token
	svc.Handlers.Sign.PushBackNamed(request.NamedHandler{
		Name: fetchTokenHandlerName,
		Fn:   tp.fetchTokenHandler,
	})
	// NamedHandler for enabling token provider
	svc.Handlers.Complete.PushBackNamed(request.NamedHandler{
		Name: enableTokenProviderHandlerName,
		Fn:   tp.enableTokenProviderHandler,
	})

	svc.Handlers.Unmarshal.PushBackNamed(unmarshalHandler)
	svc.Handlers.UnmarshalError.PushBack(unmarshalError)
	svc.Handlers.Validate.Clear()
	svc.Handlers.Validate.PushBack(validateEndpointHandler)

	// Disable the EC2 Metadata service if the environment variable is set.
	// This short-circuits the service's functionality to always fail to send
	// requests.
	if strings.ToLower(os.Getenv(disableServiceEnvVar)) == "true" {
		svc.Handlers.Send.SwapNamed(request.NamedHandler{
			Name: corehandlers.SendHandler.Name,
			Fn: func(r *request.Request) {
				r.HTTPResponse = &http.Response{
					Header: http.Header{},
				}
				r.Error = awserr.New(
					request.CanceledErrorCode,
					"EC2 IMDS access disabled via "+disableServiceEnvVar+" env var",
					nil)
			},
		})
	}

	// Add additional options to the service config
	for _, option := range opts {
		option(svc.Client)
	}
	return svc
}

func httpClientZero(c *http.Client) bool {
	return c == nil || (c.Transport == nil && c.CheckRedirect == nil && c.Jar == nil && c.Timeout == 0)
}

type metadataOutput struct {
	Content string
}

type tokenOutput struct {
	Token string
	TTL   time.Duration
}

// unmarshal token handler is used to parse the response of a getToken operation
var unmarshalTokenHandler = request.NamedHandler{
	Name: unmarshalTokenHandlerName,
	Fn: func(r *request.Request) {
		defer r.HTTPResponse.Body.Close()
		var b bytes.Buffer
		if _, err := io.Copy(&b, r.HTTPResponse.Body); err != nil {
			r.Error = awserr.NewRequestFailure(awserr.New(request.ErrCodeSerialization,
				"unable to unmarshal EC2 metadata response", err), r.HTTPResponse.StatusCode, r.RequestID)
			return
		}

		v := r.HTTPResponse.Header.Get(ttlHeader)
		data, ok := r.Data.(*tokenOutput)
		if !ok {
			return
		}

		data.Token = b.String()
		// TTL is in seconds
		i, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			r.Error = awserr.NewRequestFailure(awserr.New(request.ParamFormatErrCode,
				"unable to parse EC2 token TTL response", err), r.HTTPResponse.StatusCode, r.RequestID)
			return
		}
		t := time.Duration(i) * time.Second
		data.TTL = t
	},
}

var unmarshalHandler = request.NamedHandler{
	Name: unmarshalMetadataHandlerName,
	Fn: func(r *request.Request) {
		defer r.HTTPResponse.Body.Close()
		var b bytes.Buffer
		if _, err := io.Copy(&b, r.HTTPResponse.Body); err != nil {
			r.Error = awserr.NewRequestFailure(awserr.New(request.ErrCodeSerialization,
				"unable to unmarshal EC2 metadata response", err), r.HTTPResponse.StatusCode, r.RequestID)
			return
		}

		if data, ok := r.Data.(*metadataOutput); ok {
			data.Content = b.String()
		}
	},
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	var b bytes.Buffer

	if _, err := io.Copy(&b, r.HTTPResponse.Body); err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization, "unable to unmarshal EC2 metadata error response", err),
			r.HTTPResponse.StatusCode, r.RequestID)
		return
	}

	// Response body format is not consistent between metadata endpoints.
	// Grab the error message as a string and include that as the source error
	r.Error = awserr.NewRequestFailure(awserr.New("EC2MetadataError", "failed to make EC2Metadata request", errors.New(b.String())),
		r.HTTPResponse.StatusCode, r.RequestID)
}

func validateEndpointHandler(r *request.Request) {
	if r.ClientInfo.Endpoint == "" {
		r.Error = aws.ErrMissingEndpoint
	}
}
