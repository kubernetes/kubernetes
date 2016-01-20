// Package ec2metadata provides the client for making API calls to the
// EC2 Metadata service.
package ec2metadata

import (
	"io/ioutil"
	"net"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
)

// ServiceName is the name of the service.
const ServiceName = "ec2metadata"

// A EC2Metadata is an EC2 Metadata service Client.
type EC2Metadata struct {
	*client.Client
}

// New creates a new instance of the EC2Metadata client with a session.
// This client is safe to use across multiple goroutines.
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
func NewClient(cfg aws.Config, handlers request.Handlers, endpoint, signingRegion string, opts ...func(*client.Client)) *EC2Metadata {
	// If the default http client is provided, replace it with a custom
	// client using default timeouts.
	if cfg.HTTPClient == http.DefaultClient {
		cfg.HTTPClient = &http.Client{
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
				Dial: (&net.Dialer{
					// use a shorter timeout than default because the metadata
					// service is local if it is running, and to fail faster
					// if not running on an ec2 instance.
					Timeout:   5 * time.Second,
					KeepAlive: 30 * time.Second,
				}).Dial,
				TLSHandshakeTimeout: 10 * time.Second,
			},
		}
	}

	svc := &EC2Metadata{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName: ServiceName,
				Endpoint:    endpoint,
				APIVersion:  "latest",
			},
			handlers,
		),
	}

	svc.Handlers.Unmarshal.PushBack(unmarshalHandler)
	svc.Handlers.UnmarshalError.PushBack(unmarshalError)
	svc.Handlers.Validate.Clear()
	svc.Handlers.Validate.PushBack(validateEndpointHandler)

	// Add additional options to the service config
	for _, option := range opts {
		option(svc.Client)
	}

	return svc
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
	if r.ClientInfo.Endpoint == "" {
		r.Error = aws.ErrMissingEndpoint
	}
}
