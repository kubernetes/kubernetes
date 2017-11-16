// Package endpointcreds provides support for retrieving credentials from an
// arbitrary HTTP endpoint.
//
// The credentials endpoint Provider can receive both static and refreshable
// credentials that will expire. Credentials are static when an "Expiration"
// value is not provided in the endpoint's response.
//
// Static credentials will never expire once they have been retrieved. The format
// of the static credentials response:
//    {
//        "AccessKeyId" : "MUA...",
//        "SecretAccessKey" : "/7PC5om....",
//    }
//
// Refreshable credentials will expire within the "ExpiryWindow" of the Expiration
// value in the response. The format of the refreshable credentials response:
//    {
//        "AccessKeyId" : "MUA...",
//        "SecretAccessKey" : "/7PC5om....",
//        "Token" : "AQoDY....=",
//        "Expiration" : "2016-02-25T06:03:31Z"
//    }
//
// Errors should be returned in the following format and only returned with 400
// or 500 HTTP status codes.
//    {
//        "code": "ErrorCode",
//        "message": "Helpful error message."
//    }
package endpointcreds

import (
	"encoding/json"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
)

// ProviderName is the name of the credentials provider.
const ProviderName = `CredentialsEndpointProvider`

// Provider satisfies the credentials.Provider interface, and is a client to
// retrieve credentials from an arbitrary endpoint.
type Provider struct {
	staticCreds bool
	credentials.Expiry

	// Requires a AWS Client to make HTTP requests to the endpoint with.
	// the Endpoint the request will be made to is provided by the aws.Config's
	// Endpoint value.
	Client *client.Client

	// ExpiryWindow will allow the credentials to trigger refreshing prior to
	// the credentials actually expiring. This is beneficial so race conditions
	// with expiring credentials do not cause request to fail unexpectedly
	// due to ExpiredTokenException exceptions.
	//
	// So a ExpiryWindow of 10s would cause calls to IsExpired() to return true
	// 10 seconds before the credentials are actually expired.
	//
	// If ExpiryWindow is 0 or less it will be ignored.
	ExpiryWindow time.Duration
}

// NewProviderClient returns a credentials Provider for retrieving AWS credentials
// from arbitrary endpoint.
func NewProviderClient(cfg aws.Config, handlers request.Handlers, endpoint string, options ...func(*Provider)) credentials.Provider {
	p := &Provider{
		Client: client.New(
			cfg,
			metadata.ClientInfo{
				ServiceName: "CredentialsEndpoint",
				Endpoint:    endpoint,
			},
			handlers,
		),
	}

	p.Client.Handlers.Unmarshal.PushBack(unmarshalHandler)
	p.Client.Handlers.UnmarshalError.PushBack(unmarshalError)
	p.Client.Handlers.Validate.Clear()
	p.Client.Handlers.Validate.PushBack(validateEndpointHandler)

	for _, option := range options {
		option(p)
	}

	return p
}

// NewCredentialsClient returns a Credentials wrapper for retrieving credentials
// from an arbitrary endpoint concurrently. The client will request the
func NewCredentialsClient(cfg aws.Config, handlers request.Handlers, endpoint string, options ...func(*Provider)) *credentials.Credentials {
	return credentials.NewCredentials(NewProviderClient(cfg, handlers, endpoint, options...))
}

// IsExpired returns true if the credentials retrieved are expired, or not yet
// retrieved.
func (p *Provider) IsExpired() bool {
	if p.staticCreds {
		return false
	}
	return p.Expiry.IsExpired()
}

// Retrieve will attempt to request the credentials from the endpoint the Provider
// was configured for. And error will be returned if the retrieval fails.
func (p *Provider) Retrieve() (credentials.Value, error) {
	resp, err := p.getCredentials()
	if err != nil {
		return credentials.Value{ProviderName: ProviderName},
			awserr.New("CredentialsEndpointError", "failed to load credentials", err)
	}

	if resp.Expiration != nil {
		p.SetExpiration(*resp.Expiration, p.ExpiryWindow)
	} else {
		p.staticCreds = true
	}

	return credentials.Value{
		AccessKeyID:     resp.AccessKeyID,
		SecretAccessKey: resp.SecretAccessKey,
		SessionToken:    resp.Token,
		ProviderName:    ProviderName,
	}, nil
}

type getCredentialsOutput struct {
	Expiration      *time.Time
	AccessKeyID     string
	SecretAccessKey string
	Token           string
}

type errorOutput struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (p *Provider) getCredentials() (*getCredentialsOutput, error) {
	op := &request.Operation{
		Name:       "GetCredentials",
		HTTPMethod: "GET",
	}

	out := &getCredentialsOutput{}
	req := p.Client.NewRequest(op, nil, out)
	req.HTTPRequest.Header.Set("Accept", "application/json")

	return out, req.Send()
}

func validateEndpointHandler(r *request.Request) {
	if len(r.ClientInfo.Endpoint) == 0 {
		r.Error = aws.ErrMissingEndpoint
	}
}

func unmarshalHandler(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	out := r.Data.(*getCredentialsOutput)
	if err := json.NewDecoder(r.HTTPResponse.Body).Decode(&out); err != nil {
		r.Error = awserr.New("SerializationError",
			"failed to decode endpoint credentials",
			err,
		)
	}
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	var errOut errorOutput
	if err := json.NewDecoder(r.HTTPResponse.Body).Decode(&errOut); err != nil {
		r.Error = awserr.New("SerializationError",
			"failed to decode endpoint credentials",
			err,
		)
	}

	// Response body format is not consistent between metadata endpoints.
	// Grab the error message as a string and include that as the source error
	r.Error = awserr.New(errOut.Code, errOut.Message, nil)
}
