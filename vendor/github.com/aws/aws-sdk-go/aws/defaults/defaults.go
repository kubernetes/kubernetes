// Package defaults is a collection of helpers to retrieve the SDK's default
// configuration and handlers.
//
// Generally this package shouldn't be used directly, but session.Session
// instead. This package is useful when you need to reset the defaults
// of a session or service client to the SDK defaults before setting
// additional parameters.
package defaults

import (
	"net/http"
	"os"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/endpoints"
)

// A Defaults provides a collection of default values for SDK clients.
type Defaults struct {
	Config   *aws.Config
	Handlers request.Handlers
}

// Get returns the SDK's default values with Config and handlers pre-configured.
func Get() Defaults {
	cfg := Config()
	handlers := Handlers()
	cfg.Credentials = CredChain(cfg, handlers)

	return Defaults{
		Config:   cfg,
		Handlers: handlers,
	}
}

// Config returns the default configuration without credentials.
// To retrieve a config with credentials also included use
// `defaults.Get().Config` instead.
//
// Generally you shouldn't need to use this method directly, but
// is available if you need to reset the configuration of an
// existing service client or session.
func Config() *aws.Config {
	return aws.NewConfig().
		WithCredentials(credentials.AnonymousCredentials).
		WithRegion(os.Getenv("AWS_REGION")).
		WithHTTPClient(http.DefaultClient).
		WithMaxRetries(aws.UseServiceDefaultRetries).
		WithLogger(aws.NewDefaultLogger()).
		WithLogLevel(aws.LogOff).
		WithSleepDelay(time.Sleep)
}

// Handlers returns the default request handlers.
//
// Generally you shouldn't need to use this method directly, but
// is available if you need to reset the request handlers of an
// existing service client or session.
func Handlers() request.Handlers {
	var handlers request.Handlers

	handlers.Validate.PushBackNamed(corehandlers.ValidateEndpointHandler)
	handlers.Build.PushBackNamed(corehandlers.SDKVersionUserAgentHandler)
	handlers.Build.AfterEachFn = request.HandlerListStopOnError
	handlers.Sign.PushBackNamed(corehandlers.BuildContentLengthHandler)
	handlers.Send.PushBackNamed(corehandlers.SendHandler)
	handlers.AfterRetry.PushBackNamed(corehandlers.AfterRetryHandler)
	handlers.ValidateResponse.PushBackNamed(corehandlers.ValidateResponseHandler)

	return handlers
}

// CredChain returns the default credential chain.
//
// Generally you shouldn't need to use this method directly, but
// is available if you need to reset the credentials of an
// existing service client or session's Config.
func CredChain(cfg *aws.Config, handlers request.Handlers) *credentials.Credentials {
	endpoint, signingRegion := endpoints.EndpointForRegion(ec2metadata.ServiceName, *cfg.Region, true)

	return credentials.NewCredentials(&credentials.ChainProvider{
		VerboseErrors: aws.BoolValue(cfg.CredentialsChainVerboseErrors),
		Providers: []credentials.Provider{
			&credentials.EnvProvider{},
			&credentials.SharedCredentialsProvider{Filename: "", Profile: ""},
			&ec2rolecreds.EC2RoleProvider{
				Client:       ec2metadata.NewClient(*cfg, handlers, endpoint, signingRegion),
				ExpiryWindow: 5 * time.Minute,
			},
		}})
}
