// Package session provides a way to create service clients with shared configuration
// and handlers.
//
// Generally this package should be used instead of the `defaults` package.
//
// A session should be used to share configurations and request handlers between multiple
// service clients. When service clients need specific configuration aws.Config can be
// used to provide additional configuration directly to the service client.
package session

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/defaults"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/endpoints"
)

// A Session provides a central location to create service clients from and
// store configurations and request handlers for those services.
//
// Sessions are safe to create service clients concurrently, but it is not safe
// to mutate the session concurrently.
type Session struct {
	Config   *aws.Config
	Handlers request.Handlers
}

// New creates a new instance of the handlers merging in the provided Configs
// on top of the SDK's default configurations. Once the session is created it
// can be mutated to modify Configs or Handlers. The session is safe to be read
// concurrently, but it should not be written to concurrently.
//
// Example:
//     // Create a session with the default config and request handlers.
//     sess := session.New()
//
//     // Create a session with a custom region
//     sess := session.New(&aws.Config{Region: aws.String("us-east-1")})
//
//     // Create a session, and add additional handlers for all service
//     // clients created with the session to inherit. Adds logging handler.
//     sess := session.New()
//     sess.Handlers.Send.PushFront(func(r *request.Request) {
//          // Log every request made and its payload
//          logger.Println("Request: %s/%s, Payload: %s", r.ClientInfo.ServiceName, r.Operation, r.Params)
//     })
//
//     // Create a S3 client instance from a session
//     sess := session.New()
//     svc := s3.New(sess)
func New(cfgs ...*aws.Config) *Session {
	cfg := defaults.Config()
	handlers := defaults.Handlers()

	// Apply the passed in configs so the configuration can be applied to the
	// default credential chain
	cfg.MergeIn(cfgs...)
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

func initHandlers(s *Session) {
	// Add the Validate parameter handler if it is not disabled.
	s.Handlers.Validate.Remove(corehandlers.ValidateParametersHandler)
	if !aws.BoolValue(s.Config.DisableParamValidation) {
		s.Handlers.Validate.PushBackNamed(corehandlers.ValidateParametersHandler)
	}
}

// Copy creates and returns a copy of the current session, coping the config
// and handlers. If any additional configs are provided they will be merged
// on top of the session's copied config.
//
// Example:
//     // Create a copy of the current session, configured for the us-west-2 region.
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
//
// Example:
//     sess := session.New()
//     s3.New(sess)
func (s *Session) ClientConfig(serviceName string, cfgs ...*aws.Config) client.Config {
	s = s.Copy(cfgs...)
	endpoint, signingRegion := endpoints.NormalizeEndpoint(
		aws.StringValue(s.Config.Endpoint), serviceName,
		aws.StringValue(s.Config.Region), aws.BoolValue(s.Config.DisableSSL))

	return client.Config{
		Config:        s.Config,
		Handlers:      s.Handlers,
		Endpoint:      endpoint,
		SigningRegion: signingRegion,
	}
}
