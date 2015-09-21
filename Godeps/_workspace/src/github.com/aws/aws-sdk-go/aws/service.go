package aws

import (
	"fmt"
	"math"
	"net/http"
	"net/http/httputil"
	"regexp"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/internal/endpoints"
)

// A Service implements the base service request and response handling
// used by all services.
type Service struct {
	Config            *Config
	Handlers          Handlers
	ManualSend        bool
	ServiceName       string
	APIVersion        string
	Endpoint          string
	SigningName       string
	SigningRegion     string
	JSONVersion       string
	TargetPrefix      string
	RetryRules        func(*Request) time.Duration
	ShouldRetry       func(*Request) bool
	DefaultMaxRetries uint
}

var schemeRE = regexp.MustCompile("^([^:]+)://")

// NewService will return a pointer to a new Server object initialized.
func NewService(config *Config) *Service {
	svc := &Service{Config: config}
	svc.Initialize()
	return svc
}

// Initialize initializes the service.
func (s *Service) Initialize() {
	if s.Config == nil {
		s.Config = &Config{}
	}
	if s.Config.HTTPClient == nil {
		s.Config.HTTPClient = http.DefaultClient
	}

	if s.RetryRules == nil {
		s.RetryRules = retryRules
	}

	if s.ShouldRetry == nil {
		s.ShouldRetry = shouldRetry
	}

	s.DefaultMaxRetries = 3
	s.Handlers.Validate.PushBack(ValidateEndpointHandler)
	s.Handlers.Build.PushBack(UserAgentHandler)
	s.Handlers.Sign.PushBack(BuildContentLength)
	s.Handlers.Send.PushBack(SendHandler)
	s.Handlers.AfterRetry.PushBack(AfterRetryHandler)
	s.Handlers.ValidateResponse.PushBack(ValidateResponseHandler)
	s.AddDebugHandlers()
	s.buildEndpoint()

	if !s.Config.DisableParamValidation {
		s.Handlers.Validate.PushBack(ValidateParameters)
	}
}

// buildEndpoint builds the endpoint values the service will use to make requests with.
func (s *Service) buildEndpoint() {
	if s.Config.Endpoint != "" {
		s.Endpoint = s.Config.Endpoint
	} else {
		s.Endpoint, s.SigningRegion =
			endpoints.EndpointForRegion(s.ServiceName, s.Config.Region)
	}

	if s.Endpoint != "" && !schemeRE.MatchString(s.Endpoint) {
		scheme := "https"
		if s.Config.DisableSSL {
			scheme = "http"
		}
		s.Endpoint = scheme + "://" + s.Endpoint
	}
}

// AddDebugHandlers injects debug logging handlers into the service to log request
// debug information.
func (s *Service) AddDebugHandlers() {
	out := s.Config.Logger
	if s.Config.LogLevel == 0 {
		return
	}

	s.Handlers.Send.PushFront(func(r *Request) {
		logBody := r.Config.LogHTTPBody
		dumpedBody, _ := httputil.DumpRequestOut(r.HTTPRequest, logBody)

		fmt.Fprintf(out, "---[ REQUEST POST-SIGN ]-----------------------------\n")
		fmt.Fprintf(out, "%s\n", string(dumpedBody))
		fmt.Fprintf(out, "-----------------------------------------------------\n")
	})
	s.Handlers.Send.PushBack(func(r *Request) {
		fmt.Fprintf(out, "---[ RESPONSE ]--------------------------------------\n")
		if r.HTTPResponse != nil {
			logBody := r.Config.LogHTTPBody
			dumpedBody, _ := httputil.DumpResponse(r.HTTPResponse, logBody)
			fmt.Fprintf(out, "%s\n", string(dumpedBody))
		} else if r.Error != nil {
			fmt.Fprintf(out, "%s\n", r.Error)
		}
		fmt.Fprintf(out, "-----------------------------------------------------\n")
	})
}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API request.
func (s *Service) MaxRetries() uint {
	if s.Config.MaxRetries < 0 {
		return s.DefaultMaxRetries
	}
	return uint(s.Config.MaxRetries)
}

// retryRules returns the delay duration before retrying this request again
func retryRules(r *Request) time.Duration {
	delay := time.Duration(math.Pow(2, float64(r.RetryCount))) * 30
	return delay * time.Millisecond
}

// retryableCodes is a collection of service response codes which are retry-able
// without any further action.
var retryableCodes = map[string]struct{}{
	"RequestError":                           struct{}{},
	"ProvisionedThroughputExceededException": struct{}{},
	"Throttling":                             struct{}{},
}

// credsExpiredCodes is a collection of error codes which signify the credentials
// need to be refreshed. Expired tokens require refreshing of credentials, and
// resigning before the request can be retried.
var credsExpiredCodes = map[string]struct{}{
	"ExpiredToken":          struct{}{},
	"ExpiredTokenException": struct{}{},
	"RequestExpired":        struct{}{}, // EC2 Only
}

func isCodeRetryable(code string) bool {
	if _, ok := retryableCodes[code]; ok {
		return true
	}

	return isCodeExpiredCreds(code)
}

func isCodeExpiredCreds(code string) bool {
	_, ok := credsExpiredCodes[code]
	return ok
}

// shouldRetry returns if the request should be retried.
func shouldRetry(r *Request) bool {
	if r.HTTPResponse.StatusCode >= 500 {
		return true
	}
	if r.Error != nil {
		if err, ok := r.Error.(awserr.Error); ok {
			return isCodeRetryable(err.Code())
		}
	}
	return false
}
