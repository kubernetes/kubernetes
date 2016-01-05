package service

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httputil"
	"regexp"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/service/serviceinfo"
	"github.com/aws/aws-sdk-go/private/endpoints"
)

// A Service implements the base service request and response handling
// used by all services.
type Service struct {
	serviceinfo.ServiceInfo
	request.Retryer
	DefaultMaxRetries uint
	Handlers          request.Handlers
}

var schemeRE = regexp.MustCompile("^([^:]+)://")

// New will return a pointer to a new Server object initialized.
func New(config *aws.Config) *Service {
	svc := &Service{ServiceInfo: serviceinfo.ServiceInfo{Config: config}}
	svc.Initialize()
	return svc
}

// Initialize initializes the service.
func (s *Service) Initialize() {
	if s.Config == nil {
		s.Config = &aws.Config{}
	}
	if s.Config.HTTPClient == nil {
		s.Config.HTTPClient = http.DefaultClient
	}
	if s.Config.SleepDelay == nil {
		s.Config.SleepDelay = time.Sleep
	}

	s.Retryer = DefaultRetryer{s}
	s.DefaultMaxRetries = 3
	s.Handlers.Validate.PushBackNamed(corehandlers.ValidateEndpointHandler)
	s.Handlers.Build.PushBackNamed(corehandlers.UserAgentHandler)
	s.Handlers.Sign.PushBackNamed(corehandlers.BuildContentLengthHandler)
	s.Handlers.Send.PushBackNamed(corehandlers.SendHandler)
	s.Handlers.AfterRetry.PushBackNamed(corehandlers.AfterRetryHandler)
	s.Handlers.ValidateResponse.PushBackNamed(corehandlers.ValidateResponseHandler)
	if !aws.BoolValue(s.Config.DisableParamValidation) {
		s.Handlers.Validate.PushBackNamed(corehandlers.ValidateParametersHandler)
	}
	s.AddDebugHandlers()
	s.buildEndpoint()
}

// NewRequest returns a new Request pointer for the service API
// operation and parameters.
func (s *Service) NewRequest(operation *request.Operation, params interface{}, data interface{}) *request.Request {
	return request.New(s.ServiceInfo, s.Handlers, s.Retryer, operation, params, data)
}

// buildEndpoint builds the endpoint values the service will use to make requests with.
func (s *Service) buildEndpoint() {
	if aws.StringValue(s.Config.Endpoint) != "" {
		s.Endpoint = *s.Config.Endpoint
	} else if s.Endpoint == "" {
		s.Endpoint, s.SigningRegion =
			endpoints.EndpointForRegion(s.ServiceName, aws.StringValue(s.Config.Region))
	}

	if s.Endpoint != "" && !schemeRE.MatchString(s.Endpoint) {
		scheme := "https"
		if aws.BoolValue(s.Config.DisableSSL) {
			scheme = "http"
		}
		s.Endpoint = scheme + "://" + s.Endpoint
	}
}

// AddDebugHandlers injects debug logging handlers into the service to log request
// debug information.
func (s *Service) AddDebugHandlers() {
	if !s.Config.LogLevel.AtLeast(aws.LogDebug) {
		return
	}

	s.Handlers.Send.PushFront(logRequest)
	s.Handlers.Send.PushBack(logResponse)
}

const logReqMsg = `DEBUG: Request %s/%s Details:
---[ REQUEST POST-SIGN ]-----------------------------
%s
-----------------------------------------------------`

func logRequest(r *request.Request) {
	logBody := r.Service.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
	dumpedBody, _ := httputil.DumpRequestOut(r.HTTPRequest, logBody)

	if logBody {
		// Reset the request body because dumpRequest will re-wrap the r.HTTPRequest's
		// Body as a NoOpCloser and will not be reset after read by the HTTP
		// client reader.
		r.Body.Seek(r.BodyStart, 0)
		r.HTTPRequest.Body = ioutil.NopCloser(r.Body)
	}

	r.Service.Config.Logger.Log(fmt.Sprintf(logReqMsg, r.Service.ServiceName, r.Operation.Name, string(dumpedBody)))
}

const logRespMsg = `DEBUG: Response %s/%s Details:
---[ RESPONSE ]--------------------------------------
%s
-----------------------------------------------------`

func logResponse(r *request.Request) {
	var msg = "no reponse data"
	if r.HTTPResponse != nil {
		logBody := r.Service.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
		dumpedBody, _ := httputil.DumpResponse(r.HTTPResponse, logBody)
		msg = string(dumpedBody)
	} else if r.Error != nil {
		msg = r.Error.Error()
	}
	r.Service.Config.Logger.Log(fmt.Sprintf(logRespMsg, r.Service.ServiceName, r.Operation.Name, msg))
}
