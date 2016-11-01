package client

import (
	"fmt"
	"net/http/httputil"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
)

// A Config provides configuration to a service client instance.
type Config struct {
	Config                  *aws.Config
	Handlers                request.Handlers
	Endpoint, SigningRegion string
}

// ConfigProvider provides a generic way for a service client to receive
// the ClientConfig without circular dependencies.
type ConfigProvider interface {
	ClientConfig(serviceName string, cfgs ...*aws.Config) Config
}

// A Client implements the base client request and response handling
// used by all service clients.
type Client struct {
	request.Retryer
	metadata.ClientInfo

	Config   aws.Config
	Handlers request.Handlers
}

// New will return a pointer to a new initialized service client.
func New(cfg aws.Config, info metadata.ClientInfo, handlers request.Handlers, options ...func(*Client)) *Client {
	svc := &Client{
		Config:     cfg,
		ClientInfo: info,
		Handlers:   handlers,
	}

	switch retryer, ok := cfg.Retryer.(request.Retryer); {
	case ok:
		svc.Retryer = retryer
	case cfg.Retryer != nil && cfg.Logger != nil:
		s := fmt.Sprintf("WARNING: %T does not implement request.Retryer; using DefaultRetryer instead", cfg.Retryer)
		cfg.Logger.Log(s)
		fallthrough
	default:
		maxRetries := aws.IntValue(cfg.MaxRetries)
		if cfg.MaxRetries == nil || maxRetries == aws.UseServiceDefaultRetries {
			maxRetries = 3
		}
		svc.Retryer = DefaultRetryer{NumMaxRetries: maxRetries}
	}

	svc.AddDebugHandlers()

	for _, option := range options {
		option(svc)
	}

	return svc
}

// NewRequest returns a new Request pointer for the service API
// operation and parameters.
func (c *Client) NewRequest(operation *request.Operation, params interface{}, data interface{}) *request.Request {
	return request.New(c.Config, c.ClientInfo, c.Handlers, c.Retryer, operation, params, data)
}

// AddDebugHandlers injects debug logging handlers into the service to log request
// debug information.
func (c *Client) AddDebugHandlers() {
	if !c.Config.LogLevel.AtLeast(aws.LogDebug) {
		return
	}

	c.Handlers.Send.PushFront(logRequest)
	c.Handlers.Send.PushBack(logResponse)
}

const logReqMsg = `DEBUG: Request %s/%s Details:
---[ REQUEST POST-SIGN ]-----------------------------
%s
-----------------------------------------------------`

const logReqErrMsg = `DEBUG ERROR: Request %s/%s:
---[ REQUEST DUMP ERROR ]-----------------------------
%s
-----------------------------------------------------`

func logRequest(r *request.Request) {
	logBody := r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
	dumpedBody, err := httputil.DumpRequestOut(r.HTTPRequest, logBody)
	if err != nil {
		r.Config.Logger.Log(fmt.Sprintf(logReqErrMsg, r.ClientInfo.ServiceName, r.Operation.Name, err))
		return
	}

	if logBody {
		// Reset the request body because dumpRequest will re-wrap the r.HTTPRequest's
		// Body as a NoOpCloser and will not be reset after read by the HTTP
		// client reader.
		r.ResetBody()
	}

	r.Config.Logger.Log(fmt.Sprintf(logReqMsg, r.ClientInfo.ServiceName, r.Operation.Name, string(dumpedBody)))
}

const logRespMsg = `DEBUG: Response %s/%s Details:
---[ RESPONSE ]--------------------------------------
%s
-----------------------------------------------------`

const logRespErrMsg = `DEBUG ERROR: Response %s/%s:
---[ RESPONSE DUMP ERROR ]-----------------------------
%s
-----------------------------------------------------`

func logResponse(r *request.Request) {
	var msg = "no response data"
	if r.HTTPResponse != nil {
		logBody := r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
		dumpedBody, err := httputil.DumpResponse(r.HTTPResponse, logBody)
		if err != nil {
			r.Config.Logger.Log(fmt.Sprintf(logRespErrMsg, r.ClientInfo.ServiceName, r.Operation.Name, err))
			return
		}

		msg = string(dumpedBody)
	} else if r.Error != nil {
		msg = r.Error.Error()
	}
	r.Config.Logger.Log(fmt.Sprintf(logRespMsg, r.ClientInfo.ServiceName, r.Operation.Name, msg))
}
