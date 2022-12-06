package client

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http/httputil"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

const logReqMsg = `DEBUG: Request %s/%s Details:
---[ REQUEST POST-SIGN ]-----------------------------
%s
-----------------------------------------------------`

const logReqErrMsg = `DEBUG ERROR: Request %s/%s:
---[ REQUEST DUMP ERROR ]-----------------------------
%s
------------------------------------------------------`

type logWriter struct {
	// Logger is what we will use to log the payload of a response.
	Logger aws.Logger
	// buf stores the contents of what has been read
	buf *bytes.Buffer
}

func (logger *logWriter) Write(b []byte) (int, error) {
	return logger.buf.Write(b)
}

type teeReaderCloser struct {
	// io.Reader will be a tee reader that is used during logging.
	// This structure will read from a body and write the contents to a logger.
	io.Reader
	// Source is used just to close when we are done reading.
	Source io.ReadCloser
}

func (reader *teeReaderCloser) Close() error {
	return reader.Source.Close()
}

// LogHTTPRequestHandler is a SDK request handler to log the HTTP request sent
// to a service. Will include the HTTP request body if the LogLevel of the
// request matches LogDebugWithHTTPBody.
var LogHTTPRequestHandler = request.NamedHandler{
	Name: "awssdk.client.LogRequest",
	Fn:   logRequest,
}

func logRequest(r *request.Request) {
	if !r.Config.LogLevel.AtLeast(aws.LogDebug) || r.Config.Logger == nil {
		return
	}

	logBody := r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
	bodySeekable := aws.IsReaderSeekable(r.Body)

	b, err := httputil.DumpRequestOut(r.HTTPRequest, logBody)
	if err != nil {
		r.Config.Logger.Log(fmt.Sprintf(logReqErrMsg,
			r.ClientInfo.ServiceName, r.Operation.Name, err))
		return
	}

	if logBody {
		if !bodySeekable {
			r.SetReaderBody(aws.ReadSeekCloser(r.HTTPRequest.Body))
		}
		// Reset the request body because dumpRequest will re-wrap the
		// r.HTTPRequest's Body as a NoOpCloser and will not be reset after
		// read by the HTTP client reader.
		if err := r.Error; err != nil {
			r.Config.Logger.Log(fmt.Sprintf(logReqErrMsg,
				r.ClientInfo.ServiceName, r.Operation.Name, err))
			return
		}
	}

	r.Config.Logger.Log(fmt.Sprintf(logReqMsg,
		r.ClientInfo.ServiceName, r.Operation.Name, string(b)))
}

// LogHTTPRequestHeaderHandler is a SDK request handler to log the HTTP request sent
// to a service. Will only log the HTTP request's headers. The request payload
// will not be read.
var LogHTTPRequestHeaderHandler = request.NamedHandler{
	Name: "awssdk.client.LogRequestHeader",
	Fn:   logRequestHeader,
}

func logRequestHeader(r *request.Request) {
	if !r.Config.LogLevel.AtLeast(aws.LogDebug) || r.Config.Logger == nil {
		return
	}

	b, err := httputil.DumpRequestOut(r.HTTPRequest, false)
	if err != nil {
		r.Config.Logger.Log(fmt.Sprintf(logReqErrMsg,
			r.ClientInfo.ServiceName, r.Operation.Name, err))
		return
	}

	r.Config.Logger.Log(fmt.Sprintf(logReqMsg,
		r.ClientInfo.ServiceName, r.Operation.Name, string(b)))
}

const logRespMsg = `DEBUG: Response %s/%s Details:
---[ RESPONSE ]--------------------------------------
%s
-----------------------------------------------------`

const logRespErrMsg = `DEBUG ERROR: Response %s/%s:
---[ RESPONSE DUMP ERROR ]-----------------------------
%s
-----------------------------------------------------`

// LogHTTPResponseHandler is a SDK request handler to log the HTTP response
// received from a service. Will include the HTTP response body if the LogLevel
// of the request matches LogDebugWithHTTPBody.
var LogHTTPResponseHandler = request.NamedHandler{
	Name: "awssdk.client.LogResponse",
	Fn:   logResponse,
}

func logResponse(r *request.Request) {
	if !r.Config.LogLevel.AtLeast(aws.LogDebug) || r.Config.Logger == nil {
		return
	}

	lw := &logWriter{r.Config.Logger, bytes.NewBuffer(nil)}

	if r.HTTPResponse == nil {
		lw.Logger.Log(fmt.Sprintf(logRespErrMsg,
			r.ClientInfo.ServiceName, r.Operation.Name, "request's HTTPResponse is nil"))
		return
	}

	logBody := r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
	if logBody {
		r.HTTPResponse.Body = &teeReaderCloser{
			Reader: io.TeeReader(r.HTTPResponse.Body, lw),
			Source: r.HTTPResponse.Body,
		}
	}

	handlerFn := func(req *request.Request) {
		b, err := httputil.DumpResponse(req.HTTPResponse, false)
		if err != nil {
			lw.Logger.Log(fmt.Sprintf(logRespErrMsg,
				req.ClientInfo.ServiceName, req.Operation.Name, err))
			return
		}

		lw.Logger.Log(fmt.Sprintf(logRespMsg,
			req.ClientInfo.ServiceName, req.Operation.Name, string(b)))

		if logBody {
			b, err := ioutil.ReadAll(lw.buf)
			if err != nil {
				lw.Logger.Log(fmt.Sprintf(logRespErrMsg,
					req.ClientInfo.ServiceName, req.Operation.Name, err))
				return
			}

			lw.Logger.Log(string(b))
		}
	}

	const handlerName = "awsdk.client.LogResponse.ResponseBody"

	r.Handlers.Unmarshal.SetBackNamed(request.NamedHandler{
		Name: handlerName, Fn: handlerFn,
	})
	r.Handlers.UnmarshalError.SetBackNamed(request.NamedHandler{
		Name: handlerName, Fn: handlerFn,
	})
}

// LogHTTPResponseHeaderHandler is a SDK request handler to log the HTTP
// response received from a service. Will only log the HTTP response's headers.
// The response payload will not be read.
var LogHTTPResponseHeaderHandler = request.NamedHandler{
	Name: "awssdk.client.LogResponseHeader",
	Fn:   logResponseHeader,
}

func logResponseHeader(r *request.Request) {
	if !r.Config.LogLevel.AtLeast(aws.LogDebug) || r.Config.Logger == nil {
		return
	}

	b, err := httputil.DumpResponse(r.HTTPResponse, false)
	if err != nil {
		r.Config.Logger.Log(fmt.Sprintf(logRespErrMsg,
			r.ClientInfo.ServiceName, r.Operation.Name, err))
		return
	}

	r.Config.Logger.Log(fmt.Sprintf(logRespMsg,
		r.ClientInfo.ServiceName, r.Operation.Name, string(b)))
}
