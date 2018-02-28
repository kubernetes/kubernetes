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

func logRequest(r *request.Request) {
	logBody := r.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody)
	bodySeekable := aws.IsReaderSeekable(r.Body)
	dumpedBody, err := httputil.DumpRequestOut(r.HTTPRequest, logBody)
	if err != nil {
		r.Config.Logger.Log(fmt.Sprintf(logReqErrMsg, r.ClientInfo.ServiceName, r.Operation.Name, err))
		return
	}

	if logBody {
		if !bodySeekable {
			r.SetReaderBody(aws.ReadSeekCloser(r.HTTPRequest.Body))
		}
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
	lw := &logWriter{r.Config.Logger, bytes.NewBuffer(nil)}
	r.HTTPResponse.Body = &teeReaderCloser{
		Reader: io.TeeReader(r.HTTPResponse.Body, lw),
		Source: r.HTTPResponse.Body,
	}

	handlerFn := func(req *request.Request) {
		body, err := httputil.DumpResponse(req.HTTPResponse, false)
		if err != nil {
			lw.Logger.Log(fmt.Sprintf(logRespErrMsg, req.ClientInfo.ServiceName, req.Operation.Name, err))
			return
		}

		b, err := ioutil.ReadAll(lw.buf)
		if err != nil {
			lw.Logger.Log(fmt.Sprintf(logRespErrMsg, req.ClientInfo.ServiceName, req.Operation.Name, err))
			return
		}
		lw.Logger.Log(fmt.Sprintf(logRespMsg, req.ClientInfo.ServiceName, req.Operation.Name, string(body)))
		if req.Config.LogLevel.Matches(aws.LogDebugWithHTTPBody) {
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
