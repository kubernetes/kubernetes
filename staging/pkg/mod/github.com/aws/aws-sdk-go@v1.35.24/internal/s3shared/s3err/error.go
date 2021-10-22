package s3err

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// RequestFailure provides additional S3 specific metadata for the request
// failure.
type RequestFailure struct {
	awserr.RequestFailure

	hostID string
}

// NewRequestFailure returns a request failure error decordated with S3
// specific metadata.
func NewRequestFailure(err awserr.RequestFailure, hostID string) *RequestFailure {
	return &RequestFailure{RequestFailure: err, hostID: hostID}
}

func (r RequestFailure) Error() string {
	extra := fmt.Sprintf("status code: %d, request id: %s, host id: %s",
		r.StatusCode(), r.RequestID(), r.hostID)
	return awserr.SprintError(r.Code(), r.Message(), extra, r.OrigErr())
}
func (r RequestFailure) String() string {
	return r.Error()
}

// HostID returns the HostID request response value.
func (r RequestFailure) HostID() string {
	return r.hostID
}

// RequestFailureWrapperHandler returns a handler to rap an
// awserr.RequestFailure with the  S3 request ID 2 from the response.
func RequestFailureWrapperHandler() request.NamedHandler {
	return request.NamedHandler{
		Name: "awssdk.s3.errorHandler",
		Fn: func(req *request.Request) {
			reqErr, ok := req.Error.(awserr.RequestFailure)
			if !ok || reqErr == nil {
				return
			}

			hostID := req.HTTPResponse.Header.Get("X-Amz-Id-2")
			if req.Error == nil {
				return
			}

			req.Error = NewRequestFailure(reqErr, hostID)
		},
	}
}
