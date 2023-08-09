package protocol

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// RequireHTTPMinProtocol request handler is used to enforce that
// the target endpoint supports the given major and minor HTTP protocol version.
type RequireHTTPMinProtocol struct {
	Major, Minor int
}

// Handler will mark the request.Request with an error if the
// target endpoint did not connect with the required HTTP protocol
// major and minor version.
func (p RequireHTTPMinProtocol) Handler(r *request.Request) {
	if r.Error != nil || r.HTTPResponse == nil {
		return
	}

	if !strings.HasPrefix(r.HTTPResponse.Proto, "HTTP") {
		r.Error = newMinHTTPProtoError(p.Major, p.Minor, r)
	}

	if r.HTTPResponse.ProtoMajor < p.Major || r.HTTPResponse.ProtoMinor < p.Minor {
		r.Error = newMinHTTPProtoError(p.Major, p.Minor, r)
	}
}

// ErrCodeMinimumHTTPProtocolError error code is returned when the target endpoint
// did not match the required HTTP major and minor protocol version.
const ErrCodeMinimumHTTPProtocolError = "MinimumHTTPProtocolError"

func newMinHTTPProtoError(major, minor int, r *request.Request) error {
	return awserr.NewRequestFailure(
		awserr.New("MinimumHTTPProtocolError",
			fmt.Sprintf(
				"operation requires minimum HTTP protocol of HTTP/%d.%d, but was %s",
				major, minor, r.HTTPResponse.Proto,
			),
			nil,
		),
		r.HTTPResponse.StatusCode, r.RequestID,
	)
}
