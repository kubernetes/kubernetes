package s3

import (
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

type xmlErrorResponse struct {
	XMLName xml.Name `xml:"Error"`
	Code    string   `xml:"Code"`
	Message string   `xml:"Message"`
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	defer io.Copy(ioutil.Discard, r.HTTPResponse.Body)

	hostID := r.HTTPResponse.Header.Get("X-Amz-Id-2")

	// Bucket exists in a different region, and request needs
	// to be made to the correct region.
	if r.HTTPResponse.StatusCode == http.StatusMovedPermanently {
		r.Error = requestFailure{
			RequestFailure: awserr.NewRequestFailure(
				awserr.New("BucketRegionError",
					fmt.Sprintf("incorrect region, the bucket is not in '%s' region",
						aws.StringValue(r.Config.Region)),
					nil),
				r.HTTPResponse.StatusCode,
				r.RequestID,
			),
			hostID: hostID,
		}
		return
	}

	var errCode, errMsg string

	// Attempt to parse error from body if it is known
	resp := &xmlErrorResponse{}
	err := xml.NewDecoder(r.HTTPResponse.Body).Decode(resp)
	if err != nil && err != io.EOF {
		errCode = "SerializationError"
		errMsg = "failed to decode S3 XML error response"
	} else {
		errCode = resp.Code
		errMsg = resp.Message
		err = nil
	}

	// Fallback to status code converted to message if still no error code
	if len(errCode) == 0 {
		statusText := http.StatusText(r.HTTPResponse.StatusCode)
		errCode = strings.Replace(statusText, " ", "", -1)
		errMsg = statusText
	}

	r.Error = requestFailure{
		RequestFailure: awserr.NewRequestFailure(
			awserr.New(errCode, errMsg, err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		),
		hostID: hostID,
	}
}

// A RequestFailure provides access to the S3 Request ID and Host ID values
// returned from API operation errors. Getting the error as a string will
// return the formated error with the same information as awserr.RequestFailure,
// while also adding the HostID value from the response.
type RequestFailure interface {
	awserr.RequestFailure

	// Host ID is the S3 Host ID needed for debug, and contacting support
	HostID() string
}

type requestFailure struct {
	awserr.RequestFailure

	hostID string
}

func (r requestFailure) Error() string {
	extra := fmt.Sprintf("status code: %d, request id: %s, host id: %s",
		r.StatusCode(), r.RequestID(), r.hostID)
	return awserr.SprintError(r.Code(), r.Message(), extra, r.OrigErr())
}
func (r requestFailure) String() string {
	return r.Error()
}
func (r requestFailure) HostID() string {
	return r.hostID
}
