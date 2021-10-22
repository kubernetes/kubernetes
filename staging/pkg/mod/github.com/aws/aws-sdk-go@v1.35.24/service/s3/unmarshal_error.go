package s3

import (
	"bytes"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

type xmlErrorResponse struct {
	XMLName xml.Name `xml:"Error"`
	Code    string   `xml:"Code"`
	Message string   `xml:"Message"`
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	defer io.Copy(ioutil.Discard, r.HTTPResponse.Body)

	// Bucket exists in a different region, and request needs
	// to be made to the correct region.
	if r.HTTPResponse.StatusCode == http.StatusMovedPermanently {
		msg := fmt.Sprintf(
			"incorrect region, the bucket is not in '%s' region at endpoint '%s'",
			aws.StringValue(r.Config.Region),
			aws.StringValue(r.Config.Endpoint),
		)
		if v := r.HTTPResponse.Header.Get("x-amz-bucket-region"); len(v) != 0 {
			msg += fmt.Sprintf(", bucket is in '%s' region", v)
		}
		r.Error = awserr.NewRequestFailure(
			awserr.New("BucketRegionError", msg, nil),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	// Attempt to parse error from body if it is known
	var errResp xmlErrorResponse
	var err error
	if r.HTTPResponse.StatusCode >= 200 && r.HTTPResponse.StatusCode < 300 {
		err = s3unmarshalXMLError(&errResp, r.HTTPResponse.Body)
	} else {
		err = xmlutil.UnmarshalXMLError(&errResp, r.HTTPResponse.Body)
	}

	if err != nil {
		var errorMsg string
		if err == io.EOF {
			errorMsg = "empty response payload"
		} else {
			errorMsg = "failed to unmarshal error message"
		}

		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				errorMsg, err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	// Fallback to status code converted to message if still no error code
	if len(errResp.Code) == 0 {
		statusText := http.StatusText(r.HTTPResponse.StatusCode)
		errResp.Code = strings.Replace(statusText, " ", "", -1)
		errResp.Message = statusText
	}

	r.Error = awserr.NewRequestFailure(
		awserr.New(errResp.Code, errResp.Message, err),
		r.HTTPResponse.StatusCode,
		r.RequestID,
	)
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

// s3unmarshalXMLError is s3 specific xml error unmarshaler
// for 200 OK errors and response payloads.
// This function differs from the xmlUtil.UnmarshalXMLError
// func. It does not ignore the EOF error and passes it up.
// Related to bug fix for `s3 200 OK response with empty payload`
func s3unmarshalXMLError(v interface{}, stream io.Reader) error {
	var errBuf bytes.Buffer
	body := io.TeeReader(stream, &errBuf)

	err := xml.NewDecoder(body).Decode(v)
	if err != nil && err != io.EOF {
		return awserr.NewUnmarshalError(err,
			"failed to unmarshal error message", errBuf.Bytes())
	}

	return err
}
