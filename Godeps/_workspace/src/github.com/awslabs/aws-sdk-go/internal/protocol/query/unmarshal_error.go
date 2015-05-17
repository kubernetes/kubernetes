package query

import (
	"encoding/xml"
	"io"

	"github.com/awslabs/aws-sdk-go/aws"
)

type xmlErrorResponse struct {
	XMLName   xml.Name `xml:"ErrorResponse"`
	Code      string   `xml:"Error>Code"`
	Message   string   `xml:"Error>Message"`
	RequestID string   `xml:"RequestId"`
}

// UnmarshalError unmarshals an error response for an AWS Query service.
func UnmarshalError(r *aws.Request) {
	defer r.HTTPResponse.Body.Close()

	resp := &xmlErrorResponse{}
	err := xml.NewDecoder(r.HTTPResponse.Body).Decode(resp)
	if err != nil && err != io.EOF {
		r.Error = err
	} else {
		r.Error = aws.APIError{
			StatusCode: r.HTTPResponse.StatusCode,
			Code:       resp.Code,
			Message:    resp.Message,
		}
	}
}
