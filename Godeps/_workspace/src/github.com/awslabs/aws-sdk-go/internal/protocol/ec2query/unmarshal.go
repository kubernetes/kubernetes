package ec2query

//go:generate go run ../../fixtures/protocol/generate.go ../../fixtures/protocol/output/ec2.json unmarshal_test.go

import (
	"encoding/xml"
	"io"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/internal/protocol/xml/xmlutil"
)

// Unmarshal unmarshals a response body for the EC2 protocol.
func Unmarshal(r *aws.Request) {
	defer r.HTTPResponse.Body.Close()
	if r.DataFilled() {
		decoder := xml.NewDecoder(r.HTTPResponse.Body)
		err := xmlutil.UnmarshalXML(r.Data, decoder, "")
		if err != nil {
			r.Error = err
			return
		}
	}
}

// UnmarshalMeta unmarshals response headers for the EC2 protocol.
func UnmarshalMeta(r *aws.Request) {
	// TODO implement unmarshaling of request IDs
}

type xmlErrorResponse struct {
	XMLName   xml.Name `xml:"Response"`
	Code      string   `xml:"Errors>Error>Code"`
	Message   string   `xml:"Errors>Error>Message"`
	RequestID string   `xml:"RequestId"`
}

// UnmarshalError unmarshals a response error for the EC2 protocol.
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
