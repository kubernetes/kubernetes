package simpledb

import (
	"encoding/xml"
	"io"
	"io/ioutil"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

type xmlErrorDetail struct {
	Code    string `xml:"Code"`
	Message string `xml:"Message"`
}

type xmlErrorResponse struct {
	XMLName   xml.Name         `xml:"Response"`
	Errors    []xmlErrorDetail `xml:"Errors>Error"`
	RequestID string           `xml:"RequestID"`
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	defer io.Copy(ioutil.Discard, r.HTTPResponse.Body)

	if r.HTTPResponse.ContentLength == int64(0) {
		// No body, use status code to generate an awserr.Error
		r.Error = awserr.NewRequestFailure(
			awserr.New(strings.Replace(r.HTTPResponse.Status, " ", "", -1), r.HTTPResponse.Status, nil),
			r.HTTPResponse.StatusCode,
			"",
		)
		return
	}

	resp := &xmlErrorResponse{}
	err := xml.NewDecoder(r.HTTPResponse.Body).Decode(resp)
	if err != nil && err != io.EOF {
		r.Error = awserr.New("SerializationError", "failed to decode SimpleDB XML error response", nil)
	} else if len(resp.Errors) == 0 {
		r.Error = awserr.New("MissingError", "missing error code in SimpleDB XML error response", nil)
	} else {
		// If there are multiple error codes, return only the first as the aws.Error interface only supports
		// one error code.
		r.Error = awserr.NewRequestFailure(
			awserr.New(resp.Errors[0].Code, resp.Errors[0].Message, nil),
			r.HTTPResponse.StatusCode,
			resp.RequestID,
		)
	}
}
