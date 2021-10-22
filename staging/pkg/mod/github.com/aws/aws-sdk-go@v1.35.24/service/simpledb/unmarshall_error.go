package simpledb

import (
	"encoding/xml"
	"io"
	"io/ioutil"
	"strings"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

type xmlErrorDetail struct {
	Code    string `xml:"Code"`
	Message string `xml:"Message"`
}
type xmlErrorMessage struct {
	XMLName   xml.Name         `xml:"Response"`
	Errors    []xmlErrorDetail `xml:"Errors>Error"`
	RequestID string           `xml:"RequestID"`
}

type xmlErrorResponse struct {
	Code        string
	Message     string
	RequestID   string
	OtherErrors []xmlErrorDetail
}

func (r *xmlErrorResponse) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var errResp xmlErrorMessage
	if err := d.DecodeElement(&errResp, &start); err != nil {
		return err
	}

	r.RequestID = errResp.RequestID
	if len(errResp.Errors) == 0 {
		r.Code = "MissingError"
		r.Message = "missing error code in SimpleDB XML error response"
	} else {
		r.Code = errResp.Errors[0].Code
		r.Message = errResp.Errors[0].Message
		r.OtherErrors = errResp.Errors[1:]
	}

	return nil
}

func unmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()
	defer io.Copy(ioutil.Discard, r.HTTPResponse.Body)

	if r.HTTPResponse.ContentLength == int64(0) {
		// No body, use status code to generate an awserr.Error
		r.Error = awserr.NewRequestFailure(
			awserr.New(strings.Replace(r.HTTPResponse.Status, " ", "", -1), r.HTTPResponse.Status, nil),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	var errResp xmlErrorResponse
	err := xmlutil.UnmarshalXMLError(&errResp, r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization, "failed to unmarshal error message", err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	var otherErrs []error
	for _, e := range errResp.OtherErrors {
		otherErrs = append(otherErrs, awserr.New(e.Code, e.Message, nil))
	}

	// If there are multiple error codes, return only the first as the
	// aws.Error interface only supports one error code.
	r.Error = awserr.NewRequestFailure(
		awserr.NewBatchError(errResp.Code, errResp.Message, otherErrs),
		r.HTTPResponse.StatusCode,
		errResp.RequestID,
	)
}
