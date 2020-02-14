package query

import (
	"encoding/xml"
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

// UnmarshalErrorHandler is a name request handler to unmarshal request errors
var UnmarshalErrorHandler = request.NamedHandler{Name: "awssdk.query.UnmarshalError", Fn: UnmarshalError}

type xmlErrorResponse struct {
	Code      string `xml:"Error>Code"`
	Message   string `xml:"Error>Message"`
	RequestID string `xml:"RequestId"`
}

type xmlResponseError struct {
	xmlErrorResponse
}

func (e *xmlResponseError) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	const svcUnavailableTagName = "ServiceUnavailableException"
	const errorResponseTagName = "ErrorResponse"

	switch start.Name.Local {
	case svcUnavailableTagName:
		e.Code = svcUnavailableTagName
		e.Message = "service is unavailable"
		return d.Skip()

	case errorResponseTagName:
		return d.DecodeElement(&e.xmlErrorResponse, &start)

	default:
		return fmt.Errorf("unknown error response tag, %v", start)
	}
}

// UnmarshalError unmarshals an error response for an AWS Query service.
func UnmarshalError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	var respErr xmlResponseError
	err := xmlutil.UnmarshalXMLError(&respErr, r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				"failed to unmarshal error message", err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	reqID := respErr.RequestID
	if len(reqID) == 0 {
		reqID = r.RequestID
	}

	r.Error = awserr.NewRequestFailure(
		awserr.New(respErr.Code, respErr.Message, nil),
		r.HTTPResponse.StatusCode,
		reqID,
	)
}
