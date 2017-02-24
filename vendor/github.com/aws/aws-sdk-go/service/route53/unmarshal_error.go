package route53

import (
	"bytes"
	"encoding/xml"
	"io/ioutil"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/restxml"
)

type baseXMLErrorResponse struct {
	XMLName xml.Name
}

type standardXMLErrorResponse struct {
	XMLName   xml.Name `xml:"ErrorResponse"`
	Code      string   `xml:"Error>Code"`
	Message   string   `xml:"Error>Message"`
	RequestID string   `xml:"RequestId"`
}

type invalidChangeBatchXMLErrorResponse struct {
	XMLName  xml.Name `xml:"InvalidChangeBatch"`
	Messages []string `xml:"Messages>Message"`
}

func unmarshalChangeResourceRecordSetsError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	responseBody, err := ioutil.ReadAll(r.HTTPResponse.Body)

	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to read Route53 XML error response", err)
		return
	}

	baseError := &baseXMLErrorResponse{}

	if err := xml.Unmarshal(responseBody, baseError); err != nil {
		r.Error = awserr.New("SerializationError", "failed to decode Route53 XML error response", err)
		return
	}

	switch baseError.XMLName.Local {
	case "InvalidChangeBatch":
		unmarshalInvalidChangeBatchError(r, responseBody)
	default:
		r.HTTPResponse.Body = ioutil.NopCloser(bytes.NewReader(responseBody))
		restxml.UnmarshalError(r)
	}
}

func unmarshalInvalidChangeBatchError(r *request.Request, requestBody []byte) {
	resp := &invalidChangeBatchXMLErrorResponse{}
	err := xml.Unmarshal(requestBody, resp)

	if err != nil {
		r.Error = awserr.New("SerializationError", "failed to decode query XML error response", err)
		return
	}

	const errorCode = "InvalidChangeBatch"
	errors := []error{}

	for _, msg := range resp.Messages {
		errors = append(errors, awserr.New(errorCode, msg, nil))
	}

	r.Error = awserr.NewRequestFailure(
		awserr.NewBatchError(errorCode, "ChangeBatch errors occurred", errors),
		r.HTTPResponse.StatusCode,
		r.RequestID,
	)

}
