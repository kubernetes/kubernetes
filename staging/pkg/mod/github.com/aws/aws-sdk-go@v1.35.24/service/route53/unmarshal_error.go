package route53

import (
	"encoding/xml"
	"fmt"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/xml/xmlutil"
)

const errorRespTag = "ErrorResponse"
const invalidChangeTag = "InvalidChangeBatch"

type standardXMLErrorResponse struct {
	Code      string `xml:"Error>Code"`
	Message   string `xml:"Error>Message"`
	RequestID string `xml:"RequestId"`
}

func (e standardXMLErrorResponse) FillCommon(c *xmlErrorResponse) {
	c.Code = e.Code
	c.Message = e.Message
	c.RequestID = e.RequestID
}

type invalidChangeBatchXMLErrorResponse struct {
	Messages  []string `xml:"Messages>Message"`
	RequestID string   `xml:"RequestId"`
}

func (e invalidChangeBatchXMLErrorResponse) FillCommon(c *xmlErrorResponse) {
	c.Code = invalidChangeTag
	c.Message = "ChangeBatch errors occurred"
	c.Messages = e.Messages
	c.RequestID = e.RequestID
}

type xmlErrorResponse struct {
	Code      string
	Message   string
	Messages  []string
	RequestID string
}

func (e *xmlErrorResponse) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	type commonFiller interface {
		FillCommon(*xmlErrorResponse)
	}

	var errResp commonFiller
	switch start.Name.Local {
	case errorRespTag:
		errResp = &standardXMLErrorResponse{}

	case invalidChangeTag:
		errResp = &invalidChangeBatchXMLErrorResponse{}

	default:
		return fmt.Errorf("unknown error message, %v", start.Name.Local)
	}

	if err := d.DecodeElement(errResp, &start); err != nil {
		return err
	}

	errResp.FillCommon(e)
	return nil
}

func unmarshalChangeResourceRecordSetsError(r *request.Request) {
	defer r.HTTPResponse.Body.Close()

	var errResp xmlErrorResponse
	err := xmlutil.UnmarshalXMLError(&errResp, r.HTTPResponse.Body)
	if err != nil {
		r.Error = awserr.NewRequestFailure(
			awserr.New(request.ErrCodeSerialization,
				"failed to unmarshal error message", err),
			r.HTTPResponse.StatusCode,
			r.RequestID,
		)
		return
	}

	var baseErr awserr.Error
	if len(errResp.Messages) != 0 {
		var errs []error
		for _, msg := range errResp.Messages {
			errs = append(errs, awserr.New(invalidChangeTag, msg, nil))
		}
		baseErr = awserr.NewBatchError(errResp.Code, errResp.Message, errs)
	} else {
		baseErr = awserr.New(errResp.Code, errResp.Message, nil)
	}

	reqID := errResp.RequestID
	if len(reqID) == 0 {
		reqID = r.RequestID
	}
	r.Error = awserr.NewRequestFailure(
		baseErr,
		r.HTTPResponse.StatusCode,
		reqID,
	)
}
