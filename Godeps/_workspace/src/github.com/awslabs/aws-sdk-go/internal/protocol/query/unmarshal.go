package query

//go:generate go run ../../fixtures/protocol/generate.go ../../fixtures/protocol/output/query.json unmarshal_test.go

import (
	"encoding/xml"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/internal/protocol/xml/xmlutil"
)

// Unmarshal unmarshals a response for an AWS Query service.
func Unmarshal(r *aws.Request) {
	defer r.HTTPResponse.Body.Close()
	if r.DataFilled() {
		decoder := xml.NewDecoder(r.HTTPResponse.Body)
		err := xmlutil.UnmarshalXML(r.Data, decoder, r.Operation.Name+"Result")
		if err != nil {
			r.Error = err
			return
		}
	}
}

// UnmarshalMeta unmarshals header response values for an AWS Query service.
func UnmarshalMeta(r *aws.Request) {
	// TODO implement unmarshaling of request IDs
}
