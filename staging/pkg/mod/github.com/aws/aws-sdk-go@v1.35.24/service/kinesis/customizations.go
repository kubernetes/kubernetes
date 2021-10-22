package kinesis

import (
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

var readDuration = 5 * time.Second

func init() {
	initRequest = customizeRequest
}

func customizeRequest(r *request.Request) {
	if r.Operation.Name == opGetRecords {
		r.ApplyOptions(request.WithResponseReadTimeout(readDuration))
	}

	// Service specific error codes. Github(aws/aws-sdk-go#1376)
	r.RetryErrorCodes = append(r.RetryErrorCodes, ErrCodeLimitExceededException)
}
