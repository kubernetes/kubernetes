package kinesis

import (
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

var readDuration = 5 * time.Second

func init() {
	ops := []string{
		opGetRecords,
	}
	initRequest = func(r *request.Request) {
		for _, operation := range ops {
			if r.Operation.Name == operation {
				r.ApplyOptions(request.WithResponseReadTimeout(readDuration))
			}
		}
	}
}
