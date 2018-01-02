package polly

import (
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/private/protocol/rest"
)

func init() {
	initRequest = func(r *request.Request) {
		if r.Operation.Name == opSynthesizeSpeech {
			r.Operation.BeforePresignFn = restGETPresignStrategy
		}
	}
}

// restGETPresignStrategy will prepare the request from a POST to a GET request.
// Enabling the presigner to sign the request as a GET.
func restGETPresignStrategy(r *request.Request) error {
	r.Handlers.Build.Clear()
	r.Handlers.Build.PushBack(rest.BuildAsGET)
	r.Operation.HTTPMethod = "GET"
	r.HTTPRequest.Method = "GET"
	return nil
}
