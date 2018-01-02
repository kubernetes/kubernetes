package machinelearning

import (
	"net/url"

	"github.com/aws/aws-sdk-go/aws/request"
)

func init() {
	initRequest = func(r *request.Request) {
		switch r.Operation.Name {
		case opPredict:
			r.Handlers.Build.PushBack(updatePredictEndpoint)
		}
	}
}

// updatePredictEndpoint rewrites the request endpoint to use the
// "PredictEndpoint" parameter of the Predict operation.
func updatePredictEndpoint(r *request.Request) {
	if !r.ParamsFilled() {
		return
	}

	r.ClientInfo.Endpoint = *r.Params.(*PredictInput).PredictEndpoint

	uri, err := url.Parse(r.ClientInfo.Endpoint)
	if err != nil {
		r.Error = err
		return
	}
	r.HTTPRequest.URL = uri
}
