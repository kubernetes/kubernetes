package s3control

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/s3shared"
)

// updateAccountIDWithARNHandler is a request named handler that is used to validate and populate the request account id
// input if it may also be present in the resource ARN.
var updateAccountIDWithARNHandler = request.NamedHandler{
	Name: "updateAccountIDWithARNHandler",
	Fn: func(req *request.Request) {
		endpoint, ok := req.Params.(endpointARNGetter)
		if !ok || !endpoint.hasEndpointARN() {
			return
		}

		// fetch endpoint arn resource
		resource, err := endpoint.getEndpointARN()
		if err != nil {
			req.Error = fmt.Errorf("error while fetching endpoint ARN: %v", err)
			return
		}

		// Validate that the present account id in a request input matches the account id
		// present in an ARN. If a value for request input account id member is not provided,
		// the accountID member is populated using the account id present in the ARN
		// and a pointer to copy of updatedInput is returned.
		if accountIDValidator, ok := req.Params.(accountIDValidator); ok {
			accID := resource.GetARN().AccountID
			updatedInput, err := accountIDValidator.updateAccountID(accID)
			if err != nil {
				req.Error = s3shared.NewInvalidARNError(resource, err)
				return
			}
			// update request params to use modified account id, if not nil
			if updatedInput != nil {
				req.Params = updatedInput
			}
		}
	},
}
