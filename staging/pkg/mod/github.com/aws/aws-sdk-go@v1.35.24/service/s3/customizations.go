package s3

import (
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/s3shared/arn"
	"github.com/aws/aws-sdk-go/internal/s3shared/s3err"
)

func init() {
	initClient = defaultInitClientFn
	initRequest = defaultInitRequestFn
}

func defaultInitClientFn(c *client.Client) {
	// Support building custom endpoints based on config
	c.Handlers.Build.PushFront(endpointHandler)

	// Require SSL when using SSE keys
	c.Handlers.Validate.PushBack(validateSSERequiresSSL)
	c.Handlers.Build.PushBack(computeSSEKeyMD5)
	c.Handlers.Build.PushBack(computeCopySourceSSEKeyMD5)

	// S3 uses custom error unmarshaling logic
	c.Handlers.UnmarshalError.Clear()
	c.Handlers.UnmarshalError.PushBack(unmarshalError)
	c.Handlers.UnmarshalError.PushBackNamed(s3err.RequestFailureWrapperHandler())
}

func defaultInitRequestFn(r *request.Request) {
	// Add request handlers for specific platforms.
	// e.g. 100-continue support for PUT requests using Go 1.6
	platformRequestHandlers(r)

	switch r.Operation.Name {
	case opGetBucketLocation:
		// GetBucketLocation has custom parsing logic
		r.Handlers.Unmarshal.PushFront(buildGetBucketLocation)
	case opCreateBucket:
		// Auto-populate LocationConstraint with current region
		r.Handlers.Validate.PushFront(populateLocationConstraint)
	case opCopyObject, opUploadPartCopy, opCompleteMultipartUpload:
		r.Handlers.Unmarshal.PushFront(copyMultipartStatusOKUnmarhsalError)
		r.Handlers.Unmarshal.PushBackNamed(s3err.RequestFailureWrapperHandler())
	case opPutObject, opUploadPart:
		r.Handlers.Build.PushBack(computeBodyHashes)
		// Disabled until #1837 root issue is resolved.
		//	case opGetObject:
		//		r.Handlers.Build.PushBack(askForTxEncodingAppendMD5)
		//		r.Handlers.Unmarshal.PushBack(useMD5ValidationReader)
	}
}

// bucketGetter is an accessor interface to grab the "Bucket" field from
// an S3 type.
type bucketGetter interface {
	getBucket() string
}

// sseCustomerKeyGetter is an accessor interface to grab the "SSECustomerKey"
// field from an S3 type.
type sseCustomerKeyGetter interface {
	getSSECustomerKey() string
}

// copySourceSSECustomerKeyGetter is an accessor interface to grab the
// "CopySourceSSECustomerKey" field from an S3 type.
type copySourceSSECustomerKeyGetter interface {
	getCopySourceSSECustomerKey() string
}

// endpointARNGetter is an accessor interface to grab the
// the field corresponding to an endpoint ARN input.
type endpointARNGetter interface {
	getEndpointARN() (arn.Resource, error)
	hasEndpointARN() bool
}
