package s3

import (
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
)

func init() {
	initClient = func(c *client.Client) {
		// Support building custom host-style bucket endpoints
		c.Handlers.Build.PushFront(updateHostWithBucket)

		// Require SSL when using SSE keys
		c.Handlers.Validate.PushBack(validateSSERequiresSSL)
		c.Handlers.Build.PushBack(computeSSEKeys)

		// S3 uses custom error unmarshaling logic
		c.Handlers.UnmarshalError.Clear()
		c.Handlers.UnmarshalError.PushBack(unmarshalError)
	}

	initRequest = func(r *request.Request) {
		switch r.Operation.Name {
		case opPutBucketCors, opPutBucketLifecycle, opPutBucketPolicy, opPutBucketTagging, opDeleteObjects:
			// These S3 operations require Content-MD5 to be set
			r.Handlers.Build.PushBack(contentMD5)
		case opGetBucketLocation:
			// GetBucketLocation has custom parsing logic
			r.Handlers.Unmarshal.PushFront(buildGetBucketLocation)
		case opCreateBucket:
			// Auto-populate LocationConstraint with current region
			r.Handlers.Validate.PushFront(populateLocationConstraint)
		case opCopyObject, opUploadPartCopy, opCompleteMultipartUpload:
			r.Handlers.Unmarshal.PushFront(copyMultipartStatusOKUnmarhsalError)
		}
	}
}
