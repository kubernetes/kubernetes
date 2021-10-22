package s3

import (
	"fmt"
	"net/url"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
)

// an operationBlacklist is a list of operation names that should a
// request handler should not be executed with.
type operationBlacklist []string

// Continue will return true of the Request's operation name is not
// in the blacklist. False otherwise.
func (b operationBlacklist) Continue(r *request.Request) bool {
	for i := 0; i < len(b); i++ {
		if b[i] == r.Operation.Name {
			return false
		}
	}
	return true
}

var accelerateOpBlacklist = operationBlacklist{
	opListBuckets, opCreateBucket, opDeleteBucket,
}

// Automatically add the bucket name to the endpoint domain
// if possible. This style of bucket is valid for all bucket names which are
// DNS compatible and do not contain "."
func updateEndpointForS3Config(r *request.Request, bucketName string) {
	forceHostStyle := aws.BoolValue(r.Config.S3ForcePathStyle)
	accelerate := aws.BoolValue(r.Config.S3UseAccelerate)

	if accelerate && accelerateOpBlacklist.Continue(r) {
		if forceHostStyle {
			if r.Config.Logger != nil {
				r.Config.Logger.Log("ERROR: aws.Config.S3UseAccelerate is not compatible with aws.Config.S3ForcePathStyle, ignoring S3ForcePathStyle.")
			}
		}
		updateEndpointForAccelerate(r, bucketName)
	} else if !forceHostStyle && r.Operation.Name != opGetBucketLocation {
		updateEndpointForHostStyle(r, bucketName)
	}
}

func updateEndpointForHostStyle(r *request.Request, bucketName string) {
	if !hostCompatibleBucketName(r.HTTPRequest.URL, bucketName) {
		// bucket name must be valid to put into the host
		return
	}

	moveBucketToHost(r.HTTPRequest.URL, bucketName)
}

var (
	accelElem = []byte("s3-accelerate.dualstack.")
)

func updateEndpointForAccelerate(r *request.Request, bucketName string) {
	if !hostCompatibleBucketName(r.HTTPRequest.URL, bucketName) {
		r.Error = awserr.New("InvalidParameterException",
			fmt.Sprintf("bucket name %s is not compatible with S3 Accelerate", bucketName),
			nil)
		return
	}

	parts := strings.Split(r.HTTPRequest.URL.Host, ".")
	if len(parts) < 3 {
		r.Error = awserr.New("InvalidParameterExecption",
			fmt.Sprintf("unable to update endpoint host for S3 accelerate, hostname invalid, %s",
				r.HTTPRequest.URL.Host), nil)
		return
	}

	if parts[0] == "s3" || strings.HasPrefix(parts[0], "s3-") {
		parts[0] = "s3-accelerate"
	}
	for i := 1; i+1 < len(parts); i++ {
		if parts[i] == aws.StringValue(r.Config.Region) {
			parts = append(parts[:i], parts[i+1:]...)
			break
		}
	}

	r.HTTPRequest.URL.Host = strings.Join(parts, ".")

	moveBucketToHost(r.HTTPRequest.URL, bucketName)
}

// Attempts to retrieve the bucket name from the request input parameters.
// If no bucket is found, or the field is empty "", false will be returned.
func bucketNameFromReqParams(params interface{}) (string, bool) {
	if iface, ok := params.(bucketGetter); ok {
		b := iface.getBucket()
		return b, len(b) > 0
	}

	return "", false
}

// hostCompatibleBucketName returns true if the request should
// put the bucket in the host. This is false if S3ForcePathStyle is
// explicitly set or if the bucket is not DNS compatible.
func hostCompatibleBucketName(u *url.URL, bucket string) bool {
	// Bucket might be DNS compatible but dots in the hostname will fail
	// certificate validation, so do not use host-style.
	if u.Scheme == "https" && strings.Contains(bucket, ".") {
		return false
	}

	// if the bucket is DNS compatible
	return dnsCompatibleBucketName(bucket)
}

var reDomain = regexp.MustCompile(`^[a-z0-9][a-z0-9\.\-]{1,61}[a-z0-9]$`)
var reIPAddress = regexp.MustCompile(`^(\d+\.){3}\d+$`)

// dnsCompatibleBucketName returns true if the bucket name is DNS compatible.
// Buckets created outside of the classic region MUST be DNS compatible.
func dnsCompatibleBucketName(bucket string) bool {
	return reDomain.MatchString(bucket) &&
		!reIPAddress.MatchString(bucket) &&
		!strings.Contains(bucket, "..")
}

// moveBucketToHost moves the bucket name from the URI path to URL host.
func moveBucketToHost(u *url.URL, bucket string) {
	u.Host = bucket + "." + u.Host
	removeBucketFromPath(u)
}
