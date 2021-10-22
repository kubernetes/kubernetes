package s3

import (
	"net/url"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	awsarn "github.com/aws/aws-sdk-go/aws/arn"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/s3shared"
	"github.com/aws/aws-sdk-go/internal/s3shared/arn"
)

// Used by shapes with members decorated as endpoint ARN.
func parseEndpointARN(v string) (arn.Resource, error) {
	return arn.ParseResource(v, accessPointResourceParser)
}

func accessPointResourceParser(a awsarn.ARN) (arn.Resource, error) {
	resParts := arn.SplitResource(a.Resource)
	switch resParts[0] {
	case "accesspoint":
		if a.Service != "s3" {
			return arn.AccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "service is not s3"}
		}
		return arn.ParseAccessPointResource(a, resParts[1:])
	case "outpost":
		if a.Service != "s3-outposts" {
			return arn.OutpostAccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "service is not s3-outposts"}
		}
		return parseOutpostAccessPointResource(a, resParts[1:])
	default:
		return nil, arn.InvalidARNError{ARN: a, Reason: "unknown resource type"}
	}
}

// parseOutpostAccessPointResource attempts to parse the ARNs resource as an
// outpost access-point resource.
//
// Supported Outpost AccessPoint ARN format:
//	- ARN format: arn:{partition}:s3-outposts:{region}:{accountId}:outpost/{outpostId}/accesspoint/{accesspointName}
//	- example: arn:aws:s3-outposts:us-west-2:012345678901:outpost/op-1234567890123456/accesspoint/myaccesspoint
//
func parseOutpostAccessPointResource(a awsarn.ARN, resParts []string) (arn.OutpostAccessPointARN, error) {
	// outpost accesspoint arn is only valid if service is s3-outposts
	if a.Service != "s3-outposts" {
		return arn.OutpostAccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "service is not s3-outposts"}
	}

	if len(resParts) == 0 {
		return arn.OutpostAccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "outpost resource-id not set"}
	}

	if len(resParts) < 3 {
		return arn.OutpostAccessPointARN{}, arn.InvalidARNError{
			ARN: a, Reason: "access-point resource not set in Outpost ARN",
		}
	}

	resID := strings.TrimSpace(resParts[0])
	if len(resID) == 0 {
		return arn.OutpostAccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "outpost resource-id not set"}
	}

	var outpostAccessPointARN = arn.OutpostAccessPointARN{}
	switch resParts[1] {
	case "accesspoint":
		accessPointARN, err := arn.ParseAccessPointResource(a, resParts[2:])
		if err != nil {
			return arn.OutpostAccessPointARN{}, err
		}
		// set access-point arn
		outpostAccessPointARN.AccessPointARN = accessPointARN
	default:
		return arn.OutpostAccessPointARN{}, arn.InvalidARNError{ARN: a, Reason: "access-point resource not set in Outpost ARN"}
	}

	// set outpost id
	outpostAccessPointARN.OutpostID = resID
	return outpostAccessPointARN, nil
}

func endpointHandler(req *request.Request) {
	endpoint, ok := req.Params.(endpointARNGetter)
	if !ok || !endpoint.hasEndpointARN() {
		updateBucketEndpointFromParams(req)
		return
	}

	resource, err := endpoint.getEndpointARN()
	if err != nil {
		req.Error = s3shared.NewInvalidARNError(nil, err)
		return
	}

	resReq := s3shared.ResourceRequest{
		Resource: resource,
		Request:  req,
	}

	if resReq.IsCrossPartition() {
		req.Error = s3shared.NewClientPartitionMismatchError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
		return
	}

	if !resReq.AllowCrossRegion() && resReq.IsCrossRegion() {
		req.Error = s3shared.NewClientRegionMismatchError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
		return
	}

	if resReq.HasCustomEndpoint() {
		req.Error = s3shared.NewInvalidARNWithCustomEndpointError(resource, nil)
		return
	}

	switch tv := resource.(type) {
	case arn.AccessPointARN:
		err = updateRequestAccessPointEndpoint(req, tv)
		if err != nil {
			req.Error = err
		}
	case arn.OutpostAccessPointARN:
		// outposts does not support FIPS regions
		if resReq.ResourceConfiguredForFIPS() {
			req.Error = s3shared.NewInvalidARNWithFIPSError(resource, nil)
			return
		}

		err = updateRequestOutpostAccessPointEndpoint(req, tv)
		if err != nil {
			req.Error = err
		}
	default:
		req.Error = s3shared.NewInvalidARNError(resource, nil)
	}
}

func updateBucketEndpointFromParams(r *request.Request) {
	bucket, ok := bucketNameFromReqParams(r.Params)
	if !ok {
		// Ignore operation requests if the bucket name was not provided
		// if this is an input validation error the validation handler
		// will report it.
		return
	}
	updateEndpointForS3Config(r, bucket)
}

func updateRequestAccessPointEndpoint(req *request.Request, accessPoint arn.AccessPointARN) error {
	// Accelerate not supported
	if aws.BoolValue(req.Config.S3UseAccelerate) {
		return s3shared.NewClientConfiguredForAccelerateError(accessPoint,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}

	// Ignore the disable host prefix for access points since custom endpoints
	// are not supported.
	req.Config.DisableEndpointHostPrefix = aws.Bool(false)

	if err := accessPointEndpointBuilder(accessPoint).build(req); err != nil {
		return err
	}

	removeBucketFromPath(req.HTTPRequest.URL)

	return nil
}

func updateRequestOutpostAccessPointEndpoint(req *request.Request, accessPoint arn.OutpostAccessPointARN) error {
	// Accelerate not supported
	if aws.BoolValue(req.Config.S3UseAccelerate) {
		return s3shared.NewClientConfiguredForAccelerateError(accessPoint,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}

	// Dualstack not supported
	if aws.BoolValue(req.Config.UseDualStack) {
		return s3shared.NewClientConfiguredForDualStackError(accessPoint,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}

	// Ignore the disable host prefix for access points since custom endpoints
	// are not supported.
	req.Config.DisableEndpointHostPrefix = aws.Bool(false)

	if err := outpostAccessPointEndpointBuilder(accessPoint).build(req); err != nil {
		return err
	}

	removeBucketFromPath(req.HTTPRequest.URL)
	return nil
}

func removeBucketFromPath(u *url.URL) {
	u.Path = strings.Replace(u.Path, "/{Bucket}", "", -1)
	if u.Path == "" {
		u.Path = "/"
	}
}
