package s3control

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	awsarn "github.com/aws/aws-sdk-go/aws/arn"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/s3shared"
	"github.com/aws/aws-sdk-go/internal/s3shared/arn"
)

const (
	// outpost id header
	outpostIDHeader = "x-amz-outpost-id"

	// account id header
	accountIDHeader = "x-amz-account-id"
)

// Used by shapes with members decorated as endpoint ARN.
func parseEndpointARN(v string) (arn.Resource, error) {
	return arn.ParseResource(v, resourceParser)
}

func resourceParser(a awsarn.ARN) (arn.Resource, error) {
	resParts := arn.SplitResource(a.Resource)
	switch resParts[0] {
	case "outpost":
		return arn.ParseOutpostARNResource(a, resParts[1:])
	default:
		return nil, arn.InvalidARNError{ARN: a, Reason: "unknown resource type"}
	}
}

func endpointHandler(req *request.Request) {
	// For special case "CreateBucket" and "ListRegionalBuckets" operation
	outpostIDEndpoint, ok := req.Params.(endpointOutpostIDGetter)
	if ok && outpostIDEndpoint.hasOutpostID() {
		outpostID, err := outpostIDEndpoint.getOutpostID()
		if err != nil {
			req.Error = fmt.Errorf("expected outpost ID to be supported, %v", err)
		}
		if len(strings.TrimSpace(outpostID)) == 0 {
			return
		}
		updateRequestOutpostIDEndpoint(req)
		return
	}

	endpoint, ok := req.Params.(endpointARNGetter)
	if !ok || !endpoint.hasEndpointARN() {
		return
	}

	resource, err := endpoint.getEndpointARN()
	if err != nil {
		req.Error = s3shared.NewInvalidARNError(nil, err)
		return
	}

	// Add account-id header for the request if not present.
	// SDK must always send the x-amz-account-id header for all requests
	// where an accountId has been extracted from an ARN or the accountId field modeled as a header.
	if h := req.HTTPRequest.Header.Get(accountIDHeader); len(h) == 0 {
		req.HTTPRequest.Header.Add(accountIDHeader, resource.GetARN().AccountID)
	}

	switch tv := resource.(type) {
	case arn.OutpostAccessPointARN:
		// Add outpostID header
		req.HTTPRequest.Header.Add(outpostIDHeader, tv.OutpostID)

		// update arnable field to resource value
		updatedInput, err := endpoint.updateArnableField(tv.AccessPointName)
		if err != nil {
			req.Error = err
			return
		}

		// update request params to use modified ARN field value, if not nil
		if updatedInput != nil {
			req.Params = updatedInput
		}

		// update request for outpost access point endpoint
		err = updateRequestOutpostAccessPointEndpoint(req, tv)
		if err != nil {
			req.Error = err
		}
	case arn.OutpostBucketARN:
		// Add outpostID header
		req.HTTPRequest.Header.Add(outpostIDHeader, tv.OutpostID)

		// update arnable field to resource value
		updatedInput, err := endpoint.updateArnableField(tv.BucketName)
		if err != nil {
			req.Error = err
			return
		}

		// update request params to use modified ARN field value, if not nil
		if updatedInput != nil {
			req.Params = updatedInput
		}

		// update request for outpost bucket endpoint
		err = updateRequestOutpostBucketEndpoint(req, tv)
		if err != nil {
			req.Error = err
		}
	default:
		req.Error = s3shared.NewInvalidARNError(resource, nil)
	}
}

// updateRequestOutpostIDEndpoint is special customization to be applied for operations
// CreateBucket, ListRegionalBuckets which must resolve endpoint to s3-outposts.{region}.amazonaws.com
// with region as client region and signed by s3-control if an outpost id is provided.
func updateRequestOutpostIDEndpoint(request *request.Request) {
	serviceEndpointLabel := "s3-outposts."
	cfgRegion := aws.StringValue(request.Config.Region)

	// request url
	request.HTTPRequest.URL.Host = serviceEndpointLabel + cfgRegion + ".amazonaws.com"

	// disable the host prefix for outpost access points
	request.Config.DisableEndpointHostPrefix = aws.Bool(true)

	// signer redirection
	request.ClientInfo.SigningName = "s3-outposts"
	request.ClientInfo.SigningRegion = cfgRegion
}

func updateRequestOutpostAccessPointEndpoint(req *request.Request, accessPoint arn.OutpostAccessPointARN) error {
	// validate Outpost endpoint
	if err := validateOutpostEndpoint(req, accessPoint); err != nil {
		return err
	}

	// disable the host prefix for outpost access points
	req.Config.DisableEndpointHostPrefix = aws.Bool(true)

	if err := outpostAccessPointEndpointBuilder(accessPoint).build(req); err != nil {
		return err
	}

	return nil
}

func updateRequestOutpostBucketEndpoint(req *request.Request, bucketResource arn.OutpostBucketARN) error {
	// validate Outpost endpoint
	if err := validateOutpostEndpoint(req, bucketResource); err != nil {
		return err
	}

	// disable the host prefix for outpost bucket.
	req.Config.DisableEndpointHostPrefix = aws.Bool(true)

	if err := outpostBucketResourceEndpointBuilder(bucketResource).build(req); err != nil {
		return err
	}

	return nil
}

// validate request resource for retrieving endpoint
func validateEndpointRequestResource(req *request.Request, resource arn.Resource) error {
	resReq := s3shared.ResourceRequest{Request: req, Resource: resource}

	if resReq.IsCrossPartition() {
		return s3shared.NewClientPartitionMismatchError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}

	if !resReq.AllowCrossRegion() && resReq.IsCrossRegion() {
		return s3shared.NewClientRegionMismatchError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}

	if resReq.HasCustomEndpoint() {
		return s3shared.NewInvalidARNWithCustomEndpointError(resource, nil)
	}

	// Accelerate not supported
	if aws.BoolValue(req.Config.S3UseAccelerate) {
		return s3shared.NewClientConfiguredForAccelerateError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}
	return nil
}

// validations for fetching outpost endpoint
func validateOutpostEndpoint(req *request.Request, resource arn.Resource) error {
	resReq := s3shared.ResourceRequest{
		Request:  req,
		Resource: resource,
	}

	if err := validateEndpointRequestResource(req, resource); err != nil {
		return err
	}

	// resource configured with FIPS as region is not supported by outposts
	if resReq.ResourceConfiguredForFIPS() {
		return s3shared.NewInvalidARNWithFIPSError(resource, nil)
	}

	// DualStack not supported
	if aws.BoolValue(req.Config.UseDualStack) {
		return s3shared.NewClientConfiguredForDualStackError(resource,
			req.ClientInfo.PartitionID, aws.StringValue(req.Config.Region), nil)
	}
	return nil
}
