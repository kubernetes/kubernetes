package arn

import (
	"strings"

	"github.com/aws/aws-sdk-go/aws/arn"
)

// OutpostARN interface that should be satisfied by outpost ARNs
type OutpostARN interface {
	Resource
	GetOutpostID() string
}

// ParseOutpostARNResource will parse a provided ARNs resource using the appropriate ARN format
// and return a specific OutpostARN type
//
// Currently supported outpost ARN formats:
// * Outpost AccessPoint ARN format:
//		- ARN format: arn:{partition}:s3-outposts:{region}:{accountId}:outpost/{outpostId}/accesspoint/{accesspointName}
//		- example: arn:aws:s3-outposts:us-west-2:012345678901:outpost/op-1234567890123456/accesspoint/myaccesspoint
//
// * Outpost Bucket ARN format:
// 		- ARN format: arn:{partition}:s3-outposts:{region}:{accountId}:outpost/{outpostId}/bucket/{bucketName}
//		- example: arn:aws:s3-outposts:us-west-2:012345678901:outpost/op-1234567890123456/bucket/mybucket
//
// Other outpost ARN formats may be supported and added in the future.
//
func ParseOutpostARNResource(a arn.ARN, resParts []string) (OutpostARN, error) {
	if len(a.Region) == 0 {
		return nil, InvalidARNError{ARN: a, Reason: "region not set"}
	}

	if len(a.AccountID) == 0 {
		return nil, InvalidARNError{ARN: a, Reason: "account-id not set"}
	}

	// verify if outpost id is present and valid
	if len(resParts) == 0 || len(strings.TrimSpace(resParts[0])) == 0 {
		return nil, InvalidARNError{ARN: a, Reason: "outpost resource-id not set"}
	}

	// verify possible resource type exists
	if len(resParts) < 3 {
		return nil, InvalidARNError{
			ARN: a, Reason: "incomplete outpost resource type. Expected bucket or access-point resource to be present",
		}
	}

	// Since we know this is a OutpostARN fetch outpostID
	outpostID := strings.TrimSpace(resParts[0])

	switch resParts[1] {
	case "accesspoint":
		accesspointARN, err := ParseAccessPointResource(a, resParts[2:])
		if err != nil {
			return OutpostAccessPointARN{}, err
		}
		return OutpostAccessPointARN{
			AccessPointARN: accesspointARN,
			OutpostID:      outpostID,
		}, nil

	case "bucket":
		bucketName, err := parseBucketResource(a, resParts[2:])
		if err != nil {
			return nil, err
		}
		return OutpostBucketARN{
			ARN:        a,
			BucketName: bucketName,
			OutpostID:  outpostID,
		}, nil

	default:
		return nil, InvalidARNError{ARN: a, Reason: "unknown resource set for outpost ARN"}
	}
}

// OutpostAccessPointARN represents outpost access point ARN.
type OutpostAccessPointARN struct {
	AccessPointARN
	OutpostID string
}

// GetOutpostID returns the outpost id of outpost access point arn
func (o OutpostAccessPointARN) GetOutpostID() string {
	return o.OutpostID
}

// OutpostBucketARN represents the outpost bucket ARN.
type OutpostBucketARN struct {
	arn.ARN
	BucketName string
	OutpostID  string
}

// GetOutpostID returns the outpost id of outpost bucket arn
func (o OutpostBucketARN) GetOutpostID() string {
	return o.OutpostID
}

// GetARN retrives the base ARN from outpost bucket ARN resource
func (o OutpostBucketARN) GetARN() arn.ARN {
	return o.ARN
}

// parseBucketResource attempts to parse the ARN's bucket resource and retrieve the
// bucket resource id.
//
// parseBucketResource only parses the bucket resource id.
//
func parseBucketResource(a arn.ARN, resParts []string) (bucketName string, err error) {
	if len(resParts) == 0 {
		return bucketName, InvalidARNError{ARN: a, Reason: "bucket resource-id not set"}
	}
	if len(resParts) > 1 {
		return bucketName, InvalidARNError{ARN: a, Reason: "sub resource not supported"}
	}

	bucketName = strings.TrimSpace(resParts[0])
	if len(bucketName) == 0 {
		return bucketName, InvalidARNError{ARN: a, Reason: "bucket resource-id not set"}
	}
	return bucketName, err
}
