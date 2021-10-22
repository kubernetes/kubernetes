package arn

import (
	"strings"

	"github.com/aws/aws-sdk-go/aws/arn"
)

// AccessPointARN provides representation
type AccessPointARN struct {
	arn.ARN
	AccessPointName string
}

// GetARN returns the base ARN for the Access Point resource
func (a AccessPointARN) GetARN() arn.ARN {
	return a.ARN
}

// ParseAccessPointResource attempts to parse the ARN's resource as an
// AccessPoint resource.
//
// Supported Access point resource format:
//	- Access point format: arn:{partition}:s3:{region}:{accountId}:accesspoint/{accesspointName}
//	- example: arn.aws.s3.us-west-2.012345678901:accesspoint/myaccesspoint
//
func ParseAccessPointResource(a arn.ARN, resParts []string) (AccessPointARN, error) {
	if len(a.Region) == 0 {
		return AccessPointARN{}, InvalidARNError{ARN: a, Reason: "region not set"}
	}
	if len(a.AccountID) == 0 {
		return AccessPointARN{}, InvalidARNError{ARN: a, Reason: "account-id not set"}
	}
	if len(resParts) == 0 {
		return AccessPointARN{}, InvalidARNError{ARN: a, Reason: "resource-id not set"}
	}
	if len(resParts) > 1 {
		return AccessPointARN{}, InvalidARNError{ARN: a, Reason: "sub resource not supported"}
	}

	resID := resParts[0]
	if len(strings.TrimSpace(resID)) == 0 {
		return AccessPointARN{}, InvalidARNError{ARN: a, Reason: "resource-id not set"}
	}

	return AccessPointARN{
		ARN:             a,
		AccessPointName: resID,
	}, nil
}
