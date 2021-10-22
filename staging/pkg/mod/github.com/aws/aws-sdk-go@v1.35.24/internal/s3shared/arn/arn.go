package arn

import (
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go/aws/arn"
)

// Resource provides the interfaces abstracting ARNs of specific resource
// types.
type Resource interface {
	GetARN() arn.ARN
	String() string
}

// ResourceParser provides the function for parsing an ARN's resource
// component into a typed resource.
type ResourceParser func(arn.ARN) (Resource, error)

// ParseResource parses an AWS ARN into a typed resource for the S3 API.
func ParseResource(s string, resParser ResourceParser) (resARN Resource, err error) {
	a, err := arn.Parse(s)
	if err != nil {
		return nil, err
	}

	if len(a.Partition) == 0 {
		return nil, InvalidARNError{ARN: a, Reason: "partition not set"}
	}

	if a.Service != "s3" && a.Service != "s3-outposts" {
		return nil, InvalidARNError{ARN: a, Reason: "service is not supported"}
	}
	if len(a.Resource) == 0 {
		return nil, InvalidARNError{ARN: a, Reason: "resource not set"}
	}

	return resParser(a)
}

// SplitResource splits the resource components by the ARN resource delimiters.
func SplitResource(v string) []string {
	var parts []string
	var offset int

	for offset <= len(v) {
		idx := strings.IndexAny(v[offset:], "/:")
		if idx < 0 {
			parts = append(parts, v[offset:])
			break
		}
		parts = append(parts, v[offset:idx+offset])
		offset += idx + 1
	}

	return parts
}

// IsARN returns whether the given string is an ARN
func IsARN(s string) bool {
	return arn.IsARN(s)
}

// InvalidARNError provides the error for an invalid ARN error.
type InvalidARNError struct {
	ARN    arn.ARN
	Reason string
}

// Error returns a string denoting the occurred InvalidARNError
func (e InvalidARNError) Error() string {
	return fmt.Sprintf("invalid Amazon %s ARN, %s, %s", e.ARN.Service, e.Reason, e.ARN.String())
}
