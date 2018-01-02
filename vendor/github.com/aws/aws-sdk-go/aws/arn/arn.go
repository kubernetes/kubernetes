// Package arn provides a parser for interacting with Amazon Resource Names.
package arn

import (
	"errors"
	"strings"
)

const (
	arnDelimiter = ":"
	arnSections  = 6
	arnPrefix    = "arn:"

	// zero-indexed
	sectionPartition = 1
	sectionService   = 2
	sectionRegion    = 3
	sectionAccountID = 4
	sectionResource  = 5

	// errors
	invalidPrefix   = "arn: invalid prefix"
	invalidSections = "arn: not enough sections"
)

// ARN captures the individual fields of an Amazon Resource Name.
// See http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html for more information.
type ARN struct {
	// The partition that the resource is in. For standard AWS regions, the partition is "aws". If you have resources in
	// other partitions, the partition is "aws-partitionname". For example, the partition for resources in the China
	// (Beijing) region is "aws-cn".
	Partition string

	// The service namespace that identifies the AWS product (for example, Amazon S3, IAM, or Amazon RDS). For a list of
	// namespaces, see
	// http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html#genref-aws-service-namespaces.
	Service string

	// The region the resource resides in. Note that the ARNs for some resources do not require a region, so this
	// component might be omitted.
	Region string

	// The ID of the AWS account that owns the resource, without the hyphens. For example, 123456789012. Note that the
	// ARNs for some resources don't require an account number, so this component might be omitted.
	AccountID string

	// The content of this part of the ARN varies by service. It often includes an indicator of the type of resource —
	// for example, an IAM user or Amazon RDS database - followed by a slash (/) or a colon (:), followed by the
	// resource name itself. Some services allows paths for resource names, as described in
	// http://docs.aws.amazon.com/general/latest/gr/aws-arns-and-namespaces.html#arns-paths.
	Resource string
}

// Parse parses an ARN into its constituent parts.
//
// Some example ARNs:
// arn:aws:elasticbeanstalk:us-east-1:123456789012:environment/My App/MyEnvironment
// arn:aws:iam::123456789012:user/David
// arn:aws:rds:eu-west-1:123456789012:db:mysql-db
// arn:aws:s3:::my_corporate_bucket/exampleobject.png
func Parse(arn string) (ARN, error) {
	if !strings.HasPrefix(arn, arnPrefix) {
		return ARN{}, errors.New(invalidPrefix)
	}
	sections := strings.SplitN(arn, arnDelimiter, arnSections)
	if len(sections) != arnSections {
		return ARN{}, errors.New(invalidSections)
	}
	return ARN{
		Partition: sections[sectionPartition],
		Service:   sections[sectionService],
		Region:    sections[sectionRegion],
		AccountID: sections[sectionAccountID],
		Resource:  sections[sectionResource],
	}, nil
}

// String returns the canonical representation of the ARN
func (arn ARN) String() string {
	return arnPrefix +
		arn.Partition + arnDelimiter +
		arn.Service + arnDelimiter +
		arn.Region + arnDelimiter +
		arn.AccountID + arnDelimiter +
		arn.Resource
}
