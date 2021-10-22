// +build go1.7

package arn

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/arn"
)

func TestParseOutpostAccessPointARNResource(t *testing.T) {
	cases := map[string]struct {
		ARN       arn.ARN
		ExpectErr string
		ExpectARN OutpostAccessPointARN
	}{
		"region not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/accesspoint/myendpoint",
			},
			ExpectErr: "region not set",
		},
		"account-id not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				Resource:  "outpost/myoutpost/accesspoint/myendpoint",
			},
			ExpectErr: "account-id not set",
		},
		"resource-id not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "myoutpost",
			},
			ExpectErr: "resource-id not set",
		},
		"resource-id empty": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost:",
			},
			ExpectErr: "resource-id not set",
		},
		"resource not supported": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/accesspoint/endpoint/object/key",
			},
			ExpectErr: "sub resource not supported",
		},
		"access-point not defined": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/endpoint/object/key",
			},
			ExpectErr: "unknown resource set for outpost ARN",
		},
		"valid resource-id": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/accesspoint/myaccesspoint",
			},
			ExpectARN: OutpostAccessPointARN{
				AccessPointARN: AccessPointARN{
					ARN: arn.ARN{
						Partition: "aws",
						Service:   "s3-outposts",
						Region:    "us-west-2",
						AccountID: "012345678901",
						Resource:  "outpost/myoutpost/accesspoint/myaccesspoint",
					},
					AccessPointName: "myaccesspoint",
				},
				OutpostID: "myoutpost",
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			resParts := SplitResource(c.ARN.Resource)
			a, err := ParseOutpostARNResource(c.ARN, resParts[1:])

			if len(c.ExpectErr) == 0 && err != nil {
				t.Fatalf("expect no error but got %v", err)
			} else if len(c.ExpectErr) != 0 && err == nil {
				t.Fatalf("expect error %q, but got nil", c.ExpectErr)
			} else if len(c.ExpectErr) != 0 && err != nil {
				if e, a := c.ExpectErr, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect error %q, got %q", e, a)
				}
				return
			}

			if e, a := c.ExpectARN, a; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestParseOutpostBucketARNResource(t *testing.T) {
	cases := map[string]struct {
		ARN       arn.ARN
		ExpectErr string
		ExpectARN OutpostBucketARN
	}{
		"region not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/bucket/mybucket",
			},
			ExpectErr: "region not set",
		},
		"resource-id empty": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost:",
			},
			ExpectErr: "resource-id not set",
		},
		"resource not supported": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/bucket/mybucket/object/key",
			},
			ExpectErr: "sub resource not supported",
		},
		"bucket not defined": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/endpoint/object/key",
			},
			ExpectErr: "unknown resource set for outpost ARN",
		},
		"valid resource-id": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3-outposts",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "outpost/myoutpost/bucket/mybucket",
			},
			ExpectARN: OutpostBucketARN{
				ARN: arn.ARN{
					Partition: "aws",
					Service:   "s3-outposts",
					Region:    "us-west-2",
					AccountID: "012345678901",
					Resource:  "outpost/myoutpost/bucket/mybucket",
				},
				BucketName: "mybucket",
				OutpostID:  "myoutpost",
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			resParts := SplitResource(c.ARN.Resource)
			a, err := ParseOutpostARNResource(c.ARN, resParts[1:])

			if len(c.ExpectErr) == 0 && err != nil {
				t.Fatalf("expect no error but got %v", err)
			} else if len(c.ExpectErr) != 0 && err == nil {
				t.Fatalf("expect error %q, but got nil", c.ExpectErr)
			} else if len(c.ExpectErr) != 0 && err != nil {
				if e, a := c.ExpectErr, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect error %q, got %q", e, a)
				}
				return
			}

			if e, a := c.ExpectARN, a; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}

func TestParseBucketResource(t *testing.T) {
	cases := map[string]struct {
		ARN              arn.ARN
		ExpectErr        string
		ExpectBucketName string
	}{
		"resource-id empty": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "bucket:",
			},
			ExpectErr: "bucket resource-id not set",
		},
		"resource not supported": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "bucket/mybucket/object/key",
			},
			ExpectErr: "sub resource not supported",
		},
		"valid resource-id": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "bucket/mybucket",
			},
			ExpectBucketName: "mybucket",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			resParts := SplitResource(c.ARN.Resource)
			a, err := parseBucketResource(c.ARN, resParts[1:])

			if len(c.ExpectErr) == 0 && err != nil {
				t.Fatalf("expect no error but got %v", err)
			} else if len(c.ExpectErr) != 0 && err == nil {
				t.Fatalf("expect error %q, but got nil", c.ExpectErr)
			} else if len(c.ExpectErr) != 0 && err != nil {
				if e, a := c.ExpectErr, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect error %q, got %q", e, a)
				}
				return
			}

			if e, a := c.ExpectBucketName, a; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}
