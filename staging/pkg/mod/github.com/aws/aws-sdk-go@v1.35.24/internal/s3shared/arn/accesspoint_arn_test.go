// +build go1.7

package arn

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/arn"
)

func TestParseAccessPointResource(t *testing.T) {
	cases := map[string]struct {
		ARN       arn.ARN
		ExpectErr string
		ExpectARN AccessPointARN
	}{
		"region not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				AccountID: "012345678901",
				Resource:  "accesspoint/myendpoint",
			},
			ExpectErr: "region not set",
		},
		"account-id not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				Resource:  "accesspoint/myendpoint",
			},
			ExpectErr: "account-id not set",
		},
		"resource-id not set": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "accesspoint",
			},
			ExpectErr: "resource-id not set",
		},
		"resource-id empty": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "accesspoint:",
			},
			ExpectErr: "resource-id not set",
		},
		"resource not supported": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "accesspoint/endpoint/object/key",
			},
			ExpectErr: "sub resource not supported",
		},
		"valid resource-id": {
			ARN: arn.ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "us-west-2",
				AccountID: "012345678901",
				Resource:  "accesspoint/endpoint",
			},
			ExpectARN: AccessPointARN{
				ARN: arn.ARN{
					Partition: "aws",
					Service:   "s3",
					Region:    "us-west-2",
					AccountID: "012345678901",
					Resource:  "accesspoint/endpoint",
				},
				AccessPointName: "endpoint",
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			resParts := SplitResource(c.ARN.Resource)
			a, err := ParseAccessPointResource(c.ARN, resParts[1:])

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
