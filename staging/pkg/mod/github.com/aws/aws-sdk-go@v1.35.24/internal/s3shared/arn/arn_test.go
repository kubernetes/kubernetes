// +build go1.7

package arn

import (
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go/aws/arn"
)

func TestParseResource(t *testing.T) {
	cases := map[string]struct {
		Input           string
		MappedResources map[string]func(arn.ARN, []string) (Resource, error)
		Expect          Resource
		ExpectErr       string
	}{
		"Empty ARN": {
			Input:     "",
			ExpectErr: "arn: invalid prefix",
		},
		"No Partition": {
			Input:     "arn::sqs:us-west-2:012345678901:accesspoint",
			ExpectErr: "partition not set",
		},
		"Not S3 ARN": {
			Input:     "arn:aws:sqs:us-west-2:012345678901:accesspoint",
			ExpectErr: "service is not supported",
		},
		"No Resource": {
			Input:     "arn:aws:s3:us-west-2:012345678901:",
			ExpectErr: "resource not set",
		},
		"Unknown Resource Type": {
			Input:     "arn:aws:s3:us-west-2:012345678901:myresource",
			ExpectErr: "unknown resource type",
		},
		"Unknown BucketARN Resource Type": {
			Input:     "arn:aws:s3:us-west-2:012345678901:bucket_name:mybucket",
			ExpectErr: "unknown resource type",
		},
		"Unknown Resource Type with Resource and Sub-Resource": {
			Input:     "arn:aws:s3:us-west-2:012345678901:somethingnew:myresource/subresource",
			ExpectErr: "unknown resource type",
		},
		"Access Point with sub resource": {
			Input: "arn:aws:s3:us-west-2:012345678901:accesspoint:myresource/subresource",
			MappedResources: map[string]func(arn.ARN, []string) (Resource, error){
				"accesspoint": func(a arn.ARN, parts []string) (Resource, error) {
					return ParseAccessPointResource(a, parts)
				},
			},
			ExpectErr: "resource not supported",
		},
		"AccessPoint Resource Type": {
			Input: "arn:aws:s3:us-west-2:012345678901:accesspoint:myendpoint",
			MappedResources: map[string]func(arn.ARN, []string) (Resource, error){
				"accesspoint": func(a arn.ARN, parts []string) (Resource, error) {
					return ParseAccessPointResource(a, parts)
				},
			},
			Expect: AccessPointARN{
				ARN: arn.ARN{
					Partition: "aws",
					Service:   "s3",
					Region:    "us-west-2",
					AccountID: "012345678901",
					Resource:  "accesspoint:myendpoint",
				},
				AccessPointName: "myendpoint",
			},
		},
		"AccessPoint Resource Type With Path Syntax": {
			Input: "arn:aws:s3:us-west-2:012345678901:accesspoint/myendpoint",
			MappedResources: map[string]func(arn.ARN, []string) (Resource, error){
				"accesspoint": func(a arn.ARN, parts []string) (Resource, error) {
					return ParseAccessPointResource(a, parts)
				},
			},
			Expect: AccessPointARN{
				ARN: arn.ARN{
					Partition: "aws",
					Service:   "s3",
					Region:    "us-west-2",
					AccountID: "012345678901",
					Resource:  "accesspoint/myendpoint",
				},
				AccessPointName: "myendpoint",
			},
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			parsed, err := ParseResource(c.Input, mappedResourceParser(c.MappedResources))

			if len(c.ExpectErr) == 0 && err != nil {
				t.Fatalf("expect no error but got %v", err)
			} else if len(c.ExpectErr) != 0 && err == nil {
				t.Fatalf("expect error but got nil")
			} else if len(c.ExpectErr) != 0 && err != nil {
				if e, a := c.ExpectErr, err.Error(); !strings.Contains(a, e) {
					t.Fatalf("expect error %q, got %q", e, a)
				}
				return
			}

			if e, a := c.Expect, parsed; !reflect.DeepEqual(e, a) {
				t.Errorf("Expect %v, got %v", e, a)
			}
		})
	}
}

func mappedResourceParser(kinds map[string]func(arn.ARN, []string) (Resource, error)) ResourceParser {
	return func(a arn.ARN) (Resource, error) {
		parts := SplitResource(a.Resource)

		fn, ok := kinds[parts[0]]
		if !ok {
			return nil, InvalidARNError{ARN: a, Reason: "unknown resource type"}
		}
		return fn(a, parts[1:])
	}
}

func TestSplitResource(t *testing.T) {
	cases := []struct {
		Input  string
		Expect []string
	}{
		{
			Input:  "accesspoint:myendpoint",
			Expect: []string{"accesspoint", "myendpoint"},
		},
		{
			Input:  "accesspoint/myendpoint",
			Expect: []string{"accesspoint", "myendpoint"},
		},
		{
			Input:  "accesspoint",
			Expect: []string{"accesspoint"},
		},
		{
			Input:  "accesspoint:",
			Expect: []string{"accesspoint", ""},
		},
		{
			Input:  "accesspoint:  ",
			Expect: []string{"accesspoint", "  "},
		},
		{
			Input:  "accesspoint:endpoint/object/key",
			Expect: []string{"accesspoint", "endpoint", "object", "key"},
		},
	}

	for _, c := range cases {
		t.Run(c.Input, func(t *testing.T) {
			parts := SplitResource(c.Input)
			if e, a := c.Expect, parts; !reflect.DeepEqual(e, a) {
				t.Errorf("expect %v, got %v", e, a)
			}
		})
	}
}
