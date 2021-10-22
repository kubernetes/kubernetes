// +build go1.7

package arn

import (
	"errors"
	"testing"
)

func TestParseARN(t *testing.T) {
	cases := []struct {
		input string
		arn   ARN
		err   error
	}{
		{
			input: "invalid",
			err:   errors.New(invalidPrefix),
		},
		{
			input: "arn:nope",
			err:   errors.New(invalidSections),
		},
		{
			input: "arn:aws:ecr:us-west-2:123456789012:repository/foo/bar",
			arn: ARN{
				Partition: "aws",
				Service:   "ecr",
				Region:    "us-west-2",
				AccountID: "123456789012",
				Resource:  "repository/foo/bar",
			},
		},
		{
			input: "arn:aws:elasticbeanstalk:us-east-1:123456789012:environment/My App/MyEnvironment",
			arn: ARN{
				Partition: "aws",
				Service:   "elasticbeanstalk",
				Region:    "us-east-1",
				AccountID: "123456789012",
				Resource:  "environment/My App/MyEnvironment",
			},
		},
		{
			input: "arn:aws:iam::123456789012:user/David",
			arn: ARN{
				Partition: "aws",
				Service:   "iam",
				Region:    "",
				AccountID: "123456789012",
				Resource:  "user/David",
			},
		},
		{
			input: "arn:aws:rds:eu-west-1:123456789012:db:mysql-db",
			arn: ARN{
				Partition: "aws",
				Service:   "rds",
				Region:    "eu-west-1",
				AccountID: "123456789012",
				Resource:  "db:mysql-db",
			},
		},
		{
			input: "arn:aws:s3:::my_corporate_bucket/exampleobject.png",
			arn: ARN{
				Partition: "aws",
				Service:   "s3",
				Region:    "",
				AccountID: "",
				Resource:  "my_corporate_bucket/exampleobject.png",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			spec, err := Parse(tc.input)
			if tc.arn != spec {
				t.Errorf("Expected %q to parse as %v, but got %v", tc.input, tc.arn, spec)
			}
			if err == nil && tc.err != nil {
				t.Errorf("Expected err to be %v, but got nil", tc.err)
			} else if err != nil && tc.err == nil {
				t.Errorf("Expected err to be nil, but got %v", err)
			} else if err != nil && tc.err != nil && err.Error() != tc.err.Error() {
				t.Errorf("Expected err to be %v, but got %v", tc.err, err)
			}
		})
	}
}

func TestIsARN(t *testing.T) {

	cases := map[string]struct {
		In     string
		Expect bool
		// Params
	}{
		"valid ARN slash resource": {
			In:     "arn:aws:service:us-west-2:123456789012:restype/resvalue",
			Expect: true,
		},
		"valid ARN colon resource": {
			In:     "arn:aws:service:us-west-2:123456789012:restype:resvalue",
			Expect: true,
		},
		"valid ARN resource": {
			In:     "arn:aws:service:us-west-2:123456789012:*",
			Expect: true,
		},
		"empty sections": {
			In:     "arn:::::",
			Expect: true,
		},
		"invalid ARN": {
			In: "some random string",
		},
		"invalid ARN missing resource": {
			In: "arn:aws:service:us-west-2:123456789012",
		},
	}

	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			actual := IsARN(c.In)
			if e, a := c.Expect, actual; e != a {
				t.Errorf("expect %s valid %v, got %v", c.In, e, a)
			}
		})
	}
}
