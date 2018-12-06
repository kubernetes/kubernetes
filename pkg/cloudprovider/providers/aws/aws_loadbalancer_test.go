/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package aws

import (
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func TestElbProtocolsAreEqual(t *testing.T) {
	grid := []struct {
		L        *string
		R        *string
		Expected bool
	}{
		{
			L:        aws.String("http"),
			R:        aws.String("http"),
			Expected: true,
		},
		{
			L:        aws.String("HTTP"),
			R:        aws.String("http"),
			Expected: true,
		},
		{
			L:        aws.String("HTTP"),
			R:        aws.String("TCP"),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String("TCP"),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String(""),
			Expected: true,
		},
		{
			L:        nil,
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        nil,
			Expected: false,
		},
		{
			L:        nil,
			R:        nil,
			Expected: true,
		},
	}
	for _, g := range grid {
		actual := elbProtocolsAreEqual(g.L, g.R)
		if actual != g.Expected {
			t.Errorf("unexpected result from protocolsEquals(%v, %v)", g.L, g.R)
		}
	}
}

func TestAWSARNEquals(t *testing.T) {
	grid := []struct {
		L        *string
		R        *string
		Expected bool
	}{
		{
			L:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			R:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			Expected: true,
		},
		{
			L:        aws.String("ARN:AWS:ACM:US-EAST-1:123456789012:CERTIFICATE/12345678-1234-1234-1234-123456789012"),
			R:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			Expected: true,
		},
		{
			L:        aws.String("arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"),
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        aws.String(""),
			Expected: true,
		},
		{
			L:        nil,
			R:        aws.String(""),
			Expected: false,
		},
		{
			L:        aws.String(""),
			R:        nil,
			Expected: false,
		},
		{
			L:        nil,
			R:        nil,
			Expected: true,
		},
	}
	for _, g := range grid {
		actual := awsArnEquals(g.L, g.R)
		if actual != g.Expected {
			t.Errorf("unexpected result from awsArnEquals(%v, %v)", g.L, g.R)
		}
	}
}

func TestIsNLB(t *testing.T) {
	tests := []struct {
		name string

		annotations map[string]string
		want        bool
	}{
		{
			"NLB annotation provided",
			map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "nlb"},
			true,
		},
		{
			"NLB annotation has invalid value",
			map[string]string{"service.beta.kubernetes.io/aws-load-balancer-type": "elb"},
			false,
		},
		{
			"NLB annotation absent",
			map[string]string{},
			false,
		},
	}

	for _, test := range tests {
		t.Logf("Running test case %s", test.name)
		got := isNLB(test.annotations)

		if got != test.want {
			t.Errorf("Incorrect value for isNLB() case %s. Got %t, expected %t.", test.name, got, test.want)
		}
	}
}

func TestSecurityGroupFiltering(t *testing.T) {
	grid := []struct {
		in          []*ec2.SecurityGroup
		name        string
		expected    int
		description string
	}{
		{
			in: []*ec2.SecurityGroup{
				{
					IpPermissions: []*ec2.IpPermission{
						{
							IpRanges: []*ec2.IpRange{
								{
									Description: aws.String("an unmanaged"),
								},
							},
						},
					},
				},
			},
			name:        "unmanaged",
			expected:    0,
			description: "An environment without managed LBs should have %d, but found %d SecurityGroups",
		},
		{
			in: []*ec2.SecurityGroup{
				{
					IpPermissions: []*ec2.IpPermission{
						{
							IpRanges: []*ec2.IpRange{
								{
									Description: aws.String("an unmanaged"),
								},
								{
									Description: aws.String(fmt.Sprintf("%s=%s", NLBClientRuleDescription, "managedlb")),
								},
								{
									Description: aws.String(fmt.Sprintf("%s=%s", NLBHealthCheckRuleDescription, "managedlb")),
								},
							},
						},
					},
				},
			},
			name:        "managedlb",
			expected:    1,
			description: "Found %d, but should have %d Security Groups",
		},
	}

	for _, g := range grid {
		actual := len(filterForIPRangeDescription(g.in, g.name))
		if actual != g.expected {
			t.Errorf(g.description, actual, g.expected)
		}
	}

}
