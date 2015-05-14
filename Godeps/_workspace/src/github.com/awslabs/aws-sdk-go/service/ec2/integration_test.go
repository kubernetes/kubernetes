// +build integration

package ec2_test

import (
	"testing"

	"github.com/awslabs/aws-sdk-go/aws"
	"github.com/awslabs/aws-sdk-go/internal/test/integration"
	"github.com/awslabs/aws-sdk-go/internal/util/utilassert"
	"github.com/awslabs/aws-sdk-go/service/ec2"
	"github.com/stretchr/testify/assert"
)

var (
	_ = assert.Equal
	_ = utilassert.Match
	_ = integration.Imported
)

func TestMakingABasicRequest(t *testing.T) {
	client := ec2.New(nil)
	resp, e := client.DescribeRegions(&ec2.DescribeRegionsInput{})
	err := aws.Error(e)
	_, _, _ = resp, e, err // avoid unused warnings

	assert.NoError(t, nil, err)

}

func TestErrorHandling(t *testing.T) {
	client := ec2.New(nil)
	resp, e := client.DescribeInstances(&ec2.DescribeInstancesInput{
		InstanceIDs: []*string{
			aws.String("i-12345678"),
		},
	})
	err := aws.Error(e)
	_, _, _ = resp, e, err // avoid unused warnings

	assert.NotEqual(t, nil, err)
	assert.Equal(t, "InvalidInstanceID.NotFound", err.Code)
	utilassert.Match(t, "The instance ID 'i-12345678' does not exist", err.Message)

}
