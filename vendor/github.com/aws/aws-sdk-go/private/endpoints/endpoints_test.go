package endpoints_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/private/endpoints"
)

func TestGenericEndpoint(t *testing.T) {
	name := "service"
	region := "mock-region-1"

	ep, sr := endpoints.EndpointForRegion(name, region, false)
	assert.Equal(t, fmt.Sprintf("https://%s.%s.amazonaws.com", name, region), ep)
	assert.Empty(t, sr)
}

func TestGlobalEndpoints(t *testing.T) {
	region := "mock-region-1"
	svcs := []string{"cloudfront", "iam", "importexport", "route53", "sts", "waf"}

	for _, name := range svcs {
		ep, sr := endpoints.EndpointForRegion(name, region, false)
		assert.Equal(t, fmt.Sprintf("https://%s.amazonaws.com", name), ep)
		assert.Equal(t, "us-east-1", sr)
	}
}

func TestServicesInCN(t *testing.T) {
	region := "cn-north-1"
	svcs := []string{"cloudfront", "iam", "importexport", "route53", "sts", "s3", "waf"}

	for _, name := range svcs {
		ep, sr := endpoints.EndpointForRegion(name, region, false)
		assert.Equal(t, fmt.Sprintf("https://%s.%s.amazonaws.com.cn", name, region), ep)
		assert.Empty(t, sr)
	}
}
