package endpoints

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGlobalEndpoints(t *testing.T) {
	region := "mock-region-1"
	svcs := []string{"cloudfront", "iam", "importexport", "route53", "sts"}

	for _, name := range svcs {
		ep, sr := EndpointForRegion(name, region)
		assert.Equal(t, name+".amazonaws.com", ep)
		assert.Equal(t, "us-east-1", sr)
	}
}

func TestServicesInCN(t *testing.T) {
	region := "cn-north-1"
	svcs := []string{"cloudfront", "iam", "importexport", "route53", "sts", "s3"}

	for _, name := range svcs {
		ep, _ := EndpointForRegion(name, region)
		assert.Equal(t, name+"."+region+".amazonaws.com.cn", ep)
	}
}
