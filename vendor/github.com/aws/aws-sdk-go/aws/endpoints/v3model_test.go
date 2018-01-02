package endpoints

import (
	"encoding/json"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUnmarshalRegionRegex(t *testing.T) {
	var input = []byte(`
{
    "regionRegex": "^(us|eu|ap|sa|ca)\\-\\w+\\-\\d+$"
}`)

	p := partition{}
	err := json.Unmarshal(input, &p)
	assert.NoError(t, err)

	expectRegexp, err := regexp.Compile(`^(us|eu|ap|sa|ca)\-\w+\-\d+$`)
	assert.NoError(t, err)

	assert.Equal(t, expectRegexp.String(), p.RegionRegex.Regexp.String())
}

func TestUnmarshalRegion(t *testing.T) {
	var input = []byte(`
{
	"aws-global": {
	  "description": "AWS partition-global endpoint"
	},
	"us-east-1": {
	  "description": "US East (N. Virginia)"
	}
}`)

	rs := regions{}
	err := json.Unmarshal(input, &rs)
	assert.NoError(t, err)

	assert.Len(t, rs, 2)
	r, ok := rs["aws-global"]
	assert.True(t, ok)
	assert.Equal(t, "AWS partition-global endpoint", r.Description)

	r, ok = rs["us-east-1"]
	assert.True(t, ok)
	assert.Equal(t, "US East (N. Virginia)", r.Description)
}

func TestUnmarshalServices(t *testing.T) {
	var input = []byte(`
{
	"acm": {
	  "endpoints": {
		"us-east-1": {}
	  }
	},
	"apigateway": {
      "isRegionalized": true,
	  "endpoints": {
		"us-east-1": {},
        "us-west-2": {}
	  }
	},
	"notRegionalized": {
      "isRegionalized": false,
	  "endpoints": {
		"us-east-1": {},
        "us-west-2": {}
	  }
	}
}`)

	ss := services{}
	err := json.Unmarshal(input, &ss)
	assert.NoError(t, err)

	assert.Len(t, ss, 3)
	s, ok := ss["acm"]
	assert.True(t, ok)
	assert.Len(t, s.Endpoints, 1)
	assert.Equal(t, boxedBoolUnset, s.IsRegionalized)

	s, ok = ss["apigateway"]
	assert.True(t, ok)
	assert.Len(t, s.Endpoints, 2)
	assert.Equal(t, boxedTrue, s.IsRegionalized)

	s, ok = ss["notRegionalized"]
	assert.True(t, ok)
	assert.Len(t, s.Endpoints, 2)
	assert.Equal(t, boxedFalse, s.IsRegionalized)
}

func TestUnmarshalEndpoints(t *testing.T) {
	var inputs = []byte(`
{
	"aws-global": {
	  "hostname": "cloudfront.amazonaws.com",
	  "protocols": [
		"http",
		"https"
	  ],
	  "signatureVersions": [ "v4" ],
	  "credentialScope": {
		"region": "us-east-1",
		"service": "serviceName"
	  },
	  "sslCommonName": "commonName"
	},
	"us-east-1": {}
}`)

	es := endpoints{}
	err := json.Unmarshal(inputs, &es)
	assert.NoError(t, err)

	assert.Len(t, es, 2)
	s, ok := es["aws-global"]
	assert.True(t, ok)
	assert.Equal(t, "cloudfront.amazonaws.com", s.Hostname)
	assert.Equal(t, []string{"http", "https"}, s.Protocols)
	assert.Equal(t, []string{"v4"}, s.SignatureVersions)
	assert.Equal(t, credentialScope{"us-east-1", "serviceName"}, s.CredentialScope)
	assert.Equal(t, "commonName", s.SSLCommonName)
}

func TestEndpointResolve(t *testing.T) {
	defs := []endpoint{
		{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			SignatureVersions: []string{"v2"},
			SSLCommonName:     "sslCommonName",
		},
		{
			Hostname:  "other-hostname",
			Protocols: []string{"http"},
			CredentialScope: credentialScope{
				Region:  "signing_region",
				Service: "signing_service",
			},
		},
	}

	e := endpoint{
		Hostname:          "{service}.{region}.{dnsSuffix}",
		Protocols:         []string{"http", "https"},
		SignatureVersions: []string{"v4"},
		SSLCommonName:     "new sslCommonName",
	}

	resolved := e.resolve("service", "region", "dnsSuffix",
		defs, Options{},
	)

	assert.Equal(t, "https://service.region.dnsSuffix", resolved.URL)
	assert.Equal(t, "signing_service", resolved.SigningName)
	assert.Equal(t, "signing_region", resolved.SigningRegion)
	assert.Equal(t, "v4", resolved.SigningMethod)
}

func TestEndpointMergeIn(t *testing.T) {
	expected := endpoint{
		Hostname:          "other hostname",
		Protocols:         []string{"http"},
		SignatureVersions: []string{"v4"},
		SSLCommonName:     "ssl common name",
		CredentialScope: credentialScope{
			Region:  "region",
			Service: "service",
		},
	}

	actual := endpoint{}
	actual.mergeIn(endpoint{
		Hostname:          "other hostname",
		Protocols:         []string{"http"},
		SignatureVersions: []string{"v4"},
		SSLCommonName:     "ssl common name",
		CredentialScope: credentialScope{
			Region:  "region",
			Service: "service",
		},
	})

	assert.Equal(t, expected, actual)
}

var testPartitions = partitions{
	partition{
		ID:        "part-id",
		Name:      "partitionName",
		DNSSuffix: "amazonaws.com",
		RegionRegex: regionRegex{
			Regexp: func() *regexp.Regexp {
				reg, _ := regexp.Compile("^(us|eu|ap|sa|ca)\\-\\w+\\-\\d+$")
				return reg
			}(),
		},
		Defaults: endpoint{
			Hostname:          "{service}.{region}.{dnsSuffix}",
			Protocols:         []string{"https"},
			SignatureVersions: []string{"v4"},
		},
		Regions: regions{
			"us-east-1": region{
				Description: "region description",
			},
			"us-west-2": region{},
		},
		Services: services{
			"s3": service{},
			"service1": service{
				Endpoints: endpoints{
					"us-east-1": {},
					"us-west-2": {
						HasDualStack:      boxedTrue,
						DualStackHostname: "{service}.dualstack.{region}.{dnsSuffix}",
					},
				},
			},
			"service2": service{},
			"httpService": service{
				Defaults: endpoint{
					Protocols: []string{"http"},
				},
			},
			"globalService": service{
				IsRegionalized:    boxedFalse,
				PartitionEndpoint: "aws-global",
				Endpoints: endpoints{
					"aws-global": endpoint{
						CredentialScope: credentialScope{
							Region: "us-east-1",
						},
						Hostname: "globalService.amazonaws.com",
					},
				},
			},
		},
	},
}

func TestResolveEndpoint(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("service2", "us-west-2")

	assert.NoError(t, err)
	assert.Equal(t, "https://service2.us-west-2.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-west-2", resolved.SigningRegion)
	assert.Equal(t, "service2", resolved.SigningName)
}

func TestResolveEndpoint_DisableSSL(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("service2", "us-west-2", DisableSSLOption)

	assert.NoError(t, err)
	assert.Equal(t, "http://service2.us-west-2.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-west-2", resolved.SigningRegion)
	assert.Equal(t, "service2", resolved.SigningName)
}

func TestResolveEndpoint_UseDualStack(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("service1", "us-west-2", UseDualStackOption)

	assert.NoError(t, err)
	assert.Equal(t, "https://service1.dualstack.us-west-2.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-west-2", resolved.SigningRegion)
	assert.Equal(t, "service1", resolved.SigningName)
}

func TestResolveEndpoint_HTTPProtocol(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("httpService", "us-west-2")

	assert.NoError(t, err)
	assert.Equal(t, "http://httpService.us-west-2.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-west-2", resolved.SigningRegion)
	assert.Equal(t, "httpService", resolved.SigningName)
}

func TestResolveEndpoint_UnknownService(t *testing.T) {
	_, err := testPartitions.EndpointFor("unknownservice", "us-west-2")

	assert.Error(t, err)

	_, ok := err.(UnknownServiceError)
	assert.True(t, ok, "expect error to be UnknownServiceError")
}

func TestResolveEndpoint_ResolveUnknownService(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("unknown-service", "us-region-1",
		ResolveUnknownServiceOption)

	assert.NoError(t, err)

	assert.Equal(t, "https://unknown-service.us-region-1.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-region-1", resolved.SigningRegion)
	assert.Equal(t, "unknown-service", resolved.SigningName)
}

func TestResolveEndpoint_UnknownMatchedRegion(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("service2", "us-region-1")

	assert.NoError(t, err)
	assert.Equal(t, "https://service2.us-region-1.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-region-1", resolved.SigningRegion)
	assert.Equal(t, "service2", resolved.SigningName)
}

func TestResolveEndpoint_UnknownRegion(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("service2", "unknownregion")

	assert.NoError(t, err)
	assert.Equal(t, "https://service2.unknownregion.amazonaws.com", resolved.URL)
	assert.Equal(t, "unknownregion", resolved.SigningRegion)
	assert.Equal(t, "service2", resolved.SigningName)
}

func TestResolveEndpoint_StrictPartitionUnknownEndpoint(t *testing.T) {
	_, err := testPartitions[0].EndpointFor("service2", "unknownregion", StrictMatchingOption)

	assert.Error(t, err)

	_, ok := err.(UnknownEndpointError)
	assert.True(t, ok, "expect error to be UnknownEndpointError")
}

func TestResolveEndpoint_StrictPartitionsUnknownEndpoint(t *testing.T) {
	_, err := testPartitions.EndpointFor("service2", "us-region-1", StrictMatchingOption)

	assert.Error(t, err)

	_, ok := err.(UnknownEndpointError)
	assert.True(t, ok, "expect error to be UnknownEndpointError")
}

func TestResolveEndpoint_NotRegionalized(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("globalService", "us-west-2")

	assert.NoError(t, err)
	assert.Equal(t, "https://globalService.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-east-1", resolved.SigningRegion)
	assert.Equal(t, "globalService", resolved.SigningName)
}

func TestResolveEndpoint_AwsGlobal(t *testing.T) {
	resolved, err := testPartitions.EndpointFor("globalService", "aws-global")

	assert.NoError(t, err)
	assert.Equal(t, "https://globalService.amazonaws.com", resolved.URL)
	assert.Equal(t, "us-east-1", resolved.SigningRegion)
	assert.Equal(t, "globalService", resolved.SigningName)
}
