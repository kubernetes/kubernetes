package testing

import (
	"testing"

	"github.com/gophercloud/gophercloud/openstack/utils"
	th "github.com/gophercloud/gophercloud/testhelper"
)

type endpointTestCases struct {
	Endpoint     string
	BaseEndpoint string
}

func TestBaseEndpoint(t *testing.T) {
	tests := []endpointTestCases{
		{
			Endpoint:     "http://example.com:5000/v3",
			BaseEndpoint: "http://example.com:5000/",
		},
		{
			Endpoint:     "http://example.com:5000/v3.6",
			BaseEndpoint: "http://example.com:5000/",
		},
		{
			Endpoint:     "http://example.com:5000/v2.0",
			BaseEndpoint: "http://example.com:5000/",
		},
		{
			Endpoint:     "http://example.com:5000/",
			BaseEndpoint: "http://example.com:5000/",
		},
		{
			Endpoint:     "http://example.com:5000",
			BaseEndpoint: "http://example.com:5000",
		},
		{
			Endpoint:     "http://example.com/identity/v3",
			BaseEndpoint: "http://example.com/identity/",
		},
		{
			Endpoint:     "http://example.com/identity/v3.6",
			BaseEndpoint: "http://example.com/identity/",
		},
		{
			Endpoint:     "http://example.com/identity/v2.0",
			BaseEndpoint: "http://example.com/identity/",
		},
		{
			Endpoint:     "http://example.com/identity/v2.0/projects",
			BaseEndpoint: "http://example.com/identity/",
		},
		{
			Endpoint:     "http://example.com/v2.0/projects",
			BaseEndpoint: "http://example.com/",
		},
		{
			Endpoint:     "http://example.com/identity/",
			BaseEndpoint: "http://example.com/identity/",
		},
		{
			Endpoint:     "http://dev.example.com:5000/v3",
			BaseEndpoint: "http://dev.example.com:5000/",
		},
		{
			Endpoint:     "http://dev.example.com:5000/v3.6",
			BaseEndpoint: "http://dev.example.com:5000/",
		},
		{
			Endpoint:     "http://dev.example.com/identity/",
			BaseEndpoint: "http://dev.example.com/identity/",
		},
		{
			Endpoint:     "http://dev.example.com/identity/v2.0/projects",
			BaseEndpoint: "http://dev.example.com/identity/",
		},
		{
			Endpoint:     "http://dev.example.com/identity/v3.6",
			BaseEndpoint: "http://dev.example.com/identity/",
		},
	}

	for _, test := range tests {
		actual, err := utils.BaseEndpoint(test.Endpoint)
		th.AssertNoErr(t, err)
		th.AssertEquals(t, test.BaseEndpoint, actual)
	}
}
