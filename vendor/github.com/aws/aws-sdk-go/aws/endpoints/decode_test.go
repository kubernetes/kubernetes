package endpoints

import (
	"strings"
	"testing"
)

func TestDecodeEndpoints_V3(t *testing.T) {
	const v3Doc = `
{
  "version": 3,
  "partitions": [
    {
      "defaults": {
        "hostname": "{service}.{region}.{dnsSuffix}",
        "protocols": [
          "https"
        ],
        "signatureVersions": [
          "v4"
        ]
      },
      "dnsSuffix": "amazonaws.com",
      "partition": "aws",
      "partitionName": "AWS Standard",
      "regionRegex": "^(us|eu|ap|sa|ca)\\-\\w+\\-\\d+$",
      "regions": {
        "ap-northeast-1": {
          "description": "Asia Pacific (Tokyo)"
        }
      },
      "services": {
        "acm": {
          "endpoints": {
             "ap-northeast-1": {}
    	  }
        },
        "s3": {
          "endpoints": {
             "ap-northeast-1": {}
    	  }
        }
      }
    }
  ]
}`

	resolver, err := DecodeModel(strings.NewReader(v3Doc))
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	endpoint, err := resolver.EndpointFor("acm", "ap-northeast-1")
	if err != nil {
		t.Fatalf("failed to resolve endpoint, %v", err)
	}

	if a, e := endpoint.URL, "https://acm.ap-northeast-1.amazonaws.com"; a != e {
		t.Errorf("expected %q URL got %q", e, a)
	}

	p := resolver.(partitions)[0]

	s3Defaults := p.Services["s3"].Defaults
	if a, e := s3Defaults.HasDualStack, boxedTrue; a != e {
		t.Errorf("expect s3 service to have dualstack enabled")
	}
	if a, e := s3Defaults.DualStackHostname, "{service}.dualstack.{region}.{dnsSuffix}"; a != e {
		t.Errorf("expect s3 dualstack host pattern to be %q, got %q", e, a)
	}

	ec2metaEndpoint := p.Services["ec2metadata"].Endpoints["aws-global"]
	if a, e := ec2metaEndpoint.Hostname, "169.254.169.254/latest"; a != e {
		t.Errorf("expect ec2metadata host to be %q, got %q", e, a)
	}
}

func TestDecodeEndpoints_NoPartitions(t *testing.T) {
	const doc = `{ "version": 3 }`

	resolver, err := DecodeModel(strings.NewReader(doc))
	if err == nil {
		t.Fatalf("expected error")
	}

	if resolver != nil {
		t.Errorf("expect resolver to be nil")
	}
}

func TestDecodeEndpoints_UnsupportedVersion(t *testing.T) {
	const doc = `{ "version": 2 }`

	resolver, err := DecodeModel(strings.NewReader(doc))
	if err == nil {
		t.Fatalf("expected error decoding model")
	}

	if resolver != nil {
		t.Errorf("expect resolver to be nil")
	}
}

func TestDecodeModelOptionsSet(t *testing.T) {
	var actual DecodeModelOptions
	actual.Set(func(o *DecodeModelOptions) {
		o.SkipCustomizations = true
	})

	expect := DecodeModelOptions{
		SkipCustomizations: true,
	}

	if actual != expect {
		t.Errorf("expect %v options got %v", expect, actual)
	}
}
