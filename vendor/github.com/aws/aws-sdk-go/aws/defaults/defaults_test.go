package defaults

import (
	"fmt"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/credentials/endpointcreds"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestHTTPCredProvider(t *testing.T) {
	cases := []struct {
		Host string
		Fail bool
	}{
		{"localhost", false}, {"127.0.0.1", false},
		{"www.example.com", true}, {"169.254.170.2", true},
	}

	defer os.Clearenv()

	for i, c := range cases {
		u := fmt.Sprintf("http://%s/abc/123", c.Host)
		os.Setenv(httpProviderEnvVar, u)

		provider := RemoteCredProvider(aws.Config{}, request.Handlers{})
		if provider == nil {
			t.Fatalf("%d, expect provider not to be nil, but was", i)
		}

		if c.Fail {
			creds, err := provider.Retrieve()
			if err == nil {
				t.Fatalf("%d, expect error but got none", i)
			} else {
				aerr := err.(awserr.Error)
				if e, a := "CredentialsEndpointError", aerr.Code(); e != a {
					t.Errorf("%d, expect %s error code, got %s", i, e, a)
				}
			}
			if e, a := endpointcreds.ProviderName, creds.ProviderName; e != a {
				t.Errorf("%d, expect %s provider name got %s", i, e, a)
			}
		} else {
			httpProvider := provider.(*endpointcreds.Provider)
			if e, a := u, httpProvider.Client.Endpoint; e != a {
				t.Errorf("%d, expect %q endpoint, got %q", i, e, a)
			}
		}
	}
}

func TestECSCredProvider(t *testing.T) {
	defer os.Clearenv()
	os.Setenv(ecsCredsProviderEnvVar, "/abc/123")

	provider := RemoteCredProvider(aws.Config{}, request.Handlers{})
	if provider == nil {
		t.Fatalf("expect provider not to be nil, but was")
	}

	httpProvider := provider.(*endpointcreds.Provider)
	if httpProvider == nil {
		t.Fatalf("expect provider not to be nil, but was")
	}
	if e, a := "http://169.254.170.2/abc/123", httpProvider.Client.Endpoint; e != a {
		t.Errorf("expect %q endpoint, got %q", e, a)
	}
}

func TestDefaultEC2RoleProvider(t *testing.T) {
	provider := RemoteCredProvider(aws.Config{}, request.Handlers{})
	if provider == nil {
		t.Fatalf("expect provider not to be nil, but was")
	}

	ec2Provider := provider.(*ec2rolecreds.EC2RoleProvider)
	if ec2Provider == nil {
		t.Fatalf("expect provider not to be nil, but was")
	}
	if e, a := "http://169.254.169.254/latest", ec2Provider.Client.Endpoint; e != a {
		t.Errorf("expect %q endpoint, got %q", e, a)
	}
}
