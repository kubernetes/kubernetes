// +build go1.8,awsinclude

package plugincreds

import (
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
)

func TestProvider_Passthrough(t *testing.T) {
	p := Provider{
		RetrieveFn: func() (string, string, string, error) {
			return "key", "secret", "token", nil
		},
		IsExpiredFn: func() bool {
			return false
		},
	}

	actual, err := p.Retrieve()
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	expect := credentials.Value{
		AccessKeyID:     "key",
		SecretAccessKey: "secret",
		SessionToken:    "token",
		ProviderName:    ProviderName,
	}
	if expect != actual {
		t.Errorf("expect %+v credentials, got %+v", expect, actual)
	}
}

func TestProvider_Error(t *testing.T) {
	expectErr := fmt.Errorf("expect error")

	p := Provider{
		RetrieveFn: func() (string, string, string, error) {
			return "", "", "", expectErr
		},
		IsExpiredFn: func() bool {
			return false
		},
	}

	actual, err := p.Retrieve()
	if err == nil {
		t.Fatalf("expect error, got none")
	}

	aerr := err.(awserr.Error)
	if e, a := ErrCodePluginProviderRetrieve, aerr.Code(); e != a {
		t.Errorf("expect %s error code, got %s", e, a)
	}

	if e, a := expectErr, aerr.OrigErr(); e != a {
		t.Errorf("expect %v cause error, got %v", e, a)
	}

	expect := credentials.Value{
		ProviderName: ProviderName,
	}
	if expect != actual {
		t.Errorf("expect %+v credentials, got %+v", expect, actual)
	}
}
