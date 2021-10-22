package ec2

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/awstesting/unit"
)

func TestCustomRetryRules(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})
	req, _ := svc.ModifyNetworkInterfaceAttributeRequest(&ModifyNetworkInterfaceAttributeInput{
		NetworkInterfaceId: aws.String("foo"),
	})

	duration := req.Retryer.RetryRules(req)
	if duration < time.Second*1 || duration > time.Second*2 {
		t.Errorf("expected duration to be between 1s and 2s, but received %v", duration)
	}

	req.RetryCount = 15
	duration = req.Retryer.RetryRules(req)
	if duration < time.Second*4 || duration > time.Second*8 {
		t.Errorf("expected duration to be between 4s and 8s, but received %v", duration)
	}
}

func TestCustomRetryer_WhenRetrierSpecified(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2"),
		Retryer: client.DefaultRetryer{
			NumMaxRetries:    4,
			MinThrottleDelay: 50 * time.Millisecond,
			MinRetryDelay:    10 * time.Millisecond,
			MaxThrottleDelay: 200 * time.Millisecond,
			MaxRetryDelay:    300 * time.Millisecond,
		},
	})

	if _, ok := svc.Client.Retryer.(client.DefaultRetryer); !ok {
		t.Error("expected default retryer, but received otherwise")
	}

	req, _ := svc.AssignPrivateIpAddressesRequest(&AssignPrivateIpAddressesInput{
		NetworkInterfaceId: aws.String("foo"),
	})

	d := req.Retryer.(client.DefaultRetryer)

	if d.NumMaxRetries != 4 {
		t.Errorf("expected max retries to be %v, got %v", 4, d.NumMaxRetries)
	}

	if d.MinRetryDelay != 10*time.Millisecond {
		t.Errorf("expected min retry delay to be %v, got %v", "10 ms", d.MinRetryDelay)
	}

	if d.MinThrottleDelay != 50*time.Millisecond {
		t.Errorf("expected min throttle delay to be %v, got %v", "50 ms", d.MinThrottleDelay)
	}

	if d.MaxRetryDelay != 300*time.Millisecond {
		t.Errorf("expected max retry delay to be %v, got %v", "300 ms", d.MaxRetryDelay)
	}

	if d.MaxThrottleDelay != 200*time.Millisecond {
		t.Errorf("expected max throttle delay to be %v, got %v", "200 ms", d.MaxThrottleDelay)
	}
}

func TestCustomRetryer(t *testing.T) {
	svc := New(unit.Session, &aws.Config{Region: aws.String("us-west-2")})

	req, _ := svc.AssignPrivateIpAddressesRequest(&AssignPrivateIpAddressesInput{
		NetworkInterfaceId: aws.String("foo"),
	})

	d := req.Retryer.(client.DefaultRetryer)

	if d.NumMaxRetries != client.DefaultRetryerMaxNumRetries {
		t.Errorf("expected max retries to be %v, got %v", client.DefaultRetryerMaxNumRetries, d.NumMaxRetries)
	}

	if d.MinRetryDelay != customRetryerMinRetryDelay {
		t.Errorf("expected min retry delay to be %v, got %v", customRetryerMinRetryDelay, d.MinRetryDelay)
	}

	if d.MinThrottleDelay != customRetryerMinRetryDelay {
		t.Errorf("expected min throttle delay to be %v, got %v", customRetryerMinRetryDelay, d.MinThrottleDelay)
	}

	if d.MaxRetryDelay != customRetryerMaxRetryDelay {
		t.Errorf("expected max retry delay to be %v, got %v", customRetryerMaxRetryDelay, d.MaxRetryDelay)
	}

	if d.MaxThrottleDelay != customRetryerMaxRetryDelay {
		t.Errorf("expected max throttle delay to be %v, got %v", customRetryerMaxRetryDelay, d.MaxThrottleDelay)
	}
}
