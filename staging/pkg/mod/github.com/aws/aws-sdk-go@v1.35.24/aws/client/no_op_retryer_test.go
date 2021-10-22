package client

import (
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

func TestNoOpRetryer(t *testing.T) {
	cases := []struct {
		r                request.Request
		expectMaxRetries int
		expectRetryDelay time.Duration
		expectRetry      bool
	}{
		{
			r: request.Request{
				HTTPResponse: &http.Response{StatusCode: 200},
			},
			expectMaxRetries: 0,
			expectRetryDelay: 0,
			expectRetry:      false,
		},
	}

	d := NoOpRetryer{}
	for i, c := range cases {
		maxRetries := d.MaxRetries()
		retry := d.ShouldRetry(&c.r)
		retryDelay := d.RetryRules(&c.r)

		if e, a := c.expectMaxRetries, maxRetries; e != a {
			t.Errorf("%d: expected %v, but received %v for number of max retries", i, e, a)
		}

		if e, a := c.expectRetry, retry; e != a {
			t.Errorf("%d: expected %v, but received %v for should retry", i, e, a)
		}

		if e, a := c.expectRetryDelay, retryDelay; e != a {
			t.Errorf("%d: expected %v, but received %v as retry delay", i, e, a)
		}
	}
}
