package client

import (
	"net/http"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

func TestRetryThrottleStatusCodes(t *testing.T) {
	cases := []struct {
		expectThrottle bool
		expectRetry    bool
		r              request.Request
	}{
		{
			false,
			false,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 200},
			},
		},
		{
			true,
			true,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 429},
			},
		},
		{
			true,
			true,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 502},
			},
		},
		{
			true,
			true,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 503},
			},
		},
		{
			true,
			true,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 504},
			},
		},
		{
			false,
			true,
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 500},
			},
		},
	}

	d := DefaultRetryer{NumMaxRetries: 10}
	for i, c := range cases {
		throttle := c.r.IsErrorThrottle()
		retry := d.ShouldRetry(&c.r)

		if e, a := c.expectThrottle, throttle; e != a {
			t.Errorf("%d: expected %v, but received %v", i, e, a)
		}

		if e, a := c.expectRetry, retry; e != a {
			t.Errorf("%d: expected %v, but received %v", i, e, a)
		}
	}
}

func TestCanUseRetryAfter(t *testing.T) {
	cases := []struct {
		r request.Request
		e bool
	}{
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 200},
			},
			false,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 500},
			},
			false,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 429},
			},
			true,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 503},
			},
			true,
		},
	}

	for i, c := range cases {
		a := canUseRetryAfterHeader(&c.r)
		if c.e != a {
			t.Errorf("%d: expected %v, but received %v", i, c.e, a)
		}
	}
}

func TestGetRetryDelay(t *testing.T) {
	cases := []struct {
		r     request.Request
		e     time.Duration
		equal bool
		ok    bool
	}{
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": []string{"3600"}}},
			},
			3600 * time.Second,
			true,
			true,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 503, Header: http.Header{"Retry-After": []string{"120"}}},
			},
			120 * time.Second,
			true,
			true,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 503, Header: http.Header{"Retry-After": []string{"120"}}},
			},
			1 * time.Second,
			false,
			true,
		},
		{
			request.Request{
				HTTPResponse: &http.Response{StatusCode: 503, Header: http.Header{"Retry-After": []string{""}}},
			},
			0 * time.Second,
			true,
			false,
		},
	}

	for i, c := range cases {
		a, ok := getRetryAfterDelay(&c.r)
		if c.ok != ok {
			t.Errorf("%d: expected %v, but received %v", i, c.ok, ok)
		}

		if (c.e != a) == c.equal {
			t.Errorf("%d: expected %v, but received %v", i, c.e, a)
		}
	}
}

func TestRetryDelay(t *testing.T) {
	d := DefaultRetryer{NumMaxRetries: 100}
	r := request.Request{}
	for i := 0; i < 100; i++ {
		rTemp := r
		rTemp.HTTPResponse = &http.Response{StatusCode: 500, Header: http.Header{"Retry-After": []string{"299"}}}
		rTemp.RetryCount = i
		a := d.RetryRules(&rTemp)
		if a > 5*time.Minute {
			t.Errorf("retry delay should never be greater than five minutes, received %s for retrycount %d", a, i)
		}
	}

	for i := 0; i < 100; i++ {
		rTemp := r
		rTemp.RetryCount = i
		rTemp.HTTPResponse = &http.Response{StatusCode: 503, Header: http.Header{"Retry-After": []string{""}}}
		a := d.RetryRules(&rTemp)
		if a > 5*time.Minute {
			t.Errorf("retry delay should not be greater than five minutes, received %s for retrycount %d", a, i)
		}
	}

	rTemp := r
	rTemp.RetryCount = 1
	rTemp.HTTPResponse = &http.Response{StatusCode: 503, Header: http.Header{"Retry-After": []string{"300"}}}
	a := d.RetryRules(&rTemp)
	if a < 5*time.Minute {
		t.Errorf("retry delay should not be less than retry-after duration, received %s for retrycount %d", a, 1)
	}

}
