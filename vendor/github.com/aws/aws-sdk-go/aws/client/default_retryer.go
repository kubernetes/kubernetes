package client

import (
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/internal/sdkrand"
)

// DefaultRetryer implements basic retry logic using exponential backoff for
// most services. If you want to implement custom retry logic, implement the
// request.Retryer interface or create a structure type that composes this
// struct and override the specific methods. For example, to override only
// the MaxRetries method:
//
//   type retryer struct {
//       client.DefaultRetryer
//   }
//
//   // This implementation always has 100 max retries
//   func (d retryer) MaxRetries() int { return 100 }
type DefaultRetryer struct {
	NumMaxRetries int
}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API request.
func (d DefaultRetryer) MaxRetries() int {
	return d.NumMaxRetries
}

// RetryRules returns the delay duration before retrying this request again
func (d DefaultRetryer) RetryRules(r *request.Request) time.Duration {
	// Set the upper limit of delay in retrying at ~five minutes
	var minTime int64 = 30
	var initialDelay time.Duration

	isThrottle := r.IsErrorThrottle()
	if isThrottle {
		if delay, ok := getRetryAfterDelay(r); ok {
			initialDelay = delay
		}

		minTime = 500
	}

	retryCount := r.RetryCount
	if isThrottle && retryCount > 8 {
		retryCount = 8
	} else if retryCount > 12 {
		retryCount = 12
	}

	delay := (1 << uint(retryCount)) * (sdkrand.SeededRand.Int63n(minTime) + minTime)
	return (time.Duration(delay) * time.Millisecond) + initialDelay

}

// ShouldRetry returns true if the request should be retried.
func (d DefaultRetryer) ShouldRetry(r *request.Request) bool {
	// If one of the other handlers already set the retry state
	// we don't want to override it based on the service's state
	if r.Retryable != nil {
		return *r.Retryable
	}

	if r.HTTPResponse.StatusCode >= 500 && r.HTTPResponse.StatusCode != 501 {
		return true
	}

	return r.IsErrorRetryable() || r.IsErrorThrottle()
}

// This will look in the Retry-After header, RFC 7231, for how long
// it will wait before attempting another request
func getRetryAfterDelay(r *request.Request) (time.Duration, bool) {
	if !canUseRetryAfterHeader(r) {
		return 0, false
	}

	delayStr := r.HTTPResponse.Header.Get("Retry-After")
	if len(delayStr) == 0 {
		return 0, false
	}

	delay, err := strconv.Atoi(delayStr)
	if err != nil {
		return 0, false
	}

	return time.Duration(delay) * time.Second, true
}

// Will look at the status code to see if the retry header pertains to
// the status code.
func canUseRetryAfterHeader(r *request.Request) bool {
	switch r.HTTPResponse.StatusCode {
	case 429:
	case 503:
	default:
		return false
	}

	return true
}
