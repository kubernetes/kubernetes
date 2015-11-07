package service

import (
	"math"
	"math/rand"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

// DefaultRetryer implements basic retry logic using exponential backoff for
// most services. If you want to implement custom retry logic, implement the
// request.Retryer interface or create a structure type that composes this
// struct and override the specific methods. For example, to override only
// the MaxRetries method:
//
//		type retryer struct {
//      service.DefaultRetryer
//    }
//
//    // This implementation always has 100 max retries
//    func (d retryer) MaxRetries() uint { return 100 }
type DefaultRetryer struct {
	*Service
}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API request.
func (d DefaultRetryer) MaxRetries() uint {
	if aws.IntValue(d.Service.Config.MaxRetries) < 0 {
		return d.DefaultMaxRetries
	}
	return uint(aws.IntValue(d.Service.Config.MaxRetries))
}

// RetryRules returns the delay duration before retrying this request again
func (d DefaultRetryer) RetryRules(r *request.Request) time.Duration {
	delay := int(math.Pow(2, float64(r.RetryCount))) * (rand.Intn(30) + 30)
	return time.Duration(delay) * time.Millisecond
}

// ShouldRetry returns if the request should be retried.
func (d DefaultRetryer) ShouldRetry(r *request.Request) bool {
	if r.HTTPResponse.StatusCode >= 500 {
		return true
	}
	return r.IsErrorRetryable()
}
