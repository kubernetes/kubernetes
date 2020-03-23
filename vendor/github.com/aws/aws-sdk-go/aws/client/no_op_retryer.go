package client

import (
	"time"

	"github.com/aws/aws-sdk-go/aws/request"
)

// NoOpRetryer provides a retryer that performs no retries.
// It should be used when we do not want retries to be performed.
type NoOpRetryer struct{}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API; For NoOpRetryer the MaxRetries will always be zero.
func (d NoOpRetryer) MaxRetries() int {
	return 0
}

// ShouldRetry will always return false for NoOpRetryer, as it should never retry.
func (d NoOpRetryer) ShouldRetry(_ *request.Request) bool {
	return false
}

// RetryRules returns the delay duration before retrying this request again;
// since NoOpRetryer does not retry, RetryRules always returns 0.
func (d NoOpRetryer) RetryRules(_ *request.Request) time.Duration {
	return 0
}
