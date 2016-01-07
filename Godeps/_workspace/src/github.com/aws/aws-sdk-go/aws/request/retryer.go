package request

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
)

// Retryer is an interface to control retry logic for a given service.
// The default implementation used by most services is the service.DefaultRetryer
// structure, which contains basic retry logic using exponential backoff.
type Retryer interface {
	RetryRules(*Request) time.Duration
	ShouldRetry(*Request) bool
	MaxRetries() int
}

// WithRetryer sets a config Retryer value to the given Config returning it
// for chaining.
func WithRetryer(cfg *aws.Config, retryer Retryer) *aws.Config {
	cfg.Retryer = retryer
	return cfg
}

// retryableCodes is a collection of service response codes which are retry-able
// without any further action.
var retryableCodes = map[string]struct{}{
	"RequestError":                           {},
	"RequestTimeout":                         {},
	"ProvisionedThroughputExceededException": {},
	"Throttling":                             {},
	"ThrottlingException":                    {},
	"RequestLimitExceeded":                   {},
	"RequestThrottled":                       {},
	"LimitExceededException":                 {}, // Deleting 10+ DynamoDb tables at once
	"TooManyRequestsException":               {}, // Lambda functions
}

// credsExpiredCodes is a collection of error codes which signify the credentials
// need to be refreshed. Expired tokens require refreshing of credentials, and
// resigning before the request can be retried.
var credsExpiredCodes = map[string]struct{}{
	"ExpiredToken":          {},
	"ExpiredTokenException": {},
	"RequestExpired":        {}, // EC2 Only
}

func isCodeRetryable(code string) bool {
	if _, ok := retryableCodes[code]; ok {
		return true
	}

	return isCodeExpiredCreds(code)
}

func isCodeExpiredCreds(code string) bool {
	_, ok := credsExpiredCodes[code]
	return ok
}

// IsErrorRetryable returns whether the error is retryable, based on its Code.
// Returns false if the request has no Error set.
func (r *Request) IsErrorRetryable() bool {
	if r.Error != nil {
		if err, ok := r.Error.(awserr.Error); ok {
			return isCodeRetryable(err.Code())
		}
	}
	return false
}

// IsErrorExpired returns whether the error code is a credential expiry error.
// Returns false if the request has no Error set.
func (r *Request) IsErrorExpired() bool {
	if r.Error != nil {
		if err, ok := r.Error.(awserr.Error); ok {
			return isCodeExpiredCreds(err.Code())
		}
	}
	return false
}
