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
	"RequestError":            {},
	"RequestTimeout":          {},
	ErrCodeResponseTimeout:    {},
	"RequestTimeoutException": {}, // Glacier's flavor of RequestTimeout
}

var throttleCodes = map[string]struct{}{
	"ProvisionedThroughputExceededException": {},
	"Throttling":                             {},
	"ThrottlingException":                    {},
	"RequestLimitExceeded":                   {},
	"RequestThrottled":                       {},
	"TooManyRequestsException":               {}, // Lambda functions
	"PriorRequestNotComplete":                {}, // Route53
}

// credsExpiredCodes is a collection of error codes which signify the credentials
// need to be refreshed. Expired tokens require refreshing of credentials, and
// resigning before the request can be retried.
var credsExpiredCodes = map[string]struct{}{
	"ExpiredToken":          {},
	"ExpiredTokenException": {},
	"RequestExpired":        {}, // EC2 Only
}

func isCodeThrottle(code string) bool {
	_, ok := throttleCodes[code]
	return ok
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

var validParentCodes = map[string]struct{}{
	ErrCodeSerialization: struct{}{},
	ErrCodeRead:          struct{}{},
}

type temporaryError interface {
	Temporary() bool
}

func isNestedErrorRetryable(parentErr awserr.Error) bool {
	if parentErr == nil {
		return false
	}

	if _, ok := validParentCodes[parentErr.Code()]; !ok {
		return false
	}

	err := parentErr.OrigErr()
	if err == nil {
		return false
	}

	if aerr, ok := err.(awserr.Error); ok {
		return isCodeRetryable(aerr.Code())
	}

	if t, ok := err.(temporaryError); ok {
		return t.Temporary()
	}

	return isErrConnectionReset(err)
}

// IsErrorRetryable returns whether the error is retryable, based on its Code.
// Returns false if error is nil.
func IsErrorRetryable(err error) bool {
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			return isCodeRetryable(aerr.Code()) || isNestedErrorRetryable(aerr)
		}
	}
	return false
}

// IsErrorThrottle returns whether the error is to be throttled based on its code.
// Returns false if error is nil.
func IsErrorThrottle(err error) bool {
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			return isCodeThrottle(aerr.Code())
		}
	}
	return false
}

// IsErrorExpiredCreds returns whether the error code is a credential expiry error.
// Returns false if error is nil.
func IsErrorExpiredCreds(err error) bool {
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			return isCodeExpiredCreds(aerr.Code())
		}
	}
	return false
}

// IsErrorRetryable returns whether the error is retryable, based on its Code.
// Returns false if the request has no Error set.
//
// Alias for the utility function IsErrorRetryable
func (r *Request) IsErrorRetryable() bool {
	return IsErrorRetryable(r.Error)
}

// IsErrorThrottle returns whether the error is to be throttled based on its code.
// Returns false if the request has no Error set
//
// Alias for the utility function IsErrorThrottle
func (r *Request) IsErrorThrottle() bool {
	return IsErrorThrottle(r.Error)
}

// IsErrorExpired returns whether the error code is a credential expiry error.
// Returns false if the request has no Error set.
//
// Alias for the utility function IsErrorExpiredCreds
func (r *Request) IsErrorExpired() bool {
	return IsErrorExpiredCreds(r.Error)
}
