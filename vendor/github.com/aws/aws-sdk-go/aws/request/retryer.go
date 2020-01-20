package request

import (
	"net"
	"net/url"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
)

// Retryer provides the interface drive the SDK's request retry behavior. The
// Retryer implementation is responsible for implementing exponential backoff,
// and determine if a request API error should be retried.
//
// client.DefaultRetryer is the SDK's default implementation of the Retryer. It
// uses the which uses the Request.IsErrorRetryable and Request.IsErrorThrottle
// methods to determine if the request is retried.
type Retryer interface {
	// RetryRules return the retry delay that should be used by the SDK before
	// making another request attempt for the failed request.
	RetryRules(*Request) time.Duration

	// ShouldRetry returns if the failed request is retryable.
	//
	// Implementations may consider request attempt count when determining if a
	// request is retryable, but the SDK will use MaxRetries to limit the
	// number of attempts a request are made.
	ShouldRetry(*Request) bool

	// MaxRetries is the number of times a request may be retried before
	// failing.
	MaxRetries() int
}

// WithRetryer sets a Retryer value to the given Config returning the Config
// value for chaining. The value must not be nil.
func WithRetryer(cfg *aws.Config, retryer Retryer) *aws.Config {
	if retryer == nil {
		if cfg.Logger != nil {
			cfg.Logger.Log("ERROR: Request.WithRetryer called with nil retryer. Replacing with retry disabled Retryer.")
		}
		retryer = noOpRetryer{}
	}
	cfg.Retryer = retryer
	return cfg

}

// noOpRetryer is a internal no op retryer used when a request is created
// without a retryer.
//
// Provides a retryer that performs no retries.
// It should be used when we do not want retries to be performed.
type noOpRetryer struct{}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API; For NoOpRetryer the MaxRetries will always be zero.
func (d noOpRetryer) MaxRetries() int {
	return 0
}

// ShouldRetry will always return false for NoOpRetryer, as it should never retry.
func (d noOpRetryer) ShouldRetry(_ *Request) bool {
	return false
}

// RetryRules returns the delay duration before retrying this request again;
// since NoOpRetryer does not retry, RetryRules always returns 0.
func (d noOpRetryer) RetryRules(_ *Request) time.Duration {
	return 0
}

// retryableCodes is a collection of service response codes which are retry-able
// without any further action.
var retryableCodes = map[string]struct{}{
	ErrCodeRequestError:       {},
	"RequestTimeout":          {},
	ErrCodeResponseTimeout:    {},
	"RequestTimeoutException": {}, // Glacier's flavor of RequestTimeout
}

var throttleCodes = map[string]struct{}{
	"ProvisionedThroughputExceededException": {},
	"ThrottledException":                     {}, // SNS, XRay, ResourceGroupsTagging API
	"Throttling":                             {},
	"ThrottlingException":                    {},
	"RequestLimitExceeded":                   {},
	"RequestThrottled":                       {},
	"RequestThrottledException":              {},
	"TooManyRequestsException":               {}, // Lambda functions
	"PriorRequestNotComplete":                {}, // Route53
	"TransactionInProgressException":         {},
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
	ErrCodeSerialization: {},
	ErrCodeRead:          {},
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

	if t, ok := err.(temporary); ok {
		return t.Temporary() || isErrConnectionReset(err)
	}

	return isErrConnectionReset(err)
}

// IsErrorRetryable returns whether the error is retryable, based on its Code.
// Returns false if error is nil.
func IsErrorRetryable(err error) bool {
	if err == nil {
		return false
	}
	return shouldRetryError(err)
}

type temporary interface {
	Temporary() bool
}

func shouldRetryError(origErr error) bool {
	switch err := origErr.(type) {
	case awserr.Error:
		if err.Code() == CanceledErrorCode {
			return false
		}
		if isNestedErrorRetryable(err) {
			return true
		}

		origErr := err.OrigErr()
		var shouldRetry bool
		if origErr != nil {
			shouldRetry = shouldRetryError(origErr)
			if err.Code() == ErrCodeRequestError && !shouldRetry {
				return false
			}
		}
		if isCodeRetryable(err.Code()) {
			return true
		}
		return shouldRetry

	case *url.Error:
		if strings.Contains(err.Error(), "connection refused") {
			// Refused connections should be retried as the service may not yet
			// be running on the port. Go TCP dial considers refused
			// connections as not temporary.
			return true
		}
		// *url.Error only implements Temporary after golang 1.6 but since
		// url.Error only wraps the error:
		return shouldRetryError(err.Err)

	case temporary:
		if netErr, ok := err.(*net.OpError); ok && netErr.Op == "dial" {
			return true
		}
		// If the error is temporary, we want to allow continuation of the
		// retry process
		return err.Temporary() || isErrConnectionReset(origErr)

	case nil:
		// `awserr.Error.OrigErr()` can be nil, meaning there was an error but
		// because we don't know the cause, it is marked as retryable. See
		// TestRequest4xxUnretryable for an example.
		return true

	default:
		switch err.Error() {
		case "net/http: request canceled",
			"net/http: request canceled while waiting for connection":
			// known 1.5 error case when an http request is cancelled
			return false
		}
		// here we don't know the error; so we allow a retry.
		return true
	}
}

// IsErrorThrottle returns whether the error is to be throttled based on its code.
// Returns false if error is nil.
func IsErrorThrottle(err error) bool {
	if aerr, ok := err.(awserr.Error); ok && aerr != nil {
		return isCodeThrottle(aerr.Code())
	}
	return false
}

// IsErrorExpiredCreds returns whether the error code is a credential expiry
// error. Returns false if error is nil.
func IsErrorExpiredCreds(err error) bool {
	if aerr, ok := err.(awserr.Error); ok && aerr != nil {
		return isCodeExpiredCreds(aerr.Code())
	}
	return false
}

// IsErrorRetryable returns whether the error is retryable, based on its Code.
// Returns false if the request has no Error set.
//
// Alias for the utility function IsErrorRetryable
func (r *Request) IsErrorRetryable() bool {
	if isErrCode(r.Error, r.RetryErrorCodes) {
		return true
	}

	// HTTP response status code 501 should not be retried.
	// 501 represents Not Implemented which means the request method is not
	// supported by the server and cannot be handled.
	if r.HTTPResponse != nil {
		// HTTP response status code 500 represents internal server error and
		// should be retried without any throttle.
		if r.HTTPResponse.StatusCode == 500 {
			return true
		}
	}
	return IsErrorRetryable(r.Error)
}

// IsErrorThrottle returns whether the error is to be throttled based on its
// code. Returns false if the request has no Error set.
//
// Alias for the utility function IsErrorThrottle
func (r *Request) IsErrorThrottle() bool {
	if isErrCode(r.Error, r.ThrottleErrorCodes) {
		return true
	}

	if r.HTTPResponse != nil {
		switch r.HTTPResponse.StatusCode {
		case
			429, // error caused due to too many requests
			502, // Bad Gateway error should be throttled
			503, // caused when service is unavailable
			504: // error occurred due to gateway timeout
			return true
		}
	}

	return IsErrorThrottle(r.Error)
}

func isErrCode(err error, codes []string) bool {
	if aerr, ok := err.(awserr.Error); ok && aerr != nil {
		for _, code := range codes {
			if code == aerr.Code() {
				return true
			}
		}
	}

	return false
}

// IsErrorExpired returns whether the error code is a credential expiry error.
// Returns false if the request has no Error set.
//
// Alias for the utility function IsErrorExpiredCreds
func (r *Request) IsErrorExpired() bool {
	return IsErrorExpiredCreds(r.Error)
}
