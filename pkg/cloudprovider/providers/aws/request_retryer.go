/*
Copyright 2016 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package aws

import (
	"math/rand"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"

	"github.com/golang/glog"
)

const (
	backoffMinTimeoutMs        float64 = 500
	backoffMultiplier          float64 = 2
	backoffRandomizationFactor float64 = 0.5
)

// throttlingCodes is a collection of service response codes which indicate
// some kind of throttling error.
var throttlingCodes = map[string]struct{}{
	"ProvisionedThroughputExceededException": {},
	"Throttling":                             {},
	"ThrottlingException":                    {},
	"RequestLimitExceeded":                   {},
	"RequestThrottled":                       {},
	"LimitExceededException":                 {}, // Deleting 10+ DynamoDb tables at once
	"TooManyRequestsException":               {}, // Lambda functions
}

// RequestRetryer implements basic retry logic using exponential backoff for
// most services. In case of throttling error, it increases the backoff
// to be at least one second
type RequestRetryer struct {
	NumMaxRetries int
}

// MaxRetries returns the number of maximum returns the service will use to make
// an individual API request.
func (r RequestRetryer) MaxRetries() int {
	return r.NumMaxRetries
}

// RetryRules returns the delay duration before retrying this request again
func (_ RequestRetryer) RetryRules(r *request.Request) time.Duration {
	interval := backoffMinTimeoutMs * backoffMultiplier * float64(r.RetryCount)
	delta := backoffRandomizationFactor * float64(interval)
	minInterval := float64(interval) - delta
	maxInterval := float64(interval) + delta
	// Get a random value from the range [minInterval, maxInterval].
	delay := minInterval + (rand.Float64() * (maxInterval - minInterval + 1))
	if awsError, throttled := isThrottled(r); throttled {
		glog.Warningf("Got %s throttling error on AWS request (%s)",
			awsError.Code(), describeRequest(r))
		delay *= 2
	}
	return time.Duration(delay) * time.Millisecond
}

// ShouldRetry returns if the request should be retried.
func (_ RequestRetryer) ShouldRetry(r *request.Request) bool {
	if r.HTTPResponse.StatusCode >= 500 {
		return true
	}
	return r.IsErrorRetryable()
}

// Return a user-friendly string describing the request, for use in log messages
func describeRequest(r *request.Request) string {
	service := r.ClientInfo.ServiceName

	name := "?"
	if r.Operation != nil {
		name = r.Operation.Name
	}

	return service + "::" + name
}

// Returns the aws error and a boolean indicating if the error is a throttling error
func isThrottled(r *request.Request) (awserr.Error, bool) {
	if r.Error != nil {
		if awsError, ok := r.Error.(awserr.Error); ok {
			_, ok := throttlingCodes[awsError.Code()]
			return awsError, ok
		}
	}
	return nil, false
}
