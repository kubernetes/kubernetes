//go:build !providerless
// +build !providerless

/*
Copyright 2015 The Kubernetes Authors.

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
	"math"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/request"
	"k8s.io/klog/v2"
)

const (
	decayIntervalSeconds = 20
	decayFraction        = 0.8
	maxDelay             = 60 * time.Second
)

// CrossRequestRetryDelay inserts delays before AWS calls, when we are observing RequestLimitExceeded errors
// Note that we share a CrossRequestRetryDelay across multiple AWS requests; this is a process-wide back-off,
// whereas the aws-sdk-go implements a per-request exponential backoff/retry
type CrossRequestRetryDelay struct {
	backoff Backoff
}

// NewCrossRequestRetryDelay creates a new CrossRequestRetryDelay
func NewCrossRequestRetryDelay() *CrossRequestRetryDelay {
	c := &CrossRequestRetryDelay{}
	c.backoff.init(decayIntervalSeconds, decayFraction, maxDelay)
	return c
}

// BeforeSign is added to the Sign chain; called before each request
func (c *CrossRequestRetryDelay) BeforeSign(r *request.Request) {
	now := time.Now()
	delay := c.backoff.ComputeDelayForRequest(now)
	if delay > 0 {
		klog.Warningf("Inserting delay before AWS request (%s) to avoid RequestLimitExceeded: %s",
			describeRequest(r), delay.String())

		if sleepFn := r.Config.SleepDelay; sleepFn != nil {
			// Support SleepDelay for backwards compatibility
			sleepFn(delay)
		} else if err := aws.SleepWithContext(r.Context(), delay); err != nil {
			r.Error = awserr.New(request.CanceledErrorCode, "request context canceled", err)
			r.Retryable = aws.Bool(false)
			return
		}

		// Avoid clock skew problems
		r.Time = now
	}
}

// Return the operation name, for use in log messages and metrics
func operationName(r *request.Request) string {
	name := "?"
	if r.Operation != nil {
		name = r.Operation.Name
	}
	return name
}

// Return a user-friendly string describing the request, for use in log messages
func describeRequest(r *request.Request) string {
	service := r.ClientInfo.ServiceName
	return service + "::" + operationName(r)
}

// AfterRetry is added to the AfterRetry chain; called after any error
func (c *CrossRequestRetryDelay) AfterRetry(r *request.Request) {
	if r.Error == nil {
		return
	}
	awsError, ok := r.Error.(awserr.Error)
	if !ok {
		return
	}
	if awsError.Code() == "RequestLimitExceeded" {
		c.backoff.ReportError()
		recordAWSThrottlesMetric(operationName(r))
		klog.Warningf("Got RequestLimitExceeded error on AWS request (%s)",
			describeRequest(r))
	}
}

// Backoff manages a backoff that varies based on the recently observed failures
type Backoff struct {
	decayIntervalSeconds int64
	decayFraction        float64
	maxDelay             time.Duration

	mutex sync.Mutex

	// We count all requests & the number of requests which hit a
	// RequestLimit.  We only really care about 'recent' requests, so we
	// decay the counts exponentially to bias towards recent values.
	countErrorsRequestLimit float32
	countRequests           float32
	lastDecay               int64
}

func (b *Backoff) init(decayIntervalSeconds int, decayFraction float64, maxDelay time.Duration) {
	b.lastDecay = time.Now().Unix()
	// Bias so that if the first request hits the limit we don't immediately apply the full delay
	b.countRequests = 4
	b.decayIntervalSeconds = int64(decayIntervalSeconds)
	b.decayFraction = decayFraction
	b.maxDelay = maxDelay
}

// ComputeDelayForRequest computes the delay required for a request, also
// updates internal state to count this request
func (b *Backoff) ComputeDelayForRequest(now time.Time) time.Duration {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	// Apply exponential decay to the counters
	timeDeltaSeconds := now.Unix() - b.lastDecay
	if timeDeltaSeconds > b.decayIntervalSeconds {
		intervals := float64(timeDeltaSeconds) / float64(b.decayIntervalSeconds)
		decay := float32(math.Pow(b.decayFraction, intervals))
		b.countErrorsRequestLimit *= decay
		b.countRequests *= decay
		b.lastDecay = now.Unix()
	}

	// Count this request
	b.countRequests += 1.0

	// Compute the failure rate
	errorFraction := float32(0.0)
	if b.countRequests > 0.5 {
		// Avoid tiny residuals & rounding errors
		errorFraction = b.countErrorsRequestLimit / b.countRequests
	}

	// Ignore a low fraction of errors
	// This also allows them to time-out
	if errorFraction < 0.1 {
		return time.Duration(0)
	}

	// Delay by the max delay multiplied by the recent error rate
	// (i.e. we apply a linear delay function)
	// TODO: This is pretty arbitrary
	delay := time.Nanosecond * time.Duration(float32(b.maxDelay.Nanoseconds())*errorFraction)
	// Round down to the nearest second for sanity
	return time.Second * time.Duration(int(delay.Seconds()))
}

// ReportError is called when we observe a throttling error
func (b *Backoff) ReportError() {
	b.mutex.Lock()
	defer b.mutex.Unlock()

	b.countErrorsRequestLimit += 1.0
}
