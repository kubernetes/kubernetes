// +build !providerless

/*
Copyright 2019 The Kubernetes Authors.

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

package retry

import (
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
	"github.com/stretchr/testify/assert"
)

func TestNewBackoff(t *testing.T) {
	expected := &Backoff{Duration: time.Second, Factor: 2, Steps: 0, Cap: 3 * time.Second, Jitter: 0.5}
	result := NewBackoff(time.Second, 2, 0.5, 0, 3*time.Second)
	assert.Equal(t, expected, result)
}

func TestWithNonRetriableErrors(t *testing.T) {
	bo := &Backoff{Duration: time.Second, Factor: 2, Steps: 0, Cap: 3 * time.Second, Jitter: 0.5}
	errs := []string{"error1", "error2"}
	expected := bo
	expected.NonRetriableErrors = errs
	result := bo.WithNonRetriableErrors(errs)
	assert.Equal(t, expected, result)
}

func TestWithRetriableHTTPStatusCodes(t *testing.T) {
	bo := &Backoff{Duration: time.Second, Factor: 2, Steps: 0, Cap: 3 * time.Second, Jitter: 0.5}
	httpStatusCodes := []int{http.StatusOK, http.StatusTooManyRequests}
	expected := bo
	expected.RetriableHTTPStatusCodes = httpStatusCodes
	result := bo.WithRetriableHTTPStatusCodes(httpStatusCodes)
	assert.Equal(t, expected, result)
}

func TestIsNonRetriableError(t *testing.T) {
	// false case
	bo := &Backoff{Factor: 1.0, Steps: 3}
	ret := bo.isNonRetriableError(nil)
	assert.Equal(t, false, ret)

	// true case
	errs := []string{"error1", "error2"}
	bo2 := bo
	bo2.NonRetriableErrors = errs
	rerr := &Error{
		Retriable:      false,
		HTTPStatusCode: 429,
		RawError:       fmt.Errorf("error1"),
	}

	ret = bo2.isNonRetriableError(rerr)
	assert.Equal(t, true, ret)
}

func TestJitterWithNegativeMaxFactor(t *testing.T) {
	// jitter := duration + time.Duration(rand.Float64()*maxFactor*float64(duration))
	// If maxFactor is 0.0 or less than 0.0, a suggested default value will be chosen.
	// rand.Float64() returns, as a float64, a pseudo-random number in [0.0,1.0).
	duration := time.Duration(time.Second)
	maxFactor := float64(-3.0)
	res := jitter(duration, maxFactor)
	defaultMaxFactor := float64(1.0)
	expected := jitter(duration, defaultMaxFactor)
	assert.Equal(t, expected-res >= time.Duration(0.0*float64(duration)), true)
	assert.Equal(t, expected-res < time.Duration(1.0*float64(duration)), true)
}

func TestDoExponentialBackoffRetry(t *testing.T) {
	client := mocks.NewSender()
	bo := &Backoff{Duration: time.Second, Factor: 2, Steps: 0, Cap: 3 * time.Second, Jitter: 0.5}
	sender := autorest.DecorateSender(
		client,
		DoExponentialBackoffRetry(bo),
	)

	req := &http.Request{
		Method: "GET",
	}

	result, err := sender.Do(req)
	assert.Nil(t, result)
	assert.Nil(t, err)
}

func TestStep(t *testing.T) {
	tests := []struct {
		initial *Backoff
		want    []time.Duration
	}{
		{initial: &Backoff{Duration: time.Second, Steps: 0}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Steps: 1}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 1.0, Steps: 1}, want: []time.Duration{time.Second, time.Second, time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 3}, want: []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 3, Cap: 3 * time.Second}, want: []time.Duration{1 * time.Second, 2 * time.Second, 3 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 2, Cap: 3 * time.Second, Jitter: 0.5}, want: []time.Duration{2 * time.Second, 3 * time.Second, 3 * time.Second}},
		{initial: &Backoff{Duration: time.Second, Factor: 2, Steps: 6, Jitter: 4}, want: []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 8 * time.Second, 16 * time.Second, 32 * time.Second}},
	}
	for seed := int64(0); seed < 5; seed++ {
		for _, tt := range tests {
			initial := *tt.initial
			t.Run(fmt.Sprintf("%#v seed=%d", initial, seed), func(t *testing.T) {
				rand.Seed(seed)
				for i := 0; i < len(tt.want); i++ {
					got := initial.Step()
					t.Logf("[%d]=%s", i, got)
					if initial.Jitter > 0 {
						if got == tt.want[i] {
							// this is statistically unlikely to happen by chance
							t.Errorf("Backoff.Step(%d) = %v, no jitter", i, got)
							continue
						}
						diff := float64(tt.want[i]-got) / float64(tt.want[i])
						if diff > initial.Jitter {
							t.Errorf("Backoff.Step(%d) = %v, want %v, outside range", i, got, tt.want)
							continue
						}
					} else {
						if got != tt.want[i] {
							t.Errorf("Backoff.Step(%d) = %v, want %v", i, got, tt.want)
							continue
						}
					}
				}
			})
		}
	}
}

func TestDoBackoffRetry(t *testing.T) {
	fakeRequest := &http.Request{
		URL: &url.URL{
			Host: "localhost",
			Path: "/api",
		},
	}
	r := mocks.NewResponseWithStatus("500 InternelServerError", http.StatusInternalServerError)
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(r, 3)

	// retries up to steps on errors
	expectedErr := &Error{
		Retriable:      true,
		HTTPStatusCode: 500,
		RawError:       fmt.Errorf("HTTP status code (500)"),
	}
	resp, err := doBackoffRetry(client, fakeRequest, &Backoff{Factor: 1.0, Steps: 3})
	assert.NotNil(t, resp)
	assert.Equal(t, 500, resp.StatusCode)
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 3, client.Attempts())

	// retries with 0 steps
	respSteps0, errSteps0 := doBackoffRetry(client, fakeRequest, &Backoff{Factor: 1.0, Steps: 0})
	assert.Nil(t, respSteps0)
	assert.Nil(t, errSteps0)

	// backoff with NonRetriableErrors and RetriableHTTPStatusCodes
	r = mocks.NewResponseWithStatus("404 StatusNotFound", http.StatusNotFound)
	client = mocks.NewSender()
	client.AppendAndRepeatResponseWithDelay(r, time.Second, 1)
	client.AppendError(fmt.Errorf("HTTP status code (404)"))
	bo := &Backoff{Factor: 1.0, Steps: 3}
	bo.NonRetriableErrors = []string{"404 StatusNotFound"}
	bo.RetriableHTTPStatusCodes = []int{http.StatusNotFound}
	expectedResp := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
		ProtoMinor: 0,
		Body:       mocks.NewBody(""),
		Request:    fakeRequest,
	}

	resp, err = doBackoffRetry(client, fakeRequest, bo)
	assert.Nil(t, err)
	assert.Equal(t, 3, client.Attempts())
	assert.Equal(t, expectedResp, resp)

	// returns immediately on succeed
	r = mocks.NewResponseWithStatus("200 OK", http.StatusOK)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 1)
	resp, err = doBackoffRetry(client, fakeRequest, &Backoff{Factor: 1.0, Steps: 3})
	assert.Nil(t, err)
	assert.Equal(t, 1, client.Attempts())
	assert.NotNil(t, resp)
	assert.Equal(t, 200, resp.StatusCode)

	// returns immediately on throttling
	r = mocks.NewResponseWithStatus("429 TooManyRequests", http.StatusTooManyRequests)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 1)
	expectedErr = &Error{
		Retriable:      false,
		HTTPStatusCode: 429,
		RawError:       fmt.Errorf("HTTP status code (429)"),
	}
	resp, err = doBackoffRetry(client, fakeRequest, &Backoff{Factor: 1.0, Steps: 3})
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 1, client.Attempts())
	assert.NotNil(t, resp)
	assert.Equal(t, 429, resp.StatusCode)

	// don't retry on non retriable error
	r = mocks.NewResponseWithStatus("404 StatusNotFound", http.StatusNotFound)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 1)
	expectedErr = &Error{
		Retriable:      false,
		HTTPStatusCode: 404,
		RawError:       fmt.Errorf("HTTP status code (404)"),
	}
	resp, err = doBackoffRetry(client, fakeRequest, &Backoff{Factor: 1.0, Steps: 3})
	assert.NotNil(t, resp)
	assert.Equal(t, 404, resp.StatusCode)
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 1, client.Attempts())

	// retry on RetriableHTTPStatusCodes
	r = mocks.NewResponseWithStatus("102 StatusProcessing", http.StatusProcessing)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 3)
	expectedErr = &Error{
		Retriable:      true,
		HTTPStatusCode: 102,
		RawError:       fmt.Errorf("HTTP status code (102)"),
	}
	resp, err = doBackoffRetry(client, fakeRequest, &Backoff{
		Factor:                   1.0,
		Steps:                    3,
		RetriableHTTPStatusCodes: []int{http.StatusProcessing},
	})
	assert.NotNil(t, resp)
	assert.Equal(t, 102, resp.StatusCode)
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 3, client.Attempts())
}
