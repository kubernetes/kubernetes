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

	"github.com/Azure/go-autorest/autorest/mocks"
	"github.com/stretchr/testify/assert"
)

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
	backoff := &Backoff{Factor: 1.0, Steps: 3}
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
	resp, err := doBackoffRetry(client, fakeRequest, backoff)
	assert.NotNil(t, resp)
	assert.Equal(t, 500, resp.StatusCode)
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 3, client.Attempts())

	// returns immediately on succeed
	r = mocks.NewResponseWithStatus("200 OK", http.StatusOK)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 1)
	resp, err = doBackoffRetry(client, fakeRequest, backoff)
	assert.Nil(t, err)
	assert.Equal(t, 1, client.Attempts())
	assert.NotNil(t, resp)
	assert.Equal(t, 200, resp.StatusCode)

	// returns immediately on throttling
	r = mocks.NewResponseWithStatus("429 TooManyRequests", http.StatusTooManyRequests)
	client = mocks.NewSender()
	client.AppendAndRepeatResponse(r, 1)
	expectedErr = &Error{
		Retriable:      true,
		HTTPStatusCode: 429,
		RawError:       fmt.Errorf("HTTP status code (429)"),
	}
	resp, err = doBackoffRetry(client, fakeRequest, backoff)
	assert.Equal(t, expectedErr.Error(), err)
	assert.Equal(t, 1, client.Attempts())
	assert.NotNil(t, resp)
	assert.Equal(t, 429, resp.StatusCode)
}
