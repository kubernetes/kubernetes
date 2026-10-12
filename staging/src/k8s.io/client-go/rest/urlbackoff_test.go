/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"context"
	"errors"
	"net/url"
	"testing"
	"time"

	"github.com/onsi/gomega"

	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/ktesting"
)

func parse(raw string) *url.URL {
	theUrl, _ := url.Parse(raw)
	return theUrl
}

func TestURLBackoffFunctionalityCollisions(t *testing.T) {
	myBackoff := &URLBackoff{
		Backoff: flowcontrol.NewBackOff(1*time.Second, 60*time.Second),
	}

	// Add some noise and make sure backoff for a clean URL is zero.
	myBackoff.UpdateBackoff(parse("http://100.200.300.400:8080"), nil, 500)

	myBackoff.UpdateBackoff(parse("http://1.2.3.4:8080"), nil, 500)

	if myBackoff.CalculateBackoff(parse("http://1.2.3.4:100")) > 0 {
		t.Errorf("URLs are colliding in the backoff map!")
	}
}

// TestURLBackoffFunctionality generally tests the URLBackoff wrapper.  We avoid duplicating tests from backoff and request.
func TestURLBackoffFunctionality(t *testing.T) {
	myBackoff := &URLBackoff{
		Backoff: flowcontrol.NewBackOff(1*time.Second, 60*time.Second),
	}

	// Now test that backoff increases, then recovers.
	// 200 and 300 should both result in clearing the backoff.
	// all others like 429 should result in increased backoff.
	seconds := []int{0,
		1, 2, 4, 8, 0,
		1, 2}
	returnCodes := []int{
		429, 500, 501, 502, 300,
		500, 501, 502,
	}

	if len(seconds) != len(returnCodes) {
		t.Fatalf("responseCode to backoff arrays should be the same length... sanity check failed.")
	}

	for i, sec := range seconds {
		backoffSec := myBackoff.CalculateBackoff(parse("http://1.2.3.4:100"))
		if backoffSec < time.Duration(sec)*time.Second || backoffSec > time.Duration(sec+5)*time.Second {
			t.Errorf("Backoff out of range %v: %v %v", i, sec, backoffSec)
		}
		myBackoff.UpdateBackoff(parse("http://1.2.3.4:100/responseCodeForFuncTest"), nil, returnCodes[i])
	}

	if myBackoff.CalculateBackoff(parse("http://1.2.3.4:100")) == 0 {
		t.Errorf("The final return code %v should have resulted in a backoff ! ", returnCodes[7])
	}
}

// fakeBackoffManager is a hand-written stand-in for BackoffManager that
// records the arguments it receives so the nop-context wrapper's delegation
// can be verified.
type fakeBackoffManager struct {
	sleepCalls           []time.Duration
	calculateBackoffCall *url.URL
	calculateBackoffRet  time.Duration
	updateBackoffURL     *url.URL
	updateBackoffErr     error
	updateBackoffCode    int
}

func (f *fakeBackoffManager) UpdateBackoff(actualURL *url.URL, err error, responseCode int) {
	f.updateBackoffURL = actualURL
	f.updateBackoffErr = err
	f.updateBackoffCode = responseCode
}

func (f *fakeBackoffManager) CalculateBackoff(actualURL *url.URL) time.Duration {
	f.calculateBackoffCall = actualURL
	return f.calculateBackoffRet
}

func (f *fakeBackoffManager) Sleep(d time.Duration) {
	f.sleepCalls = append(f.sleepCalls, d)
}

func TestBackoffManagerNopContext(t *testing.T) {
	tCtx := ktesting.Init(t)

	sleepDuration := 42 * time.Second
	backoffURL := &url.URL{}
	calculateBackoffRet := time.Second
	err := errors.New("fake error")
	responseCode := 404

	mock := &fakeBackoffManager{calculateBackoffRet: calculateBackoffRet}

	ctx := context.Background()
	wrapper := backoffManagerNopContext{BackoffManager: mock}
	wrapper.SleepWithContext(ctx, sleepDuration)
	wrapper.CalculateBackoffWithContext(ctx, backoffURL)
	wrapper.UpdateBackoffWithContext(ctx, backoffURL, err, responseCode)

	tCtx.Assert(mock.sleepCalls).Should(gomega.Equal([]time.Duration{sleepDuration}))
	tCtx.Assert(mock.calculateBackoffCall).Should(gomega.Equal(backoffURL))
	tCtx.Assert(mock.updateBackoffURL).Should(gomega.Equal(backoffURL))
	tCtx.Assert(mock.updateBackoffErr).Should(gomega.Equal(err))
	tCtx.Assert(mock.updateBackoffCode).Should(gomega.Equal(responseCode))
}

func TestNoBackoff(t *testing.T) {
	tCtx := ktesting.Init(t)
	var backoff NoBackoff
	tCtx.Assert(backoff.CalculateBackoff(nil)).Should(gomega.Equal(0 * time.Second))
	tCtx.Assert(backoff.CalculateBackoffWithContext(context.Background(), nil)).Should(gomega.Equal(0 * time.Second))

	start := time.Now()
	backoff.Sleep(0 * time.Second)
	tCtx.Assert(time.Now()).Should(gomega.BeTemporally("~", start, time.Minute /* pretty generous, but we don't want to flake */), "backoff.Sleep")

	// Cancel right away to prevent sleeping.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start = time.Now()
	backoff.SleepWithContext(ctx, 10*time.Minute)
	tCtx.Assert(time.Now()).Should(gomega.BeTemporally("~", start, time.Minute /* pretty generous, but we don't want to flake */), "backoff.SleepWithContext")
}
