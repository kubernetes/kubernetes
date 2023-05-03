package app

import (
	"fmt"
	"net/http"
	"net/textproto"
	"testing"
)

func TestRejectIfNotReadyHeaderRT(t *testing.T) {
	scenarios := []struct {
		name          string
		eligibleUsers []string
		currentUser   string
		expectHeader  bool
	}{
		{
			name:          "scenario 1: happy path",
			currentUser:   "system:serviceaccount:kube-system:generic-garbage-collector",
			eligibleUsers: []string{"generic-garbage-collector", "namespace-controller"},
			expectHeader:  true,
		},
		{
			name:          "scenario 2: ineligible user",
			currentUser:   "system:serviceaccount:kube-system:service-account-controller",
			eligibleUsers: []string{"generic-garbage-collector", "namespace-controller"},
			expectHeader:  false,
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			// set up the test
			fakeRT := fakeRTFunc(func(r *http.Request) (*http.Response, error) {
				// this is where we validate if the header was set or not
				headerSet := func() bool {
					if len(r.Header.Get("X-OpenShift-Internal-If-Not-Ready")) > 0 {
						return true
					}
					return false
				}()
				if scenario.expectHeader && !headerSet {
					return nil, fmt.Errorf("%v header wasn't set", textproto.CanonicalMIMEHeaderKey("X-OpenShift-Internal-If-Not-Ready"))
				}
				if !scenario.expectHeader && headerSet {
					return nil, fmt.Errorf("didn't expect %v header", textproto.CanonicalMIMEHeaderKey("X-OpenShift-Internal-If-Not-Ready"))
				}
				if scenario.expectHeader {
					if value := r.Header.Get("X-OpenShift-Internal-If-Not-Ready"); value != "reject" {
						return nil, fmt.Errorf("unexpected value %v in the %v header, expected \"reject\"", value, textproto.CanonicalMIMEHeaderKey("X-OpenShift-Internal-If-Not-Ready"))
					}
				}
				return nil, nil
			})
			target := newRejectIfNotReadyHeaderRoundTripper(scenario.eligibleUsers)(fakeRT)
			req, err := http.NewRequest("GET", "", nil)
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("User-Agent", scenario.currentUser)

			// act and validate
			if _, err := target.RoundTrip(req); err != nil {
				t.Fatal(err)
			}
		})
	}
}

type fakeRTFunc func(r *http.Request) (*http.Response, error)

func (rt fakeRTFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return rt(r)
}

func TestMergeCh(t *testing.T) {
	testCases := []struct {
		name    string
		chan1   chan struct{}
		chan2   chan struct{}
		closeFn func(chan struct{}, chan struct{})
	}{
		{
			name:  "chan1 gets closed",
			chan1: make(chan struct{}),
			chan2: make(chan struct{}),
			closeFn: func(a, b chan struct{}) {
				close(a)
			},
		},
		{
			name:  "chan2 gets closed",
			chan1: make(chan struct{}),
			chan2: make(chan struct{}),
			closeFn: func(a, b chan struct{}) {
				close(b)
			},
		},
		{
			name:  "both channels get closed",
			chan1: make(chan struct{}),
			chan2: make(chan struct{}),
			closeFn: func(a, b chan struct{}) {
				close(a)
				close(b)
			},
		},
		{
			name:  "channel receives data and returned channel is closed",
			chan1: make(chan struct{}),
			chan2: make(chan struct{}),
			closeFn: func(a, b chan struct{}) {
				a <- struct{}{}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			go tc.closeFn(tc.chan1, tc.chan2)
			merged := mergeCh(tc.chan1, tc.chan2)
			if _, ok := <-merged; ok {
				t.Fatalf("expected closed channel, got data")
			}
		})
	}
}
