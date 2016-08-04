/*
Copyright 2016 The Kubernetes Authors.

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

package app

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"

	certificatefake "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/certificates/unversioned/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/util/config"
)

func TestValueOfAllocatableResources(t *testing.T) {
	testCases := []struct {
		kubeReserved   string
		systemReserved string
		errorExpected  bool
		name           string
	}{
		{
			kubeReserved:   "cpu=200m,memory=-150G",
			systemReserved: "cpu=200m,memory=150G",
			errorExpected:  true,
			name:           "negative quantity value",
		},
		{
			kubeReserved:   "cpu=200m,memory=150GG",
			systemReserved: "cpu=200m,memory=150G",
			errorExpected:  true,
			name:           "invalid quantity unit",
		},
		{
			kubeReserved:   "cpu=200m,memory=15G",
			systemReserved: "cpu=200m,memory=15Ki",
			errorExpected:  false,
			name:           "Valid resource quantity",
		},
	}

	for _, test := range testCases {
		kubeReservedCM := make(config.ConfigurationMap)
		systemReservedCM := make(config.ConfigurationMap)

		kubeReservedCM.Set(test.kubeReserved)
		systemReservedCM.Set(test.systemReserved)

		_, err := parseReservation(kubeReservedCM, systemReservedCM)
		if err != nil {
			t.Logf("%s: error returned: %v", test.name, err)
		}
		if test.errorExpected {
			if err == nil {
				t.Errorf("%s: error expected", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("%s: unexpected error: %v", test.name, err)
			}
		}
	}
}

func TestRequestrCertificate(t *testing.T) {
	var fake core.Fake
	fake.ReactionChain = []core.Reactor{&fakeCreationReactor{&fake}}

	tests := []struct {
		request    []byte
		resultCert []byte
		err        error
	}{
		// Case 0, no errors.
		{
			[]byte("good request"),
			[]byte("good result"),
			nil,
		},

		// Case 1, error creating requests.
		{
			[]byte("error creation request"),
			nil,
			fmt.Errorf("cannot create certificate signing request: error creation"),
		},

		// Case 2, error watching.
		{
			[]byte("error watching request"),
			nil,
			fmt.Errorf("cannot watch on the certificate signing request: error watching"),
		},

		// Case 3, watch channel closed unexpectedly (mock timeout).
		{
			[]byte("timeout request"),
			nil,
			fmt.Errorf("watch channel closed"),
		},

		// Case 4, not approved.
		{
			[]byte("denied request"),
			nil,
			fmt.Errorf("certificate signing request is not approved: no reason, no why"),
		},
	}

	client := &certificatefake.FakeCertificates{&fake}
	for i, tt := range tests {
		testHint := fmt.Sprintf("test case #%d", i)

		cert, err := requestCertificate(client, tt.request, 42)
		assert.Equal(t, tt.err, err, testHint)
		assert.Equal(t, tt.resultCert, cert, testHint)
	}
}

type fakeCreationReactor struct {
	fake *core.Fake
}

func (r *fakeCreationReactor) Handles(action core.Action) bool {
	return true
}

// React takes an csr creation action and add watch reactor according to the request.
func (r *fakeCreationReactor) React(action core.Action) (handled bool, ret runtime.Object, err error) {
	createAction := action.(core.CreateActionImpl)
	csr := createAction.GetObject().(*certificates.CertificateSigningRequest)

	if string(csr.Spec.Request) == "error creation request" {
		return true, nil, fmt.Errorf("error creation")
	}

	r.fake.WatchReactionChain = []core.WatchReactor{&fakeWatchRactor{string(csr.Spec.Request)}}

	return true, nil, nil
}

type fakeWatchRactor struct {
	response string
}

func (r *fakeWatchRactor) Handles(action core.Action) bool {
	return true
}

func (r *fakeWatchRactor) React(action core.Action) (handled bool, ret watch.Interface, err error) {
	if r.response == "error watching request" {
		return true, nil, fmt.Errorf("error watching")
	}
	return true, &watchInterface{r.response}, nil
}

type watchInterface struct {
	response string
}

func (w *watchInterface) Stop() {}

func (w *watchInterface) ResultChan() <-chan watch.Event {
	var event watch.Event

	ch := make(chan watch.Event, 10)

	switch w.response {
	case "good request": // Send a good response.
		event.Type = watch.Modified
		event.Object = &certificates.CertificateSigningRequest{
			Status: certificates.CertificateSigningRequestStatus{
				Conditions: []certificates.CertificateSigningRequestCondition{certificates.CertificateSigningRequestCondition{
					Type: certificates.CertificateApproved,
				}},
				Certificate: []byte("good result"),
			},
		}
		ch <- event
	case "timeout request": // Send an "Added" event, and close the channel.
		event.Type = watch.Added
		event.Object = &certificates.CertificateSigningRequest{}
		ch <- event
	case "denied request": // Send a response with the condition == 'Denied'.
		event.Type = watch.Modified
		event.Object = &certificates.CertificateSigningRequest{
			Status: certificates.CertificateSigningRequestStatus{
				Conditions: []certificates.CertificateSigningRequestCondition{certificates.CertificateSigningRequestCondition{
					Type:    certificates.CertificateDenied,
					Reason:  "no reason",
					Message: "no why",
				}},
				Certificate: []byte("bad result"),
			},
		}
		ch <- event
	}
	close(ch)

	return ch
}
