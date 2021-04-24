/*
Copyright 2017 The Kubernetes Authors.

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

package certificates

import (
	"context"
	"github.com/stretchr/testify/assert"
	"testing"
	"time"

	certificates "k8s.io/api/certificates/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller"
)

var (
	apprpvedCondition = certificates.CertificateSigningRequestCondition{
		Type:    certificates.CertificateApproved,
		Reason:  "Approved reason",
		Message: "Approved message",
	}
	deniedCondition = certificates.CertificateSigningRequestCondition{
		Type:    certificates.CertificateDenied,
		Reason:  "Denied reason",
		Message: "Denied message",
	}
)

func TestCertificateController(t *testing.T) {
	csr := &certificates.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-csr",
		},
	}
	client := fake.NewSimpleClientset(csr)

	testCases := map[string]struct {
		keyIsDeleted         bool
		handler              func(csr *certificates.CertificateSigningRequest) error
		expectedLatestAction expectedAction
		expectedIsApproved   bool
	}{
		"CSR approved": {
			keyIsDeleted: false,
			handler:      csrHandler(client, apprpvedCondition),
			expectedLatestAction: expectedAction{
				verb:        "update",
				resource:    "certificatesigningrequests",
				subresource: "approval",
			},
			expectedIsApproved: true,
		},
		"CSR is not approved": {
			keyIsDeleted: false,
			handler:      csrHandler(client, deniedCondition),
			expectedLatestAction: expectedAction{
				verb:        "update",
				resource:    "certificatesigningrequests",
				subresource: "approval",
			},
			expectedIsApproved: false,
		},
		"Not being able to find the csr in the cache": {
			keyIsDeleted: true,
			handler:      csrHandler(client, apprpvedCondition),
			expectedLatestAction: expectedAction{
				verb:        "watch",
				resource:    "certificatesigningrequests",
				subresource: "",
			},
		},
	}

	for name, tc := range testCases {
		client.ClearActions()
		fakeWatch := watch.NewFake()
		client.PrependWatchReactor("certificatesigningrequests", ktesting.DefaultWatchReactor(fakeWatch, nil))
		informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

		controller := NewCertificateController(
			"test-controller",
			client,
			informerFactory.Certificates().V1().CertificateSigningRequests(),
			tc.handler,
		)
		controller.csrsSynced = func() bool { return true }

		stopCh := make(chan struct{})
		defer close(stopCh)
		informerFactory.Start(stopCh)
		informerFactory.WaitForCacheSync(stopCh)
		_ = wait.PollUntil(10*time.Millisecond, func() (bool, error) {
			return controller.queue.Len() >= 1, nil
		}, stopCh)

		if tc.keyIsDeleted {
			fakeWatch.Delete(csr)

			// wait until csr is deleted
			err := wait.PollImmediate(10*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
				obj, _ := controller.csrLister.Get("test-csr")
				return obj == nil, nil
			})
			if err != nil {
				t.Fatalf("Error waiting for CertificateSigningRequest: %v", err)
			}
		}

		controller.processNextWorkItem()

		actions := client.Actions()
		latestAction := actions[len(actions)-1]
		if latestAction.GetVerb() == "update" {
			actualCSR := latestAction.(ktesting.UpdateActionImpl).GetObject().(*certificates.CertificateSigningRequest)
			assert.Equal(t, tc.expectedIsApproved, IsCertificateRequestApproved(actualCSR), name+": unexpected RequestConditionType")
		}
		assert.Equal(t, tc.expectedLatestAction.verb, latestAction.GetVerb(), name+": unexpected verb")
		assert.Equal(t, tc.expectedLatestAction.resource, latestAction.GetResource().Resource, name+": unexpected resource")
		assert.Equal(t, tc.expectedLatestAction.subresource, latestAction.GetSubresource(), name+": unexpected subresource")
	}
}

type expectedAction struct {
	verb, resource, subresource string
}

func csrHandler(client *fake.Clientset, condition certificates.CertificateSigningRequestCondition) func(csr *certificates.CertificateSigningRequest) error {
	return func(csr *certificates.CertificateSigningRequest) error {
		csr.Status.Conditions = append(csr.Status.Conditions, condition)
		_, err := client.CertificatesV1().CertificateSigningRequests().UpdateApproval(context.TODO(), csr.Name, csr, metav1.UpdateOptions{})
		if err != nil {
			return err
		}
		return nil
	}
}

func TestIgnorableError(t *testing.T) {
	testCases := []struct {
		name, errMessage string
		expectedMessage  string
		args             []interface{}
	}{
		{
			name:            "A",
			errMessage:      "foo in cache",
			expectedMessage: "foo in cache",
			args:            nil,
		},
		{
			name:            "B",
			errMessage:      "%v has %d words",
			expectedMessage: "foo has 3 words",
			args:            []interface{}{"foo", 3},
		},
	}
	for _, tc := range testCases {
		assert.Equal(t, tc.expectedMessage, IgnorableError(tc.errMessage, tc.args...).Error(), tc.name+": unexpected error message")
	}
}
