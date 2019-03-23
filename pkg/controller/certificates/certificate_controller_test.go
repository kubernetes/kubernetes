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
	"testing"
	"time"

	certificates "k8s.io/api/certificates/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/controller"
)

// TODO flesh this out to cover things like not being able to find the csr in the cache, not
// auto-approving, etc.
func TestCertificateController(t *testing.T) {

	csr := &certificates.CertificateSigningRequest{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-csr",
		},
	}

	client := fake.NewSimpleClientset(csr)
	informerFactory := informers.NewSharedInformerFactory(fake.NewSimpleClientset(csr), controller.NoResyncPeriodFunc())

	handler := func(csr *certificates.CertificateSigningRequest) error {
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:    certificates.CertificateApproved,
			Reason:  "test reason",
			Message: "test message",
		})
		_, err := client.CertificatesV1beta1().CertificateSigningRequests().UpdateApproval(csr)
		if err != nil {
			return err
		}
		return nil
	}

	controller := NewCertificateController(
		client,
		informerFactory.Certificates().V1beta1().CertificateSigningRequests(),
		handler,
	)
	controller.csrsSynced = func() bool { return true }

	stopCh := make(chan struct{})
	defer close(stopCh)
	informerFactory.Start(stopCh)
	informerFactory.WaitForCacheSync(stopCh)
	wait.PollUntil(10*time.Millisecond, func() (bool, error) {
		return controller.queue.Len() >= 1, nil
	}, stopCh)

	controller.processNextWorkItem()

	actions := client.Actions()
	if len(actions) != 1 {
		t.Errorf("expected 1 actions")
	}
	if a := actions[0]; !a.Matches("update", "certificatesigningrequests") ||
		a.GetSubresource() != "approval" {
		t.Errorf("unexpected action: %#v", a)
	}

}
