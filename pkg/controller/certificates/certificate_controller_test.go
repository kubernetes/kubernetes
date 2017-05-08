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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	certificates "k8s.io/kubernetes/pkg/apis/certificates/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
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
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	handler := func(csr *certificates.CertificateSigningRequest) error {
		csr.Status.Conditions = append(csr.Status.Conditions, certificates.CertificateSigningRequestCondition{
			Type:    certificates.CertificateApproved,
			Reason:  "test reason",
			Message: "test message",
		})
		_, err := client.Certificates().CertificateSigningRequests().UpdateApproval(csr)
		if err != nil {
			return err
		}
		return nil
	}

	controller, err := NewCertificateController(
		client,
		informerFactory.Certificates().V1beta1().CertificateSigningRequests(),
		handler,
	)
	if err != nil {
		t.Fatalf("error creating controller: %v", err)
	}
	controller.csrsSynced = func() bool { return true }

	stopCh := make(chan struct{})
	defer close(stopCh)
	go informerFactory.Start(stopCh)

	controller.processNextWorkItem()

	actions := client.Actions()
	if len(actions) != 3 {
		t.Errorf("expected 3 actions")
	}
	if a := actions[0]; !a.Matches("list", "certificatesigningrequests") {
		t.Errorf("unexpected action: %#v", a)
	}
	if a := actions[1]; !a.Matches("watch", "certificatesigningrequests") {
		t.Errorf("unexpected action: %#v", a)
	}
	if a := actions[2]; !a.Matches("update", "certificatesigningrequests") ||
		a.GetSubresource() != "approval" {
		t.Errorf("unexpected action: %#v", a)
	}

}
