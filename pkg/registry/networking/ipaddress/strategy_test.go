/*
Copyright 2022 The Kubernetes Authors.

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

package ipaddress

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func newIPAddress() networking.IPAddress {
	return networking.IPAddress{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "192.168.1.1",
			ResourceVersion: "1",
		},
		Spec: networking.IPAddressSpec{
			ParentRef: &networking.ParentReference{
				Group:     "",
				Resource:  "services",
				Name:      "foo",
				Namespace: "bar",
			},
		},
	}
}

func TestIPAddressStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("ipAddress must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ipAddress should not allow create on update")
	}

	ipAddress := newIPAddress()
	Strategy.PrepareForCreate(ctx, &ipAddress)

	errs := Strategy.Validate(ctx, &ipAddress)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from validation for ipAddress: %v", errs)
	}

	newIPAddress := ipAddress.DeepCopy()
	Strategy.PrepareForUpdate(ctx, newIPAddress, &ipAddress)
	errs = Strategy.ValidateUpdate(ctx, newIPAddress, &ipAddress)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from update validation for ipAddress: %v", errs)
	}

	invalidIPAddress := newIPAddress.DeepCopy()
	invalidIPAddress.Name = "invalid/name"
	invalidIPAddress.ResourceVersion = "4"
	errs = Strategy.Validate(ctx, invalidIPAddress)
	if len(errs) == 0 {
		t.Errorf("Expected error from validation for ipAddress, got none")
	}
	errs = Strategy.ValidateUpdate(ctx, invalidIPAddress, &ipAddress)
	if len(errs) == 0 {
		t.Errorf("Expected error from update validation for ipAddress, got none")
	}
	if invalidIPAddress.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}
