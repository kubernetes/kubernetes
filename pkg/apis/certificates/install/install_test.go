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

package install

import (
	"encoding/json"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/certificates/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestResourceVersioner(t *testing.T) {
	csr := certificates.CertificateSigningRequest{ObjectMeta: api.ObjectMeta{ResourceVersion: "10"}}
	version, err := accessor.ResourceVersion(&csr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}

	csrList := certificates.CertificateSigningRequestList{ListMeta: unversioned.ListMeta{ResourceVersion: "10"}}
	version, err = accessor.ResourceVersion(&csrList)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if version != "10" {
		t.Errorf("unexpected version %v", version)
	}
}

func TestCodec(t *testing.T) {
	csr := certificates.CertificateSigningRequest{}
	// We do want to use package registered rather than testapi here, because we
	// want to test if the package install and package registered work as expected.
	data, err := runtime.Encode(api.Codecs.LegacyCodec(registered.GroupOrDie(certificates.GroupName).GroupVersion), &csr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	other := certificates.CertificateSigningRequest{}
	if err := json.Unmarshal(data, &other); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if other.APIVersion != registered.GroupOrDie(certificates.GroupName).GroupVersion.String() || other.Kind != "CertificateSigningRequest" {
		t.Errorf("unexpected unmarshalled object %#v", other)
	}
}

func TestInterfacesFor(t *testing.T) {
	if _, err := registered.GroupOrDie(certificates.GroupName).InterfacesFor(certificates.SchemeGroupVersion); err == nil {
		t.Fatalf("unexpected non-error: %v", err)
	}
	for i, version := range registered.GroupOrDie(certificates.GroupName).GroupVersions {
		if vi, err := registered.GroupOrDie(certificates.GroupName).InterfacesFor(version); err != nil || vi == nil {
			t.Fatalf("%d: unexpected result: %v", i, err)
		}
	}
}

func TestRESTMapper(t *testing.T) {
	gv := v1alpha1.SchemeGroupVersion
	csrGVK := gv.WithKind("CertificateSigningRequest")

	if gvk, err := registered.GroupOrDie(certificates.GroupName).RESTMapper.KindFor(gv.WithResource("certificatesigningrequests")); err != nil || gvk != csrGVK {
		t.Errorf("unexpected version mapping: %v %v", gvk, err)
	}

	for _, version := range registered.GroupOrDie(certificates.GroupName).GroupVersions {
		mapping, err := registered.GroupOrDie(certificates.GroupName).RESTMapper.RESTMapping(csrGVK.GroupKind(), version.Version)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		if mapping.Resource != "certificatesigningrequests" {
			t.Errorf("incorrect resource name: %#v", mapping)
		}
		if mapping.GroupVersionKind.GroupVersion() != version {
			t.Errorf("incorrect groupVersion: %v", mapping)
		}

		interfaces, _ := registered.GroupOrDie(certificates.GroupName).InterfacesFor(version)
		if mapping.ObjectConvertor != interfaces.ObjectConvertor {
			t.Errorf("unexpected: %#v, expected: %#v", mapping, interfaces)
		}

		csr := &certificates.CertificateSigningRequest{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		name, err := mapping.MetadataAccessor.Name(csr)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if name != "foo" {
			t.Errorf("unable to retrieve object meta with: %v", mapping.MetadataAccessor)
		}
	}
}
