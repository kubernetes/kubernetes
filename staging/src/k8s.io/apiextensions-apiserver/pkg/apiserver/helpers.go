/*
Copyright 2018 The Kubernetes Authors.

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

package apiserver

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsinternalversion "k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion"
)

// updateCustomResourceDefinitionStatus updates a CRD's status, retrying up to 5 times on version conflict errors.
// It stops retrying when either the update succeeds or the update func returns true.
func updateCustomResourceDefinitionStatus(client apiextensionsinternalversion.ApiextensionsInterface, name string, update func(definition *apiextensions.CustomResourceDefinition) bool) (*apiextensions.CustomResourceDefinition, error) {
	for i := 0; i < 5; i++ {
		crd, err := client.CustomResourceDefinitions().Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get CustomResourceDefinition %q: %v", name, err)
		}
		if update(crd) {
			return crd, nil
		}
		crd, err = client.CustomResourceDefinitions().UpdateStatus(crd)
		if err == nil {
			return crd, nil
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			return nil, fmt.Errorf("failed to update CustomResourceDefinition %q: %v", name, err)
		}
	}
	return nil, fmt.Errorf("too many retries updating CustomResourceDefinition %q", name)
}

func isStoragedVersion(crd *apiextensions.CustomResourceDefinition, version string) bool {
	for _, v := range crd.Status.StoredVersions {
		if version == v {
			return true
		}
	}
	return false
}
