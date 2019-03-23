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

package apimachinery

import (
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/crd"

	. "github.com/onsi/ginkgo"
)

var storageVersionServerVersion = utilversion.MustParseSemantic("v1.13.99")
var _ = SIGDescribe("Discovery", func() {
	f := framework.NewDefaultFramework("discovery")

	var namespaceName string

	BeforeEach(func() {
		namespaceName = f.Namespace.Name

		framework.SkipUnlessServerVersionGTE(storageVersionServerVersion, f.ClientSet.Discovery())

		By("Setting up server cert")
		setupServerCert(namespaceName, serviceName)
	})

	It("[Feature:StorageVersionHash] Custom resource should have storage version hash", func() {
		testcrd, err := crd.CreateTestCRD(f)
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		spec := testcrd.Crd.Spec
		resources, err := testcrd.APIExtensionClient.Discovery().ServerResourcesForGroupVersion(spec.Group + "/" + spec.Versions[0].Name)
		if err != nil {
			framework.Failf("failed to find the discovery doc for %v: %v", resources, err)
		}
		found := false
		var storageVersion string
		for _, v := range spec.Versions {
			if v.Storage {
				storageVersion = v.Name
			}
		}
		// DISCLAIMER: the algorithm of deriving the storageVersionHash
		// is an implementation detail, which shouldn't be relied on by
		// the clients. The following calculation is for test purpose
		// only.
		expected := discovery.StorageVersionHash(spec.Group, storageVersion, spec.Names.Kind)

		for _, r := range resources.APIResources {
			if r.Name == spec.Names.Plural {
				found = true
				if r.StorageVersionHash != expected {
					framework.Failf("expected storageVersionHash of %s/%s/%s to be %s, got %s", r.Group, r.Version, r.Name, expected, r.StorageVersionHash)
				}
			}
		}
		if !found {
			framework.Failf("didn't find resource %s in the discovery doc", spec.Names.Plural)
		}
	})
})
