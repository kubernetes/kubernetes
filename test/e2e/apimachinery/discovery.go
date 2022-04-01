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
	"context"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	clientdiscovery "k8s.io/client-go/discovery"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/utils/crd"

	"github.com/onsi/ginkgo"
	e2eutils "k8s.io/kubernetes/test/e2e/framework/utils"
)

var storageVersionServerVersion = utilversion.MustParseSemantic("v1.13.99")
var _ = SIGDescribe("Discovery", func() {
	f := framework.NewDefaultFramework("discovery")

	var namespaceName string

	ginkgo.BeforeEach(func() {
		namespaceName = f.Namespace.Name

		e2eskipper.SkipUnlessServerVersionGTE(storageVersionServerVersion, f.ClientSet.Discovery())

		ginkgo.By("Setting up server cert")
		setupServerCert(namespaceName, serviceName)
	})

	ginkgo.It("should accurately determine present and missing resources", func() {
		// checks that legacy api group resources function
		ok, err := clientdiscovery.IsResourceEnabled(f.ClientSet.Discovery(), schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"})
		e2eutils.ExpectNoError(err)
		if !ok {
			e2eutils.Failf("namespace.v1 should always be present")
		}
		// checks that non-legacy api group resources function
		ok, err = clientdiscovery.IsResourceEnabled(f.ClientSet.Discovery(), schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"})
		e2eutils.ExpectNoError(err)
		if !ok {
			e2eutils.Failf("deployments.v1.apps should always be present")
		}
		// checks that nonsense resources in existing api groups function
		ok, err = clientdiscovery.IsResourceEnabled(f.ClientSet.Discovery(), schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "please-dont-ever-create-this"})
		e2eutils.ExpectNoError(err)
		if ok {
			e2eutils.Failf("please-dont-ever-create-this.v1.apps should never be present")
		}
		// checks that resources resources in nonsense api groups function
		ok, err = clientdiscovery.IsResourceEnabled(f.ClientSet.Discovery(), schema.GroupVersionResource{Group: "not-these-apps", Version: "v1", Resource: "deployments"})
		e2eutils.ExpectNoError(err)
		if ok {
			e2eutils.Failf("deployments.v1.not-these-apps should never be present")
		}
	})

	ginkgo.It("Custom resource should have storage version hash", func() {
		testcrd, err := crd.CreateTestCRD(f)
		if err != nil {
			return
		}
		defer testcrd.CleanUp()
		spec := testcrd.Crd.Spec
		resources, err := testcrd.APIExtensionClient.Discovery().ServerResourcesForGroupVersion(spec.Group + "/" + spec.Versions[0].Name)
		if err != nil {
			e2eutils.Failf("failed to find the discovery doc for %v: %v", resources, err)
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
					e2eutils.Failf("expected storageVersionHash of %s/%s/%s to be %s, got %s", r.Group, r.Version, r.Name, expected, r.StorageVersionHash)
				}
			}
		}
		if !found {
			e2eutils.Failf("didn't find resource %s in the discovery doc", spec.Names.Plural)
		}
	})

	/*
	   Release : v1.19
	   Testname: Discovery, confirm the PreferredVersion for each api group
	   Description: Ensure that a list of apis is retrieved.
	   Each api group found MUST return a valid PreferredVersion unless the group suffix is example.com.
	*/
	framework.ConformanceIt("should validate PreferredVersion for each APIGroup", func() {

		// get list of APIGroup endpoints
		list := &metav1.APIGroupList{}
		err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/").Do(context.TODO()).Into(list)
		e2eutils.ExpectNoError(err, "Failed to find /apis/")
		e2eutils.ExpectNotEqual(len(list.Groups), 0, "Missing APIGroups")

		for _, group := range list.Groups {
			if strings.HasSuffix(group.Name, ".example.com") {
				// ignore known example dynamic API groups that are added/removed during the e2e test run
				continue
			}
			e2eutils.Logf("Checking APIGroup: %v", group.Name)

			// locate APIGroup endpoint
			checkGroup := &metav1.APIGroup{}
			apiPath := "/apis/" + group.Name + "/"
			err = f.ClientSet.Discovery().RESTClient().Get().AbsPath(apiPath).Do(context.TODO()).Into(checkGroup)
			e2eutils.ExpectNoError(err, "Fail to access: %s", apiPath)
			e2eutils.ExpectNotEqual(len(checkGroup.Versions), 0, "No version found for %v", group.Name)
			e2eutils.Logf("PreferredVersion.GroupVersion: %s", checkGroup.PreferredVersion.GroupVersion)
			e2eutils.Logf("Versions found %v", checkGroup.Versions)

			// confirm that the PreferredVersion is a valid version
			match := false
			for _, version := range checkGroup.Versions {
				if version.GroupVersion == checkGroup.PreferredVersion.GroupVersion {
					e2eutils.Logf("%s matches %s", version.GroupVersion, checkGroup.PreferredVersion.GroupVersion)
					match = true
					break
				}
			}
			e2eutils.ExpectEqual(true, match, "failed to find a valid version for PreferredVersion")
		}
	})
})
