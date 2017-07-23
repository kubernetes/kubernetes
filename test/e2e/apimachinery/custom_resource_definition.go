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

package apimachinery

import (
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/testserver"
	utilversion "k8s.io/kubernetes/pkg/util/version"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var crdVersion = utilversion.MustParseSemantic("v1.7.0")

var _ = SIGDescribe("CustomResourceDefinition resources", func() {

	f := framework.NewDefaultFramework("custom-resource-definition")

	Context("Simple CustomResourceDefinition", func() {
		It("creating/deleting custom resource definition objects works [Conformance]", func() {

			framework.SkipUnlessServerVersionGTE(crdVersion, f.ClientSet.Discovery())

			config, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("failed to load config: %v", err)
			}

			apiExtensionClient, err := clientset.NewForConfig(config)
			if err != nil {
				framework.Failf("failed to initialize apiExtensionClient: %v", err)
			}

			randomDefinition := testserver.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)

			//create CRD and waits for the resource to be recognized and available.
			_, err = testserver.CreateNewCustomResourceDefinition(randomDefinition, apiExtensionClient, f.ClientPool)
			if err != nil {
				framework.Failf("failed to create CustomResourceDefinition: %v", err)
			}

			defer func() {
				err = testserver.DeleteCustomResourceDefinition(randomDefinition, apiExtensionClient)
				if err != nil {
					framework.Failf("failed to delete CustomResourceDefinition: %v", err)
				}
			}()
		})
	})
})
