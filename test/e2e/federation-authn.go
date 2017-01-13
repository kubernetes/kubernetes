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

package e2e

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("[Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federation-apiserver-authn")

	var _ = Describe("Federation API server authentication", func() {
		BeforeEach(func() {
			framework.SkipUnlessFederated(f.ClientSet)
		})

		It("should accept cluster resources when the client has right authentication credentials", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			svc := createServiceOrFail(f.FederationClientset_1_5, nsName, FederatedServiceName)
			deleteServiceOrFail(f.FederationClientset_1_5, nsName, svc.Name, nil)
		})

		It("should not accept cluster resources when the client has invalid authentication credentials", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			contexts := f.GetUnderlyingFederatedContexts()

			// `contexts` is obtained by calling
			// `f.GetUnderlyingFederatedContexts()`. This function in turn
			// checks that the contexts it returns does not include the
			// federation API server context. So `contexts` is guaranteed to
			// contain only the underlying Kubernetes cluster contexts.
			fcs, err := invalidAuthFederationClientSet(contexts[0].User)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
			if err == nil && svc != nil {
				deleteServiceOrFail(fcs, nsName, svc.Name, nil)
			}
		})

		It("should not accept cluster resources when the client has no authentication credentials", func() {
			framework.SkipUnlessFederated(f.ClientSet)

			fcs, err := invalidAuthFederationClientSet(nil)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
			if err == nil && svc != nil {
				deleteServiceOrFail(fcs, nsName, svc.Name, nil)
			}
		})
	})
})

func invalidAuthFederationClientSet(user *framework.KubeUser) (*federation_clientset.Clientset, error) {
	overrides := &clientcmd.ConfigOverrides{}
	if user != nil {
		overrides = &clientcmd.ConfigOverrides{
			AuthInfo: clientcmdapi.AuthInfo{
				Token:    user.User.Token,
				Username: user.User.Username,
				Password: user.User.Password,
			},
		}
	}

	config, err := framework.LoadFederatedConfig(overrides)
	if err != nil {
		return nil, err
	}

	if user == nil {
		config.Password = ""
		config.BearerToken = ""
		config.Username = ""
	}

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}
