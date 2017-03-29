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

package e2e_federation

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO: These tests should be integration tests rather than e2e tests, when the
// integration test harness is ready.
var _ = framework.KubeDescribe("[Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-apiserver-authn")

	var _ = Describe("Federation API server authentication [NoCluster]", func() {
		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
		})

		It("should accept cluster resources when the client has certificate authentication credentials", func() {
			fcs, err := federationClientSetWithCert()
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc := createServiceOrFail(fcs, nsName, FederatedServiceName)
			deleteServiceOrFail(f.FederationClientset, nsName, svc.Name, nil)
		})

		It("should accept cluster resources when the client has HTTP Basic authentication credentials", func() {
			fcs, err := federationClientSetWithBasicAuth(true /* valid */)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(err).NotTo(HaveOccurred())
			deleteServiceOrFail(fcs, nsName, svc.Name, nil)
		})

		It("should accept cluster resources when the client has token authentication credentials", func() {
			fcs, err := federationClientSetWithToken(true /* valid */)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(err).NotTo(HaveOccurred())
			deleteServiceOrFail(fcs, nsName, svc.Name, nil)
		})

		It("should not accept cluster resources when the client has no authentication credentials", func() {
			fcs, err := unauthenticatedFederationClientSet()
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			_, err = createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
		})

		// TODO: Add a test for invalid certificate credentials. The certificate is validated for
		// correct format, so it cannot contain random noise.

		It("should not accept cluster resources when the client has invalid HTTP Basic authentication credentials", func() {
			fcs, err := federationClientSetWithBasicAuth(false /* invalid */)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			_, err = createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
		})

		It("should not accept cluster resources when the client has invalid token authentication credentials", func() {
			fcs, err := federationClientSetWithToken(false /* invalid */)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			_, err = createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
		})

	})
})

// unauthenticatedFederationClientSet returns a Federation Clientset configured with
// no authentication credentials.
func unauthenticatedFederationClientSet() (*federation_clientset.Clientset, error) {
	config, err := fedframework.LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}
	config.Insecure = true
	config.CAData = []byte{}
	config.CertData = []byte{}
	config.KeyData = []byte{}
	config.BearerToken = ""

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}

// federationClientSetWithCert returns a Federation Clientset configured with
// certificate authentication credentials.
func federationClientSetWithCert() (*federation_clientset.Clientset, error) {
	config, err := fedframework.LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}

	config.BearerToken = ""

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}

// federationClientSetWithBasicAuth returns a Federation Clientset configured with
// HTTP Basic authentication credentials.
func federationClientSetWithBasicAuth(valid bool) (*federation_clientset.Clientset, error) {
	config, err := fedframework.LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}

	config.Insecure = true
	config.CAData = []byte{}
	config.CertData = []byte{}
	config.KeyData = []byte{}
	config.BearerToken = ""

	if !valid {
		config.Username = ""
		config.Password = ""
	} else {
		// This is a hacky approach to getting the basic auth credentials, but since
		// the token and the username/password cannot live in the same AuthInfo object,
		// and because we do not want to store basic auth credentials with token and
		// certificate credentials for security reasons, we must dig it out by hand.
		c, err := framework.RestclientConfig(framework.TestContext.FederatedKubeContext)
		if err != nil {
			return nil, err
		}
		if authInfo, ok := c.AuthInfos[fmt.Sprintf("%s-basic-auth", framework.TestContext.FederatedKubeContext)]; ok {
			config.Username = authInfo.Username
			config.Password = authInfo.Password
		}
	}

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}

// federationClientSetWithToken returns a Federation Clientset configured with
// token authentication credentials.
func federationClientSetWithToken(valid bool) (*federation_clientset.Clientset, error) {
	config, err := fedframework.LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}
	config.Insecure = true
	config.CAData = []byte{}
	config.CertData = []byte{}
	config.KeyData = []byte{}
	config.Username = ""
	config.Password = ""

	if !valid {
		config.BearerToken = "invalid"
	}

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}
