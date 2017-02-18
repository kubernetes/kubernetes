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
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// credentialType specifies which type(s) of credentials should be used
// from the configuration during the test.
type credentialType int

const (
	credentialTypeNone      credentialType = iota // No credentials
	credentialTypeCert                            // Use certificate credentials
	credentialTypeBasicAuth                       // Use HTTP Basic credentials
	credentialTypeToken                           // Use token credentials
)

// validity specifies whether a credential should be valid or invalid.
type validity int

const (
	invalid validity = iota
	valid
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
			fcs, err := federationClientSet(credentialTypeCert, valid)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc := createServiceOrFail(fcs, nsName, FederatedServiceName)
			deleteServiceOrFail(f.FederationClientset, nsName, svc.Name, nil)
		})

		It("should accept cluster resources when the client has HTTP Basic authentication credentials", func() {
			fcs, err := federationClientSet(credentialTypeBasicAuth, valid)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(err).NotTo(HaveOccurred())
			deleteServiceOrFail(fcs, nsName, svc.Name, nil)
		})

		It("should accept cluster resources when the client has token authentication credentials", func() {
			fcs, err := federationClientSet(credentialTypeToken, valid)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(err).NotTo(HaveOccurred())
			deleteServiceOrFail(fcs, nsName, svc.Name, nil)
		})

		It("should not accept cluster resources when the client has no authentication credentials", func() {
			fcs, err := federationClientSet(credentialTypeNone, valid)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
			if err == nil && svc != nil {
				deleteServiceOrFail(fcs, nsName, svc.Name, nil)
			}
		})

		// TODO: Add a test for invalid certificate credentials. The certificate is validated for
		// correct format, so it cannot contain random noise.

		It("should not accept cluster resources when the client has invalid HTTP Basic authentication credentials", func() {
			fcs, err := federationClientSet(credentialTypeBasicAuth, invalid)
			framework.ExpectNoError(err)

			nsName := f.FederationNamespace.Name
			svc, err := createService(fcs, nsName, FederatedServiceName)
			Expect(errors.IsUnauthorized(err)).To(BeTrue())
			if err == nil && svc != nil {
				deleteServiceOrFail(fcs, nsName, svc.Name, nil)
			}
		})

		It("should not accept cluster resources when the client has invalid token authentication credentials", func() {
			fcs, err := federationClientSet(credentialTypeToken, invalid)
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

// federationClientSet returns a Federation Clientset configured to authenticate
// with the given credential type and validity.
func federationClientSet(authType credentialType, validity validity) (*federation_clientset.Clientset, error) {
	config, err := fedframework.LoadFederatedConfig(&clientcmd.ConfigOverrides{})
	if err != nil {
		return nil, err
	}

	// With no credentials and no valid SSL certificate, the client must attempt
	// to authenticate in insecure mode, or else all requests fail with invalid
	// SSL certificate errors.
	if authType == credentialTypeNone {
		config.Insecure = true
	}

	if authType == credentialTypeCert {
		if validity == invalid {
			framework.Failf("Invalid certificates not implemented.")
		}
	} else {
		config.CAData = []byte{}
		config.CertData = []byte{}
		config.KeyData = []byte{}
	}

	if authType == credentialTypeToken {
		config.Insecure = true
		if validity == invalid {
			config.BearerToken = "invalid"
		}
	} else {
		config.BearerToken = ""
	}

	if authType == credentialTypeBasicAuth {
		config.Insecure = true
		if validity == invalid {
			config.Username = "invalid"
			config.Password = "invalid"
		} else {
			// This is a hacky approach to getting the basic auth credentials, but since
			// the token and the username/password cannot live in the same AuthInfo object,
			// we must dig it out by hand.
			c, err := framework.RestclientConfig(framework.TestContext.FederatedKubeContext)
			if err != nil {
				return nil, err
			}
			for name, authInfo := range c.AuthInfos {
				if strings.HasPrefix(name, framework.TestContext.FederatedKubeContext) && strings.HasSuffix(name, "basic-auth") {
					config.Username = authInfo.Username
					config.Password = authInfo.Password
					break
				}
			}
		}
	} else {
		config.Username = ""
		config.Password = ""
	}

	c, err := federation_clientset.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err)
	}

	return c, nil
}
