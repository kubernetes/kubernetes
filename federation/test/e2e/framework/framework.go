/*
Copyright 2015 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"io/ioutil"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	yaml "gopkg.in/yaml.v2"
)

// Framework extends e2e Framework and adds federation specific fields
type Framework struct {
	*framework.Framework

	// To make sure that this framework cleans up after itself, no matter what,
	// we install a Cleanup action before each test and clear it after.  If we
	// should abort, the AfterSuite hook should run all Cleanup actions.
	cleanupHandle framework.CleanupActionHandle

	FederationConfig *restclient.Config

	FederationClientset *federation_clientset.Clientset

	FederationNamespace *v1.Namespace
}

func NewDefaultFederatedFramework(baseName string) *Framework {
	f := &Framework{}

	// Register the federation cleanup before initializing the default
	// e2e framework to ensure it gets called before the default
	// framework's cleanup.
	AfterEach(f.FederationAfterEach)

	f.Framework = framework.NewDefaultFramework(baseName)
	f.Framework.SkipNamespaceCreation = true

	// Register the federation setup after initializing the default
	// e2e framework to ensure it gets called after the default
	// framework's setup.
	BeforeEach(f.FederationBeforeEach)

	return f
}

// FederationBeforeEach checks for federation apiserver is ready and makes a namespace.
func (f *Framework) FederationBeforeEach() {
	// The fact that we need this feels like a bug in ginkgo.
	// https://github.com/onsi/ginkgo/issues/222
	f.cleanupHandle = framework.AddCleanupAction(f.FederationAfterEach)

	if f.FederationConfig == nil {
		By("Reading the federation configuration")
		var err error
		f.FederationConfig, err = LoadFederatedConfig(&clientcmd.ConfigOverrides{})
		Expect(err).NotTo(HaveOccurred())
	}
	if f.FederationClientset == nil {
		By("Creating a release 1.5 federation Clientset")
		var err error
		f.FederationClientset, err = LoadFederationClientset(f.FederationConfig)
		Expect(err).NotTo(HaveOccurred())
	}
	By("Waiting for federation-apiserver to be ready")
	err := WaitForFederationApiserverReady(f.FederationClientset)
	Expect(err).NotTo(HaveOccurred())
	By("federation-apiserver is ready")

	By("Creating a federation namespace")
	ns, err := f.createFederationNamespace(f.BaseName)
	Expect(err).NotTo(HaveOccurred())
	f.FederationNamespace = ns
	By(fmt.Sprintf("Created federation namespace %s", ns.Name))
}

func (f *Framework) deleteFederationNs() {
	ns := f.FederationNamespace
	By(fmt.Sprintf("Destroying federation namespace %q for this suite.", ns.Name))
	timeout := 5 * time.Minute
	if f.NamespaceDeletionTimeout != 0 {
		timeout = f.NamespaceDeletionTimeout
	}

	clientset := f.FederationClientset
	// First delete the namespace from federation apiserver.
	// Also delete the corresponding namespaces from underlying clusters.
	orphanDependents := false
	if err := clientset.Core().Namespaces().Delete(ns.Name, &metav1.DeleteOptions{OrphanDependents: &orphanDependents}); err != nil {
		framework.Failf("Error while deleting federation namespace %s: %s", ns.Name, err)
	}
	// Verify that it got deleted.
	err := wait.PollImmediate(5*time.Second, timeout, func() (bool, error) {
		if _, err := clientset.Core().Namespaces().Get(ns.Name, metav1.GetOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			framework.Logf("Error while waiting for namespace to be terminated: %v", err)
			return false, nil
		}
		return false, nil
	})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			framework.Failf("Couldn't delete ns %q: %s", ns.Name, err)
		} else {
			framework.Logf("Namespace %v was already deleted", ns.Name)
		}
	}
}

// FederationAfterEach deletes the namespace, after reading its events.
func (f *Framework) FederationAfterEach() {
	framework.RemoveCleanupAction(f.cleanupHandle)

	// DeleteNamespace at the very end in defer, to avoid any
	// expectation failures preventing deleting the namespace.
	defer func() {
		// Whether to delete namespace is determined by 3 factors: delete-namespace flag, delete-namespace-on-failure flag and the test result
		// if delete-namespace set to false, namespace will always be preserved.
		// if delete-namespace is true and delete-namespace-on-failure is false, namespace will be preserved if test failed.
		if framework.TestContext.DeleteNamespace && (framework.TestContext.DeleteNamespaceOnFailure || !CurrentGinkgoTestDescription().Failed) {
			// Delete the federation namespace.
			f.deleteFederationNs()
		}

		// Paranoia-- prevent reuse!
		f.FederationNamespace = nil

		if f.FederationClientset == nil {
			framework.Logf("Warning: framework is marked federated, but has no federation 1.5 clientset")
			return
		}
	}()

	// Print events if the test failed.
	if CurrentGinkgoTestDescription().Failed && framework.TestContext.DumpLogsOnFailure {
		// Dump federation events in federation namespace.
		framework.DumpEventsInNamespace(func(opts metav1.ListOptions, ns string) (*v1.EventList, error) {
			return f.FederationClientset.Core().Events(ns).List(opts)
		}, f.FederationNamespace.Name)
	}
}

func (f *Framework) createFederationNamespace(baseName string) (*v1.Namespace, error) {
	clientset := f.FederationClientset
	namespaceObj := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("e2e-tests-%v-", baseName),
		},
	}
	// Be robust about making the namespace creation call.
	var got *v1.Namespace
	if err := wait.PollImmediate(framework.Poll, framework.SingleCallTimeout, func() (bool, error) {
		var err error
		got, err = clientset.Core().Namespaces().Create(namespaceObj)
		if err != nil {
			framework.Logf("Unexpected error while creating namespace: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}
	return got, nil
}

type E2EContext struct {
	// Raw context name,
	RawName string `yaml:"rawName"`
	// A valid dns subdomain which can be used as the name of kubernetes resources.
	Name    string                 `yaml:"name"`
	Cluster *framework.KubeCluster `yaml:"cluster"`
	User    *framework.KubeUser    `yaml:"user"`
}

func (f *Framework) GetUnderlyingFederatedContexts() []E2EContext {
	kubeconfig := framework.KubeConfig{}
	configBytes, err := ioutil.ReadFile(framework.TestContext.KubeConfig)
	framework.ExpectNoError(err)
	err = yaml.Unmarshal(configBytes, &kubeconfig)
	framework.ExpectNoError(err)

	e2eContexts := []E2EContext{}
	for _, context := range kubeconfig.Contexts {
		if strings.HasPrefix(context.Name, "federation") && context.Name != framework.TestContext.FederatedKubeContext {
			user := kubeconfig.FindUser(context.Context.User)
			if user == nil {
				framework.Failf("Could not find user for context %+v", context)
			}

			cluster := kubeconfig.FindCluster(context.Context.Cluster)
			if cluster == nil {
				framework.Failf("Could not find cluster for context %+v", context)
			}

			dnsSubdomainName, err := GetValidDNSSubdomainName(context.Name)
			if err != nil {
				framework.Failf("Could not convert context name %s to a valid dns subdomain name, error: %s", context.Name, err)
			}
			e2eContexts = append(e2eContexts, E2EContext{
				RawName: context.Name,
				Name:    dnsSubdomainName,
				Cluster: cluster,
				User:    user,
			})
		}
	}

	return e2eContexts
}

func (f *Framework) GetRegisteredClusters() ClusterSlice {
	if framework.TestContext.FederationConfigFromCluster {
		return registeredClustersFromSecrets(f)
	} else {
		return registeredClustersFromConfig(f)
	}
}

func (f *Framework) GetClusterClients() []kubeclientset.Interface {
	clusters := f.GetRegisteredClusters()
	var clusterClients []kubeclientset.Interface
	for _, c := range clusters {
		clusterClients = append(clusterClients, c.Clientset)
	}
	return clusterClients
}
