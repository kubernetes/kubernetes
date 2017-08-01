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

package network

import (
	"fmt"
	"net/http"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/manifest"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// getLoadBalancerControllers returns a list of LBCtesters.
func getLoadBalancerControllers(client clientset.Interface) []LBCTester {
	return []LBCTester{
		&haproxyControllerTester{
			name:   "haproxy",
			cfg:    "test/e2e/testing-manifests/serviceloadbalancer/haproxyrc.yaml",
			client: client,
		},
	}
}

// getIngManagers returns a list of ingManagers.
func getIngManagers(client clientset.Interface) []*ingManager {
	return []*ingManager{
		{
			name:        "netexec",
			rcCfgPaths:  []string{"test/e2e/testing-manifests/serviceloadbalancer/netexecrc.yaml"},
			svcCfgPaths: []string{"test/e2e/testing-manifests/serviceloadbalancer/netexecsvc.yaml"},
			svcNames:    []string{},
			client:      client,
		},
	}
}

// LBCTester is an interface used to test loadbalancer controllers.
type LBCTester interface {
	// start starts the loadbalancer controller in the given namespace
	start(namespace string) error
	// lookup returns the address (ip/hostname) associated with ingressKey
	lookup(ingressKey string) string
	// stop stops the loadbalancer controller
	stop() error
	// name returns the name of the loadbalancer
	getName() string
}

// haproxyControllerTester implements LBCTester for bare metal haproxy LBs.
type haproxyControllerTester struct {
	client      clientset.Interface
	cfg         string
	rcName      string
	rcNamespace string
	name        string
	address     []string
}

func (h *haproxyControllerTester) getName() string {
	return h.name
}

func (h *haproxyControllerTester) start(namespace string) (err error) {

	// Create a replication controller with the given configuration.
	framework.Logf("Parsing rc from %v", h.cfg)
	rc, err := manifest.RcFromManifest(h.cfg)
	Expect(err).NotTo(HaveOccurred())
	rc.Namespace = namespace
	rc.Spec.Template.Labels["name"] = rc.Name

	// Add the --namespace arg.
	// TODO: Remove this when we have proper namespace support.
	for i, c := range rc.Spec.Template.Spec.Containers {
		rc.Spec.Template.Spec.Containers[i].Args = append(
			c.Args, fmt.Sprintf("--namespace=%v", namespace))
		framework.Logf("Container args %+v", rc.Spec.Template.Spec.Containers[i].Args)
	}

	rc, err = h.client.Core().ReplicationControllers(rc.Namespace).Create(rc)
	if err != nil {
		return
	}
	if err = framework.WaitForControlledPodsRunning(h.client, namespace, rc.Name, api.Kind("ReplicationController")); err != nil {
		return
	}
	h.rcName = rc.Name
	h.rcNamespace = rc.Namespace

	// Find the pods of the rc we just created.
	labelSelector := labels.SelectorFromSet(
		labels.Set(map[string]string{"name": h.rcName}))
	options := metav1.ListOptions{LabelSelector: labelSelector.String()}
	pods, err := h.client.Core().Pods(h.rcNamespace).List(options)
	if err != nil {
		return err
	}

	// Find the external addresses of the nodes the pods are running on.
	for _, p := range pods.Items {
		wait.Poll(1*time.Second, framework.ServiceRespondingTimeout, func() (bool, error) {
			address, err := framework.GetHostExternalAddress(h.client, &p)
			if err != nil {
				framework.Logf("%v", err)
				return false, nil
			}
			h.address = append(h.address, address)
			return true, nil
		})
	}
	if len(h.address) == 0 {
		return fmt.Errorf("No external ips found for loadbalancer %v", h.getName())
	}
	return nil
}

func (h *haproxyControllerTester) stop() error {
	return h.client.Core().ReplicationControllers(h.rcNamespace).Delete(h.rcName, nil)
}

func (h *haproxyControllerTester) lookup(ingressKey string) string {
	// The address of a service is the address of the lb/servicename, currently.
	return fmt.Sprintf("http://%v/%v", h.address[0], ingressKey)
}

// ingManager starts an rc and the associated service.
type ingManager struct {
	rcCfgPaths  []string
	svcCfgPaths []string
	ingCfgPath  string
	name        string
	namespace   string
	client      clientset.Interface
	svcNames    []string
}

func (s *ingManager) getName() string {
	return s.name
}

func (s *ingManager) start(namespace string) (err error) {
	// Create rcs
	for _, rcPath := range s.rcCfgPaths {
		framework.Logf("Parsing rc from %v", rcPath)
		var rc *v1.ReplicationController
		rc, err = manifest.RcFromManifest(rcPath)
		Expect(err).NotTo(HaveOccurred())
		rc.Namespace = namespace
		rc.Spec.Template.Labels["name"] = rc.Name
		rc, err = s.client.Core().ReplicationControllers(rc.Namespace).Create(rc)
		if err != nil {
			return
		}
		if err = framework.WaitForControlledPodsRunning(s.client, rc.Namespace, rc.Name, api.Kind("ReplicationController")); err != nil {
			return
		}
	}
	// Create services.
	// Note that it's up to the caller to make sure the service actually matches
	// the pods of the rc.
	for _, svcPath := range s.svcCfgPaths {
		framework.Logf("Parsing service from %v", svcPath)
		var svc *v1.Service
		svc, err = manifest.SvcFromManifest(svcPath)
		Expect(err).NotTo(HaveOccurred())
		svc.Namespace = namespace
		svc, err = s.client.Core().Services(svc.Namespace).Create(svc)
		if err != nil {
			return
		}
		// TODO: This is short term till we have an Ingress.
		s.svcNames = append(s.svcNames, svc.Name)
	}
	s.name = s.svcNames[0]
	s.namespace = namespace
	return nil
}

func (s *ingManager) test(path string) error {
	url := fmt.Sprintf("%v/hostName", path)
	httpClient := &http.Client{}
	return wait.Poll(1*time.Second, framework.ServiceRespondingTimeout, func() (bool, error) {
		body, err := framework.SimpleGET(httpClient, url, "")
		if err != nil {
			framework.Logf("%v\n%v\n%v", url, body, err)
			return false, nil
		}
		return true, nil
	})
}

var _ = SIGDescribe("ServiceLoadBalancer [Feature:ServiceLoadBalancer]", func() {
	// These variables are initialized after framework's beforeEach.
	var ns string
	var client clientset.Interface

	f := framework.NewDefaultFramework("servicelb")

	BeforeEach(func() {
		client = f.ClientSet
		ns = f.Namespace.Name
	})

	It("should support simple GET on Ingress ips", func() {
		for _, t := range getLoadBalancerControllers(client) {
			By(fmt.Sprintf("Starting loadbalancer controller %v in namespace %v", t.getName(), ns))
			Expect(t.start(ns)).NotTo(HaveOccurred())

			for _, s := range getIngManagers(client) {
				By(fmt.Sprintf("Starting ingress manager %v in namespace %v", s.getName(), ns))
				Expect(s.start(ns)).NotTo(HaveOccurred())

				for _, sName := range s.svcNames {
					path := t.lookup(sName)
					framework.Logf("Testing path %v", path)
					Expect(s.test(path)).NotTo(HaveOccurred())
				}
			}

			Expect(t.stop()).NotTo(HaveOccurred())
		}
	})
})
