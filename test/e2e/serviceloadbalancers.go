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

package e2e

import (
	"fmt"
	"io/ioutil"
	"net/http"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/test/e2e/framework"

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
	rc := rcFromManifest(h.cfg)
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
	if err = framework.WaitForRCPodsRunning(h.client, namespace, rc.Name); err != nil {
		return
	}
	h.rcName = rc.Name
	h.rcNamespace = rc.Namespace

	// Find the pods of the rc we just created.
	labelSelector := labels.SelectorFromSet(
		labels.Set(map[string]string{"name": h.rcName}))
	options := api.ListOptions{LabelSelector: labelSelector}
	pods, err := h.client.Core().Pods(h.rcNamespace).List(options)
	if err != nil {
		return err
	}

	// Find the external addresses of the nodes the pods are running on.
	for _, p := range pods.Items {
		wait.Poll(pollInterval, framework.ServiceRespondingTimeout, func() (bool, error) {
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
		rc := rcFromManifest(rcPath)
		rc.Namespace = namespace
		rc.Spec.Template.Labels["name"] = rc.Name
		rc, err = s.client.Core().ReplicationControllers(rc.Namespace).Create(rc)
		if err != nil {
			return
		}
		if err = framework.WaitForRCPodsRunning(s.client, rc.Namespace, rc.Name); err != nil {
			return
		}
	}
	// Create services.
	// Note that it's up to the caller to make sure the service actually matches
	// the pods of the rc.
	for _, svcPath := range s.svcCfgPaths {
		svc := svcFromManifest(svcPath)
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
	return wait.Poll(pollInterval, framework.ServiceRespondingTimeout, func() (bool, error) {
		body, err := simpleGET(httpClient, url, "")
		if err != nil {
			framework.Logf("%v\n%v\n%v", url, body, err)
			return false, nil
		}
		return true, nil
	})
}

var _ = framework.KubeDescribe("ServiceLoadBalancer [Feature:ServiceLoadBalancer]", func() {
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

// simpleGET executes a get on the given url, returns error if non-200 returned.
func simpleGET(c *http.Client, url, host string) (string, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Host = host
	res, err := c.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	rawBody, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", err
	}
	body := string(rawBody)
	if res.StatusCode != http.StatusOK {
		err = fmt.Errorf(
			"GET returned http error %v", res.StatusCode)
	}
	return body, err
}

// rcFromManifest reads a .json/yaml file and returns the rc in it.
func rcFromManifest(fileName string) *api.ReplicationController {
	var controller api.ReplicationController
	framework.Logf("Parsing rc from %v", fileName)
	data := framework.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &controller)).NotTo(HaveOccurred())
	return &controller
}

// svcFromManifest reads a .json/yaml file and returns the rc in it.
func svcFromManifest(fileName string) *api.Service {
	var svc api.Service
	framework.Logf("Parsing service from %v", fileName)
	data := framework.ReadOrDie(fileName)

	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &svc)).NotTo(HaveOccurred())
	return &svc
}
