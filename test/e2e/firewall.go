/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"net/http"
	//	"strings"
	//	"time"
	//	api "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	//	"k8s.io/kubernetes/pkg/api/latest"
	//	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	//	"k8s.io/kubernetes/pkg/util/wait"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"time"
)

const (
	gceFirewallImageName               = "artfulcoder/gce-firewall:1.1"
	gceFirewallServerPodName           = "gce-firewall-pod"
	gceFirewallServerEndpointPort      = 8080
	gceFirewallLoadBalancerServiceName = "gce-fw-lb-svc"
	gceFirewallNodePortServiceName     = "gce-fw-nodeport-svc"
	gceFirewallClientDaemonName        = "gce-firewall-client"
	firewallDSRetryPeriod              = 2 * time.Second
	firewallDSRetryTimeout             = 1 * time.Minute
)

type gceFirewallTestConfig struct {
	f                   *Framework
	serverPod           *api.Pod
	loadBalancerService *api.Service
	serviceSelector     map[string]string
	nodePort            int
}

var _ = Describe("GCEFirewall", func() {
	//	if !providerIs("gce") {
	//		By("Skipping GCE-specific Test")
	//		return
	//	}
	f := NewFramework("gce-firewall")

	It("should test GCE Firewall", func() {
		By("Creating Test Configuration")
		config := createTestConfiguration(f)

		By("Creating Firewall Service that maps to Firewall Server")
		config.createFirewallServices()

		By("Creating Firewall Server")
		config.createFirewallServer()

		By("Wait for Ingress IP to be assigned")
		config.waitForLoadBalancerIngressSetup()

		By("Hitting Firewall Service from ephemeral public IP, healthcheck, and expecting it to succeed")
		config.hitFirewallServiceWithPublicIP("healthcheck")

		By("Flushing all Firewall Rules")
		config.hitFirewallServiceWithPublicIP("flush")

		By("Creating Firewall DaemonSet")
		config.createFirewallDaemonSet()

		By("Hitting Firewall Service with Node external IP and expecting it to succeed")
		config.hitFirewallServiceWithNodeExternalIP(true)

		By("Hitting Firewall Service with ephemeral Public IP and expecting it to succeed")
		config.hitFirewallServiceWithPublicIP("healthcheck")

		By("Adding all Firewall Rules")
		config.hitFirewallServiceWithPublicIP("start")

		time.Sleep(5 * time.Second)

		By("Hitting Firewall Service with Node External IP and expecting it to fail")
		config.hitFirewallServiceWithNodeExternalIP(false)

		By("Cleanup: Flush Firewall Rules")
		config.hitFirewallServiceWithPublicIP("flush")

		By("Waiting for cleanup")
		time.Sleep(10 * time.Second)
	})
})

func createTestConfiguration(f *Framework) *gceFirewallTestConfig {

	config := &gceFirewallTestConfig{
		f: f,
	}
	selectorName := "selector-" + string(util.NewUUID())
	config.serviceSelector = map[string]string{
		selectorName: "true",
	}
	return config
}

func (config *gceFirewallTestConfig) createFirewallServer() {
	podClient := config.f.Client.Pods(config.f.Namespace.Name)
	podSpec := config.getFirewallServerPodSpec()
	serverPod, err := podClient.Create(podSpec)
	if err != nil {
		Failf("Failed to create %s pod: %v", podSpec.Name, err)
	}
	// wait that all of them are up
	expectNoError(config.f.WaitForPodRunning(podSpec.Name))
	serverPod, err = podClient.Get(podSpec.Name)
	expectNoError(err)
	Expect(serverPod.Spec.NodeName).ToNot(BeEmpty(), "could not find node name on which firewall server is running. pod:%+v", serverPod)
	config.serverPod = serverPod
}

func (config *gceFirewallTestConfig) createFirewallServices() {
	lbServiceSpec := config.getFirewallLoadBalancerServiceSpec()
	config.loadBalancerService = config.createService(lbServiceSpec)
}

func (config *gceFirewallTestConfig) hitFirewallServiceWithNodeExternalIP(expectHitSuccess bool) {
	node := config.getServerNode()
	nodeExternalIP := getNodePublicAddress(node)
	endpoint := fmt.Sprintf("http://%s:%d/healthcheck", nodeExternalIP, gceFirewallServerEndpointPort)
	Logf("Hitting Node Public IP: %q", endpoint)
	hitTest(endpoint, expectHitSuccess)
}

func (config *gceFirewallTestConfig) hitFirewallServiceWithPublicIP(command string) {
	lbIP := config.loadBalancerService.Status.LoadBalancer.Ingress[0].IP
	endpoint := fmt.Sprintf("http://%s:%d/%s", lbIP, gceFirewallServerEndpointPort, command)
	Logf("Hitting Forwarding Rule IP: %q", endpoint)
	hitTest(endpoint, true)
}

func hitTest(endpoint string, expectHitSuccess bool) {
	actualHitSuccess := hitHTTPEndpoint(endpoint, 2)
	if expectHitSuccess != actualHitSuccess {
		Failf("Hit Test to %q failed. expectHitSuccess:%v, actualHitSuccess:%v", endpoint, expectHitSuccess, actualHitSuccess)
	}
}

func (config *gceFirewallTestConfig) createFirewallDaemonSet() {
	privileged := true
	daemonClient := config.f.Client.DaemonSets(config.f.Namespace.Name)
	daemonSetLabel := map[string]string{"daemonsetselector-" + string(util.NewUUID()): gceFirewallClientDaemonName}
	_, err := daemonClient.Create(&extensions.DaemonSet{
		ObjectMeta: api.ObjectMeta{
			Name: gceFirewallClientDaemonName,
		},
		Spec: extensions.DaemonSetSpec{
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: daemonSetLabel,
				},
				Spec: api.PodSpec{
					SecurityContext: &api.PodSecurityContext{
						HostNetwork: true,
					},
					Containers: []api.Container{
						{
							Name:            gceFirewallClientDaemonName,
							Image:           gceFirewallImageName,
							ImagePullPolicy: api.PullAlways,
							SecurityContext: &api.SecurityContext{
								Privileged: &privileged,
							},
							Command: []string{
								"/client",
								"--serviceName=" + gceFirewallLoadBalancerServiceName,
								"--alsologtostderr=true",
							},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())

	By("Check that daemon pods launch on every node of the cluster.")
	err = wait.Poll(firewallDSRetryPeriod, firewallDSRetryTimeout, checkRunningOnAllNodes(config.f, daemonSetLabel))
	Expect(err).NotTo(HaveOccurred(), "error waiting for GCE Firewall daemon pods to start")
	time.Sleep(15 * time.Second)
}

func (config *gceFirewallTestConfig) getFirewallServerPodSpec() *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind: "Pod",
			//			APIVersion: latest.GroupOrDie("").Versions[0],
		},
		ObjectMeta: api.ObjectMeta{
			Name:      gceFirewallServerPodName,
			Namespace: config.f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "firewallserver",
					Image:           gceFirewallImageName,
					ImagePullPolicy: api.PullAlways,
					Command: []string{
						"/server",
						fmt.Sprintf("--port=%d", gceFirewallServerEndpointPort),
						"--alsologtostderr=true",
					},
					Ports: []api.ContainerPort{
						{
							Name:          "http",
							ContainerPort: gceFirewallServerEndpointPort,
							HostPort:      gceFirewallServerEndpointPort,
						},
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "etcssl",
							MountPath: "/etc/ssl",
							ReadOnly:  true,
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "etcssl",
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{
							Path: "/etc/ssl",
						},
					},
				},
			},
		},
	}
	pod.ObjectMeta.Labels = config.serviceSelector
	return pod
}

func (config *gceFirewallTestConfig) getFirewallLoadBalancerServiceSpec() *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: gceFirewallLoadBalancerServiceName,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeLoadBalancer,
			Ports: []api.ServicePort{
				{Port: gceFirewallServerEndpointPort, Name: "http", Protocol: "TCP", TargetPort: intstr.FromInt(gceFirewallServerEndpointPort)},
			},
			Selector: config.serviceSelector,
		},
	}
}

func (config *gceFirewallTestConfig) createService(serviceSpec *api.Service) *api.Service {
	serviceClient := config.f.Client.Services(config.f.Namespace.Name)
	_, err := serviceClient.Create(serviceSpec)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))

	err = waitForService(config.f.Client, config.f.Namespace.Name, serviceSpec.Name, true, 5*time.Second, 45*time.Second)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("error while waiting for service:%s err: %v", serviceSpec.Name, err))

	createdService, err := serviceClient.Get(serviceSpec.Name)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create %s service: %v", serviceSpec.Name, err))
	Logf("Created Service %s: %v", serviceSpec.Name, createdService)
	return createdService
}

// NodeAddresses returns the first address of the given type of each node.
func getNodePublicAddress(node *api.Node) string {
	for _, address := range node.Status.Addresses {
		// Use the first external IP address we find on the node, and
		// use at most one per node.
		if address.Type == api.NodeExternalIP {
			return address.Address
		}
	}
	Fail("Public Address not found for node. Test cannot be run (pre-req not met).")
	return ""
}

func (config *gceFirewallTestConfig) getServerNode() *api.Node {
	nodeIP := config.serverPod.Status.HostIP
	nodeList, err := config.f.Client.Nodes().List(unversioned.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to find node with hostIP %q", nodeIP))
	for _, node := range nodeList.Items {
		if node.Name == config.serverPod.Spec.NodeName {
			return &node
		}
	}
	Failf("Failed to find node with name %s, on which GCE Firewall Server is running. ", nodeIP)
	return nil
}

func newHTTPClient(transport *http.Transport) *http.Client {
	client := &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}
	return client
}

func hitHTTPEndpoint(httpEndpoint string, tries int) bool {
	hitCount := 0
	for i := 0; i < tries; i++ {
		transport := &http.Transport{}
		httpClient := newHTTPClient(transport)
		_, err := httpClient.Get(httpEndpoint)
		if err == nil {
			hitCount++
		}
		transport.CloseIdleConnections()
	}
	return (hitCount > 0)
}

func (config *gceFirewallTestConfig) waitForLoadBalancerIngressSetup() {
	serviceClient := config.f.Client.Services(config.f.Namespace.Name)
	err := wait.Poll(2*time.Second, 1*time.Minute, func() (bool, error) {
		service, err := serviceClient.Get(gceFirewallLoadBalancerServiceName)
		if err != nil {
			return false, err
		} else {
			if len(service.Status.LoadBalancer.Ingress) > 0 {
				return true, nil
			} else {
				Logf("Service LoadBalancer Ingress was not setup. Waiting..")
				return false, nil
			}
		}
	})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to setup Load Balancer Service. err:%v", err))
	Logf("Load Balancer Ingress is setup.")
	config.loadBalancerService, _ = serviceClient.Get(gceFirewallLoadBalancerServiceName)
}
