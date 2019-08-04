/*
Copyright 2017 The Kubernetes Authors.

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

package master

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kube-aggregator/pkg/apis/apiregistration"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestRun(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// test whether the server is really healthy after /healthz told us so
	t.Logf("Creating Deployment directly after being healthy")
	var replicas int32 = 1
	_, err = client.AppsV1().Deployments("default").Create(&appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "test",
			Labels:    map[string]string{"foo": "bar"},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Strategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
			},
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "foo",
							Image: "foo",
						},
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}
}

func endpointReturnsStatusOK(client *kubernetes.Clientset, path string) bool {
	res := client.CoreV1().RESTClient().Get().AbsPath(path).Do()
	var status int
	res.StatusCode(&status)
	return status == http.StatusOK
}

func TestStartupSequenceHealthzAndReadyz(t *testing.T) {
	hc := &delayedCheck{}
	instanceOptions := &kubeapiservertesting.TestServerInstanceOptions{
		InjectedHealthzChecker: hc,
	}
	server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, []string{"--maximum-startup-sequence-duration", "15s"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if endpointReturnsStatusOK(client, "/readyz") {
		t.Fatalf("readyz should start unready")
	}
	// we need to wait longer than our grace period
	err = wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return !endpointReturnsStatusOK(client, "/healthz"), nil
	})
	if err != nil {
		t.Fatalf("healthz should have become unhealthy: %v", err)
	}
	hc.makeHealthy()
	err = wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return endpointReturnsStatusOK(client, "/healthz"), nil
	})
	if err != nil {
		t.Fatalf("healthz should have become healthy again: %v", err)
	}
	if !endpointReturnsStatusOK(client, "/readyz") {
		t.Fatalf("readyz should be healthy")
	}
}

// TestOpenAPIDelegationChainPlumbing is a smoke test that checks for
// the existence of some representative paths from the
// apiextensions-server and the kube-aggregator server, both part of
// the delegation chain in kube-apiserver.
func TestOpenAPIDelegationChainPlumbing(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
	defer server.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	result := kubeclient.RESTClient().Get().AbsPath("/openapi/v2").Do()
	status := 0
	result.StatusCode(&status)
	if status != http.StatusOK {
		t.Fatalf("GET /openapi/v2 failed: expected status=%d, got=%d", http.StatusOK, status)
	}

	raw, err := result.Raw()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	type openAPISchema struct {
		Paths map[string]interface{} `json:"paths"`
	}

	var doc openAPISchema
	err = json.Unmarshal(raw, &doc)
	if err != nil {
		t.Fatalf("Failed to unmarshal: %v", err)
	}

	matchedExtension := false
	extensionsPrefix := "/apis/" + apiextensions.GroupName

	matchedRegistration := false
	registrationPrefix := "/apis/" + apiregistration.GroupName

	for path := range doc.Paths {
		if strings.HasPrefix(path, extensionsPrefix) {
			matchedExtension = true
		}
		if strings.HasPrefix(path, registrationPrefix) {
			matchedRegistration = true
		}
		if matchedExtension && matchedRegistration {
			return
		}
	}

	if !matchedExtension {
		t.Errorf("missing path: %q", extensionsPrefix)
	}

	if !matchedRegistration {
		t.Errorf("missing path: %q", registrationPrefix)
	}
}

// return the unique endpoint IPs
func getEndpointIPs(endpoints *corev1.Endpoints) []string {
	endpointMap := make(map[string]bool)
	ips := make([]string, 0)
	for _, subset := range endpoints.Subsets {
		for _, address := range subset.Addresses {
			if _, ok := endpointMap[address.IP]; !ok {
				endpointMap[address.IP] = true
				ips = append(ips, address.IP)
			}
		}
	}
	return ips
}

func verifyEndpointsWithIPs(servers []*kubeapiservertesting.TestServer, ips []string) bool {
	listenAddresses := make([]string, 0)
	for _, server := range servers {
		listenAddresses = append(listenAddresses, server.ServerOpts.GenericServerRunOptions.AdvertiseAddress.String())
	}
	return reflect.DeepEqual(listenAddresses, ips)
}

func testReconcilersMasterLease(t *testing.T, leaseCount int, masterCount int) {
	var leaseServers []*kubeapiservertesting.TestServer
	var masterCountServers []*kubeapiservertesting.TestServer
	etcd := framework.SharedEtcd()

	instanceOptions := &kubeapiservertesting.TestServerInstanceOptions{
		DisableStorageCleanup: true,
	}

	// cleanup the registry storage
	defer registry.CleanupStorage()

	// 1. start masterCount api servers
	for i := 0; i < masterCount; i++ {
		// start master count api server
		server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, []string{
			"--endpoint-reconciler-type", "master-count",
			"--advertise-address", fmt.Sprintf("10.0.1.%v", i+1),
			"--apiserver-count", fmt.Sprintf("%v", masterCount),
		}, etcd)
		masterCountServers = append(masterCountServers, server)
	}

	// 2. verify master count servers have registered
	if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
		client, err := kubernetes.NewForConfig(masterCountServers[0].ClientConfig)
		endpoints, err := client.CoreV1().Endpoints("default").Get("kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return verifyEndpointsWithIPs(masterCountServers, getEndpointIPs(endpoints)), nil
	}); err != nil {
		t.Fatalf("master count endpoints failed to register: %v", err)
	}

	// 3. start lease api servers
	for i := 0; i < leaseCount; i++ {
		options := []string{
			"--endpoint-reconciler-type", "lease",
			"--advertise-address", fmt.Sprintf("10.0.1.%v", i+10),
		}
		server := kubeapiservertesting.StartTestServerOrDie(t, instanceOptions, options, etcd)
		defer server.TearDownFn()
		leaseServers = append(leaseServers, server)
	}

	time.Sleep(3 * time.Second)

	// 4. Shutdown the masterCount server
	for _, server := range masterCountServers {
		server.TearDownFn()
	}

	// 5. verify only leaseEndpoint servers left
	if err := wait.PollImmediate(3*time.Second, 2*time.Minute, func() (bool, error) {
		client, err := kubernetes.NewForConfig(leaseServers[0].ClientConfig)
		if err != nil {
			t.Logf("create client error: %v", err)
			return false, nil
		}
		endpoints, err := client.CoreV1().Endpoints("default").Get("kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return verifyEndpointsWithIPs(leaseServers, getEndpointIPs(endpoints)), nil
	}); err != nil {
		t.Fatalf("did not find only lease endpoints: %v", err)
	}
}

func TestReconcilerMasterLeaseCombined(t *testing.T) {
	testReconcilersMasterLease(t, 1, 3)
}

func TestReconcilerMasterLeaseMultiMoreMasters(t *testing.T) {
	testReconcilersMasterLease(t, 3, 2)
}

func TestReconcilerMasterLeaseMultiCombined(t *testing.T) {
	testReconcilersMasterLease(t, 3, 3)
}

type delayedCheck struct {
	healthLock sync.Mutex
	isHealthy  bool
}

func (h *delayedCheck) Name() string {
	return "delayed-check"
}

func (h *delayedCheck) Check(req *http.Request) error {
	h.healthLock.Lock()
	defer h.healthLock.Unlock()
	if h.isHealthy {
		return nil
	}
	return fmt.Errorf("isn't healthy")
}

func (h *delayedCheck) makeHealthy() {
	h.healthLock.Lock()
	defer h.healthLock.Unlock()
	h.isHealthy = true
}
