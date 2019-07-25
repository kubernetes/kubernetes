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

package service

import (
	"fmt"
	"net"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"
)

// StartServeHostnameService creates a replication controller that serves its
// hostname and a service on top of it.
func StartServeHostnameService(c clientset.Interface, svc *v1.Service, ns string, replicas int) ([]string, string, error) {
	podNames := make([]string, replicas)
	name := svc.ObjectMeta.Name
	ginkgo.By("creating service " + name + " in namespace " + ns)
	_, err := c.CoreV1().Services(ns).Create(svc)
	if err != nil {
		return podNames, "", err
	}

	var createdPods []*v1.Pod
	maxContainerFailures := 0
	config := testutils.RCConfig{
		Client:               c,
		Image:                framework.ServeHostnameImage,
		Command:              []string{"/agnhost", "serve-hostname"},
		Name:                 name,
		Namespace:            ns,
		PollInterval:         3 * time.Second,
		Timeout:              framework.PodReadyBeforeTimeout,
		Replicas:             replicas,
		CreatedPods:          &createdPods,
		MaxContainerFailures: &maxContainerFailures,
	}
	err = framework.RunRC(config)
	if err != nil {
		return podNames, "", err
	}

	if len(createdPods) != replicas {
		return podNames, "", fmt.Errorf("incorrect number of running pods: %v", len(createdPods))
	}

	for i := range createdPods {
		podNames[i] = createdPods[i].ObjectMeta.Name
	}
	sort.StringSlice(podNames).Sort()

	service, err := c.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		return podNames, "", err
	}
	if service.Spec.ClusterIP == "" {
		return podNames, "", fmt.Errorf("service IP is blank for %v", name)
	}
	serviceIP := service.Spec.ClusterIP
	return podNames, serviceIP, nil
}

// StopServeHostnameService stops the given service.
func StopServeHostnameService(clientset clientset.Interface, ns, name string) error {
	if err := framework.DeleteRCAndWaitForGC(clientset, ns, name); err != nil {
		return err
	}
	if err := clientset.CoreV1().Services(ns).Delete(name, nil); err != nil {
		return err
	}
	return nil
}

// VerifyServeHostnameServiceUp wgets the given serviceIP:servicePort from the
// given host and from within a pod. The host is expected to be an SSH-able node
// in the cluster. Each pod in the service is expected to echo its name. These
// names are compared with the given expectedPods list after a sort | uniq.
func VerifyServeHostnameServiceUp(c clientset.Interface, ns, host string, expectedPods []string, serviceIP string, servicePort int) error {
	execPod := e2epod.CreateExecPodOrFail(c, ns, "execpod-", nil)
	defer func() {
		e2epod.DeletePodOrFail(c, ns, execPod.Name)
	}()

	// Loop a bunch of times - the proxy is randomized, so we want a good
	// chance of hitting each backend at least once.
	buildCommand := func(wget string) string {
		serviceIPPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
		return fmt.Sprintf("for i in $(seq 1 %d); do %s http://%s 2>&1 || true; echo; done",
			50*len(expectedPods), wget, serviceIPPort)
	}
	commands := []func() string{
		// verify service from node
		func() string {
			cmd := "set -e; " + buildCommand("wget -q --timeout=0.2 --tries=1 -O -")
			e2elog.Logf("Executing cmd %q on host %v", cmd, host)
			result, err := e2essh.SSH(cmd, host, framework.TestContext.Provider)
			if err != nil || result.Code != 0 {
				e2essh.LogResult(result)
				e2elog.Logf("error while SSH-ing to node: %v", err)
			}
			return result.Stdout
		},
		// verify service from pod
		func() string {
			cmd := buildCommand("wget -q -T 1 -O -")
			e2elog.Logf("Executing cmd %q in pod %v/%v", cmd, ns, execPod.Name)
			// TODO: Use exec-over-http via the netexec pod instead of kubectl exec.
			output, err := framework.RunHostCmd(ns, execPod.Name, cmd)
			if err != nil {
				e2elog.Logf("error while kubectl execing %q in pod %v/%v: %v\nOutput: %v", cmd, ns, execPod.Name, err, output)
			}
			return output
		},
	}

	expectedEndpoints := sets.NewString(expectedPods...)
	ginkgo.By(fmt.Sprintf("verifying service has %d reachable backends", len(expectedPods)))
	for _, cmdFunc := range commands {
		passed := false
		gotEndpoints := sets.NewString()

		// Retry cmdFunc for a while
		for start := time.Now(); time.Since(start) < KubeProxyLagTimeout; time.Sleep(5 * time.Second) {
			for _, endpoint := range strings.Split(cmdFunc(), "\n") {
				trimmedEp := strings.TrimSpace(endpoint)
				if trimmedEp != "" {
					gotEndpoints.Insert(trimmedEp)
				}
			}
			// TODO: simply checking that the retrieved endpoints is a superset
			// of the expected allows us to ignore intermitten network flakes that
			// result in output like "wget timed out", but these should be rare
			// and we need a better way to track how often it occurs.
			if gotEndpoints.IsSuperset(expectedEndpoints) {
				if !gotEndpoints.Equal(expectedEndpoints) {
					e2elog.Logf("Ignoring unexpected output wgetting endpoints of service %s: %v", serviceIP, gotEndpoints.Difference(expectedEndpoints))
				}
				passed = true
				break
			}
			e2elog.Logf("Unable to reach the following endpoints of service %s: %v", serviceIP, expectedEndpoints.Difference(gotEndpoints))
		}
		if !passed {
			// Sort the lists so they're easier to visually diff.
			exp := expectedEndpoints.List()
			got := gotEndpoints.List()
			sort.StringSlice(exp).Sort()
			sort.StringSlice(got).Sort()
			return fmt.Errorf("service verification failed for: %s\nexpected %v\nreceived %v", serviceIP, exp, got)
		}
	}
	return nil
}

// VerifyServeHostnameServiceDown verifies that the given service isn't served.
func VerifyServeHostnameServiceDown(c clientset.Interface, host string, serviceIP string, servicePort int) error {
	ipPort := net.JoinHostPort(serviceIP, strconv.Itoa(servicePort))
	// The current versions of curl included in CentOS and RHEL distros
	// misinterpret square brackets around IPv6 as globbing, so use the -g
	// argument to disable globbing to handle the IPv6 case.
	command := fmt.Sprintf(
		"curl -g -s --connect-timeout 2 http://%s && exit 99", ipPort)

	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(command, host, framework.TestContext.Provider)
		if err != nil {
			e2essh.LogResult(result)
			e2elog.Logf("error while SSH-ing to node: %v", err)
		}
		if result.Code != 99 {
			return nil
		}
		e2elog.Logf("service still alive - still waiting")
	}
	return fmt.Errorf("waiting for service to be down timed out")
}
