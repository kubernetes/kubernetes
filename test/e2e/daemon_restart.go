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
	"strconv"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	controllerFramework "k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// This test primarily checks 2 things:
// 1. Daemons restart automatically within some sane time (10m).
// 2. They don't take abnormal actions when restarted in the steady state.
//	- Controller manager sholdn't overshoot replicas
//	- Kubelet shouldn't restart containers
//	- Scheduler should continue assigning hosts to new pods

const (
	restartPollInterval = 5 * time.Second
	restartTimeout      = 10 * time.Minute
	numPods             = 10
	sshPort             = 22
)

// nodeExec execs the given cmd on node via SSH. Note that the nodeName is an sshable name,
// eg: the name returned by getMasterHost(). This is also not guaranteed to work across
// cloud providers since it involves ssh.
func nodeExec(nodeName, cmd string) (string, string, int, error) {
	stdout, stderr, code, err := SSH(cmd, fmt.Sprintf("%v:%v", nodeName, sshPort), testContext.Provider)
	Expect(err).NotTo(HaveOccurred())
	return stdout, stderr, code, err
}

// restartDaemonConfig is a config to restart a running daemon on a node, and wait till
// it comes back up. It uses ssh to send a SIGTERM to the daemon.
type restartDaemonConfig struct {
	nodeName     string
	daemonName   string
	healthzPort  int
	pollInterval time.Duration
	pollTimeout  time.Duration
}

// NewRestartConfig creates a restartDaemonConfig for the given node and daemon.
func NewRestartConfig(nodeName, daemonName string, healthzPort int, pollInterval, pollTimeout time.Duration) *restartDaemonConfig {
	if !providerIs("gce") {
		Logf("WARNING: SSH through the restart config might not work on %s", testContext.Provider)
	}
	return &restartDaemonConfig{
		nodeName:     nodeName,
		daemonName:   daemonName,
		healthzPort:  healthzPort,
		pollInterval: pollInterval,
		pollTimeout:  pollTimeout,
	}
}

func (r *restartDaemonConfig) String() string {
	return fmt.Sprintf("Daemon %v on node %v", r.daemonName, r.nodeName)
}

// waitUp polls healthz of the daemon till it returns "ok" or the polling hits the pollTimeout
func (r *restartDaemonConfig) waitUp() {
	Logf("Checking if %v is up by polling for a 200 on its /healthz endpoint", r)
	healthzCheck := fmt.Sprintf(
		"curl -s -o /dev/null -I -w \"%%{http_code}\" http://localhost:%v/healthz", r.healthzPort)

	err := wait.Poll(r.pollInterval, r.pollTimeout, func() (bool, error) {
		stdout, stderr, code, err := nodeExec(r.nodeName, healthzCheck)
		expectNoError(err)
		if code == 0 {
			httpCode, err := strconv.Atoi(stdout)
			if err != nil {
				Logf("Unable to parse healthz http return code: %v", err)
			} else if httpCode == 200 {
				return true, nil
			}
		}
		Logf("node %v exec command, '%v' failed with exitcode %v: \n\tstdout: %v\n\tstderr: %v",
			r.nodeName, healthzCheck, code, stdout, stderr)
		return false, nil
	})
	expectNoError(err, "%v did not respond with a 200 via %v within %v", r, healthzCheck, r.pollTimeout)
}

// kill sends a SIGTERM to the daemon
func (r *restartDaemonConfig) kill() {
	Logf("Killing %v", r)
	nodeExec(r.nodeName, fmt.Sprintf("pgrep %v | xargs -I {} sudo kill {}", r.daemonName))
}

// Restart checks if the daemon is up, kills it, and waits till it comes back up
func (r *restartDaemonConfig) restart() {
	r.waitUp()
	r.kill()
	r.waitUp()
}

// replacePods replaces content of the store with the given pods.
func replacePods(pods []*api.Pod, store cache.Store) {
	found := make([]interface{}, 0, len(pods))
	for i := range pods {
		found = append(found, pods[i])
	}
	expectNoError(store.Replace(found, "0"))
}

// getContainerRestarts returns the count of container restarts across all pods matching the given labelSelector,
// and a list of nodenames across which these containers restarted.
func getContainerRestarts(c *client.Client, ns string, labelSelector labels.Selector) (int, []string) {
	pods, err := c.Pods(ns).List(labelSelector, fields.Everything())
	expectNoError(err)
	failedContainers := 0
	containerRestartNodes := sets.NewString()
	for _, p := range pods.Items {
		for _, v := range FailedContainers(&p) {
			failedContainers = failedContainers + v.restarts
			containerRestartNodes.Insert(p.Spec.NodeName)
		}
	}
	return failedContainers, containerRestartNodes.List()
}

var _ = Describe("DaemonRestart", func() {

	framework := Framework{BaseName: "daemonrestart"}
	rcName := "daemonrestart" + strconv.Itoa(numPods) + "-" + string(util.NewUUID())
	labelSelector := labels.Set(map[string]string{"name": rcName}).AsSelector()
	existingPods := cache.NewStore(cache.MetaNamespaceKeyFunc)
	var ns string
	var config RCConfig
	var controller *controllerFramework.Controller
	var newPods cache.Store
	var stopCh chan struct{}

	BeforeEach(func() {

		// These tests require SSH
		// TODO(11834): Enable this test in GKE once experimental API there is switched on
		SkipUnlessProviderIs("gce")
		framework.beforeEach()
		ns = framework.Namespace.Name

		// All the restart tests need an rc and a watch on pods of the rc.
		// Additionally some of them might scale the rc during the test.
		config = RCConfig{
			Client:      framework.Client,
			Name:        rcName,
			Namespace:   ns,
			Image:       "kubernetes/pause",
			Replicas:    numPods,
			CreatedPods: &[]*api.Pod{},
		}
		Expect(RunRC(config)).NotTo(HaveOccurred())
		replacePods(*config.CreatedPods, existingPods)

		stopCh = make(chan struct{})
		newPods, controller = controllerFramework.NewInformer(
			&cache.ListWatch{
				ListFunc: func() (runtime.Object, error) {
					return framework.Client.Pods(ns).List(labelSelector, fields.Everything())
				},
				WatchFunc: func(rv string) (watch.Interface, error) {
					return framework.Client.Pods(ns).Watch(labelSelector, fields.Everything(), rv)
				},
			},
			&api.Pod{},
			0,
			controllerFramework.ResourceEventHandlerFuncs{},
		)
		go controller.Run(stopCh)
	})

	AfterEach(func() {
		close(stopCh)
		expectNoError(DeleteRC(framework.Client, ns, rcName))
		framework.afterEach()
	})

	It("Controller Manager should not create/delete replicas across restart", func() {

		restarter := NewRestartConfig(
			getMasterHost(), "kube-controller", ports.ControllerManagerPort, restartPollInterval, restartTimeout)
		restarter.restart()

		// The intent is to ensure the replication controller manager has observed and reported status of
		// the replication controller at least once since the manager restarted, so that we can determine
		// that it had the opportunity to create/delete pods, if it were going to do so. Scaling the RC
		// to the same size achieves this, because the scale operation advances the RC's sequence number
		// and awaits it to be observed and reported back in the RC's status.
		ScaleRC(framework.Client, ns, rcName, numPods, true)

		// Only check the keys, the pods can be different if the kubelet updated it.
		// TODO: Can it really?
		existingKeys := sets.NewString()
		newKeys := sets.NewString()
		for _, k := range existingPods.ListKeys() {
			existingKeys.Insert(k)
		}
		for _, k := range newPods.ListKeys() {
			newKeys.Insert(k)
		}
		if len(newKeys.List()) != len(existingKeys.List()) ||
			!newKeys.IsSuperset(existingKeys) {
			Failf("RcManager created/deleted pods after restart")
		}
	})

	It("Scheduler should continue assigning pods to nodes across restart", func() {

		restarter := NewRestartConfig(
			getMasterHost(), "kube-scheduler", ports.SchedulerPort, restartPollInterval, restartTimeout)

		// Create pods while the scheduler is down and make sure the scheduler picks them up by
		// scaling the rc to the same size.
		restarter.waitUp()
		restarter.kill()
		// This is best effort to try and create pods while the scheduler is down,
		// since we don't know exactly when it is restarted after the kill signal.
		expectNoError(ScaleRC(framework.Client, ns, rcName, numPods+5, false))
		restarter.waitUp()
		expectNoError(ScaleRC(framework.Client, ns, rcName, numPods+5, true))
	})

	It("Kubelet should not restart containers across restart", func() {
		nodeIPs, err := getNodePublicIps(framework.Client)
		expectNoError(err)
		preRestarts, badNodes := getContainerRestarts(framework.Client, ns, labelSelector)
		if preRestarts != 0 {
			Logf("WARNING: Non-zero container restart count: %d across nodes %v", preRestarts, badNodes)
		}
		for _, ip := range nodeIPs {
			restarter := NewRestartConfig(
				ip, "kubelet", ports.KubeletReadOnlyPort, restartPollInterval, restartTimeout)
			restarter.restart()
		}
		postRestarts, badNodes := getContainerRestarts(framework.Client, ns, labelSelector)
		if postRestarts != preRestarts {
			dumpNodeDebugInfo(framework.Client, badNodes)
			Failf("Net container restart count went from %v -> %v after kubelet restart on nodes %v", preRestarts, postRestarts, badNodes)
		}
	})
})
