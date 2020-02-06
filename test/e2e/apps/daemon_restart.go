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

package apps

import (
	"fmt"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// This test primarily checks 2 things:
// 1. Daemons restart automatically within some sane time (10m).
// 2. They don't take abnormal actions when restarted in the steady state.
//	- Controller manager shouldn't overshoot replicas
//	- Kubelet shouldn't restart containers
//	- Scheduler should continue assigning hosts to new pods

const (
	restartPollInterval = 5 * time.Second
	restartTimeout      = 10 * time.Minute
	numPods             = 10
	// ADD represents the ADD event
	ADD = "ADD"
	// DEL represents the DEL event
	DEL = "DEL"
	// UPDATE represents the UPDATE event
	UPDATE = "UPDATE"
)

// RestartDaemonConfig is a config to restart a running daemon on a node, and wait till
// it comes back up. It uses ssh to send a SIGTERM to the daemon.
type RestartDaemonConfig struct {
	nodeName     string
	daemonName   string
	healthzPort  int
	pollInterval time.Duration
	pollTimeout  time.Duration
}

// NewRestartConfig creates a RestartDaemonConfig for the given node and daemon.
func NewRestartConfig(nodeName, daemonName string, healthzPort int, pollInterval, pollTimeout time.Duration) *RestartDaemonConfig {
	if !framework.ProviderIs("gce") {
		framework.Logf("WARNING: SSH through the restart config might not work on %s", framework.TestContext.Provider)
	}
	return &RestartDaemonConfig{
		nodeName:     nodeName,
		daemonName:   daemonName,
		healthzPort:  healthzPort,
		pollInterval: pollInterval,
		pollTimeout:  pollTimeout,
	}
}

func (r *RestartDaemonConfig) String() string {
	return fmt.Sprintf("Daemon %v on node %v", r.daemonName, r.nodeName)
}

// waitUp polls healthz of the daemon till it returns "ok" or the polling hits the pollTimeout
func (r *RestartDaemonConfig) waitUp() {
	framework.Logf("Checking if %v is up by polling for a 200 on its /healthz endpoint", r)
	nullDev := "/dev/null"
	if framework.NodeOSDistroIs("windows") {
		nullDev = "NUL"
	}
	healthzCheck := fmt.Sprintf(
		"curl -s -o %v -I -w \"%%{http_code}\" http://localhost:%v/healthz", nullDev, r.healthzPort)
	err := wait.Poll(r.pollInterval, r.pollTimeout, func() (bool, error) {
		result, err := e2essh.NodeExec(r.nodeName, healthzCheck, framework.TestContext.Provider)
		framework.ExpectNoError(err)
		if result.Code == 0 {
			httpCode, err := strconv.Atoi(result.Stdout)
			if err != nil {
				framework.Logf("Unable to parse healthz http return code: %v", err)
			} else if httpCode == 200 {
				return true, nil
			}
		}
		framework.Logf("node %v exec command, '%v' failed with exitcode %v: \n\tstdout: %v\n\tstderr: %v",
			r.nodeName, healthzCheck, result.Code, result.Stdout, result.Stderr)
		return false, nil
	})
	framework.ExpectNoError(err, "%v did not respond with a 200 via %v within %v", r, healthzCheck, r.pollTimeout)
}

// kill sends a SIGTERM to the daemon
func (r *RestartDaemonConfig) kill() {
	killCmd := fmt.Sprintf("pgrep %v | xargs -I {} sudo kill {}", r.daemonName)
	if framework.NodeOSDistroIs("windows") {
		killCmd = fmt.Sprintf("taskkill /im %v.exe /f", r.daemonName)
	}
	framework.Logf("Killing %v", r)
	_, err := e2essh.NodeExec(r.nodeName, killCmd, framework.TestContext.Provider)
	framework.ExpectNoError(err)
}

// Restart checks if the daemon is up, kills it, and waits till it comes back up
func (r *RestartDaemonConfig) restart() {
	r.waitUp()
	r.kill()
	r.waitUp()
}

// podTracker records a serial history of events that might've affects pods.
type podTracker struct {
	cache.ThreadSafeStore
}

func (p *podTracker) remember(pod *v1.Pod, eventType string) {
	if eventType == UPDATE && pod.Status.Phase == v1.PodRunning {
		return
	}
	p.Add(fmt.Sprintf("[%v] %v: %v", time.Now(), eventType, pod.Name), pod)
}

func (p *podTracker) String() (msg string) {
	for _, k := range p.ListKeys() {
		obj, exists := p.Get(k)
		if !exists {
			continue
		}
		pod := obj.(*v1.Pod)
		msg += fmt.Sprintf("%v Phase %v Host %v\n", k, pod.Status.Phase, pod.Spec.NodeName)
	}
	return
}

func newPodTracker() *podTracker {
	return &podTracker{cache.NewThreadSafeStore(
		cache.Indexers{}, cache.Indices{})}
}

// replacePods replaces content of the store with the given pods.
func replacePods(pods []*v1.Pod, store cache.Store) {
	found := make([]interface{}, 0, len(pods))
	for i := range pods {
		found = append(found, pods[i])
	}
	framework.ExpectNoError(store.Replace(found, "0"))
}

// getContainerRestarts returns the count of container restarts across all pods matching the given labelSelector,
// and a list of nodenames across which these containers restarted.
func getContainerRestarts(c clientset.Interface, ns string, labelSelector labels.Selector) (int, []string) {
	options := metav1.ListOptions{LabelSelector: labelSelector.String()}
	pods, err := c.CoreV1().Pods(ns).List(options)
	framework.ExpectNoError(err)
	failedContainers := 0
	containerRestartNodes := sets.NewString()
	for _, p := range pods.Items {
		for _, v := range testutils.FailedContainers(&p) {
			failedContainers = failedContainers + v.Restarts
			containerRestartNodes.Insert(p.Spec.NodeName)
		}
	}
	return failedContainers, containerRestartNodes.List()
}

var _ = SIGDescribe("DaemonRestart [Disruptive]", func() {

	f := framework.NewDefaultFramework("daemonrestart")
	rcName := "daemonrestart" + strconv.Itoa(numPods) + "-" + string(uuid.NewUUID())
	labelSelector := labels.Set(map[string]string{"name": rcName}).AsSelector()
	existingPods := cache.NewStore(cache.MetaNamespaceKeyFunc)
	var ns string
	var config testutils.RCConfig
	var controller cache.Controller
	var newPods cache.Store
	var stopCh chan struct{}
	var tracker *podTracker

	ginkgo.BeforeEach(func() {
		// These tests require SSH
		e2eskipper.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		ns = f.Namespace.Name

		// All the restart tests need an rc and a watch on pods of the rc.
		// Additionally some of them might scale the rc during the test.
		config = testutils.RCConfig{
			Client:      f.ClientSet,
			Name:        rcName,
			Namespace:   ns,
			Image:       imageutils.GetPauseImageName(),
			Replicas:    numPods,
			CreatedPods: &[]*v1.Pod{},
		}
		framework.ExpectNoError(e2erc.RunRC(config))
		replacePods(*config.CreatedPods, existingPods)

		stopCh = make(chan struct{})
		tracker = newPodTracker()
		newPods, controller = cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					options.LabelSelector = labelSelector.String()
					obj, err := f.ClientSet.CoreV1().Pods(ns).List(options)
					return runtime.Object(obj), err
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					options.LabelSelector = labelSelector.String()
					return f.ClientSet.CoreV1().Pods(ns).Watch(options)
				},
			},
			&v1.Pod{},
			0,
			cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					tracker.remember(obj.(*v1.Pod), ADD)
				},
				UpdateFunc: func(oldObj, newObj interface{}) {
					tracker.remember(newObj.(*v1.Pod), UPDATE)
				},
				DeleteFunc: func(obj interface{}) {
					tracker.remember(obj.(*v1.Pod), DEL)
				},
			},
		)
		go controller.Run(stopCh)
	})

	ginkgo.AfterEach(func() {
		close(stopCh)
	})

	ginkgo.It("Controller Manager should not create/delete replicas across restart", func() {

		// Requires master ssh access.
		e2eskipper.SkipUnlessProviderIs("gce", "aws")
		restarter := NewRestartConfig(
			framework.GetMasterHost(), "kube-controller", ports.InsecureKubeControllerManagerPort, restartPollInterval, restartTimeout)
		restarter.restart()

		// The intent is to ensure the replication controller manager has observed and reported status of
		// the replication controller at least once since the manager restarted, so that we can determine
		// that it had the opportunity to create/delete pods, if it were going to do so. Scaling the RC
		// to the same size achieves this, because the scale operation advances the RC's sequence number
		// and awaits it to be observed and reported back in the RC's status.
		e2erc.ScaleRC(f.ClientSet, f.ScalesGetter, ns, rcName, numPods, true)

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
			framework.Failf("RcManager created/deleted pods after restart \n\n %+v", tracker)
		}
	})

	ginkgo.It("Scheduler should continue assigning pods to nodes across restart", func() {

		// Requires master ssh access.
		e2eskipper.SkipUnlessProviderIs("gce", "aws")
		restarter := NewRestartConfig(
			framework.GetMasterHost(), "kube-scheduler", ports.InsecureSchedulerPort, restartPollInterval, restartTimeout)

		// Create pods while the scheduler is down and make sure the scheduler picks them up by
		// scaling the rc to the same size.
		restarter.waitUp()
		restarter.kill()
		// This is best effort to try and create pods while the scheduler is down,
		// since we don't know exactly when it is restarted after the kill signal.
		framework.ExpectNoError(e2erc.ScaleRC(f.ClientSet, f.ScalesGetter, ns, rcName, numPods+5, false))
		restarter.waitUp()
		framework.ExpectNoError(e2erc.ScaleRC(f.ClientSet, f.ScalesGetter, ns, rcName, numPods+5, true))
	})

	ginkgo.It("Kubelet should not restart containers across restart", func() {
		nodeIPs, err := e2enode.GetPublicIps(f.ClientSet)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		// TODO: write a wrapper for ExpectNoErrorWithOffset()
		framework.ExpectNoErrorWithOffset(0, err)
		preRestarts, badNodes := getContainerRestarts(f.ClientSet, ns, labelSelector)
		if preRestarts != 0 {
			framework.Logf("WARNING: Non-zero container restart count: %d across nodes %v", preRestarts, badNodes)
		}
		for _, ip := range nodeIPs {
			restarter := NewRestartConfig(
				ip, "kubelet", ports.KubeletReadOnlyPort, restartPollInterval, restartTimeout)
			restarter.restart()
		}
		postRestarts, badNodes := getContainerRestarts(f.ClientSet, ns, labelSelector)
		if postRestarts != preRestarts {
			framework.DumpNodeDebugInfo(f.ClientSet, badNodes, framework.Logf)
			framework.Failf("Net container restart count went from %v -> %v after kubelet restart on nodes %v \n\n %+v", preRestarts, postRestarts, badNodes, tracker)
		}
	})

	ginkgo.It("Kube-proxy should recover after being killed accidentally", func() {
		nodeIPs, err := e2enode.GetPublicIps(f.ClientSet)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		for _, ip := range nodeIPs {
			restarter := NewRestartConfig(
				ip, "kube-proxy", ports.ProxyHealthzPort, restartPollInterval, restartTimeout)
			// restart method will kill the kube-proxy process and wait for recovery,
			// if not able to recover, will throw test failure.
			restarter.restart()
		}
	})
})
