/*
Copyright 2014 The Kubernetes Authors.

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
	"path/filepath"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	klabels "k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	statefulsetPoll = 10 * time.Second
	// Some statefulPods install base packages via wget
	statefulsetTimeout = 10 * time.Minute
	// Timeout for stateful pods to change state
	statefulPodTimeout      = 5 * time.Minute
	zookeeperManifestPath   = "test/e2e/testing-manifests/statefulset/zookeeper"
	mysqlGaleraManifestPath = "test/e2e/testing-manifests/statefulset/mysql-galera"
	redisManifestPath       = "test/e2e/testing-manifests/statefulset/redis"
	cockroachDBManifestPath = "test/e2e/testing-manifests/statefulset/cockroachdb"
	// We don't restart MySQL cluster regardless of restartCluster, since MySQL doesn't handle restart well
	restartCluster = true

	// Timeout for reads from databases running on stateful pods.
	readTimeout = 60 * time.Second
)

// GCE Quota requirements: 3 pds, one per stateful pod manifest declared above.
// GCE Api requirements: nodes and master need storage r/w permissions.
var _ = framework.KubeDescribe("StatefulSet", func() {
	f := framework.NewDefaultFramework("statefulset")
	var ns string
	var c clientset.Interface

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("Basic StatefulSet functionality [StatefulSetBasic]", func() {
		ssName := "ss"
		labels := map[string]string{
			"foo": "bar",
			"baz": "blah",
		}
		headlessSvcName := "test"
		var statefulPodMounts, podMounts []v1.VolumeMount
		var ss *apps.StatefulSet

		BeforeEach(func() {
			statefulPodMounts = []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
			podMounts = []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
			ss = newStatefulSet(ssName, ns, headlessSvcName, 2, statefulPodMounts, podMounts, labels)

			By("Creating service " + headlessSvcName + " in namespace " + ns)
			headlessService := createServiceSpec(headlessSvcName, "", true, labels)
			_, err := c.Core().Services(ns).Create(headlessService)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				dumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			deleteAllStatefulSets(c, ns)
		})

		It("should provide basic identity", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 3
			setInitializedAnnotation(ss, "false")

			_, err := c.Apps().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			sst := statefulSetTester{c: c}

			By("Saturating stateful set " + ss.Name)
			sst.saturate(ss)

			By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(sst.checkMount(ss, "/data"))

			By("Verifying statefulset provides a stable hostname for each pod")
			framework.ExpectNoError(sst.checkHostname(ss))

			By("Verifying statefulset set proper service name")
			framework.ExpectNoError(sst.checkServiceName(ss, headlessSvcName))

			cmd := "echo $(hostname) > /data/hostname; sync;"
			By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(sst.execInStatefulPods(ss, cmd))

			By("Restarting statefulset " + ss.Name)
			sst.restart(ss)
			sst.saturate(ss)

			By("Verifying statefulset mounted data directory is usable")
			framework.ExpectNoError(sst.checkMount(ss, "/data"))

			cmd = "if [ \"$(cat /data/hostname)\" = \"$(hostname)\" ]; then exit 0; else exit 1; fi"
			By("Running " + cmd + " in all stateful pods")
			framework.ExpectNoError(sst.execInStatefulPods(ss, cmd))
		})

		It("should not deadlock when a pod's predecessor fails", func() {
			By("Creating statefulset " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 2
			setInitializedAnnotation(ss, "false")

			_, err := c.Apps().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			sst := statefulSetTester{c: c}

			sst.waitForRunningAndReady(1, ss)

			By("Marking stateful pod at index 0 as healthy.")
			sst.setHealthy(ss)

			By("Waiting for stateful pod at index 1 to enter running.")
			sst.waitForRunningAndReady(2, ss)

			// Now we have 1 healthy and 1 unhealthy stateful pod. Deleting the healthy stateful pod should *not*
			// create a new stateful pod till the remaining stateful pod becomes healthy, which won't happen till
			// we set the healthy bit.

			By("Deleting healthy stateful pod at index 0.")
			sst.deleteStatefulPodAtIndex(0, ss)

			By("Confirming stateful pod at index 0 is recreated.")
			sst.waitForRunningAndReady(2, ss)

			By("Deleting unhealthy stateful pod at index 1.")
			sst.deleteStatefulPodAtIndex(1, ss)

			By("Confirming all stateful pods in statefulset are created.")
			sst.saturate(ss)
		})

		It("should allow template updates", func() {
			By("Creating stateful set " + ssName + " in namespace " + ns)
			*(ss.Spec.Replicas) = 2

			ss, err := c.Apps().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			sst := statefulSetTester{c: c}

			sst.waitForRunningAndReady(*ss.Spec.Replicas, ss)

			newImage := newNginxImage
			oldImage := ss.Spec.Template.Spec.Containers[0].Image
			By(fmt.Sprintf("Updating stateful set template: update image from %s to %s", oldImage, newImage))
			Expect(oldImage).NotTo(Equal(newImage), "Incorrect test setup: should update to a different image")
			_, err = framework.UpdateStatefulSetWithRetries(c, ns, ss.Name, func(update *apps.StatefulSet) {
				update.Spec.Template.Spec.Containers[0].Image = newImage
			})
			Expect(err).NotTo(HaveOccurred())

			updateIndex := 0
			By(fmt.Sprintf("Deleting stateful pod at index %d", updateIndex))
			sst.deleteStatefulPodAtIndex(updateIndex, ss)

			By("Waiting for all stateful pods to be running again")
			sst.waitForRunningAndReady(*ss.Spec.Replicas, ss)

			By(fmt.Sprintf("Verifying stateful pod at index %d is updated", updateIndex))
			verify := func(pod *v1.Pod) {
				podImage := pod.Spec.Containers[0].Image
				Expect(podImage).To(Equal(newImage), fmt.Sprintf("Expected stateful pod image %s updated to %s", podImage, newImage))
			}
			sst.verifyPodAtIndex(updateIndex, ss, verify)
		})

		It("Scaling down before scale up is finished should wait until current pod will be running and ready before it will be removed", func() {
			By("Creating stateful set " + ssName + " in namespace " + ns + ", and pausing scale operations after each pod")
			testProbe := &v1.Probe{Handler: v1.Handler{HTTPGet: &v1.HTTPGetAction{
				Path: "/index.html",
				Port: intstr.IntOrString{IntVal: 80}}}}
			ss := newStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, labels)
			ss.Spec.Template.Spec.Containers[0].ReadinessProbe = testProbe
			setInitializedAnnotation(ss, "false")
			ss, err := c.Apps().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())
			sst := &statefulSetTester{c: c}
			sst.waitForRunningAndReady(1, ss)

			By("Scaling up stateful set " + ssName + " to 3 replicas and pausing after 2nd pod")
			sst.setHealthy(ss)
			sst.updateReplicas(ss, 3)
			sst.waitForRunningAndReady(2, ss)

			By("Before scale up finished setting 2nd pod to be not ready by breaking readiness probe")
			sst.breakProbe(ss, testProbe)
			sst.waitForRunningAndNotReady(2, ss)

			By("Continue scale operation after the 2nd pod, and scaling down to 1 replica")
			sst.setHealthy(ss)
			sst.updateReplicas(ss, 1)

			By("Verifying that the 2nd pod wont be removed if it is not running and ready")
			sst.confirmStatefulPodCount(2, ss, 10*time.Second)
			expectedPodName := ss.Name + "-1"
			expectedPod, err := f.ClientSet.Core().Pods(ns).Get(expectedPodName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			watcher, err := f.ClientSet.Core().Pods(ns).Watch(metav1.SingleObject(
				metav1.ObjectMeta{
					Name:            expectedPod.Name,
					ResourceVersion: expectedPod.ResourceVersion,
				},
			))
			Expect(err).NotTo(HaveOccurred())

			By("Verifying the 2nd pod is removed only when it becomes running and ready")
			sst.restoreProbe(ss, testProbe)
			_, err = watch.Until(statefulsetTimeout, watcher, func(event watch.Event) (bool, error) {
				pod := event.Object.(*v1.Pod)
				if event.Type == watch.Deleted && pod.Name == expectedPodName {
					return false, fmt.Errorf("Pod %v was deleted before enter running", pod.Name)
				}
				framework.Logf("Observed event %v for pod %v. Phase %v, Pod is ready %v",
					event.Type, pod.Name, pod.Status.Phase, v1.IsPodReady(pod))
				if pod.Name != expectedPodName {
					return false, nil
				}
				if pod.Status.Phase == v1.PodRunning && v1.IsPodReady(pod) {
					return true, nil
				}
				return false, nil
			})
			Expect(err).NotTo(HaveOccurred())
		})

		It("Scaling should happen in predictable order and halt if any stateful pod is unhealthy", func() {
			psLabels := klabels.Set(labels)
			By("Initializing watcher for selector " + psLabels.String())
			watcher, err := f.ClientSet.Core().Pods(ns).Watch(metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			Expect(err).NotTo(HaveOccurred())

			By("Creating stateful set " + ssName + " in namespace " + ns)
			testProbe := &v1.Probe{Handler: v1.Handler{HTTPGet: &v1.HTTPGetAction{
				Path: "/index.html",
				Port: intstr.IntOrString{IntVal: 80}}}}
			ss := newStatefulSet(ssName, ns, headlessSvcName, 1, nil, nil, psLabels)
			ss.Spec.Template.Spec.Containers[0].ReadinessProbe = testProbe
			ss, err = c.Apps().StatefulSets(ns).Create(ss)
			Expect(err).NotTo(HaveOccurred())

			By("Waiting until all stateful set " + ssName + " replicas will be running in namespace " + ns)
			sst := &statefulSetTester{c: c}
			sst.waitForRunningAndReady(*ss.Spec.Replicas, ss)

			By("Confirming that stateful set scale up will halt with unhealthy stateful pod")
			sst.breakProbe(ss, testProbe)
			sst.waitForRunningAndNotReady(*ss.Spec.Replicas, ss)
			sst.updateReplicas(ss, 3)
			sst.confirmStatefulPodCount(1, ss, 10*time.Second)

			By("Scaling up stateful set " + ssName + " to 3 replicas and waiting until all of them will be running in namespace " + ns)
			sst.restoreProbe(ss, testProbe)
			sst.waitForRunningAndReady(3, ss)

			By("Verifying that stateful set " + ssName + " was scaled up in order")
			expectedOrder := []string{ssName + "-0", ssName + "-1", ssName + "-2"}
			_, err = watch.Until(statefulsetTimeout, watcher, func(event watch.Event) (bool, error) {
				if event.Type != watch.Added {
					return false, nil
				}
				pod := event.Object.(*v1.Pod)
				if pod.Name == expectedOrder[0] {
					expectedOrder = expectedOrder[1:]
				}
				return len(expectedOrder) == 0, nil

			})
			Expect(err).NotTo(HaveOccurred())

			By("Scale down will halt with unhealthy stateful pod")
			watcher, err = f.ClientSet.Core().Pods(ns).Watch(metav1.ListOptions{
				LabelSelector: psLabels.AsSelector().String(),
			})
			Expect(err).NotTo(HaveOccurred())

			sst.breakProbe(ss, testProbe)
			sst.waitForRunningAndNotReady(3, ss)
			sst.updateReplicas(ss, 0)
			sst.confirmStatefulPodCount(3, ss, 10*time.Second)

			By("Scaling down stateful set " + ssName + " to 0 replicas and waiting until none of pods will run in namespace" + ns)
			sst.restoreProbe(ss, testProbe)
			sst.scale(ss, 0)

			By("Verifying that stateful set " + ssName + " was scaled down in reverse order")
			expectedOrder = []string{ssName + "-2", ssName + "-1", ssName + "-0"}
			_, err = watch.Until(statefulsetTimeout, watcher, func(event watch.Event) (bool, error) {
				if event.Type != watch.Deleted {
					return false, nil
				}
				pod := event.Object.(*v1.Pod)
				if pod.Name == expectedOrder[0] {
					expectedOrder = expectedOrder[1:]
				}
				return len(expectedOrder) == 0, nil

			})
			Expect(err).NotTo(HaveOccurred())
		})

		It("Should recreate evicted statefulset", func() {
			podName := "test-pod"
			statefulPodName := ssName + "-0"
			By("Looking for a node to schedule stateful set and pod")
			nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
			node := nodes.Items[0]

			By("Creating pod with conflicting port in namespace " + f.Namespace.Name)
			conflictingPort := v1.ContainerPort{HostPort: 21017, ContainerPort: 21017, Name: "conflict"}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "gcr.io/google_containers/nginx-slim:0.7",
							Ports: []v1.ContainerPort{conflictingPort},
						},
					},
					NodeName: node.Name,
				},
			}
			pod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(pod)
			framework.ExpectNoError(err)

			By("Creating statefulset with conflicting port in namespace " + f.Namespace.Name)
			ss := newStatefulSet(ssName, f.Namespace.Name, headlessSvcName, 1, nil, nil, labels)
			statefulPodContainer := &ss.Spec.Template.Spec.Containers[0]
			statefulPodContainer.Ports = append(statefulPodContainer.Ports, conflictingPort)
			ss.Spec.Template.Spec.NodeName = node.Name
			_, err = f.ClientSet.Apps().StatefulSets(f.Namespace.Name).Create(ss)
			framework.ExpectNoError(err)

			By("Waiting until pod " + podName + " will start running in namespace " + f.Namespace.Name)
			if err := f.WaitForPodRunning(podName); err != nil {
				framework.Failf("Pod %v did not start running: %v", podName, err)
			}

			var initialStatefulPodUID types.UID
			By("Waiting until stateful pod " + statefulPodName + " will be recreated and deleted at least once in namespace " + f.Namespace.Name)
			w, err := f.ClientSet.Core().Pods(f.Namespace.Name).Watch(metav1.SingleObject(metav1.ObjectMeta{Name: statefulPodName}))
			framework.ExpectNoError(err)
			// we need to get UID from pod in any state and wait until stateful set controller will remove pod atleast once
			_, err = watch.Until(statefulPodTimeout, w, func(event watch.Event) (bool, error) {
				pod := event.Object.(*v1.Pod)
				switch event.Type {
				case watch.Deleted:
					framework.Logf("Observed delete event for stateful pod %v in namespace %v", pod.Name, pod.Namespace)
					if initialStatefulPodUID == "" {
						return false, nil
					}
					return true, nil
				}
				framework.Logf("Observed stateful pod in namespace: %v, name: %v, uid: %v, status phase: %v. Waiting for statefulset controller to delete.",
					pod.Namespace, pod.Name, pod.UID, pod.Status.Phase)
				initialStatefulPodUID = pod.UID
				return false, nil
			})
			if err != nil {
				framework.Failf("Pod %v expected to be re-created at least once", statefulPodName)
			}

			By("Removing pod with conflicting port in namespace " + f.Namespace.Name)
			err = f.ClientSet.Core().Pods(f.Namespace.Name).Delete(pod.Name, metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			By("Waiting when stateful pod " + statefulPodName + " will be recreated in namespace " + f.Namespace.Name + " and will be in running state")
			// we may catch delete event, thats why we are waiting for running phase like this, and not with watch.Until
			Eventually(func() error {
				statefulPod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Get(statefulPodName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if statefulPod.Status.Phase != v1.PodRunning {
					return fmt.Errorf("Pod %v is not in running phase: %v", statefulPod.Name, statefulPod.Status.Phase)
				} else if statefulPod.UID == initialStatefulPodUID {
					return fmt.Errorf("Pod %v wasn't recreated: %v == %v", statefulPod.Name, statefulPod.UID, initialStatefulPodUID)
				}
				return nil
			}, statefulPodTimeout, 2*time.Second).Should(BeNil())
		})
	})

	framework.KubeDescribe("Deploy clustered applications [Feature:StatefulSet] [Slow]", func() {
		var sst *statefulSetTester
		var appTester *clusterAppTester

		BeforeEach(func() {
			sst = &statefulSetTester{c: c}
			appTester = &clusterAppTester{tester: sst, ns: ns}
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed {
				dumpDebugInfo(c, ns)
			}
			framework.Logf("Deleting all statefulset in ns %v", ns)
			deleteAllStatefulSets(c, ns)
		})

		It("should creating a working zookeeper cluster", func() {
			appTester.statefulPod = &zookeeperTester{tester: sst}
			appTester.run()
		})

		It("should creating a working redis cluster", func() {
			appTester.statefulPod = &redisTester{tester: sst}
			appTester.run()
		})

		It("should creating a working mysql cluster", func() {
			appTester.statefulPod = &mysqlGaleraTester{tester: sst}
			appTester.run()
		})

		It("should creating a working CockroachDB cluster", func() {
			appTester.statefulPod = &cockroachDBTester{tester: sst}
			appTester.run()
		})
	})
})

func dumpDebugInfo(c clientset.Interface, ns string) {
	sl, _ := c.Core().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	for _, s := range sl.Items {
		desc, _ := framework.RunKubectl("describe", "po", s.Name, fmt.Sprintf("--namespace=%v", ns))
		framework.Logf("\nOutput of kubectl describe %v:\n%v", s.Name, desc)

		l, _ := framework.RunKubectl("logs", s.Name, fmt.Sprintf("--namespace=%v", ns), "--tail=100")
		framework.Logf("\nLast 100 log lines of %v:\n%v", s.Name, l)
	}
}

func kubectlExecWithRetries(args ...string) (out string) {
	var err error
	for i := 0; i < 3; i++ {
		if out, err = framework.RunKubectl(args...); err == nil {
			return
		}
		framework.Logf("Retrying %v:\nerror %v\nstdout %v", args, err, out)
	}
	framework.Failf("Failed to execute \"%v\" with retries: %v", args, err)
	return
}

type statefulPodTester interface {
	deploy(ns string) *apps.StatefulSet
	write(statefulPodIndex int, kv map[string]string)
	read(statefulPodIndex int, key string) string
	name() string
}

type clusterAppTester struct {
	ns          string
	statefulPod statefulPodTester
	tester      *statefulSetTester
}

func (c *clusterAppTester) run() {
	By("Deploying " + c.statefulPod.name())
	ss := c.statefulPod.deploy(c.ns)

	By("Creating foo:bar in member with index 0")
	c.statefulPod.write(0, map[string]string{"foo": "bar"})

	switch c.statefulPod.(type) {
	case *mysqlGaleraTester:
		// Don't restart MySQL cluster since it doesn't handle restarts well
	default:
		if restartCluster {
			By("Restarting stateful set " + ss.Name)
			c.tester.restart(ss)
			c.tester.waitForRunningAndReady(*ss.Spec.Replicas, ss)
		}
	}

	By("Reading value under foo from member with index 2")
	if err := pollReadWithTimeout(c.statefulPod, 2, "foo", "bar"); err != nil {
		framework.Failf("%v", err)
	}
}

type zookeeperTester struct {
	ss     *apps.StatefulSet
	tester *statefulSetTester
}

func (z *zookeeperTester) name() string {
	return "zookeeper"
}

func (z *zookeeperTester) deploy(ns string) *apps.StatefulSet {
	z.ss = z.tester.createStatefulSet(zookeeperManifestPath, ns)
	return z.ss
}

func (z *zookeeperTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ss.Namespace)
	for k, v := range kv {
		cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh create /%v %v", k, v)
		framework.Logf(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
	}
}

func (z *zookeeperTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", z.ss.Name, statefulPodIndex)
	ns := fmt.Sprintf("--namespace=%v", z.ss.Namespace)
	cmd := fmt.Sprintf("/opt/zookeeper/bin/zkCli.sh get /%v", key)
	return lastLine(framework.RunKubectlOrDie("exec", ns, name, "--", "/bin/sh", "-c", cmd))
}

type mysqlGaleraTester struct {
	ss     *apps.StatefulSet
	tester *statefulSetTester
}

func (m *mysqlGaleraTester) name() string {
	return "mysql: galera"
}

func (m *mysqlGaleraTester) mysqlExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/usr/bin/mysql -u root -B -e '%v'", cmd)
	// TODO: Find a readiness probe for mysql that guarantees writes will
	// succeed and ditch retries. Current probe only reads, so there's a window
	// for a race.
	return kubectlExecWithRetries(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *mysqlGaleraTester) deploy(ns string) *apps.StatefulSet {
	m.ss = m.tester.createStatefulSet(mysqlGaleraManifestPath, ns)

	framework.Logf("Deployed statefulset %v, initializing database", m.ss.Name)
	for _, cmd := range []string{
		"create database statefulset;",
		"use statefulset; create table foo (k varchar(20), v varchar(20));",
	} {
		framework.Logf(m.mysqlExec(cmd, ns, fmt.Sprintf("%v-0", m.ss.Name)))
	}
	return m.ss
}

func (m *mysqlGaleraTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("use  statefulset; insert into foo (k, v) values (\"%v\", \"%v\");", k, v)
		framework.Logf(m.mysqlExec(cmd, m.ss.Namespace, name))
	}
}

func (m *mysqlGaleraTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	return lastLine(m.mysqlExec(fmt.Sprintf("use statefulset; select v from foo where k=\"%v\";", key), m.ss.Namespace, name))
}

type redisTester struct {
	ss     *apps.StatefulSet
	tester *statefulSetTester
}

func (m *redisTester) name() string {
	return "redis: master/slave"
}

func (m *redisTester) redisExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/opt/redis/redis-cli -h %v %v", podName, cmd)
	return framework.RunKubectlOrDie(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (m *redisTester) deploy(ns string) *apps.StatefulSet {
	m.ss = m.tester.createStatefulSet(redisManifestPath, ns)
	return m.ss
}

func (m *redisTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	for k, v := range kv {
		framework.Logf(m.redisExec(fmt.Sprintf("SET %v %v", k, v), m.ss.Namespace, name))
	}
}

func (m *redisTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", m.ss.Name, statefulPodIndex)
	return lastLine(m.redisExec(fmt.Sprintf("GET %v", key), m.ss.Namespace, name))
}

type cockroachDBTester struct {
	ss     *apps.StatefulSet
	tester *statefulSetTester
}

func (c *cockroachDBTester) name() string {
	return "CockroachDB"
}

func (c *cockroachDBTester) cockroachDBExec(cmd, ns, podName string) string {
	cmd = fmt.Sprintf("/cockroach/cockroach sql --host %s.cockroachdb -e \"%v\"", podName, cmd)
	return framework.RunKubectlOrDie(fmt.Sprintf("--namespace=%v", ns), "exec", podName, "--", "/bin/sh", "-c", cmd)
}

func (c *cockroachDBTester) deploy(ns string) *apps.StatefulSet {
	c.ss = c.tester.createStatefulSet(cockroachDBManifestPath, ns)
	framework.Logf("Deployed statefulset %v, initializing database", c.ss.Name)
	for _, cmd := range []string{
		"CREATE DATABASE IF NOT EXISTS foo;",
		"CREATE TABLE IF NOT EXISTS foo.bar (k STRING PRIMARY KEY, v STRING);",
	} {
		framework.Logf(c.cockroachDBExec(cmd, ns, fmt.Sprintf("%v-0", c.ss.Name)))
	}
	return c.ss
}

func (c *cockroachDBTester) write(statefulPodIndex int, kv map[string]string) {
	name := fmt.Sprintf("%v-%d", c.ss.Name, statefulPodIndex)
	for k, v := range kv {
		cmd := fmt.Sprintf("UPSERT INTO foo.bar VALUES ('%v', '%v');", k, v)
		framework.Logf(c.cockroachDBExec(cmd, c.ss.Namespace, name))
	}
}
func (c *cockroachDBTester) read(statefulPodIndex int, key string) string {
	name := fmt.Sprintf("%v-%d", c.ss.Name, statefulPodIndex)
	return lastLine(c.cockroachDBExec(fmt.Sprintf("SELECT v FROM foo.bar WHERE k='%v';", key), c.ss.Namespace, name))
}

func lastLine(out string) string {
	outLines := strings.Split(strings.Trim(out, "\n"), "\n")
	return outLines[len(outLines)-1]
}

func statefulSetFromManifest(fileName, ns string) *apps.StatefulSet {
	var ss apps.StatefulSet
	framework.Logf("Parsing statefulset from %v", fileName)
	data, err := ioutil.ReadFile(fileName)
	Expect(err).NotTo(HaveOccurred())
	json, err := utilyaml.ToJSON(data)
	Expect(err).NotTo(HaveOccurred())

	Expect(runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &ss)).NotTo(HaveOccurred())
	ss.Namespace = ns
	if ss.Spec.Selector == nil {
		ss.Spec.Selector = &metav1.LabelSelector{
			MatchLabels: ss.Spec.Template.Labels,
		}
	}
	return &ss
}

// statefulSetTester has all methods required to test a single statefulset.
type statefulSetTester struct {
	c clientset.Interface
}

func (s *statefulSetTester) createStatefulSet(manifestPath, ns string) *apps.StatefulSet {
	mkpath := func(file string) string {
		return filepath.Join(framework.TestContext.RepoRoot, manifestPath, file)
	}
	ss := statefulSetFromManifest(mkpath("statefulset.yaml"), ns)

	framework.Logf(fmt.Sprintf("creating " + ss.Name + " service"))
	framework.RunKubectlOrDie("create", "-f", mkpath("service.yaml"), fmt.Sprintf("--namespace=%v", ns))

	framework.Logf(fmt.Sprintf("creating statefulset %v/%v with %d replicas and selector %+v", ss.Namespace, ss.Name, *(ss.Spec.Replicas), ss.Spec.Selector))
	framework.RunKubectlOrDie("create", "-f", mkpath("statefulset.yaml"), fmt.Sprintf("--namespace=%v", ns))
	s.waitForRunningAndReady(*ss.Spec.Replicas, ss)
	return ss
}

func (s *statefulSetTester) checkMount(ss *apps.StatefulSet, mountPath string) error {
	for _, cmd := range []string{
		// Print inode, size etc
		fmt.Sprintf("ls -idlhZ %v", mountPath),
		// Print subdirs
		fmt.Sprintf("find %v", mountPath),
		// Try writing
		fmt.Sprintf("touch %v", filepath.Join(mountPath, fmt.Sprintf("%v", time.Now().UnixNano()))),
	} {
		if err := s.execInStatefulPods(ss, cmd); err != nil {
			return fmt.Errorf("failed to execute %v, error: %v", cmd, err)
		}
	}
	return nil
}

func (s *statefulSetTester) execInStatefulPods(ss *apps.StatefulSet, cmd string) error {
	podList := s.getPodList(ss)
	for _, statefulPod := range podList.Items {
		stdout, err := framework.RunHostCmd(statefulPod.Namespace, statefulPod.Name, cmd)
		framework.Logf("stdout of %v on %v: %v", cmd, statefulPod.Name, stdout)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *statefulSetTester) checkHostname(ss *apps.StatefulSet) error {
	cmd := "printf $(hostname)"
	podList := s.getPodList(ss)
	for _, statefulPod := range podList.Items {
		hostname, err := framework.RunHostCmd(statefulPod.Namespace, statefulPod.Name, cmd)
		if err != nil {
			return err
		}
		if hostname != statefulPod.Name {
			return fmt.Errorf("unexpected hostname (%s) and stateful pod name (%s) not equal", hostname, statefulPod.Name)
		}
	}
	return nil
}
func (s *statefulSetTester) saturate(ss *apps.StatefulSet) {
	// TODO: Watch events and check that creation timestamss don't overlap
	var i int32
	for i = 0; i < *(ss.Spec.Replicas); i++ {
		framework.Logf("Waiting for stateful pod at index " + fmt.Sprintf("%v", i+1) + " to enter Running")
		s.waitForRunningAndReady(i+1, ss)
		framework.Logf("Marking stateful pod at index " + fmt.Sprintf("%v", i) + " healthy")
		s.setHealthy(ss)
	}
}

func (s *statefulSetTester) deleteStatefulPodAtIndex(index int, ss *apps.StatefulSet) {
	name := getPodNameAtIndex(index, ss)
	noGrace := int64(0)
	if err := s.c.Core().Pods(ss.Namespace).Delete(name, &metav1.DeleteOptions{GracePeriodSeconds: &noGrace}); err != nil {
		framework.Failf("Failed to delete stateful pod %v for StatefulSet %v/%v: %v", name, ss.Namespace, ss.Name, err)
	}
}

type verifyPodFunc func(*v1.Pod)

func (s *statefulSetTester) verifyPodAtIndex(index int, ss *apps.StatefulSet, verify verifyPodFunc) {
	name := getPodNameAtIndex(index, ss)
	pod, err := s.c.Core().Pods(ss.Namespace).Get(name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to get stateful pod %s for StatefulSet %s/%s", name, ss.Namespace, ss.Name))
	verify(pod)
}

func getPodNameAtIndex(index int, ss *apps.StatefulSet) string {
	// TODO: we won't use "-index" as the name strategy forever,
	// pull the name out from an identity mapper.
	return fmt.Sprintf("%v-%v", ss.Name, index)
}

func (s *statefulSetTester) scale(ss *apps.StatefulSet, count int32) error {
	name := ss.Name
	ns := ss.Namespace
	s.update(ns, name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = count })

	var statefulPodList *v1.PodList
	pollErr := wait.PollImmediate(statefulsetPoll, statefulsetTimeout, func() (bool, error) {
		statefulPodList = s.getPodList(ss)
		if int32(len(statefulPodList.Items)) == count {
			return true, nil
		}
		return false, nil
	})
	if pollErr != nil {
		unhealthy := []string{}
		for _, statefulPod := range statefulPodList.Items {
			delTs, phase, readiness := statefulPod.DeletionTimestamp, statefulPod.Status.Phase, v1.IsPodReady(&statefulPod)
			if delTs != nil || phase != v1.PodRunning || !readiness {
				unhealthy = append(unhealthy, fmt.Sprintf("%v: deletion %v, phase %v, readiness %v", statefulPod.Name, delTs, phase, readiness))
			}
		}
		return fmt.Errorf("Failed to scale statefulset to %d in %v. Remaining pods:\n%v", count, statefulsetTimeout, unhealthy)
	}
	return nil
}

func (s *statefulSetTester) updateReplicas(ss *apps.StatefulSet, count int32) {
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { ss.Spec.Replicas = &count })
}

func (s *statefulSetTester) restart(ss *apps.StatefulSet) {
	oldReplicas := *(ss.Spec.Replicas)
	framework.ExpectNoError(s.scale(ss, 0))
	s.update(ss.Namespace, ss.Name, func(ss *apps.StatefulSet) { *(ss.Spec.Replicas) = oldReplicas })
}

func (s *statefulSetTester) update(ns, name string, update func(ss *apps.StatefulSet)) {
	for i := 0; i < 3; i++ {
		ss, err := s.c.Apps().StatefulSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			framework.Failf("failed to get statefulset %q: %v", name, err)
		}
		update(ss)
		ss, err = s.c.Apps().StatefulSets(ns).Update(ss)
		if err == nil {
			return
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			framework.Failf("failed to update statefulset %q: %v", name, err)
		}
	}
	framework.Failf("too many retries draining statefulset %q", name)
}

func (s *statefulSetTester) getPodList(ss *apps.StatefulSet) *v1.PodList {
	selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector)
	framework.ExpectNoError(err)
	podList, err := s.c.Core().Pods(ss.Namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
	framework.ExpectNoError(err)
	return podList
}

func (s *statefulSetTester) confirmStatefulPodCount(count int, ss *apps.StatefulSet, timeout time.Duration) {
	start := time.Now()
	deadline := start.Add(timeout)
	for t := time.Now(); t.Before(deadline); t = time.Now() {
		podList := s.getPodList(ss)
		statefulPodCount := len(podList.Items)
		if statefulPodCount != count {
			framework.Failf("StatefulSet %v scaled unexpectedly scaled to %d -> %d replicas: %+v", ss.Name, count, len(podList.Items), podList)
		}
		framework.Logf("Verifying statefulset %v doesn't scale past %d for another %+v", ss.Name, count, deadline.Sub(t))
		time.Sleep(1 * time.Second)
	}
}

func (s *statefulSetTester) waitForRunning(numStatefulPods int32, ss *apps.StatefulSet, shouldBeReady bool) {
	pollErr := wait.PollImmediate(statefulsetPoll, statefulsetTimeout,
		func() (bool, error) {
			podList := s.getPodList(ss)
			if int32(len(podList.Items)) < numStatefulPods {
				framework.Logf("Found %d stateful pods, waiting for %d", len(podList.Items), numStatefulPods)
				return false, nil
			}
			if int32(len(podList.Items)) > numStatefulPods {
				return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numStatefulPods, len(podList.Items))
			}
			for _, p := range podList.Items {
				isReady := v1.IsPodReady(&p)
				desiredReadiness := shouldBeReady == isReady
				framework.Logf("Waiting for pod %v to enter %v - Ready=%v, currently %v - Ready=%v", p.Name, v1.PodRunning, shouldBeReady, p.Status.Phase, isReady)
				if p.Status.Phase != v1.PodRunning || !desiredReadiness {
					return false, nil
				}
			}
			return true, nil
		})
	if pollErr != nil {
		framework.Failf("Failed waiting for pods to enter running: %v", pollErr)
	}
}

func (s *statefulSetTester) waitForRunningAndReady(numStatefulPods int32, ss *apps.StatefulSet) {
	s.waitForRunning(numStatefulPods, ss, true)
}

func (s *statefulSetTester) waitForRunningAndNotReady(numStatefulPods int32, ss *apps.StatefulSet) {
	s.waitForRunning(numStatefulPods, ss, false)
}

func (s *statefulSetTester) breakProbe(ss *apps.StatefulSet, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /usr/share/nginx/html%v /tmp/", path)
	return s.execInStatefulPods(ss, cmd)
}

func (s *statefulSetTester) restoreProbe(ss *apps.StatefulSet, probe *v1.Probe) error {
	path := probe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("Path expected to be not empty: %v", path)
	}
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/share/nginx/html/", path)
	return s.execInStatefulPods(ss, cmd)
}

func (s *statefulSetTester) setHealthy(ss *apps.StatefulSet) {
	podList := s.getPodList(ss)
	markedHealthyPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning {
			framework.Failf("Found pod in %v cannot set health", pod.Status.Phase)
		}
		if isInitialized(pod) {
			continue
		}
		if markedHealthyPod != "" {
			framework.Failf("Found multiple non-healthy stateful pods: %v and %v", pod.Name, markedHealthyPod)
		}
		p, err := framework.UpdatePodWithRetries(s.c, pod.Namespace, pod.Name, func(update *v1.Pod) {
			update.Annotations[apps.StatefulSetInitAnnotation] = "true"
		})
		framework.ExpectNoError(err)
		framework.Logf("Set annotation %v to %v on pod %v", apps.StatefulSetInitAnnotation, p.Annotations[apps.StatefulSetInitAnnotation], pod.Name)
		markedHealthyPod = pod.Name
	}
}

func (s *statefulSetTester) waitForStatus(ss *apps.StatefulSet, expectedReplicas int32) {
	framework.Logf("Waiting for statefulset status.replicas updated to %d", expectedReplicas)

	ns, name := ss.Namespace, ss.Name
	pollErr := wait.PollImmediate(statefulsetPoll, statefulsetTimeout,
		func() (bool, error) {
			ssGet, err := s.c.Apps().StatefulSets(ns).Get(name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if ssGet.Status.Replicas != expectedReplicas {
				framework.Logf("Waiting for stateful set status to become %d, currently %d", expectedReplicas, ssGet.Status.Replicas)
				return false, nil
			}
			return true, nil
		})
	if pollErr != nil {
		framework.Failf("Failed waiting for stateful set status.replicas updated to %d: %v", expectedReplicas, pollErr)
	}
}

func (p *statefulSetTester) checkServiceName(ps *apps.StatefulSet, expectedServiceName string) error {
	framework.Logf("Checking if statefulset spec.serviceName is %s", expectedServiceName)

	if expectedServiceName != ps.Spec.ServiceName {
		return fmt.Errorf("Wrong service name governing statefulset. Expected %s got %s", expectedServiceName, ps.Spec.ServiceName)
	}

	return nil
}

func deleteAllStatefulSets(c clientset.Interface, ns string) {
	sst := &statefulSetTester{c: c}
	ssList, err := c.Apps().StatefulSets(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	framework.ExpectNoError(err)

	// Scale down each statefulset, then delete it completely.
	// Deleting a pvc without doing this will leak volumes, #25101.
	errList := []string{}
	for _, ss := range ssList.Items {
		framework.Logf("Scaling statefulset %v to 0", ss.Name)
		if err := sst.scale(&ss, 0); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
		sst.waitForStatus(&ss, 0)
		framework.Logf("Deleting statefulset %v", ss.Name)
		if err := c.Apps().StatefulSets(ss.Namespace).Delete(ss.Name, nil); err != nil {
			errList = append(errList, fmt.Sprintf("%v", err))
		}
	}

	// pvs are global, so we need to wait for the exact ones bound to the statefulset pvcs.
	pvNames := sets.NewString()
	// TODO: Don't assume all pvcs in the ns belong to a statefulset
	pvcPollErr := wait.PollImmediate(statefulsetPoll, statefulsetTimeout, func() (bool, error) {
		pvcList, err := c.Core().PersistentVolumeClaims(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvcs, retrying %v", err)
			return false, nil
		}
		for _, pvc := range pvcList.Items {
			pvNames.Insert(pvc.Spec.VolumeName)
			// TODO: Double check that there are no pods referencing the pvc
			framework.Logf("Deleting pvc: %v with volume %v", pvc.Name, pvc.Spec.VolumeName)
			if err := c.Core().PersistentVolumeClaims(ns).Delete(pvc.Name, nil); err != nil {
				return false, nil
			}
		}
		return true, nil
	})
	if pvcPollErr != nil {
		errList = append(errList, "Timeout waiting for pvc deletion.")
	}

	pollErr := wait.PollImmediate(statefulsetPoll, statefulsetTimeout, func() (bool, error) {
		pvList, err := c.Core().PersistentVolumes().List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
		if err != nil {
			framework.Logf("WARNING: Failed to list pvs, retrying %v", err)
			return false, nil
		}
		waitingFor := []string{}
		for _, pv := range pvList.Items {
			if pvNames.Has(pv.Name) {
				waitingFor = append(waitingFor, fmt.Sprintf("%v: %+v", pv.Name, pv.Status))
			}
		}
		if len(waitingFor) == 0 {
			return true, nil
		}
		framework.Logf("Still waiting for pvs of statefulset to disappear:\n%v", strings.Join(waitingFor, "\n"))
		return false, nil
	})
	if pollErr != nil {
		errList = append(errList, "Timeout waiting for pv provisioner to delete pvs, this might mean the test leaked pvs.")
	}
	if len(errList) != 0 {
		framework.ExpectNoError(fmt.Errorf("%v", strings.Join(errList, "\n")))
	}
}

func pollReadWithTimeout(statefulPod statefulPodTester, statefulPodNumber int, key, expectedVal string) error {
	err := wait.PollImmediate(time.Second, readTimeout, func() (bool, error) {
		val := statefulPod.read(statefulPodNumber, key)
		if val == "" {
			return false, nil
		} else if val != expectedVal {
			return false, fmt.Errorf("expected value %v, found %v", expectedVal, val)
		}
		return true, nil
	})

	if err == wait.ErrWaitTimeout {
		return fmt.Errorf("timed out when trying to read value for key %v from stateful pod %d", key, statefulPodNumber)
	}
	return err
}

func isInitialized(pod v1.Pod) bool {
	initialized, ok := pod.Annotations[apps.StatefulSetInitAnnotation]
	if !ok {
		return false
	}
	inited, err := strconv.ParseBool(initialized)
	if err != nil {
		framework.Failf("Couldn't parse statefulset init annotations %v", initialized)
	}
	return inited
}

func newPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Annotations: map[string]string{
				"volume.alpha.kubernetes.io/storage-class": "anything",
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

func newStatefulSet(name, ns, governingSvcName string, replicas int32, statefulPodMounts []v1.VolumeMount, podMounts []v1.VolumeMount, labels map[string]string) *apps.StatefulSet {
	mounts := append(statefulPodMounts, podMounts...)
	claims := []v1.PersistentVolumeClaim{}
	for _, m := range statefulPodMounts {
		claims = append(claims, newPVC(m.Name))
	}

	vols := []v1.Volume{}
	for _, m := range podMounts {
		vols = append(vols, v1.Volume{
			Name: m.Name,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	privileged := true

	return &apps.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: apps.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "nginx",
							Image:        nginxImage,
							VolumeMounts: mounts,
							SecurityContext: &v1.SecurityContext{
								Privileged: &privileged,
							},
						},
					},
					Volumes: vols,
				},
			},
			VolumeClaimTemplates: claims,
			ServiceName:          governingSvcName,
		},
	}
}

func setInitializedAnnotation(ss *apps.StatefulSet, value string) {
	ss.Spec.Template.ObjectMeta.Annotations["pod.alpha.kubernetes.io/initialized"] = value
}
