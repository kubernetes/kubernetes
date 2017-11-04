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
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	rbacv1beta1 "k8s.io/api/rbac/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/generated"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	serverStartTimeout = framework.PodStartTimeout + 3*time.Minute
)

var _ = framework.KubeDescribe("[Feature:Example]", func() {
	f := framework.NewDefaultFramework("examples")

	// Reusable cluster state function.  This won't be adversly affected by lazy initialization of framework.
	clusterState := func(selectorKey string, selectorValue string) *framework.ClusterVerification {
		return f.NewClusterVerification(
			f.Namespace,
			framework.PodStateVerification{
				Selectors:   map[string]string{selectorKey: selectorValue},
				ValidPhases: []v1.PodPhase{v1.PodRunning},
			})
	}
	// Customized ForEach wrapper for this test.
	forEachPod := func(selectorKey string, selectorValue string, fn func(v1.Pod)) {
		clusterState(selectorKey, selectorValue).ForEach(fn)
	}
	var c clientset.Interface
	var ns string
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		// this test wants powerful permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		framework.BindClusterRoleInNamespace(c.RbacV1beta1(), "edit", f.Namespace.Name,
			rbacv1beta1.Subject{Kind: rbacv1beta1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})

		err := framework.WaitForAuthorizationUpdate(c.AuthorizationV1beta1(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			f.Namespace.Name, "create", schema.GroupResource{Resource: "pods"}, true)
		framework.ExpectNoError(err)
	})

	framework.KubeDescribe("Redis", func() {
		It("should create and stop redis servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storage/redis", file)
			}
			bootstrapYaml := mkpath("redis-master.yaml")
			sentinelServiceYaml := mkpath("redis-sentinel-service.yaml")
			sentinelControllerYaml := mkpath("redis-sentinel-controller.yaml")
			controllerYaml := mkpath("redis-controller.yaml")

			bootstrapPodName := "redis-master"
			redisRC := "redis"
			sentinelRC := "redis-sentinel"
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			expectedOnServer := "The server is now ready to accept connections"
			expectedOnSentinel := "+monitor master"

			By("starting redis bootstrap")
			framework.RunKubectlOrDie("create", "-f", bootstrapYaml, nsFlag)
			err := framework.WaitForPodNameRunningInNamespace(c, bootstrapPodName, ns)
			Expect(err).NotTo(HaveOccurred())

			_, err = framework.LookForStringInLog(ns, bootstrapPodName, "master", expectedOnServer, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = framework.LookForStringInLog(ns, bootstrapPodName, "sentinel", expectedOnSentinel, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("setting up services and controllers")
			framework.RunKubectlOrDie("create", "-f", sentinelServiceYaml, nsFlag)
			framework.RunKubectlOrDie("create", "-f", sentinelControllerYaml, nsFlag)
			framework.RunKubectlOrDie("create", "-f", controllerYaml, nsFlag)
			label := labels.SelectorFromSet(labels.Set(map[string]string{sentinelRC: "true"}))
			err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())
			label = labels.SelectorFromSet(labels.Set(map[string]string{"name": redisRC}))
			err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())

			By("scaling up the deployment")
			framework.RunKubectlOrDie("scale", "rc", redisRC, "--replicas=3", nsFlag)
			framework.RunKubectlOrDie("scale", "rc", sentinelRC, "--replicas=3", nsFlag)
			framework.WaitForRCToStabilize(c, ns, redisRC, framework.PodReadyBeforeTimeout)
			framework.WaitForRCToStabilize(c, ns, sentinelRC, framework.PodReadyBeforeTimeout)

			By("checking up the services")
			checkAllLogs := func() {
				selectorKey, selectorValue := "name", redisRC
				label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
				err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
				Expect(err).NotTo(HaveOccurred())
				forEachPod(selectorKey, selectorValue, func(pod v1.Pod) {
					if pod.Name != bootstrapPodName {
						_, err := framework.LookForStringInLog(ns, pod.Name, "redis", expectedOnServer, serverStartTimeout)
						Expect(err).NotTo(HaveOccurred())
					}
				})
				selectorKey, selectorValue = sentinelRC, "true"
				label = labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
				err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
				Expect(err).NotTo(HaveOccurred())
				forEachPod(selectorKey, selectorValue, func(pod v1.Pod) {
					if pod.Name != bootstrapPodName {
						_, err := framework.LookForStringInLog(ns, pod.Name, "sentinel", expectedOnSentinel, serverStartTimeout)
						Expect(err).NotTo(HaveOccurred())
					}
				})
			}
			checkAllLogs()

			By("turning down bootstrap")
			framework.RunKubectlOrDie("delete", "-f", bootstrapYaml, nsFlag)
			err = framework.WaitForRCPodToDisappear(c, ns, redisRC, bootstrapPodName)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for the new master election")
			checkAllLogs()
		})
	})

	framework.KubeDescribe("Spark", func() {
		It("should start spark master, driver and workers", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/spark", file)
			}

			// TODO: Add Zepplin and Web UI to this example.
			serviceYaml := mkpath("spark-master-service.yaml")
			masterYaml := mkpath("spark-master-controller.yaml")
			workerControllerYaml := mkpath("spark-worker-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			master := func() {
				By("starting master")
				framework.RunKubectlOrDie("create", "-f", serviceYaml, nsFlag)
				framework.RunKubectlOrDie("create", "-f", masterYaml, nsFlag)
				selectorKey, selectorValue := "component", "spark-master"
				label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
				err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
				Expect(err).NotTo(HaveOccurred())

				framework.Logf("Now polling for Master startup...")
				// Only one master pod: But its a natural way to look up pod names.
				forEachPod(selectorKey, selectorValue, func(pod v1.Pod) {
					framework.Logf("Now waiting for master to startup in %v", pod.Name)
					_, err := framework.LookForStringInLog(ns, pod.Name, "spark-master", "Starting Spark master at", serverStartTimeout)
					Expect(err).NotTo(HaveOccurred())
				})

				By("waiting for master endpoint")
				err = framework.WaitForEndpoint(c, ns, "spark-master")
				Expect(err).NotTo(HaveOccurred())
				forEachPod(selectorKey, selectorValue, func(pod v1.Pod) {
					_, maErr := framework.LookForStringInLog(f.Namespace.Name, pod.Name, "spark-master", "Starting Spark master at", serverStartTimeout)
					if maErr != nil {
						framework.Failf("Didn't find target string. error:", maErr)
					}
				})
			}
			worker := func() {
				By("starting workers")
				framework.Logf("Now starting Workers")
				framework.RunKubectlOrDie("create", "-f", workerControllerYaml, nsFlag)
				selectorKey, selectorValue := "component", "spark-worker"
				label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
				err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
				Expect(err).NotTo(HaveOccurred())

				// For now, scaling is orthogonal to the core test.
				// framework.ScaleRC(c, ns, "spark-worker-controller", 2, true)

				framework.Logf("Now polling for worker startup...")
				forEachPod(selectorKey, selectorValue,
					func(pod v1.Pod) {
						_, slaveErr := framework.LookForStringInLog(ns, pod.Name, "spark-worker", "Successfully registered with master", serverStartTimeout)
						Expect(slaveErr).NotTo(HaveOccurred())
					})
			}
			// Run the worker verification after we turn up the master.
			defer worker()
			master()
		})
	})

	framework.KubeDescribe("Cassandra", func() {
		It("should create and scale cassandra", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storage/cassandra", file)
			}
			serviceYaml := mkpath("cassandra-service.yaml")
			controllerYaml := mkpath("cassandra-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("Starting the cassandra service")
			framework.RunKubectlOrDie("create", "-f", serviceYaml, nsFlag)
			framework.Logf("wait for service")
			err := framework.WaitForService(c, ns, "cassandra", true, framework.Poll, framework.ServiceRespondingTimeout)
			Expect(err).NotTo(HaveOccurred())

			// Create an RC with n nodes in it.  Each node will then be verified.
			By("Creating a Cassandra RC")
			framework.RunKubectlOrDie("create", "-f", controllerYaml, nsFlag)
			label := labels.SelectorFromSet(labels.Set(map[string]string{"app": "cassandra"}))
			err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())
			forEachPod("app", "cassandra", func(pod v1.Pod) {
				framework.Logf("Verifying pod %v ", pod.Name)
				// TODO how do we do this better?  Ready Probe?
				_, err = framework.LookForStringInLog(ns, pod.Name, "cassandra", "Starting listening for CQL clients", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			By("Finding each node in the nodetool status lines")
			forEachPod("app", "cassandra", func(pod v1.Pod) {
				output := framework.RunKubectlOrDie("exec", pod.Name, nsFlag, "--", "nodetool", "status")
				matched, _ := regexp.MatchString("UN.*"+pod.Status.PodIP, output)
				if matched != true {
					framework.Failf("Cassandra pod ip %s is not reporting Up and Normal 'UN' via nodetool status", pod.Status.PodIP)
				}
			})
		})
	})

	framework.KubeDescribe("CassandraStatefulSet", func() {
		It("should create statefulset", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storage/cassandra", file)
			}
			serviceYaml := mkpath("cassandra-service.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			// have to change dns prefix because of the dynamic namespace
			input := generated.ReadOrDie(mkpath("cassandra-statefulset.yaml"))

			output := strings.Replace(string(input), "cassandra-0.cassandra.default.svc.cluster.local", "cassandra-0.cassandra."+ns+".svc.cluster.local", -1)

			statefulsetYaml := "/tmp/cassandra-statefulset.yaml"

			err := ioutil.WriteFile(statefulsetYaml, []byte(output), 0644)
			Expect(err).NotTo(HaveOccurred())

			By("Starting the cassandra service")
			framework.RunKubectlOrDie("create", "-f", serviceYaml, nsFlag)
			framework.Logf("wait for service")
			err = framework.WaitForService(c, ns, "cassandra", true, framework.Poll, framework.ServiceRespondingTimeout)
			Expect(err).NotTo(HaveOccurred())

			// Create an StatefulSet with n nodes in it.  Each node will then be verified.
			By("Creating a Cassandra StatefulSet")

			framework.RunKubectlOrDie("create", "-f", statefulsetYaml, nsFlag)

			statefulsetPoll := 30 * time.Second
			statefulsetTimeout := 10 * time.Minute
			// TODO - parse this number out of the yaml
			numPets := 3
			label := labels.SelectorFromSet(labels.Set(map[string]string{"app": "cassandra"}))
			err = wait.PollImmediate(statefulsetPoll, statefulsetTimeout,
				func() (bool, error) {
					podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: label.String()})
					if err != nil {
						return false, fmt.Errorf("Unable to get list of pods in statefulset %s", label)
					}
					framework.ExpectNoError(err)
					if len(podList.Items) < numPets {
						framework.Logf("Found %d pets, waiting for %d", len(podList.Items), numPets)
						return false, nil
					}
					if len(podList.Items) > numPets {
						return false, fmt.Errorf("Too many pods scheduled, expected %d got %d", numPets, len(podList.Items))
					}
					for _, p := range podList.Items {
						isReady := podutil.IsPodReady(&p)
						if p.Status.Phase != v1.PodRunning || !isReady {
							framework.Logf("Waiting for pod %v to enter %v - Ready=True, currently %v - Ready=%v", p.Name, v1.PodRunning, p.Status.Phase, isReady)
							return false, nil
						}
					}
					return true, nil
				})
			Expect(err).NotTo(HaveOccurred())

			By("Finding each node in the nodetool status lines")
			forEachPod("app", "cassandra", func(pod v1.Pod) {
				output := framework.RunKubectlOrDie("exec", pod.Name, nsFlag, "--", "nodetool", "status")
				matched, _ := regexp.MatchString("UN.*"+pod.Status.PodIP, output)
				if matched != true {
					framework.Failf("Cassandra pod ip %s is not reporting Up and Normal 'UN' via nodetool status", pod.Status.PodIP)
				}
			})
			// using out of statefulset e2e as deleting pvc is a pain
			framework.DeleteAllStatefulSets(c, ns)
		})
	})

	framework.KubeDescribe("Storm", func() {
		It("should create and stop Zookeeper, Nimbus and Storm worker servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storm", file)
			}
			zookeeperServiceJson := mkpath("zookeeper-service.json")
			zookeeperPodJson := mkpath("zookeeper.json")
			nimbusServiceJson := mkpath("storm-nimbus-service.json")
			nimbusPodJson := mkpath("storm-nimbus.json")
			workerControllerJson := mkpath("storm-worker-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			zookeeperPod := "zookeeper"
			nimbusPod := "nimbus"

			By("starting Zookeeper")
			framework.RunKubectlOrDie("create", "-f", zookeeperPodJson, nsFlag)
			framework.RunKubectlOrDie("create", "-f", zookeeperServiceJson, nsFlag)
			err := f.WaitForPodRunningSlow(zookeeperPod)
			Expect(err).NotTo(HaveOccurred())

			By("checking if zookeeper is up and running")
			_, err = framework.LookForStringInLog(ns, zookeeperPod, "zookeeper", "binding to port", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForEndpoint(c, ns, "zookeeper")
			Expect(err).NotTo(HaveOccurred())

			By("starting Nimbus")
			framework.RunKubectlOrDie("create", "-f", nimbusPodJson, nsFlag)
			framework.RunKubectlOrDie("create", "-f", nimbusServiceJson, nsFlag)
			err = f.WaitForPodRunningSlow(nimbusPod)
			Expect(err).NotTo(HaveOccurred())

			err = framework.WaitForEndpoint(c, ns, "nimbus")
			Expect(err).NotTo(HaveOccurred())

			By("starting workers")
			framework.RunKubectlOrDie("create", "-f", workerControllerJson, nsFlag)
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": "storm-worker"}))
			err = testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())
			forEachPod("name", "storm-worker", func(pod v1.Pod) {
				//do nothing, just wait for the pod to be running
			})
			// TODO: Add logging configuration to nimbus & workers images and then
			// look for a string instead of sleeping.
			time.Sleep(20 * time.Second)

			By("checking if there are established connections to Zookeeper")
			_, err = framework.LookForStringInLog(ns, zookeeperPod, "zookeeper", "Established session", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("checking if Nimbus responds to requests")
			framework.LookForString("No topologies running.", time.Minute, func() string {
				return framework.RunKubectlOrDie("exec", "nimbus", nsFlag, "--", "bin/storm", "list")
			})
		})
	})

	framework.KubeDescribe("Liveness", func() {
		It("liveness pods should be automatically restarted", func() {
			mkpath := func(file string) string {
				path := filepath.Join("test/fixtures/doc-yaml/user-guide/liveness", file)
				framework.ExpectNoError(createFileForGoBinData(path, path))
				return path
			}
			execYaml := mkpath("exec-liveness.yaml")
			httpYaml := mkpath("http-liveness.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			framework.RunKubectlOrDie("create", "-f", filepath.Join(framework.TestContext.OutputDir, execYaml), nsFlag)
			framework.RunKubectlOrDie("create", "-f", filepath.Join(framework.TestContext.OutputDir, httpYaml), nsFlag)

			// Since both containers start rapidly, we can easily run this test in parallel.
			var wg sync.WaitGroup
			passed := true
			checkRestart := func(podName string, timeout time.Duration) {
				err := framework.WaitForPodNameRunningInNamespace(c, podName, ns)
				Expect(err).NotTo(HaveOccurred())
				for t := time.Now(); time.Since(t) < timeout; time.Sleep(framework.Poll) {
					pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
					framework.ExpectNoError(err, fmt.Sprintf("getting pod %s", podName))
					stat := podutil.GetExistingContainerStatus(pod.Status.ContainerStatuses, podName)
					framework.Logf("Pod: %s, restart count:%d", stat.Name, stat.RestartCount)
					if stat.RestartCount > 0 {
						framework.Logf("Saw %v restart, succeeded...", podName)
						wg.Done()
						return
					}
				}
				framework.Logf("Failed waiting for %v restart! ", podName)
				passed = false
				wg.Done()
			}

			By("Check restarts")

			// Start the "actual test", and wait for both pods to complete.
			// If 2 fail: Something is broken with the test (or maybe even with liveness).
			// If 1 fails: Its probably just an error in the examples/ files themselves.
			wg.Add(2)
			for _, c := range []string{"liveness-http", "liveness-exec"} {
				go checkRestart(c, 2*time.Minute)
			}
			wg.Wait()
			if !passed {
				framework.Failf("At least one liveness example failed.  See the logs above.")
			}
		})
	})

	framework.KubeDescribe("Secret", func() {
		It("should create a pod that reads a secret", func() {
			mkpath := func(file string) string {
				path := filepath.Join("test/fixtures/doc-yaml/user-guide/secrets", file)
				framework.ExpectNoError(createFileForGoBinData(path, path))
				return path
			}
			secretYaml := mkpath("secret.yaml")
			podYaml := mkpath("secret-pod.yaml")

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "secret-test-pod"

			By("creating secret and pod")
			framework.RunKubectlOrDie("create", "-f", filepath.Join(framework.TestContext.OutputDir, secretYaml), nsFlag)
			framework.RunKubectlOrDie("create", "-f", filepath.Join(framework.TestContext.OutputDir, podYaml), nsFlag)
			err := framework.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if secret was read correctly")
			_, err = framework.LookForStringInLog(ns, "secret-test-pod", "test-container", "value-1", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	framework.KubeDescribe("Downward API", func() {
		It("should create a pod that prints his name and namespace", func() {
			mkpath := func(file string) string {
				path := filepath.Join("test/fixtures/doc-yaml/user-guide/downward-api", file)
				framework.ExpectNoError(createFileForGoBinData(path, path))
				return path
			}
			podYaml := mkpath("dapi-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "dapi-test-pod"

			By("creating the pod")
			framework.RunKubectlOrDie("create", "-f", filepath.Join(framework.TestContext.OutputDir, podYaml), nsFlag)
			err := framework.WaitForPodNoLongerRunningInNamespace(c, podName, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if name and namespace were passed correctly")
			_, err = framework.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAMESPACE=%v", ns), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = framework.LookForStringInLog(ns, podName, "test-container", fmt.Sprintf("MY_POD_NAME=%v", podName), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	framework.KubeDescribe("RethinkDB", func() {
		It("should create and stop rethinkdb servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storage/rethinkdb", file)
			}
			driverServiceYaml := mkpath("driver-service.yaml")
			rethinkDbControllerYaml := mkpath("rc.yaml")
			adminPodYaml := mkpath("admin-pod.yaml")
			adminServiceYaml := mkpath("admin-service.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting rethinkdb")
			framework.RunKubectlOrDie("create", "-f", driverServiceYaml, nsFlag)
			framework.RunKubectlOrDie("create", "-f", rethinkDbControllerYaml, nsFlag)
			label := labels.SelectorFromSet(labels.Set(map[string]string{"db": "rethinkdb"}))
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())
			checkDbInstances := func() {
				forEachPod("db", "rethinkdb", func(pod v1.Pod) {
					_, err = framework.LookForStringInLog(ns, pod.Name, "rethinkdb", "Server ready", serverStartTimeout)
					Expect(err).NotTo(HaveOccurred())
				})
			}
			checkDbInstances()
			err = framework.WaitForEndpoint(c, ns, "rethinkdb-driver")
			Expect(err).NotTo(HaveOccurred())

			By("scaling rethinkdb")
			framework.ScaleRC(f.ClientSet, f.InternalClientset, ns, "rethinkdb-rc", 2, true)
			checkDbInstances()

			By("starting admin")
			framework.RunKubectlOrDie("create", "-f", adminServiceYaml, nsFlag)
			framework.RunKubectlOrDie("create", "-f", adminPodYaml, nsFlag)
			err = framework.WaitForPodNameRunningInNamespace(c, "rethinkdb-admin", ns)
			Expect(err).NotTo(HaveOccurred())
			checkDbInstances()
			content, err := makeHttpRequestToService(c, ns, "rethinkdb-admin", "/", framework.EndpointRegisterTimeout)
			Expect(err).NotTo(HaveOccurred())
			if !strings.Contains(content, "<title>RethinkDB Administration Console</title>") {
				framework.Failf("RethinkDB console is not running")
			}
		})
	})

	framework.KubeDescribe("Hazelcast", func() {
		It("should create and scale hazelcast", func() {
			mkpath := func(file string) string {
				return filepath.Join(framework.TestContext.RepoRoot, "examples/storage/hazelcast", file)
			}
			serviceYaml := mkpath("hazelcast-service.yaml")
			deploymentYaml := mkpath("hazelcast-deployment.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting hazelcast")
			framework.RunKubectlOrDie("create", "-f", serviceYaml, nsFlag)
			framework.RunKubectlOrDie("create", "-f", deploymentYaml, nsFlag)
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": "hazelcast"}))
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			Expect(err).NotTo(HaveOccurred())
			forEachPod("name", "hazelcast", func(pod v1.Pod) {
				_, err := framework.LookForStringInLog(ns, pod.Name, "hazelcast", "Members [1]", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
				_, err = framework.LookForStringInLog(ns, pod.Name, "hazelcast", "is STARTED", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			err = framework.WaitForEndpoint(c, ns, "hazelcast")
			Expect(err).NotTo(HaveOccurred())

			By("scaling hazelcast")
			framework.ScaleRC(f.ClientSet, f.InternalClientset, ns, "hazelcast", 2, true)
			forEachPod("name", "hazelcast", func(pod v1.Pod) {
				_, err := framework.LookForStringInLog(ns, pod.Name, "hazelcast", "Members [2]", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})

func makeHttpRequestToService(c clientset.Interface, ns, service, path string, timeout time.Duration) (string, error) {
	var result []byte
	var err error
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(framework.Poll) {
		proxyRequest, errProxy := framework.GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
		if errProxy != nil {
			break
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		result, err = proxyRequest.Namespace(ns).
			Context(ctx).
			Name(service).
			Suffix(path).
			Do().
			Raw()
		if err != nil {
			break
		}
	}
	return string(result), err
}

func createFileForGoBinData(gobindataPath, outputFilename string) error {
	data := generated.ReadOrDie(gobindataPath)
	if len(data) == 0 {
		return fmt.Errorf("Failed to read gobindata from %v", gobindataPath)
	}
	fullPath := filepath.Join(framework.TestContext.OutputDir, outputFilename)
	err := os.MkdirAll(filepath.Dir(fullPath), 0777)
	if err != nil {
		return fmt.Errorf("Error while creating directory %v: %v", filepath.Dir(fullPath), err)
	}
	err = ioutil.WriteFile(fullPath, data, 0644)
	if err != nil {
		return fmt.Errorf("Error while trying to write to file %v: %v", fullPath, err)
	}
	return nil
}
