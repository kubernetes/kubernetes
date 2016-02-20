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
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	podListTimeout     = time.Minute
	serverStartTimeout = podStartTimeout + 3*time.Minute
)

var _ = Describe("[Feature:Example]", func() {
	framework := NewFramework("examples")
	var c *client.Client
	var ns string
	BeforeEach(func() {
		c = framework.Client
		ns = framework.Namespace.Name
	})

	Describe("Redis", func() {
		It("should create and stop redis servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/redis", file)
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
			runKubectlOrDie("create", "-f", bootstrapYaml, nsFlag)
			err := waitForPodRunningInNamespace(c, bootstrapPodName, ns)
			Expect(err).NotTo(HaveOccurred())

			_, err = lookForStringInLog(ns, bootstrapPodName, "master", expectedOnServer, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, bootstrapPodName, "sentinel", expectedOnSentinel, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("setting up services and controllers")
			runKubectlOrDie("create", "-f", sentinelServiceYaml, nsFlag)
			runKubectlOrDie("create", "-f", sentinelControllerYaml, nsFlag)
			runKubectlOrDie("create", "-f", controllerYaml, nsFlag)

			By("scaling up the deployment")
			runKubectlOrDie("scale", "rc", redisRC, "--replicas=3", nsFlag)
			runKubectlOrDie("scale", "rc", sentinelRC, "--replicas=3", nsFlag)

			By("checking up the services")
			checkAllLogs := func() {
				forEachPod(c, ns, "name", "redis", func(pod api.Pod) {
					if pod.Name != bootstrapPodName {
						_, err := lookForStringInLog(ns, pod.Name, "redis", expectedOnServer, serverStartTimeout)
						Expect(err).NotTo(HaveOccurred())
					}
				})
				forEachPod(c, ns, "name", "redis-sentinel", func(pod api.Pod) {
					if pod.Name != bootstrapPodName {
						_, err := lookForStringInLog(ns, pod.Name, "sentinel", expectedOnSentinel, serverStartTimeout)
						Expect(err).NotTo(HaveOccurred())
					}
				})
			}
			checkAllLogs()

			By("turning down bootstrap")
			runKubectlOrDie("delete", "-f", bootstrapYaml, nsFlag)
			err = waitForRCPodToDisappear(c, ns, redisRC, bootstrapPodName)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for the new master election")
			checkAllLogs()
		})
	})

	Describe("Celery-RabbitMQ", func() {
		It("should create and stop celery+rabbitmq servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "celery-rabbitmq", file)
			}
			rabbitmqServiceYaml := mkpath("rabbitmq-service.yaml")
			rabbitmqControllerYaml := mkpath("rabbitmq-controller.yaml")
			celeryControllerYaml := mkpath("celery-controller.yaml")
			flowerControllerYaml := mkpath("flower-controller.yaml")
			flowerServiceYaml := mkpath("flower-service.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting rabbitmq")
			runKubectlOrDie("create", "-f", rabbitmqServiceYaml, nsFlag)
			runKubectlOrDie("create", "-f", rabbitmqControllerYaml, nsFlag)
			forEachPod(c, ns, "component", "rabbitmq", func(pod api.Pod) {
				_, err := lookForStringInLog(ns, pod.Name, "rabbitmq", "Server startup complete", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})
			err := waitForEndpoint(c, ns, "rabbitmq-service")
			Expect(err).NotTo(HaveOccurred())

			By("starting celery")
			runKubectlOrDie("create", "-f", celeryControllerYaml, nsFlag)
			forEachPod(c, ns, "component", "celery", func(pod api.Pod) {
				_, err := lookForStringInFile(ns, pod.Name, "celery", "/data/celery.log", " ready.", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			By("starting flower")
			runKubectlOrDie("create", "-f", flowerServiceYaml, nsFlag)
			runKubectlOrDie("create", "-f", flowerControllerYaml, nsFlag)
			forEachPod(c, ns, "component", "flower", func(pod api.Pod) {
				// Do nothing. just wait for it to be up and running.
			})
			content, err := makeHttpRequestToService(c, ns, "flower-service", "/", endpointRegisterTimeout)
			Expect(err).NotTo(HaveOccurred())
			if !strings.Contains(content, "<title>Celery Flower</title>") {
				Failf("Flower HTTP request failed")
			}
		})
	})

	Describe("Spark", func() {
		It("should start spark master, driver and workers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "spark", file)
			}

			// TODO: Add Zepplin and Web UI to this example.
			serviceYaml := mkpath("spark-master-service.yaml")
			masterYaml := mkpath("spark-master-controller.yaml")
			workerControllerYaml := mkpath("spark-worker-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			master := func() {
				By("starting master")
				runKubectlOrDie("create", "-f", serviceYaml, nsFlag)
				runKubectlOrDie("create", "-f", masterYaml, nsFlag)

				Logf("Now polling for Master startup...")

				// Only one master pod: But its a natural way to look up pod names.
				forEachPod(c, ns, "component", "spark-master", func(pod api.Pod) {
					Logf("Now waiting for master to startup in %v", pod.Name)
					_, err := lookForStringInLog(ns, pod.Name, "spark-master", "Starting Spark master at", serverStartTimeout)
					Expect(err).NotTo(HaveOccurred())
				})

				By("waiting for master endpoint")
				err := waitForEndpoint(c, ns, "spark-master")
				Expect(err).NotTo(HaveOccurred())
			}
			worker := func() {
				By("starting workers")
				Logf("Now starting Workers")
				runKubectlOrDie("create", "-f", workerControllerYaml, nsFlag)

				// For now, scaling is orthogonal to the core test.
				// ScaleRC(c, ns, "spark-worker-controller", 2, true)

				Logf("Now polling for worker startup...")
				forEachPod(c, ns, "component", "spark-worker", func(pod api.Pod) {
					_, err := lookForStringInLog(ns, pod.Name, "spark-worker", "Successfully registered with master", serverStartTimeout)
					Expect(err).NotTo(HaveOccurred())
				})
			}
			// Run the worker verification after we turn up the master.
			defer worker()
			master()
		})
	})

	Describe("Cassandra", func() {
		It("should create and scale cassandra", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "cassandra", file)
			}
			serviceYaml := mkpath("cassandra-service.yaml")
			podYaml := mkpath("cassandra.yaml")
			controllerYaml := mkpath("cassandra-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting service and pod")
			runKubectlOrDie("create", "-f", serviceYaml, nsFlag)
			runKubectlOrDie("create", "-f", podYaml, nsFlag)
			err := waitForPodRunningInNamespace(c, "cassandra", ns)
			Expect(err).NotTo(HaveOccurred())

			_, err = lookForStringInLog(ns, "cassandra", "cassandra", "Listening for thrift clients", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			err = waitForEndpoint(c, ns, "cassandra")
			Expect(err).NotTo(HaveOccurred())

			By("create and scale rc")
			runKubectlOrDie("create", "-f", controllerYaml, nsFlag)
			err = ScaleRC(c, ns, "cassandra", 2, true)
			Expect(err).NotTo(HaveOccurred())
			forEachPod(c, ns, "name", "cassandra", func(pod api.Pod) {
				_, err = lookForStringInLog(ns, pod.Name, "cassandra", "Listening for thrift clients", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
				_, err = lookForStringInLog(ns, pod.Name, "cassandra", "Handshaking version", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			output := runKubectlOrDie("exec", "cassandra", nsFlag, "--", "nodetool", "status")
			forEachPod(c, ns, "name", "cassandra", func(pod api.Pod) {
				if !strings.Contains(output, pod.Status.PodIP) {
					Failf("Pod ip %s not found in nodetool status", pod.Status.PodIP)
				}
			})
		})
	})

	Describe("Storm", func() {
		It("should create and stop Zookeeper, Nimbus and Storm worker servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "storm", file)
			}
			zookeeperServiceJson := mkpath("zookeeper-service.json")
			zookeeperPodJson := mkpath("zookeeper.json")
			nimbusServiceJson := mkpath("storm-nimbus-service.json")
			nimbusPodJson := mkpath("storm-nimbus.json")
			workerControllerJson := mkpath("storm-worker-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			zookeeperPod := "zookeeper"

			By("starting Zookeeper")
			runKubectlOrDie("create", "-f", zookeeperPodJson, nsFlag)
			runKubectlOrDie("create", "-f", zookeeperServiceJson, nsFlag)
			err := waitForPodRunningInNamespace(c, zookeeperPod, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if zookeeper is up and running")
			_, err = lookForStringInLog(ns, zookeeperPod, "zookeeper", "binding to port", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			err = waitForEndpoint(c, ns, "zookeeper")
			Expect(err).NotTo(HaveOccurred())

			By("starting Nimbus")
			runKubectlOrDie("create", "-f", nimbusPodJson, nsFlag)
			runKubectlOrDie("create", "-f", nimbusServiceJson, nsFlag)
			err = waitForPodRunningInNamespace(c, "nimbus", ns)
			Expect(err).NotTo(HaveOccurred())

			err = waitForEndpoint(c, ns, "nimbus")
			Expect(err).NotTo(HaveOccurred())

			By("starting workers")
			runKubectlOrDie("create", "-f", workerControllerJson, nsFlag)
			forEachPod(c, ns, "name", "storm-worker", func(pod api.Pod) {
				//do nothing, just wait for the pod to be running
			})
			// TODO: Add logging configuration to nimbus & workers images and then
			// look for a string instead of sleeping.
			time.Sleep(20 * time.Second)

			By("checking if there are established connections to Zookeeper")
			_, err = lookForStringInLog(ns, zookeeperPod, "zookeeper", "Established session", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("checking if Nimbus responds to requests")
			lookForString("No topologies running.", time.Minute, func() string {
				return runKubectlOrDie("exec", "nimbus", nsFlag, "--", "bin/storm", "list")
			})
		})
	})

	Describe("Liveness", func() {
		It("liveness pods should be automatically restarted", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "liveness", file)
			}
			execYaml := mkpath("exec-liveness.yaml")
			httpYaml := mkpath("http-liveness.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			runKubectlOrDie("create", "-f", execYaml, nsFlag)
			runKubectlOrDie("create", "-f", httpYaml, nsFlag)
			checkRestart := func(podName string, timeout time.Duration) {
				err := waitForPodRunningInNamespace(c, podName, ns)
				Expect(err).NotTo(HaveOccurred())

				for t := time.Now(); time.Since(t) < timeout; time.Sleep(poll) {
					pod, err := c.Pods(ns).Get(podName)
					expectNoError(err, fmt.Sprintf("getting pod %s", podName))
					restartCount := api.GetExistingContainerStatus(pod.Status.ContainerStatuses, "liveness").RestartCount
					Logf("Pod: %s   restart count:%d", podName, restartCount)
					if restartCount > 0 {
						return
					}
				}
				Failf("Pod %s was not restarted", podName)
			}
			By("Check restarts")
			checkRestart("liveness-exec", time.Minute)
			checkRestart("liveness-http", time.Minute)
		})
	})

	Describe("Secret", func() {
		It("should create a pod that reads a secret", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "secrets", file)
			}
			secretYaml := mkpath("secret.yaml")
			podYaml := mkpath("secret-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("creating secret and pod")
			runKubectlOrDie("create", "-f", secretYaml, nsFlag)
			runKubectlOrDie("create", "-f", podYaml, nsFlag)

			By("checking if secret was read correctly")
			_, err := lookForStringInLog(ns, "secret-test-pod", "test-container", "value-1", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("Downward API", func() {
		It("should create a pod that prints his name and namespace", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "downward-api", file)
			}
			podYaml := mkpath("dapi-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "dapi-test-pod"

			By("creating the pod")
			runKubectlOrDie("create", "-f", podYaml, nsFlag)

			By("checking if name and namespace were passed correctly")
			_, err := lookForStringInLog(ns, podName, "test-container", fmt.Sprintf("POD_NAMESPACE=%v", ns), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, podName, "test-container", fmt.Sprintf("POD_NAME=%v", podName), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("RethinkDB", func() {
		It("should create and stop rethinkdb servers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "rethinkdb", file)
			}
			driverServiceYaml := mkpath("driver-service.yaml")
			rethinkDbControllerYaml := mkpath("rc.yaml")
			adminPodYaml := mkpath("admin-pod.yaml")
			adminServiceYaml := mkpath("admin-service.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting rethinkdb")
			runKubectlOrDie("create", "-f", driverServiceYaml, nsFlag)
			runKubectlOrDie("create", "-f", rethinkDbControllerYaml, nsFlag)
			checkDbInstances := func() {
				forEachPod(c, ns, "db", "rethinkdb", func(pod api.Pod) {
					_, err := lookForStringInLog(ns, pod.Name, "rethinkdb", "Server ready", serverStartTimeout)
					Expect(err).NotTo(HaveOccurred())
				})
			}
			checkDbInstances()
			err := waitForEndpoint(c, ns, "rethinkdb-driver")
			Expect(err).NotTo(HaveOccurred())

			By("scaling rethinkdb")
			ScaleRC(c, ns, "rethinkdb-rc", 2, true)
			checkDbInstances()

			By("starting admin")
			runKubectlOrDie("create", "-f", adminServiceYaml, nsFlag)
			runKubectlOrDie("create", "-f", adminPodYaml, nsFlag)
			err = waitForPodRunningInNamespace(c, "rethinkdb-admin", ns)
			Expect(err).NotTo(HaveOccurred())
			checkDbInstances()
			content, err := makeHttpRequestToService(c, ns, "rethinkdb-admin", "/", endpointRegisterTimeout)
			Expect(err).NotTo(HaveOccurred())
			if !strings.Contains(content, "<title>RethinkDB Administration Console</title>") {
				Failf("RethinkDB console is not running")
			}
		})
	})

	Describe("Hazelcast", func() {
		It("should create and scale hazelcast", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "hazelcast", file)
			}
			serviceYaml := mkpath("hazelcast-service.yaml")
			controllerYaml := mkpath("hazelcast-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting hazelcast")
			runKubectlOrDie("create", "-f", serviceYaml, nsFlag)
			runKubectlOrDie("create", "-f", controllerYaml, nsFlag)
			forEachPod(c, ns, "name", "hazelcast", func(pod api.Pod) {
				_, err := lookForStringInLog(ns, pod.Name, "hazelcast", "Members [1]", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
				_, err = lookForStringInLog(ns, pod.Name, "hazelcast", "is STARTED", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			err := waitForEndpoint(c, ns, "hazelcast")
			Expect(err).NotTo(HaveOccurred())

			By("scaling hazelcast")
			ScaleRC(c, ns, "hazelcast", 2, true)
			forEachPod(c, ns, "name", "hazelcast", func(pod api.Pod) {
				_, err := lookForStringInLog(ns, pod.Name, "hazelcast", "Members [2]", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})

func makeHttpRequestToService(c *client.Client, ns, service, path string, timeout time.Duration) (string, error) {
	var result []byte
	var err error
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(poll) {
		proxyRequest, errProxy := getServicesProxyRequest(c, c.Get())
		if errProxy != nil {
			break
		}
		result, err = proxyRequest.Namespace(ns).
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

// pass enough context with the 'old' parameter so that it replaces what your really intended.
func prepareResourceWithReplacedString(inputFile, old, new string) string {
	f, err := os.Open(inputFile)
	Expect(err).NotTo(HaveOccurred())
	defer f.Close()
	data, err := ioutil.ReadAll(f)
	Expect(err).NotTo(HaveOccurred())
	podYaml := strings.Replace(string(data), old, new, 1)
	return podYaml
}

func forEachPod(c *client.Client, ns, selectorKey, selectorValue string, fn func(api.Pod)) {
	pods := []*api.Pod{}
	for t := time.Now(); time.Since(t) < podListTimeout; time.Sleep(poll) {
		selector := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
		options := api.ListOptions{LabelSelector: selector}
		podList, err := c.Pods(ns).List(options)
		Expect(err).NotTo(HaveOccurred())
		for _, pod := range podList.Items {
			if pod.Status.Phase == api.PodPending || pod.Status.Phase == api.PodRunning {
				pods = append(pods, &pod)
			}
		}
		if len(pods) > 0 {
			break
		}
	}
	if pods == nil || len(pods) == 0 {
		Failf("No pods found")
	}
	for _, pod := range pods {
		err := waitForPodRunningInNamespace(c, pod.Name, ns)
		Expect(err).NotTo(HaveOccurred())
		fn(*pod)
	}
}
