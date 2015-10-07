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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	podListTimeout     = time.Minute
	serverStartTimeout = podStartTimeout + 3*time.Minute
	dnsReadyTimeout    = time.Minute
)

const queryDnsPythonTemplate string = `
import socket
try:
	socket.gethostbyname('%s')
	print 'ok'
except:
	print 'err'`

var _ = Describe("Examples e2e", func() {
	var c *client.Client
	var ns string
	var testingNs *api.Namespace
	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		testingNs, err = createTestingNS("examples", c)
		ns = testingNs.Name
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := deleteNS(c, ns, 5*time.Minute /* namespace deletion timeout */); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	Describe("[Skipped][Example]Redis", func() {
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
			runKubectl("create", "-f", bootstrapYaml, nsFlag)
			err := waitForPodRunningInNamespace(c, bootstrapPodName, ns)
			Expect(err).NotTo(HaveOccurred())

			_, err = lookForStringInLog(ns, bootstrapPodName, "master", expectedOnServer, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, bootstrapPodName, "sentinel", expectedOnSentinel, serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("setting up services and controllers")
			runKubectl("create", "-f", sentinelServiceYaml, nsFlag)
			runKubectl("create", "-f", sentinelControllerYaml, nsFlag)
			runKubectl("create", "-f", controllerYaml, nsFlag)

			By("scaling up the deployment")
			runKubectl("scale", "rc", redisRC, "--replicas=3", nsFlag)
			runKubectl("scale", "rc", sentinelRC, "--replicas=3", nsFlag)

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
			runKubectl("delete", "-f", bootstrapYaml, nsFlag)
			err = waitForRCPodToDisappear(c, ns, redisRC, bootstrapPodName)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for the new master election")
			checkAllLogs()
		})
	})

	Describe("[Skipped][Example]Celery-RabbitMQ", func() {
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
			runKubectl("create", "-f", rabbitmqServiceYaml, nsFlag)
			runKubectl("create", "-f", rabbitmqControllerYaml, nsFlag)
			forEachPod(c, ns, "component", "rabbitmq", func(pod api.Pod) {
				_, err := lookForStringInLog(ns, pod.Name, "rabbitmq", "Server startup complete", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})
			err := waitForEndpoint(c, ns, "rabbitmq-service")
			Expect(err).NotTo(HaveOccurred())

			By("starting celery")
			runKubectl("create", "-f", celeryControllerYaml, nsFlag)
			forEachPod(c, ns, "component", "celery", func(pod api.Pod) {
				_, err := lookForStringInFile(ns, pod.Name, "celery", "/data/celery.log", " ready.", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			By("starting flower")
			runKubectl("create", "-f", flowerServiceYaml, nsFlag)
			runKubectl("create", "-f", flowerControllerYaml, nsFlag)
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

	Describe("[Skipped][Example]Spark", func() {
		It("should start spark master, driver and workers", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "spark", file)
			}
			serviceJson := mkpath("spark-master-service.json")
			masterJson := mkpath("spark-master.json")
			driverJson := mkpath("spark-driver.json")
			workerControllerJson := mkpath("spark-worker-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting master")
			runKubectl("create", "-f", serviceJson, nsFlag)
			runKubectl("create", "-f", masterJson, nsFlag)
			runKubectl("create", "-f", driverJson, nsFlag)
			err := waitForPodRunningInNamespace(c, "spark-master", ns)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, "spark-master", "spark-master", "Starting Spark master at", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, "spark-driver", "spark-driver", "Use kubectl exec", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			By("waiting for master endpoint")
			err = waitForEndpoint(c, ns, "spark-master")
			Expect(err).NotTo(HaveOccurred())

			By("starting workers")
			runKubectl("create", "-f", workerControllerJson, nsFlag)
			ScaleRC(c, ns, "spark-worker-controller", 2, true)
			forEachPod(c, ns, "name", "spark-worker", func(pod api.Pod) {
				_, err := lookForStringInLog(ns, pod.Name, "spark-worker", "Successfully registered with master", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})

	Describe("[Skipped][Example]Cassandra", func() {
		It("should create and scale cassandra", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "cassandra", file)
			}
			serviceYaml := mkpath("cassandra-service.yaml")
			podYaml := mkpath("cassandra.yaml")
			controllerYaml := mkpath("cassandra-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting service and pod")
			runKubectl("create", "-f", serviceYaml, nsFlag)
			runKubectl("create", "-f", podYaml, nsFlag)
			err := waitForPodRunningInNamespace(c, "cassandra", ns)
			Expect(err).NotTo(HaveOccurred())

			_, err = lookForStringInLog(ns, "cassandra", "cassandra", "Listening for thrift clients", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())

			err = waitForEndpoint(c, ns, "cassandra")
			Expect(err).NotTo(HaveOccurred())

			By("create and scale rc")
			runKubectl("create", "-f", controllerYaml, nsFlag)
			err = ScaleRC(c, ns, "cassandra", 2, true)
			Expect(err).NotTo(HaveOccurred())
			forEachPod(c, ns, "name", "cassandra", func(pod api.Pod) {
				_, err = lookForStringInLog(ns, pod.Name, "cassandra", "Listening for thrift clients", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
				_, err = lookForStringInLog(ns, pod.Name, "cassandra", "Handshaking version", serverStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			})

			output := runKubectl("exec", "cassandra", nsFlag, "--", "nodetool", "status")
			forEachPod(c, ns, "name", "cassandra", func(pod api.Pod) {
				if !strings.Contains(output, pod.Status.PodIP) {
					Failf("Pod ip %s not found in nodetool status", pod.Status.PodIP)
				}
			})
		})
	})

	Describe("[Skipped][Example]Storm", func() {
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
			runKubectl("create", "-f", zookeeperPodJson, nsFlag)
			runKubectl("create", "-f", zookeeperServiceJson, nsFlag)
			err := waitForPodRunningInNamespace(c, zookeeperPod, ns)
			Expect(err).NotTo(HaveOccurred())

			By("checking if zookeeper is up and running")
			_, err = lookForStringInLog(ns, zookeeperPod, "zookeeper", "binding to port", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			err = waitForEndpoint(c, ns, "zookeeper")
			Expect(err).NotTo(HaveOccurred())

			By("starting Nimbus")
			runKubectl("create", "-f", nimbusPodJson, nsFlag)
			runKubectl("create", "-f", nimbusServiceJson, nsFlag)
			err = waitForPodRunningInNamespace(c, "nimbus", ns)
			Expect(err).NotTo(HaveOccurred())

			err = waitForEndpoint(c, ns, "nimbus")
			Expect(err).NotTo(HaveOccurred())

			By("starting workers")
			runKubectl("create", "-f", workerControllerJson, nsFlag)
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
				return runKubectl("exec", "nimbus", nsFlag, "--", "bin/storm", "list")
			})
		})
	})

	Describe("[Skipped][Example]Liveness", func() {
		It("liveness pods should be automatically restarted", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "liveness", file)
			}
			execYaml := mkpath("exec-liveness.yaml")
			httpYaml := mkpath("http-liveness.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			runKubectl("create", "-f", execYaml, nsFlag)
			runKubectl("create", "-f", httpYaml, nsFlag)
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

	Describe("[Skipped][Example]Secret", func() {
		It("should create a pod that reads a secret", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "secrets", file)
			}
			secretYaml := mkpath("secret.yaml")
			podYaml := mkpath("secret-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("creating secret and pod")
			runKubectl("create", "-f", secretYaml, nsFlag)
			runKubectl("create", "-f", podYaml, nsFlag)

			By("checking if secret was read correctly")
			_, err := lookForStringInLog(ns, "secret-test-pod", "test-container", "value-1", serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("[Skipped][Example]Downward API", func() {
		It("should create a pod that prints his name and namespace", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "docs", "user-guide", "downward-api", file)
			}
			podYaml := mkpath("dapi-pod.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "dapi-test-pod"

			By("creating the pod")
			runKubectl("create", "-f", podYaml, nsFlag)

			By("checking if name and namespace were passed correctly")
			_, err := lookForStringInLog(ns, podName, "test-container", fmt.Sprintf("POD_NAMESPACE=%v", ns), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
			_, err = lookForStringInLog(ns, podName, "test-container", fmt.Sprintf("POD_NAME=%v", podName), serverStartTimeout)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	Describe("[Skipped][Example]RethinkDB", func() {
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
			runKubectl("create", "-f", driverServiceYaml, nsFlag)
			runKubectl("create", "-f", rethinkDbControllerYaml, nsFlag)
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
			runKubectl("create", "-f", adminServiceYaml, nsFlag)
			runKubectl("create", "-f", adminPodYaml, nsFlag)
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

	Describe("[Skipped][Example]Hazelcast", func() {
		It("should create and scale hazelcast", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples", "hazelcast", file)
			}
			serviceYaml := mkpath("hazelcast-service.yaml")
			controllerYaml := mkpath("hazelcast-controller.yaml")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("starting hazelcast")
			runKubectl("create", "-f", serviceYaml, nsFlag)
			runKubectl("create", "-f", controllerYaml, nsFlag)
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

	Describe("[Example]ClusterDns", func() {
		It("should create pod that uses dns", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/cluster-dns", file)
			}

			// contrary to the example, this test does not use contexts, for simplicity
			// namespaces are passed directly.
			// Also, for simplicity, we don't use yamls with namespaces, but we
			// create testing namespaces instead.

			backendRcYaml := mkpath("dns-backend-rc.yaml")
			backendRcName := "dns-backend"
			backendSvcYaml := mkpath("dns-backend-service.yaml")
			backendSvcName := "dns-backend"
			backendPodName := "dns-backend"
			frontendPodYaml := mkpath("dns-frontend-pod.yaml")
			frontendPodName := "dns-frontend"
			frontendPodContainerName := "dns-frontend"

			podOutput := "Hello World!"

			// we need two namespaces anyway, so let's forget about
			// the one created in BeforeEach and create two new ones.
			namespaces := []*api.Namespace{nil, nil}
			for i := range namespaces {
				var err error
				namespaces[i], err = createTestingNS(fmt.Sprintf("dnsexample%d", i), c)
				if namespaces[i] != nil {
					defer deleteNS(c, namespaces[i].Name, 5*time.Minute /* namespace deletion timeout */)
				}
				Expect(err).NotTo(HaveOccurred())
			}

			for _, ns := range namespaces {
				runKubectl("create", "-f", backendRcYaml, getNsCmdFlag(ns))
			}

			for _, ns := range namespaces {
				runKubectl("create", "-f", backendSvcYaml, getNsCmdFlag(ns))
			}

			// wait for objects
			for _, ns := range namespaces {
				waitForRCPodsRunning(c, ns.Name, backendRcName)
				waitForService(c, ns.Name, backendSvcName, true, poll, serviceStartTimeout)
			}
			// it is not enough that pods are running because they may be set to running, but
			// the application itself may have not been initialized. Just query the application.
			for _, ns := range namespaces {
				label := labels.SelectorFromSet(labels.Set(map[string]string{"name": backendRcName}))
				pods, err := c.Pods(ns.Name).List(label, fields.Everything())
				Expect(err).NotTo(HaveOccurred())
				err = podsResponding(c, ns.Name, backendPodName, false, pods)
				Expect(err).NotTo(HaveOccurred(), "waiting for all pods to respond")
				Logf("found %d backend pods responding in namespace %s", len(pods.Items), ns.Name)

				err = serviceResponding(c, ns.Name, backendSvcName)
				Expect(err).NotTo(HaveOccurred(), "waiting for the service to respond")
			}

			// Now another tricky part:
			// It may happen that the service name is not yet in DNS.
			// So if we start our pod, it will fail. We must make sure
			// the name is already resolvable. So let's try to query DNS from
			// the pod we have, until we find our service name.
			// This complicated code may be removed if the pod itself retried after
			// dns error or timeout.
			// This code is probably unnecessary, but let's stay on the safe side.
			label := labels.SelectorFromSet(labels.Set(map[string]string{"name": backendPodName}))
			pods, err := c.Pods(namespaces[0].Name).List(label, fields.Everything())

			if err != nil || pods == nil || len(pods.Items) == 0 {
				Failf("no running pods found")
			}
			podName := pods.Items[0].Name

			queryDns := fmt.Sprintf(queryDnsPythonTemplate, backendSvcName+"."+namespaces[0].Name)
			_, err = lookForStringInPodExec(namespaces[0].Name, podName, []string{"python", "-c", queryDns}, "ok", dnsReadyTimeout)
			Expect(err).NotTo(HaveOccurred(), "waiting for output from pod exec")

			updatedPodYaml := prepareResourceWithReplacedString(frontendPodYaml, "dns-backend.development.cluster.local", fmt.Sprintf("dns-backend.%s.cluster.local", namespaces[0].Name))

			// create a pod in each namespace
			for _, ns := range namespaces {
				newKubectlCommand("create", "-f", "-", getNsCmdFlag(ns)).withStdinData(updatedPodYaml).exec()
			}
			// remember that we cannot wait for the pods to be running because our pods terminate by themselves.

			// wait for pods to print their result
			for _, ns := range namespaces {
				_, err := lookForStringInLog(ns.Name, frontendPodName, frontendPodContainerName, podOutput, podStartTimeout)
				Expect(err).NotTo(HaveOccurred())
			}
		})
	})
})

func makeHttpRequestToService(c *client.Client, ns, service, path string, timeout time.Duration) (string, error) {
	var result []byte
	var err error
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(poll) {
		result, err = c.Get().
			Prefix("proxy").
			Namespace(ns).
			Resource("services").
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

func getNsCmdFlag(ns *api.Namespace) string {
	return fmt.Sprintf("--namespace=%v", ns.Name)
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
		podList, err := c.Pods(ns).List(labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue})), fields.Everything())
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

func lookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return runKubectl("log", podName, container, fmt.Sprintf("--namespace=%v", ns))
	})
}

func lookForStringInFile(ns, podName, container, file, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return runKubectl("exec", podName, "-c", container, fmt.Sprintf("--namespace=%v", ns), "--", "cat", file)
	})
}

func lookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		// use the first container
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns), "--"}
		args = append(args, command...)
		return runKubectl(args...)
	})
}

// Looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
func lookForString(expectedString string, timeout time.Duration, fn func() string) (result string, err error) {
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(poll) {
		result = fn()
		if strings.Contains(result, expectedString) {
			return
		}
	}
	err = fmt.Errorf("Failed to find \"%s\", last result: \"%s\"", expectedString, result)
	return
}
