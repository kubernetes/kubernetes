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
	"path/filepath"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	podListTimeout     = time.Minute
	serverStartTimeout = podStartTimeout + 3*time.Minute
)

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
		if err := c.Namespaces().Delete(ns); err != nil {
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
			content, err := makeHttpRequestToService(c, ns, "flower-service", "/")
			Expect(err).NotTo(HaveOccurred())
			if !strings.Contains(content, "<title>Celery Flower</title>") {
				Failf("Flower HTTP request failed")
			}
		})
	})
})

func makeHttpRequestToService(c *client.Client, ns, service, path string) (string, error) {
	result, err := c.Get().
		Prefix("proxy").
		Namespace(ns).
		Resource("services").
		Name(service).
		Suffix(path).
		Do().
		Raw()
	return string(result), err
}

func forEachPod(c *client.Client, ns, selectorKey, selectorValue string, fn func(api.Pod)) {
	var pods *api.PodList
	var err error
	for t := time.Now(); time.Since(t) < podListTimeout; time.Sleep(poll) {
		pods, err = c.Pods(ns).List(labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue})), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		if len(pods.Items) > 0 {
			break
		}
	}
	if pods == nil || len(pods.Items) == 0 {
		Failf("No pods found")
	}
	for _, pod := range pods.Items {
		err = waitForPodRunningInNamespace(c, pod.Name, ns)
		Expect(err).NotTo(HaveOccurred())
		fn(pod)
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
	err = fmt.Errorf("Failed to find \"%s\"", expectedString)
	return
}
