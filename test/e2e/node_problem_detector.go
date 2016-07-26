/*
Copyright 2016 The Kubernetes Authors.

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
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("NodeProblemDetector", func() {
	const (
		pollInterval   = 1 * time.Second
		pollConsistent = 5 * time.Second
		pollTimeout    = 1 * time.Minute
		image          = "gcr.io/google_containers/node-problem-detector:v0.1"
	)
	f := framework.NewDefaultFramework("node-problem-detector")
	var c *client.Client
	var uid string
	var ns, name, configName, eventNamespace string
	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
		uid = string(uuid.NewUUID())
		name = "node-problem-detector-" + uid
		configName = "node-problem-detector-config-" + uid
		// There is no namespace for Node, event recorder will set default namespace for node events.
		eventNamespace = api.NamespaceDefault
	})

	// Test kernel monitor. We may add other tests if we have more problem daemons in the future.
	framework.KubeDescribe("KernelMonitor", func() {
		const (
			// Use test condition to avoid conflict with real node problem detector
			// TODO(random-liu): Now node condition could be arbitrary string, consider wether we need to
			// add TestCondition when switching to predefined condition list.
			condition      = api.NodeConditionType("TestCondition")
			defaultReason  = "Default"
			defaultMessage = "default message"
			logDir         = "/log"
			logFile        = "test.log"
			configDir      = "/config"
			configFile     = "testconfig.json"
			tempReason     = "Temporary"
			tempMessage    = "temporary error"
			permReason     = "Permanent"
			permMessage    = "permanent error"
			configVolume   = "config"
			logVolume      = "log"
		)
		var source, config, tmpDir string
		var node *api.Node
		var eventListOptions api.ListOptions
		injectCommand := func(err string, num int) string {
			var commands []string
			for i := 0; i < num; i++ {
				commands = append(commands, fmt.Sprintf("echo kernel: [%d.000000] %s >> %s/%s", i, err, tmpDir, logFile))
			}
			return strings.Join(commands, ";")
		}

		BeforeEach(func() {
			framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
			// Randomize the source name to avoid conflict with real node problem detector
			source = "kernel-monitor-" + uid
			config = `
			{
				"logPath": "` + logDir + "/" + logFile + `",
				"bufferSize": 10,
				"source": "` + source + `",
				"conditions": [
				{
					"type": "` + string(condition) + `",
					"reason": "` + defaultReason + `",
					"message": "` + defaultMessage + `"
				}
				],
				"rules": [
				{
					"type": "temporary",
					"reason": "` + tempReason + `",
					"pattern": "` + tempMessage + `"
				},
				{
					"type": "permanent",
					"condition": "` + string(condition) + `",
					"reason": "` + permReason + `",
					"pattern": "` + permMessage + `"
				}
				]
			}`
			By("Get a non master node to run the pod")
			nodes, err := c.Nodes().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			node = nil
			for _, n := range nodes.Items {
				if !system.IsMasterNode(&n) {
					node = &n
					break
				}
			}
			Expect(node).NotTo(BeNil())
			By("Generate event list options")
			selector := fields.Set{
				"involvedObject.kind":      "Node",
				"involvedObject.name":      node.Name,
				"involvedObject.namespace": api.NamespaceAll,
				"source":                   source,
			}.AsSelector()
			eventListOptions = api.ListOptions{FieldSelector: selector}
			By("Create the test log file")
			tmpDir = "/tmp/" + name
			cmd := fmt.Sprintf("mkdir %s; > %s/%s", tmpDir, tmpDir, logFile)
			Expect(framework.IssueSSHCommand(cmd, framework.TestContext.Provider, node)).To(Succeed())
			By("Create config map for the node problem detector")
			_, err = c.ConfigMaps(ns).Create(&api.ConfigMap{
				ObjectMeta: api.ObjectMeta{
					Name: configName,
				},
				Data: map[string]string{configFile: config},
			})
			Expect(err).NotTo(HaveOccurred())
			By("Create the node problem detector")
			_, err = c.Pods(ns).Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: name,
				},
				Spec: api.PodSpec{
					NodeName:        node.Name,
					SecurityContext: &api.PodSecurityContext{HostNetwork: true},
					Volumes: []api.Volume{
						{
							Name: configVolume,
							VolumeSource: api.VolumeSource{
								ConfigMap: &api.ConfigMapVolumeSource{
									LocalObjectReference: api.LocalObjectReference{Name: configName},
								},
							},
						},
						{
							Name: logVolume,
							VolumeSource: api.VolumeSource{
								HostPath: &api.HostPathVolumeSource{Path: tmpDir},
							},
						},
					},
					Containers: []api.Container{
						{
							Name:    name,
							Image:   image,
							Command: []string{"/node-problem-detector", "--kernel-monitor=" + configDir + "/" + configFile},
							VolumeMounts: []api.VolumeMount{
								{
									Name:      logVolume,
									MountPath: logDir,
								},
								{
									Name:      configVolume,
									MountPath: configDir,
								},
							},
						},
					},
				},
			})
			Expect(err).NotTo(HaveOccurred())
			By("Wait for node problem detector running")
			Expect(f.WaitForPodRunning(name)).To(Succeed())
		})

		It("should generate node condition and events for corresponding errors", func() {
			By("Make sure no events are generated")
			Consistently(func() error {
				return verifyNoEvents(c.Events(eventNamespace), eventListOptions)
			}, pollConsistent, pollInterval).Should(Succeed())
			By("Make sure the default node condition is generated")
			Eventually(func() error {
				return verifyCondition(c.Nodes(), node.Name, condition, api.ConditionFalse, defaultReason, defaultMessage)
			}, pollConsistent, pollInterval).Should(Succeed())

			num := 3
			By(fmt.Sprintf("Inject %d temporary errors", num))
			Expect(framework.IssueSSHCommand(injectCommand(tempMessage, num), framework.TestContext.Provider, node)).To(Succeed())
			By(fmt.Sprintf("Wait for %d events generated", num))
			Eventually(func() error {
				return verifyEvents(c.Events(eventNamespace), eventListOptions, num, tempReason, tempMessage)
			}, pollTimeout, pollInterval).Should(Succeed())
			By(fmt.Sprintf("Make sure only %d events generated", num))
			Consistently(func() error {
				return verifyEvents(c.Events(eventNamespace), eventListOptions, num, tempReason, tempMessage)
			}, pollConsistent, pollInterval).Should(Succeed())
			By("Make sure the node condition is still false")
			Expect(verifyCondition(c.Nodes(), node.Name, condition, api.ConditionFalse, defaultReason, defaultMessage)).To(Succeed())

			By("Inject 1 permanent error")
			Expect(framework.IssueSSHCommand(injectCommand(permMessage, 1), framework.TestContext.Provider, node)).To(Succeed())
			By("Make sure the corresponding node condition is generated")
			Eventually(func() error {
				return verifyCondition(c.Nodes(), node.Name, condition, api.ConditionTrue, permReason, permMessage)
			}, pollTimeout, pollInterval).Should(Succeed())
			By("Make sure no new events are generated")
			Consistently(func() error {
				return verifyEvents(c.Events(eventNamespace), eventListOptions, num, tempReason, tempMessage)
			}, pollConsistent, pollInterval).Should(Succeed())
		})

		AfterEach(func() {
			By("Delete the node problem detector")
			c.Pods(ns).Delete(name, api.NewDeleteOptions(0))
			By("Wait for the node problem detector to disappear")
			Expect(framework.WaitForPodToDisappear(c, ns, name, labels.Everything(), pollInterval, pollTimeout)).To(Succeed())
			By("Delete the config map")
			c.ConfigMaps(ns).Delete(configName)
			By("Clean up the events")
			Expect(c.Events(eventNamespace).DeleteCollection(api.NewDeleteOptions(0), eventListOptions)).To(Succeed())
			By("Clean up the node condition")
			patch := []byte(fmt.Sprintf(`{"status":{"conditions":[{"$patch":"delete","type":"%s"}]}}`, condition))
			c.Patch(api.StrategicMergePatchType).Resource("nodes").Name(node.Name).SubResource("status").Body(patch).Do()
			By("Clean up the temporary directory")
			framework.IssueSSHCommand(fmt.Sprintf("rm -r %s", tmpDir), framework.TestContext.Provider, node)
		})
	})
})

// verifyEvents verifies there are num specific events generated
func verifyEvents(e client.EventInterface, options api.ListOptions, num int, reason, message string) error {
	events, err := e.List(options)
	if err != nil {
		return err
	}
	count := 0
	for _, event := range events.Items {
		if event.Reason != reason || event.Message != message {
			return fmt.Errorf("unexpected event: %v", event)
		}
		count += int(event.Count)
	}
	if count != num {
		return fmt.Errorf("expect event number %d, got %d: %v", num, count, events.Items)
	}
	return nil
}

// verifyNoEvents verifies there is no event generated
func verifyNoEvents(e client.EventInterface, options api.ListOptions) error {
	events, err := e.List(options)
	if err != nil {
		return err
	}
	if len(events.Items) != 0 {
		return fmt.Errorf("unexpected events: %v", events.Items)
	}
	return nil
}

// verifyCondition verifies specific node condition is generated, if reason and message are empty, they will not be checked
func verifyCondition(n client.NodeInterface, nodeName string, condition api.NodeConditionType, status api.ConditionStatus, reason, message string) error {
	node, err := n.Get(nodeName)
	if err != nil {
		return err
	}
	_, c := api.GetNodeCondition(&node.Status, condition)
	if c == nil {
		return fmt.Errorf("node condition %q not found", condition)
	}
	if c.Status != status || c.Reason != reason || c.Message != message {
		return fmt.Errorf("unexpected node condition %q: %+v", condition, c)
	}
	return nil
}
