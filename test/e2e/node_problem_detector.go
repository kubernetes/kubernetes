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
	"path/filepath"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	coreclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5/typed/core/v1"
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
		image          = "gcr.io/google_containers/node-problem-detector:v0.2"
	)
	f := framework.NewDefaultFramework("node-problem-detector")
	var c clientset.Interface
	var uid string
	var ns, name, configName, eventNamespace string
	var nodeTime time.Time
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		uid = string(uuid.NewUUID())
		name = "node-problem-detector-" + uid
		configName = "node-problem-detector-config-" + uid
		// There is no namespace for Node, event recorder will set default namespace for node events.
		eventNamespace = v1.NamespaceDefault
	})

	// Test kernel monitor. We may add other tests if we have more problem daemons in the future.
	framework.KubeDescribe("KernelMonitor", func() {
		const (
			// Use test condition to avoid conflict with real node problem detector
			// TODO(random-liu): Now node condition could be arbitrary string, consider wether we need to
			// add TestCondition when switching to predefined condition list.
			condition    = v1.NodeConditionType("TestCondition")
			lookback     = time.Hour // Assume the test won't take more than 1 hour, in fact it usually only takes 90 seconds.
			startPattern = "test reboot"

			// File paths used in the test.
			logDir       = "/log"
			logFile      = "test.log"
			configDir    = "/config"
			configFile   = "testconfig.json"
			etcLocaltime = "/etc/localtime"

			// Volumes used in the test.
			configVolume    = "config"
			logVolume       = "log"
			localtimeVolume = "localtime"

			// Reasons and messages used in the test.
			defaultReason  = "Default"
			defaultMessage = "default message"
			tempReason     = "Temporary"
			tempMessage    = "temporary error"
			permReason     = "Permanent"
			permMessage    = "permanent error"
		)
		var source, config, tmpDir string
		var node *v1.Node
		var eventListOptions v1.ListOptions
		injectCommand := func(timestamp time.Time, log string, num int) string {
			var commands []string
			for i := 0; i < num; i++ {
				commands = append(commands, fmt.Sprintf("echo \"%s kernel: [0.000000] %s\" >> %s/%s",
					timestamp.Format(time.Stamp), log, tmpDir, logFile))
			}
			return strings.Join(commands, ";")
		}

		BeforeEach(func() {
			framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
			// Randomize the source name to avoid conflict with real node problem detector
			source = "kernel-monitor-" + uid
			config = `
			{
				"logPath": "` + filepath.Join(logDir, logFile) + `",
				"lookback": "` + lookback.String() + `",
				"startPattern": "` + startPattern + `",
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
			nodes, err := c.Core().Nodes().List(v1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			node = nil
			for _, n := range nodes.Items {
				if !system.IsMasterNode(n.Name) {
					node = &n
					break
				}
			}
			Expect(node).NotTo(BeNil())
			By("Generate event list options")
			selector := fields.Set{
				"involvedObject.kind":      "Node",
				"involvedObject.name":      node.Name,
				"involvedObject.namespace": v1.NamespaceAll,
				"source":                   source,
			}.AsSelector().String()
			eventListOptions = v1.ListOptions{FieldSelector: selector}
			By("Create the test log file")
			tmpDir = "/tmp/" + name
			cmd := fmt.Sprintf("mkdir %s; > %s/%s", tmpDir, tmpDir, logFile)
			Expect(framework.IssueSSHCommand(cmd, framework.TestContext.Provider, node)).To(Succeed())
			By("Create config map for the node problem detector")
			_, err = c.Core().ConfigMaps(ns).Create(&v1.ConfigMap{
				ObjectMeta: v1.ObjectMeta{
					Name: configName,
				},
				Data: map[string]string{configFile: config},
			})
			Expect(err).NotTo(HaveOccurred())
			By("Create the node problem detector")
			_, err = c.Core().Pods(ns).Create(&v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					NodeName:        node.Name,
					HostNetwork:     true,
					SecurityContext: &v1.PodSecurityContext{},
					Volumes: []v1.Volume{
						{
							Name: configVolume,
							VolumeSource: v1.VolumeSource{
								ConfigMap: &v1.ConfigMapVolumeSource{
									LocalObjectReference: v1.LocalObjectReference{Name: configName},
								},
							},
						},
						{
							Name: logVolume,
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: tmpDir},
							},
						},
						{
							Name: localtimeVolume,
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{Path: etcLocaltime},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:            name,
							Image:           image,
							Command:         []string{"/node-problem-detector", "--kernel-monitor=" + filepath.Join(configDir, configFile)},
							ImagePullPolicy: v1.PullAlways,
							Env: []v1.EnvVar{
								{
									Name: "NODE_NAME",
									ValueFrom: &v1.EnvVarSource{
										FieldRef: &v1.ObjectFieldSelector{
											APIVersion: "v1",
											FieldPath:  "spec.nodeName",
										},
									},
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      logVolume,
									MountPath: logDir,
								},
								{
									Name:      localtimeVolume,
									MountPath: etcLocaltime,
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
			// Get the node time
			nodeIP := framework.GetNodeExternalIP(node)
			result, err := framework.SSH("date '+%FT%T.%N%:z'", nodeIP, framework.TestContext.Provider)
			Expect(err).ShouldNot(HaveOccurred())
			Expect(result.Code).Should(BeZero())
			nodeTime, err = time.Parse(time.RFC3339, strings.TrimSpace(result.Stdout))
			Expect(err).ShouldNot(HaveOccurred())
		})

		It("should generate node condition and events for corresponding errors", func() {
			for _, test := range []struct {
				description      string
				timestamp        time.Time
				message          string
				messageNum       int
				events           int
				conditionReason  string
				conditionMessage string
				conditionType    v1.ConditionStatus
			}{
				{
					description:      "should generate default node condition",
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should not generate events for too old log",
					timestamp:        nodeTime.Add(-3 * lookback), // Assume 3*lookback is old enough
					message:          tempMessage,
					messageNum:       3,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should not change node condition for too old log",
					timestamp:        nodeTime.Add(-3 * lookback), // Assume 3*lookback is old enough
					message:          permMessage,
					messageNum:       1,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should generate event for old log within lookback duration",
					timestamp:        nodeTime.Add(-1 * time.Minute),
					message:          tempMessage,
					messageNum:       3,
					events:           3,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should change node condition for old log within lookback duration",
					timestamp:        nodeTime.Add(-1 * time.Minute),
					message:          permMessage,
					messageNum:       1,
					events:           3, // event number should not change
					conditionReason:  permReason,
					conditionMessage: permMessage,
					conditionType:    v1.ConditionTrue,
				},
				{
					description:      "should reset node condition if the node is reboot",
					timestamp:        nodeTime,
					message:          startPattern,
					messageNum:       1,
					events:           3, // event number should not change
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should generate event for new log",
					timestamp:        nodeTime.Add(5 * time.Minute),
					message:          tempMessage,
					messageNum:       3,
					events:           6,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should change node condition for new log",
					timestamp:        nodeTime.Add(5 * time.Minute),
					message:          permMessage,
					messageNum:       1,
					events:           6, // event number should not change
					conditionReason:  permReason,
					conditionMessage: permMessage,
					conditionType:    v1.ConditionTrue,
				},
			} {
				By(test.description)
				if test.messageNum > 0 {
					By(fmt.Sprintf("Inject %d logs: %q", test.messageNum, test.message))
					cmd := injectCommand(test.timestamp, test.message, test.messageNum)
					Expect(framework.IssueSSHCommand(cmd, framework.TestContext.Provider, node)).To(Succeed())
				}

				By(fmt.Sprintf("Wait for %d events generated", test.events))
				Eventually(func() error {
					return verifyEvents(c.Core().Events(eventNamespace), eventListOptions, test.events, tempReason, tempMessage)
				}, pollTimeout, pollInterval).Should(Succeed())
				By(fmt.Sprintf("Make sure only %d events generated", test.events))
				Consistently(func() error {
					return verifyEvents(c.Core().Events(eventNamespace), eventListOptions, test.events, tempReason, tempMessage)
				}, pollConsistent, pollInterval).Should(Succeed())

				By(fmt.Sprintf("Make sure node condition %q is set", condition))
				Eventually(func() error {
					return verifyCondition(c.Core().Nodes(), node.Name, condition, test.conditionType, test.conditionReason, test.conditionMessage)
				}, pollTimeout, pollInterval).Should(Succeed())
				By(fmt.Sprintf("Make sure node condition %q is stable", condition))
				Consistently(func() error {
					return verifyCondition(c.Core().Nodes(), node.Name, condition, test.conditionType, test.conditionReason, test.conditionMessage)
				}, pollConsistent, pollInterval).Should(Succeed())
			}
		})

		AfterEach(func() {
			if CurrentGinkgoTestDescription().Failed && framework.TestContext.DumpLogsOnFailure {
				By("Get node problem detector log")
				log, err := framework.GetPodLogs(c, ns, name, name)
				Expect(err).ShouldNot(HaveOccurred())
				framework.Logf("Node Problem Detector logs:\n %s", log)
			}
			By("Delete the node problem detector")
			c.Core().Pods(ns).Delete(name, v1.NewDeleteOptions(0))
			By("Wait for the node problem detector to disappear")
			Expect(framework.WaitForPodToDisappear(c, ns, name, labels.Everything(), pollInterval, pollTimeout)).To(Succeed())
			By("Delete the config map")
			c.Core().ConfigMaps(ns).Delete(configName, nil)
			By("Clean up the events")
			Expect(c.Core().Events(eventNamespace).DeleteCollection(v1.NewDeleteOptions(0), eventListOptions)).To(Succeed())
			By("Clean up the node condition")
			patch := []byte(fmt.Sprintf(`{"status":{"conditions":[{"$patch":"delete","type":"%s"}]}}`, condition))
			c.Core().RESTClient().Patch(api.StrategicMergePatchType).Resource("nodes").Name(node.Name).SubResource("status").Body(patch).Do()
			By("Clean up the temporary directory")
			framework.IssueSSHCommand(fmt.Sprintf("rm -r %s", tmpDir), framework.TestContext.Provider, node)
		})
	})
})

// verifyEvents verifies there are num specific events generated
func verifyEvents(e coreclientset.EventInterface, options v1.ListOptions, num int, reason, message string) error {
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
func verifyNoEvents(e coreclientset.EventInterface, options v1.ListOptions) error {
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
func verifyCondition(n coreclientset.NodeInterface, nodeName string, condition v1.NodeConditionType, status v1.ConditionStatus, reason, message string) error {
	node, err := n.Get(nodeName)
	if err != nil {
		return err
	}
	_, c := v1.GetNodeCondition(&node.Status, condition)
	if c == nil {
		return fmt.Errorf("node condition %q not found", condition)
	}
	if c.Status != status || c.Reason != reason || c.Message != message {
		return fmt.Errorf("unexpected node condition %q: %+v", condition, c)
	}
	return nil
}
