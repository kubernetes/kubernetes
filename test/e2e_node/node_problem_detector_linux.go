// +build cgo,linux

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

package e2e_node

import (
	"fmt"
	"os"
	"path"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	coreclientset "k8s.io/client-go/kubernetes/typed/core/v1"
	nodeutil "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("NodeProblemDetector [NodeFeature:NodeProblemDetector]", func() {
	const (
		pollInterval   = 1 * time.Second
		pollConsistent = 5 * time.Second
		pollTimeout    = 1 * time.Minute
		image          = "k8s.gcr.io/node-problem-detector:v0.4.1"
	)
	f := framework.NewDefaultFramework("node-problem-detector")
	var c clientset.Interface
	var uid string
	var ns, name, configName, eventNamespace string
	var bootTime, nodeTime time.Time
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		uid = string(uuid.NewUUID())
		name = "node-problem-detector-" + uid
		configName = "node-problem-detector-config-" + uid
		// There is no namespace for Node, event recorder will set default namespace for node events.
		eventNamespace = metav1.NamespaceDefault
	})

	// Test system log monitor. We may add other tests if we have more problem daemons in the future.
	framework.KubeDescribe("SystemLogMonitor", func() {
		const (
			// Use test condition to avoid changing the real node condition in use.
			// TODO(random-liu): Now node condition could be arbitrary string, consider whether we need to
			// add TestCondition when switching to predefined condition list.
			condition = v1.NodeConditionType("TestCondition")

			// File paths used in the test.
			logFile      = "/log/test.log"
			configFile   = "/config/testconfig.json"
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
			permReason1    = "Permanent1"
			permMessage1   = "permanent error 1"
			permReason2    = "Permanent2"
			permMessage2   = "permanent error 2"
		)
		var source, config, hostLogFile string
		var lookback time.Duration
		var eventListOptions metav1.ListOptions

		BeforeEach(func() {
			By("Calculate Lookback duration")
			var err error

			nodeTime = time.Now()
			bootTime, err = util.GetBootTime()
			Expect(err).To(BeNil())

			// Set lookback duration longer than node up time.
			// Assume the test won't take more than 1 hour, in fact it usually only takes 90 seconds.
			lookback = nodeTime.Sub(bootTime) + time.Hour

			// Randomize the source name
			source = "kernel-monitor-" + uid
			config = `
			{
				"plugin": "filelog",
				"pluginConfig": {
					"timestamp": "^.{15}",
					"message": "kernel: \\[.*\\] (.*)",
					"timestampFormat": "` + time.Stamp + `"
				},
				"logPath": "` + logFile + `",
				"lookback": "` + lookback.String() + `",
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
					"reason": "` + permReason1 + `",
					"pattern": "` + permMessage1 + ".*" + `"
				},
				{
					"type": "permanent",
					"condition": "` + string(condition) + `",
					"reason": "` + permReason2 + `",
					"pattern": "` + permMessage2 + ".*" + `"
				}
				]
			}`
			By("Generate event list options")
			selector := fields.Set{
				"involvedObject.kind":      "Node",
				"involvedObject.name":      framework.TestContext.NodeName,
				"involvedObject.namespace": metav1.NamespaceAll,
				"source":                   source,
			}.AsSelector().String()
			eventListOptions = metav1.ListOptions{FieldSelector: selector}
			By("Create the test log file")
			Expect(err).NotTo(HaveOccurred())
			By("Create config map for the node problem detector")
			_, err = c.CoreV1().ConfigMaps(ns).Create(&v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: configName},
				Data:       map[string]string{path.Base(configFile): config},
			})
			Expect(err).NotTo(HaveOccurred())
			By("Create the node problem detector")
			hostPathType := new(v1.HostPathType)
			*hostPathType = v1.HostPathType(string(v1.HostPathFileOrCreate))
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
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
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
						{
							Name: localtimeVolume,
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: etcLocaltime,
									Type: hostPathType,
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    name,
							Image:   image,
							Command: []string{"sh", "-c", "touch " + logFile + " && /node-problem-detector --logtostderr --system-log-monitors=" + configFile + fmt.Sprintf(" --apiserver-override=%s?inClusterConfig=false", framework.TestContext.Host)},
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
									MountPath: path.Dir(logFile),
								},
								{
									Name:      localtimeVolume,
									MountPath: etcLocaltime,
								},
								{
									Name:      configVolume,
									MountPath: path.Dir(configFile),
								},
							},
						},
					},
				},
			})
			pod, err := f.PodClient().Get(name, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred())
			// TODO: remove hardcoded kubelet volume directory path
			// framework.TestContext.KubeVolumeDir is currently not populated for node e2e
			hostLogFile = "/var/lib/kubelet/pods/" + string(pod.UID) + "/volumes/kubernetes.io~empty-dir" + logFile
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
					timestamp:        bootTime.Add(-1 * time.Minute),
					message:          tempMessage,
					messageNum:       3,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should not change node condition for too old log",
					timestamp:        bootTime.Add(-1 * time.Minute),
					message:          permMessage1,
					messageNum:       1,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should generate event for old log within lookback duration",
					timestamp:        nodeTime,
					message:          tempMessage,
					messageNum:       3,
					events:           3,
					conditionReason:  defaultReason,
					conditionMessage: defaultMessage,
					conditionType:    v1.ConditionFalse,
				},
				{
					description:      "should change node condition for old log within lookback duration",
					timestamp:        nodeTime,
					message:          permMessage1,
					messageNum:       1,
					events:           3, // event number should not change
					conditionReason:  permReason1,
					conditionMessage: permMessage1,
					conditionType:    v1.ConditionTrue,
				},
				{
					description:      "should generate event for new log",
					timestamp:        nodeTime.Add(5 * time.Minute),
					message:          tempMessage,
					messageNum:       3,
					events:           6,
					conditionReason:  permReason1,
					conditionMessage: permMessage1,
					conditionType:    v1.ConditionTrue,
				},
				{
					description:      "should not update node condition with the same reason",
					timestamp:        nodeTime.Add(5 * time.Minute),
					message:          permMessage1 + "different message",
					messageNum:       1,
					events:           6, // event number should not change
					conditionReason:  permReason1,
					conditionMessage: permMessage1,
					conditionType:    v1.ConditionTrue,
				},
				{
					description:      "should change node condition for new log",
					timestamp:        nodeTime.Add(5 * time.Minute),
					message:          permMessage2,
					messageNum:       1,
					events:           6, // event number should not change
					conditionReason:  permReason2,
					conditionMessage: permMessage2,
					conditionType:    v1.ConditionTrue,
				},
			} {
				By(test.description)
				if test.messageNum > 0 {
					By(fmt.Sprintf("Inject %d logs: %q", test.messageNum, test.message))
					err := injectLog(hostLogFile, test.timestamp, test.message, test.messageNum)
					Expect(err).NotTo(HaveOccurred())
				}

				By(fmt.Sprintf("Wait for %d events generated", test.events))
				Eventually(func() error {
					return verifyEvents(c.CoreV1().Events(eventNamespace), eventListOptions, test.events, tempReason, tempMessage)
				}, pollTimeout, pollInterval).Should(Succeed())
				By(fmt.Sprintf("Make sure only %d events generated", test.events))
				Consistently(func() error {
					return verifyEvents(c.CoreV1().Events(eventNamespace), eventListOptions, test.events, tempReason, tempMessage)
				}, pollConsistent, pollInterval).Should(Succeed())

				By(fmt.Sprintf("Make sure node condition %q is set", condition))
				Eventually(func() error {
					return verifyNodeCondition(c.CoreV1().Nodes(), condition, test.conditionType, test.conditionReason, test.conditionMessage)
				}, pollTimeout, pollInterval).Should(Succeed())
				By(fmt.Sprintf("Make sure node condition %q is stable", condition))
				Consistently(func() error {
					return verifyNodeCondition(c.CoreV1().Nodes(), condition, test.conditionType, test.conditionReason, test.conditionMessage)
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
			f.PodClient().Delete(name, metav1.NewDeleteOptions(0))
			By("Wait for the node problem detector to disappear")
			Expect(framework.WaitForPodToDisappear(c, ns, name, labels.Everything(), pollInterval, pollTimeout)).To(Succeed())
			By("Delete the config map")
			c.CoreV1().ConfigMaps(ns).Delete(configName, nil)
			By("Clean up the events")
			Expect(c.CoreV1().Events(eventNamespace).DeleteCollection(metav1.NewDeleteOptions(0), eventListOptions)).To(Succeed())
			By("Clean up the node condition")
			patch := []byte(fmt.Sprintf(`{"status":{"conditions":[{"$patch":"delete","type":"%s"}]}}`, condition))
			c.CoreV1().RESTClient().Patch(types.StrategicMergePatchType).Resource("nodes").Name(framework.TestContext.NodeName).SubResource("status").Body(patch).Do()
		})
	})
})

// injectLog injects kernel log into specified file.
func injectLog(file string, timestamp time.Time, log string, num int) error {
	f, err := os.OpenFile(file, os.O_RDWR|os.O_APPEND, 0666)
	if err != nil {
		return err
	}
	defer f.Close()
	for i := 0; i < num; i++ {
		_, err := f.WriteString(fmt.Sprintf("%s kernel: [0.000000] %s\n", timestamp.Format(time.Stamp), log))
		if err != nil {
			return err
		}
	}
	return nil
}

// verifyEvents verifies there are num specific events generated
func verifyEvents(e coreclientset.EventInterface, options metav1.ListOptions, num int, reason, message string) error {
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
func verifyNoEvents(e coreclientset.EventInterface, options metav1.ListOptions) error {
	events, err := e.List(options)
	if err != nil {
		return err
	}
	if len(events.Items) != 0 {
		return fmt.Errorf("unexpected events: %v", events.Items)
	}
	return nil
}

// verifyNodeCondition verifies specific node condition is generated, if reason and message are empty, they will not be checked
func verifyNodeCondition(n coreclientset.NodeInterface, condition v1.NodeConditionType, status v1.ConditionStatus, reason, message string) error {
	node, err := n.Get(framework.TestContext.NodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	_, c := nodeutil.GetNodeCondition(&node.Status, condition)
	if c == nil {
		return fmt.Errorf("node condition %q not found", condition)
	}
	if c.Status != status || c.Reason != reason || c.Message != message {
		return fmt.Errorf("unexpected node condition %q: %+v", condition, c)
	}
	return nil
}
