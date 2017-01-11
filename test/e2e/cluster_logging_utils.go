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
	"errors"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// ingestionTimeout is how long to keep retrying to wait for all the
	// logs to be ingested.
	ingestionTimeout = 10 * time.Minute
	// ingestionRetryDelay is how long test should wait between
	// two attempts to check for ingestion
	ingestionRetryDelay = 25 * time.Second

	synthLoggerPodName = "synthlogger"

	// expectedLinesCount is the number of log lines emitted (and checked) for each synthetic logging pod.
	expectedLinesCount = 100
)

func createSynthLogger(f *framework.Framework, linesCount int) {
	f.PodClient().Create(&v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name:      synthLoggerPodName,
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyOnFailure,
			Containers: []v1.Container{
				{
					Name:  synthLoggerPodName,
					Image: "gcr.io/google_containers/busybox:1.24",
					// notice: the subshell syntax is escaped with `$$`
					Command: []string{"/bin/sh", "-c", fmt.Sprintf("i=0; while [ $i -lt %d ]; do echo $i; i=`expr $i + 1`; done", linesCount)},
				},
			},
		},
	})
}

func reportLogsFromFluentdPod(f *framework.Framework) error {
	synthLoggerPod, err := f.PodClient().Get(synthLoggerPodName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("Failed to get synth logger pod due to %v", err)
	}

	synthLoggerNodeName := synthLoggerPod.Spec.NodeName
	if synthLoggerNodeName == "" {
		return errors.New("Synthlogger pod is not assigned to the node")
	}

	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": "fluentd-logging"}))
	options := v1.ListOptions{LabelSelector: label.String()}
	fluentdPods, err := f.ClientSet.Core().Pods(api.NamespaceSystem).List(options)

	for _, fluentdPod := range fluentdPods.Items {
		if fluentdPod.Spec.NodeName == synthLoggerNodeName {
			containerName := fluentdPod.Spec.Containers[0].Name
			logs, err := framework.GetPodLogs(f.ClientSet, api.NamespaceSystem, fluentdPod.Name, containerName)
			if err != nil {
				return fmt.Errorf("Failed to get logs from fluentd pod %s due to %v", fluentdPod.Name, err)
			}
			framework.Logf("Logs from fluentd pod %s:\n%s", fluentdPod.Name, logs)
			return nil
		}
	}

	return fmt.Errorf("Failed to find fluentd pod running on node %s", synthLoggerNodeName)
}
