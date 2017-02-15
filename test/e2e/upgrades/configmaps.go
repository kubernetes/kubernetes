/*
Copyright 2017 The Kubernetes Authors.

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

package upgrades

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/wait"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// ConfigMapUpgradeTest tests that a ConfigMap is available before, during and after
// a cluster upgrade.
type ConfigMapUpgradeTest struct {
	configMap *v1.ConfigMap
	pod       *v1.Pod
}

// Setup creates a ConfigMap and a Pod to read it and verifies
// the read works
func (t *ConfigMapUpgradeTest) Setup(f *framework.Framework) {
	configMapName := "upgrade-configmap"

	// Grab a unique namespace so we don't collide.
	ns, err := f.CreateNamespace("configmap-upgrade", nil)
	framework.ExpectNoError(err)

	t.configMap = &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      configMapName,
		},
		Data: map[string]string{
			"data": "some configmap data",
		},
	}

	By("Creating a ConfigMap")
	if t.configMap, err = f.ClientSet.Core().ConfigMaps(ns.Name).Create(t.configMap); err != nil {
		framework.Failf("unable to create test ConfigMap %s: %v", t.configMap.Name, err)
	}

	By("Creating a pod to consume ConfigMap as a volume mount and env vars")
	pod, err := t.createTestPodforConfigMap(f)
	framework.ExpectNoError(err)
	t.pod = pod

	By("Waiting for ConfigMap pod to be ready")
	err = framework.WaitForPodsReady(f.ClientSet, t.configMap.Namespace, t.pod.Name, 1)
	framework.ExpectNoError(err)

	By("Making sure the ConfigMap is consumable")
	t.testPod(f)
}

// Test validates that the ConfigMap is consumable from the Pod during an upgrade (if applicable)
// and after upgrade
func (t *ConfigMapUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	testDuringUpgrade := upgrade == MasterUpgrade

	if testDuringUpgrade {
		By("Validating that the ConfigMap is consumable during upgrade")
		wait.Until(func() {
			t.testPod(f)
		}, framework.Poll, done)
	} else {
		<-done
	}

	By("Consuming the ConfigMap after upgrade")
	t.testPod(f)
}

// Teardown cleans up any remaining resources.
func (t *ConfigMapUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// create a Pod with two containers; the first mounts the ConfigMap as a volume
// and the second loads the CM as environment vars
func (t *ConfigMapUpgradeTest) createTestPodforConfigMap(f *framework.Framework) (*v1.Pod, error) {
	volumeName := "configmap-volume"
	volumeMountPath := "/etc/configmap-volume"
	podName := "pod-configmap"

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: t.configMap.Namespace,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: t.configMap.ObjectMeta.Name,
							},
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:    "configmap-volume-test",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", "while true; do sleep 5; done"},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
						},
					},
				},
				{
					Name:    "configmap-env-test",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sh", "-c", "while true; do sleep 5; done"},
					Env: []v1.EnvVar{
						{
							Name: "CONFIGMAP_DATA",
							ValueFrom: &v1.EnvVarSource{
								ConfigMapKeyRef: &v1.ConfigMapKeySelector{
									LocalObjectReference: v1.LocalObjectReference{
										Name: t.configMap.ObjectMeta.Name,
									},
									Key: "data",
								},
							},
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	pod, err := f.ClientSet.Core().Pods(t.configMap.Namespace).Create(pod)
	framework.ExpectNoError(err)

	return pod, err
}

// validate that the ConfigMap data is read correctly using both methods
func (t *ConfigMapUpgradeTest) testPod(f *framework.Framework) {
	expected := t.configMap.Data["data"]

	By("Checking contents of ConfigMap volume mount")
	res, err := framework.RunHostCmdOnContainer(t.configMap.Namespace, t.pod.Name, "configmap-volume-test", "cat /etc/configmap-volume/data")
	framework.ExpectNoError(err)

	res = strings.TrimSpace(res)

	if res != expected {
		framework.Failf("Expected '%v', got '%v'", expected, res)
	}

	By("Checking contents of ConfigMap env var")
	res, err = framework.RunHostCmdOnContainer(t.configMap.Namespace, t.pod.Name, "configmap-env-test", "echo $CONFIGMAP_DATA")
	framework.ExpectNoError(err)

	res = strings.TrimSpace(res)

	if res != expected {
		framework.Failf("Expected '%v', got '%v'", expected, res)
	}
}
