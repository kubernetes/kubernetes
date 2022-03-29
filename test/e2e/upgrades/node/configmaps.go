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

package node

import (
	"context"
	"fmt"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
)

// ConfigMapUpgradeTest tests that a ConfigMap is available before and after
// a cluster upgrade.
type ConfigMapUpgradeTest struct {
	configMap *v1.ConfigMap
}

// Name returns the tracking name of the test.
func (ConfigMapUpgradeTest) Name() string {
	return "[sig-storage] [sig-api-machinery] configmap-upgrade"
}

// Setup creates a ConfigMap and then verifies that a pod can consume it.
func (t *ConfigMapUpgradeTest) Setup(f *framework.Framework) {
	configMapName := "upgrade-configmap"

	ns := f.Namespace

	t.configMap = &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      configMapName,
		},
		Data: map[string]string{
			"data": "some configmap data",
		},
	}

	ginkgo.By("Creating a ConfigMap")
	var err error
	if t.configMap, err = f.ClientSet.CoreV1().ConfigMaps(ns.Name).Create(context.TODO(), t.configMap, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test ConfigMap %s: %v", t.configMap.Name, err)
	}

	ginkgo.By("Making sure the ConfigMap is consumable")
	t.testPod(f)
}

// Test waits for the upgrade to complete, and then verifies that a
// pod can still consume the ConfigMap.
func (t *ConfigMapUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Consuming the ConfigMap after upgrade")
	t.testPod(f)
}

// Teardown cleans up any remaining resources.
func (t *ConfigMapUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// testPod creates a pod that consumes a ConfigMap and prints it out. The
// output is then verified.
func (t *ConfigMapUpgradeTest) testPod(f *framework.Framework) {
	volumeName := "configmap-volume"
	volumeMountPath := "/etc/configmap-volume"

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-configmap-" + string(uuid.NewUUID()),
			Namespace: t.configMap.ObjectMeta.Namespace,
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
					Name:  "configmap-volume-test",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args: []string{
						"mounttest",
						fmt.Sprintf("--file_content=%s/data", volumeMountPath),
						fmt.Sprintf("--file_mode=%s/data", volumeMountPath),
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
						},
					},
				},
				{
					Name:    "configmap-env-test",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "env"},
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

	expectedOutput := []string{
		"content of file \"/etc/configmap-volume/data\": some configmap data",
		"mode of file \"/etc/configmap-volume/data\": -rw-r--r--",
	}
	f.TestContainerOutput("volume consume configmap", pod, 0, expectedOutput)

	expectedOutput = []string{"CONFIGMAP_DATA=some configmap data"}
	f.TestContainerOutput("env consume configmap", pod, 1, expectedOutput)
}
