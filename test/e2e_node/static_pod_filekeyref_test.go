/*
Copyright 2025 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("FileKeyRefEnv", feature.EnvFiles, func() {
	f := framework.NewDefaultFramework("filekeyref-env")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should inject env from file using FileKeyRef in static pod", func(ctx context.Context) {
		ns := f.Namespace.Name
		staticPodName := "static-pod-" + string(uuid.NewUUID())
		mirrorPodName := staticPodName + "-" + framework.TestContext.NodeName
		podPath := kubeletCfg.StaticPodPath
		configKey := "CONFIG_1"
		configValue := "hello_static"

		// Compose the static pod manifest
		manifest := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Pod",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      staticPodName,
				Namespace: ns,
			},
			Spec: v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:         "setup-envfile",
						Image:        "busybox",
						Command:      []string{"sh", "-c", fmt.Sprintf("echo '%s=%s' > /data/config.env", configKey, configValue)},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "main",
						Image:   "busybox",
						Command: []string{"sh", "-c", fmt.Sprintf("echo $%s; sleep 10", configKey)},
						Env: []v1.EnvVar{{
							Name: configKey,
							ValueFrom: &v1.EnvVarSource{
								FileKeyRef: &v1.FileKeySelector{
									Path:       "config.env",
									VolumeName: "config",
									Key:        configKey,
									Optional:   ptr.To(false),
								},
							},
						}},
						VolumeMounts: []v1.VolumeMount{{Name: "config", MountPath: "/data"}},
					},
				},
				Volumes: []v1.Volume{{
					Name: "config",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		// Write the static pod manifest
		filePath := staticPodPath(podPath, staticPodName, ns)
		file, err := os.OpenFile(filePath, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
		framework.ExpectNoError(err)
		defer func() {
			_ = file.Close()
		}()

		podYaml, err := yaml.Marshal(manifest)
		framework.ExpectNoError(err)
		_, err = file.WriteString(string(podYaml))
		framework.ExpectNoError(err)
		defer func() {
			_ = os.Remove(filePath)
		}()

		// Wait for the mirror pod to be running
		gomega.Eventually(ctx, func(ctx context.Context) error {
			return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
		}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())

		// Get logs and check env value
		gomega.Eventually(ctx, func(ctx context.Context) error {
			logs, err := f.ClientSet.CoreV1().Pods(ns).GetLogs(mirrorPodName, &v1.PodLogOptions{Container: "main"}).Do(ctx).Raw()
			if err != nil {
				return fmt.Errorf("failed to get logs: %w", err)
			}
			if !strings.Contains(string(logs), configValue) {
				return fmt.Errorf("expected %q in logs, got: %q", configValue, logs)
			}
			return nil
		}, 1*time.Minute, 2*time.Second).Should(gomega.Succeed())
	})
})
