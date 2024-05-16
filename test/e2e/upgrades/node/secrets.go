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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	"k8s.io/kubernetes/test/e2e/upgrades"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
)

// SecretUpgradeTest test that a secret is available before and after
// a cluster upgrade.
type SecretUpgradeTest struct {
	secret *v1.Secret
}

// Name returns the tracking name of the test.
func (SecretUpgradeTest) Name() string { return "[sig-storage] [sig-api-machinery] secret-upgrade" }

// Setup creates a secret and then verifies that a pod can consume it.
func (t *SecretUpgradeTest) Setup(ctx context.Context, f *framework.Framework) {
	secretName := "upgrade-secret"

	ns := f.Namespace

	t.secret = &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: ns.Name,
			Name:      secretName,
		},
		Data: map[string][]byte{
			"data": []byte("keep it secret"),
		},
	}

	ginkgo.By("Creating a secret")
	var err error
	if t.secret, err = f.ClientSet.CoreV1().Secrets(ns.Name).Create(ctx, t.secret, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test secret %s: %v", t.secret.Name, err)
	}

	ginkgo.By("Making sure the secret is consumable")
	t.testPod(ctx, f)
}

// Test waits for the upgrade to complete, and then verifies that a
// pod can still consume the secret.
func (t *SecretUpgradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	ginkgo.By("Consuming the secret after upgrade")
	t.testPod(ctx, f)
}

// Teardown cleans up any remaining resources.
func (t *SecretUpgradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// testPod creates a pod that consumes a secret and prints it out. The
// output is then verified.
func (t *SecretUpgradeTest) testPod(ctx context.Context, f *framework.Framework) {
	volumeName := "secret-volume"
	volumeMountPath := "/etc/secret-volume"

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-secrets-" + string(uuid.NewUUID()),
			Namespace: t.secret.ObjectMeta.Namespace,
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{
							SecretName: t.secret.ObjectMeta.Name,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "secret-volume-test",
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
					Name:    "secret-env-test",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "env"},
					Env: []v1.EnvVar{
						{
							Name: "SECRET_DATA",
							ValueFrom: &v1.EnvVarSource{
								SecretKeyRef: &v1.SecretKeySelector{
									LocalObjectReference: v1.LocalObjectReference{
										Name: t.secret.ObjectMeta.Name,
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
		"content of file \"/etc/secret-volume/data\": keep it secret",
		"mode of file \"/etc/secret-volume/data\": -rw-r--r--",
	}

	e2eoutput.TestContainerOutput(ctx, f, "volume consume secrets", pod, 0, expectedOutput)

	expectedOutput = []string{"SECRET_DATA=keep it secret"}
	e2eoutput.TestContainerOutput(ctx, f, "env consume secrets", pod, 1, expectedOutput)
}
