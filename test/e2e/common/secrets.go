/*
Copyright 2014 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"os"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Secrets", func() {
	f := framework.NewDefaultFramework("secrets")

	It("should be consumable from pods in volume [Conformance]", func() {
		doSecretE2EWithoutMapping(f, nil /* default mode */, "secret-test-"+string(uuid.NewUUID()), nil, nil)
	})

	It("should be consumable from pods in volume with defaultMode set [Conformance]", func() {
		defaultMode := int32(0400)
		doSecretE2EWithoutMapping(f, &defaultMode, "secret-test-"+string(uuid.NewUUID()), nil, nil)
	})

	It("should be consumable from pods in volume as non-root with defaultMode and fsGroup set [Conformance]", func() {
		defaultMode := int32(0440) /* setting fsGroup sets mode to at least 440 */
		fsGroup := int64(1001)
		uid := int64(1000)
		doSecretE2EWithoutMapping(f, &defaultMode, "secret-test-"+string(uuid.NewUUID()), &fsGroup, &uid)
	})

	It("should be consumable from pods in volume with mappings [Conformance]", func() {
		doSecretE2EWithMapping(f, nil)
	})

	It("should be consumable from pods in volume with mappings and Item Mode set [Conformance]", func() {
		mode := int32(0400)
		doSecretE2EWithMapping(f, &mode)
	})

	It("should be able to mount in a volume regardless of a different secret existing with same name in different namespace", func() {
		var (
			namespace2  *api.Namespace
			err         error
			secret2Name = "secret-test-" + string(uuid.NewUUID())
		)

		if namespace2, err = f.CreateNamespace("secret-namespace", nil); err != nil {
			framework.Failf("unable to create new namespace %s: %v", namespace2.Name, err)
		}

		secret2 := secretForTest(namespace2.Name, secret2Name)
		secret2.Data = map[string][]byte{
			"this_should_not_match_content_of_other_secret": []byte("similarly_this_should_not_match_content_of_other_secret\n"),
		}
		if secret2, err = f.ClientSet.Core().Secrets(namespace2.Name).Create(secret2); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret2.Name, err)
		}
		doSecretE2EWithoutMapping(f, nil /* default mode */, secret2.Name, nil, nil)
	})

	It("should be consumable in multiple volumes in a pod [Conformance]", func() {
		// This test ensures that the same secret can be mounted in multiple
		// volumes in the same pod.  This test case exists to prevent
		// regressions that break this use-case.
		var (
			name             = "secret-test-" + string(uuid.NewUUID())
			volumeName       = "secret-volume"
			volumeMountPath  = "/etc/secret-volume"
			volumeName2      = "secret-volume-2"
			volumeMountPath2 = "/etc/secret-volume-2"
			secret           = secretForTest(f.Namespace.Name, name)
		)

		By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		var err error
		if secret, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-secrets-" + string(uuid.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							Secret: &api.SecretVolumeSource{
								SecretName: name,
							},
						},
					},
					{
						Name: volumeName2,
						VolumeSource: api.VolumeSource{
							Secret: &api.SecretVolumeSource{
								SecretName: name,
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:  "secret-volume-test",
						Image: "gcr.io/google_containers/mounttest:0.7",
						Args: []string{
							"--file_content=/etc/secret-volume/data-1",
							"--file_mode=/etc/secret-volume/data-1"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
							{
								Name:      volumeName2,
								MountPath: volumeMountPath2,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume secrets", pod, 0, []string{
			"content of file \"/etc/secret-volume/data-1\": value-1",
			"mode of file \"/etc/secret-volume/data-1\": -rw-r--r--",
		})
	})

	It("should be consumable from pods in env vars [Conformance]", func() {
		name := "secret-test-" + string(uuid.NewUUID())
		secret := secretForTest(f.Namespace.Name, name)

		By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		var err error
		if secret, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-secrets-" + string(uuid.NewUUID()),
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:    "secret-env-test",
						Image:   "gcr.io/google_containers/busybox:1.24",
						Command: []string{"sh", "-c", "env"},
						Env: []api.EnvVar{
							{
								Name: "SECRET_DATA",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: name,
										},
										Key: "data-1",
									},
								},
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume secrets", pod, 0, []string{
			"SECRET_DATA=value-1",
		})
	})
})

func secretForTest(namespace, name string) *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1\n"),
			"data-2": []byte("value-2\n"),
			"data-3": []byte("value-3\n"),
		},
	}
}

func doSecretE2EWithoutMapping(f *framework.Framework, defaultMode *int32, secretName string, fsGroup *int64, uid *int64) {
	var (
		volumeName      = "secret-volume"
		volumeMountPath = "/etc/secret-volume"
		secret          = secretForTest(f.Namespace.Name, secretName)
	)

	By(fmt.Sprintf("Creating secret with name %s", secret.Name))
	var err error
	if secret, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret); err != nil {
		framework.Failf("unable to create test secret %s: %v", secret.Name, err)
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name:      "pod-secrets-" + string(uuid.NewUUID()),
			Namespace: f.Namespace.Name,
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: volumeName,
					VolumeSource: api.VolumeSource{
						Secret: &api.SecretVolumeSource{
							SecretName: secretName,
						},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "secret-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{
						"--file_content=/etc/secret-volume/data-1",
						"--file_mode=/etc/secret-volume/data-1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}

	if defaultMode != nil {
		pod.Spec.Volumes[0].VolumeSource.Secret.DefaultMode = defaultMode
	} else {
		mode := int32(0644)
		defaultMode = &mode
	}

	if fsGroup != nil || uid != nil {
		pod.Spec.SecurityContext = &api.PodSecurityContext{
			FSGroup:   fsGroup,
			RunAsUser: uid,
		}
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*defaultMode))
	expectedOutput := []string{
		"content of file \"/etc/secret-volume/data-1\": value-1",
		"mode of file \"/etc/secret-volume/data-1\": " + modeString,
	}

	f.TestContainerOutput("consume secrets", pod, 0, expectedOutput)
}

func doSecretE2EWithMapping(f *framework.Framework, mode *int32) {
	var (
		name            = "secret-test-map-" + string(uuid.NewUUID())
		volumeName      = "secret-volume"
		volumeMountPath = "/etc/secret-volume"
		secret          = secretForTest(f.Namespace.Name, name)
	)

	By(fmt.Sprintf("Creating secret with name %s", secret.Name))
	var err error
	if secret, err = f.ClientSet.Core().Secrets(f.Namespace.Name).Create(secret); err != nil {
		framework.Failf("unable to create test secret %s: %v", secret.Name, err)
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: "pod-secrets-" + string(uuid.NewUUID()),
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: volumeName,
					VolumeSource: api.VolumeSource{
						Secret: &api.SecretVolumeSource{
							SecretName: name,
							Items: []api.KeyToPath{
								{
									Key:  "data-1",
									Path: "new-path-data-1",
								},
							},
						},
					},
				},
			},
			Containers: []api.Container{
				{
					Name:  "secret-volume-test",
					Image: "gcr.io/google_containers/mounttest:0.7",
					Args: []string{
						"--file_content=/etc/secret-volume/new-path-data-1",
						"--file_mode=/etc/secret-volume/new-path-data-1"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumeMountPath,
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
		},
	}

	if mode != nil {
		pod.Spec.Volumes[0].VolumeSource.Secret.Items[0].Mode = mode
	} else {
		defaultItemMode := int32(0644)
		mode = &defaultItemMode
	}

	modeString := fmt.Sprintf("%v", os.FileMode(*mode))
	expectedOutput := []string{
		"content of file \"/etc/secret-volume/new-path-data-1\": value-1",
		"mode of file \"/etc/secret-volume/new-path-data-1\": " + modeString,
	}

	f.TestContainerOutput("consume secrets", pod, 0, expectedOutput)
}
