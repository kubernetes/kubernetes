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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	"strconv"

	. "github.com/onsi/ginkgo"
)

// This test will create a pod with a secret volume and gitRepo volume
// Thus requests a secret, a git server pod, and a git server service
var _ = framework.KubeDescribe("EmptyDir wrapper volumes", func() {
	f := framework.NewDefaultFramework("emptydir-wrapper")

	It("should becomes running", func() {
		name := "emptydir-wrapper-test-" + string(uuid.NewUUID())
		volumeName := "secret-volume"
		volumeMountPath := "/etc/secret-volume"

		secret := &api.Secret{
			ObjectMeta: api.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1\n"),
			},
		}

		var err error
		if secret, err = f.Client.Secrets(f.Namespace.Name).Create(secret); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		gitServerPodName := "git-server-" + string(uuid.NewUUID())
		containerPort := 8000

		labels := map[string]string{"name": gitServerPodName}

		gitServerPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   gitServerPodName,
				Labels: labels,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:            "git-repo",
						Image:           "gcr.io/google_containers/fakegitserver:0.1",
						ImagePullPolicy: "IfNotPresent",
						Ports: []api.ContainerPort{
							{ContainerPort: int32(containerPort)},
						},
					},
				},
			},
		}

		if gitServerPod, err = f.Client.Pods(f.Namespace.Name).Create(gitServerPod); err != nil {
			framework.Failf("unable to create test git server pod %s: %v", gitServerPod.Name, err)
		}

		// Portal IP and port
		httpPort := 2345

		gitServerSvc := &api.Service{
			ObjectMeta: api.ObjectMeta{
				Name: "git-server-svc",
			},
			Spec: api.ServiceSpec{
				Selector: labels,
				Ports: []api.ServicePort{
					{
						Name:       "http-portal",
						Port:       int32(httpPort),
						TargetPort: intstr.FromInt(containerPort),
					},
				},
			},
		}

		if gitServerSvc, err = f.Client.Services(f.Namespace.Name).Create(gitServerSvc); err != nil {
			framework.Failf("unable to create test git server service %s: %v", gitServerSvc.Name, err)
		}

		gitVolumeName := "git-volume"
		gitVolumeMountPath := "/etc/git-volume"
		gitURL := "http://" + gitServerSvc.Spec.ClusterIP + ":" + strconv.Itoa(httpPort)
		gitRepo := "test"

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
						Name: gitVolumeName,
						VolumeSource: api.VolumeSource{
							GitRepo: &api.GitRepoVolumeSource{
								Repository: gitURL,
								Directory:  gitRepo,
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:  "secret-test",
						Image: "gcr.io/google_containers/test-webserver:e2e",
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
							{
								Name:      gitVolumeName,
								MountPath: gitVolumeMountPath,
							},
						},
					},
				},
			},
		}

		pod, err = f.Client.Pods(f.Namespace.Name).Create(pod)
		if err != nil {
			framework.Failf("unable to create pod %v: %v", pod.Name, err)
		}

		defer func() {
			By("Cleaning up the secret")
			if err := f.Client.Secrets(f.Namespace.Name).Delete(secret.Name); err != nil {
				framework.Failf("unable to delete secret %v: %v", secret.Name, err)
			}
			By("Cleaning up the git server pod")
			if err = f.Client.Pods(f.Namespace.Name).Delete(gitServerPod.Name, api.NewDeleteOptions(0)); err != nil {
				framework.Failf("unable to delete git server pod %v: %v", gitServerPod.Name, err)
			}
			By("Cleaning up the git server svc")
			if err = f.Client.Services(f.Namespace.Name).Delete(gitServerSvc.Name); err != nil {
				framework.Failf("unable to delete git server svc %v: %v", gitServerSvc.Name, err)
			}
			By("Cleaning up the git vol pod")
			if err = f.Client.Pods(f.Namespace.Name).Delete(pod.Name, api.NewDeleteOptions(0)); err != nil {
				framework.Failf("unable to delete git vol pod %v: %v", pod.Name, err)
			}
		}()

		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(f.Client, pod))
	})
})
