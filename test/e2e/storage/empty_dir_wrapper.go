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

package storage

import (
	"context"
	"fmt"
	"strconv"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

const (
	// These numbers are obtained empirically.
	// If you make them too low, you'll get flaky
	// tests instead of failing ones if the race bug reappears.
	// If you make volume counts or pod counts too high,
	// the tests may fail because mounting configmap/git_repo
	// volumes is not very fast and the tests may time out
	// waiting for pods to become Running.
	// And of course the higher are the numbers, the
	// slower are the tests.
	wrappedVolumeRaceConfigMapVolumeCount    = 50
	wrappedVolumeRaceConfigMapPodCount       = 5
	wrappedVolumeRaceConfigMapIterationCount = 3
	wrappedVolumeRaceGitRepoVolumeCount      = 50
	wrappedVolumeRaceGitRepoPodCount         = 5
	wrappedVolumeRaceGitRepoIterationCount   = 3
	wrappedVolumeRaceRCNamePrefix            = "wrapped-volume-race-"
)

var _ = utils.SIGDescribe("EmptyDir wrapper volumes", func() {
	f := framework.NewDefaultFramework("emptydir-wrapper")

	/*
		Release : v1.13
		Testname: EmptyDir Wrapper Volume, Secret and ConfigMap volumes, no conflict
		Description: Secret volume and ConfigMap volume is created with data. Pod MUST be able to start with Secret and ConfigMap volumes mounted into the container.
	*/
	framework.ConformanceIt("should not conflict", func() {
		name := "emptydir-wrapper-test-" + string(uuid.NewUUID())
		volumeName := "secret-volume"
		volumeMountPath := "/etc/secret-volume"

		secret := &v1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1\n"),
			},
		}

		var err error
		if secret, err = f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		configMapVolumeName := "configmap-volume"
		configMapVolumeMountPath := "/etc/configmap-volume"

		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      name,
			},
			BinaryData: map[string][]byte{
				"data-1": []byte("value-1\n"),
			},
		}

		if configMap, err = f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{}); err != nil {
			framework.Failf("unable to create test configMap %s: %v", configMap.Name, err)
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-secrets-" + string(uuid.NewUUID()),
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: name,
							},
						},
					},
					{
						Name: configMapVolumeName,
						VolumeSource: v1.VolumeSource{
							ConfigMap: &v1.ConfigMapVolumeSource{
								LocalObjectReference: v1.LocalObjectReference{
									Name: name,
								},
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:  "secret-test",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
						Args:  []string{"test-webserver"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
							{
								Name:      configMapVolumeName,
								MountPath: configMapVolumeMountPath,
							},
						},
					},
				},
			},
		}
		pod = f.PodClient().CreateSync(pod)

		defer func() {
			ginkgo.By("Cleaning up the secret")
			if err := f.ClientSet.CoreV1().Secrets(f.Namespace.Name).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{}); err != nil {
				framework.Failf("unable to delete secret %v: %v", secret.Name, err)
			}
			ginkgo.By("Cleaning up the configmap")
			if err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), configMap.Name, metav1.DeleteOptions{}); err != nil {
				framework.Failf("unable to delete configmap %v: %v", configMap.Name, err)
			}
			ginkgo.By("Cleaning up the pod")
			if err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0)); err != nil {
				framework.Failf("unable to delete pod %v: %v", pod.Name, err)
			}
		}()
	})

	// The following two tests check for the problem fixed in #29641.
	// In order to reproduce it you need to revert the fix, e.g. via
	// git revert -n df1e925143daf34199b55ffb91d0598244888cce
	// or
	// curl -sL https://github.com/kubernetes/kubernetes/pull/29641.patch | patch -p1 -R
	//
	// After that these tests will fail because some of the pods
	// they create never enter Running state.
	//
	// They need to be [Serial] and [Slow] because they try to induce
	// the race by creating pods with many volumes and container volume mounts,
	// which takes considerable time and may interfere with other tests.
	//
	// Probably should also try making tests for secrets and downwardapi,
	// but these cases are harder because tmpfs-based emptyDir
	// appears to be less prone to the race problem.

	/*
		Release : v1.13
		Testname: EmptyDir Wrapper Volume, ConfigMap volumes, no race
		Description: Create 50 ConfigMaps Volumes and 5 replicas of pod with these ConfigMapvolumes mounted. Pod MUST NOT fail waiting for Volumes.
	*/
	framework.ConformanceIt("should not cause race condition when used for configmaps [Serial]", func() {
		configMapNames := createConfigmapsForRace(f)
		defer deleteConfigMaps(f, configMapNames)
		volumes, volumeMounts := makeConfigMapVolumes(configMapNames)
		for i := 0; i < wrappedVolumeRaceConfigMapIterationCount; i++ {
			testNoWrappedVolumeRace(f, volumes, volumeMounts, wrappedVolumeRaceConfigMapPodCount)
		}
	})

	// Slow by design [~150 Seconds].
	// This test uses deprecated GitRepo VolumeSource so it MUST not be promoted to Conformance.
	// To provision a container with a git repo, mount an EmptyDir into an InitContainer that clones the repo using git, then mount the EmptyDir into the Pod's container.
	// This projected volume maps approach can also be tested with secrets and downwardapi VolumeSource but are less prone to the race problem.
	ginkgo.It("should not cause race condition when used for git_repo [Serial] [Slow]", func() {
		gitURL, gitRepo, cleanup := createGitServer(f)
		defer cleanup()
		volumes, volumeMounts := makeGitRepoVolumes(gitURL, gitRepo)
		for i := 0; i < wrappedVolumeRaceGitRepoIterationCount; i++ {
			testNoWrappedVolumeRace(f, volumes, volumeMounts, wrappedVolumeRaceGitRepoPodCount)
		}
	})
})

func createGitServer(f *framework.Framework) (gitURL string, gitRepo string, cleanup func()) {
	var err error
	gitServerPodName := "git-server-" + string(uuid.NewUUID())
	containerPort := 8000

	labels := map[string]string{"name": gitServerPodName}

	gitServerPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   gitServerPodName,
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "git-repo",
					Image:           imageutils.GetE2EImage(imageutils.Agnhost),
					Args:            []string{"fake-gitserver"},
					ImagePullPolicy: "IfNotPresent",
					Ports: []v1.ContainerPort{
						{ContainerPort: int32(containerPort)},
					},
				},
			},
		},
	}
	f.PodClient().CreateSync(gitServerPod)

	// Portal IP and port
	httpPort := 2345

	gitServerSvc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "git-server-svc",
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
			Ports: []v1.ServicePort{
				{
					Name:       "http-portal",
					Port:       int32(httpPort),
					TargetPort: intstr.FromInt(containerPort),
				},
			},
		},
	}

	if gitServerSvc, err = f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(context.TODO(), gitServerSvc, metav1.CreateOptions{}); err != nil {
		framework.Failf("unable to create test git server service %s: %v", gitServerSvc.Name, err)
	}

	return "http://" + gitServerSvc.Spec.ClusterIP + ":" + strconv.Itoa(httpPort), "test", func() {
		ginkgo.By("Cleaning up the git server pod")
		if err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), gitServerPod.Name, *metav1.NewDeleteOptions(0)); err != nil {
			framework.Failf("unable to delete git server pod %v: %v", gitServerPod.Name, err)
		}
		ginkgo.By("Cleaning up the git server svc")
		if err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Delete(context.TODO(), gitServerSvc.Name, metav1.DeleteOptions{}); err != nil {
			framework.Failf("unable to delete git server svc %v: %v", gitServerSvc.Name, err)
		}
	}
}

func makeGitRepoVolumes(gitURL, gitRepo string) (volumes []v1.Volume, volumeMounts []v1.VolumeMount) {
	for i := 0; i < wrappedVolumeRaceGitRepoVolumeCount; i++ {
		volumeName := fmt.Sprintf("racey-git-repo-%d", i)
		volumes = append(volumes, v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				GitRepo: &v1.GitRepoVolumeSource{
					Repository: gitURL,
					Directory:  gitRepo,
				},
			},
		})
		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      volumeName,
			MountPath: fmt.Sprintf("/etc/git-volume-%d", i),
		})
	}
	return
}

func createConfigmapsForRace(f *framework.Framework) (configMapNames []string) {
	ginkgo.By(fmt.Sprintf("Creating %d configmaps", wrappedVolumeRaceConfigMapVolumeCount))
	for i := 0; i < wrappedVolumeRaceConfigMapVolumeCount; i++ {
		configMapName := fmt.Sprintf("racey-configmap-%d", i)
		configMapNames = append(configMapNames, configMapName)
		configMap := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: f.Namespace.Name,
				Name:      configMapName,
			},
			Data: map[string]string{
				"data-1": "value-1",
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(context.TODO(), configMap, metav1.CreateOptions{})
		framework.ExpectNoError(err)
	}
	return
}

func deleteConfigMaps(f *framework.Framework, configMapNames []string) {
	ginkgo.By("Cleaning up the configMaps")
	for _, configMapName := range configMapNames {
		err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Delete(context.TODO(), configMapName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "unable to delete configMap %v", configMapName)
	}
}

func makeConfigMapVolumes(configMapNames []string) (volumes []v1.Volume, volumeMounts []v1.VolumeMount) {
	for i, configMapName := range configMapNames {
		volumeName := fmt.Sprintf("racey-configmap-%d", i)
		volumes = append(volumes, v1.Volume{
			Name: volumeName,
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{
					LocalObjectReference: v1.LocalObjectReference{
						Name: configMapName,
					},
					Items: []v1.KeyToPath{
						{
							Key:  "data-1",
							Path: "data-1",
						},
					},
				},
			},
		})
		volumeMounts = append(volumeMounts, v1.VolumeMount{
			Name:      volumeName,
			MountPath: fmt.Sprintf("/etc/config-%d", i),
		})
	}
	return
}

func testNoWrappedVolumeRace(f *framework.Framework, volumes []v1.Volume, volumeMounts []v1.VolumeMount, podCount int32) {
	const nodeHostnameLabelKey = "kubernetes.io/hostname"

	rcName := wrappedVolumeRaceRCNamePrefix + string(uuid.NewUUID())
	targetNode, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
	framework.ExpectNoError(err)

	ginkgo.By("Creating RC which spawns configmap-volume pods")
	affinity := &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      nodeHostnameLabelKey,
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{targetNode.Labels[nodeHostnameLabelKey]},
							},
						},
					},
				},
			},
		},
	}

	rc := &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: rcName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &podCount,
			Selector: map[string]string{
				"name": rcName,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"name": rcName},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "test-container",
							Image:        imageutils.GetE2EImage(imageutils.Pause),
							VolumeMounts: volumeMounts,
						},
					},
					Affinity:  affinity,
					DNSPolicy: v1.DNSDefault,
					Volumes:   volumes,
				},
			},
		},
	}
	_, err = f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(context.TODO(), rc, metav1.CreateOptions{})
	framework.ExpectNoError(err, "error creating replication controller")

	defer func() {
		err := e2erc.DeleteRCAndWaitForGC(f.ClientSet, f.Namespace.Name, rcName)
		framework.ExpectNoError(err)
	}()

	pods, err := e2epod.PodsCreated(f.ClientSet, f.Namespace.Name, rcName, podCount)
	framework.ExpectNoError(err, "error creating pods")

	ginkgo.By("Ensuring each pod is running")

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed waiting for pod %s to enter running state", pod.Name)
	}
}
