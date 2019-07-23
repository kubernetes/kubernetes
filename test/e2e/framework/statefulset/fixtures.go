/*
Copyright 2019 The Kubernetes Authors.

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

package statefulset

import (
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	e2efwk "k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// NewStatefulSet creates a new Webserver StatefulSet for testing. The StatefulSet is named name, is in namespace ns,
// statefulPodsMounts are the mounts that will be backed by PVs. podsMounts are the mounts that are mounted directly
// to the Pod. labels are the labels that will be usd for the StatefulSet selector.
func NewStatefulSet(name, ns, governingSvcName string, replicas int32, statefulPodMounts []v1.VolumeMount, podMounts []v1.VolumeMount, labels map[string]string) *appsv1.StatefulSet {
	mounts := append(statefulPodMounts, podMounts...)
	claims := []v1.PersistentVolumeClaim{}
	for _, m := range statefulPodMounts {
		claims = append(claims, NewStatefulSetPVC(m.Name))
	}

	vols := []v1.Volume{}
	for _, m := range podMounts {
		vols = append(vols, v1.Volume{
			Name: m.Name,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fmt.Sprintf("/tmp/%v", m.Name),
				},
			},
		})
	}

	return &appsv1.StatefulSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "StatefulSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:         "webserver",
							Image:        imageutils.GetE2EImage(imageutils.Httpd),
							VolumeMounts: mounts,
						},
					},
					Volumes: vols,
				},
			},
			UpdateStrategy:       appsv1.StatefulSetUpdateStrategy{Type: appsv1.RollingUpdateStatefulSetStrategyType},
			VolumeClaimTemplates: claims,
			ServiceName:          governingSvcName,
		},
	}
}

// NewStatefulSetPVC returns a PersistentVolumeClaim named name, for testing StatefulSets.
func NewStatefulSetPVC(name string) v1.PersistentVolumeClaim {
	return v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{
				v1.ReadWriteOnce,
			},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceStorage: *resource.NewQuantity(1, resource.BinarySI),
				},
			},
		},
	}
}

// CreateStatefulSetService creates a Headless Service with Name name and Selector set to match labels.
func CreateStatefulSetService(name string, labels map[string]string) *v1.Service {
	headlessService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
		},
	}
	headlessService.Spec.Ports = []v1.ServicePort{
		{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
	}
	headlessService.Spec.ClusterIP = "None"
	return headlessService
}

// SetHTTPProbe sets the pod template's ReadinessProbe for Webserver StatefulSet containers.
// This probe can then be controlled with BreakHTTPProbe() and RestoreHTTPProbe().
// Note that this cannot be used together with PauseNewPods().
func SetHTTPProbe(ss *appsv1.StatefulSet) {
	ss.Spec.Template.Spec.Containers[0].ReadinessProbe = httpProbe
}

// BreakHTTPProbe breaks the readiness probe for Nginx StatefulSet containers in ss.
func BreakHTTPProbe(c clientset.Interface, ss *appsv1.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/local/apache2/htdocs%v /tmp/ || true", path)
	return ExecInStatefulPods(c, ss, cmd)
}

// BreakPodHTTPProbe breaks the readiness probe for Nginx StatefulSet containers in one pod.
func BreakPodHTTPProbe(ss *appsv1.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /usr/local/apache2/htdocs%v /tmp/ || true", path)
	stdout, err := e2efwk.RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
	e2elog.Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

// RestoreHTTPProbe restores the readiness probe for Nginx StatefulSet containers in ss.
func RestoreHTTPProbe(c clientset.Interface, ss *appsv1.StatefulSet) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/local/apache2/htdocs/ || true", path)
	return ExecInStatefulPods(c, ss, cmd)
}

// RestorePodHTTPProbe restores the readiness probe for Nginx StatefulSet containers in pod.
func RestorePodHTTPProbe(ss *appsv1.StatefulSet, pod *v1.Pod) error {
	path := httpProbe.HTTPGet.Path
	if path == "" {
		return fmt.Errorf("path expected to be not empty: %v", path)
	}
	// Ignore 'mv' errors to make this idempotent.
	cmd := fmt.Sprintf("mv -v /tmp%v /usr/local/apache2/htdocs/ || true", path)
	stdout, err := e2efwk.RunHostCmdWithRetries(pod.Namespace, pod.Name, cmd, StatefulSetPoll, StatefulPodTimeout)
	e2elog.Logf("stdout of %v on %v: %v", cmd, pod.Name, stdout)
	return err
}

func hasPauseProbe(pod *v1.Pod) bool {
	probe := pod.Spec.Containers[0].ReadinessProbe
	return probe != nil && reflect.DeepEqual(probe.Exec.Command, pauseProbe.Exec.Command)
}

var httpProbe = &v1.Probe{
	Handler: v1.Handler{
		HTTPGet: &v1.HTTPGetAction{
			Path: "/index.html",
			Port: intstr.IntOrString{IntVal: 80},
		},
	},
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 1,
}

var pauseProbe = &v1.Probe{
	Handler: v1.Handler{
		Exec: &v1.ExecAction{Command: []string{"test", "-f", "/data/statefulset-continue"}},
	},
	PeriodSeconds:    1,
	SuccessThreshold: 1,
	FailureThreshold: 1,
}

type statefulPodsByOrdinal []v1.Pod

func (sp statefulPodsByOrdinal) Len() int {
	return len(sp)
}

func (sp statefulPodsByOrdinal) Swap(i, j int) {
	sp[i], sp[j] = sp[j], sp[i]
}

func (sp statefulPodsByOrdinal) Less(i, j int) bool {
	return getStatefulPodOrdinal(&sp[i]) < getStatefulPodOrdinal(&sp[j])
}

// PauseNewPods adds an always-failing ReadinessProbe to the StatefulSet PodTemplate.
// This causes all newly-created Pods to stay Unready until they are manually resumed
// with ResumeNextPod().
// Note that this cannot be used together with SetHTTPProbe().
func PauseNewPods(ss *appsv1.StatefulSet) {
	ss.Spec.Template.Spec.Containers[0].ReadinessProbe = pauseProbe
}

// ResumeNextPod allows the next Pod in the StatefulSet to continue by removing the ReadinessProbe
// added by PauseNewPods(), if it's still there.
// It fails the test if it finds any pods that are not in phase Running,
// or if it finds more than one paused Pod existing at the same time.
// This is a no-op if there are no paused pods.
func ResumeNextPod(c clientset.Interface, ss *appsv1.StatefulSet) {
	podList := GetPodList(c, ss)
	resumedPod := ""
	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning {
			e2elog.Failf("Found pod in phase %q, cannot resume", pod.Status.Phase)
		}
		if podutil.IsPodReady(&pod) || !hasPauseProbe(&pod) {
			continue
		}
		if resumedPod != "" {
			e2elog.Failf("Found multiple paused stateful pods: %v and %v", pod.Name, resumedPod)
		}
		_, err := e2efwk.RunHostCmdWithRetries(pod.Namespace, pod.Name, "dd if=/dev/zero of=/data/statefulset-continue bs=1 count=1 conv=fsync", StatefulSetPoll, StatefulPodTimeout)
		e2efwk.ExpectNoError(err)
		e2elog.Logf("Resumed pod %v", pod.Name)
		resumedPod = pod.Name
	}
}

// SortStatefulPods sorts pods by their ordinals
func SortStatefulPods(pods *v1.PodList) {
	sort.Sort(statefulPodsByOrdinal(pods.Items))
}

var statefulPodRegex = regexp.MustCompile("(.*)-([0-9]+)$")

func getStatefulPodOrdinal(pod *v1.Pod) int {
	ordinal := -1
	subMatches := statefulPodRegex.FindStringSubmatch(pod.Name)
	if len(subMatches) < 3 {
		return ordinal
	}
	if i, err := strconv.ParseInt(subMatches[2], 10, 32); err == nil {
		ordinal = int(i)
	}
	return ordinal
}
