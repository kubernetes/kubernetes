package managed

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestStaticPodManaged(t *testing.T) {
	testCases := []struct {
		pod                 *v1.Pod
		expectedAnnotations map[string]string
	}{
		{
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
					Annotations: map[string]string{
						"target.workload.openshift.io/management": `{"effect": "PreferredDuringScheduling"}`,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expectedAnnotations: map[string]string{
				"target.workload.openshift.io/management": `{"effect": "PreferredDuringScheduling"}`,
				"resources.workload.openshift.io/nginx":   `{"cpushares":102}`,
			},
		},
		{
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
					Annotations: map[string]string{
						"target.workload.openshift.io/management": `{"effect": "PreferredDuringScheduling"}`,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "c1",
							Image: "test/nginx",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
						{
							Name:  "c2",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
						{
							Name:  "c_3",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expectedAnnotations: map[string]string{
				"target.workload.openshift.io/management": `{"effect": "PreferredDuringScheduling"}`,
				"resources.workload.openshift.io/c1":      `{"cpushares":102}`,
				"resources.workload.openshift.io/c2":      `{"cpushares":1024}`,
				"resources.workload.openshift.io/c_3":     `{"cpushares":1024}`,
			},
		},
	}

	for _, tc := range testCases {
		pod, workloadName, err := ModifyStaticPodForPinnedManagement(tc.pod)
		if err != nil {
			t.Errorf("ModifyStaticPodForPinned should not error")
		}
		for expectedKey, expectedValue := range tc.expectedAnnotations {
			value, exists := pod.Annotations[expectedKey]
			if !exists {
				t.Errorf("%v key not found", expectedKey)
			}
			if expectedValue != value {
				t.Errorf("'%v' key's value does not equal '%v' and got '%v'", expectedKey, expectedValue, value)
			}
		}
		for _, container := range pod.Spec.Containers {
			if container.Resources.Requests.Cpu().String() != "0" {
				t.Errorf("cpu requests should be 0 got %v", container.Resources.Requests.Cpu().String())
			}
			if container.Resources.Requests.Memory().String() == "0" {
				t.Errorf("memory requests were %v but should be %v", container.Resources.Requests.Memory().String(), container.Resources.Requests.Memory().String())
			}
			if _, exists := container.Resources.Requests[GenerateResourceName(workloadName)]; !exists {
				t.Errorf("managed capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
			if _, exists := container.Resources.Limits[GenerateResourceName(workloadName)]; !exists {
				t.Errorf("managed capacity label missing from pod %v and container %v limits", tc.pod.Name, container.Name)
			}
		}
	}
}

func TestStaticPodThrottle(t *testing.T) {
	testCases := []struct {
		pod                 *v1.Pod
		expectedAnnotations map[string]string
	}{
		{
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
					Annotations: map[string]string{
						"target.workload.openshift.io/throttle": `{"effect": "PreferredDuringScheduling"}`,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "nginx",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expectedAnnotations: map[string]string{
				"target.workload.openshift.io/throttle": `{"effect": "PreferredDuringScheduling"}`,
				"resources.workload.openshift.io/nginx": `{"cpushares":102}`,
			},
		},
		{
			pod: &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					UID:       "12345",
					Namespace: "mynamespace",
					Annotations: map[string]string{
						"target.workload.openshift.io/throttle": `{"effect": "PreferredDuringScheduling"}`,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "c1",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
						{
							Name:  "c2",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
						{
							Name:  "c_3",
							Image: "test/image",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
							},
						},
					},
					SecurityContext: &v1.PodSecurityContext{},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			},
			expectedAnnotations: map[string]string{
				"target.workload.openshift.io/throttle": `{"effect": "PreferredDuringScheduling"}`,
				"resources.workload.openshift.io/c1":    `{"cpushares":102}`,
				"resources.workload.openshift.io/c2":    `{"cpushares":1024}`,
				"resources.workload.openshift.io/c_3":   `{"cpushares":1024}`,
			},
		},
	}

	for _, tc := range testCases {
		pod, workloadName, err := ModifyStaticPodForPinnedManagement(tc.pod)
		if err != nil {
			t.Errorf("ModifyStaticPodForPinned should not error")
		}
		for expectedKey, expectedValue := range tc.expectedAnnotations {
			value, exists := pod.Annotations[expectedKey]
			if !exists {
				t.Errorf("%v key not found", expectedKey)
			}
			if expectedValue != value {
				t.Errorf("'%v' key's value does not equal '%v' and got '%v'", expectedKey, expectedValue, value)
			}
		}
		for _, container := range pod.Spec.Containers {
			if container.Resources.Requests.Cpu().String() != "0" {
				t.Errorf("cpu requests should be 0 got %v", container.Resources.Requests.Cpu().String())
			}
			if container.Resources.Requests.Memory().String() == "0" {
				t.Errorf("memory requests were %v but should be %v", container.Resources.Requests.Memory().String(), container.Resources.Requests.Memory().String())
			}
			if _, exists := container.Resources.Requests[GenerateResourceName(workloadName)]; !exists {
				t.Errorf("managed capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
			if _, exists := container.Resources.Limits[GenerateResourceName(workloadName)]; !exists {
				t.Errorf("managed limits capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
		}
	}
}
