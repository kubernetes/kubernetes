package managed

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestModifyStaticPodForPinnedManagementErrorStates(t *testing.T) {

	workloadAnnotations := map[string]string{
		"target.workload.openshift.io/management": `{"effect": "PreferredDuringScheduling"}`,
	}

	testCases := []struct {
		pod           *v1.Pod
		expectedError error
	}{
		{
			pod: createPod(workloadAnnotations, nil,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: nil,
					},
				}),
			expectedError: fmt.Errorf("managed container nginx does not have Resource.Requests"),
		},
		{
			pod: createPod(workloadAnnotations, nil,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
						},
					},
				}),
			expectedError: fmt.Errorf("managed container nginx does not have cpu requests"),
		},
		{
			pod: createPod(workloadAnnotations, nil,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU): resource.MustParse("100m"),
						},
					},
				}),
			expectedError: fmt.Errorf("managed container nginx does not have memory requests"),
		},
		{
			pod: createPod(workloadAnnotations,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
						},
					},
				}, nil),
			expectedError: fmt.Errorf("managed container nginx does not have cpu requests"),
		},
		{
			pod: createPod(workloadAnnotations,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU): resource.MustParse("100m"),
						},
					},
				}, nil),
			expectedError: fmt.Errorf("managed container nginx does not have memory requests"),
		},
		{
			pod: createPod(nil, nil,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
						},
					},
				}),
			expectedError: fmt.Errorf("managed container nginx does not have cpu requests"),
		},
		{
			pod: createPod(map[string]string{"something": "else"}, nil,
				&v1.Container{
					Name:  "nginx",
					Image: "test/image",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU): resource.MustParse("100m"),
						},
					},
				}),
			expectedError: fmt.Errorf("managed container nginx does not have memory requests"),
		},
	}

	for _, tc := range testCases {
		pod, workloadName, err := ModifyStaticPodForPinnedManagement(tc.pod)
		if err != nil && err.Error() != tc.expectedError.Error() {
			t.Errorf("ModifyStaticPodForPinned got error of (%v) but expected (%v)", err, tc.expectedError)
		}
		if pod != nil {
			t.Errorf("ModifyStaticPodForPinned should return pod with nil value")
		}
		if workloadName != "" {
			t.Errorf("ModifyStaticPodForPinned should return empty workloadName but got %v", workloadName)
		}
	}
}

func TestStaticPodManaged(t *testing.T) {
	testCases := []struct {
		pod                 *v1.Pod
		expectedAnnotations map[string]string
		isGuaranteed        bool
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
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("20m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
								Limits: v1.ResourceList{
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
				"resources.workload.openshift.io/c1":      `{"cpushares":20}`,
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
						"resources.workload.openshift.io/c1":      `{"cpushares":20}`,
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
								Limits: v1.ResourceList{
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
				WorkloadAnnotationWarning: qosWarning,
			},
			isGuaranteed: true,
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
								Limits: v1.ResourceList{
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
				WorkloadAnnotationWarning: qosWarning,
			},
			isGuaranteed: true,
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
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("0m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
								Limits: v1.ResourceList{
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
				WorkloadAnnotationWarning: qosWarning,
			},
			isGuaranteed: true,
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
			if container.Resources.Requests.Cpu().String() != "0" && !tc.isGuaranteed {
				t.Errorf("cpu requests should be 0 got %v", container.Resources.Requests.Cpu().String())
			}
			if container.Resources.Requests.Memory().String() == "0" && !tc.isGuaranteed {
				t.Errorf("memory requests were %v but should be %v", container.Resources.Requests.Memory().String(), container.Resources.Requests.Memory().String())
			}
			if _, exists := container.Resources.Requests[GenerateResourceName(workloadName)]; !exists && !tc.isGuaranteed {
				t.Errorf("managed capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
			if _, exists := container.Resources.Limits[GenerateResourceName(workloadName)]; !exists && !tc.isGuaranteed {
				t.Errorf("managed capacity label missing from pod %v and container %v limits", tc.pod.Name, container.Name)
			}
		}
	}
}

func TestStaticPodThrottle(t *testing.T) {
	testCases := []struct {
		pod                 *v1.Pod
		expectedAnnotations map[string]string
		isGuaranteed        bool
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
							Image: "test/nginx",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
									v1.ResourceName(v1.ResourceMemory): resource.MustParse("100m"),
								},
								Limits: v1.ResourceList{
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
				WorkloadAnnotationWarning: qosWarning,
			},
			isGuaranteed: true,
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
							Image: "test/nginx",
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
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
				WorkloadAnnotationWarning: qosWarning,
			},
			isGuaranteed: true,
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
			if container.Resources.Requests.Cpu().String() != "0" && !tc.isGuaranteed {
				t.Errorf("cpu requests should be 0 got %v", container.Resources.Requests.Cpu().String())
			}
			if container.Resources.Requests.Memory().String() == "0" && !tc.isGuaranteed {
				t.Errorf("memory requests were %v but should be %v", container.Resources.Requests.Memory().String(), container.Resources.Requests.Memory().String())
			}
			if _, exists := container.Resources.Requests[GenerateResourceName(workloadName)]; !exists && !tc.isGuaranteed {
				t.Errorf("managed capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
			if _, exists := container.Resources.Limits[GenerateResourceName(workloadName)]; !exists && !tc.isGuaranteed {
				t.Errorf("managed limits capacity label missing from pod %v and container %v", tc.pod.Name, container.Name)
			}
		}
	}
}

func createPod(annotations map[string]string, initContainer, container *v1.Container) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test",
			UID:         "12345",
			Namespace:   "mynamespace",
			Annotations: annotations,
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{},
		},
		Status: v1.PodStatus{
			Phase: v1.PodPending,
		},
	}

	if initContainer != nil {
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, *initContainer)
	}

	if container != nil {
		pod.Spec.Containers = append(pod.Spec.Containers, *container)
	}

	return pod
}
