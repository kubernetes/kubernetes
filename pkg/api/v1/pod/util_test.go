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

package pod

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestFindPort(t *testing.T) {
	testCases := []struct {
		name       string
		containers []v1.Container
		port       intstr.IntOrString
		expected   int
		pass       bool
	}{{
		name:       "valid int, no ports",
		containers: []v1.Container{{}},
		port:       intstr.FromInt(93),
		expected:   93,
		pass:       true,
	}, {
		name: "valid int, with ports",
		containers: []v1.Container{{Ports: []v1.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "TCP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     intstr.FromInt(93),
		expected: 93,
		pass:     true,
	}, {
		name:       "valid str, no ports",
		containers: []v1.Container{{}},
		port:       intstr.FromString("p"),
		expected:   0,
		pass:       false,
	}, {
		name: "valid str, one ctr with ports",
		containers: []v1.Container{{Ports: []v1.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}, {
			Name:          "q",
			ContainerPort: 33,
			Protocol:      "TCP",
		}}}},
		port:     intstr.FromString("q"),
		expected: 33,
		pass:     true,
	}, {
		name: "valid str, two ctr with ports",
		containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}, {
			Name:          "q",
			ContainerPort: 33,
			Protocol:      "TCP",
		}}}},
		port:     intstr.FromString("q"),
		expected: 33,
		pass:     true,
	}, {
		name: "valid str, two ctr with same port",
		containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
			Name:          "",
			ContainerPort: 11,
			Protocol:      "UDP",
		}, {
			Name:          "p",
			ContainerPort: 22,
			Protocol:      "TCP",
		}, {
			Name:          "q",
			ContainerPort: 22,
			Protocol:      "TCP",
		}}}},
		port:     intstr.FromString("q"),
		expected: 22,
		pass:     true,
	}, {
		name: "valid str, invalid protocol",
		containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
			Name:          "a",
			ContainerPort: 11,
			Protocol:      "snmp",
		},
		}}},
		port:     intstr.FromString("a"),
		expected: 0,
		pass:     false,
	}, {
		name: "valid hostPort",
		containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
			Name:          "a",
			ContainerPort: 11,
			HostPort:      81,
			Protocol:      "TCP",
		},
		}}},
		port:     intstr.FromString("a"),
		expected: 11,
		pass:     true,
	},
		{
			name: "invalid hostPort",
			containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
				Name:          "a",
				ContainerPort: 11,
				HostPort:      -1,
				Protocol:      "TCP",
			},
			}}},
			port:     intstr.FromString("a"),
			expected: 11,
			pass:     true,
			//this should fail but passes.
		},
		{
			name: "invalid ContainerPort",
			containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
				Name:          "a",
				ContainerPort: -1,
				Protocol:      "TCP",
			},
			}}},
			port:     intstr.FromString("a"),
			expected: -1,
			pass:     true,
			//this should fail but passes
		},
		{
			name: "HostIP Address",
			containers: []v1.Container{{}, {Ports: []v1.ContainerPort{{
				Name:          "a",
				ContainerPort: 11,
				HostIP:        "192.168.1.1",
				Protocol:      "TCP",
			},
			}}},
			port:     intstr.FromString("a"),
			expected: 11,
			pass:     true,
		},
	}

	for _, tc := range testCases {
		port, err := FindPort(&v1.Pod{Spec: v1.PodSpec{Containers: tc.containers}},
			&v1.ServicePort{Protocol: "TCP", TargetPort: tc.port})
		if err != nil && tc.pass {
			t.Errorf("unexpected error for %s: %v", tc.name, err)
		}
		if err == nil && !tc.pass {
			t.Errorf("unexpected non-error for %s: %d", tc.name, port)
		}
		if port != tc.expected {
			t.Errorf("wrong result for %s: expected %d, got %d", tc.name, tc.expected, port)
		}
	}
}

func TestPodSecrets(t *testing.T) {
	// Stub containing all possible secret references in a pod.
	// The names of the referenced secrets match struct paths detected by reflection.
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				EnvFrom: []v1.EnvFromSource{{
					SecretRef: &v1.SecretEnvSource{
						LocalObjectReference: v1.LocalObjectReference{
							Name: "Spec.Containers[*].EnvFrom[*].SecretRef"}}}},
				Env: []v1.EnvVar{{
					ValueFrom: &v1.EnvVarSource{
						SecretKeyRef: &v1.SecretKeySelector{
							LocalObjectReference: v1.LocalObjectReference{
								Name: "Spec.Containers[*].Env[*].ValueFrom.SecretKeyRef"}}}}}}},
			ImagePullSecrets: []v1.LocalObjectReference{{
				Name: "Spec.ImagePullSecrets"}},
			InitContainers: []v1.Container{{
				EnvFrom: []v1.EnvFromSource{{
					SecretRef: &v1.SecretEnvSource{
						LocalObjectReference: v1.LocalObjectReference{
							Name: "Spec.InitContainers[*].EnvFrom[*].SecretRef"}}}},
				Env: []v1.EnvVar{{
					ValueFrom: &v1.EnvVarSource{
						SecretKeyRef: &v1.SecretKeySelector{
							LocalObjectReference: v1.LocalObjectReference{
								Name: "Spec.InitContainers[*].Env[*].ValueFrom.SecretKeyRef"}}}}}}},
			Volumes: []v1.Volume{{
				VolumeSource: v1.VolumeSource{
					AzureFile: &v1.AzureFileVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.AzureFile.SecretName"}}}, {
				VolumeSource: v1.VolumeSource{
					CephFS: &v1.CephFSVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.CephFS.SecretRef"}}}}, {
				VolumeSource: v1.VolumeSource{
					FlexVolume: &v1.FlexVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.FlexVolume.SecretRef"}}}}, {
				VolumeSource: v1.VolumeSource{
					Projected: &v1.ProjectedVolumeSource{
						Sources: []v1.VolumeProjection{{
							Secret: &v1.SecretProjection{
								LocalObjectReference: v1.LocalObjectReference{
									Name: "Spec.Volumes[*].VolumeSource.Projected.Sources[*].Secret"}}}}}}}, {
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.RBD.SecretRef"}}}}, {
				VolumeSource: v1.VolumeSource{
					Secret: &v1.SecretVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.Secret.SecretName"}}}, {
				VolumeSource: v1.VolumeSource{
					Secret: &v1.SecretVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.Secret"}}}, {
				VolumeSource: v1.VolumeSource{
					ScaleIO: &v1.ScaleIOVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.ScaleIO.SecretRef"}}}}, {
				VolumeSource: v1.VolumeSource{
					ISCSI: &v1.ISCSIVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.ISCSI.SecretRef"}}}}, {
				VolumeSource: v1.VolumeSource{
					StorageOS: &v1.StorageOSVolumeSource{
						SecretRef: &v1.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.StorageOS.SecretRef"}}}}},
		},
	}
	extractedNames := sets.NewString()
	VisitPodSecretNames(pod, func(name string) bool {
		extractedNames.Insert(name)
		return true
	})

	// excludedSecretPaths holds struct paths to fields with "secret" in the name that are not actually references to secret API objects
	excludedSecretPaths := sets.NewString(
		"Spec.Volumes[*].VolumeSource.CephFS.SecretFile",
	)
	// expectedSecretPaths holds struct paths to fields with "secret" in the name that are references to secret API objects.
	// every path here should be represented as an example in the Pod stub above, with the secret name set to the path.
	expectedSecretPaths := sets.NewString(
		"Spec.Containers[*].EnvFrom[*].SecretRef",
		"Spec.Containers[*].Env[*].ValueFrom.SecretKeyRef",
		"Spec.ImagePullSecrets",
		"Spec.InitContainers[*].EnvFrom[*].SecretRef",
		"Spec.InitContainers[*].Env[*].ValueFrom.SecretKeyRef",
		"Spec.Volumes[*].VolumeSource.AzureFile.SecretName",
		"Spec.Volumes[*].VolumeSource.CephFS.SecretRef",
		"Spec.Volumes[*].VolumeSource.FlexVolume.SecretRef",
		"Spec.Volumes[*].VolumeSource.Projected.Sources[*].Secret",
		"Spec.Volumes[*].VolumeSource.RBD.SecretRef",
		"Spec.Volumes[*].VolumeSource.Secret",
		"Spec.Volumes[*].VolumeSource.Secret.SecretName",
		"Spec.Volumes[*].VolumeSource.ScaleIO.SecretRef",
		"Spec.Volumes[*].VolumeSource.ISCSI.SecretRef",
		"Spec.Volumes[*].VolumeSource.StorageOS.SecretRef",
	)
	secretPaths := collectSecretPaths(t, nil, "", reflect.TypeOf(&v1.Pod{}))
	secretPaths = secretPaths.Difference(excludedSecretPaths)
	if missingPaths := expectedSecretPaths.Difference(secretPaths); len(missingPaths) > 0 {
		t.Logf("Missing expected secret paths:\n%s", strings.Join(missingPaths.List(), "\n"))
		t.Error("Missing expected secret paths. Verify VisitPodSecretNames() is correctly finding the missing paths, then correct expectedSecretPaths")
	}
	if extraPaths := secretPaths.Difference(expectedSecretPaths); len(extraPaths) > 0 {
		t.Logf("Extra secret paths:\n%s", strings.Join(extraPaths.List(), "\n"))
		t.Error("Extra fields with 'secret' in the name found. Verify VisitPodSecretNames() is including these fields if appropriate, then correct expectedSecretPaths")
	}

	if missingNames := expectedSecretPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected secret names:\n%s", strings.Join(missingNames.List(), "\n"))
		t.Error("Missing expected secret names. Verify the pod stub above includes these references, then verify VisitPodSecretNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedSecretPaths); len(extraNames) > 0 {
		t.Logf("Extra secret names:\n%s", strings.Join(extraNames.List(), "\n"))
		t.Error("Extra secret names extracted. Verify VisitPodSecretNames() is correctly extracting secret names")
	}
}

// collectSecretPaths traverses the object, computing all the struct paths that lead to fields with "secret" in the name.
func collectSecretPaths(t *testing.T, path *field.Path, name string, tp reflect.Type) sets.String {
	secretPaths := sets.NewString()

	if tp.Kind() == reflect.Ptr {
		secretPaths.Insert(collectSecretPaths(t, path, name, tp.Elem()).List()...)
		return secretPaths
	}

	if strings.Contains(strings.ToLower(name), "secret") {
		secretPaths.Insert(path.String())
	}

	switch tp.Kind() {
	case reflect.Ptr:
		secretPaths.Insert(collectSecretPaths(t, path, name, tp.Elem()).List()...)
	case reflect.Struct:
		for i := 0; i < tp.NumField(); i++ {
			field := tp.Field(i)
			secretPaths.Insert(collectSecretPaths(t, path.Child(field.Name), field.Name, field.Type).List()...)
		}
	case reflect.Interface:
		t.Errorf("cannot find secret fields in interface{} field %s", path.String())
	case reflect.Map:
		secretPaths.Insert(collectSecretPaths(t, path.Key("*"), "", tp.Elem()).List()...)
	case reflect.Slice:
		secretPaths.Insert(collectSecretPaths(t, path.Key("*"), "", tp.Elem()).List()...)
	default:
		// all primitive types
	}

	return secretPaths
}

func newPod(now metav1.Time, ready bool, beforeSec int) *v1.Pod {
	conditionStatus := v1.ConditionFalse
	if ready {
		conditionStatus = v1.ConditionTrue
	}
	return &v1.Pod{
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:               v1.PodReady,
					LastTransitionTime: metav1.NewTime(now.Time.Add(-1 * time.Duration(beforeSec) * time.Second)),
					Status:             conditionStatus,
				},
			},
		},
	}
}

func TestIsPodAvailable(t *testing.T) {
	now := metav1.Now()
	tests := []struct {
		pod             *v1.Pod
		minReadySeconds int32
		expected        bool
	}{
		{
			pod:             newPod(now, false, 0),
			minReadySeconds: 0,
			expected:        false,
		},
		{
			pod:             newPod(now, true, 0),
			minReadySeconds: 1,
			expected:        false,
		},
		{
			pod:             newPod(now, true, 0),
			minReadySeconds: 0,
			expected:        true,
		},
		{
			pod:             newPod(now, true, 51),
			minReadySeconds: 50,
			expected:        true,
		},
	}

	for i, test := range tests {
		isAvailable := IsPodAvailable(test.pod, test.minReadySeconds, now)
		if isAvailable != test.expected {
			t.Errorf("[tc #%d] expected available pod: %t, got: %t", i, test.expected, isAvailable)
		}
	}
}

func TestSetInitContainersStatusesAnnotations(t *testing.T) {
	testStatuses := []v1.ContainerStatus{
		{
			Name: "test",
		},
	}
	value, _ := json.Marshal(testStatuses)
	testAnnotation := string(value)
	tests := []struct {
		name        string
		pod         *v1.Pod
		annotations map[string]string
	}{
		{
			name: "Populate annotations from status",
			pod: &v1.Pod{
				Status: v1.PodStatus{
					InitContainerStatuses: testStatuses,
				},
			},
			annotations: map[string]string{
				v1.PodInitContainerStatusesAnnotationKey:     testAnnotation,
				v1.PodInitContainerStatusesBetaAnnotationKey: testAnnotation,
			},
		},
		{
			name: "Clear annotations if no status",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						v1.PodInitContainerStatusesAnnotationKey:     testAnnotation,
						v1.PodInitContainerStatusesBetaAnnotationKey: testAnnotation,
					},
				},
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{},
				},
			},
			annotations: map[string]string{},
		},
	}
	for _, test := range tests {
		SetInitContainersStatusesAnnotations(test.pod)
		if !reflect.DeepEqual(test.pod.Annotations, test.annotations) {
			t.Errorf("%v, actual = %v, expected = %v", test.name, test.pod.Annotations, test.annotations)
		}
	}
}
