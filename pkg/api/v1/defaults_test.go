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

package v1_test

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
	k8s_api_v1 "k8s.io/kubernetes/pkg/api/v1"
)

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := api.Codecs.LegacyCodec(v1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = api.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultReplicationController(t *testing.T) {
	tests := []struct {
		rc             *v1.ReplicationController
		expectLabels   bool
		expectSelector bool
	}{
		{
			rc: &v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: true,
		},
		{
			rc: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: true,
		},
		{
			rc: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: false,
		},
		{
			rc: &v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: false,
		},
	}

	for _, test := range tests {
		rc := test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if test.expectSelector != reflect.DeepEqual(rc2.Spec.Selector, rc2.Spec.Template.Labels) {
			if test.expectSelector {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Spec.Selector)
			} else {
				t.Errorf("unexpected equality: %v", rc.Spec.Selector)
			}
		}
		if test.expectLabels != reflect.DeepEqual(rc2.Labels, rc2.Spec.Template.Labels) {
			if test.expectLabels {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Labels)
			} else {
				t.Errorf("unexpected equality: %v", rc.Labels)
			}
		}
	}
}

func newInt(val int32) *int32 {
	p := new(int32)
	*p = val
	return p
}

func TestSetDefaultReplicationControllerReplicas(t *testing.T) {
	tests := []struct {
		rc             v1.ReplicationController
		expectReplicas int32
	}{
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 1,
		},
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Replicas: newInt(0),
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 0,
		},
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Replicas: newInt(3),
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 3,
		},
	}

	for _, test := range tests {
		rc := &test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if rc2.Spec.Replicas == nil {
			t.Errorf("unexpected nil Replicas")
		} else if test.expectReplicas != *rc2.Spec.Replicas {
			t.Errorf("expected: %d replicas, got: %d", test.expectReplicas, *rc2.Spec.Replicas)
		}
	}
}

type InitContainerValidator func(got, expected *v1.Container) error

func TestSetDefaultReplicationControllerInitContainers(t *testing.T) {
	assertEnvFieldRef := func(got, expected *v1.Container) error {
		if len(got.Env) != len(expected.Env) {
			return fmt.Errorf("different number of env: got <%v>, expected <%v>", len(got.Env), len(expected.Env))
		}

		for j := range got.Env {
			ge := &got.Env[j]
			ee := &expected.Env[j]

			if ge.Name != ee.Name {
				return fmt.Errorf("different name of env: got <%v>, expected <%v>", ge.Name, ee.Name)
			}

			if ge.ValueFrom.FieldRef.APIVersion != ee.ValueFrom.FieldRef.APIVersion {
				return fmt.Errorf("different api version of FieldRef <%v>: got <%v>, expected <%v>",
					ge.Name, ge.ValueFrom.FieldRef.APIVersion, ee.ValueFrom.FieldRef.APIVersion)
			}
		}
		return nil
	}

	assertImagePullPolicy := func(got, expected *v1.Container) error {
		if got.ImagePullPolicy != expected.ImagePullPolicy {
			return fmt.Errorf("different image pull poicy: got <%v>, expected <%v>", got.ImagePullPolicy, expected.ImagePullPolicy)
		}
		return nil
	}

	assertContainerPort := func(got, expected *v1.Container) error {
		if len(got.Ports) != len(expected.Ports) {
			return fmt.Errorf("different number of ports: got <%v>, expected <%v>", len(got.Ports), len(expected.Ports))
		}

		for i := range got.Ports {
			gp := &got.Ports[i]
			ep := &expected.Ports[i]

			if gp.Name != ep.Name {
				return fmt.Errorf("different name of port: got <%v>, expected <%v>", gp.Name, ep.Name)
			}

			if gp.Protocol != ep.Protocol {
				return fmt.Errorf("different port protocol <%v>: got <%v>, expected <%v>", gp.Name, gp.Protocol, ep.Protocol)
			}
		}

		return nil
	}

	assertResource := func(got, expected *v1.Container) error {
		if len(got.Resources.Limits) != len(expected.Resources.Limits) {
			return fmt.Errorf("different number of resources.Limits: got <%v>, expected <%v>", len(got.Resources.Limits), (expected.Resources.Limits))
		}

		for k, v := range got.Resources.Limits {
			if ev, found := expected.Resources.Limits[v1.ResourceName(k)]; !found {
				return fmt.Errorf("failed to find resource <%v> in expected resources.Limits.", k)
			} else {
				if ev.Value() != v.Value() {
					return fmt.Errorf("different resource.Limits: got <%v>, expected <%v>.", v.Value(), ev.Value())
				}
			}
		}

		if len(got.Resources.Requests) != len(expected.Resources.Requests) {
			return fmt.Errorf("different number of resources.Requests: got <%v>, expected <%v>", len(got.Resources.Requests), (expected.Resources.Requests))
		}

		for k, v := range got.Resources.Requests {
			if ev, found := expected.Resources.Requests[v1.ResourceName(k)]; !found {
				return fmt.Errorf("failed to find resource <%v> in expected resources.Requests.", k)
			} else {
				if ev.Value() != v.Value() {
					return fmt.Errorf("different resource.Requests: got <%v>, expected <%v>.", v.Value(), ev.Value())
				}
			}
		}

		return nil
	}

	assertProb := func(got, expected *v1.Container) error {
		// Assert LivenessProbe
		if got.LivenessProbe.Handler.HTTPGet.Path != expected.LivenessProbe.Handler.HTTPGet.Path ||
			got.LivenessProbe.Handler.HTTPGet.Scheme != expected.LivenessProbe.Handler.HTTPGet.Scheme ||
			got.LivenessProbe.FailureThreshold != expected.LivenessProbe.FailureThreshold ||
			got.LivenessProbe.SuccessThreshold != expected.LivenessProbe.SuccessThreshold ||
			got.LivenessProbe.PeriodSeconds != expected.LivenessProbe.PeriodSeconds ||
			got.LivenessProbe.TimeoutSeconds != expected.LivenessProbe.TimeoutSeconds {
			return fmt.Errorf("different LivenessProbe: got <%v>, expected <%v>", got.LivenessProbe, expected.LivenessProbe)
		}

		// Assert ReadinessProbe
		if got.ReadinessProbe.Handler.HTTPGet.Path != expected.ReadinessProbe.Handler.HTTPGet.Path ||
			got.ReadinessProbe.Handler.HTTPGet.Scheme != expected.ReadinessProbe.Handler.HTTPGet.Scheme ||
			got.ReadinessProbe.FailureThreshold != expected.ReadinessProbe.FailureThreshold ||
			got.ReadinessProbe.SuccessThreshold != expected.ReadinessProbe.SuccessThreshold ||
			got.ReadinessProbe.PeriodSeconds != expected.ReadinessProbe.PeriodSeconds ||
			got.ReadinessProbe.TimeoutSeconds != expected.ReadinessProbe.TimeoutSeconds {
			return fmt.Errorf("different ReadinessProbe: got <%v>, expected <%v>", got.ReadinessProbe, expected.ReadinessProbe)
		}

		return nil
	}

	assertLifeCycle := func(got, expected *v1.Container) error {
		if got.Lifecycle.PostStart.HTTPGet.Path != expected.Lifecycle.PostStart.HTTPGet.Path ||
			got.Lifecycle.PostStart.HTTPGet.Scheme != expected.Lifecycle.PostStart.HTTPGet.Scheme {
			return fmt.Errorf("different LifeCycle: got <%v>, expected <%v>", got.Lifecycle, expected.Lifecycle)
		}

		return nil
	}

	cpu, _ := resource.ParseQuantity("100m")
	mem, _ := resource.ParseQuantity("100Mi")

	tests := []struct {
		name       string
		rc         v1.ReplicationController
		expected   []v1.Container
		validators []InitContainerValidator
	}{
		{
			name: "imagePullIPolicy",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "install",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					ImagePullPolicy: v1.PullAlways,
				},
			},
			validators: []InitContainerValidator{assertImagePullPolicy},
		},
		{
			name: "FieldRef",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Env: []v1.EnvVar{
										{
											Name: "MY_POD_IP",
											ValueFrom: &v1.EnvVarSource{
												FieldRef: &v1.ObjectFieldSelector{
													APIVersion: "",
													FieldPath:  "status.podIP",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Env: []v1.EnvVar{
						{
							Name: "MY_POD_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "status.podIP",
								},
							},
						},
					},
				},
			},
			validators: []InitContainerValidator{assertEnvFieldRef},
		},
		{
			name: "ContainerPort",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Ports: []v1.ContainerPort{
										{
											Name: "default",
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							Name:     "default",
							Protocol: v1.ProtocolTCP,
						},
					},
				},
			},
			validators: []InitContainerValidator{assertContainerPort},
		},
		{
			name: "Resources",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Resources: v1.ResourceRequirements{
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    cpu,
							v1.ResourceMemory: mem,
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu,
							v1.ResourceMemory: mem,
						},
					},
				},
			},
			validators: []InitContainerValidator{assertResource},
		},
		{
			name: "Probe",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									LivenessProbe: &v1.Probe{
										Handler: v1.Handler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
									ReadinessProbe: &v1.Probe{
										Handler: v1.Handler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					LivenessProbe: &v1.Probe{
						Handler: v1.Handler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						TimeoutSeconds:   1,
						PeriodSeconds:    10,
						SuccessThreshold: 1,
						FailureThreshold: 3,
					},
					ReadinessProbe: &v1.Probe{
						Handler: v1.Handler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						TimeoutSeconds:   1,
						PeriodSeconds:    10,
						SuccessThreshold: 1,
						FailureThreshold: 3,
					},
				},
			},
			validators: []InitContainerValidator{assertProb},
		},
		{
			name: "LifeCycle",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Ports: []v1.ContainerPort{
										{
											Name: "default",
										},
									},
									Lifecycle: &v1.Lifecycle{
										PostStart: &v1.Handler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
										PreStop: &v1.Handler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Lifecycle: &v1.Lifecycle{
						PostStart: &v1.Handler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						PreStop: &v1.Handler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
					},
				},
			},
			validators: []InitContainerValidator{assertLifeCycle},
		},
	}

	assertInitContainers := func(got, expected []v1.Container, validators []InitContainerValidator) error {
		if len(got) != len(expected) {
			return fmt.Errorf("different number of init container: got <%d>, expected <%d>",
				len(got), len(expected))
		}

		for i := range got {
			g := &got[i]
			e := &expected[i]

			for _, validator := range validators {
				if err := validator(g, e); err != nil {
					return err
				}
			}
		}

		return nil
	}

	for _, test := range tests {
		rc := &test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}

		if err := assertInitContainers(rc2.Spec.Template.Spec.InitContainers, test.expected, test.validators); err != nil {
			t.Errorf("test %v failed: %v", test.name, err)
		}
	}
}

func TestSetDefaultService(t *testing.T) {
	svc := &v1.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*v1.Service)
	if svc2.Spec.SessionAffinity != v1.ServiceAffinityNone {
		t.Errorf("Expected default session affinity type:%s, got: %s", v1.ServiceAffinityNone, svc2.Spec.SessionAffinity)
	}
	if svc2.Spec.SessionAffinityConfig != nil {
		t.Errorf("Expected empty session affinity config when session affinity type: %s, got: %v", v1.ServiceAffinityNone, svc2.Spec.SessionAffinityConfig)
	}
	if svc2.Spec.Type != v1.ServiceTypeClusterIP {
		t.Errorf("Expected default type:%s, got: %s", v1.ServiceTypeClusterIP, svc2.Spec.Type)
	}
}

func TestSetDefaultServiceSessionAffinityConfig(t *testing.T) {
	testCases := map[string]v1.Service{
		"SessionAffinityConfig is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity:       v1.ServiceAffinityClientIP,
				SessionAffinityConfig: nil,
			},
		},
		"ClientIP is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: nil,
				},
			},
		},
		"TimeoutSeconds is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: &v1.ClientIPConfig{
						TimeoutSeconds: nil,
					},
				},
			},
		},
	}
	for name, test := range testCases {
		obj2 := roundTrip(t, runtime.Object(&test))
		svc2 := obj2.(*v1.Service)
		if svc2.Spec.SessionAffinityConfig == nil || svc2.Spec.SessionAffinityConfig.ClientIP == nil || svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds == nil {
			t.Fatalf("Case: %s, unexpected empty SessionAffinityConfig/ClientIP/TimeoutSeconds when session affinity type: %s, got: %v", name, v1.ServiceAffinityClientIP, svc2.Spec.SessionAffinityConfig)
		}
		if *svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds != v1.DefaultClientIPServiceAffinitySeconds {
			t.Errorf("Case: %s, default TimeoutSeconds should be %d when session affinity type: %s, got: %d", name, v1.DefaultClientIPServiceAffinitySeconds, v1.ServiceAffinityClientIP, *svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
		}
	}
}

func TestSetDefaultSecretVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.Secret.DefaultMode
	expectedMode := v1.SecretVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected secret DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultConfigMapVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode
	expectedMode := v1.ConfigMapVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected v1.ConfigMap DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultDownwardAPIVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				DownwardAPI: &v1.DownwardAPIVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.DownwardAPI.DefaultMode
	expectedMode := v1.DownwardAPIVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected DownwardAPI DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultProjectedVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.Projected.DefaultMode
	expectedMode := v1.ProjectedVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected v1.ProjectedVolumeSource DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	s := &v1.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*v1.Secret)

	if s2.Type != v1.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", v1.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaultPersistentVolume(t *testing.T) {
	pv := &v1.PersistentVolume{}
	obj2 := roundTrip(t, runtime.Object(pv))
	pv2 := obj2.(*v1.PersistentVolume)

	if pv2.Status.Phase != v1.VolumePending {
		t.Errorf("Expected volume phase %v, got %v", v1.VolumePending, pv2.Status.Phase)
	}
	if pv2.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimRetain {
		t.Errorf("Expected pv reclaim policy %v, got %v", v1.PersistentVolumeReclaimRetain, pv2.Spec.PersistentVolumeReclaimPolicy)
	}
}

func TestSetDefaultPersistentVolumeClaim(t *testing.T) {
	pvc := &v1.PersistentVolumeClaim{}
	obj2 := roundTrip(t, runtime.Object(pvc))
	pvc2 := obj2.(*v1.PersistentVolumeClaim)

	if pvc2.Status.Phase != v1.ClaimPending {
		t.Errorf("Expected claim phase %v, got %v", v1.ClaimPending, pvc2.Status.Phase)
	}
}

func TestSetDefaulEndpointsProtocol(t *testing.T) {
	in := &v1.Endpoints{Subsets: []v1.EndpointSubset{
		{Ports: []v1.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Endpoints)

	for i := range out.Subsets {
		for j := range out.Subsets[i].Ports {
			if in.Subsets[i].Ports[j].Protocol == "" {
				if out.Subsets[i].Ports[j].Protocol != v1.ProtocolTCP {
					t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Subsets[i].Ports[j].Protocol)
				}
			} else {
				if out.Subsets[i].Ports[j].Protocol != in.Subsets[i].Ports[j].Protocol {
					t.Errorf("Expected protocol %s, got %s", in.Subsets[i].Ports[j].Protocol, out.Subsets[i].Ports[j].Protocol)
				}
			}
		}
	}
}

func TestSetDefaulServiceTargetPort(t *testing.T) {
	in := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1234}}}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(1234) {
		t.Errorf("Expected TargetPort to be defaulted, got %v", out.Spec.Ports[0].TargetPort)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1234, TargetPort: intstr.FromInt(5678)}}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(5678) {
		t.Errorf("Expected TargetPort to be unchanged, got %v", out.Spec.Ports[0].TargetPort)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
	// Unchanged if set.
	in := &v1.Service{Spec: v1.ServiceSpec{
		Ports: []v1.ServicePort{
			{Protocol: "UDP", Port: 9376, TargetPort: intstr.FromString("p")},
			{Protocol: "UDP", Port: 8675, TargetPort: intstr.FromInt(309)},
		},
	}}
	out := roundTrip(t, runtime.Object(in)).(*v1.Service)
	if out.Spec.Ports[0].Protocol != v1.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolUDP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromString("p") {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != v1.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolUDP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(309) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}

	// Defaulted.
	in = &v1.Service{Spec: v1.ServiceSpec{
		Ports: []v1.ServicePort{
			{Protocol: "", Port: 9376, TargetPort: intstr.FromString("")},
			{Protocol: "", Port: 8675, TargetPort: intstr.FromInt(0)},
		},
	}}
	out = roundTrip(t, runtime.Object(in)).(*v1.Service)
	if out.Spec.Ports[0].Protocol != v1.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromInt(int(in.Spec.Ports[0].Port)) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != v1.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt(int(in.Spec.Ports[1].Port)) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}
}

func TestSetDefaulServiceExternalTraffic(t *testing.T) {
	in := &v1.Service{}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("Expected ExternalTrafficPolicy to be empty, got %v", out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeNodePort}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != v1.ServiceExternalTrafficPolicyTypeCluster {
		t.Errorf("Expected ExternalTrafficPolicy to be %v, got %v", v1.ServiceExternalTrafficPolicyTypeCluster, out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != v1.ServiceExternalTrafficPolicyTypeCluster {
		t.Errorf("Expected ExternalTrafficPolicy to be %v, got %v", v1.ServiceExternalTrafficPolicyTypeCluster, out.Spec.ExternalTrafficPolicy)
	}
}

func TestSetDefaultNamespace(t *testing.T) {
	s := &v1.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*v1.Namespace)

	if s2.Status.Phase != v1.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", v1.NamespaceActive, s2.Status.Phase)
	}
}

func TestSetDefaultPodSpecHostNetwork(t *testing.T) {
	portNum := int32(8080)
	s := v1.PodSpec{}
	s.HostNetwork = true
	s.Containers = []v1.Container{
		{
			Ports: []v1.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Ports: []v1.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*v1.Pod)
	s2 := pod2.Spec

	hostPortNum := s2.Containers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}

	hostPortNum = s2.InitContainers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}
}

func TestSetDefaultNodeExternalID(t *testing.T) {
	name := "node0"
	n := &v1.Node{}
	n.Name = name
	obj2 := roundTrip(t, runtime.Object(n))
	n2 := obj2.(*v1.Node)
	if n2.Spec.ExternalID != name {
		t.Errorf("Expected default External ID: %s, got: %s", name, n2.Spec.ExternalID)
	}
	if n2.Spec.ProviderID != "" {
		t.Errorf("Expected empty default Cloud Provider ID, got: %s", n2.Spec.ProviderID)
	}
}

func TestSetDefaultNodeStatusAllocatable(t *testing.T) {
	capacity := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("1000m"),
		v1.ResourceMemory: resource.MustParse("10G"),
	}
	allocatable := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("500m"),
		v1.ResourceMemory: resource.MustParse("5G"),
	}
	tests := []struct {
		capacity            v1.ResourceList
		allocatable         v1.ResourceList
		expectedAllocatable v1.ResourceList
	}{{ // Everything set, no defaulting.
		capacity:            capacity,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Allocatable set, no defaulting.
		capacity:            nil,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Capacity set, allocatable defaults to capacity.
		capacity:            capacity,
		allocatable:         nil,
		expectedAllocatable: capacity,
	}, { // Nothing set, allocatable "defaults" to capacity.
		capacity:            nil,
		allocatable:         nil,
		expectedAllocatable: nil,
	}}

	copyResourceList := func(rl v1.ResourceList) v1.ResourceList {
		if rl == nil {
			return nil
		}
		copy := make(v1.ResourceList, len(rl))
		for k, v := range rl {
			copy[k] = *v.Copy()
		}
		return copy
	}

	resourceListsEqual := func(a v1.ResourceList, b v1.ResourceList) bool {
		if len(a) != len(b) {
			return false
		}
		for k, v := range a {
			vb, found := b[k]
			if !found {
				return false
			}
			if v.Cmp(vb) != 0 {
				return false
			}
		}
		return true
	}

	for i, testcase := range tests {
		node := v1.Node{
			Status: v1.NodeStatus{
				Capacity:    copyResourceList(testcase.capacity),
				Allocatable: copyResourceList(testcase.allocatable),
			},
		}
		node2 := roundTrip(t, runtime.Object(&node)).(*v1.Node)
		actual := node2.Status.Allocatable
		expected := testcase.expectedAllocatable
		if !resourceListsEqual(expected, actual) {
			t.Errorf("[%d] Expected v1.NodeStatus.Allocatable: %+v; Got: %+v", i, expected, actual)
		}
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
	s := v1.PodSpec{
		Containers: []v1.Container{
			{
				Env: []v1.EnvVar{
					{
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{},
						},
					},
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*v1.Pod)
	s2 := pod2.Spec

	apiVersion := s2.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion
	if apiVersion != "v1" {
		t.Errorf("Expected default APIVersion v1, got: %v", apiVersion)
	}
}

func TestSetMinimumScalePod(t *testing.T) {
	// verify we default if limits are specified (and that request=0 is preserved)
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("1n"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2n"),
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("1n"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2n"),
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	k8s_api_v1.SetObjectDefaults_Pod(pod)

	if expect := resource.MustParse("1m"); expect.Cmp(pod.Spec.Containers[0].Resources.Requests[v1.ResourceMemory]) != 0 {
		t.Errorf("did not round resources: %#v", pod.Spec.Containers[0].Resources)
	}
	if expect := resource.MustParse("1m"); expect.Cmp(pod.Spec.InitContainers[0].Resources.Requests[v1.ResourceMemory]) != 0 {
		t.Errorf("did not round resources: %#v", pod.Spec.InitContainers[0].Resources)
	}
}

func TestSetDefaultRequestsPod(t *testing.T) {
	// verify we default if limits are specified (and that request=0 is preserved)
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultRequest := pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	if requestValue := defaultRequest[v1.ResourceMemory]; requestValue.String() != "0" {
		t.Errorf("Expected request memory: %s, got: %s", "0", requestValue.String())
	}
	defaultRequest = pod2.Spec.InitContainers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	if requestValue := defaultRequest[v1.ResourceMemory]; requestValue.String() != "0" {
		t.Errorf("Expected request memory: %s, got: %s", "0", requestValue.String())
	}

	// verify we do nothing if no limits are specified
	s = v1.PodSpec{}
	s.Containers = []v1.Container{{}}
	s.InitContainers = []v1.Container{{}}
	pod = &v1.Pod{
		Spec: s,
	}
	output = roundTrip(t, runtime.Object(pod))
	pod2 = output.(*v1.Pod)
	defaultRequest = pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
	defaultRequest = pod2.Spec.InitContainers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestDefaultRequestIsNotSetForReplicationController(t *testing.T) {
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
			},
		},
	}
	rc := &v1.ReplicationController{
		Spec: v1.ReplicationControllerSpec{
			Replicas: newInt(3),
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: s,
			},
		},
	}
	output := roundTrip(t, runtime.Object(rc))
	rc2 := output.(*v1.ReplicationController)
	defaultRequest := rc2.Spec.Template.Spec.Containers[0].Resources.Requests
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestSetDefaultLimitRangeItem(t *testing.T) {
	limitRange := &v1.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-defaults",
		},
		Spec: v1.LimitRangeSpec{
			Limits: []v1.LimitRangeItem{{
				Type: v1.LimitTypeContainer,
				Max: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
				Min: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Default:        v1.ResourceList{},
				DefaultRequest: v1.ResourceList{},
			}},
		},
	}

	output := roundTrip(t, runtime.Object(limitRange))
	limitRange2 := output.(*v1.LimitRange)
	defaultLimit := limitRange2.Spec.Limits[0].Default
	defaultRequest := limitRange2.Spec.Limits[0].DefaultRequest

	// verify that default cpu was set to the max
	defaultValue := defaultLimit[v1.ResourceCPU]
	if defaultValue.String() != "100m" {
		t.Errorf("Expected default cpu: %s, got: %s", "100m", defaultValue.String())
	}
	// verify that default request was set to the limit
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	// verify that if a min is provided, it will be the default if no limit is specified
	requestMinValue := defaultRequest[v1.ResourceMemory]
	if requestMinValue.String() != "100Mi" {
		t.Errorf("Expected request memory: %s, got: %s", "100Mi", requestMinValue.String())
	}
}

func TestSetDefaultProbe(t *testing.T) {
	originalProbe := v1.Probe{}
	expectedProbe := v1.Probe{
		InitialDelaySeconds: 0,
		TimeoutSeconds:      1,
		PeriodSeconds:       10,
		SuccessThreshold:    1,
		FailureThreshold:    3,
	}

	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{LivenessProbe: &originalProbe}},
		},
	}

	output := roundTrip(t, runtime.Object(pod)).(*v1.Pod)
	actualProbe := *output.Spec.Containers[0].LivenessProbe
	if actualProbe != expectedProbe {
		t.Errorf("Expected probe: %+v\ngot: %+v\n", expectedProbe, actualProbe)
	}
}

func TestSetDefaultSchedulerName(t *testing.T) {
	pod := &v1.Pod{}

	output := roundTrip(t, runtime.Object(pod)).(*v1.Pod)
	if output.Spec.SchedulerName != v1.DefaultSchedulerName {
		t.Errorf("Expected scheduler name: %+v\ngot: %+v\n", v1.DefaultSchedulerName, output.Spec.SchedulerName)
	}
}

func TestSetDefaultHostPathVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "foo"},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultType := pod2.Spec.Volumes[0].VolumeSource.HostPath.Type
	expectedType := v1.HostPathUnset

	if defaultType == nil || *defaultType != expectedType {
		t.Errorf("Expected v1.HostPathVolumeSource default type %v, got %v", expectedType, defaultType)
	}
}
