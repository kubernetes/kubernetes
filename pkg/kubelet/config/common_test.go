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

package config

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/utils/ptr"
)

func noDefault(klog.Logger, *core.Pod) error { return nil }

func TestDecodeSinglePod(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				SecurityContext:          securitycontext.ValidSecurityContextWithContainerDefaults(),
			}},
			SecurityContext:    &v1.PodSecurityContext{},
			SchedulerName:      v1.DefaultSchedulerName,
			EnableServiceLinks: &enableServiceLinks,
		},
		Status: v1.PodStatus{
			PodIP: "1.2.3.4",
			PodIPs: []v1.PodIP{
				{
					IP: "1.2.3.4",
				},
			},
		},
	}
	json, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	parsed, podOut, err := tryDecodeSinglePod(logger, json, noDefault)
	if !parsed {
		t.Errorf("expected to have parsed file: (%s)", string(json))
	}
	if err != nil {
		t.Errorf("unexpected error: %v (%s)", err, string(json))
	}
	if !reflect.DeepEqual(pod, podOut) {
		t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, podOut, string(json))
	}

	for _, gv := range legacyscheme.Scheme.PrioritizedVersionsForGroup(v1.GroupName) {
		info, _ := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), "application/yaml")
		encoder := legacyscheme.Codecs.EncoderForVersion(info.Serializer, gv)
		yaml, err := runtime.Encode(encoder, pod)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		parsed, podOut, err = tryDecodeSinglePod(logger, yaml, noDefault)
		if !parsed {
			t.Errorf("expected to have parsed file: (%s)", string(yaml))
		}
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", err, string(yaml))
		}
		if !reflect.DeepEqual(pod, podOut) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, podOut, string(yaml))
		}
	}
}

func TestDecodeSinglePodRejectsClusterTrustBundleVolumes(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				SecurityContext:          securitycontext.ValidSecurityContextWithContainerDefaults(),
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      "ctb-volume",
						MountPath: "/var/run/ctb-volume",
					},
				},
			}},
			Volumes: []v1.Volume{
				{
					Name: "ctb-volume",
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									ClusterTrustBundle: &v1.ClusterTrustBundleProjection{
										Name: ptr.To("my-ctb"),
										Path: "ctb-file",
									},
								},
							},
						},
					},
				},
			},
			SecurityContext:    &v1.PodSecurityContext{},
			SchedulerName:      v1.DefaultSchedulerName,
			EnableServiceLinks: &enableServiceLinks,
		},
		Status: v1.PodStatus{
			PodIP: "1.2.3.4",
			PodIPs: []v1.PodIP{
				{
					IP: "1.2.3.4",
				},
			},
		},
	}
	json, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_, _, err = tryDecodeSinglePod(logger, json, noDefault)
	if !strings.Contains(err.Error(), "may not reference clustertrustbundles") {
		t.Errorf("Got error %q, want %q", err, fmt.Errorf("static pods may not reference clustertrustbundles API objects"))
	}
}

func TestDecodeSinglePodRejectsResourceClaims(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				SecurityContext:          securitycontext.ValidSecurityContextWithContainerDefaults(),
				Resources: v1.ResourceRequirements{
					Claims: []v1.ResourceClaim{{
						Name: "my-claim",
					}},
				},
			}},
			ResourceClaims: []v1.PodResourceClaim{{
				Name:              "my-claim",
				ResourceClaimName: ptr.To("some-external-claim"),
			}},
			SecurityContext:    &v1.PodSecurityContext{},
			SchedulerName:      v1.DefaultSchedulerName,
			EnableServiceLinks: &enableServiceLinks,
		},
		Status: v1.PodStatus{
			PodIP: "1.2.3.4",
			PodIPs: []v1.PodIP{
				{
					IP: "1.2.3.4",
				},
			},
		},
	}
	json, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	_, _, err = tryDecodeSinglePod(logger, json, noDefault)
	if !strings.Contains(err.Error(), "may not reference resourceclaims") {
		t.Errorf("Got error %q, want %q", err, fmt.Errorf("static pods may not reference resourceclaims API objects"))
	}
}

func TestDecodeSinglePodWithOptions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerRestartRules, true)
	logger, _ := ktesting.NewTestContext(t)
	restartPolicyAlways := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				SecurityContext:          securitycontext.ValidSecurityContextWithContainerDefaults(),
				RestartPolicy:            &restartPolicyAlways,
			}},
		},
	}
	json, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), pod)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	parsed, _, err := tryDecodeSinglePod(logger, json, noDefault)
	if err != nil {
		t.Errorf("unexpected error: %v (%s)", err, string(json))
	}
	if !parsed {
		t.Errorf("expected to have parsed file: (%s)", string(json))
	}
}

func TestDecodePodList(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test",
			UID:       "12345",
			Namespace: "mynamespace",
		},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			DNSPolicy:                     v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: &grace,
			Containers: []v1.Container{{
				Name:                     "image",
				Image:                    "test/image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,

				SecurityContext: securitycontext.ValidSecurityContextWithContainerDefaults(),
			}},
			SecurityContext:    &v1.PodSecurityContext{},
			SchedulerName:      v1.DefaultSchedulerName,
			EnableServiceLinks: &enableServiceLinks,
		},
		Status: v1.PodStatus{
			PodIP: "1.2.3.4",
			PodIPs: []v1.PodIP{
				{
					IP: "1.2.3.4",
				},
			},
		},
	}
	podList := &v1.PodList{
		Items: []v1.Pod{*pod},
	}
	json, err := runtime.Encode(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), podList)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	parsed, podListOut, err := tryDecodePodList(logger, json, noDefault)
	if !parsed {
		t.Errorf("expected to have parsed file: (%s)", string(json))
	}
	if err != nil {
		t.Errorf("unexpected error: %v (%s)", err, string(json))
	}
	if !reflect.DeepEqual(podList, &podListOut) {
		t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", podList, &podListOut, string(json))
	}

	for _, gv := range legacyscheme.Scheme.PrioritizedVersionsForGroup(v1.GroupName) {
		info, _ := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), "application/yaml")
		encoder := legacyscheme.Codecs.EncoderForVersion(info.Serializer, gv)
		yaml, err := runtime.Encode(encoder, podList)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		parsed, podListOut, err = tryDecodePodList(logger, yaml, noDefault)
		if !parsed {
			t.Errorf("expected to have parsed file: (%s): %v", string(yaml), err)
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", err, string(yaml))
			continue
		}
		if !reflect.DeepEqual(podList, &podListOut) {
			t.Errorf("expected:\n%#v\ngot:\n%#v\n%s", pod, &podListOut, string(yaml))
		}
	}
}

func TestStaticPodNameGenerate(t *testing.T) {
	testCases := []struct {
		nodeName  types.NodeName
		podName   string
		expected  string
		overwrite string
		shouldErr bool
	}{
		{
			"node1",
			"static-pod1",
			"static-pod1-node1",
			"",
			false,
		},
		{
			"Node1",
			"static-pod1",
			"static-pod1-node1",
			"",
			false,
		},
		{
			"NODE1",
			"static-pod1",
			"static-pod1-node1",
			"static-pod1-NODE1",
			true,
		},
	}
	for _, c := range testCases {
		assert.Equal(t, c.expected, generatePodName(c.podName, c.nodeName), "wrong pod name generated")
		pod := podtest.MakePod("")
		pod.Name = c.podName
		if c.overwrite != "" {
			pod.Name = c.overwrite
		}
		errs := validation.ValidatePodCreate(pod, validation.PodValidationOptions{})
		if c.shouldErr {
			specNameErrored := false
			for _, err := range errs {
				if err.Field == "metadata.name" {
					specNameErrored = true
				}
			}
			assert.NotEmpty(t, specNameErrored, "expecting error")
		} else {
			for _, err := range errs {
				if err.Field == "metadata.name" {
					t.Fail()
				}
			}
		}
	}
}

func TestGetStaticPodPriorityWarning(t *testing.T) {
	testCases := []struct {
		podName           string
		priority          *int32
		priorityClassName string
		shouldWarn        bool
		warning           string
	}{
		{
			podName:           "static-pod-with-priorityclassname-and-nil-priority",
			priority:          nil,
			priorityClassName: "invalid-priority-class-name",
			shouldWarn:        true,
			warning:           "Static Pod has non-nil PriorityClassName and nil Priority. Kubelet will not make use of the priority. Mirror pod creation may fail.",
		},
		{
			podName:           "static-pod-with-priority-without-priorityclassname",
			priority:          ptr.To(int32(2000001000)),
			priorityClassName: "",
			shouldWarn:        true,
			warning:           "Static Pod has Priority set without PriorityClassName. Mirror Pod creation may fail if the default priority class doesn't match the given priority",
		},
		{
			podName:           "static-pod-with-priority-and-priorityclassname",
			priority:          ptr.To(int32(2000001000)),
			priorityClassName: "system-node-critical",
			shouldWarn:        false,
			warning:           "",
		},
		{
			podName:           "static-pod-with-invalid-priority",
			priority:          ptr.To(int32(0)),
			priorityClassName: "",
			shouldWarn:        true,
			warning:           "Static Pod has Priority set without PriorityClassName. Mirror Pod creation may fail if the default priority class doesn't match the given priority",
		},
		{
			podName:           "static-pod-with-invalid-priorityclassname",
			priority:          ptr.To(int32(2000001000)),
			priorityClassName: "invalid-priority-class-name",
			shouldWarn:        true,
			warning:           "Static Pod has non-standard values for Priority and PriorityClassName. Mirror Pod may be attempted to be evicted from the node ineffectively",
		},
	}

	for _, tc := range testCases {
		pod := &v1.Pod{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				UID:       "12345",
				Namespace: "mynamespace",
			},
			Spec: v1.PodSpec{
				RestartPolicy:     v1.RestartPolicyNever,
				Priority:          tc.priority,
				PriorityClassName: tc.priorityClassName,
				Containers: []v1.Container{{
					Name:  "image",
					Image: "test/image",
				}},
			},
		}
		internalPod := &api.Pod{}
		if err := k8s_api_v1.Convert_v1_Pod_To_core_Pod(pod, internalPod, nil); err != nil {
			t.Fatalf("%s: Cannot convert pod %#v, %#v", tc.podName, pod, err)
		}

		warning := getStaticPodPriorityWarning(internalPod)
		if tc.shouldWarn && warning != tc.warning {
			t.Errorf("unexpected error: %v (%s)", warning, pod)
		}
	}
}
