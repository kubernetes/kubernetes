/*
Copyright The Kubernetes Authors.

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

package v1alpha1_test

import (
	"bytes"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	nodev1alpha1 "k8s.io/api/node/v1alpha1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/node/install"
	checkpointutil "k8s.io/kubernetes/pkg/apis/node/util"
	"k8s.io/kubernetes/pkg/features"
)

func TestPodCheckpointTemplateDefaulting(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, true)
	for _, tc := range []struct {
		name   string
		mutate func(*v1.PodTemplateSpec)
	}{
		{name: "already defaulted template"},
		{name: "omitted template defaults", mutate: func(template *v1.PodTemplateSpec) {
			template.Spec.SchedulerName = ""
			template.Spec.Containers[0].TerminationMessagePolicy = ""
		}},
		{name: "explicit captured values", mutate: func(template *v1.PodTemplateSpec) {
			template.Spec.SchedulerName = "custom-scheduler"
			template.Spec.Containers[0].TerminationMessagePolicy = v1.TerminationMessageFallbackToLogsOnError
			template.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol = new(v1.HTTPProtocolHTTP2)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := defaultingTestPod()
			legacyscheme.Scheme.Default(source)
			captured := checkpointutil.SanitizePodTemplate(source)
			if tc.mutate != nil {
				tc.mutate(captured)
			}
			want := captured.DeepCopy()
			if want.Spec.SchedulerName == "" {
				want.Spec.SchedulerName = v1.DefaultSchedulerName
			}
			if want.Spec.Containers[0].TerminationMessagePolicy == "" {
				want.Spec.Containers[0].TerminationMessagePolicy = v1.TerminationMessageReadFile
			}
			read := roundTripCheckpointTemplate(t, captured)
			if diff := cmp.Diff(want, read); diff != "" {
				t.Errorf("decoded template differs (-want +got):\n%s", diff)
			}
		})
	}
}

func TestPodCheckpointTemplateKeepsPodOnlyDefaultsSeparate(t *testing.T) {
	for _, tc := range []struct {
		name                  string
		captureGate, readGate bool
	}{
		{name: "gate stays disabled"},
		{name: "gate enabled after capture", readGate: true},
		{name: "gate disabled after capture", captureGate: true},
		{name: "gate stays enabled", captureGate: true, readGate: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, tc.captureGate)
			source := defaultingTestPod()
			legacyscheme.Scheme.Default(source)
			captured := checkpointutil.SanitizePodTemplate(source)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.H2CContainerProbe, tc.readGate)
			read := roundTripCheckpointTemplate(t, captured)
			if diff := cmp.Diff(captured, read); diff != "" {
				t.Errorf("gate change altered the captured template (-want +got):\n%s", diff)
			}
			restored := &v1.Pod{Spec: *read.Spec.DeepCopy()}
			legacyscheme.Scheme.Default(restored)
			got := checkpointutil.SanitizePodTemplate(restored).Spec
			if tc.readGate && !tc.captureGate {
				if read.Spec.Containers[0].ReadinessProbe.HTTPGet.Protocol != nil {
					t.Fatal("checkpoint decoding applied a Pod-only protocol default")
				}
				protocol := got.Containers[0].ReadinessProbe.HTTPGet.Protocol
				if protocol == nil {
					t.Fatal("restore Pod protocol is omitted, want HTTP1")
				}
				if *protocol != v1.HTTPProtocolHTTP1 {
					t.Fatalf("restore Pod protocol = %q, want HTTP1", *protocol)
				}
				if equality.Semantic.DeepEqual(read.Spec, got) {
					t.Fatal("the Pod-only default must remain visible to the restore comparison")
				}
			} else if diff := cmp.Diff(read.Spec, got); diff != "" {
				t.Errorf("template and restore Pod differ (-want +got):\n%s", diff)
			}
		})
	}
}

func defaultingTestPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "source", Namespace: "ns"},
		Spec: v1.PodSpec{Containers: []v1.Container{{
			Name:  "app",
			Image: "registry.example/app@sha256:0000000000000000000000000000000000000000000000000000000000000000",
			ReadinessProbe: &v1.Probe{ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{Port: intstr.FromInt32(8080)},
			}},
		}}},
	}
}

func roundTripCheckpointTemplate(t *testing.T, captured *v1.PodTemplateSpec) *v1.PodTemplateSpec {
	t.Helper()
	checkpoint := &nodev1alpha1.PodCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint", Namespace: "ns"},
		Spec:       nodev1alpha1.PodCheckpointSpec{SourcePod: nodev1alpha1.PodReference{Name: "source"}},
		Status:     nodev1alpha1.PodCheckpointStatus{CheckpointedPodTemplate: captured},
	}
	before := checkpoint.DeepCopy()
	stored, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(nodev1alpha1.SchemeGroupVersion), checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	encodedBefore := bytes.Clone(stored)
	read := &nodev1alpha1.PodCheckpoint{}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(nodev1alpha1.SchemeGroupVersion), stored, read); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(before, checkpoint); diff != "" {
		t.Errorf("codec changed the captured checkpoint (-want +got):\n%s", diff)
	}
	if !bytes.Equal(encodedBefore, stored) {
		t.Error("decoding changed the stored bytes")
	}
	if read.Status.CheckpointedPodTemplate == nil {
		t.Fatal("decoded checkpoint lost its template")
	}
	return read.Status.CheckpointedPodTemplate
}
