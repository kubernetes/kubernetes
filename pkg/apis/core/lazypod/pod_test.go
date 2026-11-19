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

package lazypod

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	core "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/utils/ptr"
)

func newTestProtobufCodecs(t *testing.T) (storageCodec *StorageCodec, responseEncoder runtime.Encoder, jsonEncoder runtime.Encoder) {
	t.Helper()
	protoSer := protobuf.NewSerializerWithOptions(legacyscheme.Scheme, legacyscheme.Scheme, protobuf.SerializerOptions{
		StreamingCollectionsEncoding: true,
	})
	baseStorage := versioning.NewDefaultingCodecForScheme(
		legacyscheme.Scheme,
		protoSer,
		legacyscheme.Codecs.UniversalDeserializer(),
		v1.SchemeGroupVersion,
		core.SchemeGroupVersion,
	)
	storageCodec = NewStorageCodec(baseStorage)

	cf := serializer.NewCodecFactory(legacyscheme.Scheme)
	protoInfo, ok := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeProtobuf)
	if !ok {
		t.Fatal("protobuf serializer info not found")
	}
	jsonInfo, ok := runtime.SerializerInfoForMediaType(cf.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		t.Fatal("json serializer info not found")
	}
	responseEncoder = cf.EncoderForVersion(protoInfo.Serializer, v1.SchemeGroupVersion)
	jsonEncoder = cf.EncoderForVersion(jsonInfo.Serializer, v1.SchemeGroupVersion)
	return storageCodec, responseEncoder, jsonEncoder
}

func makeDefaultedV1Pod() *v1.Pod {
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "pod-000",
			GenerateName:      "deploy-",
			Namespace:         "default",
			UID:               types.UID("eb30a43a-54b2-441a-8c42-3b7dec491d44"),
			Generation:        3,
			CreationTimestamp: metav1.NewTime(time.Date(2026, 4, 23, 12, 23, 1, 0, time.UTC)),
			Labels: map[string]string{
				"app":  "payments",
				"name": "small-deployment-00",
			},
			Annotations: map[string]string{
				"prometheus.io/scrape": "false",
			},
			Finalizers: []string{"example.com/finalizer"},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "apps/v1",
					Kind:       "ReplicaSet",
					Name:       "rs-1",
					UID:        types.UID("55338d45-98ff-4b84-949d-1619bff3c641"),
					Controller: ptr.To(true),
				},
			},
			ManagedFields: []metav1.ManagedFieldsEntry{
				{
					Manager:    "kube-controller-manager",
					Operation:  metav1.ManagedFieldsOperationUpdate,
					APIVersion: "v1",
					FieldsType: "FieldsV1",
					FieldsV1:   metav1.NewFieldsV1(`{"f:metadata":{"f:labels":{}}}`),
				},
			},
		},
		Spec: v1.PodSpec{
			NodeName:           "kind-worker0",
			ServiceAccountName: "default",
			SchedulerName:      v1.DefaultSchedulerName,
			RestartPolicy:      v1.RestartPolicyAlways,
			InitContainers: []v1.Container{
				{
					Name:    "init-0",
					Image:   "registry.k8s.io/pause:3.9",
					Command: []string{"/bin/true"},
				},
			},
			Containers: []v1.Container{
				{
					Name:  "main",
					Image: "registry.k8s.io/pause:3.9",
					Env: []v1.EnvVar{
						{
							Name: "POD_NAME",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{FieldPath: "metadata.name"},
							},
						},
					},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("5m"),
							v1.ResourceMemory: resource.MustParse("20M"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("64M"),
						},
					},
					Ports: []v1.ContainerPort{
						{Name: "http", ContainerPort: 8080},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "configmap",
					VolumeSource: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{Name: "cm-1"},
						},
					},
				},
				{
					Name: "secret",
					VolumeSource: v1.VolumeSource{
						Secret: &v1.SecretVolumeSource{SecretName: "sec-1"},
					},
				},
				{
					Name: "token",
					VolumeSource: v1.VolumeSource{
						Projected: &v1.ProjectedVolumeSource{
							Sources: []v1.VolumeProjection{
								{
									ServiceAccountToken: &v1.ServiceAccountTokenProjection{
										Path: "token",
									},
								},
								{
									DownwardAPI: &v1.DownwardAPIProjection{
										Items: []v1.DownwardAPIVolumeFile{
											{
												Path: "namespace",
												FieldRef: &v1.ObjectFieldSelector{
													FieldPath: "metadata.namespace",
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
		},
		Status: v1.PodStatus{
			Phase:             v1.PodRunning,
			NominatedNodeName: "kind-worker1",
			HostIP:            "192.168.8.8",
			HostIPs:           []v1.HostIP{{IP: "192.168.8.8"}},
			PodIP:             "10.244.7.83",
			PodIPs:            []v1.PodIP{{IP: "10.244.7.83"}},
			AllocatedResources: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("10m"),
				v1.ResourceMemory: resource.MustParse("40M"),
			},
		},
	}
	corev1.SetObjectDefaults_Pod(p)
	return p
}

func encodeV1PodProtobuf(t *testing.T, v1Pod *v1.Pod) []byte {
	t.Helper()
	corePod := &core.Pod{}
	if err := corev1.Convert_v1_Pod_To_core_Pod(v1Pod, corePod, nil); err != nil {
		t.Fatalf("convert v1->core failed: %v", err)
	}
	_, respEnc, _ := newTestProtobufCodecs(t)
	var buf bytes.Buffer
	if err := respEnc.Encode(corePod, &buf); err != nil {
		t.Fatalf("encode protobuf failed: %v", err)
	}
	return buf.Bytes()
}

func TestV1ProtobufEncoderIDMatchesCodecFactory(t *testing.T) {
	_, respEnc, _ := newTestProtobufCodecs(t)
	if respEnc.Identifier() != v1ProtobufEncoderID {
		t.Fatalf("encoder identifier mismatch:\ngot:  %s\nwant: %s", respEnc.Identifier(), v1ProtobufEncoderID)
	}
}

func TestStorageCodecAndWireParity(t *testing.T) {
	storageCodec, respEnc, jsonEnc := newTestProtobufCodecs(t)
	v1Pod := makeDefaultedV1Pod()
	rawNoRV := encodeV1PodProtobuf(t, v1Pod)

	obj, _, err := storageCodec.Decode(rawNoRV, nil, nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	lp, ok := obj.(*Pod)
	if !ok {
		t.Fatalf("expected *lazypod.Pod, got %T", obj)
	}
	if !lp.IsWireBacked() {
		t.Fatalf("expected defaulted pod to stay wire-backed")
	}

	// Test RV insertion/updates including empty -> short -> long -> empty.
	for _, rv := range []string{"7", "12345678901234567", "42", ""} {
		lp.SetResourceVersion(rv)
		if got := lp.GetResourceVersion(); got != rv {
			t.Fatalf("GetResourceVersion() = %q, want %q", got, rv)
		}
		if !lp.IsWireBacked() {
			t.Fatalf("SetResourceVersion(%q) unexpectedly materialized fallback", rv)
		}

		wantV1 := v1Pod.DeepCopy()
		wantV1.ResourceVersion = rv
		wantBytes := encodeV1PodProtobuf(t, wantV1)

		var gotBuf bytes.Buffer
		if err := respEnc.Encode(lp, &gotBuf); err != nil {
			t.Fatalf("Encode(rv=%q) failed: %v", rv, err)
		}
		if !bytes.Equal(gotBuf.Bytes(), wantBytes) {
			t.Fatalf("protobuf bytes mismatch for rv=%q (got %d bytes, want %d bytes)", rv, gotBuf.Len(), len(wantBytes))
		}

		// Verify Splice fast path (used by watchEncoder).
		spliceBuf := runtime.NewSpliceBuffer()
		if err := respEnc.Encode(lp, spliceBuf); err != nil {
			t.Fatalf("Encode splice (rv=%q) failed: %v", rv, err)
		}
		if !bytes.Equal(spliceBuf.Bytes(), wantBytes) {
			t.Fatalf("splice bytes mismatch for rv=%q", rv)
		}

		// Verify watch outer identifier delegates to inner encoder without decoding.
		watchCalled := false
		err := lp.CacheEncode(runtime.Identifier(`{"name":"watch","embeddedEncoder":"x"}`), func(inner runtime.Object, w io.Writer) error {
			watchCalled = true
			if _, ok := inner.(*Pod); !ok {
				t.Errorf("watch outer pass materialized %T instead of keeping *Pod", inner)
			}
			return respEnc.Encode(inner, w)
		}, spliceBuf)
		if err != nil || !watchCalled {
			t.Fatalf("watch outer CacheEncode failed: err=%v called=%v", err, watchCalled)
		}

		// Verify JSON fallback matches *core.Pod.
		wantCore := &core.Pod{}
		if err := corev1.Convert_v1_Pod_To_core_Pod(wantV1, wantCore, nil); err != nil {
			t.Fatalf("convert failed: %v", err)
		}
		var wantJSON, gotJSON bytes.Buffer
		if err := jsonEnc.Encode(wantCore, &wantJSON); err != nil {
			t.Fatalf("json encode core failed: %v", err)
		}
		if err := jsonEnc.Encode(lp, &gotJSON); err != nil {
			t.Fatalf("json encode lazy failed: %v", err)
		}
		if !bytes.Equal(gotJSON.Bytes(), wantJSON.Bytes()) {
			t.Fatalf("json mismatch for rv=%q:\ngot:  %s\nwant: %s", rv, gotJSON.String(), wantJSON.String())
		}

		// Verify GetObject() semantic equality.
		decodedCore := lp.GetObject().(*core.Pod)
		if !apiequality.Semantic.DeepEqual(decodedCore, wantCore) {
			t.Fatalf("GetObject() semantic mismatch for rv=%q", rv)
		}
	}
}

func TestExemplarPodWireBacked(t *testing.T) {
	exemplarPath := filepath.Join("..", "..", "..", "..", "test", "integration", "apiserver", "flowcontrol", "cost", "testdata", "exemplar_pod.yaml")
	rawYAML, err := os.ReadFile(exemplarPath)
	if err != nil {
		t.Skipf("exemplar_pod.yaml not found at %s: %v", exemplarPath, err)
	}
	var exemplar v1.Pod
	if err := yaml.Unmarshal(rawYAML, &exemplar); err != nil {
		t.Fatalf("yaml unmarshal exemplar failed: %v", err)
	}
	corev1.SetObjectDefaults_Pod(&exemplar)
	exemplar.ResourceVersion = ""

	storageCodec, respEnc, _ := newTestProtobufCodecs(t)
	rawBytes := encodeV1PodProtobuf(t, &exemplar)

	obj, _, err := storageCodec.Decode(rawBytes, nil, nil)
	if err != nil {
		t.Fatalf("Decode exemplar failed: %v", err)
	}
	lp := obj.(*Pod)
	if !lp.IsWireBacked() {
		t.Fatalf("expected BenchmarkRequestCost exemplar_pod.yaml to be wire-backed on fast path")
	}

	lp.SetResourceVersion("1001")
	wantV1 := exemplar.DeepCopy()
	wantV1.ResourceVersion = "1001"
	wantBytes := encodeV1PodProtobuf(t, wantV1)

	var gotBuf bytes.Buffer
	if err := respEnc.Encode(lp, &gotBuf); err != nil {
		t.Fatalf("Encode exemplar failed: %v", err)
	}
	if !bytes.Equal(gotBuf.Bytes(), wantBytes) {
		t.Fatalf("exemplar protobuf bytes mismatch: got %d bytes, want %d bytes", gotBuf.Len(), len(wantBytes))
	}
}

func TestPodListStreamingParity(t *testing.T) {
	storageCodec, respEnc, _ := newTestProtobufCodecs(t)
	rem := int64(42)

	for _, count := range []int{0, 1, 5} {
		lazyList := &PodList{
			ListMeta: metav1.ListMeta{
				ResourceVersion:    "999",
				Continue:           "eyJ2IjoibWV0YS5rOHMuaW8vdjEifQ",
				RemainingItemCount: &rem,
			},
			Items: make([]Pod, count),
		}
		coreList := &core.PodList{
			ListMeta: *lazyList.ListMeta.DeepCopy(),
			Items:    make([]core.Pod, count),
		}
		for i := range count {
			p := makeDefaultedV1Pod()
			p.Name = "pod-" + string(rune('a'+i))
			p.ResourceVersion = ""
			raw := encodeV1PodProtobuf(t, p)
			obj, _, err := storageCodec.Decode(raw, nil, nil)
			if err != nil {
				t.Fatalf("Decode item %d failed: %v", i, err)
			}
			lp := obj.(*Pod)
			lp.SetResourceVersion(string(rune('1' + i)))
			lazyList.Items[i] = *lp

			wantV1 := p.DeepCopy()
			wantV1.ResourceVersion = string(rune('1' + i))
			if err := corev1.Convert_v1_Pod_To_core_Pod(wantV1, &coreList.Items[i], nil); err != nil {
				t.Fatalf("convert failed: %v", err)
			}
		}

		var wantBuf, gotBuf bytes.Buffer
		if err := respEnc.Encode(coreList, &wantBuf); err != nil {
			t.Fatalf("encode coreList (n=%d) failed: %v", count, err)
		}
		if err := respEnc.Encode(lazyList, &gotBuf); err != nil {
			t.Fatalf("encode lazyList (n=%d) failed: %v", count, err)
		}
		if !bytes.Equal(gotBuf.Bytes(), wantBuf.Bytes()) {
			t.Fatalf("PodList protobuf bytes mismatch (n=%d): got %d bytes, want %d bytes", count, gotBuf.Len(), wantBuf.Len())
		}
	}

	// Mixed list with one fallback item must still encode identically.
	v1Pod := makeDefaultedV1Pod()
	v1Pod.ResourceVersion = "12"
	corePod := &core.Pod{}
	if err := corev1.Convert_v1_Pod_To_core_Pod(v1Pod, corePod, nil); err != nil {
		t.Fatalf("convert failed: %v", err)
	}
	mixed := &PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "12"},
		Items:    []Pod{*NewFromCorePod(corePod)},
	}
	wantList := &core.PodList{
		ListMeta: metav1.ListMeta{ResourceVersion: "12"},
		Items:    []core.Pod{*corePod},
	}
	var wantBuf, gotBuf bytes.Buffer
	if err := respEnc.Encode(wantList, &wantBuf); err != nil {
		t.Fatalf("encode wantList failed: %v", err)
	}
	if err := respEnc.Encode(mixed, &gotBuf); err != nil {
		t.Fatalf("encode mixed failed: %v", err)
	}
	if !bytes.Equal(gotBuf.Bytes(), wantBuf.Bytes()) {
		t.Fatalf("mixed PodList protobuf mismatch")
	}
}

func TestPodAccessorAndMutationSafety(t *testing.T) {
	storageCodec, _, _ := newTestProtobufCodecs(t)
	v1Pod := makeDefaultedV1Pod()
	v1Pod.DeletionGracePeriodSeconds = ptr.To[int64](30)
	raw := encodeV1PodProtobuf(t, v1Pod)

	obj, _, err := storageCodec.Decode(raw, nil, nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	lp := obj.(*Pod)
	lp.SetResourceVersion("55")

	wantCore := &core.Pod{}
	wantV1 := v1Pod.DeepCopy()
	wantV1.ResourceVersion = "55"
	if err := corev1.Convert_v1_Pod_To_core_Pod(wantV1, wantCore, nil); err != nil {
		t.Fatalf("convert failed: %v", err)
	}
	wantAcc := podutil.MustAccessor(wantCore)
	gotAcc, err := podutil.Accessor(lp)
	if err != nil {
		t.Fatalf("podutil.Accessor failed: %v", err)
	}

	if gotAcc.GetName() != wantAcc.GetName() ||
		gotAcc.GetNamespace() != wantAcc.GetNamespace() ||
		gotAcc.GetGenerateName() != wantAcc.GetGenerateName() ||
		gotAcc.GetUID() != wantAcc.GetUID() ||
		gotAcc.GetResourceVersion() != wantAcc.GetResourceVersion() ||
		gotAcc.GetGeneration() != wantAcc.GetGeneration() ||
		gotAcc.GetSpecNodeName() != wantAcc.GetSpecNodeName() ||
		gotAcc.GetSpecRestartPolicy() != wantAcc.GetSpecRestartPolicy() ||
		gotAcc.GetSpecSchedulerName() != wantAcc.GetSpecSchedulerName() ||
		gotAcc.GetSpecServiceAccountName() != wantAcc.GetSpecServiceAccountName() ||
		gotAcc.GetSpecHostNetwork() != wantAcc.GetSpecHostNetwork() ||
		gotAcc.GetStatusPhase() != wantAcc.GetStatusPhase() ||
		gotAcc.GetStatusNominatedNodeName() != wantAcc.GetStatusNominatedNodeName() {
		t.Fatalf("scalar getter mismatch between lazy PodAccessor and *core.Pod accessor")
	}

	tgps, ok := gotAcc.GetSpecTerminationGracePeriodSeconds()
	wantTGPS, wantOK := wantAcc.GetSpecTerminationGracePeriodSeconds()
	if tgps != wantTGPS || ok != wantOK {
		t.Fatalf("TerminationGracePeriodSeconds = (%d, %v), want (%d, %v)", tgps, ok, wantTGPS, wantOK)
	}

	// Mutating returned maps/slices must not affect underlying pod.
	labels := gotAcc.GetLabels()
	labels["app"] = "mutated"
	if got := gotAcc.GetLabels()["app"]; got != "payments" {
		t.Fatalf("GetLabels mutation leaked: got %q", got)
	}

	podIPs := gotAcc.GetStatusPodIPs()
	podIPs[0].IP = "1.2.3.4"
	if got := gotAcc.GetStatusPodIPs()[0].IP; got != "10.244.7.83" {
		t.Fatalf("GetStatusPodIPs mutation leaked: got %q", got)
	}

	hostIPs := gotAcc.GetStatusHostIPs()
	hostIPs[0].IP = "9.9.9.9"
	if got := gotAcc.GetStatusHostIPs()[0].IP; got != "192.168.8.8" {
		t.Fatalf("GetStatusHostIPs mutation leaked: got %q", got)
	}

	// Shallow struct copy isolation (simulating Cacher.Get / GetList).
	shallowCopy := *lp
	shallowCopy.SetResourceVersion("999")
	if lp.GetResourceVersion() != "55" {
		t.Fatalf("shallow copy SetResourceVersion mutated original: got %q", lp.GetResourceVersion())
	}

	shallowCopy2 := *lp
	shallowCopy2.SetManagedFields(nil)
	if len(lp.GetManagedFields()) == 0 {
		t.Fatalf("shallow copy SetManagedFields(nil) mutated original")
	}
	if len(shallowCopy2.GetManagedFields()) != 0 {
		t.Fatalf("expected shallowCopy2 managedFields cleared")
	}
}

func TestDefaultingParityAndFallback(t *testing.T) {
	storageCodec, _, _ := newTestProtobufCodecs(t)

	cases := []struct {
		name   string
		mutate func(*v1.Pod)
	}{
		{name: "missing restartPolicy", mutate: func(p *v1.Pod) { p.Spec.RestartPolicy = "" }},
		{name: "missing dnsPolicy", mutate: func(p *v1.Pod) { p.Spec.DNSPolicy = "" }},
		{name: "missing schedulerName", mutate: func(p *v1.Pod) { p.Spec.SchedulerName = "" }},
		{name: "missing terminationGracePeriodSeconds", mutate: func(p *v1.Pod) { p.Spec.TerminationGracePeriodSeconds = nil }},
		{name: "negative terminationGracePeriodSeconds", mutate: func(p *v1.Pod) { p.Spec.TerminationGracePeriodSeconds = ptr.To[int64](-5) }},
		{name: "missing securityContext", mutate: func(p *v1.Pod) { p.Spec.SecurityContext = nil }},
		{name: "missing enableServiceLinks", mutate: func(p *v1.Pod) { p.Spec.EnableServiceLinks = nil }},
		{name: "mismatched deprecated serviceAccount", mutate: func(p *v1.Pod) { p.Spec.DeprecatedServiceAccount = "" }},
		{name: "hostNetwork with ports", mutate: func(p *v1.Pod) { p.Spec.HostNetwork = true }},
		{name: "container missing pullPolicy", mutate: func(p *v1.Pod) { p.Spec.Containers[0].ImagePullPolicy = "" }},
		{name: "container missing port protocol", mutate: func(p *v1.Pod) { p.Spec.Containers[0].Ports[0].Protocol = "" }},
		{name: "container limits without request", mutate: func(p *v1.Pod) {
			p.Spec.Containers[0].Resources.Requests = nil
		}},
		{name: "sub-milli quantity", mutate: func(p *v1.Pod) {
			p.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse("100n")
		}},
		{name: "probe present", mutate: func(p *v1.Pod) {
			p.Spec.Containers[0].LivenessProbe = &v1.Probe{
				ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{Path: "/healthz"}},
			}
		}},
		{name: "empty volume source", mutate: func(p *v1.Pod) {
			p.Spec.Volumes = append(p.Spec.Volumes, v1.Volume{Name: "empty"})
		}},
		{name: "secret missing defaultMode", mutate: func(p *v1.Pod) {
			p.Spec.Volumes[1].Secret.DefaultMode = nil
		}},
		{name: "projected token missing expirationSeconds", mutate: func(p *v1.Pod) {
			p.Spec.Volumes[2].Projected.Sources[0].ServiceAccountToken.ExpirationSeconds = nil
		}},
		{name: "podIP without podIPs", mutate: func(p *v1.Pod) { p.Status.PodIPs = nil }},
		{name: "podIPs without podIP", mutate: func(p *v1.Pod) { p.Status.PodIP = "" }},
		{name: "podIP mismatch with podIPs", mutate: func(p *v1.Pod) { p.Status.PodIP = "10.0.0.2" }},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := makeDefaultedV1Pod()
			tc.mutate(p)
			raw := encodeV1PodProtobuf(t, p)

			obj, _, err := storageCodec.Decode(raw, nil, nil)
			if err != nil {
				t.Fatalf("Decode failed: %v", err)
			}
			lp, ok := obj.(*Pod)
			if !ok {
				t.Fatalf("expected *Pod, got %T", obj)
			}
			if lp.IsWireBacked() {
				t.Fatalf("expected %q to fall back to corev1.SetObjectDefaults_Pod, not stay wire-backed", tc.name)
			}

			wantV1 := p.DeepCopy()
			corev1.SetObjectDefaults_Pod(wantV1)
			wantCore := &core.Pod{}
			if err := corev1.Convert_v1_Pod_To_core_Pod(wantV1, wantCore, nil); err != nil {
				t.Fatalf("convert failed: %v", err)
			}
			gotCore := lp.GetObject().(*core.Pod)
			if !apiequality.Semantic.DeepEqual(gotCore, wantCore) {
				t.Fatalf("defaulted fallback mismatch for %q", tc.name)
			}
		})
	}

	// Corrupt bytes must fail with decode error instead of silently accepting.
	truncated := encodeV1PodProtobuf(t, makeDefaultedV1Pod())
	truncated = truncated[:len(truncated)/2]
	if _, _, err := storageCodec.Decode(truncated, nil, nil); err == nil {
		t.Fatalf("expected error on truncated protobuf bytes")
	}

	// JSON-encoded pod in storage must fall back cleanly.
	_, _, jsonEnc := newTestProtobufCodecs(t)
	corePod := &core.Pod{}
	if err := corev1.Convert_v1_Pod_To_core_Pod(makeDefaultedV1Pod(), corePod, nil); err != nil {
		t.Fatalf("convert failed: %v", err)
	}
	var jsonBuf bytes.Buffer
	if err := jsonEnc.Encode(corePod, &jsonBuf); err != nil {
		t.Fatalf("json encode failed: %v", err)
	}
	obj, _, err := storageCodec.Decode(jsonBuf.Bytes(), nil, nil)
	if err != nil {
		t.Fatalf("json Decode failed: %v", err)
	}
	if _, ok := obj.(*Pod); !ok {
		t.Fatalf("expected *Pod wrapper for JSON fallback, got %T", obj)
	}
}

func TestConcurrentShallowCopyRace(t *testing.T) {
	storageCodec, respEnc, jsonEnc := newTestProtobufCodecs(t)
	raw := encodeV1PodProtobuf(t, makeDefaultedV1Pod())
	obj, _, err := storageCodec.Decode(raw, nil, nil)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	canonical := obj.(*Pod)
	canonical.SetResourceVersion("100")

	var wg sync.WaitGroup
	for g := range 8 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := range 50 {
				cp := *canonical
				var buf bytes.Buffer
				if err := respEnc.Encode(&cp, &buf); err != nil {
					t.Errorf("proto encode failed: %v", err)
					return
				}
				if i%10 == 0 {
					cp.SetResourceVersion(string(rune('1' + (id+i)%9)))
					buf.Reset()
					if err := respEnc.Encode(&cp, &buf); err != nil {
						t.Errorf("proto encode after RV splice failed: %v", err)
						return
					}
				}
				if i%17 == 0 {
					cp.SetManagedFields(nil)
					buf.Reset()
					if err := jsonEnc.Encode(&cp, &buf); err != nil {
						t.Errorf("json encode after SetManagedFields failed: %v", err)
						return
					}
				}
				_ = canonical.GetSpecNodeName()
				_ = canonical.GetLabels()
				_ = canonical.GetStatusPodIPs()
				_ = canonical.DeepCopyObject()
			}
		}(g)
	}
	wg.Wait()

	if canonical.GetResourceVersion() != "100" {
		t.Fatalf("canonical RV corrupted: got %q", canonical.GetResourceVersion())
	}
}
