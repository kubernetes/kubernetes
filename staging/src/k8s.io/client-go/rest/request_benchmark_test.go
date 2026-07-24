/*
Copyright 2026 The Kubernetes Authors.

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

package rest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	k8swatch "k8s.io/apimachinery/pkg/watch"
	restclientwatch "k8s.io/client-go/rest/watch"
)

var kueueGV = schema.GroupVersion{Group: "kueue.x-k8s.io", Version: "v1beta1"}

var workloadBenchmarkScheme = func() *runtime.Scheme {
	s := runtime.NewScheme()
	s.AddKnownTypes(kueueGV, &Workload{}, &WorkloadList{})
	metav1.AddToGroupVersion(s, kueueGV)
	return s
}()

var workloadBenchmarkCodecs = serializer.NewCodecFactory(workloadBenchmarkScheme, serializer.WithSerializer(cbor.NewSerializerInfo))

var workloadBenchmarkSizes = []int{
	10 << 10,
	15 << 10,
	20 << 10,
	30 << 10,
	40 << 10,
	50 << 10,
	100 << 10,
	200 << 10,
	300 << 10,
	400 << 10,
	500 << 10,
	600 << 10,
	700 << 10,
	800 << 10,
	900 << 10,
	1000 << 10,
	1100 << 10,
	1200 << 10,
	1300 << 10,
	1400 << 10,
	1500 << 10,
}

func BenchmarkWorkloadJSONRESTClientGet(b *testing.B) {
	benchmarkWorkloadRESTClientGet(b, k8sruntime.ContentTypeJSON)
}

func BenchmarkWorkloadCBORRESTClientGet(b *testing.B) {
	benchmarkWorkloadRESTClientGet(b, k8sruntime.ContentTypeCBOR)
}

func BenchmarkWorkloadJSONEncode(b *testing.B) {
	benchmarkWorkloadEncode(b, k8sruntime.ContentTypeJSON)
}

func BenchmarkWorkloadCBOREncode(b *testing.B) {
	benchmarkWorkloadEncode(b, k8sruntime.ContentTypeCBOR)
}

func benchmarkWorkloadEncode(b *testing.B, contentType string) {
	for _, targetSize := range workloadBenchmarkSizes {
		b.Run(fmt.Sprintf("workload_%s_size=%dKB", benchmarkContentTypeName(contentType), targetSize>>10), func(b *testing.B) {
			workload := makeWorkloadForResponseBody(b, targetSize, contentType)
			encoded := encodeWorkloadForBenchmark(b, workload, contentType)

			b.ReportAllocs()
			b.SetBytes(int64(len(encoded)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchmarkBytes = encodeWorkloadForBenchmark(b, workload, contentType)
			}
			b.ReportMetric(float64(len(encoded))/1024, benchmarkContentTypeName(contentType)+"_KB")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "workloads/s")
		})
	}
}

func benchmarkWorkloadRESTClientGet(b *testing.B, contentType string) {
	ctx := context.Background()
	baseURL := &url.URL{Scheme: "https", Host: "benchmark.local"}
	contentConfig := ClientContentConfig{
		ContentType:  contentType,
		GroupVersion: kueueGV,
		Negotiator:   k8sruntime.NewClientNegotiator(workloadBenchmarkCodecs.WithoutConversion(), kueueGV),
	}

	for _, targetSize := range workloadBenchmarkSizes {
		b.Run(fmt.Sprintf("workload_%s_size=%dKB", benchmarkContentTypeName(contentType), targetSize>>10), func(b *testing.B) {
			responseBody := makeWorkloadResponseBody(b, targetSize, contentType)
			httpClient := &http.Client{
				Transport: workloadBenchmarkTransport(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode:    http.StatusOK,
						Header:        http.Header{"Content-Type": []string{contentType}},
						Body:          io.NopCloser(bytes.NewReader(responseBody)),
						ContentLength: int64(len(responseBody)),
					}, nil
				}),
			}
			request := NewRequestWithClient(baseURL, "", contentConfig, httpClient).
				Verb("GET").
				AbsPath("/apis/kueue.x-k8s.io/v1beta1/namespaces/default/workloads/kueue-benchmark")

			b.ReportAllocs()
			b.SetBytes(int64(len(responseBody)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out := &Workload{}
				if err := request.Do(ctx).Into(out); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(len(responseBody))/1024, benchmarkContentTypeName(contentType)+"_KB")
			b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "workloads/s")
		})
	}
}

func BenchmarkWorkloadJSONRESTClientWatch(b *testing.B) {
	benchmarkWorkloadRESTClientWatch(b, k8sruntime.ContentTypeJSON)
}

func BenchmarkWorkloadCBORRESTClientWatch(b *testing.B) {
	benchmarkWorkloadRESTClientWatch(b, k8sruntime.ContentTypeCBOR)
}

func BenchmarkWorkloadJSONWatchStreamEncode(b *testing.B) {
	benchmarkWorkloadWatchStreamEncode(b, k8sruntime.ContentTypeJSON)
}

func BenchmarkWorkloadCBORWatchStreamEncode(b *testing.B) {
	benchmarkWorkloadWatchStreamEncode(b, k8sruntime.ContentTypeCBOR)
}

func BenchmarkWorkloadJSONWatchStreamFrameRead(b *testing.B) {
	const eventsPerWatch = 100
	for _, targetSize := range workloadBenchmarkSizes {
		b.Run(fmt.Sprintf("workload_json_size=%dKB/events=%d", targetSize>>10, eventsPerWatch), func(b *testing.B) {
			workload := makeWorkloadForResponseBody(b, targetSize, k8sruntime.ContentTypeJSON)
			streamBody := makeWorkloadWatchStreamBody(b, workload, eventsPerWatch, k8sruntime.ContentTypeJSON)
			readBuffer := make([]byte, len(streamBody)/eventsPerWatch+4096)
			info := serializerInfoForBenchmark(b, k8sruntime.ContentTypeJSON)

			b.ReportAllocs()
			b.SetBytes(int64(len(streamBody)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				reader := info.StreamSerializer.Framer.NewFrameReader(io.NopCloser(bytes.NewReader(streamBody)))
				seen := 0
				for seen < eventsPerWatch {
					n, err := reader.Read(readBuffer)
					if err == io.ErrShortBuffer {
						continue
					}
					if err != nil {
						b.Fatalf("failed reading JSON watch frame %d/%d: %v", seen+1, eventsPerWatch, err)
					}
					benchmarkBytes = readBuffer[:n]
					seen++
				}
				if err := reader.Close(); err != nil {
					b.Fatalf("failed closing JSON watch frame reader: %v", err)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*eventsPerWatch), "ns/event")
			b.ReportMetric(float64(len(streamBody))/1024/float64(eventsPerWatch), "json_KB/event")
			b.ReportMetric(float64(b.N*eventsPerWatch)/b.Elapsed().Seconds(), "workloads/s")
		})
	}
}

func benchmarkWorkloadWatchStreamEncode(b *testing.B, contentType string) {
	const eventsPerWatch = 100
	for _, targetSize := range workloadBenchmarkSizes {
		b.Run(fmt.Sprintf("workload_%s_size=%dKB/events=%d", benchmarkContentTypeName(contentType), targetSize>>10, eventsPerWatch), func(b *testing.B) {
			workload := makeWorkloadForResponseBody(b, targetSize, contentType)
			streamBody := makeWorkloadWatchStreamBody(b, workload, eventsPerWatch, contentType)

			b.ReportAllocs()
			b.SetBytes(int64(len(streamBody)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchmarkBytes = makeWorkloadWatchStreamBody(b, workload, eventsPerWatch, contentType)
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*eventsPerWatch), "ns/event")
			b.ReportMetric(float64(len(streamBody))/1024/float64(eventsPerWatch), benchmarkContentTypeName(contentType)+"_KB/event")
			b.ReportMetric(float64(b.N*eventsPerWatch)/b.Elapsed().Seconds(), "workloads/s")
		})
	}
}

func benchmarkWorkloadRESTClientWatch(b *testing.B, contentType string) {
	ctx := context.Background()
	baseURL := &url.URL{Scheme: "https", Host: "benchmark.local"}
	contentConfig := ClientContentConfig{
		ContentType:  contentType,
		GroupVersion: kueueGV,
		Negotiator:   k8sruntime.NewClientNegotiator(workloadBenchmarkCodecs.WithoutConversion(), kueueGV),
	}
	const eventsPerWatch = 100

	// Toggle the ConcurrentWatchObjectDecode feature gate by setting
	// KUBE_FEATURE_ConcurrentWatchObjectDecode=true in the env. Run this
	// benchmark twice (once with, once without) and feed the two outputs
	// through benchstat to compare.
	for _, targetSize := range workloadBenchmarkSizes {
		b.Run(fmt.Sprintf("workload_%s_size=%dKB/events=%d", benchmarkContentTypeName(contentType), targetSize>>10, eventsPerWatch), func(b *testing.B) {
			workload := makeWorkloadForResponseBody(b, targetSize, contentType)
			streamBody := makeWorkloadWatchStreamBody(b, workload, eventsPerWatch, contentType)
			httpClient := &http.Client{
				Transport: workloadBenchmarkTransport(func(req *http.Request) (*http.Response, error) {
					return &http.Response{
						StatusCode:    http.StatusOK,
						Header:        http.Header{"Content-Type": []string{contentType + ";stream=watch"}},
						Body:          io.NopCloser(bytes.NewReader(streamBody)),
						ContentLength: int64(len(streamBody)),
					}, nil
				}),
			}
			request := NewRequestWithClient(baseURL, "", contentConfig, httpClient).
				Verb("GET").
				AbsPath("/apis/kueue.x-k8s.io/v1beta1/namespaces/default/workloads").
				Param("watch", "true")

			b.ReportAllocs()
			b.SetBytes(int64(len(streamBody)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				watcher, err := request.Watch(ctx)
				if err != nil {
					b.Fatal(err)
				}
				seen := 0
				for event := range watcher.ResultChan() {
					if event.Type == k8swatch.Error {
						b.Fatalf("watch returned error event: %#v", event.Object)
					}
					benchmarkWorkloadObject = event.Object
					seen++
					if seen == eventsPerWatch {
						break
					}
				}
				watcher.Stop()
				if seen != eventsPerWatch {
					b.Fatalf("decoded %d watch events, expected %d", seen, eventsPerWatch)
				}
			}
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*eventsPerWatch), "ns/event")
			b.ReportMetric(float64(len(streamBody))/1024/float64(eventsPerWatch), benchmarkContentTypeName(contentType)+"_KB/event")
			b.ReportMetric(float64(b.N*eventsPerWatch)/b.Elapsed().Seconds(), "workloads/s")
		})
	}
}

type workloadBenchmarkTransport func(*http.Request) (*http.Response, error)

func (rt workloadBenchmarkTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return rt(req)
}

func makeWorkloadResponseBody(b testing.TB, targetBytes int, contentType string) []byte {
	b.Helper()
	return encodeWorkloadForBenchmark(b, makeWorkloadForResponseBody(b, targetBytes, contentType), contentType)
}

func makeWorkloadForResponseBody(b testing.TB, targetBytes int, contentType string) *Workload {
	b.Helper()
	low, high := 0, targetBytes
	bestWorkload := newBenchmarkWorkload(high)
	bestSize := len(encodeWorkloadForBenchmark(b, bestWorkload, contentType))
	for bestSize < targetBytes {
		low = high + 1
		high *= 2
		bestWorkload = newBenchmarkWorkload(high)
		bestSize = len(encodeWorkloadForBenchmark(b, bestWorkload, contentType))
	}
	for low <= high {
		mid := low + (high-low)/2
		workload := newBenchmarkWorkload(mid)
		encodedSize := len(encodeWorkloadForBenchmark(b, workload, contentType))
		if encodedSize >= targetBytes {
			bestWorkload = workload
			high = mid - 1
			continue
		}
		low = mid + 1
	}
	return bestWorkload
}

// newBenchmarkWorkload builds a Kueue Workload-shaped object whose embedded
// PodTemplateSpec is sized to drive total encoded bytes near payloadBytes.
// The Pod-shaped helpers below generate the bulk of the payload via
// containers/volumes/env padding, matching the size profile of real Workload
// traffic in apps that wrap large training Pods.
func newBenchmarkWorkload(payloadBytes int) *Workload {
	pod := newBenchmarkPod(payloadBytes)
	template := v1.PodTemplateSpec{
		ObjectMeta: pod.ObjectMeta,
		Spec:       pod.Spec,
	}
	minCount := int32(1)
	priority := int32(1000)
	active := true
	flavors := map[v1.ResourceName]ResourceFlavorReference{
		v1.ResourceCPU:                       "default",
		v1.ResourceMemory:                    "default",
		v1.ResourceName("nvidia.com/gpu"):    "h100-80g",
		v1.ResourceName("ephemeral-storage"): "fast-ssd",
	}
	now := metav1.Now()
	return &Workload{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Workload",
			APIVersion: kueueGV.String(),
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kueue-benchmark",
			Namespace: "default",
			UID:       "00000000-0000-0000-0000-000000000001",
			Labels: map[string]string{
				"kueue.x-k8s.io/queue-name":     "benchmark-queue",
				"kueue.x-k8s.io/priority-class": "high",
				"kueue.x-k8s.io/job-uid":        "abcdef-0123",
				"app":                           "kueue-benchmark",
			},
			Annotations: map[string]string{
				"kueue.x-k8s.io/owner-name": "training-job",
				"kueue.x-k8s.io/owner-uid":  "00000000-0000-0000-0000-000000000010",
			},
		},
		Spec: WorkloadSpec{
			QueueName:         LocalQueueName("benchmark-queue"),
			PriorityClassName: "high-priority",
			Priority:          &priority,
			Active:            &active,
			PodSets: []PodSet{
				{
					Name:     PodSetReference("main"),
					Count:    1,
					MinCount: &minCount,
					Template: template,
				},
			},
		},
		Status: WorkloadStatus{
			Admission: &Admission{
				ClusterQueue: ClusterQueueReference("benchmark-cluster-queue"),
				PodSetAssignments: []PodSetAssignment{
					{
						Name:    PodSetReference("main"),
						Flavors: flavors,
						Count:   &minCount,
					},
				},
			},
			Conditions: []metav1.Condition{
				{
					Type:               "Admitted",
					Status:             metav1.ConditionTrue,
					Reason:             "QuotaReserved",
					Message:            "Workload admitted by ClusterQueue benchmark-cluster-queue",
					LastTransitionTime: now,
					ObservedGeneration: 1,
				},
				{
					Type:               "PodsReady",
					Status:             metav1.ConditionTrue,
					Reason:             "PodsReady",
					Message:            "All pods are ready",
					LastTransitionTime: now,
					ObservedGeneration: 1,
				},
			},
		},
	}
}

// newBenchmarkPod generates the Pod-shaped data that fills the embedded
// PodTemplateSpec inside the Workload. payloadBytes is a knob that drives
// the number of containers/volumes/env vars and the length of padded
// values; it does not directly equal the encoded byte count.
func newBenchmarkPod(payloadBytes int) *v1.Pod {
	scaleKB := maxInt(1, payloadBytes/1024)
	containerCount := clampInt(1+scaleKB/25, 1, 12)
	initContainerCount := clampInt(scaleKB/64, 0, 4)
	envCount := clampInt(6+scaleKB/3, 8, 96)
	volumeCount := clampInt(2+scaleKB/5, 4, 48)
	itemsPerVolume := clampInt(2+scaleKB/16, 2, 16)
	volumeMountCount := minInt(volumeCount, 16)
	envValueLength := maxInt(32, 16+payloadBytes/maxInt(1, containerCount*envCount*2))
	paddingBytes := minInt(payloadBytes/20, 2048)

	containers := make([]v1.Container, 0, containerCount)
	containerStatuses := make([]v1.ContainerStatus, 0, containerCount)
	for i := 0; i < containerCount; i++ {
		name := fmt.Sprintf("app-%02d", i)
		containers = append(containers, v1.Container{
			Name:         name,
			Image:        fmt.Sprintf("registry.k8s.io/pause:3.%d", 10+i%3),
			Command:      []string{"/pause"},
			Args:         []string{fmt.Sprintf("--shard=%d", i), "--config=/etc/app/config.yaml"},
			Ports:        benchmarkContainerPorts(i),
			Env:          benchmarkEnvVars(i, envCount, envValueLength),
			VolumeMounts: benchmarkVolumeMounts(volumeMountCount),
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{},
				Limits:   v1.ResourceList{},
			},
		})
		containerStatuses = append(containerStatuses, v1.ContainerStatus{
			Name:         name,
			Ready:        true,
			RestartCount: int32(i % 3),
			Image:        fmt.Sprintf("registry.k8s.io/pause:3.%d", 10+i%3),
			ImageID:      fmt.Sprintf("registry.k8s.io/pause@sha256:%064d", i+1),
			ContainerID:  fmt.Sprintf("containerd://%064d", i+1),
			State: v1.ContainerState{
				Running: &v1.ContainerStateRunning{},
			},
		})
	}

	initContainers := make([]v1.Container, 0, initContainerCount)
	for i := 0; i < initContainerCount; i++ {
		initContainers = append(initContainers, v1.Container{
			Name:         fmt.Sprintf("init-%02d", i),
			Image:        "registry.k8s.io/pause:3.10",
			Command:      []string{"/bin/sh", "-c"},
			Args:         []string{"prepare config and warm local cache"},
			Env:          benchmarkEnvVars(100+i, maxInt(4, envCount/4), maxInt(32, envValueLength/2)),
			VolumeMounts: benchmarkVolumeMounts(minInt(volumeMountCount, 4)),
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{},
				Limits:   v1.ResourceList{},
			},
		})
	}

	podIPs := []v1.PodIP{{IP: "10.244.0.10"}}
	for i := 1; i < minInt(containerCount, 8); i++ {
		podIPs = append(podIPs, v1.PodIP{IP: fmt.Sprintf("10.244.%d.%d", i/255, i%255+10)})
	}

	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kueue-benchmark",
			Namespace: "default",
			Labels: map[string]string{
				"app":       "kueue-benchmark",
				"component": "client-go-rest",
				"tier":      "benchmark",
			},
			Annotations: map[string]string{
				"checksum/config":                   fmt.Sprintf("%064d", payloadBytes),
				"kubectl.kubernetes.io/restartedAt": "2026-05-06T00:00:00Z",
				"prometheus.io/path":                "/metrics",
				"prometheus.io/scrape":              "true",
				"benchmark.k8s.io/padding":          strings.Repeat("x", paddingBytes),
			},
		},
		Spec: v1.PodSpec{
			ServiceAccountName: "default",
			NodeName:           "node-0001",
			InitContainers:     initContainers,
			Containers:         containers,
			Volumes:            benchmarkVolumes(volumeCount, itemsPerVolume),
			RestartPolicy:      v1.RestartPolicyAlways,
			DNSPolicy:          v1.DNSClusterFirst,
			Tolerations: []v1.Toleration{
				{
					Key:      "dedicated",
					Operator: v1.TolerationOpEqual,
					Value:    "benchmark",
					Effect:   v1.TaintEffectNoSchedule,
				},
			},
		},
		Status: v1.PodStatus{
			Phase:             v1.PodRunning,
			HostIP:            "192.168.0.10",
			PodIP:             podIPs[0].IP,
			PodIPs:            podIPs,
			ContainerStatuses: containerStatuses,
			Conditions: []v1.PodCondition{
				{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				{Type: v1.PodReady, Status: v1.ConditionTrue},
				{Type: v1.ContainersReady, Status: v1.ConditionTrue},
				{Type: v1.PodInitialized, Status: v1.ConditionTrue},
			},
		},
	}
}

func benchmarkContainerPorts(containerIndex int) []v1.ContainerPort {
	return []v1.ContainerPort{
		{Name: "http", ContainerPort: int32(8080 + containerIndex%100), Protocol: v1.ProtocolTCP},
		{Name: "metrics", ContainerPort: int32(9090 + containerIndex%100), Protocol: v1.ProtocolTCP},
	}
}

func benchmarkEnvVars(containerIndex, count, valueLength int) []v1.EnvVar {
	env := make([]v1.EnvVar, 0, count)
	for i := 0; i < count; i++ {
		env = append(env, v1.EnvVar{
			Name:  fmt.Sprintf("CONFIG_%02d_%03d", containerIndex, i),
			Value: fmt.Sprintf("https://service-%02d.default.svc.cluster.local/%s", i%16, strings.Repeat("x", valueLength)),
		})
	}
	return env
}

func benchmarkVolumeMounts(count int) []v1.VolumeMount {
	mounts := make([]v1.VolumeMount, 0, count)
	for i := 0; i < count; i++ {
		mounts = append(mounts, v1.VolumeMount{
			Name:      fmt.Sprintf("config-%02d", i),
			MountPath: fmt.Sprintf("/etc/app/config-%02d", i),
			ReadOnly:  i%3 != 0,
		})
	}
	return mounts
}

func benchmarkVolumes(count, itemsPerVolume int) []v1.Volume {
	volumes := make([]v1.Volume, 0, count)
	for i := 0; i < count; i++ {
		volume := v1.Volume{Name: fmt.Sprintf("config-%02d", i)}
		switch i % 4 {
		case 0:
			volume.ConfigMap = &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{Name: fmt.Sprintf("app-config-%02d", i)},
				Items:                benchmarkKeyToPaths("config", i, itemsPerVolume),
			}
		case 1:
			volume.Secret = &v1.SecretVolumeSource{
				SecretName: fmt.Sprintf("app-secret-%02d", i),
				Items:      benchmarkKeyToPaths("secret", i, itemsPerVolume),
			}
		case 2:
			volume.Projected = &v1.ProjectedVolumeSource{
				Sources: []v1.VolumeProjection{
					{
						ConfigMap: &v1.ConfigMapProjection{
							LocalObjectReference: v1.LocalObjectReference{Name: fmt.Sprintf("projected-config-%02d", i)},
							Items:                benchmarkKeyToPaths("projected", i, itemsPerVolume),
						},
					},
					{
						DownwardAPI: &v1.DownwardAPIProjection{
							Items: []v1.DownwardAPIVolumeFile{
								{Path: "pod-name", FieldRef: &v1.ObjectFieldSelector{FieldPath: "metadata.name"}},
								{Path: "pod-namespace", FieldRef: &v1.ObjectFieldSelector{FieldPath: "metadata.namespace"}},
							},
						},
					},
				},
			}
		default:
			volume.EmptyDir = &v1.EmptyDirVolumeSource{}
		}
		volumes = append(volumes, volume)
	}
	return volumes
}

func benchmarkKeyToPaths(prefix string, volumeIndex, count int) []v1.KeyToPath {
	items := make([]v1.KeyToPath, 0, count)
	for i := 0; i < count; i++ {
		items = append(items, v1.KeyToPath{
			Key:  fmt.Sprintf("%s-key-%02d-%03d", prefix, volumeIndex, i),
			Path: fmt.Sprintf("%s/%02d/%03d.conf", prefix, volumeIndex, i),
		})
	}
	return items
}

func clampInt(v, low, high int) int {
	return minInt(maxInt(v, low), high)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func makeWorkloadWatchStreamBody(b testing.TB, workload *Workload, eventCount int, contentType string) []byte {
	b.Helper()

	info := serializerInfoForBenchmark(b, contentType)
	var buf bytes.Buffer
	frameWriter := info.StreamSerializer.Framer.NewFrameWriter(&buf)
	watchEventEncoder := restclientwatch.NewEncoder(
		streaming.NewEncoder(frameWriter, info.StreamSerializer.Serializer),
		workloadBenchmarkCodecs.EncoderForVersion(info.Serializer, kueueGV),
	)
	for i := 0; i < eventCount; i++ {
		if err := watchEventEncoder.Encode(&k8swatch.Event{Type: k8swatch.Modified, Object: workload}); err != nil {
			b.Fatalf("failed to encode %s watch event: %v", contentType, err)
		}
	}
	return buf.Bytes()
}

func encodeWorkloadForBenchmark(b testing.TB, workload *Workload, contentType string) []byte {
	b.Helper()

	info := serializerInfoForBenchmark(b, contentType)
	encoder := workloadBenchmarkCodecs.EncoderForVersion(info.Serializer, kueueGV)
	data, err := k8sruntime.Encode(encoder, workload)
	if err != nil {
		b.Fatalf("failed to encode workload as %s: %v", contentType, err)
	}
	return data
}

func serializerInfoForBenchmark(b testing.TB, contentType string) k8sruntime.SerializerInfo {
	b.Helper()

	info, ok := k8sruntime.SerializerInfoForMediaType(workloadBenchmarkCodecs.SupportedMediaTypes(), contentType)
	if !ok {
		b.Fatalf("serializer for %q was not found", contentType)
	}
	if info.StreamSerializer == nil {
		b.Fatalf("stream serializer for %q was not found", contentType)
	}
	return info
}

func benchmarkContentTypeName(contentType string) string {
	switch contentType {
	case k8sruntime.ContentTypeJSON:
		return "json"
	case k8sruntime.ContentTypeCBOR:
		return "cbor"
	default:
		return strings.ReplaceAll(contentType, "/", "_")
	}
}

var benchmarkWorkloadObject k8sruntime.Object
var benchmarkBytes []byte
