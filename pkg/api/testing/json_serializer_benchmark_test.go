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

package testing

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func BenchmarkJSONSerializerEncode(b *testing.B) {
	payloads := map[string]func() runtime.Object{
		"SmallPodList500": sync.OnceValue(func() runtime.Object { return podList(500, smallPod) }),
		"PodList500":      sync.OnceValue(func() runtime.Object { return podList(500, webPod) }),
		"PodList500x5IOS": sync.OnceValue(func() runtime.Object { return podList(500, webPod5IOS) }),
		"Hetero1MiB":      sync.OnceValue(func() runtime.Object { return cmHetero(1000, 1000, 500, 1<<20) }),
	}
	modes := map[string]json.SerializerOptions{
		"Compact": {},
		"Pretty":  {Pretty: true},
		"Stream":  {StreamingCollectionsEncoding: true},
	}
	for _, c := range []struct{ mode, payload string }{
		{"Stream", "SmallPodList500"},
		{"Stream", "PodList500"},
		{"Stream", "PodList500x5IOS"},
		{"Stream", "Hetero1MiB"},
		{"Pretty", "SmallPodList500"},
		{"Pretty", "PodList500"},
		{"Compact", "PodList500"},
	} {
		b.Run(c.mode+"/"+c.payload, func(b *testing.B) {
			obj := payloads[c.payload]()
			s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, modes[c.mode])
			if err := s.Encode(obj, io.Discard); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			for b.Loop() {
				if err := s.Encode(obj, io.Discard); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkJSONSerializerPretty(b *testing.B) {
	s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{Pretty: true})
	for _, c := range []struct {
		name string
		obj  runtime.Object
	}{
		{"Pod", nginxPod()},
		{"PodList100", nginxPodList(100)},
		{"PodList1000", nginxPodList(1000)},
	} {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if err := s.Encode(c.obj, io.Discard); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func podList(n int, mk func(int) v1.Pod) *v1.PodList {
	l := &v1.PodList{
		TypeMeta: metav1.TypeMeta{Kind: "PodList", APIVersion: "v1"},
		ListMeta: metav1.ListMeta{ResourceVersion: "48299999"},
		Items:    make([]v1.Pod, n),
	}
	for i := range n {
		l.Items[i] = mk(i)
	}
	return l
}

func webPod5IOS(i int) v1.Pod {
	p := webPod(i)
	p.Spec.Containers[1].LivenessProbe = probe("/live", 15000, 3)
	return p
}

var epoch = time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)

func ts(offsetSec int) metav1.Time {
	return metav1.NewTime(epoch.Add(time.Duration(offsetSec) * time.Second))
}

const fieldsSpec = `{"f:metadata":{"f:generateName":{},"f:labels":{".":{},"f:app.kubernetes.io/name":{},"f:app.kubernetes.io/component":{},"f:pod-template-hash":{},"f:tier":{}},"f:ownerReferences":{".":{},"k:{\"uid\":\"6c0b1c9e-0f4e-4b8e-9a53-2d8a3f1e7b10\"}":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"app\"}":{".":{},"f:env":{".":{},"k:{\"name\":\"POD_NAME\"}":{".":{},"f:name":{},"f:valueFrom":{".":{},"f:fieldRef":{}}},"k:{\"name\":\"LOG_LEVEL\"}":{".":{},"f:name":{},"f:value":{}}},"f:image":{},"f:imagePullPolicy":{},"f:livenessProbe":{".":{},"f:failureThreshold":{},"f:httpGet":{".":{},"f:path":{},"f:port":{},"f:scheme":{}},"f:periodSeconds":{}},"f:name":{},"f:ports":{".":{},"k:{\"containerPort\":8080,\"protocol\":\"TCP\"}":{".":{},"f:containerPort":{},"f:name":{},"f:protocol":{}}},"f:resources":{".":{},"f:limits":{".":{},"f:cpu":{},"f:memory":{}},"f:requests":{".":{},"f:cpu":{},"f:memory":{}}},"f:volumeMounts":{".":{},"k:{\"mountPath\":\"/etc/config\"}":{".":{},"f:mountPath":{},"f:name":{}}}},"k:{\"name\":\"sidecar\"}":{".":{},"f:image":{},"f:name":{},"f:resources":{".":{},"f:requests":{".":{},"f:cpu":{},"f:memory":{}}}}},"f:dnsPolicy":{},"f:enableServiceLinks":{},"f:restartPolicy":{},"f:schedulerName":{},"f:securityContext":{".":{},"f:runAsNonRoot":{}},"f:terminationGracePeriodSeconds":{},"f:volumes":{".":{},"k:{\"name\":\"config\"}":{".":{},"f:configMap":{".":{},"f:defaultMode":{},"f:name":{}},"f:name":{}}}}}`

const fieldsStatus = `{"f:status":{"f:conditions":{"k:{\"type\":\"ContainersReady\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Initialized\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"PodReadyToStartContainers\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Ready\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}}},"f:containerStatuses":{},"f:hostIP":{},"f:hostIPs":{},"f:phase":{},"f:podIP":{},"f:podIPs":{".":{},"k:{\"ip\":\"10.244.3.17\"}":{".":{},"f:ip":{}}},"f:startTime":{}}}`

const fieldsBinding = `{"f:spec":{"f:nodeName":{}}}`

const fieldsLabels = `{"f:metadata":{"f:annotations":{".":{},"f:prometheus.io/port":{},"f:prometheus.io/scrape":{}},"f:labels":{"f:rollout-id":{}}}}`

const fieldsApply = `{"f:metadata":{"f:annotations":{"f:kubectl.kubernetes.io/last-applied-configuration":{}}}}`

const fieldsFinalizer = `{"f:metadata":{"f:finalizers":{".":{},"v:\"example.com/cleanup\"":{}}}}`

func fv1(s string) *metav1.FieldsV1 { return &metav1.FieldsV1{Raw: []byte(s)} }

func q(s string) resource.Quantity { return resource.MustParse(s) }

func probe(path string, port int, initial int32) *v1.Probe {
	return &v1.Probe{
		ProbeHandler: v1.ProbeHandler{HTTPGet: &v1.HTTPGetAction{
			Path: path, Port: intstr.FromInt32(int32(port)), Scheme: v1.URISchemeHTTP,
		}},
		InitialDelaySeconds: initial, TimeoutSeconds: 1, PeriodSeconds: 10, SuccessThreshold: 1, FailureThreshold: 3,
	}
}

func webPod(i int) v1.Pod {
	name := fmt.Sprintf("web-frontend-7d9f8b6c5d-%05d", i)
	uid := types.UID(fmt.Sprintf("%08x-1f2e-4d3c-8b4a-%012x", i, i*7919))
	lastApplied := `{"apiVersion":"v1","kind":"Pod","metadata":{"annotations":{"prometheus.io/port":"9090","prometheus.io/scrape":"true"},"labels":{"app.kubernetes.io/name":"web-frontend","tier":"frontend"},"name":"` + name + `","namespace":"production"},"spec":{"containers":[{"image":"registry.example.com/web/frontend:v2.14.3","name":"app","ports":[{"containerPort":8080}]}]}}` + "\n"
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			GenerateName:      "web-frontend-7d9f8b6c5d-",
			Namespace:         "production",
			UID:               uid,
			ResourceVersion:   strconv.Itoa(48213000 + i),
			Generation:        1,
			CreationTimestamp: ts(i),
			Labels: map[string]string{
				"app.kubernetes.io/name":      "web-frontend",
				"app.kubernetes.io/component": "frontend",
				"app.kubernetes.io/part-of":   "storefront",
				"pod-template-hash":           "7d9f8b6c5d",
				"tier":                        "frontend",
				"rollout-id":                  fmt.Sprintf("r-%04d", i%97),
			},
			Annotations: map[string]string{
				"prometheus.io/scrape":                             "true",
				"prometheus.io/port":                               "9090",
				"kubectl.kubernetes.io/restartedAt":                "2026-01-01T00:00:00Z",
				"kubectl.kubernetes.io/last-applied-configuration": lastApplied,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "web-frontend-7d9f8b6c5d",
				UID:        "6c0b1c9e-0f4e-4b8e-9a53-2d8a3f1e7b10",
				Controller: new(true), BlockOwnerDeletion: new(true),
			}},
			Finalizers: []string{"example.com/cleanup"},
			ManagedFields: []metav1.ManagedFieldsEntry{
				{Manager: "kube-controller-manager", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: new(ts(i)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsSpec)},
				{Manager: "kube-scheduler", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: new(ts(i + 1)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsBinding), Subresource: "binding"},
				{Manager: "kubelet", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: new(ts(i + 9)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsStatus), Subresource: "status"},
				{Manager: "kubectl-client-side-apply", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: new(ts(i + 30)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsApply)},
				{Manager: "rollout-operator", Operation: metav1.ManagedFieldsOperationApply, APIVersion: "v1", Time: new(ts(i + 60)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsLabels)},
				{Manager: "cleanup-controller", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: new(ts(i + 61)), FieldsType: "FieldsV1", FieldsV1: fv1(fieldsFinalizer)},
			},
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{Name: "config", VolumeSource: v1.VolumeSource{ConfigMap: &v1.ConfigMapVolumeSource{LocalObjectReference: v1.LocalObjectReference{Name: "web-frontend-config"}, DefaultMode: new(int32(420))}}},
				{Name: "tls", VolumeSource: v1.VolumeSource{Secret: &v1.SecretVolumeSource{SecretName: "web-frontend-tls", DefaultMode: new(int32(256))}}},
				{Name: "cache", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{Medium: v1.StorageMediumMemory, SizeLimit: new(q("256Mi"))}}},
				{Name: "kube-api-access-" + fmt.Sprintf("%05d", i), VolumeSource: v1.VolumeSource{Projected: &v1.ProjectedVolumeSource{
					DefaultMode: new(int32(420)),
					Sources: []v1.VolumeProjection{
						{ServiceAccountToken: &v1.ServiceAccountTokenProjection{ExpirationSeconds: new(int64(3607)), Path: "token"}},
						{ConfigMap: &v1.ConfigMapProjection{LocalObjectReference: v1.LocalObjectReference{Name: "kube-root-ca.crt"}, Items: []v1.KeyToPath{{Key: "ca.crt", Path: "ca.crt"}}}},
						{DownwardAPI: &v1.DownwardAPIProjection{Items: []v1.DownwardAPIVolumeFile{{Path: "namespace", FieldRef: &v1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.namespace"}}}}},
					},
				}}},
			},
			InitContainers: []v1.Container{{
				Name: "migrate", Image: "registry.example.com/web/migrate:v2.14.3",
				Command: []string{"/bin/migrate", "--database=$(DB_URL)", "--timeout=120s"},
				Env:     []v1.EnvVar{{Name: "DB_URL", ValueFrom: &v1.EnvVarSource{SecretKeyRef: &v1.SecretKeySelector{LocalObjectReference: v1.LocalObjectReference{Name: "db"}, Key: "url"}}}},
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: q("100m"), v1.ResourceMemory: q("64Mi")},
				},
				TerminationMessagePath: "/dev/termination-log", TerminationMessagePolicy: v1.TerminationMessageReadFile, ImagePullPolicy: v1.PullIfNotPresent,
			}},
			Containers: []v1.Container{
				{
					Name: "app", Image: "registry.example.com/web/frontend:v2.14.3",
					Args:  []string{"--port=8080", "--metrics-port=9090", "--log-format=json", "--feature-gates=Checkout=true,Search=false"},
					Ports: []v1.ContainerPort{{Name: "http", ContainerPort: 8080, Protocol: v1.ProtocolTCP}, {Name: "metrics", ContainerPort: 9090, Protocol: v1.ProtocolTCP}},
					Env: []v1.EnvVar{
						{Name: "POD_NAME", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "metadata.name"}}},
						{Name: "POD_IP", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{APIVersion: "v1", FieldPath: "status.podIP"}}},
						{Name: "LOG_LEVEL", Value: "info"},
						{Name: "UPSTREAM_URL", Value: "http://catalog.production.svc.cluster.local:8080/api?x=1&y=<2>"},
						{Name: "GOMAXPROCS", ValueFrom: &v1.EnvVarSource{ResourceFieldRef: &v1.ResourceFieldSelector{Resource: "limits.cpu", Divisor: q("1")}}},
						{Name: "API_TOKEN", ValueFrom: &v1.EnvVarSource{SecretKeyRef: &v1.SecretKeySelector{LocalObjectReference: v1.LocalObjectReference{Name: "api"}, Key: "token", Optional: new(false)}}},
					},
					Resources: v1.ResourceRequirements{
						Limits:   v1.ResourceList{v1.ResourceCPU: q("2"), v1.ResourceMemory: q("1Gi")},
						Requests: v1.ResourceList{v1.ResourceCPU: q("500m"), v1.ResourceMemory: q("512Mi"), v1.ResourceEphemeralStorage: q("1Gi")},
					},
					VolumeMounts: []v1.VolumeMount{
						{Name: "config", MountPath: "/etc/config", ReadOnly: true},
						{Name: "tls", MountPath: "/etc/tls", ReadOnly: true},
						{Name: "cache", MountPath: "/var/cache/app"},
						{Name: "kube-api-access-" + fmt.Sprintf("%05d", i), MountPath: "/var/run/secrets/kubernetes.io/serviceaccount", ReadOnly: true},
					},
					LivenessProbe:            probe("/healthz", 8080, 10),
					ReadinessProbe:           probe("/readyz", 8080, 5),
					StartupProbe:             probe("/startupz", 8080, 0),
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: v1.TerminationMessageReadFile,
					ImagePullPolicy:          v1.PullIfNotPresent,
					SecurityContext: &v1.SecurityContext{
						AllowPrivilegeEscalation: new(false), ReadOnlyRootFilesystem: new(true),
						Capabilities: &v1.Capabilities{Drop: []v1.Capability{"ALL"}},
					},
				},
				{
					Name: "sidecar", Image: "registry.example.com/infra/envoy:v1.31.2",
					Args:  []string{"-c", "/etc/envoy/envoy.yaml", "--service-cluster", "web-frontend"},
					Ports: []v1.ContainerPort{{Name: "envoy-admin", ContainerPort: 15000, Protocol: v1.ProtocolTCP}},
					Resources: v1.ResourceRequirements{
						Limits:   v1.ResourceList{v1.ResourceMemory: q("256Mi")},
						Requests: v1.ResourceList{v1.ResourceCPU: q("100m"), v1.ResourceMemory: q("128Mi")},
					},
					VolumeMounts:             []v1.VolumeMount{{Name: "config", MountPath: "/etc/envoy", SubPath: "envoy", ReadOnly: true}},
					ReadinessProbe:           probe("/ready", 15000, 1),
					TerminationMessagePath:   "/dev/termination-log",
					TerminationMessagePolicy: v1.TerminationMessageReadFile,
					ImagePullPolicy:          v1.PullIfNotPresent,
				},
			},
			RestartPolicy:                 v1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: new(int64(30)),
			DNSPolicy:                     v1.DNSClusterFirst,
			ServiceAccountName:            "web-frontend",
			DeprecatedServiceAccount:      "web-frontend",
			NodeName:                      fmt.Sprintf("node-pool-a-%03d", i%250),
			SecurityContext:               &v1.PodSecurityContext{RunAsNonRoot: new(true), RunAsUser: new(int64(10001)), FSGroup: new(int64(10001))},
			SchedulerName:                 "default-scheduler",
			Tolerations: []v1.Toleration{
				{Key: "node.kubernetes.io/not-ready", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute, TolerationSeconds: new(int64(300))},
				{Key: "node.kubernetes.io/unreachable", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute, TolerationSeconds: new(int64(300))},
				{Key: "dedicated", Operator: v1.TolerationOpEqual, Value: "frontend", Effect: v1.TaintEffectNoSchedule},
			},
			Priority:           new(int32(0)),
			EnableServiceLinks: new(true),
			PreemptionPolicy:   new(v1.PreemptLowerPriority),
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			Conditions: []v1.PodCondition{
				{Type: "PodReadyToStartContainers", Status: v1.ConditionTrue, LastTransitionTime: ts(i + 5)},
				{Type: v1.PodInitialized, Status: v1.ConditionTrue, LastTransitionTime: ts(i + 7)},
				{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: ts(i + 12)},
				{Type: v1.ContainersReady, Status: v1.ConditionTrue, LastTransitionTime: ts(i + 12)},
				{Type: v1.PodScheduled, Status: v1.ConditionTrue, LastTransitionTime: ts(i + 1)},
			},
			HostIP:    fmt.Sprintf("10.0.%d.%d", (i/250)%256, i%250+1),
			HostIPs:   []v1.HostIP{{IP: fmt.Sprintf("10.0.%d.%d", (i/250)%256, i%250+1)}},
			PodIP:     fmt.Sprintf("10.244.%d.%d", (i/200)%256, i%200+2),
			PodIPs:    []v1.PodIP{{IP: fmt.Sprintf("10.244.%d.%d", (i/200)%256, i%200+2)}},
			StartTime: new(ts(i + 2)),
			InitContainerStatuses: []v1.ContainerStatus{{
				Name: "migrate", Ready: true, Image: "registry.example.com/web/migrate:v2.14.3",
				ImageID:     "registry.example.com/web/migrate@sha256:1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f809",
				ContainerID: fmt.Sprintf("containerd://%064x", i*31+1),
				State:       v1.ContainerState{Terminated: &v1.ContainerStateTerminated{ExitCode: 0, Reason: "Completed", StartedAt: ts(i + 3), FinishedAt: ts(i + 6), ContainerID: fmt.Sprintf("containerd://%064x", i*31+1)}},
			}},
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name: "app", Ready: true, RestartCount: int32(i % 3), Started: new(true),
					Image:       "registry.example.com/web/frontend:v2.14.3",
					ImageID:     "registry.example.com/web/frontend@sha256:9f8e7d6c5b4a39281706f5e4d3c2b1a09f8e7d6c5b4a39281706f5e4d3c2b1a0",
					ContainerID: fmt.Sprintf("containerd://%064x", i*31+2),
					State:       v1.ContainerState{Running: &v1.ContainerStateRunning{StartedAt: ts(i + 8)}},
				},
				{
					Name: "sidecar", Ready: true, Started: new(true),
					Image:       "registry.example.com/infra/envoy:v1.31.2",
					ImageID:     "registry.example.com/infra/envoy@sha256:0011223344556677889900aabbccddeeff0011223344556677889900aabbccdd",
					ContainerID: fmt.Sprintf("containerd://%064x", i*31+3),
					State:       v1.ContainerState{Running: &v1.ContainerStateRunning{StartedAt: ts(i + 8)}},
				},
			},
			QOSClass: v1.PodQOSBurstable,
		},
	}
}

func configText(seed, n int) string {
	var sb strings.Builder
	sb.Grow(n + 128)
	for line := 0; sb.Len() < n; line++ {
		fmt.Fprintf(&sb, "service.%d.endpoint.%d = \"https://svc-%d.internal/path?a=1&b=<%d>\"  # tuned value %08x\n", seed, line, (seed*131+line)%9973, line%17, (seed+1)*(line+7)*2654435761%4294967296)
	}
	return sb.String()
}

func makeConfigMap(i int, data map[string]string) v1.ConfigMap {
	return v1.ConfigMap{
		TypeMeta: metav1.TypeMeta{Kind: "ConfigMap", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("app-config-%04d", i), Namespace: "production",
			UID:               types.UID(fmt.Sprintf("%08x-aaaa-4bbb-8ccc-%012x", i, i*104729)),
			ResourceVersion:   strconv.Itoa(7700000 + i),
			CreationTimestamp: ts(i),
			Labels:            map[string]string{"app.kubernetes.io/name": "storefront", "config-shard": strconv.Itoa(i % 8)},
		},
		Data: data,
	}
}

func cmHetero(n, bigEvery, bigOffset, bigSize int) *v1.ConfigMapList {
	l := &v1.ConfigMapList{
		TypeMeta: metav1.TypeMeta{Kind: "ConfigMapList", APIVersion: "v1"},
		ListMeta: metav1.ListMeta{ResourceVersion: "7799999"},
	}
	for i := range n {
		data := map[string]string{"LOG_LEVEL": "info", "FEATURE_X": "enabled", "ENDPOINT": fmt.Sprintf("https://svc-%d.internal:8443", i)}
		if i%bigEvery == bigOffset {
			data = map[string]string{"bundle.txt": configText(i, bigSize)}
		}
		l.Items = append(l.Items, makeConfigMap(i, data))
	}
	return l
}

func smallPod(i int) v1.Pod {
	raw := `{"f:metadata":{"f:labels":{".":{},"f:app":{},"f:pod-template-hash":{}},"f:ownerReferences":{".":{},"k:{\"uid\":\"5d2e5c1a-1111-2222-3333-444455556666\"}":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"nginx\"}":{".":{},"f:image":{},"f:imagePullPolicy":{},"f:name":{},"f:ports":{".":{},"k:{\"containerPort\":80,\"protocol\":\"TCP\"}":{".":{},"f:containerPort":{},"f:protocol":{}}},"f:resources":{".":{},"f:limits":{".":{},"f:cpu":{},"f:memory":{}},"f:requests":{".":{},"f:cpu":{},"f:memory":{}}},"f:terminationMessagePath":{},"f:terminationMessagePolicy":{}}},"f:dnsPolicy":{},"f:enableServiceLinks":{},"f:restartPolicy":{},"f:schedulerName":{},"f:securityContext":{},"f:terminationGracePeriodSeconds":{}}}`
	statusRaw := `{"f:status":{"f:conditions":{"k:{\"type\":\"ContainersReady\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Initialized\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Ready\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}}},"f:containerStatuses":{},"f:hostIP":{},"f:phase":{},"f:podIP":{},"f:podIPs":{".":{},"k:{\"ip\":\"10.244.0.5\"}":{".":{},"f:ip":{}}},"f:startTime":{}}}`
	now := ts(i)
	return v1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("nginx-deployment-7c79c4bf97-%05d", i), Namespace: "default", UID: "0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b",
			ResourceVersion: strconv.Itoa(123456 + i), CreationTimestamp: now,
			Labels:          map[string]string{"app": "nginx", "pod-template-hash": "7c79c4bf97"},
			OwnerReferences: []metav1.OwnerReference{{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "nginx-deployment-7c79c4bf97", UID: "5d2e5c1a-1111-2222-3333-444455556666"}},
			ManagedFields: []metav1.ManagedFieldsEntry{
				{Manager: "kube-controller-manager", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: &now, FieldsType: "FieldsV1", FieldsV1: &metav1.FieldsV1{Raw: []byte(raw)}},
				{Manager: "kubelet", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Time: &now, FieldsType: "FieldsV1", FieldsV1: &metav1.FieldsV1{Raw: []byte(statusRaw)}, Subresource: "status"},
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "nginx", Image: "nginx:1.14.2", Ports: []v1.ContainerPort{{ContainerPort: 80, Protocol: "TCP"}},
				TerminationMessagePath: "/dev/termination-log", TerminationMessagePolicy: "File", ImagePullPolicy: "IfNotPresent"}},
			RestartPolicy: "Always", DNSPolicy: "ClusterFirst", SchedulerName: "default-scheduler", NodeName: fmt.Sprintf("node-%d", i%50),
		},
		Status: v1.PodStatus{Phase: "Running", HostIP: "10.0.0.1", PodIP: "10.244.0.5", StartTime: &now,
			Conditions: []v1.PodCondition{{Type: "Ready", Status: "True", LastTransitionTime: now}, {Type: "Initialized", Status: "True", LastTransitionTime: now}}},
	}
}

func nginxPodList(n int) *v1.PodList {
	l := &v1.PodList{TypeMeta: metav1.TypeMeta{Kind: "PodList", APIVersion: "v1"}, ListMeta: metav1.ListMeta{ResourceVersion: "99"}}
	for i := range n {
		p := nginxPod()
		p.Name = fmt.Sprintf("web-%d", i)
		l.Items = append(l.Items, *p)
	}
	return l
}

func nginxPod() *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{Kind: "Pod", APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Name: "web-0", Namespace: "default", UID: "5f1c6f0e-1b0e-4f2a-9f6d-2d8c3b7a9e11",
			ResourceVersion: "123456", CreationTimestamp: metav1.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC),
			Labels:      map[string]string{"app": "web", "tier": "frontend", "pod-template-hash": "7d9f8c6b5"},
			Annotations: map[string]string{"kubectl.kubernetes.io/restartedAt": "2026-09-24T12:00:00Z", "html": "<a href=\"x\">&</a>"},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "apps/v1", Kind: "StatefulSet", Name: "web", UID: "0b7c", Controller: new(true), BlockOwnerDeletion: new(true),
			}},
			ManagedFields: []metav1.ManagedFieldsEntry{{
				Manager: "kube-controller-manager", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1",
				Time: new(metav1.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)), FieldsType: "FieldsV1",
				FieldsV1: &metav1.FieldsV1{Raw: []byte(`{"f:metadata":{"f:generateName":{},"f:labels":{".":{},"f:app":{},"f:controller-revision-hash":{},"f:statefulset.kubernetes.io/pod-name":{}},"f:ownerReferences":{".":{},"k:{\"uid\":\"0b7c\"}":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"nginx\"}":{".":{},"f:image":{},"f:imagePullPolicy":{},"f:name":{},"f:ports":{".":{},"k:{\"containerPort\":80,\"protocol\":\"TCP\"}":{".":{},"f:containerPort":{},"f:name":{},"f:protocol":{}}},"f:resources":{},"f:terminationMessagePath":{},"f:terminationMessagePolicy":{},"f:volumeMounts":{".":{},"k:{\"mountPath\":\"/usr/share/nginx/html\"}":{".":{},"f:mountPath":{},"f:name":{}}}}},"f:dnsPolicy":{},"f:enableServiceLinks":{},"f:hostname":{},"f:restartPolicy":{},"f:schedulerName":{},"f:securityContext":{},"f:subdomain":{},"f:terminationGracePeriodSeconds":{},"f:volumes":{".":{},"k:{\"name\":\"www\"}":{".":{},"f:name":{},"f:persistentVolumeClaim":{".":{},"f:claimName":{}}}}}}`)},
			}, {
				Manager: "kubelet", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Subresource: "status",
				Time: new(metav1.Date(2026, 9, 24, 12, 0, 5, 0, time.UTC)), FieldsType: "FieldsV1",
				FieldsV1: &metav1.FieldsV1{Raw: []byte(`{"f:status":{"f:conditions":{"k:{\"type\":\"ContainersReady\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Initialized\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"PodScheduled\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}},"k:{\"type\":\"Ready\"}":{".":{},"f:lastProbeTime":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}}},"f:containerStatuses":{},"f:hostIP":{},"f:phase":{},"f:podIP":{},"f:podIPs":{".":{},"k:{\"ip\":\"10.244.0.5\"}":{".":{},"f:ip":{}}},"f:startTime":{}}}`)},
			}},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "nginx", Image: "registry.k8s.io/nginx-slim:0.8",
				Ports: []v1.ContainerPort{{Name: "web", ContainerPort: 80, Protocol: v1.ProtocolTCP}},
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m"), v1.ResourceMemory: resource.MustParse("1.5Gi")},
					Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("2Gi")},
				},
				VolumeMounts:             []v1.VolumeMount{{Name: "www", MountPath: "/usr/share/nginx/html"}},
				TerminationMessagePath:   "/dev/termination-log",
				TerminationMessagePolicy: v1.TerminationMessageReadFile,
				ImagePullPolicy:          v1.PullIfNotPresent,
				Env:                      []v1.EnvVar{{Name: "A", Value: "1"}, {Name: "B", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: "metadata.name"}}}},
			}},
			RestartPolicy: v1.RestartPolicyAlways, DNSPolicy: v1.DNSClusterFirst,
			TerminationGracePeriodSeconds: new(int64(30)), Hostname: "web-0", Subdomain: "nginx",
			SchedulerName: "default-scheduler", SecurityContext: &v1.PodSecurityContext{},
			Volumes: []v1.Volume{{Name: "www", VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "www-web-0"}}}},
			Tolerations: []v1.Toleration{
				{Key: "node.kubernetes.io/not-ready", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute, TolerationSeconds: new(int64(300))},
				{Key: "node.kubernetes.io/unreachable", Operator: v1.TolerationOpExists, Effect: v1.TaintEffectNoExecute, TolerationSeconds: new(int64(300))},
			},
			EnableServiceLinks: new(true),
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning, HostIP: "10.0.0.3", PodIP: "10.244.0.5",
			PodIPs:    []v1.PodIP{{IP: "10.244.0.5"}},
			StartTime: new(metav1.Date(2026, 9, 24, 12, 0, 1, 0, time.UTC)),
			Conditions: []v1.PodCondition{
				{Type: v1.PodInitialized, Status: v1.ConditionTrue, LastTransitionTime: metav1.Date(2026, 9, 24, 12, 0, 1, 0, time.UTC)},
				{Type: v1.PodReady, Status: v1.ConditionTrue, LastTransitionTime: metav1.Date(2026, 9, 24, 12, 0, 5, 0, time.UTC)},
				{Type: v1.ContainersReady, Status: v1.ConditionTrue, LastTransitionTime: metav1.Date(2026, 9, 24, 12, 0, 5, 0, time.UTC)},
				{Type: v1.PodScheduled, Status: v1.ConditionTrue, LastTransitionTime: metav1.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)},
			},
			ContainerStatuses: []v1.ContainerStatus{{
				Name: "nginx", Ready: true, RestartCount: 0, Image: "registry.k8s.io/nginx-slim:0.8",
				ImageID: "registry.k8s.io/nginx-slim@sha256:8b4501fe0fe221df663c22e16539f399e89594552f400408303c42f3dd8d0e52", ContainerID: "containerd://0a1b2c",
				Started: new(true), State: v1.ContainerState{Running: &v1.ContainerStateRunning{StartedAt: metav1.Date(2026, 9, 24, 12, 0, 4, 0, time.UTC)}},
			}},
			QOSClass: v1.PodQOSBurstable,
		},
	}
}
