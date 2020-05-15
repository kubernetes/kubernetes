/*
Copyright 2020 The Kubernetes Authors.

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

package scheme

import (
	"bytes"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-scheduler/config/v1alpha2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
	"sigs.k8s.io/yaml"
)

func TestCodecsDecodePluginConfig(t *testing.T) {
	testCases := []struct {
		name         string
		data         []byte
		wantErr      string
		wantProfiles []config.KubeSchedulerProfile
	}{
		{
			name: "v1alpha2 all plugin args in default profile",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: InterPodAffinity
    args:
      hardPodAffinityWeight: 5
  - name: NodeLabel
    args:
      presentLabels: ["foo"]
  - name: NodeResourcesFit
    args:
      ignoredResources: ["foo"]
  - name: RequestedToCapacityRatio
    args:
      shape:
      - utilization: 1
  - name: PodTopologySpread
    args:
      defaultConstraints:
      - maxSkew: 1
        topologyKey: zone
        whenUnsatisfiable: ScheduleAnyway
  - name: ServiceAffinity
    args:
      affinityLabels: ["bar"]
  - name: NodeResourcesLeastAllocated
    args:
      resources:
      - name: cpu
        weight: 2
      - name: unknown
        weight: 1
  - name: NodeResourcesMostAllocated
    args:
      resources:
      - name: memory
        weight: 1
  - name: VolumeBinding
    args:
      bindTimeoutSeconds: 300
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "InterPodAffinity",
							Args: &config.InterPodAffinityArgs{HardPodAffinityWeight: 5},
						},
						{
							Name: "NodeLabel",
							Args: &config.NodeLabelArgs{PresentLabels: []string{"foo"}},
						},
						{
							Name: "NodeResourcesFit",
							Args: &config.NodeResourcesFitArgs{IgnoredResources: []string{"foo"}},
						},
						{
							Name: "RequestedToCapacityRatio",
							Args: &config.RequestedToCapacityRatioArgs{
								Shape:     []config.UtilizationShapePoint{{Utilization: 1}},
								Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
							},
						},
						{
							Name: "PodTopologySpread",
							Args: &config.PodTopologySpreadArgs{
								DefaultConstraints: []v1.TopologySpreadConstraint{
									{MaxSkew: 1, TopologyKey: "zone", WhenUnsatisfiable: v1.ScheduleAnyway},
								},
							},
						},
						{
							Name: "ServiceAffinity",
							Args: &config.ServiceAffinityArgs{
								AffinityLabels: []string{"bar"},
							},
						},
						{
							Name: "NodeResourcesLeastAllocated",
							Args: &config.NodeResourcesLeastAllocatedArgs{
								Resources: []config.ResourceSpec{{Name: "cpu", Weight: 2}, {Name: "unknown", Weight: 1}},
							},
						},
						{
							Name: "NodeResourcesMostAllocated",
							Args: &config.NodeResourcesMostAllocatedArgs{
								Resources: []config.ResourceSpec{{Name: "memory", Weight: 1}},
							},
						},
						{
							Name: "VolumeBinding",
							Args: &config.VolumeBindingArgs{
								BindTimeoutSeconds: 300,
							},
						},
					},
				},
			},
		},
		{
			name: "v1alpha2 plugins can include version and kind",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: NodeLabel
    args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      kind: NodeLabelArgs
      presentLabels: ["bars"]
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "NodeLabel",
							Args: &config.NodeLabelArgs{PresentLabels: []string{"bars"}},
						},
					},
				},
			},
		},
		{
			name: "plugin group and kind should match the type",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: NodeLabel
    args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      kind: InterPodAffinityArgs
`),
			wantErr: "decoding .profiles[0].pluginConfig[0]: args for plugin NodeLabel were not of type NodeLabelArgs.kubescheduler.config.k8s.io, got InterPodAffinityArgs.kubescheduler.config.k8s.io",
		},
		{
			// TODO: do not replicate this case for v1beta1.
			name: "v1alpha2 case insensitive RequestedToCapacityRatioArgs",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: RequestedToCapacityRatio
    args:
      shape:
      - utilization: 1
        score: 2
      - Utilization: 3
        Score: 4
      resources:
      - name: Upper
        weight: 1
      - Name: lower
        weight: 2
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "RequestedToCapacityRatio",
							Args: &config.RequestedToCapacityRatioArgs{
								Shape: []config.UtilizationShapePoint{
									{Utilization: 1, Score: 2},
									{Utilization: 3, Score: 4},
								},
								Resources: []config.ResourceSpec{
									{Name: "Upper", Weight: 1},
									{Name: "lower", Weight: 2},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "out-of-tree plugin args",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: OutOfTreePlugin
    args:
      foo: bar
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "OutOfTreePlugin",
							Args: &runtime.Unknown{
								ContentType: "application/json",
								Raw:         []byte(`{"foo":"bar"}`),
							},
						},
					},
				},
			},
		},
		{
			name: "empty and no plugin args",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1alpha2
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: InterPodAffinity
    args:
  - name: NodeResourcesFit
  - name: OutOfTreePlugin
    args:
  - name: NodeResourcesLeastAllocated
    args:
  - name: NodeResourcesMostAllocated
    args:
  - name: VolumeBinding
    args:
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "InterPodAffinity",
							Args: &config.InterPodAffinityArgs{
								HardPodAffinityWeight: 1,
							},
						},
						{
							Name: "NodeResourcesFit",
							Args: &config.NodeResourcesFitArgs{},
						},
						{Name: "OutOfTreePlugin"},
						{
							Name: "NodeResourcesLeastAllocated",
							Args: &config.NodeResourcesLeastAllocatedArgs{
								Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
							},
						},
						{
							Name: "NodeResourcesMostAllocated",
							Args: &config.NodeResourcesMostAllocatedArgs{
								Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}, {Name: "memory", Weight: 1}},
							},
						},
						{
							Name: "VolumeBinding",
							Args: &config.VolumeBindingArgs{
								BindTimeoutSeconds: 600,
							},
						},
					},
				},
			},
		},
	}
	decoder := Codecs.UniversalDecoder()
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			obj, gvk, err := decoder.Decode(tt.data, nil, nil)
			if err != nil {
				if tt.wantErr != err.Error() {
					t.Fatalf("got err %w, want %w", err, tt.wantErr)
				}
				return
			}
			if len(tt.wantErr) != 0 {
				t.Fatal("no error produced, wanted %w", tt.wantErr)
			}
			got, ok := obj.(*config.KubeSchedulerConfiguration)
			if !ok {
				t.Fatalf("decoded into %s, want %s", gvk, config.SchemeGroupVersion.WithKind("KubeSchedulerConfiguration"))
			}
			if diff := cmp.Diff(tt.wantProfiles, got.Profiles); diff != "" {
				t.Errorf("unexpected configuration (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestCodecsEncodePluginConfig(t *testing.T) {
	testCases := []struct {
		name    string
		obj     runtime.Object
		version schema.GroupVersion
		want    string
	}{
		{
			name:    "v1alpha2 in-tree and out-of-tree plugins",
			version: v1alpha2.SchemeGroupVersion,
			obj: &v1alpha2.KubeSchedulerConfiguration{
				Profiles: []v1alpha2.KubeSchedulerProfile{
					{
						PluginConfig: []v1alpha2.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1alpha2.InterPodAffinityArgs{
										HardPodAffinityWeight: pointer.Int32Ptr(5),
									},
								},
							},
							{
								Name: "VolumeBinding",
								Args: runtime.RawExtension{
									Object: &v1alpha2.VolumeBindingArgs{
										BindTimeoutSeconds: pointer.Int64Ptr(300),
									},
								},
							},
							{
								Name: "RequestedToCapacityRatio",
								Args: runtime.RawExtension{
									Object: &v1alpha2.RequestedToCapacityRatioArgs{
										Shape: []v1alpha2.UtilizationShapePoint{
											{Utilization: 1, Score: 2},
										},
										Resources: []v1alpha2.ResourceSpec{
											{Name: "lower", Weight: 2},
										},
									},
								},
							},
							{
								Name: "NodeResourcesLeastAllocated",
								Args: runtime.RawExtension{
									Object: &v1alpha2.NodeResourcesLeastAllocatedArgs{
										Resources: []v1alpha2.ResourceSpec{
											{Name: "mem", Weight: 2},
										},
									},
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: runtime.RawExtension{
									Raw: []byte(`{"foo":"bar"}`),
								},
							},
						},
					},
				},
			},
			want: `apiVersion: kubescheduler.config.k8s.io/v1alpha2
clientConnection:
  acceptContentTypes: ""
  burst: 0
  contentType: ""
  kubeconfig: ""
  qps: 0
kind: KubeSchedulerConfiguration
leaderElection:
  leaderElect: null
  leaseDuration: 0s
  renewDeadline: 0s
  resourceLock: ""
  resourceName: ""
  resourceNamespace: ""
  retryPeriod: 0s
profiles:
- pluginConfig:
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      hardPodAffinityWeight: 5
      kind: InterPodAffinityArgs
    name: InterPodAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      bindTimeoutSeconds: 300
      kind: VolumeBindingArgs
    name: VolumeBinding
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      kind: RequestedToCapacityRatioArgs
      resources:
      - Name: lower
        Weight: 2
      shape:
      - Score: 2
        Utilization: 1
    name: RequestedToCapacityRatio
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      kind: NodeResourcesLeastAllocatedArgs
      resources:
      - Name: mem
        Weight: 2
    name: NodeResourcesLeastAllocated
  - args:
      foo: bar
    name: OutOfTreePlugin
`,
		},
		{
			name:    "v1alpha2 in-tree and out-of-tree plugins from internal",
			version: v1alpha2.SchemeGroupVersion,
			obj: &config.KubeSchedulerConfiguration{
				Profiles: []config.KubeSchedulerProfile{
					{
						PluginConfig: []config.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: &config.InterPodAffinityArgs{
									HardPodAffinityWeight: 5,
								},
							},
							{
								Name: "NodeResourcesMostAllocated",
								Args: &config.NodeResourcesMostAllocatedArgs{
									Resources: []config.ResourceSpec{{Name: "cpu", Weight: 1}},
								},
							},
							{
								Name: "VolumeBinding",
								Args: &config.VolumeBindingArgs{
									BindTimeoutSeconds: 300,
								},
							},
							{
								Name: "OutOfTreePlugin",
								Args: &runtime.Unknown{
									Raw: []byte(`{"foo":"bar"}`),
								},
							},
						},
					},
				},
			},
			want: `apiVersion: kubescheduler.config.k8s.io/v1alpha2
bindTimeoutSeconds: 0
clientConnection:
  acceptContentTypes: ""
  burst: 0
  contentType: ""
  kubeconfig: ""
  qps: 0
disablePreemption: false
enableContentionProfiling: false
enableProfiling: false
healthzBindAddress: ""
kind: KubeSchedulerConfiguration
leaderElection:
  leaderElect: false
  leaseDuration: 0s
  renewDeadline: 0s
  resourceLock: ""
  resourceName: ""
  resourceNamespace: ""
  retryPeriod: 0s
metricsBindAddress: ""
percentageOfNodesToScore: 0
podInitialBackoffSeconds: 0
podMaxBackoffSeconds: 0
profiles:
- pluginConfig:
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      hardPodAffinityWeight: 5
      kind: InterPodAffinityArgs
    name: InterPodAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      kind: NodeResourcesMostAllocatedArgs
      resources:
      - Name: cpu
        Weight: 1
    name: NodeResourcesMostAllocated
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1alpha2
      bindTimeoutSeconds: 300
      kind: VolumeBindingArgs
    name: VolumeBinding
  - args:
      foo: bar
    name: OutOfTreePlugin
  schedulerName: ""
`,
		},
	}
	yamlInfo, ok := runtime.SerializerInfoForMediaType(Codecs.SupportedMediaTypes(), runtime.ContentTypeYAML)
	if !ok {
		t.Fatalf("unable to locate encoder -- %q is not a supported media type", runtime.ContentTypeYAML)
	}
	jsonInfo, ok := runtime.SerializerInfoForMediaType(Codecs.SupportedMediaTypes(), runtime.ContentTypeJSON)
	if !ok {
		t.Fatalf("unable to locate encoder -- %q is not a supported media type", runtime.ContentTypeJSON)
	}
	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			encoder := Codecs.EncoderForVersion(yamlInfo.Serializer, tt.version)
			var buf bytes.Buffer
			if err := encoder.Encode(tt.obj, &buf); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, buf.String()); diff != "" {
				t.Errorf("unexpected encoded configuration:\n%s", diff)
			}
			encoder = Codecs.EncoderForVersion(jsonInfo.Serializer, tt.version)
			buf = bytes.Buffer{}
			if err := encoder.Encode(tt.obj, &buf); err != nil {
				t.Fatal(err)
			}
			out, err := yaml.JSONToYAML(buf.Bytes())
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(tt.want, string(out)); diff != "" {
				t.Errorf("unexpected encoded configuration:\n%s", diff)
			}
		})
	}
}
