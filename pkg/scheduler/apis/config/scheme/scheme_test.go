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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kube-scheduler/config/v1beta1"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/utils/pointer"
	"sigs.k8s.io/yaml"
)

// TestCodecsDecodePluginConfig tests that embedded plugin args get decoded
// into their appropriate internal types and defaults are applied.
func TestCodecsDecodePluginConfig(t *testing.T) {
	testCases := []struct {
		name         string
		data         []byte
		wantErr      string
		wantProfiles []config.KubeSchedulerProfile
	}{
		{
			name: "v1beta1 all plugin args in default profile",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: DefaultPreemption
    args:
      minCandidateNodesPercentage: 50
      minCandidateNodesAbsolute: 500
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
  - name: NodeAffinity
    args:
      addedAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
          - matchExpressions:
            - key: foo
              operator: In
              values: ["bar"]
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "DefaultPreemption",
							Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 50, MinCandidateNodesAbsolute: 500},
						},
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
								DefaultConstraints: []corev1.TopologySpreadConstraint{
									{MaxSkew: 1, TopologyKey: "zone", WhenUnsatisfiable: corev1.ScheduleAnyway},
								},
								DefaultingType: config.ListDefaulting,
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
						{
							Name: "NodeAffinity",
							Args: &config.NodeAffinityArgs{
								AddedAffinity: &corev1.NodeAffinity{
									RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
										NodeSelectorTerms: []corev1.NodeSelectorTerm{
											{
												MatchExpressions: []corev1.NodeSelectorRequirement{
													{
														Key:      "foo",
														Operator: corev1.NodeSelectorOpIn,
														Values:   []string{"bar"},
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
		},
		{
			name: "v1beta1 plugins can include version and kind",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: NodeLabel
    args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
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
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: NodeLabel
    args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: InterPodAffinityArgs
`),
			wantErr: `decoding .profiles[0].pluginConfig[0]: args for plugin NodeLabel were not of type NodeLabelArgs.kubescheduler.config.k8s.io, got InterPodAffinityArgs.kubescheduler.config.k8s.io`,
		},
		{
			name: "v1beta1 RequestedToCapacityRatioArgs shape encoding is strict",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: RequestedToCapacityRatio
    args:
      shape:
      - Utilization: 1
        Score: 2
`),
			wantErr: `decoding .profiles[0].pluginConfig[0]: decoding args for plugin RequestedToCapacityRatio: strict decoder error for {"shape":[{"Score":2,"Utilization":1}]}: v1beta1.RequestedToCapacityRatioArgs.Shape: []v1beta1.UtilizationShapePoint: v1beta1.UtilizationShapePoint.ReadObject: found unknown field: Score, error found in #10 byte of ...|:[{"Score":2,"Utiliz|..., bigger context ...|{"shape":[{"Score":2,"Utilization":1}]}|...`,
		},
		{
			name: "v1beta1 RequestedToCapacityRatioArgs resources encoding is strict",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: RequestedToCapacityRatio
    args:
      shape:
      - utilization: 1
        score: 2
      resources:
      - Name: 1
        Weight: 2
`),
			wantErr: `decoding .profiles[0].pluginConfig[0]: decoding args for plugin RequestedToCapacityRatio: strict decoder error for {"resources":[{"Name":1,"Weight":2}],"shape":[{"score":2,"utilization":1}]}: v1beta1.RequestedToCapacityRatioArgs.Shape: []v1beta1.UtilizationShapePoint: Resources: []v1beta1.ResourceSpec: v1beta1.ResourceSpec.ReadObject: found unknown field: Name, error found in #10 byte of ...|":[{"Name":1,"Weight|..., bigger context ...|{"resources":[{"Name":1,"Weight":2}],"shape":[{"score":2,"utilization":|...`,
		},
		{
			name: "out-of-tree plugin args",
			data: []byte(`
apiVersion: kubescheduler.config.k8s.io/v1beta1
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
apiVersion: kubescheduler.config.k8s.io/v1beta1
kind: KubeSchedulerConfiguration
profiles:
- pluginConfig:
  - name: DefaultPreemption
    args:
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
  - name: PodTopologySpread
  - name: NodeAffinity
`),
			wantProfiles: []config.KubeSchedulerProfile{
				{
					SchedulerName: "default-scheduler",
					PluginConfig: []config.PluginConfig{
						{
							Name: "DefaultPreemption",
							Args: &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 10, MinCandidateNodesAbsolute: 100},
						},
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
						{
							Name: "PodTopologySpread",
							Args: &config.PodTopologySpreadArgs{
								DefaultingType: config.SystemDefaulting,
							},
						},
						{
							Name: "NodeAffinity",
							Args: &config.NodeAffinityArgs{},
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
					t.Fatalf("\ngot err:\n\t%v\nwant:\n\t%s", err, tt.wantErr)
				}
				return
			}
			if len(tt.wantErr) != 0 {
				t.Fatalf("no error produced, wanted %v", tt.wantErr)
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
			name:    "v1beta1 in-tree and out-of-tree plugins",
			version: v1beta1.SchemeGroupVersion,
			obj: &v1beta1.KubeSchedulerConfiguration{
				Profiles: []v1beta1.KubeSchedulerProfile{
					{
						PluginConfig: []v1beta1.PluginConfig{
							{
								Name: "InterPodAffinity",
								Args: runtime.RawExtension{
									Object: &v1beta1.InterPodAffinityArgs{
										HardPodAffinityWeight: pointer.Int32Ptr(5),
									},
								},
							},
							{
								Name: "VolumeBinding",
								Args: runtime.RawExtension{
									Object: &v1beta1.VolumeBindingArgs{
										BindTimeoutSeconds: pointer.Int64Ptr(300),
									},
								},
							},
							{
								Name: "RequestedToCapacityRatio",
								Args: runtime.RawExtension{
									Object: &v1beta1.RequestedToCapacityRatioArgs{
										Shape: []v1beta1.UtilizationShapePoint{
											{Utilization: 1, Score: 2},
										},
										Resources: []v1beta1.ResourceSpec{
											{Name: "cpu", Weight: 2},
										},
									},
								},
							},
							{
								Name: "NodeResourcesLeastAllocated",
								Args: runtime.RawExtension{
									Object: &v1beta1.NodeResourcesLeastAllocatedArgs{
										Resources: []v1beta1.ResourceSpec{
											{Name: "mem", Weight: 2},
										},
									},
								},
							},
							{
								Name: "PodTopologySpread",
								Args: runtime.RawExtension{
									Object: &v1beta1.PodTopologySpreadArgs{
										DefaultConstraints: []corev1.TopologySpreadConstraint{},
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
			want: `apiVersion: kubescheduler.config.k8s.io/v1beta1
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
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      hardPodAffinityWeight: 5
      kind: InterPodAffinityArgs
    name: InterPodAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      bindTimeoutSeconds: 300
      kind: VolumeBindingArgs
    name: VolumeBinding
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: RequestedToCapacityRatioArgs
      resources:
      - name: cpu
        weight: 2
      shape:
      - score: 2
        utilization: 1
    name: RequestedToCapacityRatio
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: NodeResourcesLeastAllocatedArgs
      resources:
      - name: mem
        weight: 2
    name: NodeResourcesLeastAllocated
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: PodTopologySpreadArgs
    name: PodTopologySpread
  - args:
      foo: bar
    name: OutOfTreePlugin
`,
		},
		{
			name:    "v1beta1 in-tree and out-of-tree plugins from internal",
			version: v1beta1.SchemeGroupVersion,
			obj: &config.KubeSchedulerConfiguration{
				Parallelism: 8,
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
								Name: "PodTopologySpread",
								Args: &config.PodTopologySpreadArgs{},
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
			want: `apiVersion: kubescheduler.config.k8s.io/v1beta1
clientConnection:
  acceptContentTypes: ""
  burst: 0
  contentType: ""
  kubeconfig: ""
  qps: 0
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
parallelism: 8
percentageOfNodesToScore: 0
podInitialBackoffSeconds: 0
podMaxBackoffSeconds: 0
profiles:
- pluginConfig:
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      hardPodAffinityWeight: 5
      kind: InterPodAffinityArgs
    name: InterPodAffinity
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: NodeResourcesMostAllocatedArgs
      resources:
      - name: cpu
        weight: 1
    name: NodeResourcesMostAllocated
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      bindTimeoutSeconds: 300
      kind: VolumeBindingArgs
    name: VolumeBinding
  - args:
      apiVersion: kubescheduler.config.k8s.io/v1beta1
      kind: PodTopologySpreadArgs
    name: PodTopologySpread
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
