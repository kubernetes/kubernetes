/*
Copyright 2024 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"net"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	extensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	aggregatorscheme "k8s.io/kube-aggregator/pkg/apiserver/scheme"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controlplane/apiserver/options"
	generatedopenapi "k8s.io/kubernetes/pkg/generated/openapi"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

func TestBuildGenericConfig(t *testing.T) {
	opts := options.NewOptions()
	s := (&apiserveroptions.SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
	}).WithLoopback()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on 127.0.0.1:0")
	}
	defer ln.Close()
	s.Listener = ln
	s.BindPort = ln.Addr().(*net.TCPAddr).Port
	opts.SecureServing = s

	completedOptions, err := opts.Complete(context.TODO(), nil, nil)
	if err != nil {
		t.Fatalf("Failed to complete apiserver options: %v", err)
	}

	genericConfig, _, storageFactory, err := BuildGenericConfig(
		completedOptions,
		[]*runtime.Scheme{legacyscheme.Scheme, extensionsapiserver.Scheme, aggregatorscheme.Scheme},
		nil,
		generatedopenapi.GetOpenAPIDefinitions,
		nil,
	)
	if err != nil {
		t.Fatalf("Failed to build generic config: %v", err)
	}
	if genericConfig.StorageObjectCountTracker == nil {
		t.Errorf("genericConfig StorageObjectCountTracker is absent")
	}
	if genericConfig.StorageObjectCountTracker != storageFactory.StorageConfig.StorageObjectCountTracker {
		t.Errorf("There are different StorageObjectCountTracker in genericConfig and storageFactory")
	}

	restOptions, err := genericConfig.RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: "", Resource: ""}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if restOptions.StorageConfig.StorageObjectCountTracker != genericConfig.StorageObjectCountTracker {
		t.Errorf("There are different StorageObjectCountTracker in restOptions and serverConfig")
	}
}

// fullPodSpecForTrimTest returns a Pod with every field relevant to
// trimPodSpec set to a non-zero value.
func fullPodSpecForTrimTest() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-pod",
			Namespace:   "test-ns",
			Annotations: map[string]string{"example.com/foo": "bar"},
		},
		Spec: corev1.PodSpec{
			// Fields trimPodSpec must clear.
			RestartPolicy:                 corev1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: ptr.To(int64(30)),
			DNSPolicy:                     corev1.DNSClusterFirst,
			NodeSelector:                  map[string]string{"kubernetes.io/os": "linux"},
			DeprecatedServiceAccount:      "legacy-sa",
			AutomountServiceAccountToken:  ptr.To(true),
			ShareProcessNamespace:         ptr.To(true),
			Hostname:                      "my-host",
			Subdomain:                     "my-subdomain",
			SchedulerName:                 "default-scheduler",
			Tolerations:                   []corev1.Toleration{{Key: "k", Operator: corev1.TolerationOpExists}},
			HostAliases:                   []corev1.HostAlias{{IP: "10.0.0.1", Hostnames: []string{"foo"}}},
			Priority:                      ptr.To(int32(10)),
			DNSConfig:                     &corev1.PodDNSConfig{Nameservers: []string{"8.8.8.8"}},
			ReadinessGates:                []corev1.PodReadinessGate{{ConditionType: "my-condition"}},
			EnableServiceLinks:            ptr.To(true),
			PreemptionPolicy:              ptr.To(corev1.PreemptLowerPriority),
			TopologySpreadConstraints:     []corev1.TopologySpreadConstraint{{MaxSkew: 1, TopologyKey: "kubernetes.io/hostname"}},
			SetHostnameAsFQDN:             ptr.To(true),
			SchedulingGates:               []corev1.PodSchedulingGate{{Name: "my-gate"}},

			// Fields at least one consumer reads; trimPodSpec must leave these alone.
			NodeName:           "node-1",
			ServiceAccountName: "my-sa",
			Volumes: []corev1.Volume{
				{Name: "v", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "s"}}},
			},
			ResourceClaims: []corev1.PodResourceClaim{{Name: "c"}},
			Containers: []corev1.Container{
				{
					Name: "c",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("100m")},
					},
				},
			},
		},
	}
}

func TestTrimPodSpec(t *testing.T) {
	pod := fullPodSpecForTrimTest()
	trimPodSpec(pod)

	want := fullPodSpecForTrimTest()
	want.Spec.RestartPolicy = ""
	want.Spec.TerminationGracePeriodSeconds = nil
	want.Spec.DNSPolicy = ""
	want.Spec.NodeSelector = nil
	want.Spec.DeprecatedServiceAccount = ""
	want.Spec.AutomountServiceAccountToken = nil
	want.Spec.ShareProcessNamespace = nil
	want.Spec.Hostname = ""
	want.Spec.Subdomain = ""
	want.Spec.SchedulerName = ""
	want.Spec.Tolerations = nil
	want.Spec.HostAliases = nil
	want.Spec.Priority = nil
	want.Spec.DNSConfig = nil
	want.Spec.ReadinessGates = nil
	want.Spec.EnableServiceLinks = nil
	want.Spec.PreemptionPolicy = nil
	want.Spec.TopologySpreadConstraints = nil
	want.Spec.SetHostnameAsFQDN = nil
	want.Spec.SchedulingGates = nil

	if diff := cmp.Diff(want, pod); diff != "" {
		t.Errorf("trimPodSpec() mismatch (-want +got):\n%s", diff)
	}
}
