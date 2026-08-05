/*
Copyright 2016 The Kubernetes Authors.

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

package top

import (
	"bytes"
	"encoding/json"
	"io"
	"time"

	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1api "k8s.io/metrics/pkg/apis/metrics/v1"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
)

// metricsAPIVersions lists the metrics API versions the top commands support,
// so tests can run every case against each version.
var metricsAPIVersions = []struct {
	name     string
	apisBody string
}{
	{"v1", apisV1BodyWithMetrics},
	{"v1beta1", apisV1beta1BodyWithMetrics},
}

func TestTopSubcommandsExist(t *testing.T) {
	cmdtesting.InitTestErrorHandler(t)

	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	cmd := NewCmdTop(f, genericiooptions.NewTestIOStreamsDiscard())
	if !cmd.HasSubCommands() {
		t.Error("top command should have subcommands")
	}
}

func marshallBody(metrics interface{}) (io.ReadCloser, error) {
	result, err := json.Marshal(metrics)
	if err != nil {
		return nil, err
	}
	return io.NopCloser(bytes.NewReader(result)), nil
}

// versionedNodeMetricsList converts internal node metrics to the given
// metrics API version, returning the list and its first item for use as fake
// clientset reactor return values.
func versionedNodeMetricsList(t *testing.T, version string, in *metricsapi.NodeMetricsList) (list, first runtime.Object) {
	t.Helper()
	switch version {
	case "v1":
		out := &metricsv1api.NodeMetricsList{}
		if err := metricsv1api.Convert_metrics_NodeMetricsList_To_v1_NodeMetricsList(in, out, nil); err != nil {
			t.Fatal(err)
		}
		return out, &out.Items[0]
	case "v1beta1":
		out := &metricsv1beta1api.NodeMetricsList{}
		if err := metricsv1beta1api.Convert_metrics_NodeMetricsList_To_v1beta1_NodeMetricsList(in, out, nil); err != nil {
			t.Fatal(err)
		}
		return out, &out.Items[0]
	default:
		t.Fatalf("unsupported metrics API version %q", version)
		return nil, nil
	}
}

// versionedPodMetricsList converts internal pod metrics to the given metrics
// API version, returning the list and its first item (nil for an empty list)
// for use as fake clientset reactor return values.
func versionedPodMetricsList(t *testing.T, version string, in *metricsapi.PodMetricsList) (list, first runtime.Object) {
	t.Helper()
	switch version {
	case "v1":
		out := &metricsv1api.PodMetricsList{}
		if err := metricsv1api.Convert_metrics_PodMetricsList_To_v1_PodMetricsList(in, out, nil); err != nil {
			t.Fatal(err)
		}
		if len(out.Items) == 0 {
			return out, nil
		}
		return out, &out.Items[0]
	case "v1beta1":
		out := &metricsv1beta1api.PodMetricsList{}
		if err := metricsv1beta1api.Convert_metrics_PodMetricsList_To_v1beta1_PodMetricsList(in, out, nil); err != nil {
			t.Fatal(err)
		}
		if len(out.Items) == 0 {
			return out, nil
		}
		return out, &out.Items[0]
	default:
		t.Fatalf("unsupported metrics API version %q", version)
		return nil, nil
	}
}

func testNodeMetricsData() (*metricsapi.NodeMetricsList, *v1.NodeList) {
	metrics := &metricsapi.NodeMetricsList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "1",
		},
		Items: []metricsapi.NodeMetrics{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
				Window:     metav1.Duration{Duration: time.Minute},
				Usage: v1.ResourceList{
					v1.ResourceCPU:     *resource.NewMilliQuantity(1, resource.DecimalSI),
					"swap":             *resource.NewQuantity(1*(1024*1024), resource.DecimalSI),
					v1.ResourceMemory:  *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
					v1.ResourceStorage: *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node2", ResourceVersion: "11"},
				Window:     metav1.Duration{Duration: time.Minute},
				Usage: v1.ResourceList{
					v1.ResourceCPU:     *resource.NewMilliQuantity(5, resource.DecimalSI),
					"swap":             *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
					v1.ResourceMemory:  *resource.NewQuantity(6*(1024*1024), resource.DecimalSI),
					v1.ResourceStorage: *resource.NewQuantity(7*(1024*1024), resource.DecimalSI),
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node3", ResourceVersion: "11"},
				Window:     metav1.Duration{Duration: time.Minute},
				Usage: v1.ResourceList{
					v1.ResourceCPU:     *resource.NewMilliQuantity(3, resource.DecimalSI),
					"swap":             *resource.NewQuantity(0, resource.DecimalSI),
					v1.ResourceMemory:  *resource.NewQuantity(4*(1024*1024), resource.DecimalSI),
					v1.ResourceStorage: *resource.NewQuantity(5*(1024*1024), resource.DecimalSI),
				},
			},
		},
	}
	nodes := &v1.NodeList{
		ListMeta: metav1.ListMeta{
			ResourceVersion: "15",
		},
		Items: []v1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(10, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(20*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(30*(1024*1024), resource.DecimalSI),
					},
					NodeInfo: v1.NodeSystemInfo{
						Swap: &v1.NodeSwapStatus{
							Capacity: new(int64(10 * (1024 * 1024 * 1024))),
						},
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node2", ResourceVersion: "11"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(50, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(60*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(70*(1024*1024), resource.DecimalSI),
					},
					NodeInfo: v1.NodeSystemInfo{
						Swap: &v1.NodeSwapStatus{
							Capacity: new(int64(20 * (1024 * 1024 * 1024))),
						},
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "node3", ResourceVersion: "11"},
				Status: v1.NodeStatus{
					Allocatable: v1.ResourceList{
						v1.ResourceCPU:     *resource.NewMilliQuantity(30, resource.DecimalSI),
						v1.ResourceMemory:  *resource.NewQuantity(40*(1024*1024), resource.DecimalSI),
						v1.ResourceStorage: *resource.NewQuantity(50*(1024*1024), resource.DecimalSI),
					},
					NodeInfo: v1.NodeSystemInfo{
						Swap: &v1.NodeSwapStatus{
							Capacity: new(int64(30 * (1024 * 1024 * 1024))),
						},
					},
				},
			},
		},
	}
	return metrics, nodes
}
