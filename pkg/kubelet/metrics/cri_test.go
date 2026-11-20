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

package metrics

import (
	"context"
	"strings"
	"testing"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	crimetrics "k8s.io/cri-client/pkg/metrics"
	"k8s.io/kubernetes/pkg/features"
)

func TestCRIStreamingMetrics(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CRIListStreaming, true)
	CRIListStreamingTotal.Reset()
	t.Cleanup(CRIListStreamingTotal.Reset)

	RegisterCRIStreamingMetrics()
	crimetrics.ListStreamingTotal.Increment(context.Background(), "list_containers", "success")
	crimetrics.ListStreamingTotal.Increment(context.Background(), "list_containers", "error")
	crimetrics.ListStreamingTotal.Increment(context.Background(), "list_images", "fallback")

	// Observations before kubelet's general metric registration must be preserved.
	RegisterCRIStreamingMetrics()
	Register()

	want := `
# HELP kubelet_cri_list_streaming_total [ALPHA] Cumulative number of CRI list streaming RPC attempts by operation type and result (success, error, fallback).
# TYPE kubelet_cri_list_streaming_total counter
kubelet_cri_list_streaming_total{operation_type="list_containers",result="success"} 1
kubelet_cri_list_streaming_total{operation_type="list_containers",result="error"} 1
kubelet_cri_list_streaming_total{operation_type="list_images",result="fallback"} 1
`
	if err := testutil.GatherAndCompare(GetGather(), strings.NewReader(want),
		KubeletSubsystem+"_"+CRIListStreamingTotalKey); err != nil {
		t.Fatal(err)
	}
}
