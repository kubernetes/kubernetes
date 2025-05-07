/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func init() {
	// Metrics must be registered before they can be set and retrieved.
	utilfeature.DefaultMutableFeatureGate.SetFromMap(map[string]bool{string(features.DynamicResourceAllocation): true})
	metrics.Register()
}

func testClaimsInUseMetric(tCtx ktesting.TContext, expectedMetric string) {
	tCtx.Helper()
	err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(expectedMetric), metrics.DRAResourceClaimsInUse.FQName())
	if err != nil {
		tCtx.Error(err)
	}
}
