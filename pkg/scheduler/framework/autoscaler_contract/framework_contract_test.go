/*
Copyright 2022 The Kubernetes Authors.

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

// This file helps to detect whether we changed the interface of Framework.
// This is important for projects under `github.com/kubernetes*` orgs
// who import the Framework to not breaking their codes, e.g. cluster-autoscaler.

package contract

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

type frameworkContract interface {
	RunPreFilterPlugins(ctx context.Context, state fwk.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *fwk.Status, sets.Set[string])
	RunFilterPlugins(context.Context, fwk.CycleState, *v1.Pod, *framework.NodeInfo) *fwk.Status
	RunReservePluginsReserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status
}

func TestFrameworkContract(t *testing.T) {
	var f framework.Framework
	var c frameworkContract = f
	assert.Nil(t, c)
}

func TestNewFramework(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	var f interface{}
	if f, _ = runtime.NewFramework(ctx, nil, nil); f != nil {
		_, ok := f.(framework.Framework)
		assert.True(t, ok)
	}
}

func TestNewCycleState(t *testing.T) {
	var state interface{} = framework.NewCycleState()
	_, ok := state.(*framework.CycleState)
	assert.True(t, ok)
}
