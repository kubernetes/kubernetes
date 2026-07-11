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

package mutating

import (
	"github.com/stretchr/testify/assert"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestFullReinvocation(t *testing.T) {
	key1 := key{PolicyUID: types.NamespacedName{Name: "p1"}, BindingUID: types.NamespacedName{Name: "b1"}}
	key2 := key{PolicyUID: types.NamespacedName{Name: "p2"}, BindingUID: types.NamespacedName{Name: "b2"}}
	key3 := key{PolicyUID: types.NamespacedName{Name: "p3"}, BindingUID: types.NamespacedName{Name: "b3"}}

	cm1v1 := &v1.ConfigMap{Data: map[string]string{"v": "1"}}
	cm1v2 := &v1.ConfigMap{Data: map[string]string{"v": "2"}}

	rc := policyReinvokeContext{}

	// key1 is invoked and it updates the configmap
	rc.SetLastPolicyInvocationOutput(cm1v1)
	rc.RequireReinvokingPreviouslyInvokedPlugins()
	rc.AddReinvocablePolicyToPreviouslyInvoked(key1)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v2))

	// key2 is invoked and it updates the configmap
	rc.SetLastPolicyInvocationOutput(cm1v2)
	rc.RequireReinvokingPreviouslyInvokedPlugins()
	rc.AddReinvocablePolicyToPreviouslyInvoked(key2)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))

	// key3 is invoked but it does not change anything
	rc.AddReinvocablePolicyToPreviouslyInvoked(key3)

	assert.False(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v2))

	// key1 is reinvoked
	assert.True(t, rc.ShouldReinvoke(key1))
	rc.AddReinvocablePolicyToPreviouslyInvoked(key1)
	rc.SetLastPolicyInvocationOutput(cm1v1)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v2))
	rc.RequireReinvokingPreviouslyInvokedPlugins()

	// key2 is reinvoked
	assert.True(t, rc.ShouldReinvoke(key2))
	rc.AddReinvocablePolicyToPreviouslyInvoked(key2)
	rc.SetLastPolicyInvocationOutput(cm1v2)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))
	rc.RequireReinvokingPreviouslyInvokedPlugins()

	// key3 is reinvoked, because the reinvocations have changed the resource
	assert.True(t, rc.ShouldReinvoke(key3))
}

func TestPartialReinvocation(t *testing.T) {
	key1 := key{PolicyUID: types.NamespacedName{Name: "p1"}, BindingUID: types.NamespacedName{Name: "b1"}}
	key2 := key{PolicyUID: types.NamespacedName{Name: "p2"}, BindingUID: types.NamespacedName{Name: "b2"}}
	key3 := key{PolicyUID: types.NamespacedName{Name: "p3"}, BindingUID: types.NamespacedName{Name: "b3"}}

	cm1v1 := &v1.ConfigMap{Data: map[string]string{"v": "1"}}
	cm1v2 := &v1.ConfigMap{Data: map[string]string{"v": "2"}}

	rc := policyReinvokeContext{}

	// key1 is invoked and it updates the configmap
	rc.SetLastPolicyInvocationOutput(cm1v1)
	rc.RequireReinvokingPreviouslyInvokedPlugins()
	rc.AddReinvocablePolicyToPreviouslyInvoked(key1)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v2))

	// key2 is invoked and it updates the configmap
	rc.SetLastPolicyInvocationOutput(cm1v2)
	rc.RequireReinvokingPreviouslyInvokedPlugins()
	rc.AddReinvocablePolicyToPreviouslyInvoked(key2)

	assert.True(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))

	// key3 is invoked but it does not change anything
	rc.AddReinvocablePolicyToPreviouslyInvoked(key3)

	assert.False(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v2))

	// key1 is reinvoked but does not change anything
	assert.True(t, rc.ShouldReinvoke(key1))

	// key2 is not reinvoked because nothing changed since last invocation
	assert.False(t, rc.ShouldReinvoke(key2))

	// key3 is not reinvoked because nothing changed since last invocation
	assert.False(t, rc.ShouldReinvoke(key3))
}

func TestNoReinvocation(t *testing.T) {
	key1 := key{PolicyUID: types.NamespacedName{Name: "p1"}, BindingUID: types.NamespacedName{Name: "b1"}}
	key2 := key{PolicyUID: types.NamespacedName{Name: "p2"}, BindingUID: types.NamespacedName{Name: "b2"}}
	key3 := key{PolicyUID: types.NamespacedName{Name: "p3"}, BindingUID: types.NamespacedName{Name: "b3"}}

	cm1v1 := &v1.ConfigMap{Data: map[string]string{"v": "1"}}

	rc := policyReinvokeContext{}

	// key1 is invoked and it updates the configmap
	rc.AddReinvocablePolicyToPreviouslyInvoked(key1)
	rc.SetLastPolicyInvocationOutput(cm1v1)

	assert.False(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))

	// key2 is invoked but does not change anything
	rc.AddReinvocablePolicyToPreviouslyInvoked(key2)
	rc.SetLastPolicyInvocationOutput(cm1v1)

	assert.False(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))

	// key3 is invoked but it does not change anything
	rc.AddReinvocablePolicyToPreviouslyInvoked(key3)
	rc.SetLastPolicyInvocationOutput(cm1v1)

	assert.False(t, rc.IsOutputChangedSinceLastPolicyInvocation(cm1v1))

	// no keys are reinvoked
	assert.False(t, rc.ShouldReinvoke(key1))
	assert.False(t, rc.ShouldReinvoke(key2))
	assert.False(t, rc.ShouldReinvoke(key3))

}
