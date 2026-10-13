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

package internalbootstrap

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	flowcontrol "k8s.io/api/flowcontrol/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
)

func TestBootstrapConfigurationWithDefaulted(t *testing.T) {
	scheme := NewAPFScheme()

	bootstrapFlowSchemas := make([]*flowcontrol.FlowSchema, 0)
	bootstrapFlowSchemas = append(bootstrapFlowSchemas, bootstrap.MandatoryFlowSchemas...)
	bootstrapFlowSchemas = append(bootstrapFlowSchemas, bootstrap.SuggestedFlowSchemas...)
	for _, original := range bootstrapFlowSchemas {
		t.Run(fmt.Sprintf("FlowSchema/%s", original.Name), func(t *testing.T) {
			defaulted := original.DeepCopyObject().(*flowcontrol.FlowSchema)
			scheme.Default(defaulted)
			if apiequality.Semantic.DeepEqual(original, defaulted) {
				t.Logf("Defaulting makes no change to FlowSchema: %q", original.Name)
				return
			}
			t.Errorf("Expected defaulting to not change FlowSchema: %q, diff: %s", original.Name, cmp.Diff(original, defaulted))
		})
	}

	bootstrapPriorityLevels := make([]*flowcontrol.PriorityLevelConfiguration, 0)
	bootstrapPriorityLevels = append(bootstrapPriorityLevels, bootstrap.MandatoryPriorityLevelConfigurations...)
	bootstrapPriorityLevels = append(bootstrapPriorityLevels, bootstrap.SuggestedPriorityLevelConfigurations...)
	for _, original := range bootstrapPriorityLevels {
		t.Run(fmt.Sprintf("PriorityLevelConfiguration/%s", original.Name), func(t *testing.T) {
			defaulted := original.DeepCopyObject().(*flowcontrol.PriorityLevelConfiguration)
			scheme.Default(defaulted)
			if apiequality.Semantic.DeepEqual(original, defaulted) {
				t.Logf("Defaulting makes no change to PriorityLevelConfiguration: %q", original.Name)
				return
			}
			t.Errorf("Expected defaulting to not change PriorityLevelConfiguration: %q, diff: %s", original.Name, cmp.Diff(original, defaulted))
		})
	}
}
