/*
Copyright 2021 The Kubernetes Authors.

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

package resourcequota

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledFields removes disabled fields from the ResourceQuota spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources
// containing a ResourceQuota spec.
func DropDisabledFields(newResSpec, oldResSpec *api.ResourceQuotaSpec) {
	if utilfeature.DefaultFeatureGate.Enabled(features.PodAffinityNamespaceSelector) || newResSpec == nil {
		return
	}
	if len(newResSpec.Scopes) != 0 && !crossNamespaceScopeInUse(oldResSpec) {
		var scopes []api.ResourceQuotaScope
		for _, s := range newResSpec.Scopes {
			if s != api.ResourceQuotaScopeCrossNamespacePodAffinity {
				scopes = append(scopes, s)
			}
		}
		newResSpec.Scopes = scopes
	}
	if newResSpec.ScopeSelector != nil && len(newResSpec.ScopeSelector.MatchExpressions) != 0 &&
		!crossNamespaceScopeSelectorInUse(oldResSpec) {
		var expressions []api.ScopedResourceSelectorRequirement
		for _, e := range newResSpec.ScopeSelector.MatchExpressions {
			if e.ScopeName != api.ResourceQuotaScopeCrossNamespacePodAffinity {
				expressions = append(expressions, e)
			}
		}
		newResSpec.ScopeSelector.MatchExpressions = expressions
	}
}

func crossNamespaceScopeInUse(resSpec *api.ResourceQuotaSpec) bool {
	if resSpec == nil {
		return false
	}
	for _, s := range resSpec.Scopes {
		if s == api.ResourceQuotaScopeCrossNamespacePodAffinity {
			return true
		}
	}
	return false
}

func crossNamespaceScopeSelectorInUse(resSpec *api.ResourceQuotaSpec) bool {
	if resSpec == nil || resSpec.ScopeSelector == nil {
		return false
	}
	for _, m := range resSpec.ScopeSelector.MatchExpressions {
		if m.ScopeName == api.ResourceQuotaScopeCrossNamespacePodAffinity {
			return true
		}
	}
	return false
}
