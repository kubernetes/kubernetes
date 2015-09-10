/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util"
)

func addDefaultingFuncs() {
	api.Scheme.AddDefaultingFuncs(
		func(obj *APIVersion) {
			if len(obj.APIGroup) == 0 {
				obj.APIGroup = "experimental"
			}
		},
		func(obj *DaemonSet) {
			var labels map[string]string
			if obj.Spec.Template != nil {
				labels = obj.Spec.Template.Labels
			}
			// TODO: support templates defined elsewhere when we support them in the API
			if labels != nil {
				if len(obj.Spec.Selector) == 0 {
					obj.Spec.Selector = labels
				}
				if len(obj.Labels) == 0 {
					obj.Labels = labels
				}
			}
		},
		func(obj *Deployment) {
			// Set DeploymentSpec.Replicas to 1 if it is not set.
			if obj.Spec.Replicas == nil {
				obj.Spec.Replicas = new(int)
				*obj.Spec.Replicas = 1
			}
			strategy := &obj.Spec.Strategy
			// Set default DeploymentType as RollingUpdate.
			if strategy.Type == "" {
				strategy.Type = DeploymentRollingUpdate
			}
			if strategy.Type == DeploymentRollingUpdate {
				if strategy.RollingUpdate == nil {
					rollingUpdate := RollingUpdateDeployment{}
					strategy.RollingUpdate = &rollingUpdate
				}
				if strategy.RollingUpdate.MaxUnavailable == nil {
					// Set default MaxUnavailable as 1 by default.
					maxUnavailable := util.NewIntOrStringFromInt(1)
					strategy.RollingUpdate.MaxUnavailable = &maxUnavailable
				}
				if strategy.RollingUpdate.MaxSurge == nil {
					// Set default MaxSurge as 1 by default.
					maxSurge := util.NewIntOrStringFromInt(1)
					strategy.RollingUpdate.MaxSurge = &maxSurge
				}
			}
			if obj.Spec.UniqueLabelKey == nil {
				obj.Spec.UniqueLabelKey = new(string)
				*obj.Spec.UniqueLabelKey = "deployment.kubernetes.io/podTemplateHash"
			}
		},
	)
}
