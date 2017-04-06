/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	extensions_v1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// SetDaemonSetDefaults sets defaults for a daemonset retrieved from a
// member cluster to allow backwards-compatible comparison.
func SetDaemonSetDefaults(daemonSet *extensions_v1.DaemonSet, templateGeneration int64) {
	extensions_v1.SetDefaults_DaemonSet(daemonSet)
	setPodSpecDefaults(&daemonSet.Spec.Template.Spec)
	// TemplateGeneration will have a non-zero value in 1.6 clusters
	// but be 0 in previous versions.  Since it can only change when
	// the template changes, it isn't useful to compare, so just set
	// it to an explicit value.
	daemonSet.Spec.TemplateGeneration = templateGeneration
}
