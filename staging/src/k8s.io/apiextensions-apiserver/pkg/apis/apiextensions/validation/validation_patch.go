/*
Copyright 2022 The KCP Authors.

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

package validation

import "k8s.io/apimachinery/pkg/util/sets"

var kubernetesAPIGroups = sets.NewString(
	"admissionregistration.k8s.io",
	"apps",
	"authentication.k8s.io",
	"authorization.k8s.io",
	"autoscaling",
	"batch",
	"certificates.k8s.io",
	"coordination.k8s.io",
	"discovery.k8s.io",
	"events.k8s.io",
	"extensions",
	"flowcontrol.apiserver.k8s.io",
	"imagepolicy.k8s.io",
	"policy",
	"rbac.authorization.k8s.io",
	"scheduling.k8s.io",
	"storage.k8s.io",
	"storagemigration.k8s.io",
	"",
)

func isKubernetesAPIGroup(group string) bool {
	return kubernetesAPIGroups.Has(group)
}
