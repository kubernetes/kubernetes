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

package version

import (
	apimachineryversion "k8s.io/kubernetes/staging/src/k8s.io/apimachinery/version"
)

// KubernetesRef is the git sha1 reference this k8s.io/client-go was
// exported from. As long as it is in-tree (= unexported), it is just "HEAD".
// The value is replaced during export to the external repositories.
const KubernetesRef = "HEAD"

// compatibleApiMachineryKubernetesRef is a git sha1 reference (or "HEAD") which
// is known to be compatible with this k8s.io/client-go version. If the repo publish
// script in Kubernetes CI update client-go, but not apimachinery, this value is set
// to the "old" apimachinery git sha1 reference.
const compatibleApiMachineryKubernetesRef = "HEAD"

func init() {
	if apimachineryversion.KubernetesRef == KubernetesRef {
		// both repos were exported at the same time
		return
	}
	if apimachineryversion.KubernetesRef == compatibleApiMachineryKubernetesRef {
		// client-go was exported, but apimachinery was not
		return
	}
	if apimachineryversion.CompatibleClientGoKubernetesRef == KubernetesRef {
		// apimachinery was exported, but client-go was not
		return
	}

	panic("k8s.io/client-go and k8s.io/apimachinery version mismatch. Update both to the same branch or tag.")
}
