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

// KubernetesRef is the git sha1 reference of k8s.io/kubernetes this
// k8s.io/apimachinery version was exported from. As long as it is in-tree (= unexported),
// it is just "HEAD". The value is replaced during export to the external repositories.
const KubernetesRef = "HEAD"

// CompatibleClientGoKubernetesRef is an optional clientgoversion.KubernetesRef or "".
// The former means that the k8s.io/client-go export from that Kubernetes version is
// known to be compatible with this apimachinery version.
//
// Normally, the value is "". But if the repo publish script in Kubernetes CI has
// to update apimachinery but not client-go, the "old" clientgoversion.KubernetesRef
// (which did not change during that export) is set here. When both repositories are
// updated again at the same time, this value can be reset to "".
var CompatibleClientGoKubernetesRef = ""
