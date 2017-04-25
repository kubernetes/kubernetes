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

package federation

// FederationNameAnnotation is the annotation which holds the name of
// the federation that a federation control plane component is associated
// with. It must be applied to all the API types that represent that federations
// control plane's components in the host cluster and in joining clusters.
const FederationNameAnnotation = "federation.alpha.kubernetes.io/federation-name"

// ClusterNameAnnotation is the annotation which holds the name of
// the cluster that an object is associated with. If the object is
// not associated with any cluster, then this annotation is not
// required.
const ClusterNameAnnotation = "federation.alpha.kubernetes.io/cluster-name"
