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

// Package kubeletplugin provides helper functions for running a dynamic
// resource allocation kubelet plugin.
//
// A DRA driver using this package can be deployed as a DaemonSet on suitable
// nodes. Node labeling, for example through NFD
// (https://github.com/kubernetes-sigs/node-feature-discovery), can be used
// to run the driver only on nodes which have the necessary hardware.
//
// The service account of the DaemonSet must have sufficient RBAC permissions
// to read ResourceClaims and to create and update ResourceSlices, if
// the driver intends to publish per-node ResourceSlices. It is good
// security practice (but not required) to limit access to ResourceSlices
// associated with the node a specific Pod is running on. This can be done
// with a Validating Admission Policy (VAP). For more information,
// see the deployment of the DRA example driver
// (https://github.com/kubernetes-sigs/dra-example-driver/tree/main/deployments/helm/dra-example-driver/templates).
//
// Traditionally, the kubelet has not supported rolling updates of plugins.
// Therefore the DaemonSet must not set `maxSurge` to a value larger than
// zero. With the default `maxSurge: 0`, updating the DaemonSet of the driver
// will first shut down the old driver Pod, then start the replacement.
//
// This leads to a short downtime for operations that need the driver:
//   - Pods cannot start unless the claims they depend on were already
//     prepared for use.
//   - Cleanup after the last pod which used a claim gets delayed
//     until the driver is available again. The pod is not marked
//     as terminated. This prevents reusing the resources used by
//     the pod for other pods.
//   - Running pods are *not* affected as far as Kubernetes is
//     concerned. However, a DRA driver might provide required runtime
//     services. Vendors need to document this.
//
// Note that the second point also means that draining a node should
// first evict normal pods, then the driver DaemonSet Pod.
//
// Starting with Kubernetes 1.33, the kubelet supports rolling updates
// such that old and new Pod run at the same time for a short while
// and hand over work gracefully, with no downtime.
// However, there is no mechanism for determining in advance whether
// the node the DaemonSet runs on supports that. Trying
// to do a rolling update with a kubelet which does not support it yet
// will fail because shutting down the old Pod unregisters the driver
// even though the new Pod is running. See https://github.com/kubernetes/kubernetes/pull/129832
// for details (TODO: link to doc after merging instead).
//
// A DRA driver can either require 1.33 as minimal Kubernetes version or
// provide two variants of its DaemonSet deployment. In the variant with
// support for rolling updates, `maxSurge` can be set to a non-zero
// value. Administrators have to be careful about running the right variant.
package kubeletplugin
