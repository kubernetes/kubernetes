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

// Package corev1 defines functions which should satisfy one of the following:
//
// - Be used by more than one core component (kube-scheduler, kubelet, kube-apiserver, etc.)
// - Be used by a core component and another kubernetes project (cluster-autoscaler, descheduler)
//
// And be a scheduling feature.
package corev1
