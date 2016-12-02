/*
Copyright 2016 The Kubernetes Authors.

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

package qos

import "k8s.io/kubernetes/pkg/api/v1"

// QOSClass defines the supported qos classes of Pods/Containers.
type QOSClass string

const (
	// Guaranteed is the Guaranteed qos class.
	Guaranteed v1.PodQOSClass = v1.PodQOSGuaranteed
	// Burstable is the Burstable qos class.
	Burstable v1.PodQOSClass = v1.PodQOSBurstable
	// BestEffort is the BestEffort qos class.
	BestEffort v1.PodQOSClass = v1.PodQOSBestEffort
)
