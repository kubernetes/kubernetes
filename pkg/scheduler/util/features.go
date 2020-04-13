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

package util

import "k8s.io/component-base/featuregate"

// NOTE: constants in this file are copied from pkg/features package to avoid external scheduler dependency

const (
	// owner: @egernst
	// alpha: v1.16
	// beta: v1.18
	//
	// Enables PodOverhead, for accounting pod overheads which are specific to a given RuntimeClass
	PodOverhead featuregate.Feature = "PodOverhead"

	// owner: @aveshagarwal
	// alpha: v1.9
	//
	// Enable resource limits priority function
	ResourceLimitsPriorityFunction featuregate.Feature = "ResourceLimitsPriorityFunction"

	// owner: @Huang-Wei
	// beta: v1.18
	//
	// Schedule pods evenly across available topology domains.
	EvenPodsSpread featuregate.Feature = "EvenPodsSpread"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the in-tree storage to CSI Plugin migration feature.
	CSIMigration featuregate.Feature = "CSIMigration"

	// owner: @davidz627
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the GCE PD in-tree driver to GCE CSI Driver migration feature.
	CSIMigrationGCE featuregate.Feature = "CSIMigrationGCE"

	// owner: @leakingtapan
	// alpha: v1.14
	// beta: v1.17
	//
	// Enables the AWS EBS in-tree driver to AWS EBS CSI Driver migration feature.
	CSIMigrationAWS featuregate.Feature = "CSIMigrationAWS"

	// owner: @andyzhangx
	// alpha: v1.15
	//
	// Enables the Azure Disk in-tree driver to Azure Disk Driver migration feature.
	CSIMigrationAzureDisk featuregate.Feature = "CSIMigrationAzureDisk"

	// owner: @adisky
	// alpha: v1.14
	// beta: v1.18
	//
	// Enables the OpenStack Cinder in-tree driver to OpenStack Cinder CSI Driver migration feature.
	CSIMigrationOpenStack featuregate.Feature = "CSIMigrationOpenStack"

	// owner: @ravig
	// alpha: v1.11
	//
	// Include volume count on node to be considered for balanced resource allocation while scheduling.
	// A node which has closer cpu,memory utilization and volume count is favoured by scheduler
	// while making decisions.
	BalanceAttachedNodeVolumes featuregate.Feature = "BalanceAttachedNodeVolumes"

	// owner: @verult
	// alpha: v1.12
	// beta:  v1.14
	// ga:    v1.17
	// Enable all logic related to the CSINode API object in storage.k8s.io
	CSINodeInfo featuregate.Feature = "CSINodeInfo"

	// owner: @denkensk
	// alpha: v1.15
	//
	// Enables NonPreempting option for priorityClass and pod.
	NonPreemptingPriority featuregate.Feature = "NonPreemptingPriority"

	// owner: @mortent
	// alpha: v1.3
	// beta:  v1.5
	//
	// Enable all logic related to the PodDisruptionBudget API object in policy
	PodDisruptionBudget featuregate.Feature = "PodDisruptionBudget"
)
