/*
Copyright 2018 The Kubernetes Authors.

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

// This file is a common place holder for volume limit utility constants
// shared between volume package and scheduler

const (
	// EBSVolumeLimitKey resource name that will store volume limits for EBS
	EBSVolumeLimitKey = "attachable-volumes-aws-ebs"
	// AzureVolumeLimitKey stores resource name that will store volume limits for Azure
	AzureVolumeLimitKey = "attachable-volumes-azure-disk"
	// GCEVolumeLimitKey stores resource name that will store volume limits for GCE node
	GCEVolumeLimitKey = "attachable-volumes-gce-pd"
)
