/*
Copyright 2019 The Kubernetes Authors.

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

package volume

const (
	// ProvisionedVolumeName is the name of a volume in an external cloud
	// that is being provisioned and thus should be ignored by rest of Kubernetes.
	ProvisionedVolumeName = "placeholder-for-provisioning"

	// LabelMultiZoneDelimiter separates zones for volumes
	LabelMultiZoneDelimiter = "__"
)
