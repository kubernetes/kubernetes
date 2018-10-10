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

package volumes

const (
	// Name of a volume in external cloud that is being provisioned and thus
	// should be ignored by rest of Kubernetes.
	ProvisionedVolumeName = "placeholder-for-provisioning"
)

// NewDeletedVolumeInUseError returns a new instance of DeletedVolumeInUseError
// error.
func NewDeletedVolumeInUseError(message string) error {
	return deletedVolumeInUseError(message)
}

type deletedVolumeInUseError string

var _ error = deletedVolumeInUseError("")

func (err deletedVolumeInUseError) Error() string {
	return string(err)
}
