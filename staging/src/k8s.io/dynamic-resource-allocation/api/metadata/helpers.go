/*
Copyright The Kubernetes Authors.

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

package metadata

import "path/filepath"

// MetadataFileName returns the metadata file name for the given driver.
func MetadataFileName(driverName string) string {
	return driverName + MetadataFileSuffix
}

// ResourceClaimFilePath returns the in-container path for a directly
// referenced ResourceClaim's metadata file.
func ResourceClaimFilePath(driverName, claimName, requestName string) string {
	return filepath.Join(ContainerDir, ResourceClaimsSubDir, claimName, requestName, MetadataFileName(driverName))
}

// ResourceClaimTemplateFilePath returns the in-container path for a
// template-generated claim's metadata file.
func ResourceClaimTemplateFilePath(driverName, podClaimName, requestName string) string {
	return filepath.Join(ContainerDir, ResourceClaimTemplatesSubDir, podClaimName, requestName, MetadataFileName(driverName))
}
