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

package kubeletplugin

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

// MetadataUpdater allows drivers to update device metadata after
// PrepareResourceClaims has returned. This is used by network DRA drivers
// whose device configuration (IP addresses, interface names, MAC addresses)
// only becomes known after CNI runs during RunPodSandbox.
//
// A typical usage pattern:
//  1. During PrepareResourceClaims, the driver returns devices without
//     Metadata (nil). The framework writes a metadata file containing only
//     the structural claim/device information and creates a CDI bind-mount.
//  2. From an NRI RunPodSandbox hook (after CNI), the driver calls
//     UpdateRequestMetadata with the now-known network data.
//  3. The framework overwrites the metadata file with the complete data
//     and increments the generation number.
//
// The [Helper] returned by [Start] implements this interface when device
// metadata is enabled via [EnableDeviceMetadata].
type MetadataUpdater interface {
	// UpdateRequestMetadata overwrites the metadata file for a specific
	// request in a prepared claim. It validates that the claim was prepared
	// by this driver and increments the metadata generation number.
	//
	// The devices slice should contain the same devices that were in the
	// original PrepareResult for this request, now with their Metadata
	// field populated.
	//
	// Returns an error if device metadata is not enabled, the claim was
	// not prepared by this driver, or the request name is not recognized.
	UpdateRequestMetadata(
		ctx context.Context,
		claimNamespace, claimName string,
		claimUID types.UID,
		requestName string,
		devices []Device,
	) error
}

// Compile-time check that Helper implements MetadataUpdater.
var _ MetadataUpdater = &Helper{}

// UpdateRequestMetadata implements [MetadataUpdater].
func (d *Helper) UpdateRequestMetadata(
	ctx context.Context,
	claimNamespace, claimName string,
	claimUID types.UID,
	requestName string,
	devices []Device,
) error {
	if d.metadataWriter == nil {
		return fmt.Errorf("device metadata is not enabled")
	}
	ref := claimRef{
		namespace: claimNamespace,
		name:      claimName,
		uid:       claimUID,
	}
	return d.metadataWriter.updateRequestMetadata(ref, requestName, devices)
}
