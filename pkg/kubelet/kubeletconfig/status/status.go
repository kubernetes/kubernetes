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

package status

import (
	apiv1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
)

const (
	// LoadError indicates that the Kubelet failed to load the config checkpoint
	LoadError = "failed to load config, see Kubelet log for details"
	// ValidateError indicates that the Kubelet failed to validate the config checkpoint
	ValidateError = "failed to validate config, see Kubelet log for details"
	// AllNilSubfieldsError is used when no subfields are set
	// This could happen in the case that an old client tries to read an object from a newer API server with a set subfield it does not know about
	AllNilSubfieldsError = "invalid NodeConfigSource, exactly one subfield must be non-nil, but all were nil"
	// DownloadError is used when the download fails, e.g. due to network issues
	DownloadError = "failed to download config, see Kubelet log for details"
	// InternalError indicates that some internal error happened while trying to sync config, e.g. filesystem issues
	InternalError = "internal failure, see Kubelet log for details"

	// SyncErrorFmt is used when the system couldn't sync the config, due to a malformed Node.Spec.ConfigSource, a download failure, etc.
	SyncErrorFmt = "failed to sync: %s"
)

// NodeConfigStatus represents Node.Status.Config
type NodeConfigStatus interface {
	// SetActive sets the active source in the status
	SetActive(source *apiv1.NodeConfigSource)
	// SetAssigned sets the assigned source in the status
	SetAssigned(source *apiv1.NodeConfigSource)
	// SetLastKnownGood sets the last-known-good source in the status
	SetLastKnownGood(source *apiv1.NodeConfigSource)
	// SetError sets the error associated with the status
	SetError(err string)
	// SetErrorOverride sets an error that overrides the base error set by SetError.
	// If the override is set to the empty string, the base error is reported in
	// the status, otherwise the override is reported.
	SetErrorOverride(err string)
	// Sync patches the current status into the Node identified by `nodeName` if an update is pending
	Sync(client clientset.Interface, nodeName string)
}
