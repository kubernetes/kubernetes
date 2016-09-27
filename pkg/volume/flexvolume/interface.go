package flexvolume

import (
	"k8s.io/kubernetes/pkg/volume/flexvolume/api"
)

type Interface interface {
	// Probe Plugin
	//
	// This call probes the plugin for it’s capabilities and supported options.
	// Supported options are used to validate and reject pod spec.
	// This is executed from both the Controller-manager and Kubelet.
	Probe() (*api.FlexVolumeDriver, error)

	// Provision/Create a volume
	//
	// This call creates a volume. It is executed from Controller-manager.
	Create(volume *api.FlexVolume) (*api.FlexVolume, error)

	// Delete a volume
	//
	// This call deletes a volume. It is executed from Controller-manager.
	Delete(volume *api.FlexVolume) error

	// Attach a volume
	//
	// This call attaches a volume to a remote host or local host.
	// It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.
	// This is only valid for drivers which support attach & detach.
	//
	// TODO specific struct instead
	Attach(volume *api.FlexVolume) (*api.FlexVolumeAttachement, error)

	// Wait for attach
	//
	// This call wait for a volume to be attached on the local host.
	// It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.
	// This is only valid for drivers which support attach & detach.
	//
	// TODO specific struct instead
	WaitForAttach(volume *api.FlexVolume) (*api.FlexVolumeAttachement, error)

	// Detach a volume
	//
	// This call detaches a volume to a remote host or local host.
	// It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.
	// This call only valid for drivers which support attach & detach.
	//
	// TODO specific struct instead
	Detach(volume *api.FlexVolume) error

	// Wait for detach
	//
	// This call wait for a volume to be detached from the local host.
	// It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.
	// This is only valid for drivers which support attach & detach.
	//
	// TODO specific struct instead
	WaitForDetach(volume *api.FlexVolume) error

	// Mount a volume
	//
	// This call mounts a volume on the node. It is executed from Kubelet.
	// This is only valid for plugins which do not support attach/detach to a node.
	//
	// TODO specific struct instead
	Mount(volume *api.FlexVolume) (*api.FlexVolumeMount, error)

	// Unmount a volume
	//
	// This call unmounts a volume on the node.
	// It is executed from Kubelet.
	// This is only valid for plugins which do not support attach/detach to a node.
	// Example: NFS/CIFS.
	Unmount(mount *api.FlexVolumeMount) error

	// Metrics call
	//
	// This call gets the metrics of a volume. It is executed from Kubelet.
	Metrics(mount *api.FlexVolumeMount) (*api.FlexVolumeMetrics, error)
}
