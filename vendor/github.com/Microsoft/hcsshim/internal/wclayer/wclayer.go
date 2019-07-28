package wclayer

import "github.com/Microsoft/hcsshim/internal/guid"

//go:generate go run ../../mksyscall_windows.go -output zsyscall_windows.go wclayer.go

//sys activateLayer(info *driverInfo, id string) (hr error) = vmcompute.ActivateLayer?
//sys copyLayer(info *driverInfo, srcId string, dstId string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.CopyLayer?
//sys createLayer(info *driverInfo, id string, parent string) (hr error) = vmcompute.CreateLayer?
//sys createSandboxLayer(info *driverInfo, id string, parent uintptr, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.CreateSandboxLayer?
//sys expandSandboxSize(info *driverInfo, id string, size uint64) (hr error) = vmcompute.ExpandSandboxSize?
//sys deactivateLayer(info *driverInfo, id string) (hr error) = vmcompute.DeactivateLayer?
//sys destroyLayer(info *driverInfo, id string) (hr error) = vmcompute.DestroyLayer?
//sys exportLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.ExportLayer?
//sys getLayerMountPath(info *driverInfo, id string, length *uintptr, buffer *uint16) (hr error) = vmcompute.GetLayerMountPath?
//sys getBaseImages(buffer **uint16) (hr error) = vmcompute.GetBaseImages?
//sys importLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.ImportLayer?
//sys layerExists(info *driverInfo, id string, exists *uint32) (hr error) = vmcompute.LayerExists?
//sys nameToGuid(name string, guid *_guid) (hr error) = vmcompute.NameToGuid?
//sys prepareLayer(info *driverInfo, id string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.PrepareLayer?
//sys unprepareLayer(info *driverInfo, id string) (hr error) = vmcompute.UnprepareLayer?
//sys processBaseImage(path string) (hr error) = vmcompute.ProcessBaseImage?
//sys processUtilityImage(path string) (hr error) = vmcompute.ProcessUtilityImage?

//sys grantVmAccess(vmid string, filepath string) (hr error) = vmcompute.GrantVmAccess?

type _guid = guid.GUID
