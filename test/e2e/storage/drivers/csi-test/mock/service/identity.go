package service

import (
	"golang.org/x/net/context"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/golang/protobuf/ptypes/wrappers"
)

func (s *service) GetPluginInfo(
	ctx context.Context,
	req *csi.GetPluginInfoRequest) (
	*csi.GetPluginInfoResponse, error) {

	return &csi.GetPluginInfoResponse{
		Name:          s.config.DriverName,
		VendorVersion: VendorVersion,
		Manifest:      Manifest,
	}, nil
}

func (s *service) Probe(
	ctx context.Context,
	req *csi.ProbeRequest) (
	*csi.ProbeResponse, error) {

	return &csi.ProbeResponse{
		Ready: &wrappers.BoolValue{Value: true},
	}, nil
}

func (s *service) GetPluginCapabilities(
	ctx context.Context,
	req *csi.GetPluginCapabilitiesRequest) (
	*csi.GetPluginCapabilitiesResponse, error) {

	volExpType := csi.PluginCapability_VolumeExpansion_ONLINE

	if s.config.DisableOnlineExpansion {
		volExpType = csi.PluginCapability_VolumeExpansion_OFFLINE
	}

	capabilities := []*csi.PluginCapability{
		{
			Type: &csi.PluginCapability_Service_{
				Service: &csi.PluginCapability_Service{
					Type: csi.PluginCapability_Service_CONTROLLER_SERVICE,
				},
			},
		},
		{
			Type: &csi.PluginCapability_VolumeExpansion_{
				VolumeExpansion: &csi.PluginCapability_VolumeExpansion{
					Type: volExpType,
				},
			},
		},
	}

	if s.config.EnableTopology {
		capabilities = append(capabilities,
			&csi.PluginCapability{
				Type: &csi.PluginCapability_Service_{
					Service: &csi.PluginCapability_Service{
						Type: csi.PluginCapability_Service_VOLUME_ACCESSIBILITY_CONSTRAINTS,
					},
				},
			})
	}

	return &csi.GetPluginCapabilitiesResponse{
		Capabilities: capabilities,
	}, nil
}
