/*
Copyright 2021 The Kubernetes Authors.

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

package service

import (
	"fmt"
	"path"
	"reflect"
	"strconv"

	"github.com/container-storage-interface/spec/lib/go/csi"
	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/klog/v2"
)

const (
	MaxStorageCapacity = tib
	ReadOnlyKey        = "readonly"
)

func (s *service) CreateVolume(
	ctx context.Context,
	req *csi.CreateVolumeRequest) (
	*csi.CreateVolumeResponse, error) {

	if len(req.Name) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume Name cannot be empty")
	}
	if req.VolumeCapabilities == nil {
		return nil, status.Error(codes.InvalidArgument, "Volume Capabilities cannot be empty")
	}

	// Check to see if the volume already exists.
	if i, v := s.findVolByName(ctx, req.Name); i >= 0 {
		// Requested volume name already exists, need to check if the existing volume's
		// capacity is more or equal to new request's capacity.
		if v.GetCapacityBytes() < req.GetCapacityRange().GetRequiredBytes() {
			return nil, status.Error(codes.AlreadyExists,
				fmt.Sprintf("Volume with name %s already exists", req.GetName()))
		}
		return &csi.CreateVolumeResponse{Volume: &v}, nil
	}

	// If no capacity is specified then use 100GiB
	capacity := gib100
	if cr := req.CapacityRange; cr != nil {
		if rb := cr.RequiredBytes; rb > 0 {
			capacity = rb
		}
		if lb := cr.LimitBytes; lb > 0 {
			capacity = lb
		}
	}
	// Check for maximum available capacity
	if capacity >= MaxStorageCapacity {
		return nil, status.Errorf(codes.OutOfRange, "Requested capacity %d exceeds maximum allowed %d", capacity, MaxStorageCapacity)
	}

	var v csi.Volume
	// Create volume from content source if provided.
	if req.GetVolumeContentSource() != nil {
		switch req.GetVolumeContentSource().GetType().(type) {
		case *csi.VolumeContentSource_Snapshot:
			sid := req.GetVolumeContentSource().GetSnapshot().GetSnapshotId()
			// Check if the source snapshot exists.
			if snapID, _ := s.snapshots.FindSnapshot("id", sid); snapID >= 0 {
				v = s.newVolumeFromSnapshot(req.Name, capacity, snapID)
			} else {
				return nil, status.Errorf(codes.NotFound, "Requested source snapshot %s not found", sid)
			}
		case *csi.VolumeContentSource_Volume:
			vid := req.GetVolumeContentSource().GetVolume().GetVolumeId()
			// Check if the source volume exists.
			if volID, _ := s.findVolNoLock("id", vid); volID >= 0 {
				v = s.newVolumeFromVolume(req.Name, capacity, volID)
			} else {
				return nil, status.Errorf(codes.NotFound, "Requested source volume %s not found", vid)
			}
		}
	} else {
		v = s.newVolume(req.Name, capacity)
	}

	// Add the created volume to the service's in-mem volume slice.
	s.volsRWL.Lock()
	defer s.volsRWL.Unlock()
	s.vols = append(s.vols, v)
	MockVolumes[v.GetVolumeId()] = Volume{
		VolumeCSI:       v,
		NodeID:          "",
		ISStaged:        false,
		ISPublished:     false,
		StageTargetPath: "",
		TargetPath:      "",
	}

	if hookVal, hookMsg := s.execHook("CreateVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.CreateVolumeResponse{Volume: &v}, nil
}

func (s *service) DeleteVolume(
	ctx context.Context,
	req *csi.DeleteVolumeRequest) (
	*csi.DeleteVolumeResponse, error) {

	s.volsRWL.Lock()
	defer s.volsRWL.Unlock()

	//  If the volume is not specified, return error
	if len(req.VolumeId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID cannot be empty")
	}

	if hookVal, hookMsg := s.execHook("DeleteVolumeStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	// If the volume does not exist then return an idempotent response.
	i, _ := s.findVolNoLock("id", req.VolumeId)
	if i < 0 {
		return &csi.DeleteVolumeResponse{}, nil
	}

	// This delete logic preserves order and prevents potential memory
	// leaks. The slice's elements may not be pointers, but the structs
	// themselves have fields that are.
	copy(s.vols[i:], s.vols[i+1:])
	s.vols[len(s.vols)-1] = csi.Volume{}
	s.vols = s.vols[:len(s.vols)-1]
	klog.V(5).InfoS("mock delete volume", "volumeID", req.VolumeId)

	if hookVal, hookMsg := s.execHook("DeleteVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}
	return &csi.DeleteVolumeResponse{}, nil
}

func (s *service) ControllerPublishVolume(
	ctx context.Context,
	req *csi.ControllerPublishVolumeRequest) (
	*csi.ControllerPublishVolumeResponse, error) {

	if s.config.DisableAttach {
		return nil, status.Error(codes.Unimplemented, "ControllerPublish is not supported")
	}

	if len(req.VolumeId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID cannot be empty")
	}
	if len(req.NodeId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Node ID cannot be empty")
	}
	if req.VolumeCapability == nil {
		return nil, status.Error(codes.InvalidArgument, "Volume Capabilities cannot be empty")
	}

	if req.NodeId != s.nodeID {
		return nil, status.Errorf(codes.NotFound, "Not matching Node ID %s to Mock Node ID %s", req.NodeId, s.nodeID)
	}

	if hookVal, hookMsg := s.execHook("ControllerPublishVolumeStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	s.volsRWL.Lock()
	defer s.volsRWL.Unlock()

	i, v := s.findVolNoLock("id", req.VolumeId)
	if i < 0 {
		return nil, status.Error(codes.NotFound, req.VolumeId)
	}

	// devPathKey is the key in the volume's attributes that is set to a
	// mock device path if the volume has been published by the controller
	// to the specified node.
	devPathKey := path.Join(req.NodeId, "dev")

	// Check to see if the volume is already published.
	if device := v.VolumeContext[devPathKey]; device != "" {
		var volRo bool
		var roVal string
		if ro, ok := v.VolumeContext[ReadOnlyKey]; ok {
			roVal = ro
		}

		if roVal == "true" {
			volRo = true
		} else {
			volRo = false
		}

		// Check if readonly flag is compatible with the publish request.
		if req.GetReadonly() != volRo {
			return nil, status.Error(codes.AlreadyExists, "Volume published but has incompatible readonly flag")
		}

		return &csi.ControllerPublishVolumeResponse{
			PublishContext: map[string]string{
				"device":   device,
				"readonly": roVal,
			},
		}, nil
	}

	// Check attach limit before publishing only if attach limit is set.
	if s.config.AttachLimit > 0 && s.getAttachCount(devPathKey) >= s.config.AttachLimit {
		return nil, status.Errorf(codes.ResourceExhausted, "Cannot attach any more volumes to this node")
	}

	var roVal string
	if req.GetReadonly() {
		roVal = "true"
	} else {
		roVal = "false"
	}

	// Publish the volume.
	device := "/dev/mock"
	v.VolumeContext[devPathKey] = device
	v.VolumeContext[ReadOnlyKey] = roVal
	s.vols[i] = v

	if volInfo, ok := MockVolumes[req.VolumeId]; ok {
		volInfo.ISControllerPublished = true
		MockVolumes[req.VolumeId] = volInfo
	}

	if hookVal, hookMsg := s.execHook("ControllerPublishVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.ControllerPublishVolumeResponse{
		PublishContext: map[string]string{
			"device":   device,
			"readonly": roVal,
		},
	}, nil
}

func (s *service) ControllerUnpublishVolume(
	ctx context.Context,
	req *csi.ControllerUnpublishVolumeRequest) (
	*csi.ControllerUnpublishVolumeResponse, error) {

	if s.config.DisableAttach {
		return nil, status.Error(codes.Unimplemented, "ControllerPublish is not supported")
	}

	if len(req.VolumeId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID cannot be empty")
	}
	nodeID := req.NodeId
	if len(nodeID) == 0 {
		// If node id is empty, no failure as per Spec
		nodeID = s.nodeID
	}

	if req.NodeId != s.nodeID {
		return nil, status.Errorf(codes.NotFound, "Node ID %s does not match to expected Node ID %s", req.NodeId, s.nodeID)
	}

	if hookVal, hookMsg := s.execHook("ControllerUnpublishVolumeStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	s.volsRWL.Lock()
	defer s.volsRWL.Unlock()

	i, v := s.findVolNoLock("id", req.VolumeId)
	if i < 0 {
		// Not an error: a non-existent volume is not published.
		// See also https://github.com/kubernetes-csi/external-attacher/pull/165
		return &csi.ControllerUnpublishVolumeResponse{}, nil
	}

	// devPathKey is the key in the volume's attributes that is set to a
	// mock device path if the volume has been published by the controller
	// to the specified node.
	devPathKey := path.Join(nodeID, "dev")

	// Check to see if the volume is already unpublished.
	if v.VolumeContext[devPathKey] == "" {
		return &csi.ControllerUnpublishVolumeResponse{}, nil
	}

	// Unpublish the volume.
	delete(v.VolumeContext, devPathKey)
	delete(v.VolumeContext, ReadOnlyKey)
	s.vols[i] = v

	if hookVal, hookMsg := s.execHook("ControllerUnpublishVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.ControllerUnpublishVolumeResponse{}, nil
}

func (s *service) ValidateVolumeCapabilities(
	ctx context.Context,
	req *csi.ValidateVolumeCapabilitiesRequest) (
	*csi.ValidateVolumeCapabilitiesResponse, error) {

	if len(req.GetVolumeId()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID cannot be empty")
	}
	if len(req.VolumeCapabilities) == 0 {
		return nil, status.Error(codes.InvalidArgument, req.VolumeId)
	}
	i, _ := s.findVolNoLock("id", req.VolumeId)
	if i < 0 {
		return nil, status.Error(codes.NotFound, req.VolumeId)
	}

	if hookVal, hookMsg := s.execHook("ValidateVolumeCapabilities"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.ValidateVolumeCapabilitiesResponse{
		Confirmed: &csi.ValidateVolumeCapabilitiesResponse_Confirmed{
			VolumeContext:      req.GetVolumeContext(),
			VolumeCapabilities: req.GetVolumeCapabilities(),
			Parameters:         req.GetParameters(),
		},
	}, nil
}

func (s *service) ControllerGetVolume(
	ctx context.Context,
	req *csi.ControllerGetVolumeRequest) (
	*csi.ControllerGetVolumeResponse, error) {

	if hookVal, hookMsg := s.execHook("GetVolumeStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	resp := &csi.ControllerGetVolumeResponse{
		Status: &csi.ControllerGetVolumeResponse_VolumeStatus{
			VolumeCondition: &csi.VolumeCondition{},
		},
	}
	i, v := s.findVolByID(ctx, req.VolumeId)
	if i < 0 {
		resp.Status.VolumeCondition.Abnormal = true
		resp.Status.VolumeCondition.Message = "volume not found"
		return resp, status.Error(codes.NotFound, req.VolumeId)
	}

	resp.Volume = &v
	if !s.config.DisableAttach {
		resp.Status.PublishedNodeIds = []string{
			s.nodeID,
		}
	}

	if hookVal, hookMsg := s.execHook("GetVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return resp, nil
}

func (s *service) ListVolumes(
	ctx context.Context,
	req *csi.ListVolumesRequest) (
	*csi.ListVolumesResponse, error) {

	if hookVal, hookMsg := s.execHook("ListVolumesStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	// Copy the mock volumes into a new slice in order to avoid
	// locking the service's volume slice for the duration of the
	// ListVolumes RPC.
	var vols []csi.Volume
	func() {
		s.volsRWL.RLock()
		defer s.volsRWL.RUnlock()
		vols = make([]csi.Volume, len(s.vols))
		copy(vols, s.vols)
	}()

	var (
		ulenVols      = int32(len(vols))
		maxEntries    = req.MaxEntries
		startingToken int32
	)

	if v := req.StartingToken; v != "" {
		i, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return nil, status.Errorf(
				codes.Aborted,
				"startingToken=%s: %v",
				v, err)
		}
		startingToken = int32(i)
	}

	if startingToken > ulenVols {
		return nil, status.Errorf(
			codes.Aborted,
			"startingToken=%d > len(vols)=%d",
			startingToken, ulenVols)
	}

	// Discern the number of remaining entries.
	rem := ulenVols - startingToken

	// If maxEntries is 0 or greater than the number of remaining entries then
	// set maxEntries to the number of remaining entries.
	if maxEntries == 0 || maxEntries > rem {
		maxEntries = rem
	}

	var (
		i       int
		j       = startingToken
		entries = make(
			[]*csi.ListVolumesResponse_Entry,
			maxEntries)
	)

	for i = 0; i < len(entries); i++ {
		volumeStatus := &csi.ListVolumesResponse_VolumeStatus{
			VolumeCondition: &csi.VolumeCondition{},
		}

		if !s.config.DisableAttach {
			volumeStatus.PublishedNodeIds = []string{
				s.nodeID,
			}
		}

		entries[i] = &csi.ListVolumesResponse_Entry{
			Volume: &vols[j],
			Status: volumeStatus,
		}
		j++
	}

	var nextToken string
	if n := startingToken + int32(i); n < ulenVols {
		nextToken = fmt.Sprintf("%d", n)
	}

	if hookVal, hookMsg := s.execHook("ListVolumesEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.ListVolumesResponse{
		Entries:   entries,
		NextToken: nextToken,
	}, nil
}

func (s *service) GetCapacity(
	ctx context.Context,
	req *csi.GetCapacityRequest) (
	*csi.GetCapacityResponse, error) {

	if hookVal, hookMsg := s.execHook("GetCapacity"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.GetCapacityResponse{
		AvailableCapacity: MaxStorageCapacity,
	}, nil
}

func (s *service) ControllerGetCapabilities(
	ctx context.Context,
	req *csi.ControllerGetCapabilitiesRequest) (
	*csi.ControllerGetCapabilitiesResponse, error) {

	if hookVal, hookMsg := s.execHook("ControllerGetCapabilitiesStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	caps := []*csi.ControllerServiceCapability{
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_CREATE_DELETE_VOLUME,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_LIST_VOLUMES,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_LIST_VOLUMES_PUBLISHED_NODES,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_GET_CAPACITY,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_LIST_SNAPSHOTS,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_CREATE_DELETE_SNAPSHOT,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_PUBLISH_READONLY,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_CLONE_VOLUME,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_GET_VOLUME,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_VOLUME_CONDITION,
				},
			},
		},
		{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_MODIFY_VOLUME,
				},
			},
		},
	}

	if !s.config.DisableAttach {
		caps = append(caps, &csi.ControllerServiceCapability{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_PUBLISH_UNPUBLISH_VOLUME,
				},
			},
		})
	}

	if !s.config.DisableControllerExpansion {
		caps = append(caps, &csi.ControllerServiceCapability{
			Type: &csi.ControllerServiceCapability_Rpc{
				Rpc: &csi.ControllerServiceCapability_RPC{
					Type: csi.ControllerServiceCapability_RPC_EXPAND_VOLUME,
				},
			},
		})
	}

	if hookVal, hookMsg := s.execHook("ControllerGetCapabilitiesEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.ControllerGetCapabilitiesResponse{
		Capabilities: caps,
	}, nil
}

func (s *service) CreateSnapshot(ctx context.Context,
	req *csi.CreateSnapshotRequest) (*csi.CreateSnapshotResponse, error) {
	// Check arguments
	if len(req.GetName()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Snapshot Name cannot be empty")
	}
	if len(req.GetSourceVolumeId()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Snapshot SourceVolumeId cannot be empty")
	}

	// Check to see if the snapshot already exists.
	if i, v := s.snapshots.FindSnapshot("name", req.GetName()); i >= 0 {
		// Requested snapshot name already exists
		if v.SnapshotCSI.GetSourceVolumeId() != req.GetSourceVolumeId() || !reflect.DeepEqual(v.Parameters, req.GetParameters()) {
			return nil, status.Error(codes.AlreadyExists,
				fmt.Sprintf("Snapshot with name %s already exists", req.GetName()))
		}
		return &csi.CreateSnapshotResponse{Snapshot: &v.SnapshotCSI}, nil
	}

	// Create the snapshot and add it to the service's in-mem snapshot slice.
	snapshot := s.newSnapshot(req.GetName(), req.GetSourceVolumeId(), req.GetParameters())
	s.snapshots.Add(snapshot)

	if hookVal, hookMsg := s.execHook("CreateSnapshotEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.CreateSnapshotResponse{Snapshot: &snapshot.SnapshotCSI}, nil
}

func (s *service) DeleteSnapshot(ctx context.Context,
	req *csi.DeleteSnapshotRequest) (*csi.DeleteSnapshotResponse, error) {

	//  If the snapshot is not specified, return error
	if len(req.SnapshotId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Snapshot ID cannot be empty")
	}

	if hookVal, hookMsg := s.execHook("DeleteSnapshotStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	// If the snapshot does not exist then return an idempotent response.
	i, _ := s.snapshots.FindSnapshot("id", req.SnapshotId)
	if i < 0 {
		return &csi.DeleteSnapshotResponse{}, nil
	}

	// This delete logic preserves order and prevents potential memory
	// leaks. The slice's elements may not be pointers, but the structs
	// themselves have fields that are.
	s.snapshots.Delete(i)
	klog.V(5).InfoS("mock delete snapshot", "snapshotId", req.SnapshotId)

	if hookVal, hookMsg := s.execHook("DeleteSnapshotEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return &csi.DeleteSnapshotResponse{}, nil
}

func (s *service) ListSnapshots(ctx context.Context,
	req *csi.ListSnapshotsRequest) (*csi.ListSnapshotsResponse, error) {

	if hookVal, hookMsg := s.execHook("ListSnapshots"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	// case 1: SnapshotId is not empty, return snapshots that match the snapshot id.
	if len(req.GetSnapshotId()) != 0 {
		return getSnapshotById(s, req)
	}

	// case 2: SourceVolumeId is not empty, return snapshots that match the source volume id.
	if len(req.GetSourceVolumeId()) != 0 {
		return getSnapshotByVolumeId(s, req)
	}

	// case 3: no parameter is set, so we return all the snapshots.
	return getAllSnapshots(s, req)
}

func (s *service) ControllerExpandVolume(
	ctx context.Context,
	req *csi.ControllerExpandVolumeRequest) (*csi.ControllerExpandVolumeResponse, error) {
	if len(req.VolumeId) == 0 {
		return nil, status.Error(codes.InvalidArgument, "Volume ID cannot be empty")
	}

	if req.CapacityRange == nil {
		return nil, status.Error(codes.InvalidArgument, "Request capacity cannot be empty")
	}

	if hookVal, hookMsg := s.execHook("ControllerExpandVolumeStart"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	s.volsRWL.Lock()
	defer s.volsRWL.Unlock()

	i, v := s.findVolNoLock("id", req.VolumeId)
	if i < 0 {
		return nil, status.Error(codes.NotFound, req.VolumeId)
	}

	if s.config.DisableOnlineExpansion && MockVolumes[v.GetVolumeId()].ISControllerPublished {
		return nil, status.Error(codes.FailedPrecondition, "volume is published and online volume expansion is not supported")
	}

	requestBytes := req.CapacityRange.RequiredBytes

	if v.CapacityBytes > requestBytes {
		return nil, status.Error(codes.InvalidArgument, "cannot change volume capacity to a smaller size")
	}

	resp := &csi.ControllerExpandVolumeResponse{
		CapacityBytes:         requestBytes,
		NodeExpansionRequired: s.config.NodeExpansionRequired,
	}

	// Check to see if the volume already satisfied request size.
	if v.CapacityBytes == requestBytes {
		klog.V(5).InfoS("volume capacity sufficient, no need to expand", "requested", requestBytes, "current", v.CapacityBytes, "volumeID", v.VolumeId)
		return resp, nil
	}

	// Update volume's capacity to the requested size.
	v.CapacityBytes = requestBytes
	s.vols[i] = v

	if hookVal, hookMsg := s.execHook("ControllerExpandVolumeEnd"); hookVal != codes.OK {
		return nil, status.Errorf(hookVal, hookMsg)
	}

	return resp, nil
}

func (s *service) ControllerModifyVolume(
	ctx context.Context,
	req *csi.ControllerModifyVolumeRequest) (*csi.ControllerModifyVolumeResponse, error) {
	// todo: implement the functionality while we add the modifyVolume test
	resp := &csi.ControllerModifyVolumeResponse{}
	return resp, nil
}

func getSnapshotById(s *service, req *csi.ListSnapshotsRequest) (*csi.ListSnapshotsResponse, error) {
	if len(req.GetSnapshotId()) != 0 {
		i, snapshot := s.snapshots.FindSnapshot("id", req.GetSnapshotId())
		if i < 0 {
			return &csi.ListSnapshotsResponse{}, nil
		}

		if len(req.GetSourceVolumeId()) != 0 {
			if snapshot.SnapshotCSI.GetSourceVolumeId() != req.GetSourceVolumeId() {
				return &csi.ListSnapshotsResponse{}, nil
			}
		}

		return &csi.ListSnapshotsResponse{
			Entries: []*csi.ListSnapshotsResponse_Entry{
				{
					Snapshot: &snapshot.SnapshotCSI,
				},
			},
		}, nil
	}
	return nil, nil
}

func getSnapshotByVolumeId(s *service, req *csi.ListSnapshotsRequest) (*csi.ListSnapshotsResponse, error) {
	if len(req.GetSourceVolumeId()) != 0 {
		i, snapshot := s.snapshots.FindSnapshot("sourceVolumeId", req.SourceVolumeId)
		if i < 0 {
			return &csi.ListSnapshotsResponse{}, nil
		}
		return &csi.ListSnapshotsResponse{
			Entries: []*csi.ListSnapshotsResponse_Entry{
				{
					Snapshot: &snapshot.SnapshotCSI,
				},
			},
		}, nil
	}
	return nil, nil
}

func getAllSnapshots(s *service, req *csi.ListSnapshotsRequest) (*csi.ListSnapshotsResponse, error) {
	// Copy the mock snapshots into a new slice in order to avoid
	// locking the service's snapshot slice for the duration of the
	// ListSnapshots RPC.
	readyToUse := true
	snapshots := s.snapshots.List(readyToUse)

	var (
		ulenSnapshots = int32(len(snapshots))
		maxEntries    = req.MaxEntries
		startingToken int32
	)

	if v := req.StartingToken; v != "" {
		i, err := strconv.ParseUint(v, 10, 32)
		if err != nil {
			return nil, status.Errorf(
				codes.Aborted,
				"startingToken=%s: %v",
				v, err)
		}
		startingToken = int32(i)
	}

	if startingToken > ulenSnapshots {
		return nil, status.Errorf(
			codes.Aborted,
			"startingToken=%d > len(snapshots)=%d",
			startingToken, ulenSnapshots)
	}

	// Discern the number of remaining entries.
	rem := ulenSnapshots - startingToken

	// If maxEntries is 0 or greater than the number of remaining entries then
	// set maxEntries to the number of remaining entries.
	if maxEntries == 0 || maxEntries > rem {
		maxEntries = rem
	}

	var (
		i       int
		j       = startingToken
		entries = make(
			[]*csi.ListSnapshotsResponse_Entry,
			maxEntries)
	)

	for i = 0; i < len(entries); i++ {
		entries[i] = &csi.ListSnapshotsResponse_Entry{
			Snapshot: &snapshots[j],
		}
		j++
	}

	var nextToken string
	if n := startingToken + int32(i); n < ulenSnapshots {
		nextToken = fmt.Sprintf("%d", n)
	}

	return &csi.ListSnapshotsResponse{
		Entries:   entries,
		NextToken: nextToken,
	}, nil
}
