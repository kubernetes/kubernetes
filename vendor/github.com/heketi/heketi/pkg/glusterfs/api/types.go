//
// Copyright (c) 2016 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//
// Please see https://github.com/heketi/heketi/wiki/API
// for documentation
//
package api

import (
	"fmt"
	"sort"
)

// State
type EntryState string

const (
	EntryStateUnknown EntryState = ""
	EntryStateOnline  EntryState = "online"
	EntryStateOffline EntryState = "offline"
	EntryStateFailed  EntryState = "failed"
)

type DurabilityType string

const (
	DurabilityReplicate      DurabilityType = "replicate"
	DurabilityDistributeOnly DurabilityType = "none"
	DurabilityEC             DurabilityType = "disperse"
)

// Common
type StateRequest struct {
	State EntryState `json:"state"`
}

// Storage values in KB
type StorageSize struct {
	Total uint64 `json:"total"`
	Free  uint64 `json:"free"`
	Used  uint64 `json:"used"`
}

type HostAddresses struct {
	Manage  sort.StringSlice `json:"manage"`
	Storage sort.StringSlice `json:"storage"`
}

// Brick
type BrickInfo struct {
	Id       string `json:"id"`
	Path     string `json:"path"`
	DeviceId string `json:"device"`
	NodeId   string `json:"node"`

	// Size in KB
	Size uint64 `json:"size"`
}

// Device
type Device struct {
	Name string `json:"name"`
}

type DeviceAddRequest struct {
	Device
	NodeId string `json:"node"`
}

type DeviceInfo struct {
	Device
	Storage StorageSize `json:"storage"`
	Id      string      `json:"id"`
}

type DeviceInfoResponse struct {
	DeviceInfo
	State  EntryState  `json:"state"`
	Bricks []BrickInfo `json:"bricks"`
}

// Node
type NodeAddRequest struct {
	Zone      int           `json:"zone"`
	Hostnames HostAddresses `json:"hostnames"`
	ClusterId string        `json:"cluster"`
}

type NodeInfo struct {
	NodeAddRequest
	Id string `json:"id"`
}

type NodeInfoResponse struct {
	NodeInfo
	State       EntryState           `json:"state"`
	DevicesInfo []DeviceInfoResponse `json:"devices"`
}

// Cluster
type Cluster struct {
	Volumes []VolumeInfoResponse `json:"volumes"`
	Nodes   []NodeInfoResponse   `json:"nodes"`
	Id      string               `json:"id"`
}

type TopologyInfoResponse struct {
	ClusterList []Cluster `json:"clusters"`
}

type ClusterInfoResponse struct {
	Id      string           `json:"id"`
	Nodes   sort.StringSlice `json:"nodes"`
	Volumes sort.StringSlice `json:"volumes"`
}

type ClusterListResponse struct {
	Clusters []string `json:"clusters"`
}

// Durabilities
type ReplicaDurability struct {
	Replica int `json:"replica,omitempty"`
}

type DisperseDurability struct {
	Data       int `json:"data,omitempty"`
	Redundancy int `json:"redundancy,omitempty"`
}

// Volume
type VolumeDurabilityInfo struct {
	Type      DurabilityType     `json:"type,omitempty"`
	Replicate ReplicaDurability  `json:"replicate,omitempty"`
	Disperse  DisperseDurability `json:"disperse,omitempty"`
}

type VolumeCreateRequest struct {
	// Size in GB
	Size       int                  `json:"size"`
	Clusters   []string             `json:"clusters,omitempty"`
	Name       string               `json:"name"`
	Durability VolumeDurabilityInfo `json:"durability,omitempty"`
	Snapshot   struct {
		Enable bool    `json:"enable"`
		Factor float32 `json:"factor"`
	} `json:"snapshot"`
}

type VolumeInfo struct {
	VolumeCreateRequest
	Id      string `json:"id"`
	Cluster string `json:"cluster"`
	Mount   struct {
		GlusterFS struct {
			Hosts      []string          `json:"hosts"`
			MountPoint string            `json:"device"`
			Options    map[string]string `json:"options"`
		} `json:"glusterfs"`
	} `json:"mount"`
}

type VolumeInfoResponse struct {
	VolumeInfo
	Bricks []BrickInfo `json:"bricks"`
}

type VolumeListResponse struct {
	Volumes []string `json:"volumes"`
}

type VolumeExpandRequest struct {
	Size int `json:"expand_size"`
}

// Constructors

func NewVolumeInfoResponse() *VolumeInfoResponse {

	info := &VolumeInfoResponse{}
	info.Mount.GlusterFS.Options = make(map[string]string)
	info.Bricks = make([]BrickInfo, 0)

	return info
}

// String functions
func (v *VolumeInfoResponse) String() string {
	s := fmt.Sprintf("Name: %v\n"+
		"Size: %v\n"+
		"Volume Id: %v\n"+
		"Cluster Id: %v\n"+
		"Mount: %v\n"+
		"Mount Options: backup-volfile-servers=%v\n"+
		"Durability Type: %v\n",
		v.Name,
		v.Size,
		v.Id,
		v.Cluster,
		v.Mount.GlusterFS.MountPoint,
		v.Mount.GlusterFS.Options["backup-volfile-servers"],
		v.Durability.Type)

	switch v.Durability.Type {
	case DurabilityEC:
		s += fmt.Sprintf("Disperse Data: %v\n"+
			"Disperse Redundancy: %v\n",
			v.Durability.Disperse.Data,
			v.Durability.Disperse.Redundancy)
	case DurabilityReplicate:
		s += fmt.Sprintf("Distributed+Replica: %v\n",
			v.Durability.Replicate.Replica)
	}

	if v.Snapshot.Enable {
		s += fmt.Sprintf("Snapshot Factor: %.2f\n",
			v.Snapshot.Factor)
	}

	/*
		s += "\nBricks:\n"
		for _, b := range v.Bricks {
			s += fmt.Sprintf("Id: %v\n"+
				"Path: %v\n"+
				"Size (GiB): %v\n"+
				"Node: %v\n"+
				"Device: %v\n\n",
				b.Id,
				b.Path,
				b.Size/(1024*1024),
				b.NodeId,
				b.DeviceId)
		}
	*/

	return s
}
