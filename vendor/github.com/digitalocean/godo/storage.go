package godo

import (
	"fmt"
	"time"
)

const (
	storageBasePath  = "v2"
	storageAllocPath = storageBasePath + "/volumes"
	storageSnapPath  = storageBasePath + "/snapshots"
)

// StorageService is an interface for interfacing with the storage
// endpoints of the Digital Ocean API.
// See: https://developers.digitalocean.com/documentation/v2#storage
type StorageService interface {
	ListVolumes(*ListOptions) ([]Volume, *Response, error)
	GetVolume(string) (*Volume, *Response, error)
	CreateVolume(*VolumeCreateRequest) (*Volume, *Response, error)
	DeleteVolume(string) (*Response, error)
}

// BetaStorageService is an interface for the storage services that are
// not yet stable. The interface is not exposed in the godo.Client and
// requires type-asserting the `StorageService` to make it available.
//
// Note that Beta features will change and compiling against those
// symbols (using type-assertion) is prone to breaking your build
// if you use our master.
type BetaStorageService interface {
	StorageService

	ListSnapshots(volumeID string, opts *ListOptions) ([]Snapshot, *Response, error)
	GetSnapshot(string) (*Snapshot, *Response, error)
	CreateSnapshot(*SnapshotCreateRequest) (*Snapshot, *Response, error)
	DeleteSnapshot(string) (*Response, error)
}

// StorageServiceOp handles communication with the storage volumes related methods of the
// DigitalOcean API.
type StorageServiceOp struct {
	client *Client
}

var _ StorageService = &StorageServiceOp{}

// Volume represents a Digital Ocean block store volume.
type Volume struct {
	ID            string    `json:"id"`
	Region        *Region   `json:"region"`
	Name          string    `json:"name"`
	SizeGigaBytes int64     `json:"size_gigabytes"`
	Description   string    `json:"description"`
	DropletIDs    []int     `json:"droplet_ids"`
	CreatedAt     time.Time `json:"created_at"`
}

func (f Volume) String() string {
	return Stringify(f)
}

type storageVolumesRoot struct {
	Volumes []Volume `json:"volumes"`
	Links   *Links   `json:"links"`
}

type storageVolumeRoot struct {
	Volume *Volume `json:"volume"`
	Links  *Links  `json:"links,omitempty"`
}

// VolumeCreateRequest represents a request to create a block store
// volume.
type VolumeCreateRequest struct {
	Region        string `json:"region"`
	Name          string `json:"name"`
	Description   string `json:"description"`
	SizeGigaBytes int64  `json:"size_gigabytes"`
}

// ListVolumes lists all storage volumes.
func (svc *StorageServiceOp) ListVolumes(opt *ListOptions) ([]Volume, *Response, error) {
	path, err := addOptions(storageAllocPath, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := svc.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumesRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Volumes, resp, nil
}

// CreateVolume creates a storage volume. The name must be unique.
func (svc *StorageServiceOp) CreateVolume(createRequest *VolumeCreateRequest) (*Volume, *Response, error) {
	path := storageAllocPath

	req, err := svc.client.NewRequest("POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumeRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	return root.Volume, resp, nil
}

// GetVolume retrieves an individual storage volume.
func (svc *StorageServiceOp) GetVolume(id string) (*Volume, *Response, error) {
	path := fmt.Sprintf("%s/%s", storageAllocPath, id)

	req, err := svc.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumeRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Volume, resp, nil
}

// DeleteVolume deletes a storage volume.
func (svc *StorageServiceOp) DeleteVolume(id string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", storageAllocPath, id)

	req, err := svc.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}
	return svc.client.Do(req, nil)
}

// Snapshot represents a Digital Ocean block store snapshot.
type Snapshot struct {
	ID            string    `json:"id"`
	VolumeID      string    `json:"volume_id"`
	Region        *Region   `json:"region"`
	Name          string    `json:"name"`
	SizeGigaBytes int64     `json:"size_gigabytes"`
	Description   string    `json:"description"`
	CreatedAt     time.Time `json:"created_at"`
}

type storageSnapsRoot struct {
	Snapshots []Snapshot `json:"snapshots"`
	Links     *Links     `json:"links"`
}

type storageSnapRoot struct {
	Snapshot *Snapshot `json:"snapshot"`
	Links    *Links    `json:"links,omitempty"`
}

// SnapshotCreateRequest represents a request to create a block store
// volume.
type SnapshotCreateRequest struct {
	VolumeID    string `json:"volume_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// ListSnapshots lists all snapshots related to a storage volume.
func (svc *StorageServiceOp) ListSnapshots(volumeID string, opt *ListOptions) ([]Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s/snapshots", storageAllocPath, volumeID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := svc.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageSnapsRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Snapshots, resp, nil
}

// CreateSnapshot creates a snapshot of a storage volume.
func (svc *StorageServiceOp) CreateSnapshot(createRequest *SnapshotCreateRequest) (*Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s/snapshots", storageAllocPath, createRequest.VolumeID)

	req, err := svc.client.NewRequest("POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageSnapRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	return root.Snapshot, resp, nil
}

// GetSnapshot retrieves an individual snapshot.
func (svc *StorageServiceOp) GetSnapshot(id string) (*Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s", storageSnapPath, id)

	req, err := svc.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageSnapRoot)
	resp, err := svc.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Snapshot, resp, nil
}

// DeleteSnapshot deletes a snapshot.
func (svc *StorageServiceOp) DeleteSnapshot(id string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", storageSnapPath, id)

	req, err := svc.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}
	return svc.client.Do(req, nil)
}
