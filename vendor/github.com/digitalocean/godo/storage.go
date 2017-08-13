package godo

import (
	"fmt"
	"time"

	"github.com/digitalocean/godo/context"
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
	ListVolumes(context.Context, *ListVolumeParams) ([]Volume, *Response, error)
	GetVolume(context.Context, string) (*Volume, *Response, error)
	CreateVolume(context.Context, *VolumeCreateRequest) (*Volume, *Response, error)
	DeleteVolume(context.Context, string) (*Response, error)
	ListSnapshots(ctx context.Context, volumeID string, opts *ListOptions) ([]Snapshot, *Response, error)
	GetSnapshot(context.Context, string) (*Snapshot, *Response, error)
	CreateSnapshot(context.Context, *SnapshotCreateRequest) (*Snapshot, *Response, error)
	DeleteSnapshot(context.Context, string) (*Response, error)
}

// StorageServiceOp handles communication with the storage volumes related methods of the
// DigitalOcean API.
type StorageServiceOp struct {
	client *Client
}

// ListVolumeParams stores the options you can set for a ListVolumeCall
type ListVolumeParams struct {
	Region      string       `json:"region"`
	Name        string       `json:"name"`
	ListOptions *ListOptions `json:"list_options,omitempty"`
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
	SnapshotID    string `json:"snapshot_id"`
}

// ListVolumes lists all storage volumes.
func (svc *StorageServiceOp) ListVolumes(ctx context.Context, params *ListVolumeParams) ([]Volume, *Response, error) {
	path := storageAllocPath
	if params != nil {
		if params.Region != "" && params.Name != "" {
			path = fmt.Sprintf("%s?name=%s&region=%s", path, params.Name, params.Region)
		}

		if params.ListOptions != nil {
			var err error
			path, err = addOptions(path, params.ListOptions)
			if err != nil {
				return nil, nil, err
			}
		}
	}

	req, err := svc.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumesRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Volumes, resp, nil
}

// CreateVolume creates a storage volume. The name must be unique.
func (svc *StorageServiceOp) CreateVolume(ctx context.Context, createRequest *VolumeCreateRequest) (*Volume, *Response, error) {
	path := storageAllocPath

	req, err := svc.client.NewRequest(ctx, "POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumeRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	return root.Volume, resp, nil
}

// GetVolume retrieves an individual storage volume.
func (svc *StorageServiceOp) GetVolume(ctx context.Context, id string) (*Volume, *Response, error) {
	path := fmt.Sprintf("%s/%s", storageAllocPath, id)

	req, err := svc.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(storageVolumeRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Volume, resp, nil
}

// DeleteVolume deletes a storage volume.
func (svc *StorageServiceOp) DeleteVolume(ctx context.Context, id string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", storageAllocPath, id)

	req, err := svc.client.NewRequest(ctx, "DELETE", path, nil)
	if err != nil {
		return nil, err
	}
	return svc.client.Do(ctx, req, nil)
}

// SnapshotCreateRequest represents a request to create a block store
// volume.
type SnapshotCreateRequest struct {
	VolumeID    string `json:"volume_id"`
	Name        string `json:"name"`
	Description string `json:"description"`
}

// ListSnapshots lists all snapshots related to a storage volume.
func (svc *StorageServiceOp) ListSnapshots(ctx context.Context, volumeID string, opt *ListOptions) ([]Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s/snapshots", storageAllocPath, volumeID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := svc.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(snapshotsRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Snapshots, resp, nil
}

// CreateSnapshot creates a snapshot of a storage volume.
func (svc *StorageServiceOp) CreateSnapshot(ctx context.Context, createRequest *SnapshotCreateRequest) (*Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s/snapshots", storageAllocPath, createRequest.VolumeID)

	req, err := svc.client.NewRequest(ctx, "POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(snapshotRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	return root.Snapshot, resp, nil
}

// GetSnapshot retrieves an individual snapshot.
func (svc *StorageServiceOp) GetSnapshot(ctx context.Context, id string) (*Snapshot, *Response, error) {
	path := fmt.Sprintf("%s/%s", storageSnapPath, id)

	req, err := svc.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(snapshotRoot)
	resp, err := svc.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Snapshot, resp, nil
}

// DeleteSnapshot deletes a snapshot.
func (svc *StorageServiceOp) DeleteSnapshot(ctx context.Context, id string) (*Response, error) {
	path := fmt.Sprintf("%s/%s", storageSnapPath, id)

	req, err := svc.client.NewRequest(ctx, "DELETE", path, nil)
	if err != nil {
		return nil, err
	}
	return svc.client.Do(ctx, req, nil)
}
