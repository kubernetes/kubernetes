package godo

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/digitalocean/godo/context"
)

const dropletBasePath = "v2/droplets"

var errNoNetworks = errors.New("no networks have been defined")

// DropletsService is an interface for interfacing with the Droplet
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#droplets
type DropletsService interface {
	List(context.Context, *ListOptions) ([]Droplet, *Response, error)
	ListByTag(context.Context, string, *ListOptions) ([]Droplet, *Response, error)
	Get(context.Context, int) (*Droplet, *Response, error)
	Create(context.Context, *DropletCreateRequest) (*Droplet, *Response, error)
	CreateMultiple(context.Context, *DropletMultiCreateRequest) ([]Droplet, *Response, error)
	Delete(context.Context, int) (*Response, error)
	DeleteByTag(context.Context, string) (*Response, error)
	Kernels(context.Context, int, *ListOptions) ([]Kernel, *Response, error)
	Snapshots(context.Context, int, *ListOptions) ([]Image, *Response, error)
	Backups(context.Context, int, *ListOptions) ([]Image, *Response, error)
	Actions(context.Context, int, *ListOptions) ([]Action, *Response, error)
	Neighbors(context.Context, int) ([]Droplet, *Response, error)
}

// DropletsServiceOp handles communication with the Droplet related methods of the
// DigitalOcean API.
type DropletsServiceOp struct {
	client *Client
}

var _ DropletsService = &DropletsServiceOp{}

// Droplet represents a DigitalOcean Droplet
type Droplet struct {
	ID               int           `json:"id,float64,omitempty"`
	Name             string        `json:"name,omitempty"`
	Memory           int           `json:"memory,omitempty"`
	Vcpus            int           `json:"vcpus,omitempty"`
	Disk             int           `json:"disk,omitempty"`
	Region           *Region       `json:"region,omitempty"`
	Image            *Image        `json:"image,omitempty"`
	Size             *Size         `json:"size,omitempty"`
	SizeSlug         string        `json:"size_slug,omitempty"`
	BackupIDs        []int         `json:"backup_ids,omitempty"`
	NextBackupWindow *BackupWindow `json:"next_backup_window,omitempty"`
	SnapshotIDs      []int         `json:"snapshot_ids,omitempty"`
	Features         []string      `json:"features,omitempty"`
	Locked           bool          `json:"locked,bool,omitempty"`
	Status           string        `json:"status,omitempty"`
	Networks         *Networks     `json:"networks,omitempty"`
	Created          string        `json:"created_at,omitempty"`
	Kernel           *Kernel       `json:"kernel,omitempty"`
	Tags             []string      `json:"tags,omitempty"`
	VolumeIDs        []string      `json:"volume_ids"`
}

// PublicIPv4 returns the public IPv4 address for the Droplet.
func (d *Droplet) PublicIPv4() (string, error) {
	if d.Networks == nil {
		return "", errNoNetworks
	}

	for _, v4 := range d.Networks.V4 {
		if v4.Type == "public" {
			return v4.IPAddress, nil
		}
	}

	return "", nil
}

// PrivateIPv4 returns the private IPv4 address for the Droplet.
func (d *Droplet) PrivateIPv4() (string, error) {
	if d.Networks == nil {
		return "", errNoNetworks
	}

	for _, v4 := range d.Networks.V4 {
		if v4.Type == "private" {
			return v4.IPAddress, nil
		}
	}

	return "", nil
}

// PublicIPv6 returns the public IPv6 address for the Droplet.
func (d *Droplet) PublicIPv6() (string, error) {
	if d.Networks == nil {
		return "", errNoNetworks
	}

	for _, v6 := range d.Networks.V6 {
		if v6.Type == "public" {
			return v6.IPAddress, nil
		}
	}

	return "", nil
}

// Kernel object
type Kernel struct {
	ID      int    `json:"id,float64,omitempty"`
	Name    string `json:"name,omitempty"`
	Version string `json:"version,omitempty"`
}

// BackupWindow object
type BackupWindow struct {
	Start *Timestamp `json:"start,omitempty"`
	End   *Timestamp `json:"end,omitempty"`
}

// Convert Droplet to a string
func (d Droplet) String() string {
	return Stringify(d)
}

// DropletRoot represents a Droplet root
type dropletRoot struct {
	Droplet *Droplet `json:"droplet"`
	Links   *Links   `json:"links,omitempty"`
}

type dropletsRoot struct {
	Droplets []Droplet `json:"droplets"`
	Links    *Links    `json:"links"`
}

type kernelsRoot struct {
	Kernels []Kernel `json:"kernels,omitempty"`
	Links   *Links   `json:"links"`
}

type dropletSnapshotsRoot struct {
	Snapshots []Image `json:"snapshots,omitempty"`
	Links     *Links  `json:"links"`
}

type backupsRoot struct {
	Backups []Image `json:"backups,omitempty"`
	Links   *Links  `json:"links"`
}

// DropletCreateImage identifies an image for the create request. It prefers slug over ID.
type DropletCreateImage struct {
	ID   int
	Slug string
}

// MarshalJSON returns either the slug or id of the image. It returns the id
// if the slug is empty.
func (d DropletCreateImage) MarshalJSON() ([]byte, error) {
	if d.Slug != "" {
		return json.Marshal(d.Slug)
	}

	return json.Marshal(d.ID)
}

// DropletCreateVolume identifies a volume to attach for the create request. It
// prefers Name over ID,
type DropletCreateVolume struct {
	ID   string
	Name string
}

// MarshalJSON returns an object with either the name or id of the volume. It
// returns the id if the name is empty.
func (d DropletCreateVolume) MarshalJSON() ([]byte, error) {
	if d.Name != "" {
		return json.Marshal(struct {
			Name string `json:"name"`
		}{Name: d.Name})
	}

	return json.Marshal(struct {
		ID string `json:"id"`
	}{ID: d.ID})
}

// DropletCreateSSHKey identifies a SSH Key for the create request. It prefers fingerprint over ID.
type DropletCreateSSHKey struct {
	ID          int
	Fingerprint string
}

// MarshalJSON returns either the fingerprint or id of the ssh key. It returns
// the id if the fingerprint is empty.
func (d DropletCreateSSHKey) MarshalJSON() ([]byte, error) {
	if d.Fingerprint != "" {
		return json.Marshal(d.Fingerprint)
	}

	return json.Marshal(d.ID)
}

// DropletCreateRequest represents a request to create a Droplet.
type DropletCreateRequest struct {
	Name              string                `json:"name"`
	Region            string                `json:"region"`
	Size              string                `json:"size"`
	Image             DropletCreateImage    `json:"image"`
	SSHKeys           []DropletCreateSSHKey `json:"ssh_keys"`
	Backups           bool                  `json:"backups"`
	IPv6              bool                  `json:"ipv6"`
	PrivateNetworking bool                  `json:"private_networking"`
	Monitoring        bool                  `json:"monitoring"`
	UserData          string                `json:"user_data,omitempty"`
	Volumes           []DropletCreateVolume `json:"volumes,omitempty"`
	Tags              []string              `json:"tags"`
}

// DropletMultiCreateRequest is a request to create multiple Droplets.
type DropletMultiCreateRequest struct {
	Names             []string              `json:"names"`
	Region            string                `json:"region"`
	Size              string                `json:"size"`
	Image             DropletCreateImage    `json:"image"`
	SSHKeys           []DropletCreateSSHKey `json:"ssh_keys"`
	Backups           bool                  `json:"backups"`
	IPv6              bool                  `json:"ipv6"`
	PrivateNetworking bool                  `json:"private_networking"`
	Monitoring        bool                  `json:"monitoring"`
	UserData          string                `json:"user_data,omitempty"`
	Tags              []string              `json:"tags"`
}

func (d DropletCreateRequest) String() string {
	return Stringify(d)
}

func (d DropletMultiCreateRequest) String() string {
	return Stringify(d)
}

// Networks represents the Droplet's Networks.
type Networks struct {
	V4 []NetworkV4 `json:"v4,omitempty"`
	V6 []NetworkV6 `json:"v6,omitempty"`
}

// NetworkV4 represents a DigitalOcean IPv4 Network.
type NetworkV4 struct {
	IPAddress string `json:"ip_address,omitempty"`
	Netmask   string `json:"netmask,omitempty"`
	Gateway   string `json:"gateway,omitempty"`
	Type      string `json:"type,omitempty"`
}

func (n NetworkV4) String() string {
	return Stringify(n)
}

// NetworkV6 represents a DigitalOcean IPv6 network.
type NetworkV6 struct {
	IPAddress string `json:"ip_address,omitempty"`
	Netmask   int    `json:"netmask,omitempty"`
	Gateway   string `json:"gateway,omitempty"`
	Type      string `json:"type,omitempty"`
}

func (n NetworkV6) String() string {
	return Stringify(n)
}

// Performs a list request given a path.
func (s *DropletsServiceOp) list(ctx context.Context, path string) ([]Droplet, *Response, error) {
	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Droplets, resp, err
}

// List all Droplets.
func (s *DropletsServiceOp) List(ctx context.Context, opt *ListOptions) ([]Droplet, *Response, error) {
	path := dropletBasePath
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	return s.list(ctx, path)
}

// ListByTag lists all Droplets matched by a Tag.
func (s *DropletsServiceOp) ListByTag(ctx context.Context, tag string, opt *ListOptions) ([]Droplet, *Response, error) {
	path := fmt.Sprintf("%s?tag_name=%s", dropletBasePath, tag)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	return s.list(ctx, path)
}

// Get individual Droplet.
func (s *DropletsServiceOp) Get(ctx context.Context, dropletID int) (*Droplet, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", dropletBasePath, dropletID)

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Droplet, resp, err
}

// Create Droplet
func (s *DropletsServiceOp) Create(ctx context.Context, createRequest *DropletCreateRequest) (*Droplet, *Response, error) {
	if createRequest == nil {
		return nil, nil, NewArgError("createRequest", "cannot be nil")
	}

	path := dropletBasePath

	req, err := s.client.NewRequest(ctx, "POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Droplet, resp, err
}

// CreateMultiple creates multiple Droplets.
func (s *DropletsServiceOp) CreateMultiple(ctx context.Context, createRequest *DropletMultiCreateRequest) ([]Droplet, *Response, error) {
	if createRequest == nil {
		return nil, nil, NewArgError("createRequest", "cannot be nil")
	}

	path := dropletBasePath

	req, err := s.client.NewRequest(ctx, "POST", path, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Droplets, resp, err
}

// Performs a delete request given a path
func (s *DropletsServiceOp) delete(ctx context.Context, path string) (*Response, error) {
	req, err := s.client.NewRequest(ctx, "DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(ctx, req, nil)

	return resp, err
}

// Delete Droplet.
func (s *DropletsServiceOp) Delete(ctx context.Context, dropletID int) (*Response, error) {
	if dropletID < 1 {
		return nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", dropletBasePath, dropletID)

	return s.delete(ctx, path)
}

// DeleteByTag deletes Droplets matched by a Tag.
func (s *DropletsServiceOp) DeleteByTag(ctx context.Context, tag string) (*Response, error) {
	if tag == "" {
		return nil, NewArgError("tag", "cannot be empty")
	}

	path := fmt.Sprintf("%s?tag_name=%s", dropletBasePath, tag)

	return s.delete(ctx, path)
}

// Kernels lists kernels available for a Droplet.
func (s *DropletsServiceOp) Kernels(ctx context.Context, dropletID int, opt *ListOptions) ([]Kernel, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d/kernels", dropletBasePath, dropletID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(kernelsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Kernels, resp, err
}

// Actions lists the actions for a Droplet.
func (s *DropletsServiceOp) Actions(ctx context.Context, dropletID int, opt *ListOptions) ([]Action, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d/actions", dropletBasePath, dropletID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(actionsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Actions, resp, err
}

// Backups lists the backups for a Droplet.
func (s *DropletsServiceOp) Backups(ctx context.Context, dropletID int, opt *ListOptions) ([]Image, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d/backups", dropletBasePath, dropletID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(backupsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Backups, resp, err
}

// Snapshots lists the snapshots available for a Droplet.
func (s *DropletsServiceOp) Snapshots(ctx context.Context, dropletID int, opt *ListOptions) ([]Image, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d/snapshots", dropletBasePath, dropletID)
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletSnapshotsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.Snapshots, resp, err
}

// Neighbors lists the neighbors for a Droplet.
func (s *DropletsServiceOp) Neighbors(ctx context.Context, dropletID int) ([]Droplet, *Response, error) {
	if dropletID < 1 {
		return nil, nil, NewArgError("dropletID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d/neighbors", dropletBasePath, dropletID)

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(dropletsRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Droplets, resp, err
}

func (s *DropletsServiceOp) dropletActionStatus(ctx context.Context, uri string) (string, error) {
	action, _, err := s.client.DropletActions.GetByURI(ctx, uri)

	if err != nil {
		return "", err
	}

	return action.Status, nil
}
