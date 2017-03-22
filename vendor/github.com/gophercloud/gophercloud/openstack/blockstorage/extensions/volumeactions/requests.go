package volumeactions

import (
	"github.com/gophercloud/gophercloud"
)

// AttachOptsBuilder allows extensions to add additional parameters to the
// Attach request.
type AttachOptsBuilder interface {
	ToVolumeAttachMap() (map[string]interface{}, error)
}

// AttachMode describes the attachment mode for volumes.
type AttachMode string

// These constants determine how a volume is attached
const (
	ReadOnly  AttachMode = "ro"
	ReadWrite AttachMode = "rw"
)

// AttachOpts contains options for attaching a Volume.
type AttachOpts struct {
	// The mountpoint of this volume
	MountPoint string `json:"mountpoint,omitempty"`
	// The nova instance ID, can't set simultaneously with HostName
	InstanceUUID string `json:"instance_uuid,omitempty"`
	// The hostname of baremetal host, can't set simultaneously with InstanceUUID
	HostName string `json:"host_name,omitempty"`
	// Mount mode of this volume
	Mode AttachMode `json:"mode,omitempty"`
}

// ToVolumeAttachMap assembles a request body based on the contents of a
// AttachOpts.
func (opts AttachOpts) ToVolumeAttachMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "os-attach")
}

// Attach will attach a volume based on the values in AttachOpts.
func Attach(client *gophercloud.ServiceClient, id string, opts AttachOptsBuilder) (r AttachResult) {
	b, err := opts.ToVolumeAttachMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(attachURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// BeginDetach will mark the volume as detaching
func BeginDetaching(client *gophercloud.ServiceClient, id string) (r BeginDetachingResult) {
	b := map[string]interface{}{"os-begin_detaching": make(map[string]interface{})}
	_, r.Err = client.Post(beginDetachingURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// DetachOptsBuilder allows extensions to add additional parameters to the
// Detach request.
type DetachOptsBuilder interface {
	ToVolumeDetachMap() (map[string]interface{}, error)
}

type DetachOpts struct {
	AttachmentID string `json:"attachment_id,omitempty"`
}

// ToVolumeDetachMap assembles a request body based on the contents of a
// DetachOpts.
func (opts DetachOpts) ToVolumeDetachMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "os-detach")
}

// Detach will detach a volume based on volume id.
func Detach(client *gophercloud.ServiceClient, id string, opts DetachOptsBuilder) (r DetachResult) {
	b, err := opts.ToVolumeDetachMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(detachURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// Reserve will reserve a volume based on volume id.
func Reserve(client *gophercloud.ServiceClient, id string) (r ReserveResult) {
	b := map[string]interface{}{"os-reserve": make(map[string]interface{})}
	_, r.Err = client.Post(reserveURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// Unreserve will unreserve a volume based on volume id.
func Unreserve(client *gophercloud.ServiceClient, id string) (r UnreserveResult) {
	b := map[string]interface{}{"os-unreserve": make(map[string]interface{})}
	_, r.Err = client.Post(unreserveURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// InitializeConnectionOptsBuilder allows extensions to add additional parameters to the
// InitializeConnection request.
type InitializeConnectionOptsBuilder interface {
	ToVolumeInitializeConnectionMap() (map[string]interface{}, error)
}

// InitializeConnectionOpts hosts options for InitializeConnection.
type InitializeConnectionOpts struct {
	IP        string   `json:"ip,omitempty"`
	Host      string   `json:"host,omitempty"`
	Initiator string   `json:"initiator,omitempty"`
	Wwpns     []string `json:"wwpns,omitempty"`
	Wwnns     string   `json:"wwnns,omitempty"`
	Multipath *bool    `json:"multipath,omitempty"`
	Platform  string   `json:"platform,omitempty"`
	OSType    string   `json:"os_type,omitempty"`
}

// ToVolumeInitializeConnectionMap assembles a request body based on the contents of a
// InitializeConnectionOpts.
func (opts InitializeConnectionOpts) ToVolumeInitializeConnectionMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "connector")
	return map[string]interface{}{"os-initialize_connection": b}, err
}

// InitializeConnection initializes iscsi connection.
func InitializeConnection(client *gophercloud.ServiceClient, id string, opts InitializeConnectionOptsBuilder) (r InitializeConnectionResult) {
	b, err := opts.ToVolumeInitializeConnectionMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(initializeConnectionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	return
}

// TerminateConnectionOptsBuilder allows extensions to add additional parameters to the
// TerminateConnection request.
type TerminateConnectionOptsBuilder interface {
	ToVolumeTerminateConnectionMap() (map[string]interface{}, error)
}

// TerminateConnectionOpts hosts options for TerminateConnection.
type TerminateConnectionOpts struct {
	IP        string   `json:"ip,omitempty"`
	Host      string   `json:"host,omitempty"`
	Initiator string   `json:"initiator,omitempty"`
	Wwpns     []string `json:"wwpns,omitempty"`
	Wwnns     string   `json:"wwnns,omitempty"`
	Multipath *bool    `json:"multipath,omitempty"`
	Platform  string   `json:"platform,omitempty"`
	OSType    string   `json:"os_type,omitempty"`
}

// ToVolumeTerminateConnectionMap assembles a request body based on the contents of a
// TerminateConnectionOpts.
func (opts TerminateConnectionOpts) ToVolumeTerminateConnectionMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "connector")
	return map[string]interface{}{"os-terminate_connection": b}, err
}

// TerminateConnection terminates iscsi connection.
func TerminateConnection(client *gophercloud.ServiceClient, id string, opts TerminateConnectionOptsBuilder) (r TerminateConnectionResult) {
	b, err := opts.ToVolumeTerminateConnectionMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(teminateConnectionURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// ExtendSizeOptsBuilder allows extensions to add additional parameters to the
// ExtendSize request.
type ExtendSizeOptsBuilder interface {
	ToVolumeExtendSizeMap() (map[string]interface{}, error)
}

// ExtendSizeOpts contain options for extending the size of an existing Volume. This object is passed
// to the volumes.ExtendSize function.
type ExtendSizeOpts struct {
	// NewSize is the new size of the volume, in GB
	NewSize int `json:"new_size" required:"true"`
}

// ToVolumeExtendSizeMap assembles a request body based on the contents of an
// ExtendSizeOpts.
func (opts ExtendSizeOpts) ToVolumeExtendSizeMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "os-extend")
}

// ExtendSize will extend the size of the volume based on the provided information.
// This operation does not return a response body.
func ExtendSize(client *gophercloud.ServiceClient, id string, opts ExtendSizeOptsBuilder) (r ExtendSizeResult) {
	b, err := opts.ToVolumeExtendSizeMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(extendSizeURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}
