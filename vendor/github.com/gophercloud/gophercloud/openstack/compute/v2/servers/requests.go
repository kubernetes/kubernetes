package servers

import (
	"encoding/base64"
	"encoding/json"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/flavors"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/images"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToServerListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the server attributes you want to see returned. Marker and Limit are used
// for pagination.
type ListOpts struct {
	// ChangesSince is a time/date stamp for when the server last changed status.
	ChangesSince string `q:"changes-since"`

	// Image is the name of the image in URL format.
	Image string `q:"image"`

	// Flavor is the name of the flavor in URL format.
	Flavor string `q:"flavor"`

	// Name of the server as a string; can be queried with regular expressions.
	// Realize that ?name=bob returns both bob and bobb. If you need to match bob
	// only, you can use a regular expression matching the syntax of the
	// underlying database server implemented for Compute.
	Name string `q:"name"`

	// Status is the value of the status of the server so that you can filter on
	// "ACTIVE" for example.
	Status string `q:"status"`

	// Host is the name of the host as a string.
	Host string `q:"host"`

	// Marker is a UUID of the server at which you want to set a marker.
	Marker string `q:"marker"`

	// Limit is an integer value for the limit of values to return.
	Limit int `q:"limit"`

	// AllTenants is a bool to show all tenants.
	AllTenants bool `q:"all_tenants"`

	// TenantID lists servers for a particular tenant.
	// Setting "AllTenants = true" is required.
	TenantID string `q:"tenant_id"`
}

// ToServerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToServerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list servers accessible to you.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listDetailURL(client)
	if opts != nil {
		query, err := opts.ToServerListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ServerPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToServerCreateMap() (map[string]interface{}, error)
}

// Network is used within CreateOpts to control a new server's network
// attachments.
type Network struct {
	// UUID of a network to attach to the newly provisioned server.
	// Required unless Port is provided.
	UUID string

	// Port of a neutron network to attach to the newly provisioned server.
	// Required unless UUID is provided.
	Port string

	// FixedIP specifies a fixed IPv4 address to be used on this network.
	FixedIP string
}

// Personality is an array of files that are injected into the server at launch.
type Personality []*File

// File is used within CreateOpts and RebuildOpts to inject a file into the
// server at launch.
// File implements the json.Marshaler interface, so when a Create or Rebuild
// operation is requested, json.Marshal will call File's MarshalJSON method.
type File struct {
	// Path of the file.
	Path string

	// Contents of the file. Maximum content size is 255 bytes.
	Contents []byte
}

// MarshalJSON marshals the escaped file, base64 encoding the contents.
func (f *File) MarshalJSON() ([]byte, error) {
	file := struct {
		Path     string `json:"path"`
		Contents string `json:"contents"`
	}{
		Path:     f.Path,
		Contents: base64.StdEncoding.EncodeToString(f.Contents),
	}
	return json.Marshal(file)
}

// CreateOpts specifies server creation parameters.
type CreateOpts struct {
	// Name is the name to assign to the newly launched server.
	Name string `json:"name" required:"true"`

	// ImageRef [optional; required if ImageName is not provided] is the ID or
	// full URL to the image that contains the server's OS and initial state.
	// Also optional if using the boot-from-volume extension.
	ImageRef string `json:"imageRef"`

	// ImageName [optional; required if ImageRef is not provided] is the name of
	// the image that contains the server's OS and initial state.
	// Also optional if using the boot-from-volume extension.
	ImageName string `json:"-"`

	// FlavorRef [optional; required if FlavorName is not provided] is the ID or
	// full URL to the flavor that describes the server's specs.
	FlavorRef string `json:"flavorRef"`

	// FlavorName [optional; required if FlavorRef is not provided] is the name of
	// the flavor that describes the server's specs.
	FlavorName string `json:"-"`

	// SecurityGroups lists the names of the security groups to which this server
	// should belong.
	SecurityGroups []string `json:"-"`

	// UserData contains configuration information or scripts to use upon launch.
	// Create will base64-encode it for you, if it isn't already.
	UserData []byte `json:"-"`

	// AvailabilityZone in which to launch the server.
	AvailabilityZone string `json:"availability_zone,omitempty"`

	// Networks dictates how this server will be attached to available networks.
	// By default, the server will be attached to all isolated networks for the
	// tenant.
	Networks []Network `json:"-"`

	// Metadata contains key-value pairs (up to 255 bytes each) to attach to the
	// server.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Personality includes files to inject into the server at launch.
	// Create will base64-encode file contents for you.
	Personality Personality `json:"personality,omitempty"`

	// ConfigDrive enables metadata injection through a configuration drive.
	ConfigDrive *bool `json:"config_drive,omitempty"`

	// AdminPass sets the root user password. If not set, a randomly-generated
	// password will be created and returned in the response.
	AdminPass string `json:"adminPass,omitempty"`

	// AccessIPv4 specifies an IPv4 address for the instance.
	AccessIPv4 string `json:"accessIPv4,omitempty"`

	// AccessIPv6 pecifies an IPv6 address for the instance.
	AccessIPv6 string `json:"accessIPv6,omitempty"`

	// ServiceClient will allow calls to be made to retrieve an image or
	// flavor ID by name.
	ServiceClient *gophercloud.ServiceClient `json:"-"`
}

// ToServerCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToServerCreateMap() (map[string]interface{}, error) {
	sc := opts.ServiceClient
	opts.ServiceClient = nil
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if opts.UserData != nil {
		var userData string
		if _, err := base64.StdEncoding.DecodeString(string(opts.UserData)); err != nil {
			userData = base64.StdEncoding.EncodeToString(opts.UserData)
		} else {
			userData = string(opts.UserData)
		}
		b["user_data"] = &userData
	}

	if len(opts.SecurityGroups) > 0 {
		securityGroups := make([]map[string]interface{}, len(opts.SecurityGroups))
		for i, groupName := range opts.SecurityGroups {
			securityGroups[i] = map[string]interface{}{"name": groupName}
		}
		b["security_groups"] = securityGroups
	}

	if len(opts.Networks) > 0 {
		networks := make([]map[string]interface{}, len(opts.Networks))
		for i, net := range opts.Networks {
			networks[i] = make(map[string]interface{})
			if net.UUID != "" {
				networks[i]["uuid"] = net.UUID
			}
			if net.Port != "" {
				networks[i]["port"] = net.Port
			}
			if net.FixedIP != "" {
				networks[i]["fixed_ip"] = net.FixedIP
			}
		}
		b["networks"] = networks
	}

	// If ImageRef isn't provided, check if ImageName was provided to ascertain
	// the image ID.
	if opts.ImageRef == "" {
		if opts.ImageName != "" {
			if sc == nil {
				err := ErrNoClientProvidedForIDByName{}
				err.Argument = "ServiceClient"
				return nil, err
			}
			imageID, err := images.IDFromName(sc, opts.ImageName)
			if err != nil {
				return nil, err
			}
			b["imageRef"] = imageID
		}
	}

	// If FlavorRef isn't provided, use FlavorName to ascertain the flavor ID.
	if opts.FlavorRef == "" {
		if opts.FlavorName == "" {
			err := ErrNeitherFlavorIDNorFlavorNameProvided{}
			err.Argument = "FlavorRef/FlavorName"
			return nil, err
		}
		if sc == nil {
			err := ErrNoClientProvidedForIDByName{}
			err.Argument = "ServiceClient"
			return nil, err
		}
		flavorID, err := flavors.IDFromName(sc, opts.FlavorName)
		if err != nil {
			return nil, err
		}
		b["flavorRef"] = flavorID
	}

	return map[string]interface{}{"server": b}, nil
}

// Create requests a server to be provisioned to the user in the current tenant.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	reqBody, err := opts.ToServerCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(listURL(client), reqBody, &r.Body, nil)
	return
}

// Delete requests that a server previously provisioned be removed from your
// account.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ForceDelete forces the deletion of a server.
func ForceDelete(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	_, r.Err = client.Post(actionURL(client, id), map[string]interface{}{"forceDelete": ""}, nil, nil)
	return
}

// Get requests details on a single server, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 203},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional attributes to the
// Update request.
type UpdateOptsBuilder interface {
	ToServerUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts specifies the base attributes that may be updated on an existing
// server.
type UpdateOpts struct {
	// Name changes the displayed name of the server.
	// The server host name will *not* change.
	// Server names are not constrained to be unique, even within the same tenant.
	Name string `json:"name,omitempty"`

	// AccessIPv4 provides a new IPv4 address for the instance.
	AccessIPv4 string `json:"accessIPv4,omitempty"`

	// AccessIPv6 provides a new IPv6 address for the instance.
	AccessIPv6 string `json:"accessIPv6,omitempty"`
}

// ToServerUpdateMap formats an UpdateOpts structure into a request body.
func (opts UpdateOpts) ToServerUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "server")
}

// Update requests that various attributes of the indicated server be changed.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToServerUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// ChangeAdminPassword alters the administrator or root password for a specified
// server.
func ChangeAdminPassword(client *gophercloud.ServiceClient, id, newPassword string) (r ActionResult) {
	b := map[string]interface{}{
		"changePassword": map[string]string{
			"adminPass": newPassword,
		},
	}
	_, r.Err = client.Post(actionURL(client, id), b, nil, nil)
	return
}

// RebootMethod describes the mechanisms by which a server reboot can be requested.
type RebootMethod string

// These constants determine how a server should be rebooted.
// See the Reboot() function for further details.
const (
	SoftReboot RebootMethod = "SOFT"
	HardReboot RebootMethod = "HARD"
	OSReboot                = SoftReboot
	PowerCycle              = HardReboot
)

// RebootOptsBuilder allows extensions to add additional parameters to the
// reboot request.
type RebootOptsBuilder interface {
	ToServerRebootMap() (map[string]interface{}, error)
}

// RebootOpts provides options to the reboot request.
type RebootOpts struct {
	// Type is the type of reboot to perform on the server.
	Type RebootMethod `json:"type" required:"true"`
}

// ToServerRebootMap builds a body for the reboot request.
func (opts *RebootOpts) ToServerRebootMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "reboot")
}

/*
	Reboot requests that a given server reboot.

	Two methods exist for rebooting a server:

	HardReboot (aka PowerCycle) starts the server instance by physically cutting
	power to the machine, or if a VM, terminating it at the hypervisor level.
	It's done. Caput. Full stop.
	Then, after a brief while, power is rtored or the VM instance restarted.

	SoftReboot (aka OSReboot) simply tells the OS to restart under its own
	procedure.
	E.g., in Linux, asking it to enter runlevel 6, or executing
	"sudo shutdown -r now", or by asking Windows to rtart the machine.
*/
func Reboot(client *gophercloud.ServiceClient, id string, opts RebootOptsBuilder) (r ActionResult) {
	b, err := opts.ToServerRebootMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, nil, nil)
	return
}

// RebuildOptsBuilder allows extensions to provide additional parameters to the
// rebuild request.
type RebuildOptsBuilder interface {
	ToServerRebuildMap() (map[string]interface{}, error)
}

// RebuildOpts represents the configuration options used in a server rebuild
// operation.
type RebuildOpts struct {
	// AdminPass is the server's admin password
	AdminPass string `json:"adminPass,omitempty"`

	// ImageID is the ID of the image you want your server to be provisioned on.
	ImageID string `json:"imageRef"`

	// ImageName is readable name of an image.
	ImageName string `json:"-"`

	// Name to set the server to
	Name string `json:"name,omitempty"`

	// AccessIPv4 [optional] provides a new IPv4 address for the instance.
	AccessIPv4 string `json:"accessIPv4,omitempty"`

	// AccessIPv6 [optional] provides a new IPv6 address for the instance.
	AccessIPv6 string `json:"accessIPv6,omitempty"`

	// Metadata [optional] contains key-value pairs (up to 255 bytes each)
	// to attach to the server.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Personality [optional] includes files to inject into the server at launch.
	// Rebuild will base64-encode file contents for you.
	Personality Personality `json:"personality,omitempty"`

	// ServiceClient will allow calls to be made to retrieve an image or
	// flavor ID by name.
	ServiceClient *gophercloud.ServiceClient `json:"-"`
}

// ToServerRebuildMap formats a RebuildOpts struct into a map for use in JSON
func (opts RebuildOpts) ToServerRebuildMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	// If ImageRef isn't provided, check if ImageName was provided to ascertain
	// the image ID.
	if opts.ImageID == "" {
		if opts.ImageName != "" {
			if opts.ServiceClient == nil {
				err := ErrNoClientProvidedForIDByName{}
				err.Argument = "ServiceClient"
				return nil, err
			}
			imageID, err := images.IDFromName(opts.ServiceClient, opts.ImageName)
			if err != nil {
				return nil, err
			}
			b["imageRef"] = imageID
		}
	}

	return map[string]interface{}{"rebuild": b}, nil
}

// Rebuild will reprovision the server according to the configuration options
// provided in the RebuildOpts struct.
func Rebuild(client *gophercloud.ServiceClient, id string, opts RebuildOptsBuilder) (r RebuildResult) {
	b, err := opts.ToServerRebuildMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, &r.Body, nil)
	return
}

// ResizeOptsBuilder allows extensions to add additional parameters to the
// resize request.
type ResizeOptsBuilder interface {
	ToServerResizeMap() (map[string]interface{}, error)
}

// ResizeOpts represents the configuration options used to control a Resize
// operation.
type ResizeOpts struct {
	// FlavorRef is the ID of the flavor you wish your server to become.
	FlavorRef string `json:"flavorRef" required:"true"`
}

// ToServerResizeMap formats a ResizeOpts as a map that can be used as a JSON
// request body for the Resize request.
func (opts ResizeOpts) ToServerResizeMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "resize")
}

// Resize instructs the provider to change the flavor of the server.
//
// Note that this implies rebuilding it.
//
// Unfortunately, one cannot pass rebuild parameters to the resize function.
// When the resize completes, the server will be in RESIZE_VERIFY state.
// While in this state, you can explore the use of the new server's
// configuration. If you like it, call ConfirmResize() to commit the resize
// permanently. Otherwise, call RevertResize() to restore the old configuration.
func Resize(client *gophercloud.ServiceClient, id string, opts ResizeOptsBuilder) (r ActionResult) {
	b, err := opts.ToServerResizeMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, nil, nil)
	return
}

// ConfirmResize confirms a previous resize operation on a server.
// See Resize() for more details.
func ConfirmResize(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	_, r.Err = client.Post(actionURL(client, id), map[string]interface{}{"confirmResize": nil}, nil, &gophercloud.RequestOpts{
		OkCodes: []int{201, 202, 204},
	})
	return
}

// RevertResize cancels a previous resize operation on a server.
// See Resize() for more details.
func RevertResize(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	_, r.Err = client.Post(actionURL(client, id), map[string]interface{}{"revertResize": nil}, nil, nil)
	return
}

// RescueOptsBuilder is an interface that allows extensions to override the
// default structure of a Rescue request.
type RescueOptsBuilder interface {
	ToServerRescueMap() (map[string]interface{}, error)
}

// RescueOpts represents the configuration options used to control a Rescue
// option.
type RescueOpts struct {
	// AdminPass is the desired administrative password for the instance in
	// RESCUE mode. If it's left blank, the server will generate a password.
	AdminPass string `json:"adminPass,omitempty"`
}

// ToServerRescueMap formats a RescueOpts as a map that can be used as a JSON
// request body for the Rescue request.
func (opts RescueOpts) ToServerRescueMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "rescue")
}

// Rescue instructs the provider to place the server into RESCUE mode.
func Rescue(client *gophercloud.ServiceClient, id string, opts RescueOptsBuilder) (r RescueResult) {
	b, err := opts.ToServerRescueMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// ResetMetadataOptsBuilder allows extensions to add additional parameters to
// the Reset request.
type ResetMetadataOptsBuilder interface {
	ToMetadataResetMap() (map[string]interface{}, error)
}

// MetadataOpts is a map that contains key-value pairs.
type MetadataOpts map[string]string

// ToMetadataResetMap assembles a body for a Reset request based on the contents
// of a MetadataOpts.
func (opts MetadataOpts) ToMetadataResetMap() (map[string]interface{}, error) {
	return map[string]interface{}{"metadata": opts}, nil
}

// ToMetadataUpdateMap assembles a body for an Update request based on the
// contents of a MetadataOpts.
func (opts MetadataOpts) ToMetadataUpdateMap() (map[string]interface{}, error) {
	return map[string]interface{}{"metadata": opts}, nil
}

// ResetMetadata will create multiple new key-value pairs for the given server
// ID.
// Note: Using this operation will erase any already-existing metadata and
// create the new metadata provided. To keep any already-existing metadata,
// use the UpdateMetadatas or UpdateMetadata function.
func ResetMetadata(client *gophercloud.ServiceClient, id string, opts ResetMetadataOptsBuilder) (r ResetMetadataResult) {
	b, err := opts.ToMetadataResetMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(metadataURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Metadata requests all the metadata for the given server ID.
func Metadata(client *gophercloud.ServiceClient, id string) (r GetMetadataResult) {
	_, r.Err = client.Get(metadataURL(client, id), &r.Body, nil)
	return
}

// UpdateMetadataOptsBuilder allows extensions to add additional parameters to
// the Create request.
type UpdateMetadataOptsBuilder interface {
	ToMetadataUpdateMap() (map[string]interface{}, error)
}

// UpdateMetadata updates (or creates) all the metadata specified by opts for
// the given server ID. This operation does not affect already-existing metadata
// that is not specified by opts.
func UpdateMetadata(client *gophercloud.ServiceClient, id string, opts UpdateMetadataOptsBuilder) (r UpdateMetadataResult) {
	b, err := opts.ToMetadataUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(metadataURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// MetadatumOptsBuilder allows extensions to add additional parameters to the
// Create request.
type MetadatumOptsBuilder interface {
	ToMetadatumCreateMap() (map[string]interface{}, string, error)
}

// MetadatumOpts is a map of length one that contains a key-value pair.
type MetadatumOpts map[string]string

// ToMetadatumCreateMap assembles a body for a Create request based on the
// contents of a MetadataumOpts.
func (opts MetadatumOpts) ToMetadatumCreateMap() (map[string]interface{}, string, error) {
	if len(opts) != 1 {
		err := gophercloud.ErrInvalidInput{}
		err.Argument = "servers.MetadatumOpts"
		err.Info = "Must have 1 and only 1 key-value pair"
		return nil, "", err
	}
	metadatum := map[string]interface{}{"meta": opts}
	var key string
	for k := range metadatum["meta"].(MetadatumOpts) {
		key = k
	}
	return metadatum, key, nil
}

// CreateMetadatum will create or update the key-value pair with the given key
// for the given server ID.
func CreateMetadatum(client *gophercloud.ServiceClient, id string, opts MetadatumOptsBuilder) (r CreateMetadatumResult) {
	b, key, err := opts.ToMetadatumCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(metadatumURL(client, id, key), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Metadatum requests the key-value pair with the given key for the given
// server ID.
func Metadatum(client *gophercloud.ServiceClient, id, key string) (r GetMetadatumResult) {
	_, r.Err = client.Get(metadatumURL(client, id, key), &r.Body, nil)
	return
}

// DeleteMetadatum will delete the key-value pair with the given key for the
// given server ID.
func DeleteMetadatum(client *gophercloud.ServiceClient, id, key string) (r DeleteMetadatumResult) {
	_, r.Err = client.Delete(metadatumURL(client, id, key), nil)
	return
}

// ListAddresses makes a request against the API to list the servers IP
// addresses.
func ListAddresses(client *gophercloud.ServiceClient, id string) pagination.Pager {
	return pagination.NewPager(client, listAddressesURL(client, id), func(r pagination.PageResult) pagination.Page {
		return AddressPage{pagination.SinglePageBase(r)}
	})
}

// ListAddressesByNetwork makes a request against the API to list the servers IP
// addresses for the given network.
func ListAddressesByNetwork(client *gophercloud.ServiceClient, id, network string) pagination.Pager {
	return pagination.NewPager(client, listAddressesByNetworkURL(client, id, network), func(r pagination.PageResult) pagination.Page {
		return NetworkAddressPage{pagination.SinglePageBase(r)}
	})
}

// CreateImageOptsBuilder allows extensions to add additional parameters to the
// CreateImage request.
type CreateImageOptsBuilder interface {
	ToServerCreateImageMap() (map[string]interface{}, error)
}

// CreateImageOpts provides options to pass to the CreateImage request.
type CreateImageOpts struct {
	// Name of the image/snapshot.
	Name string `json:"name" required:"true"`

	// Metadata contains key-value pairs (up to 255 bytes each) to attach to
	// the created image.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ToServerCreateImageMap formats a CreateImageOpts structure into a request
// body.
func (opts CreateImageOpts) ToServerCreateImageMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "createImage")
}

// CreateImage makes a request against the nova API to schedule an image to be
// created of the server
func CreateImage(client *gophercloud.ServiceClient, id string, opts CreateImageOptsBuilder) (r CreateImageResult) {
	b, err := opts.ToServerCreateImageMap()
	if err != nil {
		r.Err = err
		return
	}
	resp, err := client.Post(actionURL(client, id), b, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	r.Err = err
	r.Header = resp.Header
	return
}

// IDFromName is a convienience function that returns a server's ID given its
// name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	count := 0
	id := ""
	allPages, err := List(client, nil).AllPages()
	if err != nil {
		return "", err
	}

	all, err := ExtractServers(allPages)
	if err != nil {
		return "", err
	}

	for _, f := range all {
		if f.Name == name {
			count++
			id = f.ID
		}
	}

	switch count {
	case 0:
		return "", gophercloud.ErrResourceNotFound{Name: name, ResourceType: "server"}
	case 1:
		return id, nil
	default:
		return "", gophercloud.ErrMultipleResourcesFound{Name: name, Count: count, ResourceType: "server"}
	}
}

// GetPassword makes a request against the nova API to get the encrypted
// administrative password.
func GetPassword(client *gophercloud.ServiceClient, serverId string) (r GetPasswordResult) {
	_, r.Err = client.Get(passwordURL(client, serverId), &r.Body, nil)
	return
}
