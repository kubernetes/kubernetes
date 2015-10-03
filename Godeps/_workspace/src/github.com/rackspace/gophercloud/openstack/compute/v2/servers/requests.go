package servers

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/compute/v2/flavors"
	"github.com/rackspace/gophercloud/openstack/compute/v2/images"
	"github.com/rackspace/gophercloud/pagination"
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
	// A time/date stamp for when the server last changed status.
	ChangesSince string `q:"changes-since"`

	// Name of the image in URL format.
	Image string `q:"image"`

	// Name of the flavor in URL format.
	Flavor string `q:"flavor"`

	// Name of the server as a string; can be queried with regular expressions.
	// Realize that ?name=bob returns both bob and bobb. If you need to match bob
	// only, you can use a regular expression matching the syntax of the
	// underlying database server implemented for Compute.
	Name string `q:"name"`

	// Value of the status of the server so that you can filter on "ACTIVE" for example.
	Status string `q:"status"`

	// Name of the host as a string.
	Host string `q:"host"`

	// UUID of the server at which you want to set a marker.
	Marker string `q:"marker"`

	// Integer value for the limit of values to return.
	Limit int `q:"limit"`

	// Bool to show all tenants
	AllTenants bool `q:"all_tenants"`
}

// ToServerListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToServerListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
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

	createPageFn := func(r pagination.PageResult) pagination.Page {
		return ServerPage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, url, createPageFn)
}

// CreateOptsBuilder describes struct types that can be accepted by the Create call.
// The CreateOpts struct in this package does.
type CreateOptsBuilder interface {
	ToServerCreateMap() (map[string]interface{}, error)
}

// Network is used within CreateOpts to control a new server's network attachments.
type Network struct {
	// UUID of a nova-network to attach to the newly provisioned server.
	// Required unless Port is provided.
	UUID string

	// Port of a neutron network to attach to the newly provisioned server.
	// Required unless UUID is provided.
	Port string

	// FixedIP [optional] specifies a fixed IPv4 address to be used on this network.
	FixedIP string
}

// Personality is an array of files that are injected into the server at launch.
type Personality []*File

// File is used within CreateOpts and RebuildOpts to inject a file into the server at launch.
// File implements the json.Marshaler interface, so when a Create or Rebuild operation is requested,
// json.Marshal will call File's MarshalJSON method.
type File struct {
	// Path of the file
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
	// Name [required] is the name to assign to the newly launched server.
	Name string

	// ImageRef [optional; required if ImageName is not provided] is the ID or full
	// URL to the image that contains the server's OS and initial state.
	// Also optional if using the boot-from-volume extension.
	ImageRef string

	// ImageName [optional; required if ImageRef is not provided] is the name of the
	// image that contains the server's OS and initial state.
	// Also optional if using the boot-from-volume extension.
	ImageName string

	// FlavorRef [optional; required if FlavorName is not provided] is the ID or
	// full URL to the flavor that describes the server's specs.
	FlavorRef string

	// FlavorName [optional; required if FlavorRef is not provided] is the name of
	// the flavor that describes the server's specs.
	FlavorName string

	// SecurityGroups [optional] lists the names of the security groups to which this server should belong.
	SecurityGroups []string

	// UserData [optional] contains configuration information or scripts to use upon launch.
	// Create will base64-encode it for you.
	UserData []byte

	// AvailabilityZone [optional] in which to launch the server.
	AvailabilityZone string

	// Networks [optional] dictates how this server will be attached to available networks.
	// By default, the server will be attached to all isolated networks for the tenant.
	Networks []Network

	// Metadata [optional] contains key-value pairs (up to 255 bytes each) to attach to the server.
	Metadata map[string]string

	// Personality [optional] includes files to inject into the server at launch.
	// Create will base64-encode file contents for you.
	Personality Personality

	// ConfigDrive [optional] enables metadata injection through a configuration drive.
	ConfigDrive bool

	// AdminPass [optional] sets the root user password. If not set, a randomly-generated
	// password will be created and returned in the response.
	AdminPass string

	// AccessIPv4 [optional] specifies an IPv4 address for the instance.
	AccessIPv4 string

	// AccessIPv6 [optional] specifies an IPv6 address for the instance.
	AccessIPv6 string
}

// ToServerCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToServerCreateMap() (map[string]interface{}, error) {
	server := make(map[string]interface{})

	server["name"] = opts.Name
	server["imageRef"] = opts.ImageRef
	server["imageName"] = opts.ImageName
	server["flavorRef"] = opts.FlavorRef
	server["flavorName"] = opts.FlavorName

	if opts.UserData != nil {
		encoded := base64.StdEncoding.EncodeToString(opts.UserData)
		server["user_data"] = &encoded
	}
	if opts.ConfigDrive {
		server["config_drive"] = "true"
	}
	if opts.AvailabilityZone != "" {
		server["availability_zone"] = opts.AvailabilityZone
	}
	if opts.Metadata != nil {
		server["metadata"] = opts.Metadata
	}
	if opts.AdminPass != "" {
		server["adminPass"] = opts.AdminPass
	}
	if opts.AccessIPv4 != "" {
		server["accessIPv4"] = opts.AccessIPv4
	}
	if opts.AccessIPv6 != "" {
		server["accessIPv6"] = opts.AccessIPv6
	}

	if len(opts.SecurityGroups) > 0 {
		securityGroups := make([]map[string]interface{}, len(opts.SecurityGroups))
		for i, groupName := range opts.SecurityGroups {
			securityGroups[i] = map[string]interface{}{"name": groupName}
		}
		server["security_groups"] = securityGroups
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
		server["networks"] = networks
	}

	if len(opts.Personality) > 0 {
		server["personality"] = opts.Personality
	}

	return map[string]interface{}{"server": server}, nil
}

// Create requests a server to be provisioned to the user in the current tenant.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToServerCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// If ImageRef isn't provided, use ImageName to ascertain the image ID.
	if reqBody["server"].(map[string]interface{})["imageRef"].(string) == "" {
		imageName := reqBody["server"].(map[string]interface{})["imageName"].(string)
		if imageName == "" {
			res.Err = errors.New("One and only one of ImageRef and ImageName must be provided.")
			return res
		}
		imageID, err := images.IDFromName(client, imageName)
		if err != nil {
			res.Err = err
			return res
		}
		reqBody["server"].(map[string]interface{})["imageRef"] = imageID
	}
	delete(reqBody["server"].(map[string]interface{}), "imageName")

	// If FlavorRef isn't provided, use FlavorName to ascertain the flavor ID.
	if reqBody["server"].(map[string]interface{})["flavorRef"].(string) == "" {
		flavorName := reqBody["server"].(map[string]interface{})["flavorName"].(string)
		if flavorName == "" {
			res.Err = errors.New("One and only one of FlavorRef and FlavorName must be provided.")
			return res
		}
		flavorID, err := flavors.IDFromName(client, flavorName)
		if err != nil {
			res.Err = err
			return res
		}
		reqBody["server"].(map[string]interface{})["flavorRef"] = flavorID
	}
	delete(reqBody["server"].(map[string]interface{}), "flavorName")

	_, res.Err = client.Post(listURL(client), reqBody, &res.Body, nil)
	return res
}

// Delete requests that a server previously provisioned be removed from your account.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var res DeleteResult
	_, res.Err = client.Delete(deleteURL(client, id), nil)
	return res
}

// Get requests details on a single server, by ID.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var result GetResult
	_, result.Err = client.Get(getURL(client, id), &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 203},
	})
	return result
}

// UpdateOptsBuilder allows extensions to add additional attributes to the Update request.
type UpdateOptsBuilder interface {
	ToServerUpdateMap() map[string]interface{}
}

// UpdateOpts specifies the base attributes that may be updated on an existing server.
type UpdateOpts struct {
	// Name [optional] changes the displayed name of the server.
	// The server host name will *not* change.
	// Server names are not constrained to be unique, even within the same tenant.
	Name string

	// AccessIPv4 [optional] provides a new IPv4 address for the instance.
	AccessIPv4 string

	// AccessIPv6 [optional] provides a new IPv6 address for the instance.
	AccessIPv6 string
}

// ToServerUpdateMap formats an UpdateOpts structure into a request body.
func (opts UpdateOpts) ToServerUpdateMap() map[string]interface{} {
	server := make(map[string]string)
	if opts.Name != "" {
		server["name"] = opts.Name
	}
	if opts.AccessIPv4 != "" {
		server["accessIPv4"] = opts.AccessIPv4
	}
	if opts.AccessIPv6 != "" {
		server["accessIPv6"] = opts.AccessIPv6
	}
	return map[string]interface{}{"server": server}
}

// Update requests that various attributes of the indicated server be changed.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) UpdateResult {
	var result UpdateResult
	reqBody := opts.ToServerUpdateMap()
	_, result.Err = client.Put(updateURL(client, id), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return result
}

// ChangeAdminPassword alters the administrator or root password for a specified server.
func ChangeAdminPassword(client *gophercloud.ServiceClient, id, newPassword string) ActionResult {
	var req struct {
		ChangePassword struct {
			AdminPass string `json:"adminPass"`
		} `json:"changePassword"`
	}

	req.ChangePassword.AdminPass = newPassword

	var res ActionResult
	_, res.Err = client.Post(actionURL(client, id), req, nil, nil)
	return res
}

// ErrArgument errors occur when an argument supplied to a package function
// fails to fall within acceptable values.  For example, the Reboot() function
// expects the "how" parameter to be one of HardReboot or SoftReboot.  These
// constants are (currently) strings, leading someone to wonder if they can pass
// other string values instead, perhaps in an effort to break the API of their
// provider.  Reboot() returns this error in this situation.
//
// Function identifies which function was called/which function is generating
// the error.
// Argument identifies which formal argument was responsible for producing the
// error.
// Value provides the value as it was passed into the function.
type ErrArgument struct {
	Function, Argument string
	Value              interface{}
}

// Error yields a useful diagnostic for debugging purposes.
func (e *ErrArgument) Error() string {
	return fmt.Sprintf("Bad argument in call to %s, formal parameter %s, value %#v", e.Function, e.Argument, e.Value)
}

func (e *ErrArgument) String() string {
	return e.Error()
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

// Reboot requests that a given server reboot.
// Two methods exist for rebooting a server:
//
// HardReboot (aka PowerCycle) restarts the server instance by physically cutting power to the machine, or if a VM,
// terminating it at the hypervisor level.
// It's done. Caput. Full stop.
// Then, after a brief while, power is restored or the VM instance restarted.
//
// SoftReboot (aka OSReboot) simply tells the OS to restart under its own procedures.
// E.g., in Linux, asking it to enter runlevel 6, or executing "sudo shutdown -r now", or by asking Windows to restart the machine.
func Reboot(client *gophercloud.ServiceClient, id string, how RebootMethod) ActionResult {
	var res ActionResult

	if (how != SoftReboot) && (how != HardReboot) {
		res.Err = &ErrArgument{
			Function: "Reboot",
			Argument: "how",
			Value:    how,
		}
		return res
	}

	reqBody := struct {
		C map[string]string `json:"reboot"`
	}{
		map[string]string{"type": string(how)},
	}

	_, res.Err = client.Post(actionURL(client, id), reqBody, nil, nil)
	return res
}

// RebuildOptsBuilder is an interface that allows extensions to override the
// default behaviour of rebuild options
type RebuildOptsBuilder interface {
	ToServerRebuildMap() (map[string]interface{}, error)
}

// RebuildOpts represents the configuration options used in a server rebuild
// operation
type RebuildOpts struct {
	// Required. The ID of the image you want your server to be provisioned on
	ImageID string

	// Name to set the server to
	Name string

	// Required. The server's admin password
	AdminPass string

	// AccessIPv4 [optional] provides a new IPv4 address for the instance.
	AccessIPv4 string

	// AccessIPv6 [optional] provides a new IPv6 address for the instance.
	AccessIPv6 string

	// Metadata [optional] contains key-value pairs (up to 255 bytes each) to attach to the server.
	Metadata map[string]string

	// Personality [optional] includes files to inject into the server at launch.
	// Rebuild will base64-encode file contents for you.
	Personality Personality
}

// ToServerRebuildMap formats a RebuildOpts struct into a map for use in JSON
func (opts RebuildOpts) ToServerRebuildMap() (map[string]interface{}, error) {
	var err error
	server := make(map[string]interface{})

	if opts.AdminPass == "" {
		err = fmt.Errorf("AdminPass is required")
	}

	if opts.ImageID == "" {
		err = fmt.Errorf("ImageID is required")
	}

	if err != nil {
		return server, err
	}

	server["name"] = opts.Name
	server["adminPass"] = opts.AdminPass
	server["imageRef"] = opts.ImageID

	if opts.AccessIPv4 != "" {
		server["accessIPv4"] = opts.AccessIPv4
	}

	if opts.AccessIPv6 != "" {
		server["accessIPv6"] = opts.AccessIPv6
	}

	if opts.Metadata != nil {
		server["metadata"] = opts.Metadata
	}

	if len(opts.Personality) > 0 {
		server["personality"] = opts.Personality
	}

	return map[string]interface{}{"rebuild": server}, nil
}

// Rebuild will reprovision the server according to the configuration options
// provided in the RebuildOpts struct.
func Rebuild(client *gophercloud.ServiceClient, id string, opts RebuildOptsBuilder) RebuildResult {
	var result RebuildResult

	if id == "" {
		result.Err = fmt.Errorf("ID is required")
		return result
	}

	reqBody, err := opts.ToServerRebuildMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Post(actionURL(client, id), reqBody, &result.Body, nil)
	return result
}

// ResizeOptsBuilder is an interface that allows extensions to override the default structure of
// a Resize request.
type ResizeOptsBuilder interface {
	ToServerResizeMap() (map[string]interface{}, error)
}

// ResizeOpts represents the configuration options used to control a Resize operation.
type ResizeOpts struct {
	// FlavorRef is the ID of the flavor you wish your server to become.
	FlavorRef string
}

// ToServerResizeMap formats a ResizeOpts as a map that can be used as a JSON request body for the
// Resize request.
func (opts ResizeOpts) ToServerResizeMap() (map[string]interface{}, error) {
	resize := map[string]interface{}{
		"flavorRef": opts.FlavorRef,
	}

	return map[string]interface{}{"resize": resize}, nil
}

// Resize instructs the provider to change the flavor of the server.
// Note that this implies rebuilding it.
// Unfortunately, one cannot pass rebuild parameters to the resize function.
// When the resize completes, the server will be in RESIZE_VERIFY state.
// While in this state, you can explore the use of the new server's configuration.
// If you like it, call ConfirmResize() to commit the resize permanently.
// Otherwise, call RevertResize() to restore the old configuration.
func Resize(client *gophercloud.ServiceClient, id string, opts ResizeOptsBuilder) ActionResult {
	var res ActionResult
	reqBody, err := opts.ToServerResizeMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Post(actionURL(client, id), reqBody, nil, nil)
	return res
}

// ConfirmResize confirms a previous resize operation on a server.
// See Resize() for more details.
func ConfirmResize(client *gophercloud.ServiceClient, id string) ActionResult {
	var res ActionResult

	reqBody := map[string]interface{}{"confirmResize": nil}
	_, res.Err = client.Post(actionURL(client, id), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{201, 202, 204},
	})
	return res
}

// RevertResize cancels a previous resize operation on a server.
// See Resize() for more details.
func RevertResize(client *gophercloud.ServiceClient, id string) ActionResult {
	var res ActionResult
	reqBody := map[string]interface{}{"revertResize": nil}
	_, res.Err = client.Post(actionURL(client, id), reqBody, nil, nil)
	return res
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
	AdminPass string
}

// ToServerRescueMap formats a RescueOpts as a map that can be used as a JSON
// request body for the Rescue request.
func (opts RescueOpts) ToServerRescueMap() (map[string]interface{}, error) {
	server := make(map[string]interface{})
	if opts.AdminPass != "" {
		server["adminPass"] = opts.AdminPass
	}
	return map[string]interface{}{"rescue": server}, nil
}

// Rescue instructs the provider to place the server into RESCUE mode.
func Rescue(client *gophercloud.ServiceClient, id string, opts RescueOptsBuilder) RescueResult {
	var result RescueResult

	if id == "" {
		result.Err = fmt.Errorf("ID is required")
		return result
	}
	reqBody, err := opts.ToServerRescueMap()
	if err != nil {
		result.Err = err
		return result
	}

	_, result.Err = client.Post(actionURL(client, id), reqBody, &result.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	return result
}

// ResetMetadataOptsBuilder allows extensions to add additional parameters to the
// Reset request.
type ResetMetadataOptsBuilder interface {
	ToMetadataResetMap() (map[string]interface{}, error)
}

// MetadataOpts is a map that contains key-value pairs.
type MetadataOpts map[string]string

// ToMetadataResetMap assembles a body for a Reset request based on the contents of a MetadataOpts.
func (opts MetadataOpts) ToMetadataResetMap() (map[string]interface{}, error) {
	return map[string]interface{}{"metadata": opts}, nil
}

// ToMetadataUpdateMap assembles a body for an Update request based on the contents of a MetadataOpts.
func (opts MetadataOpts) ToMetadataUpdateMap() (map[string]interface{}, error) {
	return map[string]interface{}{"metadata": opts}, nil
}

// ResetMetadata will create multiple new key-value pairs for the given server ID.
// Note: Using this operation will erase any already-existing metadata and create
// the new metadata provided. To keep any already-existing metadata, use the
// UpdateMetadatas or UpdateMetadata function.
func ResetMetadata(client *gophercloud.ServiceClient, id string, opts ResetMetadataOptsBuilder) ResetMetadataResult {
	var res ResetMetadataResult
	metadata, err := opts.ToMetadataResetMap()
	if err != nil {
		res.Err = err
		return res
	}
	_, res.Err = client.Put(metadataURL(client, id), metadata, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Metadata requests all the metadata for the given server ID.
func Metadata(client *gophercloud.ServiceClient, id string) GetMetadataResult {
	var res GetMetadataResult
	_, res.Err = client.Get(metadataURL(client, id), &res.Body, nil)
	return res
}

// UpdateMetadataOptsBuilder allows extensions to add additional parameters to the
// Create request.
type UpdateMetadataOptsBuilder interface {
	ToMetadataUpdateMap() (map[string]interface{}, error)
}

// UpdateMetadata updates (or creates) all the metadata specified by opts for the given server ID.
// This operation does not affect already-existing metadata that is not specified
// by opts.
func UpdateMetadata(client *gophercloud.ServiceClient, id string, opts UpdateMetadataOptsBuilder) UpdateMetadataResult {
	var res UpdateMetadataResult
	metadata, err := opts.ToMetadataUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}
	_, res.Err = client.Post(metadataURL(client, id), metadata, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// MetadatumOptsBuilder allows extensions to add additional parameters to the
// Create request.
type MetadatumOptsBuilder interface {
	ToMetadatumCreateMap() (map[string]interface{}, string, error)
}

// MetadatumOpts is a map of length one that contains a key-value pair.
type MetadatumOpts map[string]string

// ToMetadatumCreateMap assembles a body for a Create request based on the contents of a MetadataumOpts.
func (opts MetadatumOpts) ToMetadatumCreateMap() (map[string]interface{}, string, error) {
	if len(opts) != 1 {
		return nil, "", errors.New("CreateMetadatum operation must have 1 and only 1 key-value pair.")
	}
	metadatum := map[string]interface{}{"meta": opts}
	var key string
	for k := range metadatum["meta"].(MetadatumOpts) {
		key = k
	}
	return metadatum, key, nil
}

// CreateMetadatum will create or update the key-value pair with the given key for the given server ID.
func CreateMetadatum(client *gophercloud.ServiceClient, id string, opts MetadatumOptsBuilder) CreateMetadatumResult {
	var res CreateMetadatumResult
	metadatum, key, err := opts.ToMetadatumCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = client.Put(metadatumURL(client, id, key), metadatum, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Metadatum requests the key-value pair with the given key for the given server ID.
func Metadatum(client *gophercloud.ServiceClient, id, key string) GetMetadatumResult {
	var res GetMetadatumResult
	_, res.Err = client.Request("GET", metadatumURL(client, id, key), gophercloud.RequestOpts{
		JSONResponse: &res.Body,
	})
	return res
}

// DeleteMetadatum will delete the key-value pair with the given key for the given server ID.
func DeleteMetadatum(client *gophercloud.ServiceClient, id, key string) DeleteMetadatumResult {
	var res DeleteMetadatumResult
	_, res.Err = client.Delete(metadatumURL(client, id, key), nil)
	return res
}

// ListAddresses makes a request against the API to list the servers IP addresses.
func ListAddresses(client *gophercloud.ServiceClient, id string) pagination.Pager {
	createPageFn := func(r pagination.PageResult) pagination.Page {
		return AddressPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, listAddressesURL(client, id), createPageFn)
}

// ListAddressesByNetwork makes a request against the API to list the servers IP addresses
// for the given network.
func ListAddressesByNetwork(client *gophercloud.ServiceClient, id, network string) pagination.Pager {
	createPageFn := func(r pagination.PageResult) pagination.Page {
		return NetworkAddressPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(client, listAddressesByNetworkURL(client, id, network), createPageFn)
}

type CreateImageOpts struct {
	// Name [required] of the image/snapshot
	Name string
	// Metadata [optional] contains key-value pairs (up to 255 bytes each) to attach to the created image.
	Metadata map[string]string
}

type CreateImageOptsBuilder interface {
	ToServerCreateImageMap() (map[string]interface{}, error)
}

// ToServerCreateImageMap formats a CreateImageOpts structure into a request body.
func (opts CreateImageOpts) ToServerCreateImageMap() (map[string]interface{}, error) {
	var err error
	img := make(map[string]interface{})
	if opts.Name == "" {
		return nil, fmt.Errorf("Cannot create a server image without a name")
	}
	img["name"] = opts.Name
	if opts.Metadata != nil {
		img["metadata"] = opts.Metadata
	}
	createImage := make(map[string]interface{})
	createImage["createImage"] = img
	return createImage, err
}

// CreateImage makes a request against the nova API to schedule an image to be created of the server
func CreateImage(client *gophercloud.ServiceClient, serverId string, opts CreateImageOptsBuilder) CreateImageResult {
	var res CreateImageResult
	reqBody, err := opts.ToServerCreateImageMap()
	if err != nil {
		res.Err = err
		return res
	}
	response, err := client.Post(actionURL(client, serverId), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	res.Err = err
	res.Header = response.Header
	return res
}

// IDFromName is a convienience function that returns a server's ID given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	serverCount := 0
	serverID := ""
	if name == "" {
		return "", fmt.Errorf("A server name must be provided.")
	}
	pager := List(client, nil)
	pager.EachPage(func(page pagination.Page) (bool, error) {
		serverList, err := ExtractServers(page)
		if err != nil {
			return false, err
		}

		for _, s := range serverList {
			if s.Name == name {
				serverCount++
				serverID = s.ID
			}
		}
		return true, nil
	})

	switch serverCount {
	case 0:
		return "", fmt.Errorf("Unable to find server: %s", name)
	case 1:
		return serverID, nil
	default:
		return "", fmt.Errorf("Found %d servers matching %s", serverCount, name)
	}
}
