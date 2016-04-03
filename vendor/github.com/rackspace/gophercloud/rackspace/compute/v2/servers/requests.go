package servers

import (
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/diskconfig"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
)

// CreateOpts specifies all of the options that Rackspace accepts in its Create request, including
// the union of all extensions that Rackspace supports.
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
	Networks []os.Network

	// Metadata [optional] contains key-value pairs (up to 255 bytes each) to attach to the server.
	Metadata map[string]string

	// Personality [optional] includes files to inject into the server at launch.
	// Create will base64-encode file contents for you.
	Personality os.Personality

	// ConfigDrive [optional] enables metadata injection through a configuration drive.
	ConfigDrive bool

	// AdminPass [optional] sets the root user password. If not set, a randomly-generated
	// password will be created and returned in the response.
	AdminPass string

	// Rackspace-specific extensions begin here.

	// KeyPair [optional] specifies the name of the SSH KeyPair to be injected into the newly launched
	// server. See the "keypairs" extension in OpenStack compute v2.
	KeyPair string

	// DiskConfig [optional] controls how the created server's disk is partitioned. See the "diskconfig"
	// extension in OpenStack compute v2.
	DiskConfig diskconfig.DiskConfig

	// BlockDevice [optional] will create the server from a volume, which is created from an image,
	// a snapshot, or another volume.
	BlockDevice []bootfromvolume.BlockDevice
}

// ToServerCreateMap constructs a request body using all of the OpenStack extensions that are
// active on Rackspace.
func (opts CreateOpts) ToServerCreateMap() (map[string]interface{}, error) {
	base := os.CreateOpts{
		Name:             opts.Name,
		ImageRef:         opts.ImageRef,
		ImageName:        opts.ImageName,
		FlavorRef:        opts.FlavorRef,
		FlavorName:       opts.FlavorName,
		SecurityGroups:   opts.SecurityGroups,
		UserData:         opts.UserData,
		AvailabilityZone: opts.AvailabilityZone,
		Networks:         opts.Networks,
		Metadata:         opts.Metadata,
		Personality:      opts.Personality,
		ConfigDrive:      opts.ConfigDrive,
		AdminPass:        opts.AdminPass,
	}

	drive := diskconfig.CreateOptsExt{
		CreateOptsBuilder: base,
		DiskConfig:        opts.DiskConfig,
	}

	res, err := drive.ToServerCreateMap()
	if err != nil {
		return nil, err
	}

	if len(opts.BlockDevice) != 0 {
		bfv := bootfromvolume.CreateOptsExt{
			CreateOptsBuilder: drive,
			BlockDevice:       opts.BlockDevice,
		}

		res, err = bfv.ToServerCreateMap()
		if err != nil {
			return nil, err
		}
	}

	// key_name doesn't actually come from the extension (or at least isn't documented there) so
	// we need to add it manually.
	serverMap := res["server"].(map[string]interface{})
	if opts.KeyPair != "" {
		serverMap["key_name"] = opts.KeyPair
	}

	return res, nil
}

// RebuildOpts represents all of the configuration options used in a server rebuild operation that
// are supported by Rackspace.
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
	Personality os.Personality

	// Rackspace-specific stuff begins here.

	// DiskConfig [optional] controls how the created server's disk is partitioned. See the "diskconfig"
	// extension in OpenStack compute v2.
	DiskConfig diskconfig.DiskConfig
}

// ToServerRebuildMap constructs a request body using all of the OpenStack extensions that are
// active on Rackspace.
func (opts RebuildOpts) ToServerRebuildMap() (map[string]interface{}, error) {
	base := os.RebuildOpts{
		ImageID:     opts.ImageID,
		Name:        opts.Name,
		AdminPass:   opts.AdminPass,
		AccessIPv4:  opts.AccessIPv4,
		AccessIPv6:  opts.AccessIPv6,
		Metadata:    opts.Metadata,
		Personality: opts.Personality,
	}

	drive := diskconfig.RebuildOptsExt{
		RebuildOptsBuilder: base,
		DiskConfig:         opts.DiskConfig,
	}

	return drive.ToServerRebuildMap()
}
