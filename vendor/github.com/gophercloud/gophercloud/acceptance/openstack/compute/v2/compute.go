// Package v2 contains common functions for creating compute-based resources
// for use in acceptance tests. See the `*_test.go` files for example usages.
package v2

import (
	"crypto/rand"
	"crypto/rsa"
	"fmt"
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	dsr "github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/defsecrules"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/floatingips"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/keypairs"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/networks"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/quotasets"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/schedulerhints"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/secgroups"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/servergroups"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/tenantnetworks"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/flavors"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/images"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"

	"golang.org/x/crypto/ssh"
)

// AssociateFloatingIP will associate a floating IP with an instance. An error
// will be returned if the floating IP was unable to be associated.
func AssociateFloatingIP(t *testing.T, client *gophercloud.ServiceClient, floatingIP *floatingips.FloatingIP, server *servers.Server) error {
	associateOpts := floatingips.AssociateOpts{
		FloatingIP: floatingIP.IP,
	}

	t.Logf("Attempting to associate floating IP %s to instance %s", floatingIP.IP, server.ID)
	err := floatingips.AssociateInstance(client, server.ID, associateOpts).ExtractErr()
	if err != nil {
		return err
	}

	return nil
}

// AssociateFloatingIPWithFixedIP will associate a floating IP with an
// instance's specific fixed IP. An error will be returend if the floating IP
// was unable to be associated.
func AssociateFloatingIPWithFixedIP(t *testing.T, client *gophercloud.ServiceClient, floatingIP *floatingips.FloatingIP, server *servers.Server, fixedIP string) error {
	associateOpts := floatingips.AssociateOpts{
		FloatingIP: floatingIP.IP,
		FixedIP:    fixedIP,
	}

	t.Logf("Attempting to associate floating IP %s to fixed IP %s on instance %s", floatingIP.IP, fixedIP, server.ID)
	err := floatingips.AssociateInstance(client, server.ID, associateOpts).ExtractErr()
	if err != nil {
		return err
	}

	return nil
}

// CreateBootableVolumeServer works like CreateServer but is configured with
// one or more block devices defined by passing in []bootfromvolume.BlockDevice.
// An error will be returned if a server was unable to be created.
func CreateBootableVolumeServer(t *testing.T, client *gophercloud.ServiceClient, blockDevices []bootfromvolume.BlockDevice, choices *clients.AcceptanceTestChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create bootable volume server: %s", name)

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
	}

	if blockDevices[0].SourceType == bootfromvolume.SourceImage && blockDevices[0].DestinationType == bootfromvolume.DestinationLocal {
		serverCreateOpts.ImageRef = blockDevices[0].UUID
	}

	server, err = bootfromvolume.Create(client, bootfromvolume.CreateOptsExt{
		serverCreateOpts,
		blockDevices,
	}).Extract()

	if err != nil {
		return server, err
	}

	if err := WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		return server, err
	}

	newServer, err := servers.Get(client, server.ID).Extract()

	return newServer, nil
}

// CreateDefaultRule will create a default security group rule with a
// random port range between 80 and 90. An error will be returned if
// a default rule was unable to be created.
func CreateDefaultRule(t *testing.T, client *gophercloud.ServiceClient) (dsr.DefaultRule, error) {
	createOpts := dsr.CreateOpts{
		FromPort:   tools.RandomInt(80, 89),
		ToPort:     tools.RandomInt(90, 99),
		IPProtocol: "TCP",
		CIDR:       "0.0.0.0/0",
	}

	defaultRule, err := dsr.Create(client, createOpts).Extract()
	if err != nil {
		return *defaultRule, err
	}

	t.Logf("Created default rule: %s", defaultRule.ID)

	return *defaultRule, nil
}

// CreateFloatingIP will allocate a floating IP.
// An error will be returend if one was unable to be allocated.
func CreateFloatingIP(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices) (*floatingips.FloatingIP, error) {
	createOpts := floatingips.CreateOpts{
		Pool: choices.FloatingIPPoolName,
	}
	floatingIP, err := floatingips.Create(client, createOpts).Extract()
	if err != nil {
		return floatingIP, err
	}

	t.Logf("Created floating IP: %s", floatingIP.ID)
	return floatingIP, nil
}

func createKey() (string, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return "", err
	}

	publicKey := privateKey.PublicKey
	pub, err := ssh.NewPublicKey(&publicKey)
	if err != nil {
		return "", err
	}

	pubBytes := ssh.MarshalAuthorizedKey(pub)
	pk := string(pubBytes)
	return pk, nil
}

// CreateKeyPair will create a KeyPair with a random name. An error will occur
// if the keypair failed to be created. An error will be returned if the
// keypair was unable to be created.
func CreateKeyPair(t *testing.T, client *gophercloud.ServiceClient) (*keypairs.KeyPair, error) {
	keyPairName := tools.RandomString("keypair_", 5)

	t.Logf("Attempting to create keypair: %s", keyPairName)
	createOpts := keypairs.CreateOpts{
		Name: keyPairName,
	}
	keyPair, err := keypairs.Create(client, createOpts).Extract()
	if err != nil {
		return keyPair, err
	}

	t.Logf("Created keypair: %s", keyPairName)
	return keyPair, nil
}

// CreateMultiEphemeralServer works like CreateServer but is configured with
// one or more block devices defined by passing in []bootfromvolume.BlockDevice.
// These block devices act like block devices when booting from a volume but
// are actually local ephemeral disks.
// An error will be returned if a server was unable to be created.
func CreateMultiEphemeralServer(t *testing.T, client *gophercloud.ServiceClient, blockDevices []bootfromvolume.BlockDevice, choices *clients.AcceptanceTestChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create bootable volume server: %s", name)

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
	}

	server, err = bootfromvolume.Create(client, bootfromvolume.CreateOptsExt{
		serverCreateOpts,
		blockDevices,
	}).Extract()

	if err != nil {
		return server, err
	}

	if err := WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		return server, err
	}

	newServer, err := servers.Get(client, server.ID).Extract()

	return newServer, nil
}

// CreateSecurityGroup will create a security group with a random name.
// An error will be returned if one was failed to be created.
func CreateSecurityGroup(t *testing.T, client *gophercloud.ServiceClient) (secgroups.SecurityGroup, error) {
	createOpts := secgroups.CreateOpts{
		Name:        tools.RandomString("secgroup_", 5),
		Description: "something",
	}

	securityGroup, err := secgroups.Create(client, createOpts).Extract()
	if err != nil {
		return *securityGroup, err
	}

	t.Logf("Created security group: %s", securityGroup.ID)
	return *securityGroup, nil
}

// CreateSecurityGroupRule will create a security group rule with a random name
// and a random TCP port range between port 80 and 99. An error will be
// returned if the rule failed to be created.
func CreateSecurityGroupRule(t *testing.T, client *gophercloud.ServiceClient, securityGroupID string) (secgroups.Rule, error) {
	createOpts := secgroups.CreateRuleOpts{
		ParentGroupID: securityGroupID,
		FromPort:      tools.RandomInt(80, 89),
		ToPort:        tools.RandomInt(90, 99),
		IPProtocol:    "TCP",
		CIDR:          "0.0.0.0/0",
	}

	rule, err := secgroups.CreateRule(client, createOpts).Extract()
	if err != nil {
		return *rule, err
	}

	t.Logf("Created security group rule: %s", rule.ID)
	return *rule, nil
}

// CreateServer creates a basic instance with a randomly generated name.
// The flavor of the instance will be the value of the OS_FLAVOR_ID environment variable.
// The image will be the value of the OS_IMAGE_ID environment variable.
// The instance will be launched on the network specified in OS_NETWORK_NAME.
// An error will be returned if the instance was unable to be created.
func CreateServer(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s", name)

	pwd := tools.MakeNewPassword("")

	server, err = servers.Create(client, servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		AdminPass: pwd,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
		Metadata: map[string]string{
			"abc": "def",
		},
		Personality: servers.Personality{
			&servers.File{
				Path:     "/etc/test",
				Contents: []byte("hello world"),
			},
		},
	}).Extract()
	if err != nil {
		return server, err
	}

	if err := WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		return server, err
	}

	return server, nil
}

// CreateServerWithoutImageRef creates a basic instance with a randomly generated name.
// The flavor of the instance will be the value of the OS_FLAVOR_ID environment variable.
// The image is intentionally missing to trigger an error.
// The instance will be launched on the network specified in OS_NETWORK_NAME.
// An error will be returned if the instance was unable to be created.
func CreateServerWithoutImageRef(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s", name)

	pwd := tools.MakeNewPassword("")

	server, err = servers.Create(client, servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		AdminPass: pwd,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
		Personality: servers.Personality{
			&servers.File{
				Path:     "/etc/test",
				Contents: []byte("hello world"),
			},
		},
	}).Extract()
	if err != nil {
		return server, err
	}

	if err := WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		return server, err
	}

	return server, nil
}

// CreateServerGroup will create a server with a random name. An error will be
// returned if the server group failed to be created.
func CreateServerGroup(t *testing.T, client *gophercloud.ServiceClient, policy string) (*servergroups.ServerGroup, error) {
	sg, err := servergroups.Create(client, &servergroups.CreateOpts{
		Name:     "test",
		Policies: []string{policy},
	}).Extract()

	if err != nil {
		return sg, err
	}

	return sg, nil
}

// CreateServerInServerGroup works like CreateServer but places the instance in
// a specified Server Group.
func CreateServerInServerGroup(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices, serverGroup *servergroups.ServerGroup) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s", name)

	pwd := tools.MakeNewPassword("")

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		AdminPass: pwd,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
	}

	schedulerHintsOpts := schedulerhints.CreateOptsExt{
		serverCreateOpts,
		schedulerhints.SchedulerHints{
			Group: serverGroup.ID,
		},
	}
	server, err = servers.Create(client, schedulerHintsOpts).Extract()
	if err != nil {
		return server, err
	}

	return server, nil
}

// CreateServerWithPublicKey works the same as CreateServer, but additionally
// configures the server with a specified Key Pair name.
func CreateServerWithPublicKey(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices, keyPairName string) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	networkID, err := GetNetworkIDFromTenantNetworks(t, client, choices.NetworkName)
	if err != nil {
		return server, err
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s", name)

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
		Networks: []servers.Network{
			servers.Network{UUID: networkID},
		},
	}

	server, err = servers.Create(client, keypairs.CreateOptsExt{
		serverCreateOpts,
		keyPairName,
	}).Extract()
	if err != nil {
		return server, err
	}

	if err := WaitForComputeStatus(client, server, "ACTIVE"); err != nil {
		return server, err
	}

	return server, nil
}

// CreateVolumeAttachment will attach a volume to a server. An error will be
// returned if the volume failed to attach.
func CreateVolumeAttachment(t *testing.T, client *gophercloud.ServiceClient, blockClient *gophercloud.ServiceClient, server *servers.Server, volume *volumes.Volume) (*volumeattach.VolumeAttachment, error) {
	volumeAttachOptions := volumeattach.CreateOpts{
		VolumeID: volume.ID,
	}

	t.Logf("Attempting to attach volume %s to server %s", volume.ID, server.ID)
	volumeAttachment, err := volumeattach.Create(client, server.ID, volumeAttachOptions).Extract()
	if err != nil {
		return volumeAttachment, err
	}

	if err := volumes.WaitForStatus(blockClient, volume.ID, "in-use", 60); err != nil {
		return volumeAttachment, err
	}

	return volumeAttachment, nil
}

// DeleteDefaultRule deletes a default security group rule.
// A fatal error will occur if the rule failed to delete. This works best when
// using it as a deferred function.
func DeleteDefaultRule(t *testing.T, client *gophercloud.ServiceClient, defaultRule dsr.DefaultRule) {
	err := dsr.Delete(client, defaultRule.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete default rule %s: %v", defaultRule.ID, err)
	}

	t.Logf("Deleted default rule: %s", defaultRule.ID)
}

// DeleteFloatingIP will de-allocate a floating IP. A fatal error will occur if
// the floating IP failed to de-allocate. This works best when using it as a
// deferred function.
func DeleteFloatingIP(t *testing.T, client *gophercloud.ServiceClient, floatingIP *floatingips.FloatingIP) {
	err := floatingips.Delete(client, floatingIP.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete floating IP %s: %v", floatingIP.ID, err)
	}

	t.Logf("Deleted floating IP: %s", floatingIP.ID)
}

// DeleteKeyPair will delete a specified keypair. A fatal error will occur if
// the keypair failed to be deleted. This works best when used as a deferred
// function.
func DeleteKeyPair(t *testing.T, client *gophercloud.ServiceClient, keyPair *keypairs.KeyPair) {
	err := keypairs.Delete(client, keyPair.Name).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete keypair %s: %v", keyPair.Name, err)
	}

	t.Logf("Deleted keypair: %s", keyPair.Name)
}

// DeleteSecurityGroup will delete a security group. A fatal error will occur
// if the group failed to be deleted. This works best as a deferred function.
func DeleteSecurityGroup(t *testing.T, client *gophercloud.ServiceClient, securityGroup secgroups.SecurityGroup) {
	err := secgroups.Delete(client, securityGroup.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete security group %s: %s", securityGroup.ID, err)
	}

	t.Logf("Deleted security group: %s", securityGroup.ID)
}

// DeleteSecurityGroupRule will delete a security group rule. A fatal error
// will occur if the rule failed to be deleted. This works best when used
// as a deferred function.
func DeleteSecurityGroupRule(t *testing.T, client *gophercloud.ServiceClient, rule secgroups.Rule) {
	err := secgroups.DeleteRule(client, rule.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete rule: %v", err)
	}

	t.Logf("Deleted security group rule: %s", rule.ID)
}

// DeleteServer deletes an instance via its UUID.
// A fatal error will occur if the instance failed to be destroyed. This works
// best when using it as a deferred function.
func DeleteServer(t *testing.T, client *gophercloud.ServiceClient, server *servers.Server) {
	err := servers.Delete(client, server.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete server %s: %s", server.ID, err)
	}

	t.Logf("Deleted server: %s", server.ID)
}

// DeleteServerGroup will delete a server group. A fatal error will occur if
// the server group failed to be deleted. This works best when used as a
// deferred function.
func DeleteServerGroup(t *testing.T, client *gophercloud.ServiceClient, serverGroup *servergroups.ServerGroup) {
	err := servergroups.Delete(client, serverGroup.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete server group %s: %v", serverGroup.ID, err)
	}

	t.Logf("Deleted server group %s", serverGroup.ID)
}

// DeleteVolumeAttachment will disconnect a volume from an instance. A fatal
// error will occur if the volume failed to detach. This works best when used
// as a deferred function.
func DeleteVolumeAttachment(t *testing.T, client *gophercloud.ServiceClient, blockClient *gophercloud.ServiceClient, server *servers.Server, volumeAttachment *volumeattach.VolumeAttachment) {

	err := volumeattach.Delete(client, server.ID, volumeAttachment.VolumeID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to detach volume: %v", err)
	}

	if err := volumes.WaitForStatus(blockClient, volumeAttachment.ID, "available", 60); err != nil {
		t.Fatalf("Unable to wait for volume: %v", err)
	}
	t.Logf("Deleted volume: %s", volumeAttachment.VolumeID)
}

// DisassociateFloatingIP will disassociate a floating IP from an instance. A
// fatal error will occur if the floating IP failed to disassociate. This works
// best when using it as a deferred function.
func DisassociateFloatingIP(t *testing.T, client *gophercloud.ServiceClient, floatingIP *floatingips.FloatingIP, server *servers.Server) {
	disassociateOpts := floatingips.DisassociateOpts{
		FloatingIP: floatingIP.IP,
	}

	err := floatingips.DisassociateInstance(client, server.ID, disassociateOpts).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to disassociate floating IP %s from server %s: %v", floatingIP.IP, server.ID, err)
	}

	t.Logf("Disassociated floating IP %s from server %s", floatingIP.IP, server.ID)
}

// GetNetworkIDFromNetworks will return the network ID from a specified network
// UUID using the os-networks API extension. An error will be returned if the
// network could not be retrieved.
func GetNetworkIDFromNetworks(t *testing.T, client *gophercloud.ServiceClient, networkName string) (string, error) {
	allPages, err := networks.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkList, err := networks.ExtractNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to list networks: %v", err)
	}

	networkID := ""
	for _, network := range networkList {
		t.Logf("Network: %v", network)
		if network.Label == networkName {
			networkID = network.ID
		}
	}

	t.Logf("Found network ID for %s: %s", networkName, networkID)

	return networkID, nil
}

// GetNetworkIDFromTenantNetworks will return the network UUID for a given
// network name using the os-tenant-networks API extension. An error will be
// returned if the network could not be retrieved.
func GetNetworkIDFromTenantNetworks(t *testing.T, client *gophercloud.ServiceClient, networkName string) (string, error) {
	allPages, err := tenantnetworks.List(client).AllPages()
	if err != nil {
		return "", err
	}

	allTenantNetworks, err := tenantnetworks.ExtractNetworks(allPages)
	if err != nil {
		return "", err
	}

	for _, network := range allTenantNetworks {
		if network.Name == networkName {
			return network.ID, nil
		}
	}

	return "", fmt.Errorf("Failed to obtain network ID for network %s", networkName)
}

// ImportPublicKey will create a KeyPair with a random name and a specified
// public key. An error will be returned if the keypair failed to be created.
func ImportPublicKey(t *testing.T, client *gophercloud.ServiceClient, publicKey string) (*keypairs.KeyPair, error) {
	keyPairName := tools.RandomString("keypair_", 5)

	t.Logf("Attempting to create keypair: %s", keyPairName)
	createOpts := keypairs.CreateOpts{
		Name:      keyPairName,
		PublicKey: publicKey,
	}
	keyPair, err := keypairs.Create(client, createOpts).Extract()
	if err != nil {
		return keyPair, err
	}

	t.Logf("Created keypair: %s", keyPairName)
	return keyPair, nil
}

// ResizeServer performs a resize action on an instance. An error will be
// returned if the instance failed to resize.
// The new flavor that the instance will be resized to is specified in OS_FLAVOR_ID_RESIZE.
func ResizeServer(t *testing.T, client *gophercloud.ServiceClient, server *servers.Server, choices *clients.AcceptanceTestChoices) error {
	opts := &servers.ResizeOpts{
		FlavorRef: choices.FlavorIDResize,
	}
	if res := servers.Resize(client, server.ID, opts); res.Err != nil {
		return res.Err
	}

	if err := WaitForComputeStatus(client, server, "VERIFY_RESIZE"); err != nil {
		return err
	}

	return nil
}

// WaitForComputeStatus will poll an instance's status until it either matches
// the specified status or the status becomes ERROR.
func WaitForComputeStatus(client *gophercloud.ServiceClient, server *servers.Server, status string) error {
	return tools.WaitFor(func() (bool, error) {
		latest, err := servers.Get(client, server.ID).Extract()
		if err != nil {
			return false, err
		}

		if latest.Status == status {
			// Success!
			return true, nil
		}

		if latest.Status == "ERROR" {
			return false, fmt.Errorf("Instance in ERROR state")
		}

		return false, nil
	})
}

// PrintServer will print an instance and all of its attributes.
func PrintServer(t *testing.T, server *servers.Server) {
	t.Logf("ID: %s", server.ID)
	t.Logf("TenantID: %s", server.TenantID)
	t.Logf("UserID: %s", server.UserID)
	t.Logf("Name: %s", server.Name)
	t.Logf("Updated: %s", server.Updated)
	t.Logf("Created: %s", server.Created)
	t.Logf("HostID: %s", server.HostID)
	t.Logf("Status: %s", server.Status)
	t.Logf("Progress: %d", server.Progress)
	t.Logf("AccessIPv4: %s", server.AccessIPv4)
	t.Logf("AccessIPv6: %s", server.AccessIPv6)
	t.Logf("Image: %s", server.Image)
	t.Logf("Flavor: %s", server.Flavor)
	t.Logf("Addresses: %#v", server.Addresses)
	t.Logf("Metadata: %#v", server.Metadata)
	t.Logf("Links: %#v", server.Links)
	t.Logf("KeyName: %s", server.KeyName)
	t.Logf("AdminPass: %s", server.AdminPass)
	t.Logf("SecurityGroups: %#v", server.SecurityGroups)
}

// PrintDefaultRule will print a default security group rule and all of its attributes.
func PrintDefaultRule(t *testing.T, defaultRule *dsr.DefaultRule) {
	t.Logf("\tID: %s", defaultRule.ID)
	t.Logf("\tFrom Port: %d", defaultRule.FromPort)
	t.Logf("\tTo Port: %d", defaultRule.ToPort)
	t.Logf("\tIP Protocol: %s", defaultRule.IPProtocol)
	t.Logf("\tIP Range: %s", defaultRule.IPRange.CIDR)
	t.Logf("\tParent Group ID: %s", defaultRule.ParentGroupID)
	t.Logf("\tGroup Tenant ID: %s", defaultRule.Group.TenantID)
	t.Logf("\tGroup Name: %s", defaultRule.Group.Name)
}

// PrintFlavor will print a flavor and all of its attributes.
func PrintFlavor(t *testing.T, flavor *flavors.Flavor) {
	t.Logf("ID: %s", flavor.ID)
	t.Logf("Name: %s", flavor.Name)
	t.Logf("RAM: %d", flavor.RAM)
	t.Logf("Disk: %d", flavor.Disk)
	t.Logf("Swap: %d", flavor.Swap)
	t.Logf("RxTxFactor: %f", flavor.RxTxFactor)
}

// PrintFloatingIP will print a floating IP and all of its attributes.
func PrintFloatingIP(t *testing.T, floatingIP *floatingips.FloatingIP) {
	t.Logf("ID: %s", floatingIP.ID)
	t.Logf("Fixed IP: %s", floatingIP.FixedIP)
	t.Logf("Instance ID: %s", floatingIP.InstanceID)
	t.Logf("IP: %s", floatingIP.IP)
	t.Logf("Pool: %s", floatingIP.Pool)
}

// PrintImage will print an image and all of its attributes.
func PrintImage(t *testing.T, image images.Image) {
	t.Logf("ID: %s", image.ID)
	t.Logf("Name: %s", image.Name)
	t.Logf("MinDisk: %d", image.MinDisk)
	t.Logf("MinRAM: %d", image.MinRAM)
	t.Logf("Status: %s", image.Status)
	t.Logf("Progress: %d", image.Progress)
	t.Logf("Metadata: %#v", image.Metadata)
	t.Logf("Created: %s", image.Created)
	t.Logf("Updated: %s", image.Updated)
}

// PrintKeyPair will print keypair and all of its attributes.
func PrintKeyPair(t *testing.T, keypair *keypairs.KeyPair) {
	t.Logf("Name: %s", keypair.Name)
	t.Logf("Fingerprint: %s", keypair.Fingerprint)
	t.Logf("Public Key: %s", keypair.PublicKey)
	t.Logf("Private Key: %s", keypair.PrivateKey)
	t.Logf("UserID: %s", keypair.UserID)
}

//  PrintNetwork will print an os-networks based network and all of its attributes.
func PrintNetwork(t *testing.T, network *networks.Network) {
	t.Logf("Bridge: %s", network.Bridge)
	t.Logf("BridgeInterface: %s", network.BridgeInterface)
	t.Logf("Broadcast: %s", network.Broadcast)
	t.Logf("CIDR: %s", network.CIDR)
	t.Logf("CIDRv6: %s", network.CIDRv6)
	t.Logf("CreatedAt: %v", network.CreatedAt)
	t.Logf("Deleted: %t", network.Deleted)
	t.Logf("DeletedAt: %v", network.DeletedAt)
	t.Logf("DHCPStart: %s", network.DHCPStart)
	t.Logf("DNS1: %s", network.DNS1)
	t.Logf("DNS2: %s", network.DNS2)
	t.Logf("Gateway: %s", network.Gateway)
	t.Logf("Gatewayv6: %s", network.Gatewayv6)
	t.Logf("Host: %s", network.Host)
	t.Logf("ID: %s", network.ID)
	t.Logf("Injected: %t", network.Injected)
	t.Logf("Label: %s", network.Label)
	t.Logf("MultiHost: %t", network.MultiHost)
	t.Logf("Netmask: %s", network.Netmask)
	t.Logf("Netmaskv6: %s", network.Netmaskv6)
	t.Logf("Priority: %d", network.Priority)
	t.Logf("ProjectID: %s", network.ProjectID)
	t.Logf("RXTXBase: %d", network.RXTXBase)
	t.Logf("UpdatedAt: %v", network.UpdatedAt)
	t.Logf("VLAN: %d", network.VLAN)
	t.Logf("VPNPrivateAddress: %s", network.VPNPrivateAddress)
	t.Logf("VPNPublicAddress: %s", network.VPNPublicAddress)
	t.Logf("VPNPublicPort: %d", network.VPNPublicPort)
}

//  PrintQuotaSet will print a quota set and all of its attributes.
func PrintQuotaSet(t *testing.T, quotaSet *quotasets.QuotaSet) {
	t.Logf("instances: %d\n", quotaSet.Instances)
	t.Logf("cores: %d\n", quotaSet.Cores)
	t.Logf("ram: %d\n", quotaSet.Ram)
	t.Logf("key_pairs: %d\n", quotaSet.KeyPairs)
	t.Logf("metadata_items: %d\n", quotaSet.MetadataItems)
	t.Logf("security_groups: %d\n", quotaSet.SecurityGroups)
	t.Logf("security_group_rules: %d\n", quotaSet.SecurityGroupRules)
	t.Logf("fixed_ips: %d\n", quotaSet.FixedIps)
	t.Logf("floating_ips: %d\n", quotaSet.FloatingIps)
	t.Logf("injected_file_content_bytes: %d\n", quotaSet.InjectedFileContentBytes)
	t.Logf("injected_file_path_bytes: %d\n", quotaSet.InjectedFilePathBytes)
	t.Logf("injected_files: %d\n", quotaSet.InjectedFiles)
}

//  PrintSecurityGroup will print a security group and all of its attributes and rules.
func PrintSecurityGroup(t *testing.T, securityGroup *secgroups.SecurityGroup) {
	t.Logf("ID: %s", securityGroup.ID)
	t.Logf("Name: %s", securityGroup.Name)
	t.Logf("Description: %s", securityGroup.Description)
	t.Logf("Tenant ID: %s", securityGroup.TenantID)
	t.Logf("Rules:")

	for _, rule := range securityGroup.Rules {
		t.Logf("\tID: %s", rule.ID)
		t.Logf("\tFrom Port: %d", rule.FromPort)
		t.Logf("\tTo Port: %d", rule.ToPort)
		t.Logf("\tIP Protocol: %s", rule.IPProtocol)
		t.Logf("\tIP Range: %s", rule.IPRange.CIDR)
		t.Logf("\tParent Group ID: %s", rule.ParentGroupID)
		t.Logf("\tGroup Tenant ID: %s", rule.Group.TenantID)
		t.Logf("\tGroup Name: %s", rule.Group.Name)
	}
}

// PrintServerGroup will print a server group and all of its attributes.
func PrintServerGroup(t *testing.T, serverGroup *servergroups.ServerGroup) {
	t.Logf("ID: %s", serverGroup.ID)
	t.Logf("Name: %s", serverGroup.Name)
	t.Logf("Policies: %#v", serverGroup.Policies)
	t.Logf("Members: %#v", serverGroup.Members)
	t.Logf("Metadata: %#v", serverGroup.Metadata)
}

// PrintTenantNetwork will print an os-tenant-networks based network and all of its attributes.
func PrintTenantNetwork(t *testing.T, network *tenantnetworks.Network) {
	t.Logf("ID: %s", network.ID)
	t.Logf("Name: %s", network.Name)
	t.Logf("CIDR: %s", network.CIDR)
}

// PrintVolumeAttachment will print a volume attachment and all of its attributes.
func PrintVolumeAttachment(t *testing.T, volumeAttachment *volumeattach.VolumeAttachment) {
	t.Logf("ID: %s", volumeAttachment.ID)
	t.Logf("Device: %s", volumeAttachment.Device)
	t.Logf("VolumeID: %s", volumeAttachment.VolumeID)
	t.Logf("ServerID: %s", volumeAttachment.ServerID)
}
