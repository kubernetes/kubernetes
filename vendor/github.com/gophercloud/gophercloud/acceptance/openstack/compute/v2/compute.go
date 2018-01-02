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
func CreateBootableVolumeServer(t *testing.T, client *gophercloud.ServiceClient, blockDevices []bootfromvolume.BlockDevice) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
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

// CreateFlavor will create a flavor with a random name.
// An error will be returned if the flavor could not be created.
func CreateFlavor(t *testing.T, client *gophercloud.ServiceClient) (*flavors.Flavor, error) {
	flavorName := tools.RandomString("flavor_", 5)
	t.Logf("Attempting to create flavor %s", flavorName)

	isPublic := true
	createOpts := flavors.CreateOpts{
		Name:     flavorName,
		RAM:      1,
		VCPUs:    1,
		Disk:     gophercloud.IntToPointer(1),
		IsPublic: &isPublic,
	}

	flavor, err := flavors.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created flavor %s", flavor.ID)

	return flavor, nil
}

// CreateFloatingIP will allocate a floating IP.
// An error will be returend if one was unable to be allocated.
func CreateFloatingIP(t *testing.T, client *gophercloud.ServiceClient) (*floatingips.FloatingIP, error) {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
func CreateMultiEphemeralServer(t *testing.T, client *gophercloud.ServiceClient, blockDevices []bootfromvolume.BlockDevice) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
		CreateOptsBuilder: serverCreateOpts,
		BlockDevice:       blockDevices,
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

// CreatePrivateFlavor will create a private flavor with a random name.
// An error will be returned if the flavor could not be created.
func CreatePrivateFlavor(t *testing.T, client *gophercloud.ServiceClient) (*flavors.Flavor, error) {
	flavorName := tools.RandomString("flavor_", 5)
	t.Logf("Attempting to create flavor %s", flavorName)

	isPublic := false
	createOpts := flavors.CreateOpts{
		Name:     flavorName,
		RAM:      1,
		VCPUs:    1,
		Disk:     gophercloud.IntToPointer(1),
		IsPublic: &isPublic,
	}

	flavor, err := flavors.Create(client, createOpts).Extract()
	if err != nil {
		return nil, err
	}

	t.Logf("Successfully created flavor %s", flavor.ID)

	return flavor, nil
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
func CreateServer(t *testing.T, client *gophercloud.ServiceClient) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
func CreateServerWithoutImageRef(t *testing.T, client *gophercloud.ServiceClient) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
func CreateServerInServerGroup(t *testing.T, client *gophercloud.ServiceClient, serverGroup *servergroups.ServerGroup) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
		CreateOptsBuilder: serverCreateOpts,
		SchedulerHints: schedulerhints.SchedulerHints{
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
func CreateServerWithPublicKey(t *testing.T, client *gophercloud.ServiceClient, keyPairName string) (*servers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	var server *servers.Server

	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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
		CreateOptsBuilder: serverCreateOpts,
		KeyName:           keyPairName,
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

// DeleteFlavor will delete a flavor. A fatal error will occur if the flavor
// could not be deleted. This works best when using it as a deferred function.
func DeleteFlavor(t *testing.T, client *gophercloud.ServiceClient, flavor *flavors.Flavor) {
	err := flavors.Delete(client, flavor.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete flavor %s", flavor.ID)
	}

	t.Logf("Deleted flavor: %s", flavor.ID)
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
func ResizeServer(t *testing.T, client *gophercloud.ServiceClient, server *servers.Server) error {
	choices, err := clients.AcceptanceTestChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

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

//Convenience method to fill an QuotaSet-UpdateOpts-struct from a QuotaSet-struct
func FillUpdateOptsFromQuotaSet(src quotasets.QuotaSet, dest *quotasets.UpdateOpts) {
	dest.FixedIPs = &src.FixedIPs
	dest.FloatingIPs = &src.FloatingIPs
	dest.InjectedFileContentBytes = &src.InjectedFileContentBytes
	dest.InjectedFilePathBytes = &src.InjectedFilePathBytes
	dest.InjectedFiles = &src.InjectedFiles
	dest.KeyPairs = &src.KeyPairs
	dest.RAM = &src.RAM
	dest.SecurityGroupRules = &src.SecurityGroupRules
	dest.SecurityGroups = &src.SecurityGroups
	dest.Cores = &src.Cores
	dest.Instances = &src.Instances
	dest.ServerGroups = &src.ServerGroups
	dest.ServerGroupMembers = &src.ServerGroupMembers
	dest.MetadataItems = &src.MetadataItems
}
