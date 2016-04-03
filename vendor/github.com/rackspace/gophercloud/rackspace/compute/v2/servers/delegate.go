package servers

import (
	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
)

// List makes a request against the API to list servers accessible to you.
func List(client *gophercloud.ServiceClient, opts os.ListOptsBuilder) pagination.Pager {
	return os.List(client, opts)
}

// Create requests a server to be provisioned to the user in the current tenant.
func Create(client *gophercloud.ServiceClient, opts os.CreateOptsBuilder) os.CreateResult {
	return os.Create(client, opts)
}

// Update requests an existing server to be updated with the supplied options.
func Update(client *gophercloud.ServiceClient, id string, opts os.UpdateOptsBuilder) os.UpdateResult {
	return os.Update(client, id, opts)
}

// Delete requests that a server previously provisioned be removed from your account.
func Delete(client *gophercloud.ServiceClient, id string) os.DeleteResult {
	return os.Delete(client, id)
}

// Get requests details on a single server, by ID.
func Get(client *gophercloud.ServiceClient, id string) os.GetResult {
	return os.Get(client, id)
}

// ChangeAdminPassword alters the administrator or root password for a specified server.
func ChangeAdminPassword(client *gophercloud.ServiceClient, id, newPassword string) os.ActionResult {
	return os.ChangeAdminPassword(client, id, newPassword)
}

// Reboot requests that a given server reboot. Two methods exist for rebooting a server:
//
// os.HardReboot (aka PowerCycle) restarts the server instance by physically cutting power to the
// machine, or if a VM, terminating it at the hypervisor level. It's done. Caput. Full stop. Then,
// after a brief wait, power is restored or the VM instance restarted.
//
// os.SoftReboot (aka OSReboot) simply tells the OS to restart under its own procedures. E.g., in
// Linux, asking it to enter runlevel 6, or executing "sudo shutdown -r now", or by asking Windows to restart the machine.
func Reboot(client *gophercloud.ServiceClient, id string, how os.RebootMethod) os.ActionResult {
	return os.Reboot(client, id, how)
}

// Rebuild will reprovision the server according to the configuration options provided in the
// RebuildOpts struct.
func Rebuild(client *gophercloud.ServiceClient, id string, opts os.RebuildOptsBuilder) os.RebuildResult {
	return os.Rebuild(client, id, opts)
}

// Resize instructs the provider to change the flavor of the server.
// Note that this implies rebuilding it.
// Unfortunately, one cannot pass rebuild parameters to the resize function.
// When the resize completes, the server will be in RESIZE_VERIFY state.
// While in this state, you can explore the use of the new server's configuration.
// If you like it, call ConfirmResize() to commit the resize permanently.
// Otherwise, call RevertResize() to restore the old configuration.
func Resize(client *gophercloud.ServiceClient, id string, opts os.ResizeOptsBuilder) os.ActionResult {
	return os.Resize(client, id, opts)
}

// ConfirmResize confirms a previous resize operation on a server.
// See Resize() for more details.
func ConfirmResize(client *gophercloud.ServiceClient, id string) os.ActionResult {
	return os.ConfirmResize(client, id)
}

// WaitForStatus will continually poll a server until it successfully transitions to a specified
// status. It will do this for at most the number of seconds specified.
func WaitForStatus(c *gophercloud.ServiceClient, id, status string, secs int) error {
	return os.WaitForStatus(c, id, status, secs)
}

// ExtractServers interprets the results of a single page from a List() call, producing a slice of Server entities.
func ExtractServers(page pagination.Page) ([]os.Server, error) {
	return os.ExtractServers(page)
}

// ListAddresses makes a request against the API to list the servers IP addresses.
func ListAddresses(client *gophercloud.ServiceClient, id string) pagination.Pager {
	return os.ListAddresses(client, id)
}

// ExtractAddresses interprets the results of a single page from a ListAddresses() call, producing a map of Address slices.
func ExtractAddresses(page pagination.Page) (map[string][]os.Address, error) {
	return os.ExtractAddresses(page)
}

// ListAddressesByNetwork makes a request against the API to list the servers IP addresses
// for the given network.
func ListAddressesByNetwork(client *gophercloud.ServiceClient, id, network string) pagination.Pager {
	return os.ListAddressesByNetwork(client, id, network)
}

// ExtractNetworkAddresses interprets the results of a single page from a ListAddressesByNetwork() call, producing a map of Address slices.
func ExtractNetworkAddresses(page pagination.Page) ([]os.Address, error) {
	return os.ExtractNetworkAddresses(page)
}

// Metadata requests all the metadata for the given server ID.
func Metadata(client *gophercloud.ServiceClient, id string) os.GetMetadataResult {
	return os.Metadata(client, id)
}

// UpdateMetadata updates (or creates) all the metadata specified by opts for the given server ID.
// This operation does not affect already-existing metadata that is not specified
// by opts.
func UpdateMetadata(client *gophercloud.ServiceClient, id string, opts os.UpdateMetadataOptsBuilder) os.UpdateMetadataResult {
	return os.UpdateMetadata(client, id, opts)
}
