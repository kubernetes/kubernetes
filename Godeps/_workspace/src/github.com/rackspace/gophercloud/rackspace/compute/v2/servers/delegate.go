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

// WaitForStatus will continually poll a server until it successfully transitions to a specified
// status. It will do this for at most the number of seconds specified.
func WaitForStatus(c *gophercloud.ServiceClient, id, status string, secs int) error {
	return os.WaitForStatus(c, id, status, secs)
}

// ExtractServers interprets the results of a single page from a List() call, producing a slice of Server entities.
func ExtractServers(page pagination.Page) ([]os.Server, error) {
	return os.ExtractServers(page)
}
