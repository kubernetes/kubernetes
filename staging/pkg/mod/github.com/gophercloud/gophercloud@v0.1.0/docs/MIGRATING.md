# Compute

## Floating IPs

* `github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/floatingip` is now `github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/floatingips`
* `floatingips.Associate` and `floatingips.Disassociate` have been removed.
* `floatingips.DisassociateOpts` is now required to disassociate a Floating IP.

## Security Groups

* `secgroups.AddServerToGroup` is now `secgroups.AddServer`.
* `secgroups.RemoveServerFromGroup` is now `secgroups.RemoveServer`.

## Servers

* `servers.Reboot` now requires a `servers.RebootOpts` struct:

  ```golang
  rebootOpts := servers.RebootOpts{
          Type: servers.SoftReboot,
  }
  res := servers.Reboot(client, server.ID, rebootOpts)
  ```

# Identity

## V3

### Tokens

* `Token.ExpiresAt` is now of type `gophercloud.JSONRFC3339Milli` instead of
  `time.Time`
