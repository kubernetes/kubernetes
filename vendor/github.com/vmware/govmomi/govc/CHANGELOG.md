# changelog

### 0.14.0 (2017-04-08)

* Add find command

* Add '-wait' option to vm.ip to allow a non-waiting query

* Add datastore.disk.info command

* Add bash completion script

* Add metric commands: change, ls, info, sample, reset, interval.change, interval.info

### 0.13.0 (2017-03-02)

* Add vm.guest.tools command

* Add datastore.disk.create command

* Add datastore.vsan.dom.ls and datastore.vsan.dom.rm commands

* Add vm.disk.change command

* Add vm.rdm attach and ls commands

* Add '-n' option to vm.ip to wait for a specific NIC

* Add '-annotation' option to vm.create and vm.clone commands

* Add '-sync-time-with-host-' flag to vm.change command

* Add object.collect command (MOB for cli + Emacs)

### 0.12.1 (2016-12-19)

* Add '-f' flag to logs command

* Add storage support to vm.migrate

* Add support for file backed serialport devices

### 0.12.0 (2016-12-01)

* Add optional '-host' flag to datastore download/tail commands

* Support InjectOvfEnv without PowerOn when importing

* Support stdin as import options source

* Add basic NVME controller support

### 0.11.4 (2016-11-15)

* Add role create, remove, update, ls and usage commands

### 0.11.3 (2016-11-08)

* Add `-product-version` flag to dvs.create

* datastore.tail -f will exit without error if the file no longer exists

### 0.11.2 (2016-11-01)

* Add object.reload command

* Add ESX 5.5 support to host.account commands

### 0.11.1 (2016-10-27)

* Add support for VirtualApp in pool.change command

### 0.11.0 (2016-10-25)

* Add object.destroy and object.rename commands

* Remove datacenter.destroy command (use object.destroy instead)

* Remove folder.destroy command (use object.destroy instead)

* Rename folder.move_into -> object.mv

* Add dvs.portgroup.change command

* Add vlan flag to dvs.portgroup.add command

### 0.10.0 (2016-10-20)

* Add generated govc/USAGE.md

* Add host.date info and change commands

* Add session ls and rm commands

* Add `-tls-known-hosts` and `-tls-ca-certs` flags

* Add host.cert commands : info, csr, import

* Add about.cert command (similar to the Chrome Certificate Viewer)

* Add `-vspc-proxy` flag to device.serial.connect command

* Rename license.list -> license.ls, license.assigned.list -> license.assigned.ls

### 0.9.0 (2016-09-09)

* Add `-R` option to datastore.ls

* Add datastore.tail command

* Add vm.migrate command

* Add govc vm.register and vm.unregister commands

* Add govc vm snapshot commands: create, remove, revert, tree

* Add device.usb.add command

* Support stdin/stdout in datastore upload/download

* Add host.portgroup.change command

* Add host.portgroup.info command

* Add HostNetworkPolicy to host.vswitch.info

* Add `-json` support to host.vswitch.info command

* Support instance uuid in SearchFlag

* Add `-json` support to esxcli command

* Add `-unclaimed` flag to host.storage.info command

* Support Network mapping in import.{ova,ovf} commands

### 0.8.0 (2016-06-30)

* If username (`-u` / GOVC_USERNAME) is empty, attempt login via local ticket (Workstation)

* Add StoragePod support to govc folder.create

* Add `-folder` flag to datacenter.create command

* Logout when session persistence is disabled

* Add `-L` flag to ls command for resolving by managed object reference

* Add `-i` flag to ls command for listing the managed object reference

* Add vm.markasvm command

* Add vm.markastemplate command

### 0.7.1 (2016-06-03)

* Fix datastore.{upload,download} against VirtualCenter

### 0.7.0 (2016-06-02)

* Add `-require` flag to version command

* Add support for local type in the datastore.create command

* Add `-namespace` option to datastore.mkdir and datastore.rm to create/remove namespaces on VSANs

* Add host.service command

* Add host.storage.mark command

* Add `-rescan` option to host.storage.info command

### 0.6.0 (2016-04-29)

* Add folder commands: info, create, destroy, rename, moveinto

* Add datastore.info command

* Add `-a` and `-v4` flags to vm.ip command

* Add host.account.{create,update,remove} commands

* Add env command

* Add vm.clone command

### 0.5.0 (2016-03-30)

* Add dvs.portgroup.info command

* Add `-folder` flag to vm.create command

* Add `-dump` flag to OutputFlag

* Add `-f` flag to events command

* Add `-mode` flag to vm.disk.create command

* Add `-net` flag to device.info command

* Add `-eager` and `-thick` options to vm.create command

### 0.4.0 (2016-02-26)

* Add support for placement in datastore cluster to vm.create command

* Add support for creating new disks in vm.create command

* Add `-p` and `-a` options to govc datastore.ls command

### 0.3.0 (2016-01-16)

* Add permissions.{ls,set,remove} commands

* Add datastore.{create,remove} commands.
  The new create command supports both creating NAS and VMFS datastores.

* Add dvs.{create,add} and dvs.portgroup.add commands

* Add host.vnic.{service,info} commands

* Add cluster.{create,change,add} commands

* Add host.{disconnect,reconnect,remove,maintenance.enter,maintenance.exit} commands

* Add license.decode, license.assigned.list and license.assign commands

* Add firewall.ruleset.find command

* Add logs, logs.ls and logs.download commands

* Add support for LoginExtensionByCertificate with new `-cert` and `-key` flags

* Add govc extension.{info,register,unregister,setcert} commands

* Add govc vapp.{info,destroy,power} commands

### 0.2.0 (2015-09-15)

* The `vm.power` guest `-s` and `-r` options will fallback to hard `-off` / `-reset` if tools is unavailable and `-force` flag is given

* Add `PowerOn, InjectOvfEnv, WaitForIP` options to `import.ovf` and `import.ova` option spec file

* Add `import.spec` to produce an example json document

* Add `-options` to `import.ovf` and `import.ova`

* Add `-folder` to `import.ovf` and `import.ova`

* Add `fields` command to manage custom fields

* Add `datastore.info` command

* Add `events` command

* Add `-net.address` (Hardware Address) option to `vm.change` and `vm.create`

* Add `host.add` command to add host to datacenter.

* Add `GOVC_USERNAME` and `GOVC_PASSWORD` to allow overriding username and/or
  password (used when they contain special characters that prevent them from
  being embedded in the URL).

* Add `-e' (ExtraConfig) option to `vm.change` and `vm.info`

* Retry twice on temporary network errors.

* Add `host.autostart` commands to manage VM autostart.

* Add `-persist-session` flag to control whether or not the session is
  persisted to disk (defaults to true).

### 0.1.0 (2015-03-17)

Prior to this version the changes to govc's command set were not documented.
