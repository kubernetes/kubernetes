# govc usage

This document is generated from `govc -h` and `govc $cmd -h` commands.

The following common options are filtered out in this document,
but appear via `govc $cmd -h`:

```
  -cert=                    Certificate [GOVC_CERTIFICATE]
  -debug=false              Store debug logs [GOVC_DEBUG]
  -dump=false               Enable output dump
  -json=false               Enable JSON output
  -k=false                  Skip verification of server certificate [GOVC_INSECURE]
  -key=                     Private key [GOVC_PRIVATE_KEY]
  -persist-session=true     Persist session to disk [GOVC_PERSIST_SESSION]
  -tls-ca-certs=            TLS CA certificates file [GOVC_TLS_CA_CERTS]
  -tls-known-hosts=         TLS known hosts file [GOVC_TLS_KNOWN_HOSTS]
  -u=                       ESX or vCenter URL [GOVC_URL]
  -vim-namespace=urn:vim25  Vim namespace [GOVC_VIM_NAMESPACE]
  -vim-version=6.0          Vim version [GOVC_VIM_VERSION]
  -dc=                      Datacenter [GOVC_DATACENTER]
  -host.dns=                Find host by FQDN
  -host.ip=                 Find host by IP address
  -host.ipath=              Find host by inventory path
  -host.uuid=               Find host by UUID
  -vm.dns=                  Find VM by FQDN
  -vm.ip=                   Find VM by IP address
  -vm.ipath=                Find VM by inventory path
  -vm.path=                 Find VM by path to .vmx file
  -vm.uuid=                 Find VM by UUID
```

## about

```
Usage: govc about [OPTIONS]

Display About info for HOST.

System information including the name, type, version, and build number.

Examples:
  govc about
  govc about -json | jq -r .About.ProductLineId

Options:
  -l=false                  Include service content
```

## about.cert

```
Usage: govc about.cert [OPTIONS]

Display TLS certificate info for HOST.

If the HOST certificate cannot be verified, about.cert will return with exit code 60 (as curl does).
If the '-k' flag is provided, about.cert will return with exit code 0 in this case.
The SHA1 thumbprint can also be used as '-thumbprint' for the 'host.add' and 'cluster.add' commands.

Examples:
  govc about.cert -k -json | jq -r .ThumbprintSHA1
  govc about.cert -k -show | sudo tee /usr/local/share/ca-certificates/host.crt
  govc about.cert -k -thumbprint | tee -a ~/.govmomi/known_hosts

Options:
  -show=false               Show PEM encoded server certificate only
  -thumbprint=false         Output host hash and thumbprint only
```

## cluster.add

```
Usage: govc cluster.add [OPTIONS]

Add HOST to CLUSTER.

The host is added to the cluster specified by the 'cluster' flag.

Examples:
  thumbprint=$(govc about.cert -k -u host.example.com -thumbprint | awk '{print $2}')
  govc cluster.add -cluster ClusterA -hostname host.example.com -username root -password pass -thumbprint $thumbprint
  govc cluster.add -cluster ClusterB -hostname 10.0.6.1 -username root -password pass -noverify

Options:
  -cluster=*                Path to cluster
  -connect=true             Immediately connect to host
  -force=false              Force when host is managed by another VC
  -hostname=                Hostname or IP address of the host
  -license=                 Assign license key
  -noverify=false           Accept host thumbprint without verification
  -password=                Password of administration account on the host
  -thumbprint=              SHA-1 thumbprint of the host's SSL certificate
  -username=                Username of administration account on the host
```

## cluster.change

```
Usage: govc cluster.change [OPTIONS] CLUSTER...

Change configuration of the given clusters.

Examples:
  govc cluster.change -drs-enabled -vsan-enabled -vsan-autoclaim ClusterA
  govc cluster.change -drs-enabled=false ClusterB

Options:
  -drs-enabled=<nil>        Enable DRS
  -drs-mode=                DRS behavior for virtual machines: manual, partiallyAutomated, fullyAutomated
  -ha-enabled=<nil>         Enable HA
  -vsan-autoclaim=<nil>     Autoclaim storage on cluster hosts
  -vsan-enabled=<nil>       Enable vSAN
```

## cluster.create

```
Usage: govc cluster.create [OPTIONS] CLUSTER

Create CLUSTER in datacenter.

The cluster is added to the folder specified by the 'folder' flag. If not given,
this defaults to the host folder in the specified or default datacenter.

Examples:
  govc cluster.create ClusterA
  govc cluster.create -folder /dc2/test-folder ClusterB

Options:
  -folder=                  Inventory folder [GOVC_FOLDER]
```

## datacenter.create

```
Usage: govc datacenter.create [OPTIONS] NAME...

Options:
  -folder=                  Inventory folder [GOVC_FOLDER]
```

## datacenter.info

```
Usage: govc datacenter.info [OPTIONS] [PATH]...

Options:
```

## datastore.cp

```
Usage: govc datastore.cp [OPTIONS] SRC DST

Copy SRC to DST on DATASTORE.

Examples:
  govc datastore.cp foo/foo.vmx foo/foo.vmx.old
  govc datastore.cp -f my.vmx foo/foo.vmx

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -f=false                  If true, overwrite any identically named file at the destination
```

## datastore.create

```
Usage: govc datastore.create [OPTIONS] HOST...

Create datastore on HOST.

Examples:
  govc datastore.create -type nfs -name nfsDatastore -remote-host 10.143.2.232 -remote-path /share cluster1
  govc datastore.create -type vmfs -name vmfsDatastore -disk=mpx.vmhba0:C0:T0:L0 cluster1
  govc datastore.create -type local -name localDatastore -path /var/datastore host1

Options:
  -disk=                    Canonical name of disk (VMFS only)
  -force=false              Ignore DuplicateName error if datastore is already mounted on a host
  -host=                    Host system [GOVC_HOST]
  -mode=readOnly            Access mode for the mount point (readOnly|readWrite)
  -name=                    Datastore name
  -password=                Password to use when connecting (CIFS only)
  -path=                    Local directory path for the datastore (local only)
  -remote-host=             Remote hostname of the NAS datastore
  -remote-path=             Remote path of the NFS mount point
  -type=                    Datastore type (NFS|NFS41|CIFS|VMFS|local)
  -username=                Username to use when connecting (CIFS only)
```

## datastore.disk.create

```
Usage: govc datastore.disk.create [OPTIONS] VMDK

Create VMDK on DS.

Examples:
  govc datastore.mkdir disks
  govc datastore.disk.create -size 24G disks/disk1.vmdk

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -size=10.0GB              Size of new disk
```

## datastore.disk.info

```
Usage: govc datastore.disk.info [OPTIONS] VMDK

Query VMDK info on DS.

Examples:
  govc datastore.disk.info disks/disk1.vmdk

Options:
  -c=false                  Chain format
  -d=false                  Include datastore in output
  -ds=                      Datastore [GOVC_DATASTORE]
  -p=true                   Include parents
```

## datastore.download

```
Usage: govc datastore.download [OPTIONS] SOURCE DEST

Copy SOURCE from DS to DEST on the local system.

If DEST name is "-", source is written to stdout.

Examples:
  govc datastore.download vm-name/vmware.log ./local.log
  govc datastore.download vm-name/vmware.log - | grep -i error

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -host=                    Host system [GOVC_HOST]
```

## datastore.info

```
Usage: govc datastore.info [OPTIONS] [PATH]...

Options:
```

## datastore.ls

```
Usage: govc datastore.ls [OPTIONS] [FILE]...

Options:
  -R=false                  List subdirectories recursively
  -a=false                  Do not ignore entries starting with .
  -ds=                      Datastore [GOVC_DATASTORE]
  -l=false                  Long listing format
  -p=false                  Append / indicator to directories
```

## datastore.mkdir

```
Usage: govc datastore.mkdir [OPTIONS] DIRECTORY

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -namespace=false          Return uuid of namespace created on vsan datastore
  -p=false                  Create intermediate directories as needed
```

## datastore.mv

```
Usage: govc datastore.mv [OPTIONS] SRC DST

Move SRC to DST on DATASTORE.

Examples:
  govc datastore.mv foo/foo.vmx foo/foo.vmx.old
  govc datastore.mv -f my.vmx foo/foo.vmx

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -f=false                  If true, overwrite any identically named file at the destination
```

## datastore.remove

```
Usage: govc datastore.remove [OPTIONS] HOST...

Remove datastore from HOST.

Examples:
  govc datastore.remove -ds nfsDatastore cluster1
  govc datastore.remove -ds nasDatastore host1 host2 host3

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -host=                    Host system [GOVC_HOST]
```

## datastore.rm

```
Usage: govc datastore.rm [OPTIONS] FILE

Remove FILE from DATASTORE.

Examples:
  govc datastore.rm vm/vmware.log
  govc datastore.rm vm
  govc datastore.rm -f images/base.vmdk

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -f=false                  Force; ignore nonexistent files and arguments
  -namespace=false          Path is uuid of namespace on vsan datastore
  -t=true                   Use file type to choose disk or file manager
```

## datastore.tail

```
Usage: govc datastore.tail [OPTIONS] PATH

Output the last part of datastore files.

Examples:
  govc datastore.tail -n 100 vm-name/vmware.log
  govc datastore.tail -n 0 -f vm-name/vmware.log

Options:
  -c=-1                     Output the last NUM bytes
  -ds=                      Datastore [GOVC_DATASTORE]
  -f=false                  Output appended data as the file grows
  -host=                    Host system [GOVC_HOST]
  -n=10                     Output the last NUM lines
```

## datastore.upload

```
Usage: govc datastore.upload [OPTIONS] SOURCE DEST

Copy SOURCE from the local system to DEST on DS.

If SOURCE name is "-", read source from stdin.

Examples:
  govc datastore.upload -ds datastore1 ./config.iso vm-name/config.iso
  genisoimage ... | govc datastore.upload -ds datastore1 - vm-name/config.iso

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
```

## datastore.vsan.dom.ls

```
Usage: govc datastore.vsan.dom.ls [OPTIONS] [UUID]...

List vSAN DOM objects in DS.

Examples:
  govc datastore.vsan.dom.ls
  govc datastore.vsan.dom.ls -ds vsanDatastore -l
  govc datastore.vsan.dom.ls -l d85aa758-63f5-500a-3150-0200308e589c

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -l=false                  Long listing
  -o=false                  List orphan objects
```

## datastore.vsan.dom.rm

```
Usage: govc datastore.vsan.dom.rm [OPTIONS] UUID...

Remove vSAN DOM objects in DS.

Examples:
  govc datastore.vsan.dom.rm d85aa758-63f5-500a-3150-0200308e589c
  govc datastore.vsan.dom.rm -f d85aa758-63f5-500a-3150-0200308e589c
  govc datastore.vsan.dom.ls -o | xargs govc datastore.vsan.dom.rm

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -f=false                  Force delete
  -v=false                  Print deleted UUIDs to stdout, failed to stderr
```

## device.boot

```
Usage: govc device.boot [OPTIONS]

Configure VM boot settings.

Examples:
  govc device.boot -vm $vm -delay 1000 -order floppy,cdrom,ethernet,disk

Options:
  -delay=0                  Delay in ms before starting the boot sequence
  -order=                   Boot device order
  -retry=false              If true, retry boot after retry-delay
  -retry-delay=0            Delay in ms before a boot retry
  -setup=false              If true, enter BIOS setup on next boot
  -vm=                      Virtual machine [GOVC_VM]
```

## device.cdrom.add

```
Usage: govc device.cdrom.add [OPTIONS]

Add CD-ROM device to VM.

Examples:
  govc device.cdrom.add -vm $vm
  govc device.ls -vm $vm | grep ide-
  govc device.cdrom.add -vm $vm -controller ide-200
  govc device.info cdrom-*

Options:
  -controller=              IDE controller name
  -vm=                      Virtual machine [GOVC_VM]
```

## device.cdrom.eject

```
Usage: govc device.cdrom.eject [OPTIONS]

Eject media from CD-ROM device.

If device is not specified, the first CD-ROM device is used.

Examples:
  govc device.cdrom.eject -vm vm-1
  govc device.cdrom.eject -vm vm-1 -device floppy-1

Options:
  -device=                  CD-ROM device name
  -vm=                      Virtual machine [GOVC_VM]
```

## device.cdrom.insert

```
Usage: govc device.cdrom.insert [OPTIONS] ISO

Insert media on datastore into CD-ROM device.

If device is not specified, the first CD-ROM device is used.

Examples:
  govc device.cdrom.insert -vm vm-1 -device cdrom-3000 images/boot.iso

Options:
  -device=                  CD-ROM device name
  -ds=                      Datastore [GOVC_DATASTORE]
  -vm=                      Virtual machine [GOVC_VM]
```

## device.connect

```
Usage: govc device.connect [OPTIONS] DEVICE...

Options:
  -vm=                      Virtual machine [GOVC_VM]
```

## device.disconnect

```
Usage: govc device.disconnect [OPTIONS] DEVICE...

Options:
  -vm=                      Virtual machine [GOVC_VM]
```

## device.floppy.add

```
Usage: govc device.floppy.add [OPTIONS]

Add floppy device to VM.

Examples:
  govc device.floppy.add -vm $vm
  govc device.info floppy-*

Options:
  -vm=                      Virtual machine [GOVC_VM]
```

## device.floppy.eject

```
Usage: govc device.floppy.eject [OPTIONS]

Eject image from floppy device.

If device is not specified, the first floppy device is used.

Examples:
  govc device.floppy.eject -vm vm-1

Options:
  -device=                  Floppy device name
  -vm=                      Virtual machine [GOVC_VM]
```

## device.floppy.insert

```
Usage: govc device.floppy.insert [OPTIONS] IMG

Insert IMG on datastore into floppy device.

If device is not specified, the first floppy device is used.

Examples:
  govc device.floppy.insert -vm vm-1 vm-1/config.img

Options:
  -device=                  Floppy device name
  -ds=                      Datastore [GOVC_DATASTORE]
  -vm=                      Virtual machine [GOVC_VM]
```

## device.info

```
Usage: govc device.info [OPTIONS] [DEVICE]...

Options:
  -net=                     Network [GOVC_NETWORK]
  -net.adapter=e1000        Network adapter type
  -net.address=             Network hardware address
  -vm=                      Virtual machine [GOVC_VM]
```

## device.ls

```
Usage: govc device.ls [OPTIONS]

Options:
  -boot=false               List devices configured in the VM's boot options
  -vm=                      Virtual machine [GOVC_VM]
```

## device.remove

```
Usage: govc device.remove [OPTIONS] DEVICE...

Options:
  -keep=false               Keep files in datastore
  -vm=                      Virtual machine [GOVC_VM]
```

## device.scsi.add

```
Usage: govc device.scsi.add [OPTIONS]

Add SCSI controller to VM.

Examples:
  govc device.scsi.add -vm $vm
  govc device.scsi.add -vm $vm -type pvscsi
  govc device.info -vm $vm {lsi,pv}*

Options:
  -hot=false                Enable hot-add/remove
  -sharing=noSharing        SCSI sharing
  -type=lsilogic            SCSI controller type (lsilogic|buslogic|pvscsi|lsilogic-sas)
  -vm=                      Virtual machine [GOVC_VM]
```

## device.serial.add

```
Usage: govc device.serial.add [OPTIONS]

Add serial port to VM.

Examples:
  govc device.serial.add -vm $vm
  govc device.info -vm $vm serialport-*

Options:
  -vm=                      Virtual machine [GOVC_VM]
```

## device.serial.connect

```
Usage: govc device.serial.connect [OPTIONS] URI

Connect service URI to serial port.

If "-" is given as URI, connects file backed device with file name of
device name + .log suffix in the VM Config.Files.LogDirectory.

Defaults to the first serial port if no DEVICE is given.

Examples:
  govc device.ls | grep serialport-
  govc device.serial.connect -vm $vm -device serialport-8000 telnet://:33233
  govc device.info -vm $vm serialport-*
  govc device.serial.connect -vm $vm "[datastore1] $vm/console.log"
  govc device.serial.connect -vm $vm -
  govc datastore.tail -f $vm/serialport-8000.log

Options:
  -client=false             Use client direction
  -device=                  serial port device name
  -vm=                      Virtual machine [GOVC_VM]
  -vspc-proxy=              vSPC proxy URI
```

## device.serial.disconnect

```
Usage: govc device.serial.disconnect [OPTIONS]

Disconnect service URI from serial port.

Examples:
  govc device.ls | grep serialport-
  govc device.serial.disconnect -vm $vm -device serialport-8000
  govc device.info -vm $vm serialport-*

Options:
  -device=                  serial port device name
  -vm=                      Virtual machine [GOVC_VM]
```

## device.usb.add

```
Usage: govc device.usb.add [OPTIONS]

Add USB device to VM.

Examples:
  govc device.usb.add -vm $vm
  govc device.usb.add -type xhci -vm $vm
  govc device.info usb*

Options:
  -auto=true                Enable ability to hot plug devices
  -ehci=true                Enable enhanced host controller interface (USB 2.0)
  -type=usb                 USB controller type (usb|xhci)
  -vm=                      Virtual machine [GOVC_VM]
```

## dvs.add

```
Usage: govc dvs.add [OPTIONS] HOST...

Add hosts to DVS.

Examples:
  govc dvs.add -dvs dvsName -pnic vmnic1 hostA hostB hostC

Options:
  -dvs=                     DVS path
  -host=                    Host system [GOVC_HOST]
  -pnic=vmnic0              Name of the host physical NIC
```

## dvs.create

```
Usage: govc dvs.create [OPTIONS] DVS

Create DVS (DistributedVirtualSwitch) in datacenter.

The dvs is added to the folder specified by the 'folder' flag. If not given,
this defaults to the network folder in the specified or default datacenter.

Examples:
  govc dvs.create DSwitch
  govc dvs.create -product-version 5.5.0 DSwitch

Options:
  -folder=                  Inventory folder [GOVC_FOLDER]
  -product-version=         DVS product version
```

## dvs.portgroup.add

```
Usage: govc dvs.portgroup.add [OPTIONS] NAME

Add portgroup to DVS.

Examples:
  govc dvs.create DSwitch
  govc dvs.portgroup.add -dvs DSwitch -type earlyBinding -nports 16 ExternalNetwork
  govc dvs.portgroup.add -dvs DSwitch -type ephemeral InternalNetwork

Options:
  -dvs=                     DVS path
  -nports=128               Number of ports
  -type=earlyBinding        Portgroup type (earlyBinding|lateBinding|ephemeral)
  -vlan=0                   VLAN ID
```

## dvs.portgroup.change

```
Usage: govc dvs.portgroup.change [OPTIONS] PATH

Change DVS portgroup configuration.

Examples:
  govc dvs.portgroup.change -nports 26 ExternalNetwork
  govc dvs.portgroup.change -vlan 3214 ExternalNetwork

Options:
  -nports=0                 Number of ports
  -type=earlyBinding        Portgroup type (earlyBinding|lateBinding|ephemeral)
  -vlan=0                   VLAN ID
```

## dvs.portgroup.info

```
Usage: govc dvs.portgroup.info [OPTIONS]

Options:
  -active=false             Filter by port active or inactive status
  -connected=false          Filter by port connected or disconnected status
  -count=0                  Number of matches to return (0 = unlimited)
  -inside=true              Filter by port inside or outside status
  -pg=                      Distributed Virtual Portgroup
  -uplinkPort=false         Filter for uplink ports
  -vlan=0                   Filter by VLAN ID (0 = unfiltered)
```

## env

```
Usage: govc env [OPTIONS]

Output the environment variables for this client.

If credentials are included in the url, they are split into separate variables.
Useful as bash scripting helper to parse GOVC_URL.

Options:
  -x=false                  Output variables for each GOVC_URL component
```

## events

```
Usage: govc events [OPTIONS] [PATH]...

Display events.

Examples:
  govc events vm/my-vm1 vm/my-vm2
  govc events /dc1/vm/* /dc2/vm/*
  govc ls -t HostSystem host/* | xargs govc events | grep -i vsan

Options:
  -f=false                  Follow event stream
  -force=false              Disable number objects to monitor limit
  -n=25                     Output the last N events
```

## extension.info

```
Usage: govc extension.info [OPTIONS] [KEY]...

Options:
```

## extension.register

```
Usage: govc extension.register [OPTIONS]

Options:
  -update=false             Update extension
```

## extension.setcert

```
Usage: govc extension.setcert [OPTIONS] ID

Set certificate for the extension ID.

The '-cert-pem' option can be one of the following:
'-' : Read the certificate from stdin
'+' : Generate a new key pair and save locally to ID.crt and ID.key
... : Any other value is passed as-is to ExtensionManager.SetCertificate

Options:
  -cert-pem=-               PEM encoded certificate
  -org=VMware               Organization for generated certificate
```

## extension.unregister

```
Usage: govc extension.unregister [OPTIONS]

Options:
```

## fields.add

```
Usage: govc fields.add [OPTIONS] NAME

Options:
```

## fields.ls

```
Usage: govc fields.ls [OPTIONS]

Options:
```

## fields.rename

```
Usage: govc fields.rename [OPTIONS] KEY NAME

Options:
```

## fields.rm

```
Usage: govc fields.rm [OPTIONS] KEY...

Options:
```

## fields.set

```
Usage: govc fields.set [OPTIONS] KEY VALUE PATH...

Options:
```

## find

```
Usage: govc find [OPTIONS] [ROOT] [KEY VAL]...

Find managed objects.

ROOT can be an inventory path or ManagedObjectReference.
ROOT defaults to '.', an alias for the root folder or DC if set.

Optional KEY VAL pairs can be used to filter results against object instance properties.

The '-type' flag value can be a managed entity type or one of the following aliases:

  a    VirtualApp
  c    ClusterComputeResource
  d    Datacenter
  f    Folder
  g    DistributedVirtualPortgroup
  h    HostSystem
  m    VirtualMachine
  n    Network
  o    OpaqueNetwork
  p    ResourcePool
  r    ComputeResource
  s    Datastore
  w    DistributedVirtualSwitch

Examples:
  govc find
  govc find /dc1 -type c
  govc find vm -name my-vm-*
  govc find . -type n
  govc find . -type m -runtime.powerState poweredOn
  govc find . -type m -datastore $(govc find -i datastore -name vsanDatastore)
  govc find . -type s -summary.type vsan
  govc find . -type h -hardware.cpuInfo.numCpuCores 16

Options:
  -i=false                  Print the managed object reference
  -maxdepth=-1              Max depth
  -name=*                   Resource name
  -type=[]                  Resource type
```

## firewall.ruleset.find

```
Usage: govc firewall.ruleset.find [OPTIONS]

Find firewall rulesets matching the given rule.

For a complete list of rulesets: govc host.esxcli network firewall ruleset list
For a complete list of rules:    govc host.esxcli network firewall ruleset rule list

Examples:
  govc firewall.ruleset.find -direction inbound -port 22
  govc firewall.ruleset.find -direction outbound -port 2377

Options:
  -c=true                   Check if esx firewall is enabled
  -direction=outbound       Direction
  -enabled=true             Find enabled rule sets if true, disabled if false
  -host=                    Host system [GOVC_HOST]
  -port=0                   Port
  -proto=tcp                Protocol
  -type=dst                 Port type
```

## folder.create

```
Usage: govc folder.create [OPTIONS] PATH...

Create folder with PATH.

Examples:
  govc folder.create /dc1/vm/folder-foo
  govc object.mv /dc1/vm/vm-foo-* /dc1/vm/folder-foo
  govc folder.create -pod /dc1/datastore/sdrs
  govc object.mv /dc1/datastore/iscsi-* /dc1/datastore/sdrs

Options:
  -pod=false                Create folder(s) of type StoragePod (DatastoreCluster)
```

## folder.info

```
Usage: govc folder.info [OPTIONS] [PATH]...

Options:
```

## guest.chmod

```
Usage: govc guest.chmod [OPTIONS]

Options:
  -gid=0                    Group ID
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -perm=0                   File permissions
  -uid=0                    User ID
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.download

```
Usage: govc guest.download [OPTIONS] SOURCE DEST

Copy SOURCE from the guest VM to DEST on the local system.

If DEST name is "-", source is written to stdout.

Examples:
  govc guest.download -l user:pass -vm=my-vm /var/log/my.log ./local.log
  govc guest.download -l user:pass -vm=my-vm /etc/motd -

Options:
  -f=false                  If set, the local destination file is clobbered
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.getenv

```
Usage: govc guest.getenv [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.kill

```
Usage: govc guest.kill [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=[]                     Process ID
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.ls

```
Usage: govc guest.ls [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.mkdir

```
Usage: govc guest.mkdir [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=false                  Create intermediate directories as needed
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.mktemp

```
Usage: govc guest.mktemp [OPTIONS]

Options:
  -d=false                  Make a directory instead of a file
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -s=                       Suffix
  -t=                       Prefix
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.ps

```
Usage: govc guest.ps [OPTIONS]

Options:
  -U=                       Select by process UID
  -X=false                  Wait for process to exit
  -e=false                  Select all processes
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=[]                     Select by process ID
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.rm

```
Usage: govc guest.rm [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.rmdir

```
Usage: govc guest.rmdir [OPTIONS]

Options:
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -r=false                  Recursive removal
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.start

```
Usage: govc guest.start [OPTIONS]

Options:
  -C=                       The absolute path of the working directory for the program to start
  -e=[]                     Set environment variable (key=val)
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                      Virtual machine [GOVC_VM]
```

## guest.upload

```
Usage: govc guest.upload [OPTIONS] SOURCE DEST

Copy SOURCE from the local system to DEST in the guest VM.

If SOURCE name is "-", read source from stdin.

Examples:
  govc guest.upload -l user:pass -vm=my-vm ~/.ssh/id_rsa.pub /home/$USER/.ssh/authorized_keys
  cowsay "have a great day" | govc guest.upload -l user:pass -vm=my-vm - /etc/motd

Options:
  -f=false                  If set, the guest destination file is clobbered
  -gid=0                    Group ID
  -l=:                      Guest VM credentials [GOVC_GUEST_LOGIN]
  -perm=0                   File permissions
  -uid=0                    User ID
  -vm=                      Virtual machine [GOVC_VM]
```

## host.account.create

```
Usage: govc host.account.create [OPTIONS]

Create local account on HOST.

Examples:
  govc host.account.create -id $USER -password password-for-esx60

Options:
  -description=             The description of the specified account
  -host=                    Host system [GOVC_HOST]
  -id=                      The ID of the specified account
  -password=                The password for the specified account id
```

## host.account.remove

```
Usage: govc host.account.remove [OPTIONS]

Remove local account on HOST.

Examples:
  govc host.account.remove -id $USER

Options:
  -description=             The description of the specified account
  -host=                    Host system [GOVC_HOST]
  -id=                      The ID of the specified account
  -password=                The password for the specified account id
```

## host.account.update

```
Usage: govc host.account.update [OPTIONS]

Update local account on HOST.

Examples:
  govc host.account.update -id root -password password-for-esx60

Options:
  -description=             The description of the specified account
  -host=                    Host system [GOVC_HOST]
  -id=                      The ID of the specified account
  -password=                The password for the specified account id
```

## host.add

```
Usage: govc host.add [OPTIONS]

Add host to datacenter.

The host is added to the folder specified by the 'folder' flag. If not given,
this defaults to the host folder in the specified or default datacenter.

Examples:
  thumbprint=$(govc about.cert -k -u host.example.com -thumbprint | awk '{print $2}')
  govc host.add -hostname host.example.com -username root -password pass -thumbprint $thumbprint
  govc host.add -hostname 10.0.6.1 -username root -password pass -noverify

Options:
  -connect=true             Immediately connect to host
  -folder=                  Inventory folder [GOVC_FOLDER]
  -force=false              Force when host is managed by another VC
  -hostname=                Hostname or IP address of the host
  -noverify=false           Accept host thumbprint without verification
  -password=                Password of administration account on the host
  -thumbprint=              SHA-1 thumbprint of the host's SSL certificate
  -username=                Username of administration account on the host
```

## host.autostart.add

```
Usage: govc host.autostart.add [OPTIONS] VM...

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.autostart.configure

```
Usage: govc host.autostart.configure [OPTIONS] 

Options:
  -enabled=<nil>             
  -host=                     Host system [GOVC_HOST]
  -start-delay=0             
  -stop-action=              
  -stop-delay=0              
  -wait-for-heartbeat=<nil>  
```

## host.autostart.info

```
Usage: govc host.autostart.info [OPTIONS] 

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.autostart.remove

```
Usage: govc host.autostart.remove [OPTIONS] VM...

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.cert.csr

```
Usage: govc host.cert.csr [OPTIONS]

Generate a certificate-signing request (CSR) for HOST.

Options:
  -host=                    Host system [GOVC_HOST]
  -ip=false                 Use IP address as CN
```

## host.cert.import

```
Usage: govc host.cert.import [OPTIONS] FILE

Install SSL certificate FILE on HOST.

If FILE name is "-", read certificate from stdin.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.cert.info

```
Usage: govc host.cert.info [OPTIONS]

Display SSL certificate info for HOST.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.date.change

```
Usage: govc host.date.change [OPTIONS]

Change date and time for HOST.

Examples:
  govc host.date.change -date "$(date -u)"
  govc host.date.change -server time.vmware.com
  govc host.service enable ntpd
  govc host.service start ntpd

Options:
  -date=                    Update the date/time on the host
  -host=                    Host system [GOVC_HOST]
  -server=                  IP or FQDN for NTP server(s)
  -tz=                      Change timezone of the host
```

## host.date.info

```
Usage: govc host.date.info [OPTIONS]

Display date and time info for HOST.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.disconnect

```
Usage: govc host.disconnect [OPTIONS]

Disconnect HOST from vCenter.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.esxcli

```
Usage: govc host.esxcli [OPTIONS] COMMAND [ARG]...

Invoke esxcli command on HOST.

Output is rendered in table form when possible, unless disabled with '-hints=false'.

Examples:
  govc host.esxcli network ip connection list
  govc host.esxcli system settings advanced set -o /Net/GuestIPHack -i 1
  govc host.esxcli network firewall ruleset set -r remoteSerialPort -e true
  govc host.esxcli network firewall set -e false

Options:
  -hints=true               Use command info hints when formatting output
  -host=                    Host system [GOVC_HOST]
```

## host.info

```
Usage: govc host.info [OPTIONS]

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.maintenance.enter

```
Usage: govc host.maintenance.enter [OPTIONS] HOST...

Put HOST in maintenance mode.

While this task is running and when the host is in maintenance mode,
no VMs can be powered on and no provisioning operations can be performed on the host.

Options:
  -evacuate=false           Evacuate powered off VMs
  -host=                    Host system [GOVC_HOST]
  -timeout=0                Timeout
```

## host.maintenance.exit

```
Usage: govc host.maintenance.exit [OPTIONS] HOST...

Take HOST out of maintenance mode.

This blocks if any concurrent running maintenance-only host configurations operations are being performed.
For example, if VMFS volumes are being upgraded.

The 'timeout' flag is the number of seconds to wait for the exit maintenance mode to succeed.
If the timeout is less than or equal to zero, there is no timeout.

Options:
  -host=                    Host system [GOVC_HOST]
  -timeout=0                Timeout
```

## host.option.ls

```
Usage: govc host.option.ls [OPTIONS] NAME

List option with the given NAME.

If NAME ends with a dot, all options for that subtree are listed.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.option.set

```
Usage: govc host.option.set [OPTIONS] NAME VALUE

Set host option NAME to VALUE.

Examples:
  govc host.option.set Config.HostAgent.plugins.solo.enableMob true
  govc host.option.set Config.HostAgent.log.level verbose

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.portgroup.add

```
Usage: govc host.portgroup.add [OPTIONS] NAME

Add portgroup to HOST.

Examples:
  govc host.portgroup.add -vswitch vSwitch0 -vlan 3201 bridge

Options:
  -host=                    Host system [GOVC_HOST]
  -vlan=0                   VLAN ID
  -vswitch=                 vSwitch Name
```

## host.portgroup.change

```
Usage: govc host.portgroup.change [OPTIONS] NAME

Change configuration of HOST portgroup NAME.

Examples:
  govc host.portgroup.change -allow-promiscuous -forged-transmits -mac-changes "VM Network"
  govc host.portgroup.change -vswitch-name vSwitch1 "Management Network"

Options:
  -allow-promiscuous=<nil>  Allow promiscuous mode
  -forged-transmits=<nil>   Allow forged transmits
  -host=                    Host system [GOVC_HOST]
  -mac-changes=<nil>        Allow MAC changes
  -name=                    Portgroup name
  -vlan-id=-1               VLAN ID
  -vswitch-name=            vSwitch name
```

## host.portgroup.info

```
Usage: govc host.portgroup.info [OPTIONS]

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.portgroup.remove

```
Usage: govc host.portgroup.remove [OPTIONS] NAME

Remove portgroup from HOST.

Examples:
  govc host.portgroup.remove bridge

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.reconnect

```
Usage: govc host.reconnect [OPTIONS]

Reconnect HOST to vCenter.

This command can also be used to change connection properties (hostname, fingerprint, username, password),
without disconnecting the host.

Options:
  -force=false              Force when host is managed by another VC
  -host=                    Host system [GOVC_HOST]
  -hostname=                Hostname or IP address of the host
  -noverify=false           Accept host thumbprint without verification
  -password=                Password of administration account on the host
  -sync-state=false         Sync state
  -thumbprint=              SHA-1 thumbprint of the host's SSL certificate
  -username=                Username of administration account on the host
```

## host.remove

```
Usage: govc host.remove [OPTIONS] HOST...

Remove HOST from vCenter.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.service

```
Usage: govc host.service [OPTIONS] ACTION ID

Apply host service ACTION to service ID.

Where ACTION is one of: start, stop, restart, status, enable, disable

Examples:
  govc host.service enable TSM-SSH
  govc host.service start TSM-SSH

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.service.ls

```
Usage: govc host.service.ls [OPTIONS]

List HOST services.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.storage.info

```
Usage: govc host.storage.info [OPTIONS]

Show HOST storage system information.

Examples:
  govc ls -t HostSystem host/* | xargs -n1 govc host.storage.info -unclaimed -host

Options:
  -host=                    Host system [GOVC_HOST]
  -rescan=false             Rescan for new storage devices
  -t=lun                    Type (hba,lun)
  -unclaimed=false          Only show disks that can be used as new VMFS datastores
```

## host.storage.mark

```
Usage: govc host.storage.mark [OPTIONS] DEVICE_PATH

Mark device at DEVICE_PATH.

Options:
  -host=                    Host system [GOVC_HOST]
  -local=<nil>              Mark as local
  -ssd=<nil>                Mark as SSD
```

## host.storage.partition

```
Usage: govc host.storage.partition [OPTIONS] DEVICE_PATH

Show partition table for device at DEVICE_PATH.

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.vnic.info

```
Usage: govc host.vnic.info [OPTIONS]

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.vnic.service

```
Usage: govc host.vnic.service [OPTIONS] SERVICE DEVICE


Enable or disable service on a virtual nic device.

Where SERVICE is one of: vmotion|faultToleranceLogging|vSphereReplication|vSphereReplicationNFC|management|vsan|vSphereProvisioning
Where DEVICE is one of: vmk0|vmk1|...

Examples:
  govc host.vnic.service -host hostname -enable vsan vmk0


Options:
  -disable=false            Disable service
  -enable=false             Enable service
  -host=                    Host system [GOVC_HOST]
```

## host.vswitch.add

```
Usage: govc host.vswitch.add [OPTIONS] NAME

Options:
  -host=                    Host system [GOVC_HOST]
  -mtu=0                    MTU
  -nic=                     Bridge nic device
  -ports=128                Number of ports
```

## host.vswitch.info

```
Usage: govc host.vswitch.info [OPTIONS]

Options:
  -host=                    Host system [GOVC_HOST]
```

## host.vswitch.remove

```
Usage: govc host.vswitch.remove [OPTIONS] NAME

Options:
  -host=                    Host system [GOVC_HOST]
```

## import.ova

```
Usage: govc import.ova [OPTIONS] PATH_TO_OVA

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -folder=                  Path to folder to add the VM to
  -host=                    Host system [GOVC_HOST]
  -name=                    Name to use for new entity
  -options=                 Options spec file path for VM deployment
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
```

## import.ovf

```
Usage: govc import.ovf [OPTIONS] PATH_TO_OVF

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -folder=                  Path to folder to add the VM to
  -host=                    Host system [GOVC_HOST]
  -name=                    Name to use for new entity
  -options=                 Options spec file path for VM deployment
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
```

## import.spec

```
Usage: govc import.spec [OPTIONS] PATH_TO_OVF_OR_OVA

Options:
  -verbose=false  Verbose spec output
```

## import.vmdk

```
Usage: govc import.vmdk [OPTIONS] PATH_TO_VMDK [REMOTE_DIRECTORY]

Options:
  -ds=                      Datastore [GOVC_DATASTORE]
  -force=false              Overwrite existing disk
  -keep=false               Keep uploaded disk after import
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
  -upload=true              Upload specified disk
```

## license.add

```
Usage: govc license.add [OPTIONS] KEY...

Options:
```

## license.assign

```
Usage: govc license.assign [OPTIONS] KEY

Options:
  -host=                    Host system [GOVC_HOST]
  -name=                    Display name
  -remove=false             Remove assignment
```

## license.assigned.ls

```
Usage: govc license.assigned.ls [OPTIONS]

Options:
  -id=                      Entity ID
```

## license.decode

```
Usage: govc license.decode [OPTIONS] KEY...

Options:
  -feature=                 List licenses with given feature
```

## license.ls

```
Usage: govc license.ls [OPTIONS]

Options:
  -feature=                 List licenses with given feature
```

## license.remove

```
Usage: govc license.remove [OPTIONS] KEY...

Options:
```

## logs

```
Usage: govc logs [OPTIONS]

View VPX and ESX logs.

The '-log' option defaults to "hostd" when connected directly to a host or
when connected to VirtualCenter and a '-host' option is given.  Otherwise,
the '-log' option defaults to "vpxd:vpxd.log".  The '-host' option is ignored
when connected directly to a host.  See 'govc logs.ls' for other '-log' options.

Examples:
  govc logs -n 1000 -f
  govc logs -host esx1
  govc logs -host esx1 -log vmkernel

Options:
  -f=false                  Follow log file changes
  -host=                    Host system [GOVC_HOST]
  -log=                     Log file key
  -n=25                     Output the last N log lines
```

## logs.download

```
Usage: govc logs.download [OPTIONS] [PATH]...

Generate diagnostic bundles.

A diagnostic bundle includes log files and other configuration information.

Use PATH to include a specific set of hosts to include.

Examples:
  govc logs.download
  govc logs.download host-a host-b

Options:
  -default=true             Specifies if the bundle should include the default server
```

## logs.ls

```
Usage: govc logs.ls [OPTIONS]

List diagnostic log keys.

Examples:
  govc logs.ls
  govc logs.ls -host host-a

Options:
  -host=                    Host system [GOVC_HOST]
```

## ls

```
Usage: govc ls [OPTIONS] [PATH]...

List inventory items.

Examples:
  govc ls -l '*'
  govc ls -t ClusterComputeResource host
  govc ls -t Datastore host/ClusterA/* | grep -v local | xargs -n1 basename | sort | uniq

Options:
  -L=false                  Follow managed object references
  -i=false                  Print the managed object reference
  -l=false                  Long listing format
  -t=                       Object type
```

## metric.change

```
Usage: govc metric.change [OPTIONS] NAME...

Change counter NAME levels.

Examples:
  govc metric.change -level 1 net.bytesRx.average net.bytesTx.average

Options:
  -device-level=0           Level for the per device counter
  -i=0                      Interval ID
  -level=0                  Level for the aggregate counter
```

## metric.info

```
Usage: govc metric.info [OPTIONS] PATH [NAME]...

Metric info for NAME.

If PATH is a value other than '-', provider summary and instance list are included
for the given object type.

If NAME is not specified, all available metrics for the given INTERVAL are listed.
An object PATH must be provided in this case.

Examples:
  govc metric.info vm/my-vm
  govc metric.info -i 300 vm/my-vm
  govc metric.info - cpu.usage.average
  govc metric.info /dc1/host/cluster cpu.usage.average

Options:
  -i=0                      Interval ID
```

## metric.interval.change

```
Usage: govc metric.interval.change [OPTIONS]

Change historical metric intervals.

Examples:
  govc metric.interval.change -i 300 -level 2
  govc metric.interval.change -i 86400 -enabled=false

Options:
  -enabled=<nil>            Enable or disable
  -i=0                      Interval ID
  -level=0                  Level
```

## metric.interval.info

```
Usage: govc metric.interval.info [OPTIONS]

List historical metric intervals.

Examples:
  govc metric.interval.info
  govc metric.interval.info -i 300

Options:
  -i=0                      Interval ID
```

## metric.ls

```
Usage: govc metric.ls [OPTIONS] PATH

List available metrics for PATH.

Examples:
  govc metric.ls /dc1/host/cluster1
  govc metric.ls datastore/*
  govc metric.ls vm/* | grep mem. | xargs govc metric.sample vm/*

Options:
  -i=0                      Interval ID
  -l=false                  Long listing format
```

## metric.reset

```
Usage: govc metric.reset [OPTIONS] NAME...

Reset counter NAME to the default level of data collection.

Examples:
  govc metric.reset net.bytesRx.average net.bytesTx.average

Options:
  -i=0                      Interval ID
```

## metric.sample

```
Usage: govc metric.sample [OPTIONS] PATH... NAME...

Sample for object PATH of metric NAME.

Interval ID defaults to 20 (realtime) if supported, otherwise 300 (5m interval).

If PLOT value is set to '-', output a gnuplot script.  If non-empty with another
value, PLOT will pipe the script to gnuplot for you.  The value is also used to set
the gnuplot 'terminal' variable, unless the value is that of the DISPLAY env var.
Only 1 metric NAME can be specified when the PLOT flag is set.

Examples:
  govc metric.sample host/cluster1/* cpu.usage.average
  govc metric.sample -plot .png host/cluster1/* cpu.usage.average | xargs open
  govc metric.sample vm/* net.bytesTx.average net.bytesTx.average

Options:
  -d=30                     Limit object display name to D chars
  -i=0                      Interval ID
  -instance=*               Instance
  -n=6                      Max number of samples
  -plot=                    Plot data using gnuplot
  -t=false                  Include sample times
```

## object.collect

```
Usage: govc object.collect [OPTIONS] [MOID] [PROPERTY]...

Collect managed object properties.

MOID can be an inventory path or ManagedObjectReference.
MOID defaults to '-', an alias for 'ServiceInstance:ServiceInstance'.

By default only the current property value(s) are collected.  Use the '-n' flag to wait for updates.

Examples:
  govc object.collect - content
  govc object.collect -s HostSystem:ha-host hardware.systemInfo.uuid
  govc object.collect -s /ha-datacenter/vm/foo overallStatus
  govc object.collect -json -n=-1 EventManager:ha-eventmgr latestEvent | jq .
  govc object.collect -json -s $(govc object.collect -s - content.perfManager) description.counterType | jq .

Options:
  -n=0                      Wait for N property updates
  -s=false                  Output property value only
```

## object.destroy

```
Usage: govc object.destroy [OPTIONS] PATH...

Destroy managed objects.

Examples:
  govc object.destroy /dc1/network/dvs /dc1/host/cluster

Options:
```

## object.mv

```
Usage: govc object.mv [OPTIONS] PATH... FOLDER

Move managed entities to FOLDER.

Examples:
  govc folder.create /dc1/host/example
  govc object.mv /dc2/host/*.example.com /dc1/host/example

Options:
```

## object.reload

```
Usage: govc object.reload [OPTIONS] PATH...

Reload managed object state.

Examples:
  govc datastore.upload $vm.vmx $vm/$vm.vmx
  govc object.reload /dc1/vm/$vm

Options:
```

## object.rename

```
Usage: govc object.rename [OPTIONS] PATH NAME

Rename managed objects.

Examples:
  govc object.rename /dc1/network/dvs1 Switch1

Options:
```

## permissions.ls

```
Usage: govc permissions.ls [OPTIONS] [PATH]...

List the permissions defined on or effective on managed entities.

Examples:
  govc permissions.ls
  govc permissions.ls /dc1/host/cluster1

Options:
  -a=true                   Include inherited permissions defined by parent entities
  -i=false                  Use moref instead of inventory path
```

## permissions.remove

```
Usage: govc permissions.remove [OPTIONS] [PATH]...

Removes a permission rule from managed entities.

Examples:
  govc permissions.remove -principal root
  govc permissions.remove -principal $USER@vsphere.local -role Admin /dc1/host/cluster1

Options:
  -group=false              True, if principal refers to a group name; false, for a user name
  -i=false                  Use moref instead of inventory path
  -principal=               User or group for which the permission is defined
```

## permissions.set

```
Usage: govc permissions.set [OPTIONS] [PATH]...

Set the permissions managed entities.

Examples:
  govc permissions.set -principal root -role Admin
  govc permissions.set -principal $USER@vsphere.local -role Admin /dc1/host/cluster1

Options:
  -group=false              True, if principal refers to a group name; false, for a user name
  -i=false                  Use moref instead of inventory path
  -principal=               User or group for which the permission is defined
  -propagate=true           Whether or not this permission propagates down the hierarchy to sub-entities
  -role=Admin               Permission role name
```

## pool.change

```
Usage: govc pool.change [OPTIONS] POOL...

Change the configuration of one or more resource POOLs.

POOL may be an absolute or relative path to a resource pool or a (clustered)
compute host. If it resolves to a compute host, the associated root resource
pool is returned. If a relative path is specified, it is resolved with respect
to the current datacenter's "host" folder (i.e. /ha-datacenter/host).

Paths to nested resource pools must traverse through the root resource pool of
the selected compute host, i.e. "compute-host/Resources/nested-pool".

The same globbing rules that apply to the "ls" command apply here. For example,
POOL may be specified as "*/Resources/*" to expand to all resource pools that
are nested one level under the root resource pool, on all (clustered) compute
hosts in the current datacenter.

Options:
  -cpu.expandable=<nil>     CPU expandable reservation
  -cpu.limit=0              CPU limit in MHz
  -cpu.reservation=0        CPU reservation in MHz
  -cpu.shares=              CPU shares level or number
  -mem.expandable=<nil>     Memory expandable reservation
  -mem.limit=0              Memory limit in MB
  -mem.reservation=0        Memory reservation in MB
  -mem.shares=              Memory shares level or number
  -name=                    Resource pool name
```

## pool.create

```
Usage: govc pool.create [OPTIONS] POOL...

Create one or more resource POOLs.

POOL may be an absolute or relative path to a resource pool. The parent of the
specified POOL must be an existing resource pool. If a relative path is
specified, it is resolved with respect to the current datacenter's "host"
folder (i.e. /ha-datacenter/host). The basename of the specified POOL is used
as the name for the new resource pool.

The same globbing rules that apply to the "ls" command apply here. For example,
the path to the parent resource pool in POOL may be specified as "*/Resources"
to expand to the root resource pools on all (clustered) compute hosts in the
current datacenter.

For example:
  */Resources/test             Create resource pool "test" on all (clustered)
                               compute hosts in the current datacenter.
  somehost/Resources/*/nested  Create resource pool "nested" in every
                               resource pool that is a direct descendant of
                               the root resource pool on "somehost".

Options:
  -cpu.expandable=true      CPU expandable reservation
  -cpu.limit=0              CPU limit in MHz
  -cpu.reservation=0        CPU reservation in MHz
  -cpu.shares=normal        CPU shares level or number
  -mem.expandable=true      Memory expandable reservation
  -mem.limit=0              Memory limit in MB
  -mem.reservation=0        Memory reservation in MB
  -mem.shares=normal        Memory shares level or number
```

## pool.destroy

```
Usage: govc pool.destroy [OPTIONS] POOL...

Destroy one or more resource POOLs.

POOL may be an absolute or relative path to a resource pool or a (clustered)
compute host. If it resolves to a compute host, the associated root resource
pool is returned. If a relative path is specified, it is resolved with respect
to the current datacenter's "host" folder (i.e. /ha-datacenter/host).

Paths to nested resource pools must traverse through the root resource pool of
the selected compute host, i.e. "compute-host/Resources/nested-pool".

The same globbing rules that apply to the "ls" command apply here. For example,
POOL may be specified as "*/Resources/*" to expand to all resource pools that
are nested one level under the root resource pool, on all (clustered) compute
hosts in the current datacenter.

Options:
  -children=false           Remove all children pools
```

## pool.info

```
Usage: govc pool.info [OPTIONS] POOL...

Retrieve information about one or more resource POOLs.

POOL may be an absolute or relative path to a resource pool or a (clustered)
compute host. If it resolves to a compute host, the associated root resource
pool is returned. If a relative path is specified, it is resolved with respect
to the current datacenter's "host" folder (i.e. /ha-datacenter/host).

Paths to nested resource pools must traverse through the root resource pool of
the selected compute host, i.e. "compute-host/Resources/nested-pool".

The same globbing rules that apply to the "ls" command apply here. For example,
POOL may be specified as "*/Resources/*" to expand to all resource pools that
are nested one level under the root resource pool, on all (clustered) compute
hosts in the current datacenter.

Options:
  -a=false                  List virtual app resource pools
  -p=true                   List resource pools
```

## role.create

```
Usage: govc role.create [OPTIONS] NAME [PRIVILEGE]...

Create authorization role.

Optionally populate the role with the given PRIVILEGE(s).

Examples:
  govc role.create MyRole
  govc role.create NoDC $(govc role.ls Admin | grep -v Datacenter.)

Options:
  -i=false                  Use moref instead of inventory path
```

## role.ls

```
Usage: govc role.ls [OPTIONS] [NAME]

List authorization roles.

If NAME is provided, list privileges for the role.

Examples:
  govc role.ls
  govc role.ls Admin

Options:
  -i=false                  Use moref instead of inventory path
```

## role.remove

```
Usage: govc role.remove [OPTIONS] NAME

Remove authorization role.

Examples:
  govc role.remove MyRole
  govc role.remove MyRole -force

Options:
  -force=false              Force removal if role is in use
  -i=false                  Use moref instead of inventory path
```

## role.update

```
Usage: govc role.update [OPTIONS] NAME [PRIVILEGE]...

Update authorization role.

Set, Add or Remove role PRIVILEGE(s).

Examples:
  govc role.update MyRole $(govc role.ls Admin | grep VirtualMachine.)
  govc role.update -r MyRole $(govc role.ls Admin | grep VirtualMachine.GuestOperations.)
  govc role.update -a MyRole $(govc role.ls Admin | grep Datastore.)
  govc role.update -name RockNRole MyRole

Options:
  -a=false                  Remove given PRIVILEGE(s)
  -i=false                  Use moref instead of inventory path
  -name=                    Change role name
  -r=false                  Remove given PRIVILEGE(s)
```

## role.usage

```
Usage: govc role.usage [OPTIONS] NAME...

List usage for role NAME.

Examples:
  govc role.usage
  govc role.usage Admin

Options:
  -i=false                  Use moref instead of inventory path
```

## session.ls

```
Usage: govc session.ls [OPTIONS]

List active sessions.

Examples:
  govc session.ls
  govc session.ls -json | jq -r .CurrentSession.Key

Options:
```

## session.rm

```
Usage: govc session.rm [OPTIONS] KEY...

Remove active sessions.

Examples:
  govc session.ls | grep root
  govc session.rm 5279e245-e6f1-4533-4455-eb94353b213a

Options:
```

## snapshot.create

```
Usage: govc snapshot.create [OPTIONS] NAME

Create snapshot of VM with NAME.

Examples:
  govc snapshot.create -vm my-vm happy-vm-state

Options:
  -d=                       Snapshot description
  -m=true                   Include memory state
  -q=false                  Quiesce guest file system
  -vm=                      Virtual machine [GOVC_VM]
```

## snapshot.remove

```
Usage: govc snapshot.remove [OPTIONS] NAME

Remove snapshot of VM with given NAME.

NAME can be the snapshot name, tree path, moid or '*' to remove all snapshots.

Examples:
  govc snapshot.remove -vm my-vm happy-vm-state

Options:
  -c=true                   Consolidate disks
  -r=false                  Remove snapshot children
  -vm=                      Virtual machine [GOVC_VM]
```

## snapshot.revert

```
Usage: govc snapshot.revert [OPTIONS] [NAME]

Revert to snapshot of VM with given NAME.

If NAME is not provided, revert to the current snapshot.
Otherwise, NAME can be the snapshot name, tree path or moid.

Examples:
  govc snapshot.revert -vm my-vm happy-vm-state

Options:
  -s=false                  Suppress power on
  -vm=                      Virtual machine [GOVC_VM]
```

## snapshot.tree

```
Usage: govc snapshot.tree [OPTIONS]

List VM snapshots in a tree-like format.

The command will exit 0 with no output if VM does not have any snapshots.

Examples:
  govc snapshot.tree -vm my-vm
  govc snapshot.tree -vm my-vm -D -i

Options:
  -D=false                  Print the snapshot creation date
  -c=true                   Print the current snapshot
  -f=false                  Print the full path prefix for snapshot
  -i=false                  Print the snapshot id
  -vm=                      Virtual machine [GOVC_VM]
```

## vapp.destroy

```
Usage: govc vapp.destroy [OPTIONS] VAPP...

Options:
```

## vapp.power

```
Usage: govc vapp.power [OPTIONS]

Options:
  -force=false              Force (If force is false, the shutdown order in the vApp is executed. If force is true, all virtual machines are powered-off (regardless of shutdown order))
  -off=false                Power off
  -on=false                 Power on
  -suspend=false            Power suspend
  -vapp.ipath=              Find vapp by inventory path
```

## version

```
Usage: govc version [OPTIONS]

Options:
  -require=  Require govc version >= this value
```

## vm.change

```
Usage: govc vm.change [OPTIONS]

Change VM configuration.

Examples:
  govc vm.change -vm $vm -e smc.present=TRUE -e ich7m.present=TRUE

Options:
  -c=0                        Number of CPUs
  -e=[]                       ExtraConfig. <key>=<value>
  -g=                         Guest OS
  -m=0                        Size in MB of memory
  -name=                      Display name
  -nested-hv-enabled=<nil>    Enable nested hardware-assisted virtualization
  -sync-time-with-host=<nil>  Enable SyncTimeWithHost
  -vm=                        Virtual machine [GOVC_VM]
```

## vm.clone

```
Usage: govc vm.clone [OPTIONS] NAME

Clone VM to NAME.

Examples:
  govc vm.clone -vm template-vm new-vm

Options:
  -annotation=              VM description
  -c=0                      Number of CPUs
  -customization=           Customization Specification Name
  -datastore-cluster=       Datastore cluster [GOVC_DATASTORE_CLUSTER]
  -ds=                      Datastore [GOVC_DATASTORE]
  -folder=                  Inventory folder [GOVC_FOLDER]
  -force=false              Create VM if vmx already exists
  -host=                    Host system [GOVC_HOST]
  -m=0                      Size in MB of memory
  -net=                     Network [GOVC_NETWORK]
  -net.adapter=e1000        Network adapter type
  -net.address=             Network hardware address
  -on=true                  Power on VM
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
  -template=false           Create a Template
  -vm=                      Virtual machine [GOVC_VM]
  -waitip=false             Wait for VM to acquire IP address
```

## vm.create

```
Usage: govc vm.create [OPTIONS]

Options:
  -annotation=              VM description
  -c=1                      Number of CPUs
  -datastore-cluster=       Datastore cluster [GOVC_DATASTORE_CLUSTER]
  -disk=                    Disk path (to use existing) OR size (to create new, e.g. 20GB)
  -disk-datastore=          Datastore for disk file
  -disk.controller=scsi     Disk controller type
  -ds=                      Datastore [GOVC_DATASTORE]
  -folder=                  Inventory folder [GOVC_FOLDER]
  -force=false              Create VM if vmx already exists
  -g=otherGuest             Guest OS
  -host=                    Host system [GOVC_HOST]
  -iso=                     ISO path
  -iso-datastore=           Datastore for ISO file
  -link=true                Link specified disk
  -m=1024                   Size in MB of memory
  -net=                     Network [GOVC_NETWORK]
  -net.adapter=e1000        Network adapter type
  -net.address=             Network hardware address
  -on=true                  Power on VM. Default is true if -disk argument is given.
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
```

## vm.destroy

```
Usage: govc vm.destroy [OPTIONS]

Options:
```

## vm.disk.attach

```
Usage: govc vm.disk.attach [OPTIONS]

Options:
  -controller=              Disk controller
  -disk=                    Disk path name
  -ds=                      Datastore [GOVC_DATASTORE]
  -link=true                Link specified disk
  -persist=true             Persist attached disk
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.disk.change

```
Usage: govc vm.disk.change [OPTIONS]

Change some properties of a VM's DISK

In particular, you can change the DISK mode, and the size (as long as it is bigger)

Examples:
  govc vm.disk.change -vm VM -disk.key 2001 -size 10G
  govc vm.disk.change -vm VM -disk.label "BDD disk" -size 10G
  govc vm.disk.change -vm VM -disk.name "hard-1000-0" -size 12G
  govc vm.disk.change -vm VM -disk.filePath "[DS] VM/VM-1.vmdk" -mode nonpersistent

Options:
  -disk.filePath=           Disk file name
  -disk.key=0               Disk unique key
  -disk.label=              Disk label
  -disk.name=               Disk name
  -mode=                    Disk mode (persistent|nonpersistent|undoable|independent_persistent|independent_nonpersistent|append)
  -size=0B                  New disk size
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.disk.create

```
Usage: govc vm.disk.create [OPTIONS]

Create disk and attach to VM.

Examples:
  govc vm.disk.create -vm $name -name $name/disk1 -size 10G

Options:
  -controller=              Disk controller
  -ds=                      Datastore [GOVC_DATASTORE]
  -eager=false              Eagerly scrub new disk
  -mode=persistent          Disk mode (persistent|nonpersistent|undoable|independent_persistent|independent_nonpersistent|append)
  -name=                    Name for new disk
  -size=10.0GB              Size of new disk
  -thick=false              Thick provision new disk
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.guest.tools

```
Usage: govc vm.guest.tools [OPTIONS] VM...

Manage guest tools in VM.

Examples:
  govc vm.guest.tools -mount VM
  govc vm.guest.tools -unmount VM
  govc vm.guest.tools -upgrade -options "opt1 opt2" VM

Options:
  -mount=false              Mount tools CD installer in the guest
  -options=                 Installer options
  -unmount=false            Unmount tools CD installer in the guest
  -upgrade=false            Upgrade tools in the guest
```

## vm.info

```
Usage: govc vm.info [OPTIONS]

Options:
  -e=false                  Show ExtraConfig
  -g=true                   Show general summary
  -r=false                  Show resource summary
  -t=false                  Show ToolsConfigInfo
  -waitip=false             Wait for VM to acquire IP address
```

## vm.ip

```
Usage: govc vm.ip [OPTIONS] VM...

List IPs for VM.

By default the vm.ip command depends on vmware-tools to report the 'guest.ipAddress' field and will
wait until it has done so.  This value can also be obtained using:

  govc vm.info -json $vm | jq -r .VirtualMachines[].Guest.IpAddress

When given the '-a' flag, only IP addresses for which there is a corresponding virtual nic are listed.
If there are multiple nics, the listed addresses will be comma delimited.  The '-a' flag depends on
vmware-tools to report the 'guest.net' field and will wait until it has done so for all nics.
Note that this list includes IPv6 addresses if any, use '-v4' to filter them out.  IP addresses reported
by tools for which there is no virtual nic are not included, for example that of the 'docker0' interface.

These values can also be obtained using:

  govc vm.info -json $vm | jq -r .VirtualMachines[].Guest.Net[].IpConfig.IpAddress[].IpAddress

When given the '-n' flag, filters '-a' behavior to the nic specified by MAC address or device name.

The 'esxcli' flag does not require vmware-tools to be installed, but does require the ESX host to
have the /Net/GuestIPHack setting enabled.

The 'wait' flag default to 1hr (original default was infinite).  If a VM does not obtain an IP within
the wait time, the command will still exit with status 0.

Examples:
  govc vm.ip $vm
  govc vm.ip -wait 5m $vm
  govc vm.ip -a -v4 $vm
  govc vm.ip -n 00:0c:29:57:7b:c3 $vm
  govc vm.ip -n ethernet-0 $vm
  govc host.esxcli system settings advanced set -o /Net/GuestIPHack -i 1
  govc vm.ip -esxcli $vm

Options:
  -a=false                  Wait for an IP address on all NICs
  -esxcli=false             Use esxcli instead of guest tools
  -n=                       Wait for IP address on NIC, specified by device name or MAC
  -v4=false                 Only report IPv4 addresses
  -wait=1h0m0s              Wait time for the VM obtain an IP address
```

## vm.markastemplate

```
Usage: govc vm.markastemplate [OPTIONS] VM...

Mark VM as a virtual machine template.

Examples:
  govc vm.markastemplate $name

Options:
```

## vm.markasvm

```
Usage: govc vm.markasvm [OPTIONS] VM...

Mark VM template as a virtual machine.

Examples:
  govc vm.markasvm $name -host host1
  govc vm.markasvm $name -pool cluster1/Resources

Options:
  -host=                    Host system [GOVC_HOST]
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
```

## vm.migrate

```
Usage: govc vm.migrate [OPTIONS] VM...

Migrates VM to a specific resource pool, host or datastore.

Examples:
  govc vm.migrate -host another-host vm-1 vm-2 vm-3
  govc vm.migrate -ds another-ds vm-1 vm-2 vm-3

Options:
  -ds=                       Datastore [GOVC_DATASTORE]
  -host=                     Host system [GOVC_HOST]
  -pool=                     Resource pool [GOVC_RESOURCE_POOL]
  -priority=defaultPriority  The task priority
```

## vm.network.add

```
Usage: govc vm.network.add [OPTIONS]

Add network adapter to VM.

Examples:
  govc vm.network.add -vm $vm -net "VM Network" -net.adapter e1000e
  govc device.info -vm $vm ethernet-*

Options:
  -net=                     Network [GOVC_NETWORK]
  -net.adapter=e1000        Network adapter type
  -net.address=             Network hardware address
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.network.change

```
Usage: govc vm.network.change [OPTIONS] DEVICE

Change network DEVICE configuration.

Examples:
  govc vm.network.change -vm $vm -net PG2 ethernet-0
  govc vm.network.change -vm $vm -net.address 00:00:0f:2e:5d:69 ethernet-0
  govc device.info -vm $vm ethernet-*

Options:
  -net=                     Network [GOVC_NETWORK]
  -net.adapter=e1000        Network adapter type
  -net.address=             Network hardware address
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.power

```
Usage: govc vm.power [OPTIONS]

Options:
  -force=false              Force (ignore state error and hard shutdown/reboot if tools unavailable)
  -off=false                Power off
  -on=false                 Power on
  -r=false                  Reboot guest
  -reset=false              Power reset
  -s=false                  Shutdown guest
  -suspend=false            Power suspend
```

## vm.question

```
Usage: govc vm.question [OPTIONS]

Options:
  -answer=                  Answer to question
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.rdm.attach

```
Usage: govc vm.rdm.attach [OPTIONS]

Attach DEVICE to VM with RDM.

Examples:
  govc vm.rdm.attach -vm VM -device /vmfs/devices/disks/naa.000000000000000000000000000000000

Options:
  -device=                  Device Name
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.rdm.ls

```
Usage: govc vm.rdm.ls [OPTIONS]

List available devices that could be attach to VM with RDM.

Examples:
  govc vm.rdm.ls -vm VM

Options:
  -vm=                      Virtual machine [GOVC_VM]
```

## vm.register

```
Usage: govc vm.register [OPTIONS] VMX

Add an existing VM to the inventory.

VMX is a path to the vm config file, relative to DATASTORE.

Examples:
  govc vm.register path/name.vmx

Options:
  -as-template=false        Mark VM as template
  -ds=                      Datastore [GOVC_DATASTORE]
  -folder=                  Inventory folder [GOVC_FOLDER]
  -host=                    Host system [GOVC_HOST]
  -name=                    Name of the VM
  -pool=                    Resource pool [GOVC_RESOURCE_POOL]
```

## vm.unregister

```
Usage: govc vm.unregister [OPTIONS] VM...

Remove VM from inventory without removing any of the VM files on disk.

Options:
```

## vm.vnc

```
Usage: govc vm.vnc [OPTIONS] VM...

Enable or disable VNC for VM.

Port numbers are automatically chosen if not specified.

If neither -enable or -disable is specified, the current state is returned.

Examples:
  govc vm.vnc -enable -password 1234 $vm | awk '{print $2}' | xargs open

Options:
  -disable=false            Disable VNC
  -enable=false             Enable VNC
  -password=                VNC password
  -port=-1                  VNC port (-1 for auto-select)
  -port-range=5900-5999     VNC port auto-select range
```

