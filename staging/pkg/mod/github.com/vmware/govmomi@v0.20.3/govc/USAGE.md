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

<details><summary>Contents</summary>

 - [about](#about)
 - [about.cert](#aboutcert)
 - [cluster.add](#clusteradd)
 - [cluster.change](#clusterchange)
 - [cluster.create](#clustercreate)
 - [cluster.group.change](#clustergroupchange)
 - [cluster.group.create](#clustergroupcreate)
 - [cluster.group.ls](#clustergroupls)
 - [cluster.group.remove](#clustergroupremove)
 - [cluster.override.change](#clusteroverridechange)
 - [cluster.override.info](#clusteroverrideinfo)
 - [cluster.override.remove](#clusteroverrideremove)
 - [cluster.rule.change](#clusterrulechange)
 - [cluster.rule.create](#clusterrulecreate)
 - [cluster.rule.info](#clusterruleinfo)
 - [cluster.rule.ls](#clusterrulels)
 - [cluster.rule.remove](#clusterruleremove)
 - [datacenter.create](#datacentercreate)
 - [datacenter.info](#datacenterinfo)
 - [datastore.cluster.change](#datastoreclusterchange)
 - [datastore.cluster.info](#datastoreclusterinfo)
 - [datastore.cp](#datastorecp)
 - [datastore.create](#datastorecreate)
 - [datastore.disk.create](#datastorediskcreate)
 - [datastore.disk.inflate](#datastorediskinflate)
 - [datastore.disk.info](#datastorediskinfo)
 - [datastore.disk.shrink](#datastorediskshrink)
 - [datastore.download](#datastoredownload)
 - [datastore.info](#datastoreinfo)
 - [datastore.ls](#datastorels)
 - [datastore.mkdir](#datastoremkdir)
 - [datastore.mv](#datastoremv)
 - [datastore.remove](#datastoreremove)
 - [datastore.rm](#datastorerm)
 - [datastore.tail](#datastoretail)
 - [datastore.upload](#datastoreupload)
 - [datastore.vsan.dom.ls](#datastorevsandomls)
 - [datastore.vsan.dom.rm](#datastorevsandomrm)
 - [device.boot](#deviceboot)
 - [device.cdrom.add](#devicecdromadd)
 - [device.cdrom.eject](#devicecdromeject)
 - [device.cdrom.insert](#devicecdrominsert)
 - [device.connect](#deviceconnect)
 - [device.disconnect](#devicedisconnect)
 - [device.floppy.add](#devicefloppyadd)
 - [device.floppy.eject](#devicefloppyeject)
 - [device.floppy.insert](#devicefloppyinsert)
 - [device.info](#deviceinfo)
 - [device.ls](#devicels)
 - [device.remove](#deviceremove)
 - [device.scsi.add](#devicescsiadd)
 - [device.serial.add](#deviceserialadd)
 - [device.serial.connect](#deviceserialconnect)
 - [device.serial.disconnect](#deviceserialdisconnect)
 - [device.usb.add](#deviceusbadd)
 - [disk.create](#diskcreate)
 - [disk.ls](#diskls)
 - [disk.register](#diskregister)
 - [disk.rm](#diskrm)
 - [disk.snapshot.create](#disksnapshotcreate)
 - [disk.snapshot.ls](#disksnapshotls)
 - [disk.snapshot.rm](#disksnapshotrm)
 - [disk.tags.attach](#disktagsattach)
 - [disk.tags.detach](#disktagsdetach)
 - [dvs.add](#dvsadd)
 - [dvs.create](#dvscreate)
 - [dvs.portgroup.add](#dvsportgroupadd)
 - [dvs.portgroup.change](#dvsportgroupchange)
 - [dvs.portgroup.info](#dvsportgroupinfo)
 - [env](#env)
 - [events](#events)
 - [export.ovf](#exportovf)
 - [extension.info](#extensioninfo)
 - [extension.register](#extensionregister)
 - [extension.setcert](#extensionsetcert)
 - [extension.unregister](#extensionunregister)
 - [fields.add](#fieldsadd)
 - [fields.info](#fieldsinfo)
 - [fields.ls](#fieldsls)
 - [fields.rename](#fieldsrename)
 - [fields.rm](#fieldsrm)
 - [fields.set](#fieldsset)
 - [find](#find)
 - [firewall.ruleset.find](#firewallrulesetfind)
 - [folder.create](#foldercreate)
 - [folder.info](#folderinfo)
 - [guest.chmod](#guestchmod)
 - [guest.chown](#guestchown)
 - [guest.download](#guestdownload)
 - [guest.getenv](#guestgetenv)
 - [guest.kill](#guestkill)
 - [guest.ls](#guestls)
 - [guest.mkdir](#guestmkdir)
 - [guest.mktemp](#guestmktemp)
 - [guest.mv](#guestmv)
 - [guest.ps](#guestps)
 - [guest.rm](#guestrm)
 - [guest.rmdir](#guestrmdir)
 - [guest.run](#guestrun)
 - [guest.start](#gueststart)
 - [guest.touch](#guesttouch)
 - [guest.upload](#guestupload)
 - [host.account.create](#hostaccountcreate)
 - [host.account.remove](#hostaccountremove)
 - [host.account.update](#hostaccountupdate)
 - [host.add](#hostadd)
 - [host.autostart.add](#hostautostartadd)
 - [host.autostart.configure](#hostautostartconfigure)
 - [host.autostart.info](#hostautostartinfo)
 - [host.autostart.remove](#hostautostartremove)
 - [host.cert.csr](#hostcertcsr)
 - [host.cert.import](#hostcertimport)
 - [host.cert.info](#hostcertinfo)
 - [host.date.change](#hostdatechange)
 - [host.date.info](#hostdateinfo)
 - [host.disconnect](#hostdisconnect)
 - [host.esxcli](#hostesxcli)
 - [host.info](#hostinfo)
 - [host.maintenance.enter](#hostmaintenanceenter)
 - [host.maintenance.exit](#hostmaintenanceexit)
 - [host.option.ls](#hostoptionls)
 - [host.option.set](#hostoptionset)
 - [host.portgroup.add](#hostportgroupadd)
 - [host.portgroup.change](#hostportgroupchange)
 - [host.portgroup.info](#hostportgroupinfo)
 - [host.portgroup.remove](#hostportgroupremove)
 - [host.reconnect](#hostreconnect)
 - [host.remove](#hostremove)
 - [host.service](#hostservice)
 - [host.service.ls](#hostservicels)
 - [host.shutdown](#hostshutdown)
 - [host.storage.info](#hoststorageinfo)
 - [host.storage.mark](#hoststoragemark)
 - [host.storage.partition](#hoststoragepartition)
 - [host.vnic.info](#hostvnicinfo)
 - [host.vnic.service](#hostvnicservice)
 - [host.vswitch.add](#hostvswitchadd)
 - [host.vswitch.info](#hostvswitchinfo)
 - [host.vswitch.remove](#hostvswitchremove)
 - [import.ova](#importova)
 - [import.ovf](#importovf)
 - [import.spec](#importspec)
 - [import.vmdk](#importvmdk)
 - [license.add](#licenseadd)
 - [license.assign](#licenseassign)
 - [license.assigned.ls](#licenseassignedls)
 - [license.decode](#licensedecode)
 - [license.label.set](#licenselabelset)
 - [license.ls](#licensels)
 - [license.remove](#licenseremove)
 - [logs](#logs)
 - [logs.download](#logsdownload)
 - [logs.ls](#logsls)
 - [ls](#ls)
 - [metric.change](#metricchange)
 - [metric.info](#metricinfo)
 - [metric.interval.change](#metricintervalchange)
 - [metric.interval.info](#metricintervalinfo)
 - [metric.ls](#metricls)
 - [metric.reset](#metricreset)
 - [metric.sample](#metricsample)
 - [object.collect](#objectcollect)
 - [object.destroy](#objectdestroy)
 - [object.method](#objectmethod)
 - [object.mv](#objectmv)
 - [object.reload](#objectreload)
 - [object.rename](#objectrename)
 - [option.ls](#optionls)
 - [option.set](#optionset)
 - [permissions.ls](#permissionsls)
 - [permissions.remove](#permissionsremove)
 - [permissions.set](#permissionsset)
 - [pool.change](#poolchange)
 - [pool.create](#poolcreate)
 - [pool.destroy](#pooldestroy)
 - [pool.info](#poolinfo)
 - [role.create](#rolecreate)
 - [role.ls](#rolels)
 - [role.remove](#roleremove)
 - [role.update](#roleupdate)
 - [role.usage](#roleusage)
 - [session.login](#sessionlogin)
 - [session.logout](#sessionlogout)
 - [session.ls](#sessionls)
 - [session.rm](#sessionrm)
 - [snapshot.create](#snapshotcreate)
 - [snapshot.remove](#snapshotremove)
 - [snapshot.revert](#snapshotrevert)
 - [snapshot.tree](#snapshottree)
 - [sso.service.ls](#ssoservicels)
 - [sso.user.create](#ssousercreate)
 - [sso.user.id](#ssouserid)
 - [sso.user.ls](#ssouserls)
 - [sso.user.rm](#ssouserrm)
 - [sso.user.update](#ssouserupdate)
 - [tags.attach](#tagsattach)
 - [tags.attached.ls](#tagsattachedls)
 - [tags.category.create](#tagscategorycreate)
 - [tags.category.info](#tagscategoryinfo)
 - [tags.category.ls](#tagscategoryls)
 - [tags.category.rm](#tagscategoryrm)
 - [tags.category.update](#tagscategoryupdate)
 - [tags.create](#tagscreate)
 - [tags.detach](#tagsdetach)
 - [tags.info](#tagsinfo)
 - [tags.ls](#tagsls)
 - [tags.rm](#tagsrm)
 - [tags.update](#tagsupdate)
 - [task.cancel](#taskcancel)
 - [tasks](#tasks)
 - [vapp.destroy](#vappdestroy)
 - [vapp.power](#vapppower)
 - [version](#version)
 - [vm.change](#vmchange)
 - [vm.clone](#vmclone)
 - [vm.console](#vmconsole)
 - [vm.create](#vmcreate)
 - [vm.destroy](#vmdestroy)
 - [vm.disk.attach](#vmdiskattach)
 - [vm.disk.change](#vmdiskchange)
 - [vm.disk.create](#vmdiskcreate)
 - [vm.guest.tools](#vmguesttools)
 - [vm.info](#vminfo)
 - [vm.ip](#vmip)
 - [vm.keystrokes](#vmkeystrokes)
 - [vm.markastemplate](#vmmarkastemplate)
 - [vm.markasvm](#vmmarkasvm)
 - [vm.migrate](#vmmigrate)
 - [vm.network.add](#vmnetworkadd)
 - [vm.network.change](#vmnetworkchange)
 - [vm.option.info](#vmoptioninfo)
 - [vm.power](#vmpower)
 - [vm.question](#vmquestion)
 - [vm.rdm.attach](#vmrdmattach)
 - [vm.rdm.ls](#vmrdmls)
 - [vm.register](#vmregister)
 - [vm.unregister](#vmunregister)
 - [vm.upgrade](#vmupgrade)
 - [vm.vnc](#vmvnc)

</details>

## about

```
Usage: govc about [OPTIONS]

Display About info for HOST.

System information including the name, type, version, and build number.

Examples:
  govc about
  govc about -json | jq -r .About.ProductLineId

Options:
  -l=false               Include service content
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
  -show=false            Show PEM encoded server certificate only
  -thumbprint=false      Output host hash and thumbprint only
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
  -cluster=              Cluster [GOVC_CLUSTER]
  -connect=true          Immediately connect to host
  -force=false           Force when host is managed by another VC
  -hostname=             Hostname or IP address of the host
  -license=              Assign license key
  -noverify=false        Accept host thumbprint without verification
  -password=             Password of administration account on the host
  -thumbprint=           SHA-1 thumbprint of the host's SSL certificate
  -username=             Username of administration account on the host
```

## cluster.change

```
Usage: govc cluster.change [OPTIONS] CLUSTER...

Change configuration of the given clusters.

Examples:
  govc cluster.change -drs-enabled -vsan-enabled -vsan-autoclaim ClusterA
  govc cluster.change -drs-enabled=false ClusterB

Options:
  -drs-enabled=<nil>     Enable DRS
  -drs-mode=             DRS behavior for virtual machines: manual, partiallyAutomated, fullyAutomated
  -ha-enabled=<nil>      Enable HA
  -vsan-autoclaim=<nil>  Autoclaim storage on cluster hosts
  -vsan-enabled=<nil>    Enable vSAN
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
  -folder=               Inventory folder [GOVC_FOLDER]
```

## cluster.group.change

```
Usage: govc cluster.group.change [OPTIONS] NAME...

Set cluster group members.

Examples:
  govc cluster.group.change -name my_group vm_a vm_b vm_c # set
  govc cluster.group.change -name my_group vm_a vm_b vm_c $(govc cluster.group.ls -name my_group) vm_d # add
  govc cluster.group.ls -name my_group | grep -v vm_b | xargs govc cluster.group.change -name my_group vm_a vm_b vm_c # remove

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -name=                 Cluster group name
```

## cluster.group.create

```
Usage: govc cluster.group.create [OPTIONS]

Create cluster group.

One of '-vm' or '-host' must be provided to specify the group type.

Examples:
  govc cluster.group.create -name my_vm_group -vm vm_a vm_b vm_c
  govc cluster.group.create -name my_host_group -host host_a host_b host_c

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -host=false            Create cluster Host group
  -name=                 Cluster group name
  -vm=false              Create cluster VM group
```

## cluster.group.ls

```
Usage: govc cluster.group.ls [OPTIONS]

List cluster groups and group members.

Examples:
  govc cluster.group.ls -cluster my_cluster
  govc cluster.group.ls -cluster my_cluster -name my_group

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -name=                 Cluster group name
```

## cluster.group.remove

```
Usage: govc cluster.group.remove [OPTIONS]

Remove cluster group.

Examples:
  govc cluster.group.remove -cluster my_cluster -name my_group

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -name=                 Cluster group name
```

## cluster.override.change

```
Usage: govc cluster.override.change [OPTIONS]

Change cluster VM overrides.

Examples:
  govc cluster.override.change -cluster cluster_1 -vm vm_1 -ha-restart-priority high
  govc cluster.override.change -cluster cluster_1 -vm vm_2 -drs-enabled=false
  govc cluster.override.change -cluster cluster_1 -vm vm_3 -drs-enabled -drs-mode fullyAutomated

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -drs-enabled=<nil>     Enable DRS
  -drs-mode=             DRS behavior for virtual machines: manual, partiallyAutomated, fullyAutomated
  -ha-restart-priority=  HA restart priority: disabled, low, medium, high
  -vm=                   Virtual machine [GOVC_VM]
```

## cluster.override.info

```
Usage: govc cluster.override.info [OPTIONS]

Cluster VM overrides info.

Examples:
  govc cluster.override.info
  govc cluster.override.info -json

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
```

## cluster.override.remove

```
Usage: govc cluster.override.remove [OPTIONS]

Remove cluster VM overrides.

Examples:
  govc cluster.override.remove -cluster cluster_1 -vm vm_1

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -vm=                   Virtual machine [GOVC_VM]
```

## cluster.rule.change

```
Usage: govc cluster.rule.change [OPTIONS] NAME...

Change cluster rule.

Examples:
  govc cluster.rule.change -cluster my_cluster -name my_rule -enable=false

Options:
  -cluster=                 Cluster [GOVC_CLUSTER]
  -enable=<nil>             Enable rule
  -host-affine-group=       Host affine group name
  -host-anti-affine-group=  Host anti-affine group name
  -l=false                  Long listing format
  -mandatory=<nil>          Enforce rule compliance
  -name=                    Cluster rule name
  -vm-group=                VM group name
```

## cluster.rule.create

```
Usage: govc cluster.rule.create [OPTIONS] NAME...

Create cluster rule.

Rules are not enabled by default, use the 'enable' flag to enable upon creation or cluster.rule.change after creation.

One of '-affinity', '-anti-affinity', '-depends' or '-vm-host' must be provided to specify the rule type.

With '-affinity' or '-anti-affinity', at least 2 vm NAME arguments must be specified.

With '-depends', vm group NAME and vm group dependency NAME arguments must be specified.

With '-vm-host', use the '-vm-group' flag combined with the '-host-affine-group' and/or '-host-anti-affine-group' flags.

Examples:
  govc cluster.rule.create -name pod1 -enable -affinity vm_a vm_b vm_c
  govc cluster.rule.create -name pod2 -enable -anti-affinity vm_d vm_e vm_f
  govc cluster.rule.create -name pod3 -enable -mandatory -vm-host -vm-group my_vms -host-affine-group my_hosts
  govc cluster.rule.create -name pod4 -depends vm_group_app vm_group_db

Options:
  -affinity=false           Keep Virtual Machines Together
  -anti-affinity=false      Separate Virtual Machines
  -cluster=                 Cluster [GOVC_CLUSTER]
  -depends=false            Virtual Machines to Virtual Machines
  -enable=<nil>             Enable rule
  -host-affine-group=       Host affine group name
  -host-anti-affine-group=  Host anti-affine group name
  -l=false                  Long listing format
  -mandatory=<nil>          Enforce rule compliance
  -name=                    Cluster rule name
  -vm-group=                VM group name
  -vm-host=false            Virtual Machines to Hosts
```

## cluster.rule.info

```
Usage: govc cluster.rule.info [OPTIONS]

Provides detailed infos about cluster rules, their types and rule members.

Examples:
  govc cluster.rule.info -cluster my_cluster
  govc cluster.rule.info -cluster my_cluster -name my_rule

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -l=false               Long listing format
  -name=                 Cluster rule name
```

## cluster.rule.ls

```
Usage: govc cluster.rule.ls [OPTIONS]

List cluster rules and rule members.

Examples:
  govc cluster.rule.ls -cluster my_cluster
  govc cluster.rule.ls -cluster my_cluster -name my_rule
  govc cluster.rule.ls -cluster my_cluster -l
  govc cluster.rule.ls -cluster my_cluster -name my_rule -l

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -l=false               Long listing format
  -name=                 Cluster rule name
```

## cluster.rule.remove

```
Usage: govc cluster.rule.remove [OPTIONS]

Remove cluster rule.

Examples:
  govc cluster.group.remove -cluster my_cluster -name my_rule

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -l=false               Long listing format
  -name=                 Cluster rule name
```

## datacenter.create

```
Usage: govc datacenter.create [OPTIONS] NAME...

Options:
  -folder=               Inventory folder [GOVC_FOLDER]
```

## datacenter.info

```
Usage: govc datacenter.info [OPTIONS] [PATH]...

Options:
```

## datastore.cluster.change

```
Usage: govc datastore.cluster.change [OPTIONS] CLUSTER...

Change configuration of the given datastore clusters.

Examples:
  govc datastore.cluster.change -drs-enabled ClusterA
  govc datastore.cluster.change -drs-enabled=false ClusterB

Options:
  -drs-enabled=<nil>     Enable Storage DRS
  -drs-mode=             Storage DRS behavior: manual, automated
```

## datastore.cluster.info

```
Usage: govc datastore.cluster.info [OPTIONS] [PATH]...

Options:
```

## datastore.cp

```
Usage: govc datastore.cp [OPTIONS] SRC DST

Copy SRC to DST on DATASTORE.

Examples:
  govc datastore.cp foo/foo.vmx foo/foo.vmx.old
  govc datastore.cp -f my.vmx foo/foo.vmx
  govc datastore.cp disks/disk1.vmdk disks/disk2.vmdk
  govc datastore.cp disks/disk1.vmdk -dc-target DC2 disks/disk2.vmdk
  govc datastore.cp disks/disk1.vmdk -ds-target NFS-2 disks/disk2.vmdk

Options:
  -dc-target=            Datacenter destination (defaults to -dc)
  -ds=                   Datastore [GOVC_DATASTORE]
  -ds-target=            Datastore destination (defaults to -ds)
  -f=false               If true, overwrite any identically named file at the destination
  -t=true                Use file type to choose disk or file manager
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
  -disk=                 Canonical name of disk (VMFS only)
  -force=false           Ignore DuplicateName error if datastore is already mounted on a host
  -host=                 Host system [GOVC_HOST]
  -mode=readOnly         Access mode for the mount point (readOnly|readWrite)
  -name=                 Datastore name
  -password=             Password to use when connecting (CIFS only)
  -path=                 Local directory path for the datastore (local only)
  -remote-host=          Remote hostname of the NAS datastore
  -remote-path=          Remote path of the NFS mount point
  -type=                 Datastore type (NFS|NFS41|CIFS|VMFS|local)
  -username=             Username to use when connecting (CIFS only)
```

## datastore.disk.create

```
Usage: govc datastore.disk.create [OPTIONS] VMDK

Create VMDK on DS.

Examples:
  govc datastore.mkdir disks
  govc datastore.disk.create -size 24G disks/disk1.vmdk
  govc datastore.disk.create disks/parent.vmdk disk/child.vmdk

Options:
  -a=lsiLogic            Disk adapter
  -d=thin                Disk format
  -ds=                   Datastore [GOVC_DATASTORE]
  -f=false               Force
  -size=10.0GB           Size of new disk
  -uuid=                 Disk UUID
```

## datastore.disk.inflate

```
Usage: govc datastore.disk.inflate [OPTIONS] VMDK

Inflate VMDK on DS.

Examples:
  govc datastore.disk.inflate disks/disk1.vmdk

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
```

## datastore.disk.info

```
Usage: govc datastore.disk.info [OPTIONS] VMDK

Query VMDK info on DS.

Examples:
  govc datastore.disk.info disks/disk1.vmdk

Options:
  -c=false               Chain format
  -d=false               Include datastore in output
  -ds=                   Datastore [GOVC_DATASTORE]
  -p=true                Include parents
  -uuid=false            Include disk UUID
```

## datastore.disk.shrink

```
Usage: govc datastore.disk.shrink [OPTIONS] VMDK

Shrink VMDK on DS.

Examples:
  govc datastore.disk.shrink disks/disk1.vmdk

Options:
  -copy=<nil>            Perform shrink in-place mode if false, copy-shrink mode otherwise
  -ds=                   Datastore [GOVC_DATASTORE]
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
  -ds=                   Datastore [GOVC_DATASTORE]
  -host=                 Host system [GOVC_HOST]
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
  -R=false               List subdirectories recursively
  -a=false               Do not ignore entries starting with .
  -ds=                   Datastore [GOVC_DATASTORE]
  -l=false               Long listing format
  -p=false               Append / indicator to directories
```

## datastore.mkdir

```
Usage: govc datastore.mkdir [OPTIONS] DIRECTORY

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -namespace=false       Return uuid of namespace created on vsan datastore
  -p=false               Create intermediate directories as needed
```

## datastore.mv

```
Usage: govc datastore.mv [OPTIONS] SRC DST

Move SRC to DST on DATASTORE.

Examples:
  govc datastore.mv foo/foo.vmx foo/foo.vmx.old
  govc datastore.mv -f my.vmx foo/foo.vmx

Options:
  -dc-target=            Datacenter destination (defaults to -dc)
  -ds=                   Datastore [GOVC_DATASTORE]
  -ds-target=            Datastore destination (defaults to -ds)
  -f=false               If true, overwrite any identically named file at the destination
  -t=true                Use file type to choose disk or file manager
```

## datastore.remove

```
Usage: govc datastore.remove [OPTIONS] HOST...

Remove datastore from HOST.

Examples:
  govc datastore.remove -ds nfsDatastore cluster1
  govc datastore.remove -ds nasDatastore host1 host2 host3

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -host=                 Host system [GOVC_HOST]
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
  -ds=                   Datastore [GOVC_DATASTORE]
  -f=false               Force; ignore nonexistent files and arguments
  -namespace=false       Path is uuid of namespace on vsan datastore
  -t=true                Use file type to choose disk or file manager
```

## datastore.tail

```
Usage: govc datastore.tail [OPTIONS] PATH

Output the last part of datastore files.

Examples:
  govc datastore.tail -n 100 vm-name/vmware.log
  govc datastore.tail -n 0 -f vm-name/vmware.log

Options:
  -c=-1                  Output the last NUM bytes
  -ds=                   Datastore [GOVC_DATASTORE]
  -f=false               Output appended data as the file grows
  -host=                 Host system [GOVC_HOST]
  -n=10                  Output the last NUM lines
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
  -ds=                   Datastore [GOVC_DATASTORE]
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
  -ds=                   Datastore [GOVC_DATASTORE]
  -l=false               Long listing
  -o=false               List orphan objects
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
  -ds=                   Datastore [GOVC_DATASTORE]
  -f=false               Force delete
  -v=false               Print deleted UUIDs to stdout, failed to stderr
```

## device.boot

```
Usage: govc device.boot [OPTIONS]

Configure VM boot settings.

Examples:
  govc device.boot -vm $vm -delay 1000 -order floppy,cdrom,ethernet,disk
  govc device.boot -vm $vm -order - # reset boot order

Options:
  -delay=0               Delay in ms before starting the boot sequence
  -order=                Boot device order [-,floppy,cdrom,ethernet,disk]
  -retry=false           If true, retry boot after retry-delay
  -retry-delay=0         Delay in ms before a boot retry
  -setup=false           If true, enter BIOS setup on next boot
  -vm=                   Virtual machine [GOVC_VM]
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
  -controller=           IDE controller name
  -vm=                   Virtual machine [GOVC_VM]
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
  -device=               CD-ROM device name
  -vm=                   Virtual machine [GOVC_VM]
```

## device.cdrom.insert

```
Usage: govc device.cdrom.insert [OPTIONS] ISO

Insert media on datastore into CD-ROM device.

If device is not specified, the first CD-ROM device is used.

Examples:
  govc device.cdrom.insert -vm vm-1 -device cdrom-3000 images/boot.iso

Options:
  -device=               CD-ROM device name
  -ds=                   Datastore [GOVC_DATASTORE]
  -vm=                   Virtual machine [GOVC_VM]
```

## device.connect

```
Usage: govc device.connect [OPTIONS] DEVICE...

Connect DEVICE on VM.

Examples:
  govc device.connect -vm $name cdrom-3000

Options:
  -vm=                   Virtual machine [GOVC_VM]
```

## device.disconnect

```
Usage: govc device.disconnect [OPTIONS] DEVICE...

Disconnect DEVICE on VM.

Examples:
  govc device.disconnect -vm $name cdrom-3000

Options:
  -vm=                   Virtual machine [GOVC_VM]
```

## device.floppy.add

```
Usage: govc device.floppy.add [OPTIONS]

Add floppy device to VM.

Examples:
  govc device.floppy.add -vm $vm
  govc device.info floppy-*

Options:
  -vm=                   Virtual machine [GOVC_VM]
```

## device.floppy.eject

```
Usage: govc device.floppy.eject [OPTIONS]

Eject image from floppy device.

If device is not specified, the first floppy device is used.

Examples:
  govc device.floppy.eject -vm vm-1

Options:
  -device=               Floppy device name
  -vm=                   Virtual machine [GOVC_VM]
```

## device.floppy.insert

```
Usage: govc device.floppy.insert [OPTIONS] IMG

Insert IMG on datastore into floppy device.

If device is not specified, the first floppy device is used.

Examples:
  govc device.floppy.insert -vm vm-1 vm-1/config.img

Options:
  -device=               Floppy device name
  -ds=                   Datastore [GOVC_DATASTORE]
  -vm=                   Virtual machine [GOVC_VM]
```

## device.info

```
Usage: govc device.info [OPTIONS] [DEVICE]...

Device info for VM.

Examples:
  govc device.info -vm $name
  govc device.info -vm $name disk-*
  govc device.info -vm $name -json ethernet-0 | jq -r .Devices[].MacAddress

Options:
  -net=                  Network [GOVC_NETWORK]
  -net.adapter=e1000     Network adapter type
  -net.address=          Network hardware address
  -vm=                   Virtual machine [GOVC_VM]
```

## device.ls

```
Usage: govc device.ls [OPTIONS]

List devices for VM.

Examples:
  govc device.ls -vm $name
  govc device.ls -vm $name disk-*
  govc device.ls -vm $name -json | jq '.Devices[].Name'

Options:
  -boot=false            List devices configured in the VM's boot options
  -vm=                   Virtual machine [GOVC_VM]
```

## device.remove

```
Usage: govc device.remove [OPTIONS] DEVICE...

Remove DEVICE from VM.

Examples:
  govc device.remove -vm $name cdrom-3000
  govc device.remove -vm $name disk-1000
  govc device.remove -vm $name -keep disk-*

Options:
  -keep=false            Keep files in datastore
  -vm=                   Virtual machine [GOVC_VM]
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
  -hot=false             Enable hot-add/remove
  -sharing=noSharing     SCSI sharing
  -type=lsilogic         SCSI controller type (lsilogic|buslogic|pvscsi|lsilogic-sas)
  -vm=                   Virtual machine [GOVC_VM]
```

## device.serial.add

```
Usage: govc device.serial.add [OPTIONS]

Add serial port to VM.

Examples:
  govc device.serial.add -vm $vm
  govc device.info -vm $vm serialport-*

Options:
  -vm=                   Virtual machine [GOVC_VM]
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
  -client=false          Use client direction
  -device=               serial port device name
  -vm=                   Virtual machine [GOVC_VM]
  -vspc-proxy=           vSPC proxy URI
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
  -device=               serial port device name
  -vm=                   Virtual machine [GOVC_VM]
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
  -auto=true             Enable ability to hot plug devices
  -ehci=true             Enable enhanced host controller interface (USB 2.0)
  -type=usb              USB controller type (usb|xhci)
  -vm=                   Virtual machine [GOVC_VM]
```

## disk.create

```
Usage: govc disk.create [OPTIONS] NAME

Create disk NAME on DS.

Examples:
  govc disk.create -size 24G my-disk

Options:
  -datastore-cluster=    Datastore cluster [GOVC_DATASTORE_CLUSTER]
  -ds=                   Datastore [GOVC_DATASTORE]
  -keep=<nil>            Keep disk after VM is deleted
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
  -size=10.0GB           Size of new disk
```

## disk.ls

```
Usage: govc disk.ls [OPTIONS] [ID]...

List disk IDs on DS.

Examples:
  govc disk.ls
  govc disk.ls -l -T
  govc disk.ls -l e9b06a8b-d047-4d3c-b15b-43ea9608b1a6
  govc disk.ls -c k8s-region -t us-west-2

Options:
  -T=false               List attached tags
  -c=                    Query tag category
  -ds=                   Datastore [GOVC_DATASTORE]
  -l=false               Long listing format
  -t=                    Query tag name
```

## disk.register

```
Usage: govc disk.register [OPTIONS] PATH [NAME]

Register existing disk on DS.

Examples:
  govc disk.register disks/disk1.vmdk my-disk

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
```

## disk.rm

```
Usage: govc disk.rm [OPTIONS] ID

Remove disk ID on DS.

Examples:
  govc disk.rm ID

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
```

## disk.snapshot.create

```
Usage: govc disk.snapshot.create [OPTIONS] ID DESC

Create snapshot of ID on DS.

Examples:
  govc disk.snapshot.create b9fe5f17-3b87-4a03-9739-09a82ddcc6b0 my-disk-snapshot

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
```

## disk.snapshot.ls

```
Usage: govc disk.snapshot.ls [OPTIONS] ID

List snapshots for disk ID on DS.

Examples:
  govc snapshot.disk.ls -l 9b06a8b-d047-4d3c-b15b-43ea9608b1a6

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -l=false               Long listing format
```

## disk.snapshot.rm

```
Usage: govc disk.snapshot.rm [OPTIONS] ID SID

Remove disk ID snapshot ID on DS.

Examples:
  govc disk.snapshot.rm ffe6a398-eb8e-4eaa-9118-e1f16b8b8e3c ecbca542-0a25-4127-a585-82e4047750d6

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
```

## disk.tags.attach

```
Usage: govc disk.tags.attach [OPTIONS] NAME ID

Attach tag NAME to disk ID.

Examples:
  govc disk.tags.attach -c k8s-region k8s-region-us $id

Options:
  -c=                    Tag category
```

## disk.tags.detach

```
Usage: govc disk.tags.detach [OPTIONS] NAME ID

Detach tag NAME from disk ID.

Examples:
  govc disk.tags.detach -c k8s-region k8s-region-us $id

Options:
  -c=                    Tag category
```

## dvs.add

```
Usage: govc dvs.add [OPTIONS] HOST...

Add hosts to DVS.

Examples:
  govc dvs.add -dvs dvsName -pnic vmnic1 hostA hostB hostC

Options:
  -dvs=                  DVS path
  -host=                 Host system [GOVC_HOST]
  -pnic=vmnic0           Name of the host physical NIC
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
  -folder=               Inventory folder [GOVC_FOLDER]
  -product-version=      DVS product version
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
  -dvs=                  DVS path
  -nports=128            Number of ports
  -type=earlyBinding     Portgroup type (earlyBinding|lateBinding|ephemeral)
  -vlan=0                VLAN ID
```

## dvs.portgroup.change

```
Usage: govc dvs.portgroup.change [OPTIONS] PATH

Change DVS portgroup configuration.

Examples:
  govc dvs.portgroup.change -nports 26 ExternalNetwork
  govc dvs.portgroup.change -vlan 3214 ExternalNetwork

Options:
  -nports=0              Number of ports
  -type=earlyBinding     Portgroup type (earlyBinding|lateBinding|ephemeral)
  -vlan=0                VLAN ID
```

## dvs.portgroup.info

```
Usage: govc dvs.portgroup.info [OPTIONS] DVS

Portgroup info for DVS.

Examples:
  govc dvs.portgroup.info DSwitch
  govc dvs.portgroup.info -pg InternalNetwork DSwitch
  govc find / -type DistributedVirtualSwitch | xargs -n1 govc dvs.portgroup.info

Options:
  -active=false          Filter by port active or inactive status
  -connected=false       Filter by port connected or disconnected status
  -count=0               Number of matches to return (0 = unlimited)
  -inside=true           Filter by port inside or outside status
  -pg=                   Distributed Virtual Portgroup
  -r=false               Show DVS rules
  -uplinkPort=false      Filter for uplink ports
  -vlan=0                Filter by VLAN ID (0 = unfiltered)
```

## env

```
Usage: govc env [OPTIONS]

Output the environment variables for this client.

If credentials are included in the url, they are split into separate variables.
Useful as bash scripting helper to parse GOVC_URL.

Options:
  -x=false               Output variables for each GOVC_URL component
```

## events

```
Usage: govc events [OPTIONS] [PATH]...

Display events.

Examples:
  govc events vm/my-vm1 vm/my-vm2
  govc events /dc1/vm/* /dc2/vm/*
  govc events -type VmPoweredOffEvent -type VmPoweredOnEvent
  govc ls -t HostSystem host/* | xargs govc events | grep -i vsan

Options:
  -f=false               Follow event stream
  -force=false           Disable number objects to monitor limit
  -l=false               Long listing format
  -n=25                  Output the last N events
  -type=[]               Include only the specified event types
```

## export.ovf

```
Usage: govc export.ovf [OPTIONS] DIR

Export VM.

Examples:
  govc export.ovf -vm $vm DIR

Options:
  -f=false               Overwrite existing
  -i=false               Include image files (*.{iso,img})
  -name=                 Specifies target name (defaults to source name)
  -sha=0                 Generate manifest using SHA 1, 256, 512 or 0 to skip
  -vm=                   Virtual machine [GOVC_VM]
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
  -update=false          Update extension
```

## extension.setcert

```
Usage: govc extension.setcert [OPTIONS] ID

Set certificate for the extension ID.

The '-cert-pem' option can be one of the following:
'-' : Read the certificate from stdin
'+' : Generate a new key pair and save locally to ID.crt and ID.key
... : Any other value is passed as-is to ExtensionManager.SetCertificate

Examples:
  govc extension.setcert -cert-pem + -org Example com.example.extname

Options:
  -cert-pem=-            PEM encoded certificate
  -org=VMware            Organization for generated certificate
```

## extension.unregister

```
Usage: govc extension.unregister [OPTIONS]

Options:
```

## fields.add

```
Usage: govc fields.add [OPTIONS] NAME

Add a custom field type with NAME.

Examples:
  govc fields.add my-field-name # adds a field to all managed object types
  govc fields.add -type VirtualMachine my-vm-field-name # adds a field to the VirtualMachine type

Options:
  -type=                 Managed object type
```

## fields.info

```
Usage: govc fields.info [OPTIONS] PATH...

Display custom field values for PATH.

Also known as "Custom Attributes".

Examples:
  govc fields.info vm/*
  govc fields.info -n my-field-name vm/*

Options:
  -n=                    Filter by custom field name
```

## fields.ls

```
Usage: govc fields.ls [OPTIONS]

List custom field definitions.

Examples:
  govc fields.ls

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

Set custom field values for PATH.

Examples:
  govc fields.set my-field-name field-value vm/my-vm
  govc fields.set -add my-new-global-field-name field-value vm/my-vm
  govc fields.set -add -type VirtualMachine my-new-vm-field-name field-value vm/my-vm

Options:
  -add=false             Adds the field if it does not exist. Use the -type flag to specify the managed object type to which the field is added. Using -add and omitting -kind causes a new, global field to be created if a field with the provided name does not already exist.
  -type=                 Managed object type on which to add the field if it does not exist. This flag is ignored unless -add=true
```

## find

```
Usage: govc find [OPTIONS] [ROOT] [KEY VAL]...

Find managed objects.

ROOT can be an inventory path or ManagedObjectReference.
ROOT defaults to '.', an alias for the root folder or DC if set.

Optional KEY VAL pairs can be used to filter results against object instance properties.
Use the govc 'object.collect' command to view possible object property keys.

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
  -i=false               Print the managed object reference
  -maxdepth=-1           Max depth
  -name=*                Resource name
  -type=[]               Resource type
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
  -c=true                Check if esx firewall is enabled
  -direction=outbound    Direction
  -enabled=true          Find enabled rule sets if true, disabled if false
  -host=                 Host system [GOVC_HOST]
  -port=0                Port
  -proto=tcp             Protocol
  -type=dst              Port type
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
  -pod=false             Create folder(s) of type StoragePod (DatastoreCluster)
```

## folder.info

```
Usage: govc folder.info [OPTIONS] [PATH]...

Options:
```

## guest.chmod

```
Usage: govc guest.chmod [OPTIONS] MODE FILE

Change FILE MODE on VM.

Examples:
  govc guest.chmod -vm $name 0644 /var/log/foo.log

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.chown

```
Usage: govc guest.chown [OPTIONS] UID[:GID] FILE

Change FILE UID and GID on VM.

Examples:
  govc guest.chown -vm $name UID[:GID] /var/log/foo.log

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
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
  -f=false               If set, the local destination file is clobbered
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.getenv

```
Usage: govc guest.getenv [OPTIONS] [NAME]...

Read NAME environment variables from VM.

Examples:
  govc guest.getenv -vm $name
  govc guest.getenv -vm $name HOME

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.kill

```
Usage: govc guest.kill [OPTIONS]

Kill process ID on VM.

Examples:
  govc guest.kill -vm $name -p 12345

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=[]                  Process ID
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.ls

```
Usage: govc guest.ls [OPTIONS] PATH

List PATH files in VM.

Examples:
  govc guest.ls -vm $name /tmp

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -s=false               Simple path only listing
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.mkdir

```
Usage: govc guest.mkdir [OPTIONS] PATH

Create directory PATH in VM.

Examples:
  govc guest.mkdir -vm $name /tmp/logs
  govc guest.mkdir -vm $name -p /tmp/logs/foo/bar

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=false               Create intermediate directories as needed
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.mktemp

```
Usage: govc guest.mktemp [OPTIONS]

Create a temporary file or directory in VM.

Examples:
  govc guest.mktemp -vm $name
  govc guest.mktemp -vm $name -d
  govc guest.mktemp -vm $name -t myprefix
  govc guest.mktemp -vm $name -p /var/tmp/$USER

Options:
  -d=false               Make a directory instead of a file
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=                    If specified, create relative to this directory
  -s=                    Suffix
  -t=                    Prefix
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.mv

```
Usage: govc guest.mv [OPTIONS] SOURCE DEST

Move (rename) files in VM.

Examples:
  govc guest.mv -vm $name /tmp/foo.sh /tmp/bar.sh
  govc guest.mv -vm $name -n /tmp/baz.sh /tmp/bar.sh

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -n=false               Do not overwrite an existing file
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.ps

```
Usage: govc guest.ps [OPTIONS]

List processes in VM.

By default, unless the '-e', '-p' or '-U' flag is specified, only processes owned
by the '-l' flag user are displayed.

The '-x' and '-X' flags only apply to processes started by vmware-tools,
such as those started with the govc guest.start command.

Examples:
  govc guest.ps -vm $name
  govc guest.ps -vm $name -e
  govc guest.ps -vm $name -p 12345
  govc guest.ps -vm $name -U root

Options:
  -U=                    Select by process UID
  -X=false               Wait for process to exit
  -e=false               Select all processes
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -p=[]                  Select by process ID
  -vm=                   Virtual machine [GOVC_VM]
  -x=false               Output exit time and code
```

## guest.rm

```
Usage: govc guest.rm [OPTIONS] PATH

Remove file PATH in VM.

Examples:
  govc guest.rm -vm $name /tmp/foo.log

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.rmdir

```
Usage: govc guest.rmdir [OPTIONS] PATH

Remove directory PATH in VM.

Examples:
  govc guest.rmdir -vm $name /tmp/empty-dir
  govc guest.rmdir -vm $name -r /tmp/non-empty-dir

Options:
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -r=false               Recursive removal
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.run

```
Usage: govc guest.run [OPTIONS] NAME [ARG]...

Run program NAME in VM and display output.

This command depends on govmomi/toolbox running in the VM guest and does not work with standard VMware tools.

If the program NAME is an HTTP verb, the toolbox's http.RoundTripper will be used as the HTTP transport.

Examples:
  govc guest.run -vm $name kubectl get pods
  govc guest.run -vm $name -d - kubectl create -f - <svc.json
  govc guest.run -vm $name kubectl delete pod,service my-service
  govc guest.run -vm $name GET http://localhost:8080/api/v1/nodes
  govc guest.run -vm $name -e Content-Type:application/json -d - POST http://localhost:8080/api/v1/namespaces/default/pods <svc.json
  govc guest.run -vm $name DELETE http://localhost:8080/api/v1/namespaces/default/services/my-service

Options:
  -C=                    The absolute path of the working directory for the program to start
  -d=                    Input data
  -e=[]                  Set environment variable or HTTP header
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -v=false               Verbose
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.start

```
Usage: govc guest.start [OPTIONS] PATH [ARG]...

Start program in VM.

The process can have its status queried with govc guest.ps.
When the process completes, its exit code and end time will be available for 5 minutes after completion.

Examples:
  govc guest.start -vm $name /bin/mount /dev/hdb1 /data
  pid=$(govc guest.start -vm $name /bin/long-running-thing)
  govc guest.ps -vm $name -p $pid -X

Options:
  -C=                    The absolute path of the working directory for the program to start
  -e=[]                  Set environment variable (key=val)
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
```

## guest.touch

```
Usage: govc guest.touch [OPTIONS] FILE

Change FILE times on VM.

Examples:
  govc guest.touch -vm $name /var/log/foo.log
  govc guest.touch -vm $name -d "$(date -d '1 day ago')" /var/log/foo.log

Options:
  -a=false               Change only the access time
  -c=false               Do not create any files
  -d=                    Use DATE instead of current time
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -vm=                   Virtual machine [GOVC_VM]
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
  -f=false               If set, the guest destination file is clobbered
  -gid=<nil>             Group ID
  -l=:                   Guest VM credentials [GOVC_GUEST_LOGIN]
  -perm=0                File permissions
  -uid=<nil>             User ID
  -vm=                   Virtual machine [GOVC_VM]
```

## host.account.create

```
Usage: govc host.account.create [OPTIONS]

Create local account on HOST.

Examples:
  govc host.account.create -id $USER -password password-for-esx60

Options:
  -description=          The description of the specified account
  -host=                 Host system [GOVC_HOST]
  -id=                   The ID of the specified account
  -password=             The password for the specified account id
```

## host.account.remove

```
Usage: govc host.account.remove [OPTIONS]

Remove local account on HOST.

Examples:
  govc host.account.remove -id $USER

Options:
  -description=          The description of the specified account
  -host=                 Host system [GOVC_HOST]
  -id=                   The ID of the specified account
  -password=             The password for the specified account id
```

## host.account.update

```
Usage: govc host.account.update [OPTIONS]

Update local account on HOST.

Examples:
  govc host.account.update -id root -password password-for-esx60

Options:
  -description=          The description of the specified account
  -host=                 Host system [GOVC_HOST]
  -id=                   The ID of the specified account
  -password=             The password for the specified account id
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
  -connect=true          Immediately connect to host
  -folder=               Inventory folder [GOVC_FOLDER]
  -force=false           Force when host is managed by another VC
  -hostname=             Hostname or IP address of the host
  -noverify=false        Accept host thumbprint without verification
  -password=             Password of administration account on the host
  -thumbprint=           SHA-1 thumbprint of the host's SSL certificate
  -username=             Username of administration account on the host
```

## host.autostart.add

```
Usage: govc host.autostart.add [OPTIONS] VM...

Options:
  -host=                      Host system [GOVC_HOST]
  -start-action=powerOn       Start Action
  -start-delay=-1             Start Delay
  -start-order=-1             Start Order
  -stop-action=systemDefault  Stop Action
  -stop-delay=-1              Stop Delay
  -wait=systemDefault         Wait for Hearbeat Setting (systemDefault|yes|no)
```

## host.autostart.configure

```
Usage: govc host.autostart.configure [OPTIONS]

Options:
  -enabled=<nil>             Enable autostart
  -host=                     Host system [GOVC_HOST]
  -start-delay=0             Start delay
  -stop-action=              Stop action
  -stop-delay=0              Stop delay
  -wait-for-heartbeat=<nil>  Wait for hearbeat
```

## host.autostart.info

```
Usage: govc host.autostart.info [OPTIONS]

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.autostart.remove

```
Usage: govc host.autostart.remove [OPTIONS] VM...

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.cert.csr

```
Usage: govc host.cert.csr [OPTIONS]

Generate a certificate-signing request (CSR) for HOST.

Options:
  -host=                 Host system [GOVC_HOST]
  -ip=false              Use IP address as CN
```

## host.cert.import

```
Usage: govc host.cert.import [OPTIONS] FILE

Install SSL certificate FILE on HOST.

If FILE name is "-", read certificate from stdin.

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.cert.info

```
Usage: govc host.cert.info [OPTIONS]

Display SSL certificate info for HOST.

Options:
  -host=                 Host system [GOVC_HOST]
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
  -date=                 Update the date/time on the host
  -host=                 Host system [GOVC_HOST]
  -server=               IP or FQDN for NTP server(s)
  -tz=                   Change timezone of the host
```

## host.date.info

```
Usage: govc host.date.info [OPTIONS]

Display date and time info for HOST.

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.disconnect

```
Usage: govc host.disconnect [OPTIONS]

Disconnect HOST from vCenter.

Options:
  -host=                 Host system [GOVC_HOST]
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
  -hints=true            Use command info hints when formatting output
  -host=                 Host system [GOVC_HOST]
```

## host.info

```
Usage: govc host.info [OPTIONS]

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.maintenance.enter

```
Usage: govc host.maintenance.enter [OPTIONS] HOST...

Put HOST in maintenance mode.

While this task is running and when the host is in maintenance mode,
no VMs can be powered on and no provisioning operations can be performed on the host.

Options:
  -evacuate=false        Evacuate powered off VMs
  -host=                 Host system [GOVC_HOST]
  -timeout=0             Timeout
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
  -host=                 Host system [GOVC_HOST]
  -timeout=0             Timeout
```

## host.option.ls

```
Usage: govc host.option.ls [OPTIONS] [NAME]

List option with the given NAME.

If NAME ends with a dot, all options for that subtree are listed.

Examples:
  govc host.option.ls
  govc host.option.ls Config.HostAgent.
  govc host.option.ls Config.HostAgent.plugins.solo.enableMob

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.option.set

```
Usage: govc host.option.set [OPTIONS] NAME VALUE

Set option NAME to VALUE.

Examples:
  govc host.option.set Config.HostAgent.plugins.solo.enableMob true
  govc host.option.set Config.HostAgent.log.level verbose

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.portgroup.add

```
Usage: govc host.portgroup.add [OPTIONS] NAME

Add portgroup to HOST.

Examples:
  govc host.portgroup.add -vswitch vSwitch0 -vlan 3201 bridge

Options:
  -host=                 Host system [GOVC_HOST]
  -vlan=0                VLAN ID
  -vswitch=              vSwitch Name
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
  -host=                 Host system [GOVC_HOST]
```

## host.portgroup.remove

```
Usage: govc host.portgroup.remove [OPTIONS] NAME

Remove portgroup from HOST.

Examples:
  govc host.portgroup.remove bridge

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.reconnect

```
Usage: govc host.reconnect [OPTIONS]

Reconnect HOST to vCenter.

This command can also be used to change connection properties (hostname, fingerprint, username, password),
without disconnecting the host.

Options:
  -force=false           Force when host is managed by another VC
  -host=                 Host system [GOVC_HOST]
  -hostname=             Hostname or IP address of the host
  -noverify=false        Accept host thumbprint without verification
  -password=             Password of administration account on the host
  -sync-state=false      Sync state
  -thumbprint=           SHA-1 thumbprint of the host's SSL certificate
  -username=             Username of administration account on the host
```

## host.remove

```
Usage: govc host.remove [OPTIONS] HOST...

Remove HOST from vCenter.

Options:
  -host=                 Host system [GOVC_HOST]
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
  -host=                 Host system [GOVC_HOST]
```

## host.service.ls

```
Usage: govc host.service.ls [OPTIONS]

List HOST services.

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.shutdown

```
Usage: govc host.shutdown [OPTIONS] HOST...

Shutdown HOST.

Options:
  -f=false               Force shutdown when host is not in maintenance mode
  -host=                 Host system [GOVC_HOST]
  -r=false               Reboot host
```

## host.storage.info

```
Usage: govc host.storage.info [OPTIONS]

Show HOST storage system information.

Examples:
  govc find / -type h | xargs -n1 govc host.storage.info -unclaimed -host

Options:
  -host=                 Host system [GOVC_HOST]
  -refresh=false         Refresh the storage system provider
  -rescan=false          Rescan all host bus adapters
  -rescan-vmfs=false     Rescan for new VMFSs
  -t=lun                 Type (hba,lun)
  -unclaimed=false       Only show disks that can be used as new VMFS datastores
```

## host.storage.mark

```
Usage: govc host.storage.mark [OPTIONS] DEVICE_PATH

Mark device at DEVICE_PATH.

Options:
  -host=                 Host system [GOVC_HOST]
  -local=<nil>           Mark as local
  -ssd=<nil>             Mark as SSD
```

## host.storage.partition

```
Usage: govc host.storage.partition [OPTIONS] DEVICE_PATH

Show partition table for device at DEVICE_PATH.

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.vnic.info

```
Usage: govc host.vnic.info [OPTIONS]

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.vnic.service

```
Usage: govc host.vnic.service [OPTIONS] SERVICE DEVICE


Enable or disable service on a virtual nic device.

Where SERVICE is one of: vmotion|faultToleranceLogging|vSphereReplication|vSphereReplicationNFC|management|vsan|vSphereProvisioning
Where DEVICE is one of: vmk0|vmk1|...

Examples:
  govc host.vnic.service -host hostname -enable vsan vmk0
  govc host.vnic.service -host hostname -enable=false vmotion vmk1

Options:
  -enable=true           Enable service
  -host=                 Host system [GOVC_HOST]
```

## host.vswitch.add

```
Usage: govc host.vswitch.add [OPTIONS] NAME

Options:
  -host=                 Host system [GOVC_HOST]
  -mtu=0                 MTU
  -nic=                  Bridge nic device
  -ports=128             Number of ports
```

## host.vswitch.info

```
Usage: govc host.vswitch.info [OPTIONS]

Options:
  -host=                 Host system [GOVC_HOST]
```

## host.vswitch.remove

```
Usage: govc host.vswitch.remove [OPTIONS] NAME

Options:
  -host=                 Host system [GOVC_HOST]
```

## import.ova

```
Usage: govc import.ova [OPTIONS] PATH_TO_OVA

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -folder=               Inventory folder [GOVC_FOLDER]
  -host=                 Host system [GOVC_HOST]
  -name=                 Name to use for new entity
  -options=              Options spec file path for VM deployment
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
```

## import.ovf

```
Usage: govc import.ovf [OPTIONS] PATH_TO_OVF

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -folder=               Inventory folder [GOVC_FOLDER]
  -host=                 Host system [GOVC_HOST]
  -name=                 Name to use for new entity
  -options=              Options spec file path for VM deployment
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
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
  -ds=                   Datastore [GOVC_DATASTORE]
  -folder=               Inventory folder [GOVC_FOLDER]
  -force=false           Overwrite existing disk
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
```

## license.add

```
Usage: govc license.add [OPTIONS] KEY...

Options:
```

## license.assign

```
Usage: govc license.assign [OPTIONS] KEY

Assign licenses to HOST or CLUSTER.

Examples:
  govc license.assign $VCSA_LICENSE_KEY
  govc license.assign -host a_host.example.com $ESX_LICENSE_KEY
  govc license.assign -cluster a_cluster $VSAN_LICENSE_KEY

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -host=                 Host system [GOVC_HOST]
  -name=                 Display name
  -remove=false          Remove assignment
```

## license.assigned.ls

```
Usage: govc license.assigned.ls [OPTIONS]

Options:
  -id=                   Entity ID
```

## license.decode

```
Usage: govc license.decode [OPTIONS] KEY...

Options:
  -feature=              List licenses with given feature
```

## license.label.set

```
Usage: govc license.label.set [OPTIONS] LICENSE KEY VAL

Set license labels.

Examples:
  govc license.label.set 00000-00000-00000-00000-00000 team cnx # add/set label
  govc license.label.set 00000-00000-00000-00000-00000 team ""  # remove label
  govc license.ls -json | jq '.[] | select(.Labels[].Key == "team") | .LicenseKey'

Options:
```

## license.ls

```
Usage: govc license.ls [OPTIONS]

Options:
  -feature=              List licenses with given feature
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
  -f=false               Follow log file changes
  -host=                 Host system [GOVC_HOST]
  -log=                  Log file key
  -n=25                  Output the last N log lines
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
  -default=true          Specifies if the bundle should include the default server
```

## logs.ls

```
Usage: govc logs.ls [OPTIONS]

List diagnostic log keys.

Examples:
  govc logs.ls
  govc logs.ls -host host-a

Options:
  -host=                 Host system [GOVC_HOST]
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
  -L=false               Follow managed object references
  -i=false               Print the managed object reference
  -l=false               Long listing format
  -t=                    Object type
```

## metric.change

```
Usage: govc metric.change [OPTIONS] NAME...

Change counter NAME levels.

Examples:
  govc metric.change -level 1 net.bytesRx.average net.bytesTx.average

Options:
  -device-level=0        Level for the per device counter
  -i=0                   Interval ID
  -level=0               Level for the aggregate counter
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
  -i=0                   Interval ID
```

## metric.interval.change

```
Usage: govc metric.interval.change [OPTIONS]

Change historical metric intervals.

Examples:
  govc metric.interval.change -i 300 -level 2
  govc metric.interval.change -i 86400 -enabled=false

Options:
  -enabled=<nil>         Enable or disable
  -i=0                   Interval ID
  -level=0               Level
```

## metric.interval.info

```
Usage: govc metric.interval.info [OPTIONS]

List historical metric intervals.

Examples:
  govc metric.interval.info
  govc metric.interval.info -i 300

Options:
  -i=0                   Interval ID
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
  -i=0                   Interval ID
  -l=false               Long listing format
```

## metric.reset

```
Usage: govc metric.reset [OPTIONS] NAME...

Reset counter NAME to the default level of data collection.

Examples:
  govc metric.reset net.bytesRx.average net.bytesTx.average

Options:
  -i=0                   Interval ID
```

## metric.sample

```
Usage: govc metric.sample [OPTIONS] PATH... NAME...

Sample for object PATH of metric NAME.

Interval ID defaults to 20 (realtime) if supported, otherwise 300 (5m interval).

By default, INSTANCE '*' samples all instances and the aggregate counter.
An INSTANCE value of '-' will only sample the aggregate counter.
An INSTANCE value other than '*' or '-' will only sample the given instance counter.

If PLOT value is set to '-', output a gnuplot script.  If non-empty with another
value, PLOT will pipe the script to gnuplot for you.  The value is also used to set
the gnuplot 'terminal' variable, unless the value is that of the DISPLAY env var.
Only 1 metric NAME can be specified when the PLOT flag is set.

Examples:
  govc metric.sample host/cluster1/* cpu.usage.average
  govc metric.sample -plot .png host/cluster1/* cpu.usage.average | xargs open
  govc metric.sample vm/* net.bytesTx.average net.bytesTx.average
  govc metric.sample -instance vmnic0 vm/* net.bytesTx.average
  govc metric.sample -instance - vm/* net.bytesTx.average

Options:
  -d=30                  Limit object display name to D chars
  -i=0                   Interval ID
  -instance=*            Instance
  -n=6                   Max number of samples
  -plot=                 Plot data using gnuplot
  -t=false               Include sample times
```

## object.collect

```
Usage: govc object.collect [OPTIONS] [MOID] [PROPERTY]...

Collect managed object properties.

MOID can be an inventory path or ManagedObjectReference.
MOID defaults to '-', an alias for 'ServiceInstance:ServiceInstance' or the root folder if a '-type' flag is given.

If a '-type' flag is given, properties are collected using a ContainerView object where MOID is the root of the view.

By default only the current property value(s) are collected.  To wait for updates, use the '-n' flag or
specify a property filter.  A property filter can be specified by prefixing the property name with a '-',
followed by the value to match.

The '-R' flag sets the Filter using the given XML encoded request, which can be captured by 'vcsim -trace' for example.
It can be useful for replaying property filters created by other clients and converting filters to Go code via '-O -dump'.

Examples:
  govc object.collect - content
  govc object.collect -s HostSystem:ha-host hardware.systemInfo.uuid
  govc object.collect -s /ha-datacenter/vm/foo overallStatus
  govc object.collect -s /ha-datacenter/vm/foo -guest.guestOperationsReady true # property filter
  govc object.collect -type m / name runtime.powerState # collect properties for multiple objects
  govc object.collect -json -n=-1 EventManager:ha-eventmgr latestEvent | jq .
  govc object.collect -json -s $(govc object.collect -s - content.perfManager) description.counterType | jq .
  govc object.collect -R create-filter-request.xml # replay filter
  govc object.collect -R create-filter-request.xml -O # convert filter to Go code
  govc object.collect -s vm/my-vm summary.runtime.host | xargs govc ls -L # inventory path of VM's host
  govc object.collect -json -type m / config.hardware.device | \ # use -json + jq to search array elements
    jq -r '. | select(.ChangeSet[].Val.VirtualDevice[].MacAddress == "00:50:56:bc:5e:3c") | \
    [.Obj.Type, .Obj.Value] | join(":")' | xargs govc ls -L

Options:
  -O=false               Output the CreateFilter request itself
  -R=                    Raw XML encoded CreateFilter request
  -n=0                   Wait for N property updates
  -s=false               Output property value only
  -type=[]               Resource type.  If specified, MOID is used for a container view root
  -wait=0s               Max wait time for updates
```

## object.destroy

```
Usage: govc object.destroy [OPTIONS] PATH...

Destroy managed objects.

Examples:
  govc object.destroy /dc1/network/dvs /dc1/host/cluster

Options:
```

## object.method

```
Usage: govc object.method [OPTIONS] PATH...

Enable or disable methods for managed objects.

Examples:
  govc object.method -name Destroy_Task -enable=false /dc1/vm/foo
  govc object.collect /dc1/vm/foo disabledMethod | grep --color Destroy_Task
  govc object.method -name Destroy_Task -enable /dc1/vm/foo

Options:
  -enable=true           Enable method
  -name=                 Method name
  -reason=               Reason for disabling method
  -source=govc           Source ID
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

## option.ls

```
Usage: govc option.ls [OPTIONS] [NAME]

List option with the given NAME.

If NAME ends with a dot, all options for that subtree are listed.

Examples:
  govc option.ls
  govc option.ls config.vpxd.sso.
  govc option.ls config.vpxd.sso.sts.uri

Options:
```

## option.set

```
Usage: govc option.set [OPTIONS] NAME VALUE

Set option NAME to VALUE.

Examples:
  govc option.set log.level info
  govc option.set logger.Vsan verbose

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
  -a=true                Include inherited permissions defined by parent entities
  -i=false               Use moref instead of inventory path
```

## permissions.remove

```
Usage: govc permissions.remove [OPTIONS] [PATH]...

Removes a permission rule from managed entities.

Examples:
  govc permissions.remove -principal root
  govc permissions.remove -principal $USER@vsphere.local /dc1/host/cluster1

Options:
  -group=false           True, if principal refers to a group name; false, for a user name
  -i=false               Use moref instead of inventory path
  -principal=            User or group for which the permission is defined
```

## permissions.set

```
Usage: govc permissions.set [OPTIONS] [PATH]...

Set the permissions managed entities.

Examples:
  govc permissions.set -principal root -role Admin
  govc permissions.set -principal $USER@vsphere.local -role Admin /dc1/host/cluster1

Options:
  -group=false           True, if principal refers to a group name; false, for a user name
  -i=false               Use moref instead of inventory path
  -principal=            User or group for which the permission is defined
  -propagate=true        Whether or not this permission propagates down the hierarchy to sub-entities
  -role=Admin            Permission role name
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
  -cpu.expandable=<nil>   CPU expandable reservation
  -cpu.limit=<nil>        CPU limit in MHz
  -cpu.reservation=<nil>  CPU reservation in MHz
  -cpu.shares=            CPU shares level or number
  -mem.expandable=<nil>   Memory expandable reservation
  -mem.limit=<nil>        Memory limit in MB
  -mem.reservation=<nil>  Memory reservation in MB
  -mem.shares=            Memory shares level or number
  -name=                  Resource pool name
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
  -cpu.expandable=true   CPU expandable reservation
  -cpu.limit=-1          CPU limit in MHz
  -cpu.reservation=0     CPU reservation in MHz
  -cpu.shares=normal     CPU shares level or number
  -mem.expandable=true   Memory expandable reservation
  -mem.limit=-1          Memory limit in MB
  -mem.reservation=0     Memory reservation in MB
  -mem.shares=normal     Memory shares level or number
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
  -children=false        Remove all children pools
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
  -a=false               List virtual app resource pools
  -p=true                List resource pools
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
  -i=false               Use moref instead of inventory path
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
  -i=false               Use moref instead of inventory path
```

## role.remove

```
Usage: govc role.remove [OPTIONS] NAME

Remove authorization role.

Examples:
  govc role.remove MyRole
  govc role.remove MyRole -force

Options:
  -force=false           Force removal if role is in use
  -i=false               Use moref instead of inventory path
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
  -a=false               Add given PRIVILEGE(s)
  -i=false               Use moref instead of inventory path
  -name=                 Change role name
  -r=false               Remove given PRIVILEGE(s)
```

## role.usage

```
Usage: govc role.usage [OPTIONS] NAME...

List usage for role NAME.

Examples:
  govc role.usage
  govc role.usage Admin

Options:
  -i=false               Use moref instead of inventory path
```

## session.login

```
Usage: govc session.login [OPTIONS]

Session login.

The session.login command is optional, all other govc commands will auto login when given credentials.
The session.login command can be used to:
- Persist a session without writing to disk via the '-cookie' flag
- Acquire a clone ticket
- Login using a clone ticket
- Login using a vCenter Extension certificate
- Issue a SAML token
- Renew a SAML token
- Login using a SAML token
- Avoid passing credentials to other govc commands

Examples:
  govc session.login -u root:password@host
  ticket=$(govc session.login -u root@host -clone)
  govc session.login -u root@host -ticket $ticket
  govc session.login -u host -extension com.vmware.vsan.health -cert rui.crt -key rui.key
  token=$(govc session.login -u host -cert user.crt -key user.key -issue) # HoK token
  bearer=$(govc session.login -u user:pass@host -issue) # Bearer token
  token=$(govc session.login -u host -cert user.crt -key user.key -issue -token "$bearer")
  govc session.login -u host -cert user.crt -key user.key -token "$token"
  token=$(govc session.login -u host -cert user.crt -key user.key -renew -lifetime 24h -token "$token")

Options:
  -clone=false           Acquire clone ticket
  -cookie=               Set HTTP cookie for an existing session
  -extension=            Extension name
  -issue=false           Issue SAML token
  -l=false               Output session cookie
  -lifetime=10m0s        SAML token lifetime
  -renew=false           Renew SAML token
  -ticket=               Use clone ticket for login
  -token=                Use SAML token for login or as issue identity
```

## session.logout

```
Usage: govc session.logout [OPTIONS]

Logout the current session.

The session.logout command can be used to end the current persisted session.
The session.rm command can be used to remove sessions other than the current session.

Examples:
  govc session.logout

Options:
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
  -d=                    Snapshot description
  -m=true                Include memory state
  -q=false               Quiesce guest file system
  -vm=                   Virtual machine [GOVC_VM]
```

## snapshot.remove

```
Usage: govc snapshot.remove [OPTIONS] NAME

Remove snapshot of VM with given NAME.

NAME can be the snapshot name, tree path, moid or '*' to remove all snapshots.

Examples:
  govc snapshot.remove -vm my-vm happy-vm-state

Options:
  -c=true                Consolidate disks
  -r=false               Remove snapshot children
  -vm=                   Virtual machine [GOVC_VM]
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
  -s=false               Suppress power on
  -vm=                   Virtual machine [GOVC_VM]
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
  -C=false               Print the current snapshot name only
  -D=false               Print the snapshot creation date
  -c=true                Print the current snapshot
  -f=false               Print the full path prefix for snapshot
  -i=false               Print the snapshot id
  -vm=                   Virtual machine [GOVC_VM]
```

## sso.service.ls

```
Usage: govc sso.service.ls [OPTIONS]

List platform services.

Examples:
  govc sso.service.ls
  govc sso.service.ls -t vcenterserver -P vmomi
  govc sso.service.ls -t sso:sts
  govc sso.service.ls -t sso:sts -U
  govc sso.service.ls -t sso:sts -json | jq -r .[].ServiceEndpoints[].Url

Options:
  -P=                    Endpoint protocol
  -T=                    Endpoint type
  -U=false               List endpoint URL(s) only
  -l=false               Long listing format
  -n=                    Node ID
  -p=                    Service product
  -s=                    Site ID
  -t=                    Service type
```

## sso.user.create

```
Usage: govc sso.user.create [OPTIONS] NAME

Create SSO users.

Examples:
  govc sso.user.create -C "$(cat cert.pem)" -A -R Administrator NAME # solution user
  govc sso.user.create -p password NAME # person user

Options:
  -A=<nil>               ActAsUser role for solution user WSTrust
  -C=                    Certificate for solution user
  -R=                    Role for solution user (RegularUser|Administrator)
  -d=                    User description
  -f=                    First name
  -l=                    Last name
  -m=                    Email address
  -p=                    Password
```

## sso.user.id

```
Usage: govc sso.user.id [OPTIONS] NAME

Print SSO user and group IDs.

Examples:
  govc sso.user.id
  govc sso.user.id Administrator
  govc sso.user.id -json Administrator

Options:
```

## sso.user.ls

```
Usage: govc sso.user.ls [OPTIONS]

List SSO users.

Examples:
  govc sso.user.ls -s

Options:
  -s=false               List solution users
```

## sso.user.rm

```
Usage: govc sso.user.rm [OPTIONS] NAME

Remove SSO users.

Examples:
  govc sso.user.rm NAME

Options:
```

## sso.user.update

```
Usage: govc sso.user.update [OPTIONS] NAME

Update SSO users.

Examples:
  govc sso.user.update -C "$(cat cert.pem)" NAME
  govc sso.user.update -p password NAME

Options:
  -A=<nil>               ActAsUser role for solution user WSTrust
  -C=                    Certificate for solution user
  -R=                    Role for solution user (RegularUser|Administrator)
  -d=                    User description
  -f=                    First name
  -l=                    Last name
  -m=                    Email address
  -p=                    Password
```

## tags.attach

```
Usage: govc tags.attach [OPTIONS] NAME PATH

Attach tag NAME to object PATH.

Examples:
  govc tags.attach k8s-region-us /dc1
  govc tags.attach -c k8s-region us-ca1 /dc1/host/cluster1

Options:
  -c=                    Tag category
```

## tags.attached.ls

```
Usage: govc tags.attached.ls [OPTIONS] NAME

List attached tags or objects.

Examples:
  govc tags.attached.ls k8s-region-us
  govc tags.attached.ls -json k8s-zone-us-ca1 | jq .
  govc tags.attached.ls -r /dc1/host/cluster1
  govc tags.attached.ls -json -r /dc1 | jq .

Options:
  -r=false               List tags attached to resource
```

## tags.category.create

```
Usage: govc tags.category.create [OPTIONS] NAME

Create tag category.

This command will output the ID of the new tag category.

Examples:
  govc tags.category.create -d "Kubernetes region" -t Datacenter k8s-region
  govc tags.category.create -d "Kubernetes zone" k8s-zone

Options:
  -d=                    Description
  -m=false               Allow multiple tags per object
  -t=[]                  Object types
```

## tags.category.info

```
Usage: govc tags.category.info [OPTIONS] [NAME]

Display category info.

If NAME is provided, display info for only that category.
Otherwise display info for all categories.

Examples:
  govc tags.category.info
  govc tags.category.info k8s-zone

Options:
```

## tags.category.ls

```
Usage: govc tags.category.ls [OPTIONS]

List all categories.

Examples:
  govc tags.category.ls
  govc tags.category.ls -json | jq .

Options:
```

## tags.category.rm

```
Usage: govc tags.category.rm [OPTIONS] NAME

Delete category NAME.

Fails if category is used by any tag, unless the '-f' flag is provided.

Examples:
  govc tags.category.rm k8s-region
  govc tags.category.rm -f k8s-zone

Options:
  -f=false               Delete tag regardless of attached objects
```

## tags.category.update

```
Usage: govc tags.category.update [OPTIONS] NAME

Update category.

The '-t' flag can only be used to add new object types.  Removing category types is not supported by vCenter.

Examples:
  govc tags.category.update -n k8s-vcp-region -d "Kubernetes VCP region" k8s-region
  govc tags.category.update -t ClusterComputeResource k8s-zone

Options:
  -d=                    Description
  -m=<nil>               Allow multiple tags per object
  -n=                    Name of category
  -t=[]                  Object types
```

## tags.create

```
Usage: govc tags.create [OPTIONS] NAME

Create tag.

The '-c' option to specify a tag category is required.
This command will output the ID of the new tag.

Examples:
  govc tags.create -d "Kubernetes Zone US CA1" -c k8s-zone k8s-zone-us-ca1

Options:
  -c=                    Category name
  -d=                    Description of tag
```

## tags.detach

```
Usage: govc tags.detach [OPTIONS] NAME PATH

Detach tag NAME from object PATH.

Examples:
  govc tags.detach k8s-region-us /dc1
  govc tags.detach -c k8s-region us-ca1 /dc1/host/cluster1

Options:
  -c=                    Tag category
```

## tags.info

```
Usage: govc tags.info [OPTIONS] NAME

Display tags info.

If NAME is provided, display info for only that tag.  Otherwise display info for all tags.

Examples:
  govc tags.info
  govc tags.info k8s-zone-us-ca1
  govc tags.info -c k8s-zone

Options:
  -C=true                Display category name instead of ID
  -c=                    Category name
```

## tags.ls

```
Usage: govc tags.ls [OPTIONS]

List tags.

Examples:
  govc tags.ls
  govc tags.ls -c k8s-zone
  govc tags.ls -json | jq .
  govc tags.ls -c k8s-region -json | jq .

Options:
  -c=                    Category name
```

## tags.rm

```
Usage: govc tags.rm [OPTIONS] NAME

Delete tag NAME.

Fails if tag is attached to any object, unless the '-f' flag is provided.

Examples:
  govc tags.rm k8s-zone-us-ca1
  govc tags.rm -f -c k8s-zone us-ca2

Options:
  -c=                    Tag category
  -f=false               Delete tag regardless of attached objects
```

## tags.update

```
Usage: govc tags.update [OPTIONS] NAME

Update tag.

Examples:
  govc tags.update -d "K8s zone US-CA1" k8s-zone-us-ca1
  govc tags.update -d "K8s zone US-CA1" -c k8s-zone us-ca1

Options:
  -c=                    Tag category
  -d=                    Description of tag
  -n=                    Name of tag
```

## task.cancel

```
Usage: govc task.cancel [OPTIONS] ID...

Cancel tasks.

Examples:
  govc task.cancel task-759

Options:
```

## tasks

```
Usage: govc tasks [OPTIONS] [PATH]

Display info for recent tasks.

When a task has completed, the result column includes the task duration on success or
error message on failure.  If a task is still in progress, the result column displays
the completion percentage and the task ID.  The task ID can be used as an argument to
the 'task.cancel' command.

By default, all recent tasks are included (via TaskManager), but can be limited by PATH
to a specific inventory object.

Examples:
  govc tasks
  govc tasks -f
  govc tasks -f /dc1/host/cluster1

Options:
  -f=false               Follow recent task updates
  -l=false               Use long task description
  -n=25                  Output the last N tasks
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
  -force=false           Force (If force is false, the shutdown order in the vApp is executed. If force is true, all virtual machines are powered-off (regardless of shutdown order))
  -off=false             Power off
  -on=false              Power on
  -suspend=false         Power suspend
  -vapp.ipath=           Find vapp by inventory path
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

To add ExtraConfig variables that can read within the guest, use the 'guestinfo.' prefix.

Examples:
  govc vm.change -vm $vm -mem.reservation 2048
  govc vm.change -vm $vm -e smc.present=TRUE -e ich7m.present=TRUE
  # Enable both cpu and memory hotplug on a guest:
  govc vm.change -vm $vm -e vcpu.hotadd=true -e mem.hotadd=true
  govc vm.change -vm $vm -e guestinfo.vmname $vm
  # Read the variable set above inside the guest:
  vmware-rpctool "info-get guestinfo.vmname"

Options:
  -annotation=                VM description
  -c=0                        Number of CPUs
  -cpu.limit=<nil>            CPU limit in MHz
  -cpu.reservation=<nil>      CPU reservation in MHz
  -cpu.shares=                CPU shares level or number
  -e=[]                       ExtraConfig. <key>=<value>
  -g=                         Guest OS
  -m=0                        Size in MB of memory
  -mem.limit=<nil>            Memory limit in MB
  -mem.reservation=<nil>      Memory reservation in MB
  -mem.shares=                Memory shares level or number
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
  govc vm.clone -vm template-vm -link new-vm
  govc vm.clone -vm template-vm -snapshot s-name new-vm
  govc vm.clone -vm template-vm -link -snapshot s-name new-vm
  govc vm.clone -vm template-vm -snapshot $(govc snapshot.tree -vm template-vm -C) new-vm

Options:
  -annotation=           VM description
  -c=0                   Number of CPUs
  -customization=        Customization Specification Name
  -datastore-cluster=    Datastore cluster [GOVC_DATASTORE_CLUSTER]
  -ds=                   Datastore [GOVC_DATASTORE]
  -folder=               Inventory folder [GOVC_FOLDER]
  -force=false           Create VM if vmx already exists
  -host=                 Host system [GOVC_HOST]
  -link=false            Creates a linked clone from snapshot or source VM
  -m=0                   Size in MB of memory
  -net=                  Network [GOVC_NETWORK]
  -net.adapter=e1000     Network adapter type
  -net.address=          Network hardware address
  -on=true               Power on VM
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
  -snapshot=             Snapshot name to clone from
  -template=false        Create a Template
  -vm=                   Virtual machine [GOVC_VM]
  -waitip=false          Wait for VM to acquire IP address
```

## vm.console

```
Usage: govc vm.console [OPTIONS] VM

Generate console URL or screen capture for VM.

One of VMRC, VMware Player, VMware Fusion or VMware Workstation must be installed to
open VMRC console URLs.

Examples:
  govc vm.console my-vm
  govc vm.console -capture screen.png my-vm  # screen capture
  govc vm.console -capture - my-vm | display # screen capture to stdout
  open $(govc vm.console my-vm)              # MacOSX VMRC
  open $(govc vm.console -h5 my-vm)          # MacOSX H5
  xdg-open $(govc vm.console my-vm)          # Linux VMRC
  xdg-open $(govc vm.console -h5 my-vm)      # Linux H5

Options:
  -capture=              Capture console screen shot to file
  -h5=false              Generate HTML5 UI console link
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.create

```
Usage: govc vm.create [OPTIONS] NAME

Create VM.

For a list of possible '-g' IDs, see:
http://pubs.vmware.com/vsphere-6-5/topic/com.vmware.wssdk.apiref.doc/vim.vm.GuestOsDescriptor.GuestOsIdentifier.html

Examples:
  govc vm.create vm-name
  govc vm.create -m 2048 -c 2 -g freebsd64Guest -net.adapter vmxnet3 -disk.controller pvscsi vm-name

Options:
  -annotation=           VM description
  -c=1                   Number of CPUs
  -datastore-cluster=    Datastore cluster [GOVC_DATASTORE_CLUSTER]
  -disk=                 Disk path (to use existing) OR size (to create new, e.g. 20GB)
  -disk-datastore=       Datastore for disk file
  -disk.controller=scsi  Disk controller type
  -ds=                   Datastore [GOVC_DATASTORE]
  -firmware=bios         Firmware type [bios|efi]
  -folder=               Inventory folder [GOVC_FOLDER]
  -force=false           Create VM if vmx already exists
  -g=otherGuest          Guest OS ID
  -host=                 Host system [GOVC_HOST]
  -iso=                  ISO path
  -iso-datastore=        Datastore for ISO file
  -link=true             Link specified disk
  -m=1024                Size in MB of memory
  -net=                  Network [GOVC_NETWORK]
  -net.adapter=e1000     Network adapter type
  -net.address=          Network hardware address
  -on=true               Power on VM. Default is true if -disk argument is given.
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
  -version=              ESXi hardware version [5.5|6.0|6.5|6.7]
```

## vm.destroy

```
Usage: govc vm.destroy [OPTIONS] VM...

Power off and delete VM.

When a VM is destroyed, any attached virtual disks are also deleted.
Use the 'device.remove -vm VM -keep disk-*' command to detach and
keep disks if needed, prior to calling vm.destroy.

Examples:
  govc vm.destroy my-vm

Options:
```

## vm.disk.attach

```
Usage: govc vm.disk.attach [OPTIONS]

Attach existing disk to VM.

Examples:
  govc vm.disk.attach -vm $name -disk $name/disk1.vmdk
  govc vm.disk.attach -vm $name -disk $name/shared.vmdk -link=false -sharing sharingMultiWriter
  govc device.remove -vm $name -keep disk-* # detach disk(s)

Options:
  -controller=           Disk controller
  -disk=                 Disk path name
  -ds=                   Datastore [GOVC_DATASTORE]
  -link=true             Link specified disk
  -mode=                 Disk mode (persistent|nonpersistent|undoable|independent_persistent|independent_nonpersistent|append)
  -persist=true          Persist attached disk
  -sharing=              Sharing (sharingNone|sharingMultiWriter)
  -vm=                   Virtual machine [GOVC_VM]
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
  -disk.filePath=        Disk file name
  -disk.key=0            Disk unique key
  -disk.label=           Disk label
  -disk.name=            Disk name
  -mode=                 Disk mode (persistent|nonpersistent|undoable|independent_persistent|independent_nonpersistent|append)
  -sharing=              Sharing (sharingNone|sharingMultiWriter)
  -size=0B               New disk size
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.disk.create

```
Usage: govc vm.disk.create [OPTIONS]

Create disk and attach to VM.

Examples:
  govc vm.disk.create -vm $name -name $name/disk1 -size 10G
  govc vm.disk.create -vm $name -name $name/disk2 -size 10G -eager -thick -sharing sharingMultiWriter

Options:
  -controller=           Disk controller
  -ds=                   Datastore [GOVC_DATASTORE]
  -eager=false           Eagerly scrub new disk
  -mode=persistent       Disk mode (persistent|nonpersistent|undoable|independent_persistent|independent_nonpersistent|append)
  -name=                 Name for new disk
  -sharing=              Sharing (sharingNone|sharingMultiWriter)
  -size=10.0GB           Size of new disk
  -thick=false           Thick provision new disk
  -vm=                   Virtual machine [GOVC_VM]
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
  -mount=false           Mount tools CD installer in the guest
  -options=              Installer options
  -unmount=false         Unmount tools CD installer in the guest
  -upgrade=false         Upgrade tools in the guest
```

## vm.info

```
Usage: govc vm.info [OPTIONS] VM...

Display info for VM.

Examples:
  govc vm.info $vm
  govc vm.info -json $vm
  govc find . -type m -runtime.powerState poweredOn | xargs govc vm.info

Options:
  -e=false               Show ExtraConfig
  -g=true                Show general summary
  -r=false               Show resource summary
  -t=false               Show ToolsConfigInfo
  -waitip=false          Wait for VM to acquire IP address
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
  -a=false               Wait for an IP address on all NICs
  -esxcli=false          Use esxcli instead of guest tools
  -n=                    Wait for IP address on NIC, specified by device name or MAC
  -v4=false              Only report IPv4 addresses
  -wait=1h0m0s           Wait time for the VM obtain an IP address
```

## vm.keystrokes

```
Usage: govc vm.keystrokes [OPTIONS] VM

Send Keystrokes to VM.

Examples:
 Default Scenario
  govc vm.keystrokes -vm $vm -s "root" 	# writes 'root' to the console
  govc vm.keystrokes -vm $vm -c 0x15 	# writes an 'r' to the console
  govc vm.keystrokes -vm $vm -r 1376263 # writes an 'r' to the console
  govc vm.keystrokes -vm $vm -c 0x28 	# presses ENTER on the console
  govc vm.keystrokes -vm $vm -c 0x4c -la true -lc true 	# sends CTRL+ALT+DEL to console

Options:
  -c=                    USB HID Code (hex)
  -la=false              Enable/Disable Left Alt
  -lc=false              Enable/Disable Left Control
  -lg=false              Enable/Disable Left Gui
  -ls=false              Enable/Disable Left Shift
  -r=0                   Raw USB HID Code Value (int32)
  -ra=false              Enable/Disable Right Alt
  -rc=false              Enable/Disable Right Control
  -rg=false              Enable/Disable Right Gui
  -rs=false              Enable/Disable Right Shift
  -s=                    Raw String to Send
  -vm=                   Virtual machine [GOVC_VM]
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
  -host=                 Host system [GOVC_HOST]
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
```

## vm.migrate

```
Usage: govc vm.migrate [OPTIONS] VM...

Migrates VM to a specific resource pool, host or datastore.

Examples:
  govc vm.migrate -host another-host vm-1 vm-2 vm-3
  govc vm.migrate -pool another-pool vm-1 vm-2 vm-3
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
  -net=                  Network [GOVC_NETWORK]
  -net.adapter=e1000     Network adapter type
  -net.address=          Network hardware address
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.network.change

```
Usage: govc vm.network.change [OPTIONS] DEVICE

Change network DEVICE configuration.

Note that '-net' is currently required with '-net.address', even when not changing the VM network.

Examples:
  govc vm.network.change -vm $vm -net PG2 ethernet-0
  govc vm.network.change -vm $vm -net PG2 -net.address 00:00:0f:2e:5d:69 ethernet-0
  govc device.info -vm $vm ethernet-*

Options:
  -net=                  Network [GOVC_NETWORK]
  -net.adapter=e1000     Network adapter type
  -net.address=          Network hardware address
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.option.info

```
Usage: govc vm.option.info [OPTIONS] [GUEST_ID]...

VM config options for CLUSTER.

The config option data contains information about the execution environment for a VM
in the given CLUSTER, and optionally for a specific HOST.

This command only supports '-json' or '-dump' output, defaulting to the latter.

Examples:
  govc vm.option.info -cluster C0
  govc vm.option.info -cluster C0 ubuntu64Guest
  govc vm.option.info -cluster C0 -json | jq .GuestOSDescriptor[].Id
  govc vm.option.info -host my_hostname
  govc vm.option.info -vm my_vm

Options:
  -cluster=              Cluster [GOVC_CLUSTER]
  -host=                 Host system [GOVC_HOST]
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.power

```
Usage: govc vm.power [OPTIONS]

Options:
  -M=false               Use Datacenter.PowerOnMultiVM method instead of VirtualMachine.PowerOnVM
  -force=false           Force (ignore state error and hard shutdown/reboot if tools unavailable)
  -off=false             Power off
  -on=false              Power on
  -r=false               Reboot guest
  -reset=false           Power reset
  -s=false               Shutdown guest
  -suspend=false         Power suspend
  -wait=true             Wait for the operation to complete
```

## vm.question

```
Usage: govc vm.question [OPTIONS]

Options:
  -answer=               Answer to question
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.rdm.attach

```
Usage: govc vm.rdm.attach [OPTIONS]

Attach DEVICE to VM with RDM.

Examples:
  govc vm.rdm.attach -vm VM -device /vmfs/devices/disks/naa.000000000000000000000000000000000

Options:
  -device=               Device Name
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.rdm.ls

```
Usage: govc vm.rdm.ls [OPTIONS]

List available devices that could be attach to VM with RDM.

Examples:
  govc vm.rdm.ls -vm VM

Options:
  -vm=                   Virtual machine [GOVC_VM]
```

## vm.register

```
Usage: govc vm.register [OPTIONS] VMX

Add an existing VM to the inventory.

VMX is a path to the vm config file, relative to DATASTORE.

Examples:
  govc vm.register path/name.vmx
  govc vm.register -template -host $host path/name.vmx

Options:
  -ds=                   Datastore [GOVC_DATASTORE]
  -folder=               Inventory folder [GOVC_FOLDER]
  -host=                 Host system [GOVC_HOST]
  -name=                 Name of the VM
  -pool=                 Resource pool [GOVC_RESOURCE_POOL]
  -template=false        Mark VM as template
```

## vm.unregister

```
Usage: govc vm.unregister [OPTIONS] VM...

Remove VM from inventory without removing any of the VM files on disk.

Options:
```

## vm.upgrade

```
Usage: govc vm.upgrade [OPTIONS]

Upgrade VMs to latest hardware version

Examples:
  govc vm.upgrade -vm $vm_name
  govc vm.upgrade -version=$version -vm $vm_name
  govc vm.upgrade -version=$version -vm.uuid $vm_uuid

Options:
  -version=0             Target vm hardware version, by default -- latest available
  -vm=                   Virtual machine [GOVC_VM]
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
  -disable=false         Disable VNC
  -enable=false          Enable VNC
  -password=             VNC password
  -port=-1               VNC port (-1 for auto-select)
  -port-range=5900-5999  VNC port auto-select range
```

