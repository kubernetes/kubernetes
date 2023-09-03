
<a name="v0.29.0"></a>
## [Release v0.29.0](https://github.com/vmware/govmomi/compare/v0.28.0...v0.29.0)

> Release Date: 2022-07-06

### 🐞 Fix

- [d6dd8fb3]	Typos in vim25/soap/client CA tests
- [e086dfe4]	generate negative device key in AssignController
- [371a24a4]	Interface conversion panic in pkg simulator
- [a982c033]	use correct controlflag for vslm SetControlFlags API test
- [37b3b24c]	avoid possible panic in govc metric commands
- [310516e2]	govc: disambiguate vm/host search flags in vm.migrate
- [6af2cdc3]	govc-tests in Go v1.18
- [142cdca4]	Security update golangci-lint
- [971079ba]	use correct vcenter.DeploymentSpec.VmConfigSpec json tag

### 💫 API Changes

- [e6b5974a]	Add versioned user-agent header
- [ca7ee510]	add VmConfigSpec field to content library DeploymentSpec

### 💫 `govc` (CLI)

- [515ca29f]	Use unique searchFlagKey when calling NewSearchFlag
- [9d4ca658]	add library.deploy '-config' flag
- [fc17df08]	add 'device.clock.add' command
- [11f2d453]	Edit disk storage IO

### 💫 `vcsim` (Simulator)

- [a1a36c9a]	Fix disk capacity fields in ReconfigVM_Task
- [361c90ca]	Remove VM Guest.Net entry when removing Ethernet card
- [578b95e5]	Fix createVM to encode VM name
- [3325da0c]	add content library VmConfigSpec support
- [8928a489]	Update Dockerfile

### 📃 Documentation

- [5f5fb51e]	Fix broken link in PR template

### 🧹 Chore

- [69ac8494]	Update version.go for v0.29.0
- [80489cb5]	Update release automation
- [e1f76e37]	Add missing copyright header
- [6ed812fe]	Add Go boilerplate check

### ⚠️ BREAKING

### 📖 Commits

- [69ac8494]	chore: Update version.go for v0.29.0
- [7d3b2b39]	Update generated types
- [a1a36c9a]	vcsim: Fix disk capacity fields in ReconfigVM_Task
- [5f5fb51e]	docs: Fix broken link in PR template
- [d6dd8fb3]	fix: Typos in vim25/soap/client CA tests
- [e086dfe4]	fix: generate negative device key in AssignController
- [361c90ca]	vcsim: Remove VM Guest.Net entry when removing Ethernet card
- [80489cb5]	chore: Update release automation
- [e6b5974a]	api: Add versioned user-agent header
- [578b95e5]	vcsim: Fix createVM to encode VM name
- [371a24a4]	fix: Interface conversion panic in pkg simulator
- [a982c033]	fix: use correct controlflag for vslm SetControlFlags API test
- [37b3b24c]	fix: avoid possible panic in govc metric commands
- [310516e2]	fix: govc: disambiguate vm/host search flags in vm.migrate
- [5929abfb]	correct SetControlFlags and ClearControlFlags APIs
- [6af2cdc3]	fix: govc-tests in Go v1.18
- [e1f76e37]	chore: Add missing copyright header
- [6ed812fe]	chore: Add Go boilerplate check
- [142cdca4]	fix: Security update golangci-lint
- [3f4993d4]	build(deps): bump chuhlomin/render-template from 1.4 to 1.5
- [971079ba]	fix: use correct vcenter.DeploymentSpec.VmConfigSpec json tag
- [892dcfcc]	build(deps): bump nokogiri from 1.13.5 to 1.13.6 in /gen
- [303f0d95]	build(deps): bump goreleaser/goreleaser-action from 2 to 3
- [7eef76c3]	build(deps): bump nokogiri from 1.13.4 to 1.13.5 in /gen
- [515ca29f]	govc: Use unique searchFlagKey when calling NewSearchFlag
- [9d4ca658]	govc: add library.deploy '-config' flag
- [c5ebd552]	fix:fail to add ssd disk into allflash disk group Closes: [#2846](https://github.com/vmware/govmomi/issues/2846)
- [88f48e02]	Updated USAGE.md
- [3325da0c]	vcsim: add content library VmConfigSpec support
- [ca7ee510]	api: add VmConfigSpec field to content library DeploymentSpec
- [8928a489]	vcsim: Update Dockerfile
- [bf5d054d]	Fixed docs for govc library.info w/-json
- [2f9fab55]	emacs: fix json mode
- [fc17df08]	govc: add 'device.clock.add' command
- [11f2d453]	govc: Edit disk storage IO

<a name="v0.28.0"></a>
## [Release v0.28.0](https://github.com/vmware/govmomi/compare/v0.27.5...v0.28.0)

> Release Date: 2022-04-27

### 🐞 Fix

- [5ef4aaaf]	DiskFileOperation must consider both capacity fields
- [3566a35d]	govc guest validate username/password
- [bbbfd7bc]	govc test workflow cp error
- [a587742b]	avoid debug trace if http.Request.Body is nil
- [7e2ce135]	Ignore concurrent deletes in GetCategories
- [a7c6f15b]	Allow go 1.17 to go install
- [0f0201ad]	vapi - special param encoding for edge cluster lookup
- [e5209e34]	rest.Client.LoginByToken invalid signature
- [ad66761e]	support govc import.spec for remote ova
- [ebeaa71b]	Add IPv6 support for signing HTTP request
- [512c168e]	govc vm.destroy only destroys the 1st argument
- [d25aba08]	govc vcsa.net.proxy.info doesnt give output in json format
- [ac7c9bf9]	avoid possible panic in HostSystem.ManagementIPs
- [10fec668]	CHANGELOG sorting and generation

### 💡 Examples

- [c5826b8f]	Add Alarm Manager Example
- [9617bded]	add HostConfigManager_OptionManager
- [a1a9d848]	add VirtualDeviceList_SelectByBackingInfo

### 💫 API Changes

- [61c40001]	add GPU support to VirtualDeviceList.SelectByBackingInfo

### 💫 `govc` (CLI)

- [d8dd7f2b]	Add CLI command cluster.module
- [49a83e71]	Fix arguments validation in datastore.disk.inflate/shrink
- [01d31b53]	Add Feature dvs.create '-num-uplinks' flag
- [40e6cbc8]	Add Appliance access API
- [949ef572]	Add Appliance shutdown API's
- [d5ed6855]	Add support for VM hardware upgrade scheduling
- [742f2893]	add support for supervisor services deploy
- [3ba25d70]	Require full or absolute paths
- [a4ae62e7]	Add library info command
- [8fde8bce]	validate library.deploy arguments

### 💫 `vcsim` (Simulator)

- [3d8ddf16]	Fix device connectivity when vm is powered off
- [111ad9fc]	Use new action type in simulator PlaceVmsXCluster response
- [e92db045]	Fix NFS datastore moid collision
- [16e6bace]	set summary.guest.{hostName,ipAddress} in CustomizeVM
- [46a85642]	add ssoadmin simulator
- [811b829c]	Fix port filtering by criteria in FetchDVPorts
- [e8425be5]	revert vapi.Status() method
- [451ec35a]	Fix keys in DistributedVirtualPorts
- [6542ccb5]	Fix CreateFolder to encode folder name
- [8629c499]	Allow updating custom fields
- [93c2afd1]	copy device list when cloning a VM
- [3214d97a]	add support for cloning HostSystems
- [9b3d6353]	Fix distribute VMs across resource pools
- [93d39917]	Add TenantManager support in simulator
- [6de12ab7]	allow VM PowerOff when Host is in maintenance mode
- [48f7a881]	emit VmMigratedEvent in RelocateVM method

### 📃 Documentation

- [9ea287c2]	Update documentation
- [b4a2d3b3]	Add git blog post to CONTRIBUTING
- [c7e103e7]	Clarify squash in CONTRIBUTING
- [9317bdaf]	Update CONTIRBUTING.md file

### 🧹 Chore

- [d60b21d5]	Optimize Go CI workflows
- [2d72f576]	Add dependabot configuration
- [5c301091]	Use powerclicore on ghcr.io
- [7d8af1e7]	Update CI to Go 1.18
- [205c0e0d]	Add api: commit prefix
- [b6cd7c1b]	Add link to Discussions in New Issue
- [15efe49f]	Replace /rest with /api in vcsa.shutdown API's
- [db7edbf4]	Update workflow Go versions
- [05c28f4a]	upgrade go directive in go.mod to 1.17
- [ebff29b7]	Add notes to PR RELEASE workflow

### ⚠️ BREAKING

Fix distribute VMs across resource pools [9b3d6353]:
The name of virtual machines deployed in `vcsim` in a cluster (and
optionally child resource pools) has changed to include the
corresponding resource pool name. VM names deployed to standalone hosts
in `vcsim` are not changed.

### 📖 Commits

- [9ea287c2]	docs: Update documentation
- [89ae0933]	build(deps): bump actions/stale from 3 to 5
- [d60b21d5]	chore: Optimize Go CI workflows
- [0d1b4189]	build(deps): bump peter-evans/create-or-update-comment from 1 to 2
- [e85b164d]	build(deps): bump github/codeql-action from 1 to 2
- [5ef4aaaf]	fix: DiskFileOperation must consider both capacity fields
- [3566a35d]	fix: govc guest validate username/password
- [1f0f8cc8]	build(deps): bump chuhlomin/render-template from 1.2 to 1.4
- [7324f647]	build(deps): bump actions/upload-artifact from 2 to 3
- [808a439a]	build(deps): bump peter-evans/create-pull-request from 3 to 4
- [bdee9992]	build(deps): bump github.com/google/uuid from 1.2.0 to 1.3.0
- [2d72f576]	chore: Add dependabot configuration
- [bbbfd7bc]	fix: govc test workflow cp error
- [d8dd7f2b]	govc: Add CLI command cluster.module
- [90c90a0a]	build(deps): bump nokogiri from 1.13.2 to 1.13.4 in /gen
- [3cb3eff1]	ConfigInfo2ConfigSpec
- [3d8ddf16]	vcsim: Fix device connectivity when vm is powered off
- [b4a2d3b3]	docs: Add git blog post to CONTRIBUTING
- [49a83e71]	govc: Fix arguments validation in datastore.disk.inflate/shrink
- [5c301091]	chore: Use powerclicore on ghcr.io
- [7d8af1e7]	chore: Update CI to Go 1.18
- [111ad9fc]	vcsim: Use new action type in simulator PlaceVmsXCluster response
- [c5826b8f]	examples: Add Alarm Manager Example
- [46583051]	Move the ClusterClusterInitialPlacementAction to unreleased types + fix linter error
- [9b1de9c8]	Fix a linter error
- [cb2b8f5c]	Add a new type of cluster action used for placing a VM. This action inherits from InitialPlacement action because it conveys the resource pool and host for placing the VM. In addition, it also has the VM's ConfigSpecwhich is used for indicating the recommended datastore for each virtual disk in VM's ConfigSpec
- [9617bded]	examples: add HostConfigManager_OptionManager
- [8e4054fa]	adding a check that number of uplinks otherwise do default
- [aada9aa1]	Reconfigure LACP API for DVS
- [a1a9d848]	examples: add VirtualDeviceList_SelectByBackingInfo
- [61c40001]	api: add GPU support to VirtualDeviceList.SelectByBackingInfo
- [e92db045]	vcsim: Fix NFS datastore moid collision
- [01d31b53]	govc: Add Feature dvs.create '-num-uplinks' flag
- [11e469a4]	build(deps): bump nokogiri from 1.12.5 to 1.13.2 in /gen
- [547c63fd]	Added Support for vrdma NIC Type Signed-off-by: C S P Nanda <cspnanda[@gmail](https://github.com/gmail).com>
- [205c0e0d]	chore: Add api: commit prefix
- [b6cd7c1b]	chore: Add link to Discussions in New Issue
- [15efe49f]	chore: Replace /rest with /api in vcsa.shutdown API's
- [40e6cbc8]	govc: Add Appliance access API
- [16e6bace]	vcsim: set summary.guest.{hostName,ipAddress} in CustomizeVM
- [a587742b]	fix: avoid debug trace if http.Request.Body is nil
- [7e2ce135]	fix: Ignore concurrent deletes in GetCategories
- [1875bac1]	Add PlaceVmsXCluster bindings and simulator
- [a7c6f15b]	fix: Allow go 1.17 to go install
- [a5498b89]	Add BackingDiskObjectId go bindings to CNS API
- [0f0201ad]	fix: vapi - special param encoding for edge cluster lookup
- [46a85642]	vcsim: add ssoadmin simulator
- [297a3cae]	ssoadmin: add IdentitySources API bindings
- [811b829c]	vcsim: Fix port filtering by criteria in FetchDVPorts
- [e5209e34]	fix: rest.Client.LoginByToken invalid signature
- [c7e103e7]	docs: Clarify squash in CONTRIBUTING
- [e8425be5]	vcsim: revert vapi.Status() method
- [ad66761e]	fix: support govc import.spec for remote ova
- [803b6362]	sts: support issuing HoK token using HoK token
- [451ec35a]	vcsim: Fix keys in DistributedVirtualPorts
- [949ef572]	govc: Add Appliance shutdown API's
- [6542ccb5]	vcsim: Fix CreateFolder to encode folder name
- [d5ed6855]	govc: Add support for VM hardware upgrade scheduling
- [9317bdaf]	docs: Update CONTIRBUTING.md file
- [8629c499]	vcsim: Allow updating custom fields
- [93c2afd1]	vcsim: copy device list when cloning a VM
- [ebeaa71b]	fix: Add IPv6 support for signing HTTP request
- [b729a862]	Fix typo in (simulator.Context).WithLock() comment
- [cd577f46]	Fixed doc
- [ca1f45ae]	Added command flag documentation
- [512240a0]	Fixed goimports issues
- [ce88635f]	Added vm::ExportSnapshot and snapshot flag to export.ovf
- [742f2893]	govc: add support for supervisor services deploy
- [3214d97a]	vcsim: add support for cloning HostSystems
- [3ba25d70]	govc: Require full or absolute paths
- [db7edbf4]	chore: Update workflow Go versions
- [512c168e]	fix: govc vm.destroy only destroys the 1st argument
- [b51418e3]	Add IsAlreadyExists error helper
- [05c28f4a]	chore: upgrade go directive in go.mod to 1.17
- [a4ae62e7]	govc: Add library info command
- [d25aba08]	fix: govc vcsa.net.proxy.info doesnt give output in json format
- [ac7c9bf9]	fix: avoid possible panic in HostSystem.ManagementIPs
- [9b3d6353]	vcsim: Fix distribute VMs across resource pools
- [1da8c5e8]	Fix: Deep compare error types in simulator VM tests
- [d3eaa9b9]	Support Creating/Reconfiguring a simulator VM with VApp properties
- [10fec668]	fix: CHANGELOG sorting and generation
- [b05ed4e0]	Independent simulator.Context per-subtask in PowerOnMultiVM.
- [a0224d91]	Fix eam/simulator tests
- [e2498fb8]	Change references from global Map to ctx.Map.
- [ab446838]	Set the Context.Map's when global Map is set
- [3b86fd0c]	Re-work TenantManager addition to ServiceContent for older clients
- [93d39917]	vcsim: Add TenantManager support in simulator
- [9f737e00]	fix updating stale url from Makefile
- [8fde8bce]	govc: validate library.deploy arguments
- [6de12ab7]	vcsim: allow VM PowerOff when Host is in maintenance mode
- [ebff29b7]	chore: Add notes to PR RELEASE workflow
- [48f7a881]	vcsim: emit VmMigratedEvent in RelocateVM method
- [bb4f739b]	Support finding Portgroups by ID in Finder.Network

<a name="v0.27.5"></a>
## [Release v0.27.5](https://github.com/vmware/govmomi/compare/v0.27.4...v0.27.5)

> Release Date: 2022-06-02

### 🐞 Fix

- [e97c9708]	use correct controlflag for vslm SetControlFlags API test

### ⚠️ BREAKING

### 📖 Commits

- [e97c9708]	fix: use correct controlflag for vslm SetControlFlags API test
- [f8cf9ef7]	correct SetControlFlags and ClearControlFlags APIs

<a name="v0.27.4"></a>
## [Release v0.27.4](https://github.com/vmware/govmomi/compare/v0.27.3...v0.27.4)

> Release Date: 2022-02-10

### 🐞 Fix

- [285e80cd]	avoid debug trace if http.Request.Body is nil
- [dde50904]	Ignore concurrent deletes in GetCategories
- [cbc68fc0]	rest.Client.LoginByToken invalid signature

### 💫 `vcsim` (Simulator)

- [df595d82]	add ssoadmin simulator

### ⚠️ BREAKING

### 📖 Commits

- [285e80cd]	fix: avoid debug trace if http.Request.Body is nil
- [dde50904]	fix: Ignore concurrent deletes in GetCategories
- [fc1fce62]	Add PlaceVmsXCluster bindings and simulator
- [df595d82]	vcsim: add ssoadmin simulator
- [9ca477aa]	ssoadmin: add IdentitySources API bindings
- [24fe60f1]	Add BackingDiskObjectId go bindings to CNS API
- [cbc68fc0]	fix: rest.Client.LoginByToken invalid signature

<a name="v0.27.3"></a>
## [Release v0.27.3](https://github.com/vmware/govmomi/compare/v0.27.2...v0.27.3)

> Release Date: 2022-02-01

### 🐞 Fix

- [2d7cd133]	Add IPv6 support for signing HTTP request

### ⚠️ BREAKING

### 📖 Commits

- [6b4e2391]	sts: support issuing HoK token using HoK token
- [2d7cd133]	fix: Add IPv6 support for signing HTTP request

<a name="v0.27.2"></a>
## [Release v0.27.2](https://github.com/vmware/govmomi/compare/v0.27.1...v0.27.2)

> Release Date: 2021-11-23

### 🐞 Fix

- [f04d77d6]	avoid possible panic in HostSystem.ManagementIPs

### ⚠️ BREAKING

### 📖 Commits

- [f04d77d6]	fix: avoid possible panic in HostSystem.ManagementIPs

<a name="v0.27.1"></a>
## [Release v0.27.1](https://github.com/vmware/govmomi/compare/v0.27.0...v0.27.1)

> Release Date: 2021-10-20

### ⚠️ BREAKING

### 📖 Commits

- [6209be5b]	Support finding Portgroups by ID in Finder.Network

<a name="v0.27.0"></a>
## [Release v0.27.0](https://github.com/vmware/govmomi/compare/v0.26.2...v0.27.0)

> Release Date: 2021-10-14

### 🐞 Fix

- [57c4be58]	multi-value query params for vAPI methods
- [815e2d8f]	avoid vNIC mapping in guest.TransferURL if URL.Host is an IP
- [81a7dbe9]	avoid use of vNIC IP in guest.TransferURL if there are multiple
- [61afce31]	Update CnsQueryAsync API request parameters to handle nil for QuerySelection
- [a601a8a8]	generate negative device key

### 💫 `govc` (CLI)

- [b5426eba]	Add feature to read file contents for ExtraConfig
- [85956c77]	fix tasks to activate option dump/json/xml
- [f4ef4d93]	Fix incorrect MoRef parsing
- [d695f4cf]	Handle powered on VMs in vm.destroy
- [94f63681]	add library.clone '-e' and '-m' options
- [2fcae372]	add vsan info and change commands

### 💫 `vcsim` (Simulator)

- [fa457940]	Fix PowerOnMultiVMTask() to return per-VM tasks
- [e67b1b11]	check if VM host InMaintenanceMode

### 📃 Documentation

- [82e447d9]	Update govc USAGE

### 🧹 Chore

- [05adcc97]	Remove -i parameter in go install
- [18ea9cc5]	Update PR and release docs

### ⚠️ BREAKING

Add task manager and collector [397c8aad]:
`event.Manager` does not embed `object.Common` anymore. Only the methods
`Client()` and `Reference()` are implemented.
`event.NewHistoryCollector()` is now unexported (to
`newHistoryCollector()`) as it was merely a helper and to comply with
the task manager implementation.

### 📖 Commits

- [78f30265]	update contributors
- [68b54585]	Refactor EAM code to use BaseAgencyConfigInfo interface instead of impl
- [d5ded1f0]	Implement mo.Reference interface in task+event managers
- [038bc3d8]	Update vslm types to vCenter 7.0U3 (build 18700403)
- [ae8161df]	Update pbm types to vCenter 7.0U3 (build 18700403)
- [f2d167de]	Update eam types to vCenter 7.0U3 (build 18700403)
- [f1c7b54d]	Update vim25 types to vCenter 7.0U3 (build 18700403)
- [445fd552]	Update gen.sh to vCenter 7.0U3 (build 18700403)
- [961f0ae6]	example: find VirtualMachine's Cluster
- [57c4be58]	fix: multi-value query params for vAPI methods
- [f7e8ed73]	Set custom HTTP headers for VAPI calls
- [05adcc97]	chore: Remove -i parameter in go install
- [82e447d9]	docs: Update govc USAGE
- [b5426eba]	govc: Add feature to read file contents for ExtraConfig
- [815e2d8f]	fix: avoid vNIC mapping in guest.TransferURL if URL.Host is an IP
- [81a7dbe9]	fix: avoid use of vNIC IP in guest.TransferURL if there are multiple
- [18ea9cc5]	chore: Update PR and release docs
- [a66d23ed]	build(deps): bump nokogiri from 1.11.4 to 1.12.5 in /gen
- [a853b300]	Fix: Nil-Pointer Exception in ResourceAllocation
- [fa457940]	vcsim: Fix PowerOnMultiVMTask() to return per-VM tasks
- [85956c77]	govc: fix tasks to activate option dump/json/xml
- [61afce31]	fix: Update CnsQueryAsync API request parameters to handle nil for QuerySelection
- [397c8aad]	feat: Add task manager and collector
- [a601a8a8]	fix: generate negative device key
- [f4ef4d93]	govc: Fix incorrect MoRef parsing
- [e67b1b11]	vcsim: check if VM host InMaintenanceMode
- [d695f4cf]	govc: Handle powered on VMs in vm.destroy
- [a55fa7dc]	feat: Add optional WaitOptions to WaitForUpdates
- [94f63681]	govc: add library.clone '-e' and '-m' options
- [2fcae372]	govc: add vsan info and change commands

<a name="v0.26.2"></a>
## [Release v0.26.2](https://github.com/vmware/govmomi/compare/v0.26.1...v0.26.2)

> Release Date: 2022-03-21

### 🐞 Fix

- [76a22af3]	avoid possible panic in HostSystem.ManagementIPs
- [566d2ac1]	avoid use of vNIC IP in guest.TransferURL if there are multiple

### ⚠️ BREAKING

### 📖 Commits

- [76a22af3]	fix: avoid possible panic in HostSystem.ManagementIPs
- [566d2ac1]	fix: avoid use of vNIC IP in guest.TransferURL if there are multiple

<a name="v0.26.1"></a>
## [Release v0.26.1](https://github.com/vmware/govmomi/compare/v0.26.0...v0.26.1)

> Release Date: 2021-08-16

### 🐞 Fix

- [a366e352]	Regenerated interface and type to include BaseAgencyConfigInfo. Closes: [#2545](https://github.com/vmware/govmomi/issues/2545).
- [d66ef551]	explicitly import eam/simulator in test

### 💡 Examples

- [0c045a63]	Add Property Wait example

### 💫 `govc` (CLI)

- [012f5348]	support updating items in library.update command
- [5743d5b6]	Allow cluster.change to set ClusterDrsConfigInfo.vmotionRate
- [03210c91]	Add object.collect type flag alias help

### 💫 `vcsim` (Simulator)

- [ce6ed634]	avoid race when fetching object Locker
- [31821de3]	use 'domain-c' prefix for cluster moids
- [3625e6dd]	propagate CustomizeVM MacAddress to Virtual NIC
- [389c0382]	Take the host parameter into account while cloning a VM on a cluster
- [6fba1da7]	Implement VSLM ExtendDisk_Task

### 🧹 Chore

- [ddc2b47a]	Include commit details in BREAKING section

### ⚠️ BREAKING

### 📖 Commits

- [a366e352]	fix: Regenerated interface and type to include BaseAgencyConfigInfo. Closes: [#2545](https://github.com/vmware/govmomi/issues/2545).
- [655f8e5c]	testing for lab
- [ce6ed634]	vcsim: avoid race when fetching object Locker
- [31821de3]	vcsim: use 'domain-c' prefix for cluster moids
- [0aa1de31]	make processing of mac addresses case insensitive ([#2510](https://github.com/vmware/govmomi/issues/2510))
- [012f5348]	govc: support updating items in library.update command
- [0c045a63]	examples: Add Property Wait example
- [f30cefc3]	Add Reauth flag to skip loading cached sessions
- [3625e6dd]	vcsim: propagate CustomizeVM MacAddress to Virtual NIC
- [5743d5b6]	govc: Allow cluster.change to set ClusterDrsConfigInfo.vmotionRate
- [389c0382]	vcsim: Take the host parameter into account while cloning a VM on a cluster
- [7bf48333]	Added CNS querySnapshots binding, simulator testcases and client testcases
- [017ab414]	Added CreateSnapshots and DeleteSnapshots methods in cns simulator as well as their test cases
- [03210c91]	govc: Add object.collect type flag alias help
- [d66ef551]	fix: explicitly import eam/simulator in test
- [ddc2b47a]	chore: Include commit details in BREAKING section
- [6fba1da7]	vcsim: Implement VSLM ExtendDisk_Task

<a name="v0.26.0"></a>
## [Release v0.26.0](https://github.com/vmware/govmomi/compare/v0.25.0...v0.26.0)

> Release Date: 2021-06-03

### 🐞 Fix

- [70b92d6d]	Isolate SSO govc tests

### 💡 Examples

- [81b1de17]	add toolbox Client.Run

### 💫 `govc` (CLI)

- [e37e515b]	fix default guest.run path for unsupported Windows guests
- [0e7012d0]	Add support for getting the VC proxy and no-proxy configuration ([#2435](https://github.com/vmware/govmomi/issues/2435))
- [6afb8ff9]	Change 'Maintenance Mode' printing in host.info

### 💫 `vcsim` (Simulator)

- [dff7f6bb]	fix panic in QueryPerfCounter method
- [df9dfde1]	set VirtualMachine ChangeTrackingSupported property ([#2468](https://github.com/vmware/govmomi/issues/2468))
- [0c0ed98a]	fix race in CloneVM_Task
- [d01d0fa7]	add simulator.RunContainer method
- [8ab0c99a]	untie datastore capacity from local fs
- [d31941c8]	Modify Usage of README. (based on v0.25.0)
- [4fea687c]	include all namespaces in /about info
- [bd3467d4]	avoid edit device panic when DeviceInfo is nil
- [35a42af5]	add guest operations process support

### 📃 Documentation

- [75eee8e4]	update govc/USAGE and CONTRIBUTORS
- [1f795d21]	Add blog to vcsim README
- [2719c229]	Document linker and GOFLAGS for build vars

### 🧹 Chore

- [b4e1f965]	Fix CONTRIB link in greeting
- [6f2597be]	Update CHANGELOG implementation
- [d3944e17]	Add and reorder commits in CHANGELOG
- [a796d3fc]	Add make help target
- [8bc8fd28]	Add issue and PR templates
- [60e33916]	Document commit prefixes
- [cac1d8d7]	Add issue greeting
- [0f1c3f89]	Add WIP Action
- [921ad37a]	Remove dep files
- [1d4ce94a]	Clean up documentation
- [991278b9]	Remove unused release script
- [16d8add5]	Automate CHANGELOG

### ⚠️ BREAKING

### 📖 Commits

- [75eee8e4]	docs: update govc/USAGE and CONTRIBUTORS
- [dff7f6bb]	vcsim: fix panic in QueryPerfCounter method
- [b4e1f965]	chore: Fix CONTRIB link in greeting
- [df9dfde1]	vcsim: set VirtualMachine ChangeTrackingSupported property ([#2468](https://github.com/vmware/govmomi/issues/2468))
- [8cbe64c5]	Fix: Protect FileProvider.files to avoid concurrent modification
- [6f2597be]	chore: Update CHANGELOG implementation
- [0c0ed98a]	vcsim: fix race in CloneVM_Task
- [81b1de17]	examples: add toolbox Client.Run
- [d01d0fa7]	vcsim: add simulator.RunContainer method
- [9223b5ae]	Add toolbox.NewClient method
- [8ab0c99a]	vcsim: untie datastore capacity from local fs
- [e37e515b]	govc: fix default guest.run path for unsupported Windows guests
- [d3944e17]	chore: Add and reorder commits in CHANGELOG
- [83e29c69]	Update GitHub Test Action to use Makefile
- [a7f2c47e]	Update vslm types vC build 17986435 (7.0U2HP4)
- [067374fd]	Update sms types vC build 17986435 (7.0U2HP4)
- [d9f507f0]	Update pbm types vC build 17986435 (7.0U2HP4)
- [c89f8dd5]	Update eam types vC build 17986435 (7.0U2HP4)
- [b72432ef]	Update vim25 types vC build 17986435 (7.0U2HP4)
- [e53716dd]	Update gen.sh to vC build 17986435 (7.0U2HP4)
- [1f795d21]	docs: Add blog to vcsim README
- [338f5529]	Ran "make fix" to correct lint issues
- [23d77ba4]	Add support for golangci-lint
- [d31941c8]	vcsim: Modify Usage of README. (based on v0.25.0)
- [7046a0d3]	Support pre-auth handlers in vC Sim
- [2e8860d1]	Add CNS Snapshot APIs in govmomi
- [13d4d376]	Remove vendor
- [a796d3fc]	chore: Add make help target
- [ef824a20]	Fix QueryAsyncVolume API test to be invoked only for vSphere 7.0.3
- [c1900234]	Fix data race in simulator.container.id
- [3212351e]	install bin doc: permalink to latest version
- [7d779833]	Setup CodeQL Analysis
- [566250ff]	build(deps): bump nokogiri from 1.11.1 to 1.11.4 in /gen
- [f814a9ca]	ESX Agent Manager (EAM) Client and Simulator
- [0e7012d0]	govc: Add support for getting the VC proxy and no-proxy configuration ([#2435](https://github.com/vmware/govmomi/issues/2435))
- [8bc8fd28]	chore: Add issue and PR templates
- [4fea687c]	vcsim: include all namespaces in /about info
- [bd3467d4]	vcsim: avoid edit device panic when DeviceInfo is nil
- [70b92d6d]	fix: Isolate SSO govc tests
- [80c9053e]	Correcting broken Kubernetes vSphere Cloud Provider links
- [60e33916]	chore: Document commit prefixes
- [57a141f3]	Update govc test docs with act
- [ff578914]	Use "vcsim uuidgen" for bats tests
- [61e12ddb]	Only greet unassociated users
- [e39dfdc8]	Add chore section to CHANGELOG
- [6afb8ff9]	govc: Change 'Maintenance Mode' printing in host.info
- [cac1d8d7]	chore: Add issue greeting
- [0f1c3f89]	chore: Add WIP Action
- [921ad37a]	chore: Remove dep files
- [2719c229]	docs: Document linker and GOFLAGS for build vars
- [f3645a96]	Clarify SetRootCAs behavior
- [c368e57f]	toolbox: add hgfs freebsd stub
- [35a42af5]	vcsim: add guest operations process support
- [64e55d81]	Set RoundTripper in ssoadmin.NewClient
- [1d4ce94a]	chore: Clean up documentation
- [991278b9]	chore: Remove unused release script
- [16d8add5]	chore: Automate CHANGELOG
- [e8805c92]	Add NotFoundFault in cns types
- [8576fe27]	Add queryAsyncVolume in simulator
- [4b9e0813]	Simplify binary download instructions
- [3062dda9]	Remove Travis CI
- [0be5632f]	adding rancher to projects and reorganizing in alpha order
- [4a63a28c]	Add bindings for CnsQueryAsyncVolume API
- [a8c80b93]	Update READMEs with artifacts and Docker images
- [26c9690c]	Fix VM Guest test and vet warnings
- [a32cd0b3]	Add RELEASE documentation
- [cc660b0e]	Increase govc tests timeout
- [d7bfaf4f]	toolbox: move process management to its own package
- [e86da96e]	Exclude go files in release tarball

<a name="v0.25.0"></a>
## [Release v0.25.0](https://github.com/vmware/govmomi/compare/v0.24.2...v0.25.0)

> Release Date: 2021-04-16

### 💡 Examples

- [38da87ff]	add NetworkReference.EthernetCardBackingInfo

### 💫 `govc` (CLI)

- [1ac314c3]	add vm.customize -dns-suffix flag
- [60e0e895]	update test images URL
- [cdf3ace6]	log invalid NetworkMapping.Name with import.ova command
- [f8b3d8a8]	revert pretty print pruning optimization
- [35481467]	add library.update command
- [749c2239]	add session.ls -S flag
- [93245c1e]	add tree command
- [790f9ce6]	include sub task fault messages on failure
- [d2a353ba]	remove device.boot -firmware default
- [de6032e0]	add '-trace' and '-verbose' flags
- [63bb5c1e]	metric command enhancements and fixes
- [7844a8c2]	fix vm.migrate search index flags
- [5dacf627]	fix cluster.usage Free field
- [f71bcf25]	fix session curl when given a URL query
- [c954c2a5]	validate license.remove
- [3b25c3f1]	validate required library.clone NAME arg
- [344b7a30]	note 'disk.ls -R' in volume.rm help
- [8942055a]	add device.info examples to get disk UUID and vmdk
- [1b0af949]	fix vm.markasvm examples
- [add8be5a]	fix incorrect DeviceID value in device.pci.add
- [1f4f5640]	add IPv6 support to vm.customize

### 💫 `vcsim` (Simulator)

- [27d8d2e4]	put verbose logging behind '-trace' flag
- [0ef4ae22]	add moid value mapping mappings
- [082f9927]	add vsan simulator
- [25970530]	fix Task.Info.Entity in RevertToSnapshot_Task
- [f0a045ac]	set VirtualMachine.Config.CreateDate property
- [e51eb2b9]	support EventFilterSpec.Time
- [8e45fa4a]	emit CustomizationSucceeded event from CustomizeVM
- [c000bd6e]	add DistributedVirtualSwitchManager
- [bcd5fa87]	set VirtualDisk backing UUID
- [ccdcbe89]	move product suffix in ServiceContent.About
- [393e7330]	use linked list for EventHistoryCollector
- [9c4dc1a1]	escape datastore name
- [9c2fe70f]	record/replay EnvironmentBrowser.QueryConfigOption
- [5fd7e264]	fix EventHistoryCollector fixes
- [0b755a59]	switch bats tests from esx to vcsim env
- [3f1caf82]	fixes for PowerCLI Get-VirtualNetwork

### 📃 Documentation

- [e18b601f]	update for 0.25 release

### ⚠️ BREAKING

### 📖 Commits

- [6fe8d60a]	Fix folder write for govc container
- [e18b601f]	docs: update for 0.25 release
- [1ac314c3]	govc: add vm.customize -dns-suffix flag
- [22d911f6]	Add Cron Docker Login Action
- [60e0e895]	govc: update test images URL
- [3385b3e0]	Add action to automate release
- [cdf3ace6]	govc: log invalid NetworkMapping.Name with import.ova command
- [27d8d2e4]	vcsim: put verbose logging behind '-trace' flag
- [f8b3d8a8]	govc: revert pretty print pruning optimization
- [0ef4ae22]	vcsim: add moid value mapping mappings
- [df08d4b2]	First step towards release automation
- [f9b79a4f]	export simulator.task.Wait()
- [917c4ec8]	Ensure lock hand-off to simulator.Task goroutine
- [b45b228f]	Simulator Task Delay
- [4b59b652]	Make Simulator Tasks Async
- [bc52c793]	Associate every registry lock with a Context.
- [054971ee]	Wait until VM creation completes before adding to folder
- [35481467]	govc: add library.update command
- [7403b470]	Fix race in simulator's PropertyCollector
- [aadb2082]	Add action to block WIP PRs
- [749c2239]	govc: add session.ls -S flag
- [bc297330]	[3ad0f415] Update Dockerfiles and .goreleaser.yml
- [082f9927]	vcsim: add vsan simulator
- [8c38d56d]	Add a stretched cluster conversion command.
- [408c531a]	gofmt
- [e8a6b126]	Update govc/flags/output.go
- [bf54a7c4]	Add more badges
- [93245c1e]	govc: add tree command
- [790f9ce6]	govc: include sub task fault messages on failure
- [07e6e923]	Use Github Actions Status Badges
- [d2a353ba]	govc: remove device.boot -firmware default
- [4ed615f6]	Add chainable RoundTripper support to endpoint clients
- [bab95d26]	Add the vSAN stretched cluster reference.
- [6ff33db7]	Fix events example
- [de6032e0]	govc: add '-trace' and '-verbose' flags
- [7aae8dfb]	Add support for calling vCenter for VLSM ExtendDisk and InflateDisk
- [7a276bf6]	Add client test file for vslm package to validate register disk and cns create volume
- [dc29aa29]	Fix performance.Manager.SampleByName truncation
- [18b53fd2]	Added UpdateServiceMessage to Session Manager
- [63bb5c1e]	govc: metric command enhancements and fixes
- [7844a8c2]	govc: fix vm.migrate search index flags
- [7ab111bd]	Drop clusterDistribution from vSAN 7.0 update and create spec elements
- [2c57a8a3]	Use Github Actions
- [52631496]	Marshal soapFaultError as JSON
- [f9e323a6]	fix tab indentation
- [ae129ba0]	add tests and implement HA Ready Condition
- [f34b3fa2]	implement vSphere HA additional delay to VM HA overrides in govc
- [25970530]	vcsim: fix Task.Info.Entity in RevertToSnapshot_Task
- [5dacf627]	govc: fix cluster.usage Free field
- [0d155a61]	Handling invalid reader size
- [b70542a5]	Using progress reader in WriteFile
- [b7f9e034]	use correct enum for vm restart priority
- [d3d49a36]	Add support for snapshot size calculations
- [61bfa072]	Use a dash to indicate empty address
- [f0a045ac]	vcsim: set VirtualMachine.Config.CreateDate property
- [4d9a9000]	vim25: fix race in TemporaryNetworkError retry func
- [2f14e4b2]	ovf: add Config and ExtraConfig to VirtualHardwareSection
- [50328780]	Add vSAN 7.0U1 release constant
- [886573de]	Update .goreleaser.yml
- [1cdb3164]	Change the address type to automatic
- [667a3791]	Remove duplicate cns bindings from vsan directory
- [f71bcf25]	govc: fix session curl when given a URL query
- [d92f41de]	Update volume ACL spec to add delete field
- [c954c2a5]	govc: validate license.remove
- [2a4f8c8a]	Update ConfigureVolumeACLs bindings in cns types
- [3b25c3f1]	govc: validate required library.clone NAME arg
- [344b7a30]	govc: note 'disk.ls -R' in volume.rm help
- [8942055a]	govc: add device.info examples to get disk UUID and vmdk
- [1b0af949]	govc: fix vm.markasvm examples
- [543e52ea]	govc-env --save default
- [0a5f2a99]	Little fix for "govc-env --save without config name"
- [4a7a0b45]	gen: require nokogiri 1.11.0 or higher
- [add8be5a]	govc: fix incorrect DeviceID value in device.pci.add
- [e51eb2b9]	vcsim: support EventFilterSpec.Time
- [1f4f5640]	govc: add IPv6 support to vm.customize
- [8e45fa4a]	vcsim: emit CustomizationSucceeded event from CustomizeVM
- [c000bd6e]	vcsim: add DistributedVirtualSwitchManager
- [bcd5fa87]	vcsim: set VirtualDisk backing UUID
- [ccdcbe89]	vcsim: move product suffix in ServiceContent.About
- [393e7330]	vcsim: use linked list for EventHistoryCollector
- [9c4dc1a1]	vcsim: escape datastore name
- [9c2fe70f]	vcsim: record/replay EnvironmentBrowser.QueryConfigOption
- [5fd7e264]	vcsim: fix EventHistoryCollector fixes
- [40a2cf0b]	Skip tests that require docker on TravisCI
- [00ee2911]	toolbox: skip tests that require Linux
- [0b755a59]	vcsim: switch bats tests from esx to vcsim env
- [c6d5264a]	Updated projects to include VMware Event Broker Appliance
- [ae44a547]	ExampleCollector_Retrieve: Add missing err return
- [38da87ff]	examples: add NetworkReference.EthernetCardBackingInfo
- [3f1caf82]	vcsim: fixes for PowerCLI Get-VirtualNetwork
- [041a98b8]	Fix DvsNetworkRuleQualifier interface
- [44e05fe4]	SHA-1 deprecated in 2011, sha256sum for releases

<a name="v0.24.2"></a>
## [Release v0.24.2](https://github.com/vmware/govmomi/compare/v0.24.1...v0.24.2)

> Release Date: 2021-10-14

### 🐞 Fix

- [b18f06b5]	avoid vNIC mapping in guest.TransferURL if URL.Host is an IP
- [5a2a8aba]	avoid use of vNIC IP in guest.TransferURL if there are multiple

### ⚠️ BREAKING

### 📖 Commits

- [b18f06b5]	fix: avoid vNIC mapping in guest.TransferURL if URL.Host is an IP
- [5a2a8aba]	fix: avoid use of vNIC IP in guest.TransferURL if there are multiple

<a name="v0.24.1"></a>
## [Release v0.24.1](https://github.com/vmware/govmomi/compare/v0.24.0...v0.24.1)

> Release Date: 2021-03-17

### 💡 Examples

- [38da87ff]	add NetworkReference.EthernetCardBackingInfo

### 💫 `govc` (CLI)

- [63bb5c1e]	metric command enhancements and fixes
- [7844a8c2]	fix vm.migrate search index flags
- [5dacf627]	fix cluster.usage Free field
- [f71bcf25]	fix session curl when given a URL query
- [c954c2a5]	validate license.remove
- [3b25c3f1]	validate required library.clone NAME arg
- [344b7a30]	note 'disk.ls -R' in volume.rm help
- [8942055a]	add device.info examples to get disk UUID and vmdk
- [1b0af949]	fix vm.markasvm examples
- [add8be5a]	fix incorrect DeviceID value in device.pci.add
- [1f4f5640]	add IPv6 support to vm.customize

### 💫 `vcsim` (Simulator)

- [25970530]	fix Task.Info.Entity in RevertToSnapshot_Task
- [f0a045ac]	set VirtualMachine.Config.CreateDate property
- [e51eb2b9]	support EventFilterSpec.Time
- [8e45fa4a]	emit CustomizationSucceeded event from CustomizeVM
- [c000bd6e]	add DistributedVirtualSwitchManager
- [bcd5fa87]	set VirtualDisk backing UUID
- [ccdcbe89]	move product suffix in ServiceContent.About
- [393e7330]	use linked list for EventHistoryCollector
- [9c4dc1a1]	escape datastore name
- [9c2fe70f]	record/replay EnvironmentBrowser.QueryConfigOption
- [5fd7e264]	fix EventHistoryCollector fixes
- [0b755a59]	switch bats tests from esx to vcsim env
- [3f1caf82]	fixes for PowerCLI Get-VirtualNetwork

### ⚠️ BREAKING

### 📖 Commits

- [7a276bf6]	Add client test file for vslm package to validate register disk and cns create volume
- [dc29aa29]	Fix performance.Manager.SampleByName truncation
- [18b53fd2]	Added UpdateServiceMessage to Session Manager
- [63bb5c1e]	govc: metric command enhancements and fixes
- [7844a8c2]	govc: fix vm.migrate search index flags
- [7ab111bd]	Drop clusterDistribution from vSAN 7.0 update and create spec elements
- [52631496]	Marshal soapFaultError as JSON
- [f9e323a6]	fix tab indentation
- [ae129ba0]	add tests and implement HA Ready Condition
- [f34b3fa2]	implement vSphere HA additional delay to VM HA overrides in govc
- [25970530]	vcsim: fix Task.Info.Entity in RevertToSnapshot_Task
- [5dacf627]	govc: fix cluster.usage Free field
- [b7f9e034]	use correct enum for vm restart priority
- [d3d49a36]	Add support for snapshot size calculations
- [61bfa072]	Use a dash to indicate empty address
- [f0a045ac]	vcsim: set VirtualMachine.Config.CreateDate property
- [4d9a9000]	vim25: fix race in TemporaryNetworkError retry func
- [2f14e4b2]	ovf: add Config and ExtraConfig to VirtualHardwareSection
- [50328780]	Add vSAN 7.0U1 release constant
- [886573de]	Update .goreleaser.yml
- [1cdb3164]	Change the address type to automatic
- [667a3791]	Remove duplicate cns bindings from vsan directory
- [f71bcf25]	govc: fix session curl when given a URL query
- [d92f41de]	Update volume ACL spec to add delete field
- [c954c2a5]	govc: validate license.remove
- [2a4f8c8a]	Update ConfigureVolumeACLs bindings in cns types
- [3b25c3f1]	govc: validate required library.clone NAME arg
- [344b7a30]	govc: note 'disk.ls -R' in volume.rm help
- [8942055a]	govc: add device.info examples to get disk UUID and vmdk
- [1b0af949]	govc: fix vm.markasvm examples
- [543e52ea]	govc-env --save default
- [0a5f2a99]	Little fix for "govc-env --save without config name"
- [4a7a0b45]	gen: require nokogiri 1.11.0 or higher
- [add8be5a]	govc: fix incorrect DeviceID value in device.pci.add
- [e51eb2b9]	vcsim: support EventFilterSpec.Time
- [1f4f5640]	govc: add IPv6 support to vm.customize
- [8e45fa4a]	vcsim: emit CustomizationSucceeded event from CustomizeVM
- [c000bd6e]	vcsim: add DistributedVirtualSwitchManager
- [bcd5fa87]	vcsim: set VirtualDisk backing UUID
- [ccdcbe89]	vcsim: move product suffix in ServiceContent.About
- [393e7330]	vcsim: use linked list for EventHistoryCollector
- [9c4dc1a1]	vcsim: escape datastore name
- [9c2fe70f]	vcsim: record/replay EnvironmentBrowser.QueryConfigOption
- [5fd7e264]	vcsim: fix EventHistoryCollector fixes
- [40a2cf0b]	Skip tests that require docker on TravisCI
- [00ee2911]	toolbox: skip tests that require Linux
- [0b755a59]	vcsim: switch bats tests from esx to vcsim env
- [c6d5264a]	Updated projects to include VMware Event Broker Appliance
- [ae44a547]	ExampleCollector_Retrieve: Add missing err return
- [38da87ff]	examples: add NetworkReference.EthernetCardBackingInfo
- [3f1caf82]	vcsim: fixes for PowerCLI Get-VirtualNetwork
- [041a98b8]	Fix DvsNetworkRuleQualifier interface
- [44e05fe4]	SHA-1 deprecated in 2011, sha256sum for releases

<a name="v0.24.0"></a>
## [Release v0.24.0](https://github.com/vmware/govmomi/compare/v0.23.1...v0.24.0)

> Release Date: 2020-12-21

### 💡 Examples

- [7178588c]	add Folder.CreateVM
- [b4f7243b]	add ContainerView retrieve clusters
- [1d21fff9]	use session.Cache
- [8af8cef6]	add events
- [e153061f]	fix simulator.RunContainer on MacOSX

### 💫 `govc` (CLI)

- [1ec59a7c]	fix build.sh git tag injection
- [31c0836e]	add cluster.usage command
- [79514c81]	add volume.ls -ds option
- [5e57b3f6]	add device.boot -firmware option
- [4d82f0ff]	add dvs.portgroup.{add,change} '-auto-expand' flag
- [4a1d05ac]	fix object.collect ContainerView updates
- [e84d0d18]	document vm.disk.attach -link behavior
- [70a9ced4]	fix vm.clone panic when target VM already exists
- [a97e6168]	support sparse backing in vm.disk.change
- [3380cd30]	add CNS volume ls and rm commands
- [f7170fd2]	add find -p flag
- [b40cdd8a]	add storage.policy commands
- [d0111d28]	add vm.console -wss flag
- [86374ea2]	support multi value flags in host.esxcli command
- [ebcfa3d4]	add namespace.cluster.ls command

### 💫 `vcsim` (Simulator)

- [bf80efab]	include stderr in log message when volume import fails
- [1f3fb17c]	include stderr in log message when container fails to start
- [e1c4b06e]	rewrite vmfs path from saved model
- [bcdfb298]	QueryConfigOptionEx Spec is optional
- [73e1af55]	support inventory updates in ContainerView
- [a76123b2]	set VirtualDevice.Connectable default for removable devices
- [b195dd57]	add AuthorizationManager methods
- [a71f6c77]	set VirtualDisk backing option defaults
- [fbde3866]	add CloneVApp_Task support
- [aae78223]	fix ListView.Modify
- [9cca13ab]	avoid ViewManager.ModifyListView race
- [156b1cb0]	add ListView to race test
- [55f6f952]	add mechanism for modeling methods
- [69942fe2]	fix save/load property collection for VmwareDistributedVirtualSwitch
- [33121b87]	Honoring the instance uuid provided in spec by caller ([#2052](https://github.com/vmware/govmomi/issues/2052))

### ⚠️ BREAKING

### 📖 Commits

- [1ec59a7c]	govc: fix build.sh git tag injection
- [164b9217]	Update docs for 0.24 release
- [bf80efab]	vcsim: include stderr in log message when volume import fails
- [4080e177]	Add batch APIs for multiple tags to object
- [31c0836e]	govc: add cluster.usage command
- [7178588c]	examples: add Folder.CreateVM
- [2b962f3f]	Add test for vsan host config
- [165d7cb4]	Add function to get vsan host config
- [79514c81]	govc: add volume.ls -ds option
- [f7ff79df]	Add Configure ACL go bindings
- [1f3fb17c]	vcsim: include stderr in log message when container fails to start
- [3b83040a]	Add wrappers for retrieving vsan properties
- [12e8969c]	Use gofmt
- [6454dbd4]	Add vSAN 7.0 API bindings
- [6a216a52]	Add vSAN 7.0 API bindings
- [be15ad6c]	Regenerate against vSphere 7.0U1 release
- [5e57b3f6]	govc: add device.boot -firmware option
- [e1c4b06e]	vcsim: rewrite vmfs path from saved model
- [26635452]	Change CnsCreateVolume to return PlacementResult for static volume provisioning. Also add unit test for this case.
- [4d82f0ff]	govc: add dvs.portgroup.{add,change} '-auto-expand' flag
- [bcdfb298]	vcsim: QueryConfigOptionEx Spec is optional
- [8b194c23]	Add Placement object in CNS CreateVolume response. Add corresponding test.
- [b085fc33]	Use available ctx in enable cluster network lookup
- [f6f336ab]	Cleanup some redundant code for cluster namespace enabling
- [d04f2b49]	change negative one to rand neg int32
- [f819befd]	go binding for CNS RelocateVolume API
- [ed93ea7d]	fix the goimports validation error
- [f402c0e1]	support trunk mode port group
- [ff575977]	change key default from -1 to rand neg int32 vsphere 7 introduced a key collision detection error when adding devices com.vmware.vim.vpxd.vmprov.duplicateDeviceKey which causes -1 keys to return an error of duplicate if you try and add two devices in the same AddDevice call
- [39acef43]	Add option to disable secure cookies with non-TLS endpoints
- [ae19e30f]	simulator: fix container vm example
- [73e1af55]	vcsim: support inventory updates in ContainerView
- [593cd20d]	Add namespace.cluster.disable cmd + formatting fixes
- [782ed95c]	Add namespace.cluster.enable cmd to govc
- [e7403032]	Make ListStorageProfiles public -> for enabling clusters in govc
- [53965796]	Adds support for enabling cluster namespaces via API
- [4a1d05ac]	govc: fix object.collect ContainerView updates
- [e84d0d18]	govc: document vm.disk.attach -link behavior
- [a76123b2]	vcsim: set VirtualDevice.Connectable default for removable devices
- [b4f7243b]	examples: add ContainerView retrieve clusters
- [b195dd57]	vcsim: add AuthorizationManager methods
- [a71f6c77]	vcsim: set VirtualDisk backing option defaults
- [1d21fff9]	examples: use session.Cache
- [8af8cef6]	examples: add events
- [3e2a8071]	Add ClusterDistribution field for CNS telemetry and Drop optional fields not known to the prior releases
- [4acfb726]	Fix for fatal error: concurrent map iteration and map write
- [01610887]	Adding VsanQueryObjectIdentities and QueryVsanObjects
- [fbde3866]	vcsim: add CloneVApp_Task support
- [70a9ced4]	govc: fix vm.clone panic when target VM already exists
- [a97e6168]	govc: support sparse backing in vm.disk.change
- [3380cd30]	govc: add CNS volume ls and rm commands
- [f9d7bfdf]	sts: fix SignRequest bodyhash for non-empty request body
- [7b4e997b]	vapi: add WCP support bundle bindings
- [aae78223]	vcsim: fix ListView.Modify
- [0e4bce43]	Add AuthorizationManager.HasUserPrivilegeOnEntities wrapper
- [81207eab]	vim25/xml: sync with Go 1.15 encoding/xml
- [f7170fd2]	govc: add find -p flag
- [d49123c9]	Add internal.InventoryPath helper
- [b40cdd8a]	govc: add storage.policy commands
- [0c5cdd5d]	add / remove pci passthrough device for one VM
- [d0111d28]	govc: add vm.console -wss flag
- [94bc8497]	Add sms generated types and methods
- [e153061f]	examples: fix simulator.RunContainer on MacOSX
- [99fe9954]	finder: simplify direct use of InventoryPath func
- [3760bd6c]	Added Instant Clone feature Resolves: [#1392](https://github.com/vmware/govmomi/issues/1392)
- [86374ea2]	govc: support multi value flags in host.esxcli command
- [9cca13ab]	vcsim: avoid ViewManager.ModifyListView race
- [156b1cb0]	vcsim: add ListView to race test
- [f903d5da]	Add ExtendDisk and InflateDisk wrappers to vlsm/object_manager
- [073cc310]	Add AttachDisk and DetachDisk wrappers for the virtualMachine object.
- [a0c7e829]	vapi: add tags.Manager.GetAttachedTagsOnObjects example
- [378a24c4]	Vsan Performance Data Collection API ([#2021](https://github.com/vmware/govmomi/issues/2021))
- [55f6f952]	vcsim: add mechanism for modeling methods
- [69942fe2]	vcsim: fix save/load property collection for VmwareDistributedVirtualSwitch
- [fe3becfa]	bats: test fixes for running on MacOSX
- [0422a070]	Merge branch 'master' into pc/HardwareInfoNotReplicatingInCloning
- [9f12aae4]	vapi: add Content Library example
- [33121b87]	vcsim: Honoring the instance uuid provided in spec by caller ([#2052](https://github.com/vmware/govmomi/issues/2052))
- [9a07942b]	Setting hardware properties in clone VM spec from template VM
- [ebcfa3d4]	govc: add namespace.cluster.ls command
- [11d45e54]	vapi: add namespace management client and vcsim support
- [cdc44d5e]	vapi: add helper support "/api" endpoint

<a name="v0.23.1"></a>
## [Release v0.23.1](https://github.com/vmware/govmomi/compare/v0.23.0...v0.23.1)

> Release Date: 2020-07-02

### 💡 Examples

- [0bbb6a7d]	add property.Collector.Retrieve example

### 💫 `vcsim` (Simulator)

- [0697d33f]	add HostNetworkSystem.QueryNetworkHint
- [d7f4bba6]	use HostNetworkSystem wrapper with -load flag
- [916b12e6]	set HostSystem IP in cluster AddHost_Task
- [e63ec002]	add PbmQueryAssociatedProfile method

### ⚠️ BREAKING

### 📖 Commits

- [b7add48c]	check if config isn't nil before returning an uuid
- [12955a6c]	added support for returning array of BaseCnsVolumeOperationResult for QueryVolumeInfo API
- [0697d33f]	vcsim: add HostNetworkSystem.QueryNetworkHint
- [a5c9e1f0]	Merge branch 'master' into master
- [c14e3bc5]	adding in link to OPS
- [d7f4bba6]	vcsim: use HostNetworkSystem wrapper with -load flag
- [916b12e6]	vcsim: set HostSystem IP in cluster AddHost_Task
- [e63ec002]	vcsim: add PbmQueryAssociatedProfile method
- [0bbb6a7d]	examples: add property.Collector.Retrieve example

<a name="v0.23.0"></a>
## [Release v0.23.0](https://github.com/vmware/govmomi/compare/v0.22.2...v0.23.0)

> Release Date: 2020-06-11

### 💡 Examples

- [0e4b487e]	Fixed error is not logging in example.go
- [c17eb769]	add ContainerView.Find

### 💫 `govc` (CLI)

- [10c22fd1]	support raw object references in import.ova NetworkMapping
- [4f19eb6d]	ipath search flag does not require a Datacenter
- [414c548d]	support find with -customValue filter
- [0bf0e761]	support VirtualApp with -pool flag
- [f1ae45f5]	add -version flag to datastore.create command
- [43e4f8c2]	add session.login -X flag
- [70b7e1b4]	vm.clone ResourcePool is optional when -cluster is specified
- [2c5ff385]	add REST support for session.login -cookie flag
- [7d66cf9a]	fix host.info CPU usage
- [244a8369]	add session.ls -r flag
- [6c68ccf2]	add a VM template clone example
- [bb6ae4ab]	ignore ManagedObjectNotFound errors in 'find' command
- [210541fe]	remove ClientFlag.WithRestClient
- [75e9e80d]	do not try to start a VM template
- [667e6fbe]	add guest directory upload/download examples
- [167f5d83]	add vm.change -uuid flag
- [bcd06cee]	enable library.checkout and library.checkin by default
- [6f087ded]	avoid truncation in object.collect
- [e9bb4772]	add import.spec support for remote URLs
- [692c1008]	support optional compute.policy.ls argument
- [814e4e5c]	add vm.change '-memory-pin' flag
- [56e878a5]	support nested groups in sso.group.update
- [84346733]	add content library helpers
- [0ccfd912]	add cluster.group.ls -l flag
- [ae84c494]	use OutputFlag for import.spec
- [2dda4daa]	add library.clone -ovf flag
- [519d302d]	fix doc for -g flag (guest id) choices
- [e582cbd1]	add object.collect -o flag
- [d2e6b7df]	output formatting enhancements
- [e64c2423]	add find -l flag
- [4db4430c]	save sessions using sha256 ID

### 💫 `vcsim` (Simulator)

- [c3fe4f84]	CreateSnapshotTask now returns moref in result
- [b0af443c]	add lookup ServiceRegistration example
- [34734712]	add AuthorizationManager.HasPrivilegeOnEntities
- [228e0a8f]	traverse configManager.datastoreSystem in object.save
- [8acac02a]	traverse configManager.virtualNicManager in object.save
- [8a4ab564]	traverse configManager.networkSystem in object.save
- [4b8a5988]	add extraConfigAlias table
- [a0fe825a]	add EventHistoryCollector.ResetCollector implementation
- [558747b3]	fixes for PowerCLI
- [9ae04495]	apply ExtraConfig after devices
- [4286d7cd]	add another test/example for DVS host member validation
- [7e24bfcb]	validate DVS membership
- [853656fd]	fix flaky library subscriber test
- [7426e2fd]	avoid panic if ovf:capacityAllocationUnits is not present
- [55599668]	support QueryConfigOptionEx GuestId param
- [67d593cc]	VM templates do not have a ResourcePool
- [469e11b9]	validate session key in TerminateSession method
- [88d298ff]	unique MAC address for VM NICs
- [c4f820dd]	create vmdk directory if needed
- [488205f0]	support VMs with the same name
- [68349a27]	support Folder in RelocateVM spec
- [ab1298d5]	add guest operations support
- [7ffb9255]	add HostStorageSystem support
- [77b31b84]	avoid possible panic in UnregisterVM_Task
- [617c18e7]	support tags with the same name
- [dfcf9437]	add docs on generated inventory names
- [4cfc2905]	add support for NSX backed networks

### ⚠️ BREAKING

### 📖 Commits

- [b639ab4c]	Update docs for 0.23 release
- [be7742f2]	vapi: use header authentication in file Upload/Download
- [50846878]	provided examples for vm.clone and host.esxcli
- [aa97c4d3]	Add appliance log forwarding config handler and govc verb ([#1994](https://github.com/vmware/govmomi/issues/1994))
- [7cdad997]	Finder: support DistributedVirtualSwitch traversal
- [10c22fd1]	govc: support raw object references in import.ova NetworkMapping
- [c3fe4f84]	vcsim: CreateSnapshotTask now returns moref in result
- [4f19eb6d]	govc: ipath search flag does not require a Datacenter
- [b0af443c]	vcsim: add lookup ServiceRegistration example
- [84f1b733]	simulator: fix handling of nil Reference in container walk
- [b5b434b0]	Adding sunProfileName in pbm.CapabilityProfileCreateSpec
- [2111324a]	providing examples for govc guest.run
- [0eef3b29]	Bump to vSphere version 7
- [b277903e]	go binding for CNS QueryVolumeInfo API
- [a048ea52]	Move simulator lookupservice registration into ServiceInstance
- [30f1a71a]	modify markdown link at simulator.Model
- [7881f541]	Add REST session keep alive support
- [3aa9aaba]	vapi: sync access to rest.Client.SessionID
- [0a53ac4b]	simulator: refactor folder children operations
- [b9152f85]	simulator: relax ResourcePool constraint for createVM operation
- [70e9d821]	simulator: relax typing condition on RP parent
- [502b7efa]	simulator: relax ViewManager typing constraints
- [634fdde1]	simulator: remove data race in VM creation flow
- [6eda0169]	simulator: protect datastore freespace updates against data races
- [414c548d]	govc: support find with -customValue filter
- [487ca0d6]	Add logic to return default HealthStatus in CnsCreateVolume.
- [0bf0e761]	govc: support VirtualApp with -pool flag
- [f1ae45f5]	govc: add -version flag to datastore.create command
- [d0751307]	Add support for attach-tag-to-multiple-objects
- [5682b1f2]	simulator: relax excessive type assertions in SearchIndex
- [39a4da90]	Modify parenthesis for markdown link
- [34734712]	vcsim: add AuthorizationManager.HasPrivilegeOnEntities
- [92d464b9]	1. Add retry for CNS Create API with backing disk url 2. Fix binding for CnsAlreadyRegisteredFault
- [235582fe]	Add sample test for Create CNS API using backing disk Url path
- [b187863a]	1. Add BackingDiskUrlPath and CnsAlreadyFault go bindings to CNS APIs 2. Update CreateVolume CNS Util  to include BackingDiskUrlPath
- [409279fa]	Add GetProfileNameByID functionality to PBM
- [228e0a8f]	vcsim: traverse configManager.datastoreSystem in object.save
- [8acac02a]	vcsim: traverse configManager.virtualNicManager in object.save
- [8a4ab564]	vcsim: traverse configManager.networkSystem in object.save
- [43e4f8c2]	govc: add session.login -X flag
- [70b7e1b4]	govc: vm.clone ResourcePool is optional when -cluster is specified
- [2c5ff385]	govc: add REST support for session.login -cookie flag
- [6ccaf303]	Add guest.FileManager.TransferURL test
- [03c7611e]	Avoid possible nil pointer dereference in guest TransferURL
- [44a78f96]	Fix delegated Holder-of-Key token signature
- [11b2aa1a]	Update to vSphere 7 APIs
- [4b8a5988]	vcsim: add extraConfigAlias table
- [a0fe825a]	vcsim: add EventHistoryCollector.ResetCollector implementation
- [558747b3]	vcsim: fixes for PowerCLI
- [9ae04495]	vcsim: apply ExtraConfig after devices
- [7d66cf9a]	govc: fix host.info CPU usage
- [4286d7cd]	vcsim: add another test/example for DVS host member validation
- [515621d1]	Revert to using sha1 for session cache file names
- [f103a87a]	Default to separate session cache directories
- [7e24bfcb]	vcsim: validate DVS membership
- [244a8369]	govc: add session.ls -r flag
- [6c68ccf2]	govc: add a VM template clone example
- [bb6ae4ab]	govc: ignore ManagedObjectNotFound errors in 'find' command
- [853656fd]	vcsim: fix flaky library subscriber test
- [571f64e7]	Fix existing goimport issue
- [7426e2fd]	vcsim: avoid panic if ovf:capacityAllocationUnits is not present
- [9e57f983]	Add non-null HostLicensableResourceInfo to HostSystem
- [210541fe]	govc: remove ClientFlag.WithRestClient
- [75e9e80d]	govc: do not try to start a VM template
- [d9220e5d]	simulator: add interface for VirtualDiskManager
- [55599668]	vcsim: support QueryConfigOptionEx GuestId param
- [67d593cc]	vcsim: VM templates do not have a ResourcePool
- [667e6fbe]	govc: add guest directory upload/download examples
- [167f5d83]	govc: add vm.change -uuid flag
- [bcd06cee]	govc: enable library.checkout and library.checkin by default
- [9d4faa6d]	Refactor govc session persistence into session/cache package
- [6f087ded]	govc: avoid truncation in object.collect
- [7a1fef65]	Remove Task from function names in Task struct receiver methods
- [dd839655]	Add SetTaskState SetTaskDescription UpdateProgress to object package
- [469e11b9]	vcsim: validate session key in TerminateSession method
- [af41ae09]	Revert compute policy support
- [ad612b3e]	Fix the types of errors returned from VSLM tasks to be their originl vim faults rather than just wrappers of localized error msg
- [9e82230f]	Remove extra err check
- [e9bb4772]	govc: add import.spec support for remote URLs
- [273aaf71]	skip tests when env is not set
- [159c423c]	removing usage of spew package
- [76caec95]	vapi: prefer header authn to cookie authn
- [6c04cfa0]	Dropping fields in entity metadata for 6.7u3
- [8d15081f]	using right version and namespace from sdk/vsanServiceVersions.xml for cns client. making cns/client.go backward compatible to vsan67u3 by dropping unknown elements
- [8dfb29f5]	Add nil check for taskInfo result before typecasting CnsVolumeOperationBatchResult
- [d68bbf9b]	fixing CnsFault go binding
- [5482bd07]	syncing vmodl changes
- [3bcace84]	fixing go binding for CnsVolumeOperationResult and CnsFault
- [3c756cbd]	Fixing govmomi binding for CNS as per latest VMODL for CnsVsanFileShareBackingDetails. Also fixed cns/client_test.go accordingly.
- [4254df70]	Adding new API to get cluster configuration
- [0eacb4ed]	removing space before omitempty tag
- [59ce7e4a]	Resolve bug in Simulator regarding BackingObjectDetails
- [6ad7e87d]	Change the backingObjectDetails attribute to point to interface BaseCnsBackingObjectDetails
- [601f1ded]	Add resize support
- [56049aa4]	Updating go binding for vsan fileshare vmodl updates
- [af798c01]	Add CnsQuerySelectionNameType and CnsKubernetesEntityType back
- [af2723fd]	Add bindings for vSANFS and extend CNS bindings to support file volume
- [4e7b9b00]	update taskClientVersion for vsphere 7.0
- [692c1008]	govc: support optional compute.policy.ls argument
- [a7d4a77d]	Modified return type for Get policy
- [4007484e]	Compute Policy support
- [88d298ff]	vcsim: unique MAC address for VM NICs
- [814e4e5c]	govc: add vm.change '-memory-pin' flag
- [de8bcf25]	reset all for recursive calls fix format error
- [57efe91f]	Fixed ContainerView.RetrieveWithFilter fetch all specs if empty list of properties given
- [5af5ac8d]	Avoid possible panic in Filter.MatchProperty
- [85889777]	Add vAPI create binding for compute policy
- [56e878a5]	govc: support nested groups in sso.group.update
- [6f46ef8a]	Added prefix toggle parameter to govc export.ovf
- [6d3196e4]	Disk mode should override default value in vm.disk.attach
- [4be7a425]	Replaced ClassOvfParams with ClassDeploymentOptionParams
- [c4f820dd]	vcsim: create vmdk directory if needed
- [1ab6fe09]	Add Content Library subscriptions support
- [488205f0]	vcsim: support VMs with the same name
- [68349a27]	vcsim: support Folder in RelocateVM spec
- [6a6a7875]	Update CONTRIBUTING to have more info about running CI tests, checks.
- [a73c0d4f]	Expose Soap client default transport (a.k.a. its http client default transport)
- [84346733]	govc: add content library helpers
- [a225a002]	build(deps): bump nokogiri from 1.10.4 to 1.10.8 in /gen
- [b4395d65]	Avoid ServiceContent requirement in lookup.NewClient
- [c1e828cb]	fix blog links
- [863430ba]	toolbox: bump test VM memory for current CoreOS release
- [0ccfd912]	govc: add cluster.group.ls -l flag
- [1af6ec1d]	Add Namespace support to UseServiceVersion
- [ab1298d5]	vcsim: add guest operations support
- [0e4b487e]	examples: Fixed error is not logging in example.go
- [f36e13fc]	Add Content Library item copy support
- [7ffb9255]	vcsim: add HostStorageSystem support
- [ae84c494]	govc: use OutputFlag for import.spec
- [2dda4daa]	govc: add library.clone -ovf flag
- [77b31b84]	vcsim: avoid possible panic in UnregisterVM_Task
- [519d302d]	govc: fix doc for -g flag (guest id) choices
- [617c18e7]	vcsim: support tags with the same name
- [e582cbd1]	govc: add object.collect -o flag
- [0c6eafc1]	Apply gomvomi vim25/xml changes
- [4da54375]	Simplify ObjectName method
- [d2e6b7df]	govc: output formatting enhancements
- [dfcf9437]	vcsim: add docs on generated inventory names
- [e64c2423]	govc: add find -l flag
- [4db4430c]	govc: save sessions using sha256 ID
- [4cfc2905]	vcsim: add support for NSX backed networks
- [c17eb769]	examples: add ContainerView.Find
- [36056ae6]	Import golang/go/src/encoding/xml v1.13.6
- [346cf59a]	Avoid encoding/xml import
- [9cbe57db]	fix simulator disk manager fault message.
- [7f685c23]	Add permissions for NoCryptoAdmin

<a name="v0.22.2"></a>
## [Release v0.22.2](https://github.com/vmware/govmomi/compare/v0.22.1...v0.22.2)

> Release Date: 2020-02-13

### ⚠️ BREAKING

### 📖 Commits

- [e7df0c11]	Avoid ServiceContent requirement in lookup.NewClient

<a name="v0.22.1"></a>
## [Release v0.22.1](https://github.com/vmware/govmomi/compare/v0.22.0...v0.22.1)

> Release Date: 2020-01-13

### ⚠️ BREAKING

### 📖 Commits

- [da368950]	Release version 0.22.1
- [a62b12cf]	Fix AttributeValue.C14N for 6.7u3
- [c3d102b1]	Add finder example for MultipleFoundError
- [802e5899]	vapi: add CreateTag example
- [15630b90]	vapi: Add cluster modules client and simulator

<a name="v0.22.0"></a>
## [Release v0.22.0](https://github.com/vmware/govmomi/compare/v0.21.0...v0.22.0)

> Release Date: 2020-01-10

### 💡 Examples

- [72b1cd92]	output VM names in performance example
- [f4b3cda7]	add Common.Rename
- [dab4ab0d]	add VirtualMachine.Customize
- [1828eee9]	add VirtualMachine.CreateSnapshot
- [6ff7040e]	fix flag parsing
- [cad9a8e2]	add ExampleVirtualMachine_Reconfigure

### 💫 `govc` (CLI)

- [aed39212]	guest -i flag only applies to ProcessManager
- [704b335f]	add 5.0 to vm.create hardware version map
- [965109ae]	guest.run improvements
- [ee28fcfd]	add vm.customize multiple IP support
- [68b3ea9f]	fix library.info output formatting
- [5bb7f391]	add optional library.info details
- [d8ac7e51]	handle xsd:string responses
- [31d3e357]	add library.info details
- [182c84a3]	fixup tasks formatting
- [08fb2b02]	remove guest.run toolbox dependency
- [a727283f]	default to simple esxcli format when hints fields is empty
- [204af3c5]	add datacenter create/delete examples
- [f6c57ee7]	fix vm.create doc regarding -on flag
- [8debfcc3]	add device.boot -secure flag
- [2bb2a6ed]	add doc on vm.info -r flag
- [e50368c6]	avoid env for -cluster placement flag
- [f16eb276]	add default library.create thumbprint
- [d8325f34]	add thumbprint flag to library.create
- [0bad2bc2]	add vm.power doc
- [45d322ea]	support vm.customize without a managed spec
- [0a058e0f]	fixup usage suggestions
- [3185f7bc]	add vm.customize command
- [1b159e27]	fix datacenter.info against nested folders
- [149ba7ad]	add vm.change -latency flag
- [c35a532d]	validate moref argument
- [3fb02b52]	add guest.df command

### 💫 `vcsim` (Simulator)

- [198b97ca]	propagate VirtualMachineCloneSpec.Template
- [168a6a04]	add -trace-file option
- [32eeeb24]	Get IP address on non-default container network
- [1427d581]	avoid possible panic in VirtualMachine.Destroy_Task
- [067d58be]	automatically set Context.Caller
- [9e8e9a5a]	remove container volumes
- [6cc814b8]	bind mount BIOS UUID DMI files
- [9aec1386]	validate VirtualDisk UnitNumber
- [d7e43b4e]	add Floppy Drive support to OVF manager
- [8646dace]	properly initialize portgroup portKeys field
- [286bd5e9]	add vim25 client helper to vapi simulator
- [c3163247]	use VMX_ prefix for guestinfo env vars
- [a3a09c04]	don't allow duplicate names for Folder/StoragePod
- [a0a2296e]	pass guestinfo vars as env vars to container vms
- [903fe182]	add CustomizationSpecManager support
- [eda6bf3b]	simplify container vm arguments input
- [0ce9b0a1]	update docs
- [7755fbda]	add record/playback functionality
- [fe000674]	add VirtualMachine.Rename_Task support
- [d87cd5ac]	add feature examples
- [2cc33fa8]	Ensure that extraConfig from clone spec is added to VM being cloned
- [70ad060e]	use exported response helpers in vapi/simulator
- [1e7aa6c2]	avoid ViewManager.ViewList
- [9b0db1c2]	avoid race in ViewManager
- [28b5fc6c]	use TLS in simulator.Run
- [f962095f]	rename Example to Run
- [43d69860]	add endpoint registration mechanism
- [c183577b]	add PlaceVm support ([#1589](https://github.com/vmware/govmomi/issues/1589))
- [b17f3a51]	DefaultDatastoreID is optional in library deploy

### ⏮ Reverts

- [7914609d]	gen: retain omitempty field tag with int pointer types

### ⚠️ BREAKING

### 📖 Commits

- [317707be]	Update docs for 0.22 release
- [aed39212]	govc: guest -i flag only applies to ProcessManager
- [22308123]	Clarify DVS EthernetCardBackingInfo error message
- [a1c98f14]	Add Content Library synchronization support
- [704b335f]	govc: add 5.0 to vm.create hardware version map
- [4e907d99]	Clarify System.Read privilege requirement for PortGroup backing
- [554d9284]	Fix guest.FileManager.TransferURL cache
- [9b8da88a]	Remove toolbox specific guest run implementation
- [965109ae]	govc: guest.run improvements
- [ee28fcfd]	govc: add vm.customize multiple IP support
- [40001828]	Add OVF properties to library deploy ([#1755](https://github.com/vmware/govmomi/issues/1755))
- [68b3ea9f]	govc: fix library.info output formatting
- [198b97ca]	vcsim: propagate VirtualMachineCloneSpec.Template
- [5bb7f391]	govc: add optional library.info details
- [2509e907]	Added the missing RetrieveSnapshotDetails API in VSLM ([#1763](https://github.com/vmware/govmomi/issues/1763))
- [d8ac7e51]	govc: handle xsd:string responses
- [45b3685d]	Add library ItemType constants
- [f3e2c3ce]	Add retry support for HTTP status codes
- [31d3e357]	govc: add library.info details
- [182c84a3]	govc: fixup tasks formatting
- [08fb2b02]	govc: remove guest.run toolbox dependency
- [b10bcbf3]	VSLM: fixed the missing param in the QueryChangedDiskArea API impl
- [168a6a04]	vcsim: add -trace-file option
- [72b1cd92]	examples: output VM names in performance example
- [32eeeb24]	vcsim: Get IP address on non-default container network
- [f9f69237]	Move to cs.identity service type for sso admin endpoint
- [1427d581]	vcsim: avoid possible panic in VirtualMachine.Destroy_Task
- [067d58be]	vcsim: automatically set Context.Caller
- [a727283f]	govc: default to simple esxcli format when hints fields is empty
- [08adb5d6]	Move to cs.identity service type for sts endpoint
- [9e8e9a5a]	vcsim: remove container volumes
- [6cc814b8]	vcsim: bind mount BIOS UUID DMI files
- [e793289c]	Content Library: add CheckOuts support
- [66c9b10c]	Content Library: VM Template support
- [f4b3cda7]	examples: add Common.Rename
- [19a726f7]	Pass vm.Config.Uuid into the "VM" container via an env var
- [204af3c5]	govc: add datacenter create/delete examples
- [dab4ab0d]	examples: add VirtualMachine.Customize
- [f6c57ee7]	govc: fix vm.create doc regarding -on flag
- [8debfcc3]	govc: add device.boot -secure flag
- [9aec1386]	vcsim: validate VirtualDisk UnitNumber
- [7914609d]	Revert "gen: retain omitempty field tag with int pointer types"
- [9b2c5cc6]	Add CustomizationSpecManager.Info method and example
- [d7e43b4e]	vcsim: add Floppy Drive support to OVF manager
- [0bf21ec2]	Implement some missing methods ("*All*" variants) on SearchIndex MOB
- [2bb2a6ed]	govc: add doc on vm.info -r flag
- [8646dace]	vcsim: properly initialize portgroup portKeys field
- [e50368c6]	govc: avoid env for -cluster placement flag
- [91b1e0a7]	Add ability to set DVS discovery protocol on create and change
- [1e130141]	Move to Go 1.13
- [f16eb276]	govc: add default library.create thumbprint
- [d8325f34]	govc: add thumbprint flag to library.create
- [62c20113]	Fix hostsystem ManagementIPs call
- [c4a3908f]	Update DVS change to use finder.Network for a single object
- [ee6fe09d]	Fix usage instructions
- [5e6f5e3f]	gen: retain omitempty field tag with int pointer types
- [286bd5e9]	vcsim: add vim25 client helper to vapi simulator
- [841386f1]	Add ability to change a vnic on a host
- [391dd80b]	Add ability to change the MTU on a DVS that has already been created
- [26a45d61]	Change MTU param to use flags.NewInt32 as the type
- [dbcfc3a8]	Add MTU flag for DVS creation
- [0399353f]	Generate pointer type for ResourceReductionToToleratePercent
- [3f6b8ef5]	Add nil checks for all HostConfigManager references
- [c3163247]	vcsim: use VMX_ prefix for guestinfo env vars
- [5381f171]	Add option to follow all struct fields in mo.References
- [04e4835c]	Refactor session KeepAlive tests to use vcsim
- [7391c241]	Avoid possible deadlock in KeepAliveHandler
- [41422ea4]	build(deps): bump nokogiri from 1.6.3.1 to 1.10.4 in /gen
- [a3a09c04]	vcsim: don't allow duplicate names for Folder/StoragePod
- [4c72d2e9]	Add a method to update ports on a distributed virtual switch
- [0bad2bc2]	govc: add vm.power doc
- [45d322ea]	govc: support vm.customize without a managed spec
- [0a058e0f]	govc: fixup usage suggestions
- [a0a2296e]	vcsim: pass guestinfo vars as env vars to container vms
- [903fe182]	vcsim: add CustomizationSpecManager support
- [eda6bf3b]	vcsim: simplify container vm arguments input
- [0ce9b0a1]	vcsim: update docs
- [c538d867]	adding managed obj type to table
- [3185f7bc]	govc: add vm.customize command
- [b2a7b47e]	Include object.save directory in output
- [e8281f87]	Initial support for hybrid Model.Load
- [7755fbda]	vcsim: add record/playback functionality
- [8a3fa4f2]	set stable vsan client version
- [9eaac5cb]	Avoid empty principal in HoK token request
- [4a8da68d]	Allow sending multiple characters through -c and name the keys
- [3e3d3515]	add simple command list filter
- [fe000674]	vcsim: add VirtualMachine.Rename_Task support
- [9166bbdb]	support two tags with the same name
- [344653c1]	added log type and password scrubber
- [d87cd5ac]	vcsim: add feature examples
- [30fc2225]	Report errors when cdrom.insert fails
- [a94f2d3a]	vslm: fix to throw errors on tasks that are completed with error state
- [37054f03]	added IsTemplate vm helper
- [d7aeb628]	Fix object.collect with moref argument
- [0765aa63]	add GetInventoryPath to NetworkReference interface
- [9fb975b0]	Fix description of vm.keystrokes
- [234aaf53]	vapi: support DeleteLibrary with subscribed libraries
- [2cc33fa8]	vcsim: Ensure that extraConfig from clone spec is added to VM being cloned
- [70ad060e]	vcsim: use exported response helpers in vapi/simulator
- [b069efc0]	vapi: refactor for external API implementations
- [1e7aa6c2]	vcsim: avoid ViewManager.ViewList
- [9b0db1c2]	vcsim: avoid race in ViewManager
- [bd298f43]	a failing testcase that triggers with -race test
- [03422dd2]	vapi: expand internal path constants
- [d296a5f8]	Support HoK tokens with Interactive Users
- [c6226542]	Fix error check in session.Secret
- [28b5fc6c]	vcsim: use TLS in simulator.Run
- [f9b4bb05]	Replace LoadRetrievePropertiesResponse with LoadObjectContent
- [d84679eb]	Add VirtualHardwareSection.StorageItem
- [a23a5cb1]	Check whether there's a NIC before updating guest.ipAddress
- [8a069c27]	Add interactiveSession flag
- [25526b21]	vm.keystrokes -s (Allow spaces)
- [1828eee9]	examples: add VirtualMachine.CreateSnapshot
- [ca3763e7]	vapi: return info with current session query
- [f962095f]	vcsim: rename Example to Run
- [43d69860]	vcsim: add endpoint registration mechanism
- [1b159e27]	govc: fix datacenter.info against nested folders
- [c183577b]	vcsim: add PlaceVm support ([#1589](https://github.com/vmware/govmomi/issues/1589))
- [3e71d6be]	Add ResourcePool.Owner method
- [b17f3a51]	vcsim: DefaultDatastoreID is optional in library deploy
- [68980704]	Update generated code to vSphere 6.7u3
- [7416741c]	Add VirtualMachine.QueryChangedDiskAreas().
- [8ef87890]	Content library: support library ID in Finder
- [e373feb8]	Add option to propagate MissingSet faults in property.WaitForUpdates
- [6ff7040e]	examples: fix flag parsing
- [149ba7ad]	govc: add vm.change -latency flag
- [c35a532d]	govc: validate moref argument
- [54df157b]	Add content library subscription support
- [b86466b7]	Fix deadlock for keep alive handlers that attempt log in
- [9ad64557]	CNS go bindings
- [9de3b854]	Add simulator.Model.Run example
- [4285b614]	Include url in Client.Download error
- [caf0b6b3]	vcsa: update to 6.7 U3
- [7ac56b64]	Update vcsim Readme.md
- [48ef35df]	Update README.md
- [a40837d8]	Use gnu xargs in bats tests on Darwin
- [51ad97e1]	Add FetchCapabilityMetadata method to Pbm client
- [d124bece]	Add v4 option to VirtualMachine.WaitForIP
- [a5a429c0]	Add support for the cis session get method
- [4513735f]	Don't limit library.Finder to local libraries
- [cad9a8e2]	examples: add ExampleVirtualMachine_Reconfigure
- [3fb02b52]	govc: add guest.df command

<a name="v0.21.0"></a>
## [Release v0.21.0](https://github.com/vmware/govmomi/compare/v0.20.3...v0.21.0)

> Release Date: 2019-07-24

### 💡 Examples

- [9495f0d8]	add CustomFieldManager.Set

### 💫 `govc` (CLI)

- [fa755779]	support library paths in tags.attach commands
- [2ddfb86b]	add datastore.info -H flag
- [b3adfff2]	add sso.group commands
- [b5372b0c]	host.vnic.info -json support
- [4c41c167]	add context to LoadX509KeyPair error
- [910dac72]	add vm.change hot-add options
- [746c314e]	change logs.download -default=false
- [05f946d4]	increase guest.ps -X poll interval
- [cc10a075]	add -options support to library.deploy
- [fe372923]	rename vcenter.deploy to library.deploy
- [436d7a04]	move library.item.update commands to library.session
- [e6514757]	consolidate library commands
- [f8249ded]	export Archive Path field
- [d2ab2782]	add vm.change vpmc-enabled flag
- [e7b801c6]	fix vm.change against templates
- [8a856429]	fix option.set for int32 type values
- [81391309]	add datastore.maintenance.{enter,exit} commands
- [18cb9142]	FCD workarounds
- [665affe5]	add datastore.cluster.info Description
- [7b7f2013]	add permission.remove -f flag

### 💫 `vcsim` (Simulator)

- [774f3800]	add support to override credentials
- [ecd7312b]	fix host uuid
- [c25c41c1]	use stable UUIDs for inventory objects
- [1345eeb8]	Press any key to exit
- [ee14bd3d]	Update NetworkInfo.Portgroup in simulator
- [5b5eaa70]	remove httptest.serve flag
- [20c1873e]	add library.deploy support
- [0b1ad552]	add ovf manager
- [6684016f]	fork httptest server package
- [48c1e0a5]	add content library support
- [8543ea4f]	set guest.toolsRunningStatus property

### ⚠️ BREAKING

### 📖 Commits

- [a0fef816]	Update docs for 0.21 release
- [a38f6e87]	Content library related cleanups
- [e4024e9c]	Fix library AddLibraryItemFileFromURI fingerprint
- [fa755779]	govc: support library paths in tags.attach commands
- [5e8cb495]	Fixed type bug in global_object_manager Task.QueryResult
- [4a67dc73]	govcsim: Support Default UplinkTeamingPolicy in DVSPG
- [9da2362d]	Added missing field in VslmExtendDisk_Task in ExtendDisk method
- [91377d77]	Add Juju to projects using govmomi
- [f9026a84]	VSLM FCD Global Object Manager client for 6.7U2+
- [9495f0d8]	examples: add CustomFieldManager.Set
- [bb170705]	govcsim: Create datastore as accessible
- [35d0b7d3]	Set the InventoryPath of the folder object in DefaultFolder ([#1515](https://github.com/vmware/govmomi/issues/1515))
- [2d13a357]	Add govmomi performance example
- [2ddfb86b]	govc: add datastore.info -H flag
- [55da29e5]	govcsim: Set datastore status as normal
- [600e9f7c]	Add various govmomi client examples
- [5cccd732]	Add http source support to library.import
- [99dd5947]	Goreleaser update for multiple archives
- [b3adfff2]	govc: add sso.group commands
- [5889d091]	tags API: add methods for association of multiple tags/objects
- [b5372b0c]	govc: host.vnic.info -json support
- [9b7688e0]	Add method that sets vim version to the endpoint service version
- [fe3488f5]	Fix tls config in soap.NewServiceClient
- [4c41c167]	govc: add context to LoadX509KeyPair error
- [d7430825]	Support external PSC lookup service
- [774f3800]	vcsim: add support to override credentials
- [47c9c070]	Fix HostNetworkSystem.QueryNetworkHint return value
- [910dac72]	govc: add vm.change hot-add options
- [4606125e]	Fix json request tracing
- [746c314e]	govc: change logs.download -default=false
- [05f946d4]	govc: increase guest.ps -X poll interval
- [77cb9df5]	Add library export support
- [cc10a075]	govc: add -options support to library.deploy
- [ecd7312b]	vcsim: fix host uuid
- [c25c41c1]	vcsim: use stable UUIDs for inventory objects
- [322d9629]	Fix pbm field type lookup
- [1345eeb8]	vcsim: Press any key to exit
- [a4f58ac6]	Update examples to use examples.Run method
- [a31db862]	Add permanager example
- [384b1b95]	Fix port signature in REST endpoint token auth
- [c222666f]	Default to running against vcsim in examples
- [199e737b]	Add generated vslm types and methods
- [ee14bd3d]	vcsim: Update NetworkInfo.Portgroup in simulator
- [dc631a2d]	Format import statement
- [f133c9e9]	Fix paths in vsan/methods
- [d8e7cc75]	Update copy rights
- [62412641]	Add vsan bindings
- [fc3f0e9d]	Support resignature of vmfs snapshots ([#1442](https://github.com/vmware/govmomi/issues/1442))
- [fe372923]	govc: rename vcenter.deploy to library.deploy
- [436d7a04]	govc: move library.item.update commands to library.session
- [e6514757]	govc: consolidate library commands
- [f8249ded]	govc: export Archive Path field
- [8a823c52]	vcsa: bump to 6.7u2
- [5b5eaa70]	vcsim: remove httptest.serve flag
- [466dc5b2]	Update to vSphere 6.7u2 API
- [e9f80882]	Add error check to VirtualMachine.WaitForNetIP
- [5611aaa2]	Add ovftool support
- [20c1873e]	vcsim: add library.deploy support
- [0b1ad552]	vcsim: add ovf manager
- [d2ab2782]	govc: add vm.change vpmc-enabled flag
- [e7b801c6]	govc: fix vm.change against templates
- [8a856429]	govc: fix option.set for int32 type values
- [9155093e]	Typo and->an
- [81391309]	govc: add datastore.maintenance.{enter,exit} commands
- [1a857b94]	Add support to reconcile FCD datastore inventory
- [18cb9142]	govc: FCD workarounds
- [499a8828]	Fix staticcheck issues value of `XXX` is never used
- [665affe5]	govc: add datastore.cluster.info Description
- [546e8897]	Add error check for deferred functions
- [367c8743]	Fix bug with multiple tags in category
- [7b7f2013]	govc: add permission.remove -f flag
- [87bc0c85]	Makefile: Fix govet target using go1.12
- [791e5434]	travis.yml: Update from golang 1.11 to 1.12
- [a86a42a2]	travis.yml: Update from Ubuntu Trusty to Xenial
- [d92ee75e]	Report local Datastore back as type OTHER
- [6684016f]	vcsim: fork httptest server package
- [48c1e0a5]	vcsim: add content library support
- [69faa2de]	Make PostEvent TaskInfo param optional
- [608ad29f]	Omit namespace tag in generated method body response types
- [a7c03228]	Fix codespell issues
- [728e77db]	Fix a race in NewServer().
- [8543ea4f]	vcsim: set guest.toolsRunningStatus property
- [e3143407]	Fix elseif gocritic issues
- [89b53312]	Fix gocritic emptyStringTest issues
- [63ba9232]	Fix some trivial gocritic issues
- [0b8d0ee7]	simulator/host_datastore_browser.go: remove commented out code
- [6c17d66c]	Fix some staticcheck issues
- [d45b5f34]	Fix some gosimple issues
- [90e501a6]	Correct the year in the govc changelog
- [8082a261]	Update XDR to use fork
- [e94ec246]	govc/USAGE.md: Update documentation
- [3fde3319]	snapshot.tree: Show snapshots description
- [1d6f743b]	Fix year in changelog
- [39b2c871]	support customize vm folder in ovf deploy
- [3ad203d3]	Use rest.Client for library uploads
- [5d24c38c]	lib/finder: Support filenames with "/"
- [087f09f9]	govc library: use govc/flags for Datastore and ResourcePool
- [d1a7f491]	Remove nested progress.Tee usage
- [7312711e]	govc/vm/*: Fix some gosec Errors unhandled issues
- [88601bb7]	vcsim/*: Fix Errors unhandled issues
- [61d04b46]	session/*: Fix Errors unhandled issues
- [f9a22349]	vmdk/*: Fix gosec Errors unhandled issues
- [ca9b71a9]	Fix gosec Expect directory permissions to be 0750 or less issues
- [6083e891]	Fix gosec potential file inclusion via variable issues
- [38091bf8]	Build changes needed for content library
- [885d4b44]	Content library additions/finder
- [3fb72d1a]	Add support for content library
- [64f2a5ea]	Fix API Version check.
- [718331e3]	govc/*: Fix some staticcheck issues
- [ba7923ae]	Fix all staticcheck "error strings should not be capitalized" issues
- [ed32a917]	simulator/*: Fix some staticcheck issues
- [f71d4efb]	govc/vm/*: Fix staticcheck issues
- [3d77e2b1]	vim25/*: Fix staticcheck issues
- [d711005a]	.gitignore: add editor files *~
- [43ff04f1]	Fix [#1173](https://github.com/vmware/govmomi/issues/1173)
- [562aa0db]	Go Mod Support

<a name="v0.20.3"></a>
## [Release v0.20.3](https://github.com/vmware/govmomi/compare/v0.20.2...v0.20.3)

> Release Date: 2019-10-08

### ⚠️ BREAKING

### 📖 Commits

- [fdd27786]	Fix tls config in soap.NewServiceClient

<a name="v0.20.2"></a>
## [Release v0.20.2](https://github.com/vmware/govmomi/compare/v0.20.1...v0.20.2)

> Release Date: 2019-07-03

### ⚠️ BREAKING

### 📖 Commits

- [bd9cfd18]	Set the InventoryPath of the folder object in DefaultFolder ([#1515](https://github.com/vmware/govmomi/issues/1515))

<a name="v0.20.1"></a>
## [Release v0.20.1](https://github.com/vmware/govmomi/compare/v0.20.0...v0.20.1)

> Release Date: 2019-05-20

### ⚠️ BREAKING

### 📖 Commits

- [4514987f]	Fix port signature in REST endpoint token auth

<a name="v0.20.0"></a>
## [Release v0.20.0](https://github.com/vmware/govmomi/compare/v0.19.0...v0.20.0)

> Release Date: 2019-02-06

### 💫 `govc` (CLI)

- [308dbf99]	fix object.collect error for multiple objects with same path
- [4635c1cc]	add device name match support to device.ls and device.remove
- [c36eb50f]	add vm.disk.attach -mode flag
- [b234cdbc]	add category option to relevant tags commands
- [afe5f42d]	add vm.create -version option
- [b733db99]	fields.set can now add missing fields
- [cad627a6]	add fields.info command

### 💫 `vcsim` (Simulator)

- [957ef0f7]	require authentication in vapi simulator
- [32148187]	Resolve issue making device changes on clone (resolves [#1355](https://github.com/vmware/govmomi/issues/1355))
- [cbb4abc9]	fix SearchDatastore task info entity
- [2682c021]	add EnvironmentBrowser support
- [3b9a4c9f]	avoid zero IP address in GOVC_URL output
- [1921f73a]	avoid panic when event template is not defined
- [d79013aa]	implement RefreshStorageInfo method for virtual machine
- [69dfdd77]	configure HostSystem port
- [bba50b40]	datastore.upload now creates missing directories in destination path.
- [d2506759]	add option to run container as vm
- [47284860]	add SessionIsActive support
- [c5ee00bf]	fix fault detail encoding
- [1284300c]	support base types in property filter
- [25ae5c67]	PropertyCollector should not require PathSet
- [4f1c89e5]	allow '.' in vm name
- [b8c04142]	populate VM guest.net field
- [223b2a2a]	add SearchIndex FindByDnsName support
- [b26e10f0]	correct property update in RemoveSnapshotTask
- [693f3fb6]	update VM snapshot methods to change VM properties with UpdateObject
- [06e13bbe]	support setting vm fields via extraConfig
- [a4330365]	update VM configureDevices method to change VM properties with UpdateObject
- [5f8acb7a]	update VM device add operation - stricter key generation, new InvalidDeviceSpec condition
- [846ae27a]	add PBM support
- [d41d18aa]	put VM into registry earlier during CreateVM
- [89b4c2ce]	add datastore access check for vm host placement
- [f9f9938e]	add task_manager description property templates
- [9bb5bde2]	fix defaults when generating vmdk paths
- [0b650fd3]	fix custom_fields_manager test
- [588bc224]	replace HostSystem template IP with vcsim listen address
- [7066f8dc]	Change CustomFieldsManager SetField to use ctx.WithLock and add InvalidArgument fault check.
- [fe070811]	update DVS methods to use UpdateObject instead of setting fields directly
- [03939cce]	add vslm support
- [c02efc3d]	add setCustomValue support
- [94804159]	add fault message to PropertyCollector RetrieveProperties
- [36035f5b]	add HistoryCollector scrollable view support

### ⚠️ BREAKING

### 📖 Commits

- [da7af247]	Fix for govc/build.sh wrong dir
- [90a863be]	Update docs for 0.20 release
- [957ef0f7]	vcsim: require authentication in vapi simulator
- [32148187]	vcsim: Resolve issue making device changes on clone (resolves [#1355](https://github.com/vmware/govmomi/issues/1355))
- [a7563c4d]	Use path id for tag-association requests
- [cbb4abc9]	vcsim: fix SearchDatastore task info entity
- [2682c021]	vcsim: add EnvironmentBrowser support
- [3b9a4c9f]	vcsim: avoid zero IP address in GOVC_URL output
- [b261f25d]	Add 2x blog posts about vcsim
- [1921f73a]	vcsim: avoid panic when event template is not defined
- [308dbf99]	govc: fix object.collect error for multiple objects with same path
- [d79013aa]	vcsim: implement RefreshStorageInfo method for virtual machine
- [69dfdd77]	vcsim: configure HostSystem port
- [4f50681f]	Fix of the missing http body close under soap client upload
- [bba50b40]	vcsim: datastore.upload now creates missing directories in destination path.
- [8ac7c5a8]	Fixed 64-bit aligment issues with atomic counters
- [7ca12ea2]	fix device.info Write output
- [3a82237c]	device.ls -json doesn't work for now
- [86f4ba29]	ssoadmin:create local group and add users to group ([#1327](https://github.com/vmware/govmomi/issues/1327))
- [2d8ef2c6]	Format with latest version of goimports
- [4635c1cc]	govc: add device name match support to device.ls and device.remove
- [d7857a13]	Updated the examples for the correct format
- [71e19136]	Updated to reflect PR feedback
- [d2506759]	vcsim: add option to run container as vm
- [61b7fe3e]	Added string support
- [a72a4c42]	Initial Support for PutUsbScanCodes
- [47284860]	vcsim: add SessionIsActive support
- [c5ee00bf]	vcsim: fix fault detail encoding
- [aaf83275]	Summary of changes:  1. Changing the pbm client's path as java client is expecting /pbm.  2. Added PbmRetrieveServiceContent method in the unauthorized list.
- [c36eb50f]	govc: add vm.disk.attach -mode flag
- [1284300c]	vcsim: support base types in property filter
- [25ae5c67]	vcsim: PropertyCollector should not require PathSet
- [b234cdbc]	govc: add category option to relevant tags commands
- [138f30f8]	Makefiles for govc/vcsim; updates  govc/build.sh
- [4f1c89e5]	vcsim: allow '.' in vm name
- [afe5f42d]	govc: add vm.create -version option
- [b8c04142]	vcsim: populate VM guest.net field
- [223b2a2a]	vcsim: add SearchIndex FindByDnsName support
- [b26e10f0]	vcsim: correct property update in RemoveSnapshotTask
- [693f3fb6]	vcsim: update VM snapshot methods to change VM properties with UpdateObject
- [e5948f44]	build: Refactored Travis-CI to use containers
- [06e13bbe]	vcsim: support setting vm fields via extraConfig
- [651d4881]	Allow pointer values in mo.ApplyPropertyChange
- [546a7df6]	Tags support for First Class Disks
- [a4330365]	vcsim: update VM configureDevices method to change VM properties with UpdateObject
- [5f8acb7a]	vcsim: update VM device add operation - stricter key generation, new InvalidDeviceSpec condition
- [86375ceb]	Merge branch 'master' into fields-info
- [bf962f18]	Update govc/fields/add.go
- [98575e0c]	Update govc/fields/add.go
- [b733db99]	govc: fields.set can now add missing fields
- [cad627a6]	govc: add fields.info command
- [ed2a4cff]	vm.power: Make waiting for op completion optional
- [846ae27a]	vcsim: add PBM support
- [d41d18aa]	vcsim: put VM into registry earlier during CreateVM
- [1926071e]	Datastore Cluster placement support for First Class Disks
- [89b4c2ce]	vcsim: add datastore access check for vm host placement
- [f9f9938e]	vcsim: add task_manager description property templates
- [9bb5bde2]	vcsim: fix defaults when generating vmdk paths
- [0b650fd3]	vcsim: fix custom_fields_manager test
- [588bc224]	vcsim: replace HostSystem template IP with vcsim listen address
- [7066f8dc]	vcsim: Change CustomFieldsManager SetField to use ctx.WithLock and add InvalidArgument fault check.
- [ef517cae]	Display category name instead of ID in govc tags.info
- [d69c9787]	goimports updates
- [fe070811]	vcsim: update DVS methods to use UpdateObject instead of setting fields directly
- [03939cce]	vcsim: add vslm support
- [accb2863]	Add vslm package and govc disk commands
- [478ebae6]	[doc] add an example for cpu and memory hotplug
- [c02efc3d]	vcsim: add setCustomValue support
- [c3c79d16]	goimports updates
- [ce71b6c2]	vcsa: bump to 6.7.0 U1
- [94804159]	vcsim: add fault message to PropertyCollector RetrieveProperties
- [1ad0d87d]	Removed NewWithDelay (not needed anymore)
- [5900feef]	Updated documentation
- [5a87902b]	Added delay functionality
- [c0518fd2]	Add LoginByToken to session KeepAliveHandler
- [e0736431]	Update Ansible link in README
- [36035f5b]	vcsim: add HistoryCollector scrollable view support
- [bc2636fe]	Move govc tags rest.Client helper to ClientFlag
- [54a181af]	Add SSO support for vAPI
- [8817c27b]	replace * by client's host+port
- [ac898b50]	change hostname only if set to * and still set thumbprint
- [7a5cc6b7]	replace hostname only if unset

<a name="v0.19.0"></a>
## [Release v0.19.0](https://github.com/vmware/govmomi/compare/v0.18.0...v0.19.0)

> Release Date: 2018-09-30

### 💫 `govc` (CLI)

- [6b4a62b1]	fix test case for new cluster.rule.info command
- [1350eea6]	add new command cluster.rule.info

### 💫 `vcsim` (Simulator)

- [f3260968]	add dvpg networks to HostSystem.Parent
- [17352fce]	add support for tags API
- [c29d4b12]	Logout should not unregister PropertyCollector singleton
- [11fb0d58]	add ResetVM and SuspendVM support
- [39e6592d]	add support for PropertyCollector incremental updates
- [619fbe28]	do not include DVS in HostSystem.Network

### ⚠️ BREAKING

### 📖 Commits

- [3617f28d]	Update docs for 0.19 release
- [4316838a]	vcsa: bump to 6.7.0d
- [64d875b9]	Added PerformanceManager simulator
- [f3260968]	vcsim: add dvpg networks to HostSystem.Parent
- [862da065]	Allowing the use of STS for exchanging tokens
- [83ce863a]	Handle empty file name in import.spec
- [a99f702d]	Bump travis golang version from 1.10 to 1.11
- [e4e8e2d6]	Clean up unit test messaging
- [8e04e3c7]	Run goimports on go source files
- [2431ae00]	Add mailmap for bruceadowns
- [e805b4ea]	Updates per dep ensure integration
- [70589fb6]	Add ignore of intellij project settings directory
- [d114fa69]	Print action for dvs security groups
- [d458266a]	fix double err check
- [3f0e0aa7]	remove providerSummary cache
- [cf9c16c4]	Avoid use of Finder all param in govc
- [c4face4f]	Print DVS rules for dvportgroup
- [91a33dd4]	Finalize tags API
- [7d54bf9f]	README: Fix path to LICENSE.txt file
- [17352fce]	vcsim: add support for tags API
- [c29d4b12]	vcsim: Logout should not unregister PropertyCollector singleton
- [8bda0ee1]	Fix format in test
- [8be5207c]	Add test for WaitOption.MaxWaitSeconds == 0 behaviour in simulator
- [900e1a35]	Fix the WaitOption.MaxWaitSeconds == 0 behaviour in simulator
- [056ad0d4]	vcsa: bump to 6.7.0c release
- [6b4a62b1]	govc: fix test case for new cluster.rule.info command
- [1350eea6]	govc: add new command cluster.rule.info
- [a05cd4b0]	add output in cluster.rule.ls -name for ClusterVmHostRuleInfo and ClusterDependencyRuleInfo rules, add -l Option to cluster.rule.ls
- [11fb0d58]	vcsim: add ResetVM and SuspendVM support
- [3e6b2d6e]	Add ability to move multiple hosts into a cluster
- [e9f9920f]	Add method to move host into cluster
- [39e6592d]	vcsim: add support for PropertyCollector incremental updates
- [b7c270c6]	Add testing support for govc tags commands
- [619fbe28]	vcsim: do not include DVS in HostSystem.Network
- [6b6060dc]	show rule details for ClusterVmHostRuleInfo rules in cluster.rule.ls
- [0c28a25d]	Use govc find instead of ls to assign licenses
- [c1377063]	Only test with Go 1.10 on Travis CI
- [4cfadda5]	Avoid panic if fault detail is nil
- [d06874e1]	Upgrade for govc tags commands
- [fdfaec9c]	Better documentation for VirtualMachine.UUID
- [e1285a03]	Add UUID helper for VirtualMachine
- [919b728c]	Complete tags management APIs ([#1162](https://github.com/vmware/govmomi/issues/1162))
- [b3251638]	vcsa: bump to 6.7.0a release
- [a1fbb6ef]	Optionally check root CAs for validity ([#1154](https://github.com/vmware/govmomi/issues/1154))
- [add38bed]	Fixed govc host.info logical CPU count
- [1ddfb011]	Tags Categories cmd available  ([#1150](https://github.com/vmware/govmomi/issues/1150))
- [83ae35fb]	default MarkAsTemplate to false in import spec
- [49f0dea7]	add option to mark VM as template on OVX import
- [1f9e19f4]	example: uniform unit for host memory
- [4cfd1376]	fix example output.

<a name="v0.18.0"></a>
## [Release v0.18.0](https://github.com/vmware/govmomi/compare/v0.17.1...v0.18.0)

> Release Date: 2018-05-24

### 💫 `govc` (CLI)

- [b841ae01]	import.ovf pool flag should be optional if host is specified
- [f5c84b98]	avoid Login() attempt if username is not set
- [d91fcbf4]	add json support to find command
- [ba2d2323]	fix host.esxcli error handling

### 💫 `vcsim` (Simulator)

- [8a5438b0]	add STS simulator
- [c0337740]	use VirtualDisk CapacityInKB for device summary
- [3d7fbac2]	add property collector field type mapping for integer arrays

### ⚠️ BREAKING

### 📖 Commits

- [e4b69fab]	Update docs for 0.18 release
- [1dbfb317]	Bump versions
- [b841ae01]	govc: import.ovf pool flag should be optional if host is specified
- [96a905c1]	Add -sharing option to vm.disk.create and vm.disk.attach
- [4b4e2aaa]	Add VirtualDiskManager wrapper to set UUID
- [40a565b3]	adjust datastore size when vm is added or updated or deleted
- [7f6479ba]	update datastore capacity and free space when it is started
- [76dfefd3]	Avoid recursive root path search in govc find command
- [623c7fa9]	Change key name according to Datacenter object
- [24d0cf1b]	added check for `InstanceUuid` when `VmSearch` is true in `FindByUuid`
- [25fc474c]	Issue token if needed for govc sso commands
- [822fd1c0]	Fixed leading "/" requirement in FindByInventoryPath
- [59d9f6a0]	Add devbox scripts
- [fd45d81c]	Add -U option to sso.service.ls
- [f5c84b98]	govc: avoid Login() attempt if username is not set
- [8a5438b0]	vcsim: add STS simulator
- [93f7fbbd]	Fix govc vm.clone -annotation flag
- [bcff5383]	save CapacityInKB in thousand delimited format
- [db12d4cb]	Avoid possible panic in portgroup EthernetCardBackingInfo
- [d120efcb]	Add STS support for token renewal
- [76b1ceaf]	Add vmxnet2, pcnet32 and sriov to VirtualDeviceList.EthernetCardTypes
- [c0337740]	vcsim: use VirtualDisk CapacityInKB for device summary
- [3d7fbac2]	vcsim: add property collector field type mapping for integer arrays
- [42b30bb6]	Finder.DefaultHostSystem should find hosts in nested folders
- [b8323d6b]	Avoid property.Filter matching against unset properties
- [64788667]	Update to vSphere 6.7 API
- [d3ae3004]	Bump vCenter and ESXi builds to the latest release
- [098fc449]	Add ssoadmin client and commands
- [80a9c20e]	vm.Snapshot should be 'nil' instead of an empty 'vim.vm.SnapshotInfo' when there are no snapshots
- [1b1b428e]	added failing tests for when vm.Snapshot should / shouldn't be 'nil'
- [a34ab4ba]	Refactor LoginExtensionByCertificate tunnel usage
- [5b36033f]	Lookup Service support
- [3f07eb74]	add empty fields, but don't return them in the case of 'RetrievePropertiesEx'
- [05bdabe0]	added failing test case for issue 1061
- [903e8644]	SAML token authentication support
- [d91fcbf4]	govc: add json support to find command
- [ba2d2323]	govc: fix host.esxcli error handling
- [ff687746]	Dep Support
- [5f701460]	Add -firmware parameter to 'govc vm.create' with values bios|efi

<a name="v0.17.1"></a>
## [Release v0.17.1](https://github.com/vmware/govmomi/compare/v0.17.0...v0.17.1)

> Release Date: 2018-03-19

### 💫 `vcsim` (Simulator)

- [0502ee9b]	add Destroy method for Folder and Datacenter types
- [0636dc8c]	add EventManager.QueryEvents

### ⚠️ BREAKING

### 📖 Commits

- [123ed177]	govc release 0.17.1
- [24d88451]	Avoid possible panic in QueryVirtualDiskInfo
- [82129fb7]	Add goreleaser to automate release process
- [ce88b296]	Fix dvs.portgroup.info filtering
- [0502ee9b]	vcsim: add Destroy method for Folder and Datacenter types
- [1620160d]	In progress.Reader emit final report on EOF.
- [0636dc8c]	vcsim: add EventManager.QueryEvents

<a name="v0.17.0"></a>
## [Release v0.17.0](https://github.com/vmware/govmomi/compare/v0.16.0...v0.17.0)

> Release Date: 2018-02-28

### 💫 `govc` (CLI)

- [29498644]	fix vm.clone to use -net flag when source does not have a NIC
- [d12b8f25]	object.collect support for raw filters
- [6cb9fef8]	fix host.info CPU usage
- [5786e7d2]	add -cluster flag to license.assign command
- [d4ee331c]	allow columns in guest login password ([#972](https://github.com/vmware/govmomi/issues/972))

### 💫 `vcsim` (Simulator)

- [d2ba47d6]	add simulator.Datastore type
- [937998a1]	set VirtualMachine summary.config.instanceUuid
- [1c76c63d]	update HostSystem.Summary.Host reference
- [274f3d63]	add EventManager support
- [cc21a5ab]	stats related fixes
- [fa2bee10]	avoid data races
- [ca6f5d1d]	respect VirtualDeviceConfigSpec FileOperation
- [7811dfce]	avoid keeping the VM log file open
- [828ce5ec]	add UpdateOptions support
- [d03f38fa]	add session support
- [a3c9ed2b]	Add VM.MarkAsTemplate support
- [50735461]	more input spec honored in ReConfig VM
- [638d972b]	Initialize VM fields properly
- [aa0382c1]	Honor the input spec in ReConfig VM
- [42f9a133]	Add HostLocalAccountManager
- [76f376a3]	workaround xml ns issue with pyvsphere ([#958](https://github.com/vmware/govmomi/issues/958))
- [45c5269b]	add MakeDirectoryResponse ([#938](https://github.com/vmware/govmomi/issues/938))
- [b4e77bd2]	copy RoleList for AuthorizationManager ([#932](https://github.com/vmware/govmomi/issues/932))
- [2a8a5168]	apply vm spec NumCoresPerSocket ([#930](https://github.com/vmware/govmomi/issues/930))
- [3a61d85f]	Configure dvs with the dvs config spec
- [5f0f4004]	Add VirtualMachine guest ID validation ([#921](https://github.com/vmware/govmomi/issues/921))
- [ef571547]	add QueryVirtualDiskUuid ([#920](https://github.com/vmware/govmomi/issues/920))
- [27229ab7]	update ServiceContent to 6.5 ([#917](https://github.com/vmware/govmomi/issues/917))

### ⚠️ BREAKING

### 📖 Commits

- [1d63da8d]	govc release 0.17
- [3017acf8]	Print Table of Contents in usage.md Found good example of toc using markdown here: https://stackoverflow.com/a/448929/1572363
- [ce54fe2c]	Fix typo
- [201fc601]	Implement Destroy task for HostSystem
- [92ce4244]	Init PortKeys in DistributedVirtualPortgroup
- [795f2cc7]	Avoid json encoding error in Go 1.10
- [e805389e]	Add 'Type' field to device.info -json output
- [d622f149]	Use VirtualDiskManager in datastore cp and mv commands
- [f219bf3b]	object: Return correct helper object for OpaqueNetwork
- [29498644]	govc: fix vm.clone to use -net flag when source does not have a NIC
- [43c95b21]	Fix build on Windows
- [38124002]	Fix session persistence in session.login command
- [144bb1cf]	Add support for Datacenter.PowerOnMultiVM
- [d2ba47d6]	vcsim: add simulator.Datastore type
- [937998a1]	vcsim: set VirtualMachine summary.config.instanceUuid
- [1c76c63d]	vcsim: update HostSystem.Summary.Host reference
- [d12b8f25]	govc: object.collect support for raw filters
- [274f3d63]	vcsim: add EventManager support
- [cc21a5ab]	vcsim: stats related fixes
- [2d30cde3]	Fix broken datastore link in VM
- [54b160b2]	Several context changes:
- [f643f0ae]	Leverage contexts in http uploads
- [fa2bee10]	vcsim: avoid data races
- [29bd00ec]	Remove omitempty tag from AffinitySet field
- [ca6f5d1d]	vcsim: respect VirtualDeviceConfigSpec FileOperation
- [7811dfce]	vcsim: avoid keeping the VM log file open
- [6cb9fef8]	govc: fix host.info CPU usage
- [5786e7d2]	govc: add -cluster flag to license.assign command
- [63c86f29]	Add datastore.disk.cp command
- [828ce5ec]	vcsim: add UpdateOptions support
- [a13ad164]	Bump vcsa scripts to use 6.5U1 EP5
- [c447244d]	Add CloneSession support to govc and vcsim
- [d03f38fa]	vcsim: add session support
- [44e8d85e]	Added AttachScsiLun function ([#987](https://github.com/vmware/govmomi/issues/987))
- [a3c9ed2b]	vcsim: Add VM.MarkAsTemplate support
- [3f8349f3]	Add cluster vm override commands ([#977](https://github.com/vmware/govmomi/issues/977))
- [91fbd1f7]	Add option to filter events by type ([#976](https://github.com/vmware/govmomi/issues/976))
- [1d8b92d9]	User server clock in session.ls ([#973](https://github.com/vmware/govmomi/issues/973))
- [50735461]	vcsim: more input spec honored in ReConfig VM
- [638d972b]	vcsim: Initialize VM fields properly
- [2892ed50]	Add '-rescan-vmfs' option to host.storage.info ([#966](https://github.com/vmware/govmomi/issues/966))
- [d4ee331c]	govc: allow columns in guest login password ([#972](https://github.com/vmware/govmomi/issues/972))
- [e15ff586]	Use IsFileNotFound helper in Datastore.Stat ([#969](https://github.com/vmware/govmomi/issues/969))
- [aa0382c1]	vcsim: Honor the input spec in ReConfig VM
- [465bd948]	Hook AccountManager to UserDirectory
- [aef2d795]	Destroy event history collectors ([#962](https://github.com/vmware/govmomi/issues/962))
- [42f9a133]	vcsim: Add HostLocalAccountManager
- [76f376a3]	vcsim: workaround xml ns issue with pyvsphere ([#958](https://github.com/vmware/govmomi/issues/958))
- [a1c49292]	Ignore AcquireLocalTicket errors ([#955](https://github.com/vmware/govmomi/issues/955))
- [bb150d50]	Add missing dependency in gen script
- [0eacf959]	toolbox: validate request offset in ListFiles ([#946](https://github.com/vmware/govmomi/issues/946))
- [1d6aed22]	Corrects datastore.disk usage which had not been generated ([#951](https://github.com/vmware/govmomi/issues/951))
- [de717389]	Corrects vm.info usage with required args ([#950](https://github.com/vmware/govmomi/issues/950))
- [c5ea3fb2]	Add datastore.disk inflate and shrink commands ([#943](https://github.com/vmware/govmomi/issues/943))
- [adf4530b]	Corrects host.shutdown ([#939](https://github.com/vmware/govmomi/issues/939))
- [45c5269b]	vcsim: add MakeDirectoryResponse ([#938](https://github.com/vmware/govmomi/issues/938))
- [b4e77bd2]	vcsim: copy RoleList for AuthorizationManager ([#932](https://github.com/vmware/govmomi/issues/932))
- [426a675a]	Fix [#933](https://github.com/vmware/govmomi/issues/933) ([#936](https://github.com/vmware/govmomi/issues/936))
- [3be5f1d9]	Add cluster.group and cluster.rule commands ([#928](https://github.com/vmware/govmomi/issues/928))
- [2a8a5168]	vcsim: apply vm spec NumCoresPerSocket ([#930](https://github.com/vmware/govmomi/issues/930))
- [3a61d85f]	vcsim: Configure dvs with the dvs config spec
- [3b25c720]	CreateChildDisk 6.7 support ([#926](https://github.com/vmware/govmomi/issues/926))
- [933ee3b2]	Add VirtualDiskManager.CreateChildDisk ([#925](https://github.com/vmware/govmomi/issues/925))
- [5f0f4004]	vcsim: Add VirtualMachine guest ID validation ([#921](https://github.com/vmware/govmomi/issues/921))
- [ef571547]	vcsim: add QueryVirtualDiskUuid ([#920](https://github.com/vmware/govmomi/issues/920))
- [0ea3b9bd]	Implemened vm.upgrade operation. ([#918](https://github.com/vmware/govmomi/issues/918))
- [27229ab7]	vcsim: update ServiceContent to 6.5 ([#917](https://github.com/vmware/govmomi/issues/917))
- [46c79c93]	Add support for cpu + mem allocation to vm.change command ([#916](https://github.com/vmware/govmomi/issues/916))

<a name="v0.16.0"></a>
## [Release v0.16.0](https://github.com/vmware/govmomi/compare/v0.15.0...v0.16.0)

> Release Date: 2017-11-08

### 💫 `govc` (CLI)

- [0295f1b0]	Fix VM clone when source doesn't have vNics
- [4fea6863]	add tasks and task.cancel commands
- [ddd32366]	add reboot option to host.shutdown

### 💫 `vcsim` (Simulator)

- [4543f4b6]	preserve order in QueryIpPools ([#914](https://github.com/vmware/govmomi/issues/914))
- [b385183e]	return moref from Task.Run ([#913](https://github.com/vmware/govmomi/issues/913))
- [e29ab54a]	Implement IpPoolManager lifecycle
- [b227a258]	add autostart option to power on VMs ([#906](https://github.com/vmware/govmomi/issues/906))
- [ecde4a89]	use soapenv namespace for Fault types
- [b1318195]	various property additions
- [c19ec714]	Generate similar ref value like VC
- [f3046058]	Add moref to vm's summary
- [5f3fba94]	validate authz privilege ids
- [c2caa6d7]	AuthorizationManager additions
- [2cb741f2]	Add IpPoolManager
- [a46ab163]	VirtualDisk file backing datastore is optional
- [d347175f]	add PerformanceManager
- [df3763d5]	Implement add/update/remove roles
- [ed18165d]	Generate device filename in CreateVM
- [e8741bf0]	add AuthorizationManager
- [8961efc1]	populate vm snapshot fields
- [add0245e]	Add UpdateNetworkConfig to HostNetworkSystem
- [2aa746c6]	Implement virtual machine snapshot
- [104ddfb7]	set VirtualDisk backing datastore
- [505b5c65]	Implement enter/exit maintenance mode
- [a1f8a328]	Implement add/remove license
- [585cf5e1]	add portgroup related operations
- [a7e79a7e]	add fields support
- [895573a5]	remove use of df program for datastore info
- [defe810c]	add FileQuery support to datastore search
- [5fcca79e]	add HostConfigInfo template
- [920a70c1]	add HostSystem hardware property
- [0833484e]	Fix merging of default devices
- [f6a734f5]	Add cdrom and scsi controller to Model VMs

### ⚠️ BREAKING

### 📖 Commits

- [7d879bac]	Doc updates ([#915](https://github.com/vmware/govmomi/issues/915))
- [4543f4b6]	vcsim: preserve order in QueryIpPools ([#914](https://github.com/vmware/govmomi/issues/914))
- [b385183e]	vcsim: return moref from Task.Run ([#913](https://github.com/vmware/govmomi/issues/913))
- [c8738903]	Remove tls-handshake-timeout flag ([#911](https://github.com/vmware/govmomi/issues/911))
- [e29ab54a]	vcsim: Implement IpPoolManager lifecycle
- [3619c1d9]	Use ProgressLogger for vm.clone command ([#909](https://github.com/vmware/govmomi/issues/909))
- [13f2aba4]	readme: fix formatting of listing ([#908](https://github.com/vmware/govmomi/issues/908))
- [b227a258]	vcsim: add autostart option to power on VMs ([#906](https://github.com/vmware/govmomi/issues/906))
- [79934451]	Add installation procedure in README.md ([#902](https://github.com/vmware/govmomi/issues/902))
- [ecde4a89]	vcsim: use soapenv namespace for Fault types
- [b1318195]	vcsim: various property additions
- [4d8737c9]	Switch to kr/pretty package for the -dump flag
- [e050b1b6]	Couple of fixes for import.spec result
- [017138ca]	import.spec not to assign deploymentOption
- [c19ec714]	vcsim: Generate similar ref value like VC
- [0295f1b0]	govc: Fix VM clone when source doesn't have vNics
- [f3046058]	vcsim: Add moref to vm's summary
- [bfed5eea]	[govc] Introduce TLSHandshakeTimeout parameter ([#890](https://github.com/vmware/govmomi/issues/890))
- [1c1291ca]	Support import ova/ovf by URL
- [3cb5cc96]	Remove BaseResourceAllocationInfo
- [5f3fba94]	vcsim: validate authz privilege ids
- [c91b9605]	Add clone methods to session manager
- [c2caa6d7]	vcsim: AuthorizationManager additions
- [2cb741f2]	vcsim: Add IpPoolManager
- [644c1859]	Updates to vm.clone link + snapshot flags
- [cf624f1a]	Add linked clone and snapshot support to vm.clone
- [024c09fe]	Fix govc events output
- [d4d94f44]	govc/events: read -json flag and output events as json
- [24e71ea4]	Fix vm.register command template flag
- [5209daf2]	Fix object name suffix matching in Finder
- [a46ab163]	vcsim: VirtualDisk file backing datastore is optional
- [d347175f]	vcsim: add PerformanceManager
- [df3763d5]	vcsim: Implement add/update/remove roles
- [8d5c1558]	Support clearing vm boot order
- [ed18165d]	vcsim: Generate device filename in CreateVM
- [df93050a]	Fix CustomFieldsManager.FindKey method signature
- [e8741bf0]	vcsim: add AuthorizationManager
- [8961efc1]	vcsim: populate vm snapshot fields
- [17fb12a5]	Add method to find a CustomFieldDef by Key
- [bc395ef0]	vscim: Implement UserDirectory
- [add0245e]	vcsim: Add UpdateNetworkConfig to HostNetworkSystem
- [2aa746c6]	vcsim: Implement virtual machine snapshot
- [104ddfb7]	vcsim: set VirtualDisk backing datastore
- [f3f51c58]	Add support for VM export
- [505b5c65]	vcsim: Implement enter/exit maintenance mode
- [a1f8a328]	vcsim: Implement add/remove license
- [585cf5e1]	vcsim: add portgroup related operations
- [a7e79a7e]	vcsim: add fields support
- [e2944227]	vim25: Move internal stuff to internal package
- [c4cab690]	Add support for SOAP request operation ID header
- [895573a5]	vcsim: remove use of df program for datastore info
- [4dd9a518]	Skip version check when using 6.7-dev API
- [cc2ed7db]	Change optional ResourceAllocationInfo fields to pointers
- [3f145230]	Use base type for DVS backing info
- [df1c3132]	Add vm.console command
- [829b3f99]	Fixup recent tasks output
- [c4e473af]	Add '-refresh' option to host.storage.info
- [3df440c8]	toolbox: avoid race when closing channels on stop
- [badad9a1]	toolbox: reset session when invalidated by the vmx
- [a1a96c8f]	Include "Name" in device.info -json
- [defe810c]	vcsim: add FileQuery support to datastore search
- [93f62ef7]	Default vm.migrate pool to the host pool
- [5fcca79e]	vcsim: add HostConfigInfo template
- [4fea6863]	govc: add tasks and task.cancel commands
- [596e51a0]	Use ovf to import vmdk
- [920a70c1]	vcsim: add HostSystem hardware property
- [9e2f8a78]	Add info about maintenance mode in host.info
- [78f3fc19]	Avoid panic if ova import task is canceled
- [11827c7a]	toolbox: default to tar format for directory archives
- [8811f9bf]	toolbox: make gzip optional for directory archive transfer
- [9703fe19]	toolbox: avoid blocking the RPC channel when transferring process IO
- [d6f60304]	Add view and filter support to object.collect command
- [3527a5f8]	Tolerate repeated Close for file follower
- [ddd32366]	govc: add reboot option to host.shutdown
- [4d9061ac]	toolbox: use host management IP for guest file transfer
- [7d956b6b]	toolbox: add Client Upload and Download methods
- [c7111c63]	toolbox: support single file download via archive handler
- [ebb77d7c]	Use vcsim in bats tests
- [4bb89668]	vCenter cluster testbed automation
- [ad960e95]	toolbox: SendGuestInfo before the vmx asks us to
- [bdea7ff3]	toolbox: update vmw-guestinfo
- [51d12609]	toolbox: remove receiver from DefaultStartCommand
- [114329fc]	Add host thumbprint for use with guest file transfer
- [5083a277]	Add FindByUuid method for Virtual Machine
- [e1ab84af]	toolbox: map exec.ErrNotFound to vix.FileNotFound
- [d1091087]	toolbox: pass URL to ArchiveHandler Read/Write methods
- [cddc353c]	toolbox: make directory archive read/write customizable
- [ba6720ce]	toolbox: add http and exec round trippers
- [b35abbc8]	Handle object names containing a '/'
- [ac4891fb]	toolbox: fix ListFiles when given a symlink
- [60a6510f]	Minor correction in README.md
- [0c583dbc]	toolbox: support transferring /proc files from guest
- [0833484e]	vcsim: Fix merging of default devices
- [c9aaa3fa]	Move toolbox from vmware/vic to govmomi
- [f6a734f5]	vcsim: Add cdrom and scsi controller to Model VMs
- [9d47dd13]	Move vcsim from vmware/vic to govmomi

<a name="v0.15.0"></a>
## [Release v0.15.0](https://github.com/vmware/govmomi/compare/v0.14.0...v0.15.0)

> Release Date: 2017-06-19

### ⚠️ BREAKING

### 📖 Commits

- [b63044e5]	Release 0.15.0
- [3d357ef2]	Add dvs.portgroup.info usage
- [72977afb]	Add support for guest.FileManager directory download
- [94837bf7]	Update examples
- [e1bbcf52]	Update wsdl generator
- [b16a3d81]	fix the WaitOptions struct, MaxWaitSeconds is optional, but we can set the value 0
- [9ca7a2b5]	Support removal of ExtraConfig entries
- [86cc210c]	Guest command updates
- [9c5f63e9]	Doc updates
- [6d714f9e]	New examples: datastores, hosts and virtualmachines using view package
- [f48e1151]	update spew to be inline with testify
- [6f5c037c]	Adjust message slice passed to include
- [48509bc3]	Fix package name
- [6f635b73]	Add host.shutdown command
- [67b13b52]	Add doc on metric.sample instance flag ([#726](https://github.com/vmware/govmomi/issues/726))
- [8bff8355]	Fix tail n=0 case ([#725](https://github.com/vmware/govmomi/issues/725))
- [10e6ced9]	Update copyright ([#723](https://github.com/vmware/govmomi/issues/723))
- [6f8ebd89]	Allow caller to supply custom tail behavior ([#722](https://github.com/vmware/govmomi/issues/722))
- [35caa01b]	Add options to host.autostart.add ([#719](https://github.com/vmware/govmomi/issues/719))
- [2030458d]	Add VC options command ([#717](https://github.com/vmware/govmomi/issues/717))
- [0ccad10c]	Exported FindSnapshot() Method ([#715](https://github.com/vmware/govmomi/issues/715))
- [34202aca]	Additional wrapper functions for SPBM
- [c7f718b1]	Add AuthorizationManager {Enable,Disable}Methods
- [d5e08cd2]	Add PBM client and wrapper methods
- [58019ca9]	Add generated types and methods for PBM
- [58960380]	Regenerate against current vmodl.db
- [f736458f]	Support non-Go clients in xml decoder

<a name="v0.14.0"></a>
## [Release v0.14.0](https://github.com/vmware/govmomi/compare/v0.13.0...v0.14.0)

> Release Date: 2017-04-08

### ⚠️ BREAKING

### 📖 Commits

- [9bfdc5ce]	Release 0.14.0
- [3ba0eba5]	Release 0.13.0
- [86063832]	Add object.find command
- [0391e8eb]	Adds FindManagedObject method.
- [796e87c8]	Include embedded fields in object.collect output
- [2536e792]	Use Duration flag for vm.ip -wait flag
- [3aa64170]	Merge commit 'b0b51b50f40da2752c35266b7535b5bbbc8659e3' into marema31/govc-vm-ip-wait
- [59466881]	Implement EthernetCardBackingInfo for OpaqueNetwork
- [0d2e1b22]	Finder: support changing object root in find mode
- [9ded9d10]	Add Bash completion script
- [3bd4ab46]	Add QueryVirtualDiskInfo
- [16f6aa4f]	Emacs: add metric select
- [3763321e]	Add unit conversion to metric CSV
- [b0b51b50]	Add -wait option to govc vm.ip to allow non-blocking query
- [f0d4774a]	Add json support to metric ls and sample commands
- [c9de0310]	Add performance manager and govc metric commands
- [d758f694]	Add check for nil envelope
- [ab595fb3]	Remove deferred Close() call in follower's Read()

<a name="v0.13.0"></a>
## [Release v0.13.0](https://github.com/vmware/govmomi/compare/v0.12.1...v0.13.0)

> Release Date: 2017-03-02

### 💫 `vcsim` (Simulator)

- [5f7efaf1]	esxcli FirewallInfo fixes ([#661](https://github.com/vmware/govmomi/issues/661))

### ⚠️ BREAKING

### 📖 Commits

- [b4a3f7a1]	Release 0.13.0
- [5bf03cb4]	Add vm.guest.tools command
- [b4ef3b73]	Host is optional for MarkAsVirtualMachine ([#675](https://github.com/vmware/govmomi/issues/675))
- [f4a3ffe5]	Add vsan and disk commands / helpers ([#672](https://github.com/vmware/govmomi/issues/672))
- [1f82c282]	Handle the case where e.VirtualSystem is nil ([#671](https://github.com/vmware/govmomi/issues/671))
- [dd346974]	Remove object.ListView ([#669](https://github.com/vmware/govmomi/issues/669))
- [4994038a]	Wraps the ContainerView managed object. ([#667](https://github.com/vmware/govmomi/issues/667))
- [93064c06]	Handle nil TaskInfo in task.Wait callback [#2](https://github.com/vmware/govmomi/issues/2) ([#666](https://github.com/vmware/govmomi/issues/666))
- [f1f5b6cb]	Handle nil TaskInfo in task.Wait callback ([#665](https://github.com/vmware/govmomi/issues/665))
- [f3cf126d]	Support alternative './...' syntax for finder ([#664](https://github.com/vmware/govmomi/issues/664))
- [9bda6c3e]	Finder: support automatic Folder recursion ([#663](https://github.com/vmware/govmomi/issues/663))
- [0a28e595]	Add a command line option to change an existing disk attached to a VM ([#658](https://github.com/vmware/govmomi/issues/658))
- [3e95cb11]	Attach and list RDM/LUN ([#656](https://github.com/vmware/govmomi/issues/656))
- [5f7efaf1]	vcsim: esxcli FirewallInfo fixes ([#661](https://github.com/vmware/govmomi/issues/661))
- [17e6545f]	Add device option to WaitForNetIP ([#660](https://github.com/vmware/govmomi/issues/660))
- [ba9e3f44]	Fix vm.change test
- [e66c8344]	Add the option to describe a VM using the annotation option in ConfigSpec ([#657](https://github.com/vmware/govmomi/issues/657))
- [505fcf9c]	Update doc
- [913c0eb4]	Add support for reading and changing SyncTimeWithHost option ([#539](https://github.com/vmware/govmomi/issues/539))
- [682494e1]	Remove _Task suffix from vapp methods
- [733acc9e]	Emacs: add govc-command-history
- [ea52d587]	Add object.collect command ([#652](https://github.com/vmware/govmomi/issues/652))
- [f49782a8]	Update email address for contributor Bruce Downs

<a name="v0.12.1"></a>
## [Release v0.12.1](https://github.com/vmware/govmomi/compare/v0.12.0...v0.12.1)

> Release Date: 2016-12-19

### ⚠️ BREAKING

### 📖 Commits

- [6103db21]	Release 0.12.1
- [45a53517]	Note 6.5 support
- [fec40b21]	Add '-f' flag to logs command ([#643](https://github.com/vmware/govmomi/issues/643))
- [40cf9f80]	govc.el: auth-source integration ([#648](https://github.com/vmware/govmomi/issues/648))
- [ca99f8de]	Add govc-command customization option ([#645](https://github.com/vmware/govmomi/issues/645))
- [ad6e5634]	Avoid Finder panic when SetDatacenter is not called ([#640](https://github.com/vmware/govmomi/issues/640))
- [b5c807e3]	Add storage support to vm.migrate ([#641](https://github.com/vmware/govmomi/issues/641))
- [1a7dc61e]	govc/version: skip first char in git version mismatch error ([#642](https://github.com/vmware/govmomi/issues/642))
- [6bc730e1]	Add Slack links
- [e152c355]	Add DatastorePath helper ([#638](https://github.com/vmware/govmomi/issues/638))
- [5b4d5215]	Add support for file backed serialport devices ([#637](https://github.com/vmware/govmomi/issues/637))
- [f49bd564]	Add vm.ip docs ([#636](https://github.com/vmware/govmomi/issues/636))

<a name="v0.12.0"></a>
## [Release v0.12.0](https://github.com/vmware/govmomi/compare/v0.11.4...v0.12.0)

> Release Date: 2016-12-01

### ⚠️ BREAKING

### 📖 Commits

- [ab40ac73]	Release 0.12.0
- [e702e188]	Disable use of service ticket for datastore HTTP access by default ([#635](https://github.com/vmware/govmomi/issues/635))
- [1fba1af7]	Attach context to HTTP requests for cancellations
- [79cb3d93]	Support InjectOvfEnv without PowerOn when importing
- [117118a2]	Support stdin as import options source
- [b10f20f4]	Don't ignore version/manifest for existing sessions
- [82929d3f]	Add basic VirtualNVMEController support
- [757a2d6d]	re-generate vim25 using 6.5.0

<a name="v0.11.4"></a>
## [Release v0.11.4](https://github.com/vmware/govmomi/compare/v0.11.3...v0.11.4)

> Release Date: 2016-11-15

### ⚠️ BREAKING

### 📖 Commits

- [b9bcc6f4]	Release 0.11.4
- [dbbf84e8]	Add authz role helpers and commands
- [765b34dc]	Add folder/pod examples
- [79cb52fd]	Add host.account examples
- [2a2cab2a]	Add host.portgroup.change examples

<a name="v0.11.3"></a>
## [Release v0.11.3](https://github.com/vmware/govmomi/compare/v0.11.2...v0.11.3)

> Release Date: 2016-11-08

### ⚠️ BREAKING

### 📖 Commits

- [e16673dd]	Release 0.11.3
- [629a573f]	Add -product-version flag to dvs.create
- [83028634]	Allow DatastoreFile follower to drain current body

<a name="v0.11.2"></a>
## [Release v0.11.2](https://github.com/vmware/govmomi/compare/v0.11.1...v0.11.2)

> Release Date: 2016-11-01

### ⚠️ BREAKING

### 📖 Commits

- [cd80b8e8]	Release 0.11.2
- [f15dcbdc]	Avoid possible NPE in VirtualMachine.Device method
- [128b352e]	Add support for OpaqueNetwork type
- [c5b9a266]	Add host account manager support for 5.5

<a name="v0.11.1"></a>
## [Release v0.11.1](https://github.com/vmware/govmomi/compare/v0.11.0...v0.11.1)

> Release Date: 2016-10-27

### ⚠️ BREAKING

### 📖 Commits

- [1a7df5e3]	Release 0.11.1
- [1ae858d1]	Add support for VirtualApp in pool.change command
- [91b2ad48]	Release script tweaks

<a name="v0.11.0"></a>
## [Release v0.11.0](https://github.com/vmware/govmomi/compare/v0.10.0...v0.11.0)

> Release Date: 2016-10-25

### ⚠️ BREAKING

### 📖 Commits

- [a16901d7]	Release 0.11.0
- [4fc9deb4]	Add object destroy and rename commands
- [82634835]	Add dvs.portgroup.change command

<a name="v0.10.0"></a>
## [Release v0.10.0](https://github.com/vmware/govmomi/compare/v0.9.0...v0.10.0)

> Release Date: 2016-10-20

### ⚠️ BREAKING

### 📖 Commits

- [bb498f73]	Release 0.10.0
- [468a15af]	Release script updates
- [1c3499c4]	Documentation updates
- [1e52d88a]	Update contributors
- [e3d59fd9]	Fix snapshot.tree on vm with no snapshots
- [711fdd9c]	Add host.date info and change commands
- [16d7514a]	Add govc session ls and rm commands
- [73c471a9]	Add HostConfigManager field checks
- [d7f94557]	Improve cluster/host add thumbprint support
- [fea8955b]	Add session.Locale var to change default locale
- [eefe6cc1]	Add service ticket thumbprint validation
- [3a0a61a6]	Set default locale to en_US
- [aa1a9a84]	TLS enhancements
- [9f0e9654]	Treat DatastoreFile follower Close as "stop"
- [838b2efa]	Support typeattr for enum string types
- [dcbc9d56]	Make vm.ip esxcli test optional
- [9e20e0ae]	Remove vca references
- [7c708b2e]	Adding vSPC proxyURI to govc

<a name="v0.9.0"></a>
## [Release v0.9.0](https://github.com/vmware/govmomi/compare/v0.8.0...v0.9.0)

> Release Date: 2016-09-09

### ⚠️ BREAKING

### 📖 Commits

- [f9184c1d]	Release 0.9.0
- [e050cb6d]	Add govc -h flag
- [a4343ea8]	Set default ScsiCtlrUnitNumber
- [a920d73d]	Add -R option to datastore.ls
- [f517decc]	Fix SCSI device unit number selection
- [abaf7597]	Add DatastoreFile helpers
- [7cfa7491]	Make Datastore ServiceTicket optional
- [9ad57862]	Add vm.migrate command
- [c66458f9]	Add govc vm.{un}register commands
- [54c0c6e5]	Checking result of reflect.TypeOf is not nil before continuing
- [ea0189ea]	Fix flags.NewOptionalBool panic
- [a9cdf437]	Add govc guest command tests
- [38dee111]	Add VirtualMachine.Unregister func
- [98b50d49]	make curl follow HTTP redirects
- [8a27691f]	make goreportcard happy
- [bf66f750]	Add govc vm snapshot commands
- [eb02131a]	Validate vm.clone -vm flag value
- [62159d11]	Add device.usb.add command
- [27e02431]	Remove a bunch of context.TODO() calls.
- [a9cee43a]	Fixing tailing for events command
- [4fa7b32a]	Bump to 1.7 and start using new context pkg
- [4b7c59bf]	Fix missing datastore name with vm.clone -force=false
- [e3642fce]	Fix deletion of powered off vApp
- [63d60025]	Support stdin/stdout in datastore upload/download
- [e149909e]	Emacs: add govc-session-network
- [0ccc1788]	Emacs: add govc json diff
- [f1d6e127]	Add host.portgroup.change command
- [6f441a84]	Add host.portgroup.info command
- [aaf40729]	Add HostNetworkPolicy to host.vswitch.info
- [5ccb0572]	Add json support to host.vswitch.info command
- [9d19d1f7]	Support instance uuid in SearchFlag
- [2d3bfc9f]	Add json support to esxcli command
- [bac04959]	Support multiple NICs with vm.ip -esxcli
- [b3177d23]	Add -unclaimed flag to host.storage.info command
- [b1234a90]	govc - popualte 'Path' fiels in xxx.info output
- [7cab0ab6]	Implemented additional ListView methods
- [498cb97d]	Add 'Annotation' attribute to importx options.
- [223168f0]	Add NetworkMapping section to importx options.
- [5c708f6b]	Remove vendor target from the Makefile
- [f8199eb8]	Handle errors in QueryVirtualDiskUUid function ([#548](https://github.com/vmware/govmomi/issues/548))
- [73dcde2c]	vendor github.com/davecgh/go-spew/spew
- [e1e407f7]	vendor golang.org/x/net/context
- [e3c3cd0a]	Populate network mapping from ovf envelope ([#546](https://github.com/vmware/govmomi/issues/546))
- [fa6668dc]	Add QueryVirtualDiskUuid function ([#545](https://github.com/vmware/govmomi/issues/545))
- [17682d5b]	Fixes panic in govc events

<a name="v0.8.0"></a>
## [Release v0.8.0](https://github.com/vmware/govmomi/compare/v0.7.1...v0.8.0)

> Release Date: 2016-06-30

### ⚠️ BREAKING

### 📖 Commits

- [c0c7ce63]	Release 0.8.0
- [ce4b0be6]	Disable datastore service ticket hostname usage
- [3e44fe88]	Add support for login via local ticket
- [acf37905]	Add StoragePod support to govc folder.create
- [94d4e2c9]	Include StoragePod in Finder.FolderList
- [473f3885]	Avoid use of eval with govc env
- [4fb7ad2e]	Add datacenter.create folder option
- [77ea6f88]	Avoid vm.info panic against vcsim
- [95b2bc4d]	Session persistence improvements
- [720bbd10]	Add type attribute to soap.Fault Detail
- [ff7b5b0d]	Add filtering for use of datastore service ticket
- [fe9d7b52]	Add support for Finder lookup via moref
- [c26c7976]	Use ticket HostName for Datastore http access
- [bea2a43c]	Add govc/vm.markasvm command
- [9101528d]	Add govc/vm.markastemplate command
- [982e64b8]	Add vm.markastemplate

<a name="v0.7.1"></a>
## [Release v0.7.1](https://github.com/vmware/govmomi/compare/v0.7.0...v0.7.1)

> Release Date: 2016-06-03

### ⚠️ BREAKING

### 📖 Commits

- [2cad28d0]	Fix Datastore upload/download against VC

<a name="v0.7.0"></a>
## [Release v0.7.0](https://github.com/vmware/govmomi/compare/v0.6.2...v0.7.0)

> Release Date: 2016-06-02

### ⚠️ BREAKING

### 📖 Commits

- [6906d301]	Release 0.7.0
- [558321df]	Move InventoryPath field to object.Common
- [4147a6ae]	Add -require flag to govc version command
- [d9fd9a4b]	Add support for local type in datastore.create
- [650b5800]	Fix vm.create disk scsi controller lookup
- [9463b5e5]	Update changelog for govc to add datastore -namespace flag
- [4aab41b8]	Update changelog with DatastoreNamespaceManager methods
- [4d6ea358]	Support mkdir/rm of namespace on vsan
- [bb7e2fd7]	InjectOvfEnv() should work with VSphere
- [91ca6bd5]	Add host.service command
- [2f369a29]	Add host.storage.mark command
- [b001e05b]	Add -rescan option to host.storage.info command

<a name="v0.6.2"></a>
## [Release v0.6.2](https://github.com/vmware/govmomi/compare/v0.6.1...v0.6.2)

> Release Date: 2016-05-13

### ⚠️ BREAKING

### 📖 Commits

- [9051bd6b]	Release 0.6.2
- [3ab0d9b2]	Get complete file details in Datastore.Stat
- [0c21607e]	Convert types when possible
- [648d945a]	Avoid xsi:type overwriting type attribute
- [4e0680c1]	adding remove all snapshots to vm objects

<a name="v0.6.1"></a>
## [Release v0.6.1](https://github.com/vmware/govmomi/compare/v0.6.0...v0.6.1)

> Release Date: 2016-04-30

### ⚠️ BREAKING

### 📖 Commits

- [18154e51]	Release 0.6.1
- [47098806]	Fix mo.Entity interface

<a name="v0.6.0"></a>
## [Release v0.6.0](https://github.com/vmware/govmomi/compare/v0.5.0...v0.6.0)

> Release Date: 2016-04-29

### ⚠️ BREAKING

### 📖 Commits

- [2c1d977a]	Release 0.6.0
- [cc686c51]	Add folder.moveinto command
- [8e85a8d2]	Add folder.{create,destroy,rename} methods
- [0ba22d24]	Add Common.Rename method
- [61792ed3]	Fix Finder.FolderList check
- [b6be92a1]	Restore optional DatacenterFlag
- [53903a3a]	Add OutputFlag support to govc about command
- [e66f7793]	Add OptionManager and host.option commands
- [9d69fe4b]	Add debug xmlformat script
- [f1786bec]	Add option to use the same path for debug runs
- [99c8c5eb]	Add folder.info command
- [eca4105a]	Add datacenter.info command
- [71484c40]	Add mo.Entity interface
- [388df2f1]	Add helper to wait for multiple VM IPs
- [fc9f58d0]	Add RevertToSnapshot
- [a4aca111]	Add govc env command
- [ef17f4bd]	Update CI config
- [fa91a600]	Add host.account commands
- [44bb6d06]	Update release install instructions
- [08ba4835]	Leave AddressType empty in EthernetCardTypes
- [f9704e39]	Add vm clone
- [e6969120]	Add datastore.Download method
- [1aca660c]	device.remove: add keep option

<a name="v0.5.0"></a>
## [Release v0.5.0](https://github.com/vmware/govmomi/compare/v0.4.0...v0.5.0)

> Release Date: 2016-03-30

### ⚠️ BREAKING

### 📖 Commits

- [c1b29993]	Release 0.5.0
- [b8549681]	Use VirtualDeviceList for import.vmdk
- [cf96f70d]	Remove debug flags from pool tests
- [f74a896d]	Switch to int32 type for xsd int fields
- [074494df]	Regenerate against 6.0u2 wsdl
- [ce9314c4]	Include license header in generated files
- [957c8827]	Add pointer field white list to generator
- [2c1d1950]	Change pool recusive destroy to children destroy
- [5d34409f]	Add dvs.portgroup.info command
- [216031c3]	Update docs
- [f7dfcc98]	Remove govc-test pools in teardown hook
- [556a9b17]	Simplify pool destroy test
- [4e47b140]	Add folder management to vm.create
- [7c33bcb3]	Update test ESX IP in Drone secrets file
- [1b6ec477]	Regenerate Drone secrets file
- [f64ea833]	Implemented the ablitiy to tail the vSphere event stream - govc tail and force flag added to events command
- [fd7d320f]	Including github.com/davecgh/go-spew/spew in go get
- [1d4efec0]	Including github.com/davecgh/go-spew/spew in go get
- [424d3611]	The -dump option now requests a recursive traversal as -json does
- [b45747f3]	Added new -dump output flag for pretty printing underlying objects using davecgh/go-spew
- [a243716c]	Run govc tests against ESX using Drone
- [fb75c63e]	Double quotes network name to prevent space in name from failing the tests
- [564944ba]	test_helper.bash updated to conditionally set env variables
- [c9c6e38f]	Added new govc vm.disk.create -mode option for selecting one the VirtualDiskMode types
- [6922c88b]	Add -net flag to device.info command
- [dff2c197]	Fix VirtualDeviceList.CreateFloppy
- [c7d8cd3e]	Ran gofmt on create.go
- [e077bcf5]	Fix issue with optional UnitNumber (v2)
- [539ad504]	Added arguments to govc vm.disk.create for thick provisioning and eager scrubbing, as requested in issue [#254](https://github.com/vmware/govmomi/issues/254)
- [e66c6df9]	Handle import statement for types too
- [265d8bdb]	Remove hardcoded urn:vim25 value from vim_wsdl.rb

<a name="v0.4.0"></a>
## [Release v0.4.0](https://github.com/vmware/govmomi/compare/v0.3.0...v0.4.0)

> Release Date: 2016-02-26

### ⚠️ BREAKING

### 📖 Commits

- [b3d202ab]	Release 0.4.0
- [749da321]	Fix vm.change's ExtraConfig values being truncated at equal signs
- [13fbc59d]	Add switch to specify protocol version in SOAPAction header
- [07013a97]	Update CHANGELOG
- [bfe414fe]	Allow vm.create to take datastore cluster argument
- [dda71761]	Include reference to datastore in CreateDisk
- [855abdb3]	Make NewKey function public
- [d0031106]	Use custom datastore flags in vm.create
- [306b613d]	Modify govc's vm.create to create VM in one shot
- [e96130b4]	Add extra datastore arguments to vm.create
- [0a2da16d]	Add datastore cluster methods to finder
- [c69e9bc1]	Allow StoragePod type to be traversed
- [4d2ea3f4]	added explicit path during clone
- [3d8eb102]	Update missing property whitelist
- [779ae0a1]	re-generate vim25 using 6.0 Update 1b (vimbase [#3024326](https://github.com/vmware/govmomi/issues/3024326))
- [53c29f6a]	Handle import statements same as include
- [a738f89d]	Update govc.el URL
- [da2a249e]	Doc updates
- [47e46425]	govc.el: minor fixes for distribution as a package
- [8459ceb9]	handle GOVC_TEST_URL=user:pass[@IP](https://github.com/IP) pattern
- [3b669760]	Add Emacs interface to govc
- [7ec8028d]	Update README to include Drone build status and local build instructions
- [2ec65fbe]	Add config for Drone CI build
- [5437c466]	introduce Datastore.Type()
- [983571af]	introduce IsVC method and start using it
- [0732f137]	Introduce AttachedClusterHosts
- [18945281]	start using new helper functions for govc/flags
- [044d904a]	Add some common functions to find/finder.go
- [534dabbd]	Support vapp in pool.info command
- [4d9c6c72]	Fix bats tests
- [5e04d5ca]	Add -p and -a options to govc datastore.ls command
- [33963263]	Added check for missing ovf deployment section

<a name="v0.3.0"></a>
## [Release v0.3.0](https://github.com/vmware/govmomi/compare/v0.2.0...v0.3.0)

> Release Date: 2016-01-15

### ⚠️ BREAKING

### 📖 Commits

- [501f6106]	Mark 0.3.0 in change log
- [83a26512]	Update contributors
- [995d970f]	Print os.Args[0] in error messages
- [0a4c9782]	Move stat function to object.Datastore
- [8a0d4217]	Support VirtualApp in the lister
- [82734ef3]	Support empty folder in SearchFlag.VirtualMachines
- [f64f878f]	Add support for custom session keep alive handler
- [2d498658]	Use OptionalBool for ExpandableReservation
- [ac9a39b0]	Script to capture vpxd traffic on VCSA
- [3f473628]	Script to capture and decrypt hostd SOAP traffic
- [eccc3e21]	Move govc url.Parse wrapper to soap.ParseURL
- [e1031f44]	Don't assume sshClient firewall rule is disabled
- [cd5d8baa]	Let the lister recurse into a ComputeHost
- [b601a586]	Specify the new entity's name upon import
- [a5e26981]	Explicitly instantiate and register flags
- [aca77c67]	Parameterize datastore in VM tests
- [37324472]	Pass context to command and flag functions
- [6f955173]	Minor optimization to encoding usage
- [0f4aee8b]	Create VMFS datastore with datastore.create
- [ec724783]	Add host storage commands
- [debdd854]	Run license script
- [64022512]	Fix license script to work for uncommitted files
- [5cb0c344]	Remove host reference from HostFirewallSystem
- [4fb4052a]	Change the comment that mentions ha-datacenter
- [b76ad0eb]	Let the ESXi to figure out datastore name
- [918188dc]	Add helper method to get VM power state
- [29a2f027]	Add permissions.{ls,set,remove} commands
- [f27787a1]	Add DatacenterFlag.ManagedObjects helper
- [0e629647]	Option to disable API version check
- [42d899d0]	Add commands to add and remove datastores
- [369e0e7f]	Check host state in Datastore.AttachedHosts
- [7adf8375]	Test that vm.info -r prints mo names
- [3198242e]	Change ComputeResource.Hosts to return HostSystem
- [b34f346e]	Support property collection for embedded types
- [8035c180]	Fix vm nested hv option
- [b1d9d3c2]	Update copyright years in code headers
- [c99e7bac]	Add dvs commands
- [c30b7f17]	Support DVS lookup in finder
- [094fbdfe]	Embed Reference interface in NetworkReference
- [0657cf76]	Add DVS helpers
- [6e96a1db]	Add host.vnic.{service,info} commands
- [ae6b0b77]	Add VsanSystem and VirtualNicManager wrappers
- [24297494]	Add vsan flags to cluster.change command
- [4088502d]	Add license.assigned.list id flag
- [d089489e]	Add cluster.add license flag
- [31ee6e03]	Add vm.change options to set hv/mmu
- [a414852e]	Refactor host.add command to use HostConnectFlag
- [51543392]	Add cluster.{create,change,add} commands
- [8262e1da]	Add cluster related host commands
- [2443b364]	Add HostConnectFlag
- [8ae7da82]	Add object.HostSystem methods
- [0f630dd9]	Add finder.Folder method
- [7cd5fbb5]	Add bash function to save/load GOVC environments
- [12f26c21]	Add object.Common.Destroy method
- [2ab8aa59]	Add ComputeResource.Reconfigure method
- [5f47f155]	Add flags.NewOptionalBool
- [25fe42b2]	Add -feature flag to license list commands
- [2e6c0476]	Add license.InfoList list wrapper
- [ef7371af]	Add license assignment commands
- [5005e6e4]	Add license.AssignmentManager
- [69a23bd4]	Use object.Common in license.Manager
- [dbce3faf]	Rename receiver variable for consistency
- [80705c11]	Pass pointer to bps uint64 in last progress report
- [26e77c8e]	VirtualMachine: Add Customize function on object.VirtualMachine
- [c2a78973]	Add license.decode command
- [b3a7e07e]	Add DistributedVirtualPortgroup support to vm.info
- [1b11ad02]	Fix KeepAlive
- [3ecfd0db]	Add HostFirewallSystem wrapper
- [9ded9c1a]	KeepAlive support with certificate based login
- [cf2a879b]	Add DiagnosticManager and logs commands
- [7b14760a]	Update README.md
- [ad694500]	Export Datastore.ServiceTicket method
- [76690239]	Added a method to create snapshot of a virtual machine
- [6d4932af]	Use service ticket for datastore file access
- [5fcc29f6]	Fix vcsa ssh config
- [ac390ec8]	Retry on empty result from property collector
- [f3041b2c]	Add methods for client certificate based auth
- [b9edc663]	Add extension manager and govc commands
- [9057659c]	Fix key composition in building OVF spec
- [f56f6e80]	Move OVF environment related code to env{,test}.go
- [b33c9aef]	Add minimal doc to ovf package
- [3d40aefb]	Added verbose option to the import.spec feature
- [1df0a81d]	change for looking up a VM using instanceUUID
- [5f4d36cd]	Introduce govc vapp.{info|destroy|power}
- [88795252]	Handle the import.spec case where no spec file is provided
- [bcdc53fb]	Add inventory path to govc info commands
- [305371a8]	Collect govc host and pool info in one call
- [bfd47026]	Relax the convention around importing an ova
- [3742a8aa]	don't start goroutine while context is nil

<a name="v0.2.0"></a>
## [Release v0.2.0](https://github.com/vmware/govmomi/compare/v0.1.0...v0.2.0)

> Release Date: 2015-09-15

### ⏮ Reverts

- [2900f2ff]	Add Host information to vm.info
- [8bec13f7]	Fix git dirty status error in build script

### ⚠️ BREAKING

### 📖 Commits

- [b3315079]	Mark 0.2.0 in change log
- [cc3bcbee]	Add mode argument to release script
- [ae4a6e53]	Build govc with new cross compilation facilities
- [4708d165]	Derive CONTRIBUTORS from commit history
- [00909f48]	Move contrib/ -> scripts/
- [a0f4f799]	Capitalization
- [13baa0e4]	Split import functionality into independent flags
- [6363d0e2]	Added ovf.Property output to import.spec
- [7af121df]	Update change log
- [f9deb385]	Fix event.Manager category cache
- [7f0a892d]	Avoid tabwriter in events command
- [29601b46]	Use vm.power force flag for hard shutdown/reboot
- [ea833cf5]	Add VirtualDiskManager CreateVirtualDisk wrapper
- [bfabd01d]	Interative clean up of bats testing
- [7cba62d9]	Clean up of vcsa creation script
- [631d6228]	Add serial port URI info to device.info output
- [0b31dcff]	Add -json support to device.info command
- [54e324d1]	Add govc vm.info resources option
- [9cc5d8f5]	Add helper method to wait for virtual machine power state.
- [9ddd6337]	Remove superfluous math.Pow calculations
- [5272b1e9]	Added common method of humanizing byte strings
- [3145d146]	Add helper method to check if VMware Tools is running in the guest OS.
- [e4f4c737]	Misc clean up
- [01f2aed0]	Add host name to vm.info
- [f24ec75a]	Use property.Collector.Retrieve() in vm.info
- [a779c3b7]	Renamed vm.info VmInfos back to VirtualMachines
- [2900f2ff]	Revert "Add Host information to vm.info"
- [2a567478]	Add -hints option to host.esxcli command
- [1f0708e2]	Add options to importing an ovf or and ova file
- [debde780]	Only retrieve "currentSession" property
- [b5187c16]	Update CONTRIBUTORS
- [3e4ced8c]	Added the ability to specify ovf properties during deployment
- [688a6b18]	Introduce more VirtualApp methods
- [b1f0cb0c]	Add flag to specify destination folder for import.ovf and import.ova
- [c9fcf1ce]	Add check for error reading ova file
- [edb0a2cf]	clone vmware/rbvmomi repo if it's missing
- [40c26fc6]	use e.Object.Reference().Type as suggested by Doug
- [c1442f95]	introduce CreateVApp and CreateChildVM_Task
- [25405362]	add VirtualAppList and VirtualApp methods to Finder
- [121f075c]	Add CustomFieldsManager wrapper and cli commands
- [dd016de3]	include VirtualApp in ls -l output
- [b5db4d6d]	Provide ability to override url username and password
- [11d5ae9c]	Add OVF unmarshalling
- [135569e7]	Update travis.yml for new infra
- [822432eb]	Make govet stop complaining
- [baf9149e]	Add datastore.info cli command
- [2b93c199]	Add serial port matcher to SelectByBackingInfo
- [26ba22de]	Merge branch 'gavrie-master'
- [62591576]	Add Host information to vm.info
- [a90019ab]	Add methods for useful properties to VirtualMachine
- [502963c4]	Add Relocate method to VirtualMachine
- [7f4b6d38]	Add String method to objects for pretty printing
- [99f57f16]	Add events helpers and cli command
- [4c989ac3]	Update CONTRIBUTORS
- [ad7d1917]	Update to vim25/6.0 API
- [ad39adb9]	Add net.address flag
- [e01555f9]	Add command to add host to datacenter
- [efbd3293]	Stop returning children from `ManagedObjectList`
- [d16670f5]	Update CONTRIBUTORS
- [97fbf898]	Mention GOVC_USERNAME and GOVC_PASSWORD in CHANGELOG
- [8766bda0]	Add test to check for flag name collisions
- [791b3365]	Remove flags for overriding username and password
- [85957949]	include GOVC_USERNAME and GOVC_PASSWORD in govc README
- [8584259a]	Export variables in release script
- [14889008]	Add test for GOVC_USERNAME and GOVC_PASSWORD
- [c0a984cd]	Only run license tests against evaluation license
- [293ac813]	Allow override of username and password
- [e053bdf2]	Add extraConfig option to vm.change and vm.info
- [1dec0695]	Update CONTRIBUTORS
- [1acf418c]	Add Usage for host.esxcli
- [2e00fdb1]	Modify archive.go bug
- [985291d5]	Add missing types to list.ToElement
- [871f5d4f]	Add script to create a draft prerelease
- [8bec13f7]	Revert "Fix git dirty status error in build script"
- [c825a3c7]	Only use annotated tags to describe a version
- [66320cb0]	Retry twice on temporary network errors in govc
- [67be5f1d]	Add retry functionality to vim25 package
- [fba0548b]	Add method to destroy a compute resource
- [2add2f7a]	Add methods to add standalone or clustered hosts
- [de297fcb]	Add ability to create, read and modify clusters
- [f10480af]	Change finder functions to no longer take varargs
- [4bc93a66]	Fix resource pool creation/modification
- [b434a9a8]	Rename persist flag to persist-session
- [d85ad215]	Ignore ManagedObjectNotFound in list results
- [4c497373]	Add example that lists datastores
- [5d153787]	Update govc CHANGELOG
- [0165e2de]	Add flag to toggle persisting session to disk
- [8acb2f28]	Add Mevan to CONTRIBUTORS
- [add15217]	Ignore missing environmentBrowser field
- [447d18cd]	Fix error when using SDRS datastores
- [e85f6d59]	Find ComputeResource objects with find package
- [55f984e8]	Test package only depends on vim25
- [dbe47230]	Drop omitempty tag from optional pointer fields
- [749f0bfa]	Interpret negative values for unsigned fields
- [49a34992]	Update CHANGELOG
- [263780f3]	Update code to work with bool pointer fields
- [93aad8da]	Make optional bool fields pointers
- [b7c51f61]	Return errors for unexpected HTTP statuses
- [62ca329a]	Abort client tests on errors
- [ae345e7f]	Rename LICENSE file
- [a783a8c6]	Add govc CHANGELOG
- [ba707586]	Add commands to configure the autostart manager
- [af6a188e]	Re-enable search index test
- [ceea450c]	Update govc README
- [ea5c9a52]	Fix git dirty status error in build script

<a name="v0.1.0"></a>
## v0.1.0

> Release Date: 2015-03-17

### ⚠️ BREAKING

### 📖 Commits

- [477dcaf9]	Cross-compile govc using gox
- [8593d9c7]	Add version variable that can be set by the linker
- [fb38ca45]	Add CHANGELOG
- [76f8f1a1]	Add package docs to client.go
- [27bf35df]	Use context.Context in client in root package
- [f3b8162f]	Comment out broken test
- [a1d9d1e7]	Drop the _gen filename suffix
- [91650a1f]	Add context.Context argument to object package
- [1814113a]	Use vim25.Client throughout codebase
- [b977114e]	Move property retrieval functions to property package
- [8c3243d8]	Add lightweight client structure to vim25 package
- [ec4b5b85]	Add context.Context argument to find/list packages
- [7eecfbc7]	Make Wait function in property package standalone
- [6c1982c8]	Add keep alive for soap.RoundTripper
- [1324d1f0]	Return nil UserSession when not authenticated
- [ae7ea3dd]	Comments for task.Wait
- [a53a6b2c]	Add context parameter to object.Task functions
- [f6f44097]	Move functionality to wait for task to new package
- [ad2303cf]	Move Ancestors function to vim25/mo
- [fb9e1439]	Move PropertyCollector to new property package
- [a6618591]	Move Reference to vim25/mo
- [bfdb90f1]	Bind virtual machine to guest operation wrappers
- [ec0c16a7]	Move HasFault to vim25/types
- [683ca537]	Move wrappers for managed objects to object package
- [223a07f8]	Add GetServiceContent function to vim25/soap
- [25b07674]	Decouple factory functions from client
- [b96cf609]	Move SessionManager to new session package
- [ea8d5d11]	Return on error in SessionManager
- [7d58a49e]	Mutate copy of parameter instead of parameter itself
- [e158fd95]	Marshal soap.Client instead of govmomi.Client
- [1336ad45]	Embed soap.Client in govmomi.Client
- [15cfd514]	Work with pointer to url.URL
- [be2936f8]	Move guest related wrappers to new guest package
- [b772ba28]	Move LicenseManager to new license package
- [7ac1477f]	Move EventManager to new event package
- [2053e065]	Retrieve dependencies before running test
- [2d14321e]	Add context.Context argument to RoundTripper
- [64f716b2]	Include type of request in summarized debug log
- [40249c87]	Store reference to http.Transport
- [ac77f0c5]	Move debugging code in soap.Client to own struct
- [c8fab31b]	Loosen .ovf match in ova.import
- [9f685e92]	And further fixing the merge... go fmt.
- [8dbb438b]	Merge remote-tracking branch 'upstream/master' into event_manager
- [e57a557c]	created session manager wrapper
- [5525d5c6]	Change return pattern in CreateDatacenter
- [8acd5512]	Update contributors
- [7138d375]	Coding style consistency
- [951e9194]	added SessionIsActive to Client
- [2211e73d]	Add CreateFolder method
- [eef40cc0]	Add Login/Logout functions to client struct
- [3c7dea04]	Update contributors
- [9c4a9202]	Fixed error when attempting to access datastore
- [05ee0e62]	Add PropertiesN function on client struct
- [01ee2fd5]	Adding EventManager so that events can be queried for
- [8d10cfc7]	Restrict permissions on session file
- [88b5d03c]	Key session file off of entire URL
- [9354d314]	Error types for getter functions on finder
- [a30287dc]	Add description for pool.create
- [77466af0]	Prefix option list in help output
- [cbb8d0b2]	Create multiple resource pools through pool.create
- [8d4699d8]	Add usage and description for pool.destroy
- [2e195a92]	Change pool.change to take multiple arguments
- [38e4a2b2]	Add usage and description for pool.info
- [2f286768]	Add usage and description for pool.create
- [413fa901]	Set insert_key = false
- [d6c2b33e]	Update travis.yml
- [b878c20a]	Adding CustomizationSpecManager
- [7c8f3e56]	Add vm mark as vm and mark as template features
- [033d02e9]	Update contributors
- [18919172]	Add cpu and memory usage to host.info
- [b29f93c1]	Adding the RegisterVM task.
- [e6bf8bb5]	Add error types for Finder
- [852578b9]	Support multiple hosts in host.info command
- [f1899c63]	Set InventoryPath field
- [3a5c1cf3]	Add InventoryPath field
- [624f21a4]	Add resource pool cli commands
- [4c7cd61f]	Add ResourcePool wrapper methods
- [761d43e5]	Include ResourcePool in ls -l output
- [d2daf706]	Support nested resource pools in lister
- [4d9d9a72]	Add vm.change cli command
- [e6ebcd7f]	bats fixup: destroy datacenter
- [65838131]	Disable vcsa box password expiration
- [7a6e737b]	Add CONTRIBUTORS file
- [1cbe968d]	Issue [#192](https://github.com/vmware/govmomi/issues/192): HostSystem doesn't seem to be returning the correct host.
- [116a4044]	fix a problem of ignored https_proxy environment variable with https scheme
- [df423c32]	Add create and destroy datacenter to govc.
- [035bd12c]	Usage for devices.{cdrom,floppy}.*
- [68e50dd3]	make storage resource manager
- [b28d6f42]	Specify default network in test helper
- [4b388e67]	Fix boot order test
- [4414a07e]	Expand vm.vnc command
- [e329e6e7]	rename the session file for windows naming check
- [706520fa]	use filepath for filesystem related path operations
- [ceb35f13]	Add -f flag to datastore.rm
- [6498890f]	Default VM memory to 1GiB
- [591b74f4]	Include description for device.cdrom commands
- [815f0286]	Add usage to device.cdrom.insert
- [f2209c2b]	Flag description casing
- [5e52668c]	Add usage to import commands
- [23cf4d35]	Expand datastore.ls
- [bca8ef73]	Expose underlying VimFault through Fault() function
- [90edb2bc]	Add Usage() function to subset of commands
- [afdc145a]	Implement subset of license manager
- [14765d07]	Add net.adapter option to network flag
- [18c2cce0]	Add CreateEthernetCard method
- [9b2730f0]	Don't run vm.destroy if there is no input
- [611ced85]	Add new ops to vm.power command
- [6cd9f466]	Add VM power ops
- [7918063c]	Work on README
- [db17cddd]	Check minimum API version from client flag
- [df075430]	Don't run datastore.rm if there is no input
- [e49a6d57]	Move environment variables names into constants
- [2cfe267f]	Add device.scsi command
- [6df44c1a]	Support scsi disk.controller type in vm.create
- [39a60bbf]	Add CreateSCSIController method
- [136fabe5]	Rename vm.create disk.adapter to disk.controller
- [9c51314c]	Change disk related commands to use new helpers
- [b0c895e5]	Add VirtualDisk support to device helpers
- [a00f4545]	Add helpers for creating disks
- [16283936]	Add FindDiskController helper
- [dda056dc]	Add VirtualDeviceList.FindSCSIController method
- [5402017a]	FindByBackingInfo -> SelectByBackingInfo
- [0ff5759c]	Add vm disk related bats tests
- [8f1e183a]	Output disk file backing in device.info
- [e7cfba4b]	Remove datastore test files
- [6b883be5]	Use DeviceAdd helper in vm.network.add command
- [eb5881ae]	Use device name in vm.network.change command
- [b7503468]	Remove vm.network.remove command
- [0b81619a]	Add vm.network.change cli command
- [0af5c4cf]	Use VirtualDeviceList helpers in vm.network.remove
- [94c62da0]	Add VirtualDeviceList FindByBackingInfo method
- [c247b80c]	Move govc resource finders to govmomi/find package
- [5f0c8dd4]	Add vm.info bats test
- [5d99454d]	mv govc/flags/list -> govmomi/list
- [028bd3ff]	Fix HostSystem.ResourcePool with cluster parent
- [48e25166]	Add ls bats test
- [c5f24bce]	Add host bats test
- [f965c9ad]	Add default GOVC_HOST to vcsim_env
- [77fc8ade]	Add network flag required test
- [b1236bf8]	Add wrapper to manually run govc against vcsim
- [68831a1f]	Fix network device.remove test
- [4649bf1f]	Default vcsim box to 4G memory
- [b3f71333]	Simplify vcsim_env helper
- [2ca11cde]	Answer pending VM questions from govc
- [b6c3ff31]	Move govc/test Vagrant boxes
- [b1b5b26e]	Change network flag to use NetworkReference
- [83f49af7]	Add network bats test
- [a8ffa576]	Add NetworkReference interface
- [6fe62e29]	Add vcsim_env helper
- [a616817d]	Fix collapse_ws helper
- [0614961e]	Add DistributedVirtualPortgroup constructor
- [1ddf6801]	Cache esxcli command info
- [c713b974]	Add table formatter to esxcli command
- [fd19a011]	Include esxcli method info in response
- [3c9a436f]	Explicit exit status check in assert_failure
- [5a63bc06]	Collapse whitespace in assert_line helper
- [c9bd4312]	Change vm.ip -esxcli to wait for ip
- [e97e5604]	boot order test fixups
- [0e128e0d]	32M is plenty of memory for ttylinux
- [85ded933]	Add test cleanup script
- [2bc707e7]	Add device.serial cli commands
- [17fb283a]	Add serial port device related helpers
- [d9b846d1]	Add device.boot tests
- [b5a21e4e]	Add device.floppy cli commands
- [d1d39fc3]	Add floppy device related helpers
- [1e2c54c0]	Refactor disk logic into disk.go
- [9dff8e74]	Fix attach disk error checks
- [0f352ec3]	Add vm.disk.attach
- [bdd7b37b]	Refactor vm.disk.add to vm.disk.create
- [ae2e990e]	Add govc functional tests
- [a707fae6]	Fix alignment for 32-bit go
- [13274292]	Default cli client url.User field
- [17df67ad]	Add device.boot cli command
- [3c345ad7]	Add device.ls -boot option
- [3b25234c]	Add boot order related VirtualDeviceList helpers
- [f996c7d0]	Add VirtualMachine BootOptions wrappers
- [4f3b935b]	Add some DeviceType constants
- [86f90c52]	Add VirtualDeviceList.Type method
- [5f3b95d7]	Output MAC Address in device.info
- [58c3c64e]	Add VirtualMachineList.PrimaryMacAddress helper
- [67fea291]	Fix import.ovf with relative ovf source path
- [22602029]	Support non-disk files in import.ovf
- [92175548]	Add Upload.Headers field
- [f095536d]	Fix import.ova command
- [5093303a]	Add device related govc commands
- [18644254]	Add device list related helpers
- [6803033e]	Add device list helpers
- [4f8cd87c]	Switch to BaseOptionValue for vm extra config
- [76662657]	Regenerate types
- [46ec389f]	Generate interface types for all base types
- [f78df469]	Remove Client param from ResourcePool methods
- [ca3cd417]	Add Client reference to ResourcePool
- [ffc306cc]	Add Client reference to Network
- [c1138fc4]	Remove Client param from HttpNfcLease methods
- [6f983a49]	Add Client reference to HttpNfcLease
- [d2d566d0]	Remove Client param from HostSystem methods
- [60bf1770]	Add Client reference to HostSystem
- [e32542c1]	Remove Client param from HostDatastoreBrowser methods
- [8956959a]	Add Client reference to HostDatastoreBrowser
- [79e7da1d]	Remove Client param from Folder methods
- [68b3e6dc]	Add Client reference to Folder
- [da5b8ec0]	Remove Client param from Datastore methods
- [f89dd25a]	Add Client reference to Datastore
- [1b372efa]	Remove Client param from Datacenter methods
- [ce320403]	Add Client reference to Datacenter
- [b99a9529]	Remove Client param from VirtualMachine methods
- [eb700d65]	Add Client reference to VirtualMachine
- [673485e4]	Remove config check from esxcli.GuestInfo.IpAddress
- [667df16a]	Add VCSA Vagrant box
- [66b7daab]	Use single consistent pattern to populate FlagSet
- [8fa06b5a]	Export NewReference function
- [a4e11a3a]	Check if info is nil before using it
- [8bbe7361]	Add ManagedObject wrappers
- [9d5df71d]	Add vm.ip -esxcli option
- [1818a2a6]	Add esxcli helper for guest related info
- [ac6efdc9]	Use vim.CLIInfo for esxcli command flags and help
- [5b9b34bc]	Remove Cdrom function from disk flag
- [01d201ee]	Use new esxcli command parser
- [7531d60e]	New esxcli command parser
- [a27c9bd5]	Refactor esxcli to esxcli.Executor
- [fdb2d2d0]	Refactor unmarshal
- [2dd9910d]	Add esxcli related types and methods
- [aad819e8]	Add IsoFlag
- [df11fc04]	Handle empty values in esxcli
- [6ceff6a4]	Fix default network in NetworkFlag
- [bc39649d]	Add DistributedVirtualPortgroup wrapper
- [a7eb1d1e]	Add DVS support to NetworkFlag
- [71898a73]	Support DistributedVirtualPortgroup in lister
- [1cf31f03]	Regenerate mo types
- [1e7c1957]	Generate mo types regardless of props
- [549a2712]	tasks are no longer generated
- [fcf2cd94]	Remove unused DiskFlag.Copy method
- [e494c312]	Add DiskFlag adpater option
- [71e5eea2]	Add host.esxcli command
- [5d0fe65c]	Replace panic by error in host system flag
- [a2a7c8ff]	Remove newOvf()
- [d8e94d8f]	Use default host system where possible
- [67835263]	Move HostNetworkSystem getter to HostSystemFlag
- [348258b5]	Move resource pool getter to host system object
- [03f94f4b]	Default URL scheme and path if not specified
- [73b11f40]	Move progress reader to vim25/progress
- [34f73f0a]	Refactored progress reporting infrastructure
- [79f15899]	Include environment variable names in help
- [1de37e80]	Don't skip certificate verification by default
- [4a533b21]	Support ClusterComputeResource in list flag
- [817df9d1]	Include remote path in importable
- [a0944d82]	Import vm.network commands
- [2fd2f026]	Add vm.network.remove command
- [e3307b6f]	Add vm.network.add command
- [2ac39a1e]	Import host.portgroup commands
- [27686532]	Add host.portgroup.remove command
- [46545dd9]	Add host.portgroup.add command
- [29d8ed38]	Add host.vswitch.remove command
- [c2bfbccf]	Add host.vswitch.info command
- [f05e3e0a]	Include host/vswitch commands
- [17094882]	Add host.vswitch.add cli command
- [febf70cb]	Add SearchFlag HostNetworkSystem helper
- [cb41663b]	Add HostSystem ConfigManager getter
- [6f482eb1]	Add HostConfigManager wrapper
- [851cb8d3]	Add HostNetworkSystem wrapper
- [8bb8b613]	Implement flag.Value interface in NetworkFlag
- [41ebd843]	Change destination path for import.vmdk command
- [a6e0f1d4]	Don't create VM if vmx already exists
- [b48f0080]	Check that DiskFlag.Disk exists
- [8fcafba3]	Use DatastoreFlag.Stat method in vmdk.PrepareDestination
- [29daec38]	Add DatastoreFlag Stat method
- [29ca9c4a]	Use aggregate progess in lease updater
- [de422b52]	Enable debug logging with environment variable
- [48690f77]	Add script that summarizes debug trace information
- [c515f6e1]	Add guest.rm cli command
- [ade53d1e]	Remove recursive arg from DeleteFileInGuest
- [63ec87fd]	Add guest.start cli command
- [a5dccc14]	Add guest.kill cli command
- [8e1abdd4]	Add guest.mktemp cli command
- [8d287c3d]	Add guest.ls cli command
- [79a67b2d]	Fix a few tabwriter outputs Stderr -> Stdout
- [6dc9803f]	Remove TODO
- [48a55bbd]	Add guest.ps cli command
- [bed7c508]	vm arg is required for guest ops
- [fc387eb6]	Add example/project links
- [6e75fbf6]	Add example: Create and configure a vCenter VM
- [068cc973]	Add vm.disk.add command
- [ae42925a]	ImportVApp host argument is optional
- [a959e782]	Use OutputFlag.Log for ovf warnings
- [db30f1d4]	Stream uploads directly from the .ova file
- [b0809106]	Add import.Archive interface
- [1faa4e8b]	Add Client.Upload method
- [47fe7028]	Split datastore.import into multiple commands
- [66a468e2]	Rename datastore.delete -> datastore.rm
- [3afcdf5d]	Register commands with explicit name
- [07a12472]	Load fewer properties where possible
- [3f2d9e5e]	Cache rich type info for managed objects
- [be3b5ab1]	Install go vet for travis
- [bcf792a0]	Add go vet to travis script
- [6be65b35]	Rename OutputWrite.WriteTo method to Write
- [b2c603f2]	go vet: format related warnings
- [7403b749]	go vet: composite literal uses unkeyed fields
- [98ac1aaf]	Add NewFolder func
- [eea431c8]	Change NewDatastore signature
- [7dbc2b25]	Add NewResourcePool func
- [8467fbfd]	Support importing VMDKs into ESXi machines
- [8a501f08]	Be specific about channel direction
- [abeb8e83]	Add DeleteVirtualDisk function
- [11d67d27]	Add datastore.import support for .ova files
- [c29ff5c0]	Improve about command output
- [bdeb77fd]	Add progress for ovf datastore.import
- [4810135c]	Ignore PowerOff error in vm.destroy command
- [48c2bbd3]	Stop Ticker in ProgressLogger
- [8dfa7db3]	Include Client.URL's port in ParseURL
- [c2330cf6]	Add progress aggregator for govc
- [d7274985]	Use virtual machine flag for vnc command
- [85649cd8]	Remove vim25/tasks pkg
- [8f1a2803]	Return task objects for every task function
- [df9af568]	Add test for progress reader
- [1ddaf841]	progessReader passthrough is progress channel is nil
- [9544be13]	Add travis ci config
- [bad48a77]	Use time.Equal when comparing time.Time
- [e4aeadc2]	Upload progress for datastore.import
- [295b4597]	Move computation of progress percentage and detail
- [c72543ac]	Remove trailing _ from command name
- [2d96f8a5]	Add progress report to datastore.{upload,download}
- [ae8509c1]	Add functions to about command
- [583a4aca]	Merge branch 'readme'
- [2b03454b]	Add ovf support to datastore.import
- [fcdfafd6]	Add options param to Client.UploadFile
- [0f218092]	Move ParseURL helper to Client
- [9f9996e2]	Published -> available
- [378e32b9]	Fix
- [dae1e4e5]	Use WaitForProperties in VirtualMachine.WaitForIP
- [a0335bac]	Initial govc readme
- [65050902]	Initial govmomi readme
- [231996ff]	Add custom HttpNfcLease helpers
- [3293be25]	Add Client.WaitForProperties method
- [a28b4fc0]	Add generated HttpNfcLease wrapper
- [b610aa5c]	Add ResourcePool.ImportVApp wrapper
- [6f9f316f]	Add OvfManager getter
- [4ab1b230]	Add generated OvfManager wrapper
- [a3e28532]	Use virtual machine flag for guest ops
- [8044501f]	Configure parent disk on create
- [1786687d]	Generate mapping for interface type names
- [f3fa15c1]	Use interface type name if type attr is missing
- [27cda4d6]	Ignore EEXIST on mkdir in guest
- [c4517301]	Use search flag from host system flag
- [9f9b0c9d]	Initialize SearchFlag from Register hook
- [02108dd4]	Call user function before recursing (govc/cli)
- [73b14a66]	Don't overwrite fields (govc/cli)
- [0a5da729]	Prefix search flags with entity name
- [cc6aa166]	Isset -> IsSet
- [af8adde5]	Consistently name pointer receiver 'flag'
- [5cd9a61e]	Rename environment variables GOVMOMI -> GOVC
- [45eca426]	Use list flag to find host system
- [15fe3728]	Use list flag to find resource pool
- [1938ff93]	List resource pool in compute resource
- [051ba306]	Use list flag to find network
- [4015bec1]	Create a VM with a read only parent disk
- [9e98ef07]	Upload disk to import to directory
- [72fa245c]	Split import into upload and import steps
- [d46b4e51]	Add datastore.import command
- [2ba133de]	Move datastore path helper to datastore struct
- [ba92fed2]	Capture request and response bodies in debug mode
- [2660649a]	Add datastore.ls cli command
- [2c8b9fd5]	Add Datastore.Browser method
- [8e6805f5]	Add HostDatastoreBrowser wrapper
- [6aef2e27]	Change generated Base interfaces to a Get method
- [cb0c5763]	Add datastore.cp cli command
- [d7cc920a]	Add datastore.mv cli command
- [3601ab3b]	Add Copy, Move FileManager methods
- [8c62e27e]	Datastore commands take paths as regular arguments
- [a740c827]	Use list flag to find datastore
- [b7d4b208]	Add guest.getenv command
- [e72b79f9]	Add guest.chmod command
- [5bd30d15]	Use FileAttr flag in guest.upload
- [8c889f03]	Add guest FileAttr flag
- [22f854ef]	Fix guest RewriteURL method
- [69af5618]	Retrieve object ancestors if listing a relative path
- [882faef4]	Import vm/guest commands
- [a8fdd5ab]	Add guest.upload command
- [93815beb]	Add guest.download command
- [108f118d]	Add guest.rmdir command
- [77c1f59d]	Add guest.mkdir command
- [98283e1a]	Common flags and helpers for guest command
- [37065ae1]	Add cli flag for guest authentication
- [fc5eb7a7]	More GuestFileManager wrappers
- [8611b851]	Rename cli datastore upload/download receivers
- [2a058397]	Move {Upload,Download}File methods to soap.Client
- [75dfb253]	Use list flag to find datacenter
- [b0557434]	Add GuestOperationManager wrapper
- [9fb9b66e]	Load datacenter name for datastore URL
- [011790a6]	Fix DatastoreFlag lookup
- [b7c12086]	Add vm.ip command
- [17b0879d]	Report progress from vm.power command
- [0a10a798]	Avoid panic if ClientFlag url is not set
- [33c26af2]	Long/short output for ls command
- [0c97323b]	Make traversal of leaf nodes in list code configurable
- [ad0e3778]	Add soap.Client.URL method
- [f3289833]	Destroy multiple VMs
- [90d80fb3]	Power on/off multiple VMs
- [97c2034b]	Rename c -> cmd, client -> c
- [8ea428a4]	Initialize vm commands with search type
- [1e175eab]	Change xml.Decoder.AddType to TypeFunc
- [754da687]	Change xml.Decoder.AddType to TypeFunc
- [8cbebfcb]	Use list flag from search flag
- [43950349]	Use list flag from vm.info cli command
- [0c3080c0]	List relative to configurable object
- [c8438410]	Extract list functionality as flag
- [ad7bac7a]	Support vm.create with -disk .iso
- [189a2231]	Add Isset function to search flag
- [bd38dd9b]	Add vm.destroy cli command
- [24da8d1c]	Add VirtualMachine.Destroy method
- [79466361]	Include client counter in debug file prefix
- [11ac68f1]	Check if session is valid before returning it
- [685f9554]	Return fault from missing set if applicable
- [44575370]	Method fault is a base class
- [f1258736]	Optionally power on vm after creation
- [dd38436e]	Function to map strings to types
- [9a0dde0a]	Return VirtualMachine from CreateVM
- [29c8d2ee]	Return result from Client.waitForTask
- [a368944d]	Move error wrapper to soap package
- [bb62b6a6]	Fix client_test compile
- [e0ce3a86]	DatastoreFlag refactoring
- [4cd1e77f]	Check for DatastorePath required flag
- [79887bf3]	Rename DatastorePath to DatastorePathFlag
- [03f2520e]	Persist session to disk
- [3a9169e2]	Unembed soap.Client from govmomi.Client
- [b510dc18]	Implement vm.create cli command
- [06d2e159]	Add cli Disk flag
- [b0ce5181]	Add cli Network flag
- [8736db1c]	Add cli VmFolder flag
- [05a5e45c]	Add Folder.CreateVM method
- [4d5eb080]	Add VirtualDiskManager wrapper
- [6b4744ac]	Move waitForTask method to Client type
- [7e4d047d]	Remove embedded ClientFlag
- [a16bada5]	Store debug logs on disk
- [e1d7c5b0]	Only care about guest.ipAddress property for -waitip
- [647bd102]	Use cli flag types for host, pool and datastore
- [14d27b9f]	Add cli HostSystemFlag
- [920a5c8a]	Add cli ResourcePoolFlag
- [0f76226c]	Cache Datastore lookup
- [7bb22ee0]	Add govmomi.ResourcePool type
- [6ffac6fc]	Wait for the guest to get an IP address
- [2c361e75]	Import datastore command package
- [5a68e03c]	Add datastore.download command
- [135eb434]	Add datastore.upload command
- [02f40085]	Add datastore.delete command
- [a25d7233]	Add datastore.mkdir command
- [d1f9dad7]	Add cli DatastorePathFlag
- [27789049]	Add cli DatastoreFlag
- [ea66997e]	Add Datastore URL, Upload/Download File methods
- [3c630d4d]	Add FileManager wrapper
- [68ca1c21]	Unembed ServiceContent in govmomi.Client
- [57dd4153]	Enable/disable VNC from govc
- [002cb1dc]	Rename field Ref -> Self
- [5a3968ad]	Add generic list command
- [31664dcf]	Assign reference to self in managed objects
- [99809e14]	Include reference to self in managed objects
- [96b65720]	Rely on response to determine managed object type
- [dcfd55a6]	Include type registry for managed objects
- [46c8fce8]	Load complete object for json output
- [ceb3cfa2]	Use search flag from power command
- [1e19e548]	Add vm.info, host.info commands
- [979d8c48]	Split govc/vm/command.go
- [24ce0371]	Add output flag
- [32693cf3]	Initial stab at listing VMs
- [59734757]	Add datacenter flag
- [382bf2bc]	More verbosity
- [e9a6152d]	Allow embedding of flag types
- [2d2386dc]	Move client flag to flags pkg
- [c2f5e99b]	Nesting of flags through reflection
- [6d6f9baa]	Add SearchIndex wrapper
- [4e06b8ae]	govc cli skeleton
- [444617bb]	Add power on/off and reset functions to VirtualMachine
- [a1377afa]	Add compute resource struct
- [b6aceec1]	Add virtual machine struct
- [415f4cd9]	Add network struct
- [a7b60eb8]	Add datastore struct
- [18ec5f35]	Function to retrieve datacenter folders
- [cebbf289]	Retrieve only childEntity property for folder
- [c7b42438]	Add folder and datacenter types
- [00dce928]	Allow custom request for mo.RetrieveProperties
- [1ac7f6df]	Embed ServiceContent type in govmomi.Client
- [887b482e]	Use cookiejar in soap client
- [3b674be4]	Add basic client structure
- [79f0006e]	Don't use pointer for enum (string) fields
- [29b2981c]	Move generated enum types to their own file
- [5a0e65e5]	Import scripts used for code generation
- [71c53d0e]	Initial import
- [6081afb9]	Add Apache license
- [ff8c717d]	Import modifications to xml package
- [57091273]	Import Go LICENSE file
- [d5645253]	Import encoding/xml from Go 1.3.1
