/*
Copyright (c) 2017-2018 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vpx

import "github.com/vmware/govmomi/vim25/types"

// Description is the default template for the TaskManager description property.
// Capture method:
//   govc object.collect -s -dump TaskManager:TaskManager description
var Description = types.TaskDescription{
	MethodInfo: []types.BaseElementDescription{
		&types.ElementDescription{
			Description: types.Description{
				Label:   "createEntry",
				Summary: "createEntry",
			},
			Key: "host.OperationCleanupManager.createEntry",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateEntry",
				Summary: "updateEntry",
			},
			Key: "host.OperationCleanupManager.updateEntry",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryEntry",
				Summary: "queryEntry",
			},
			Key: "host.OperationCleanupManager.queryEntry",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query disabled guest operations",
				Summary: "Returns a list of guest operations not supported by a virtual machine",
			},
			Key: "vm.guest.GuestOperationsManager.queryDisabledMethods",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateHostSpecification",
				Summary: "updateHostSpecification",
			},
			Key: "profile.host.HostSpecificationManager.updateHostSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateHostSubSpecification",
				Summary: "updateHostSubSpecification",
			},
			Key: "profile.host.HostSpecificationManager.updateHostSubSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveHostSpecification",
				Summary: "retrieveHostSpecification",
			},
			Key: "profile.host.HostSpecificationManager.retrieveHostSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "deleteHostSubSpecification",
				Summary: "deleteHostSubSpecification",
			},
			Key: "profile.host.HostSpecificationManager.deleteHostSubSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "deleteHostSpecification",
				Summary: "deleteHostSpecification",
			},
			Key: "profile.host.HostSpecificationManager.deleteHostSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getUpdatedHosts",
				Summary: "getUpdatedHosts",
			},
			Key: "profile.host.HostSpecificationManager.getUpdatedHosts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set graphics manager custom value",
				Summary: "Sets the value of a custom field of the graphics manager",
			},
			Key: "host.GraphicsManager.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh graphics information",
				Summary: "Refresh graphics device information",
			},
			Key: "host.GraphicsManager.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check if shared graphics is active",
				Summary: "Check if shared graphics is active on the host",
			},
			Key: "host.GraphicsManager.isSharedGraphicsActive",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateGraphicsConfig",
				Summary: "updateGraphicsConfig",
			},
			Key: "host.GraphicsManager.updateGraphicsConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query configuration option descriptor",
				Summary: "Get the list of configuration option keys available in this browser",
			},
			Key: "EnvironmentBrowser.queryConfigOptionDescriptor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure option query",
				Summary: "Search for a specific configuration option",
			},
			Key: "EnvironmentBrowser.queryConfigOption",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryConfigOptionEx",
				Summary: "queryConfigOptionEx",
			},
			Key: "EnvironmentBrowser.queryConfigOptionEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query configuration target",
				Summary: "Search for a specific configuration target",
			},
			Key: "EnvironmentBrowser.queryConfigTarget",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query target capabilities",
				Summary: "Query for compute-resource capabilities associated with this browser",
			},
			Key: "EnvironmentBrowser.queryTargetCapabilities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual machine provisioning operation policy",
				Summary: "Query environment browser for information about the virtual machine provisioning operation policy",
			},
			Key: "EnvironmentBrowser.queryProvisioningPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryConfigTargetSpec",
				Summary: "queryConfigTargetSpec",
			},
			Key: "EnvironmentBrowser.queryConfigTargetSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set scheduled task custom value",
				Summary: "Sets the value of a custom field of a scheduled task",
			},
			Key: "scheduler.ScheduledTask.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove scheduled task",
				Summary: "Remove the scheduled task",
			},
			Key: "scheduler.ScheduledTask.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure scheduled task",
				Summary: "Reconfigure the scheduled task properties",
			},
			Key: "scheduler.ScheduledTask.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Run scheduled task",
				Summary: "Run the scheduled task immediately",
			},
			Key: "scheduler.ScheduledTask.run",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query CMMDS",
				Summary: "Queries CMMDS contents in the vSAN cluster",
			},
			Key: "host.VsanInternalSystem.queryCmmds",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query physical vSAN disks",
				Summary: "Queries the physical vSAN disks",
			},
			Key: "host.VsanInternalSystem.queryPhysicalVsanDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vSAN objects",
				Summary: "Queries the vSAN objects in the cluster",
			},
			Key: "host.VsanInternalSystem.queryVsanObjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vSAN objects on physical disks",
				Summary: "Queries the vSAN objects that have at least one component on the current set of physical disks",
			},
			Key: "host.VsanInternalSystem.queryObjectsOnPhysicalVsanDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Drop ownership of DOM objects",
				Summary: "Drop ownership of the DOM objects that are owned by this host",
			},
			Key: "host.VsanInternalSystem.abdicateDomOwnership",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vSAN statistics",
				Summary: "Gathers low level statistic counters from the vSAN cluster",
			},
			Key: "host.VsanInternalSystem.queryVsanStatistics",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigures vSAN objects",
				Summary: "Reconfigures the vSAN objects in the cluster",
			},
			Key: "host.VsanInternalSystem.reconfigureDomObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vSAN objects that are currently synchronizing data",
				Summary: "Queries vSAN objects that are updating stale components or synchronizing new replicas",
			},
			Key: "host.VsanInternalSystem.querySyncingVsanObjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Run diagnostics on vSAN disks",
				Summary: "Runs diagnostic tests on vSAN physical disks and verifies if objects are successfully created on the disks",
			},
			Key: "host.VsanInternalSystem.runVsanPhysicalDiskDiagnostics",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attributes of vSAN objects",
				Summary: "Shows the extended attributes of the vSAN objects",
			},
			Key: "host.VsanInternalSystem.getVsanObjExtAttrs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configurable vSAN objects",
				Summary: "Identifies the vSAN objects that can be reconfigured using the assigned storage policy in the current cluster",
			},
			Key: "host.VsanInternalSystem.reconfigurationSatisfiable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSAN objects available for provisioning",
				Summary: "Identifies the vSAN objects that are available for provisioning using the assigned storage policy in the current cluster",
			},
			Key: "host.VsanInternalSystem.canProvisionObjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "deleteVsanObjects",
				Summary: "deleteVsanObjects",
			},
			Key: "host.VsanInternalSystem.deleteVsanObjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade vSAN object format",
				Summary: "Upgrade vSAN object format, to fit in vSAN latest features",
			},
			Key: "host.VsanInternalSystem.upgradeVsanObjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryVsanObjectUuidsByFilter",
				Summary: "queryVsanObjectUuidsByFilter",
			},
			Key: "host.VsanInternalSystem.queryVsanObjectUuidsByFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSAN entities available for decommissioning",
				Summary: "Identifies the vSAN entities that are available for decommissioning in the current cluster",
			},
			Key: "host.VsanInternalSystem.canDecommission",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Authenticate credentials in guest",
				Summary: "Authenticate credentials in the guest operating system",
			},
			Key: "vm.guest.AuthManager.validateCredentials",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire credentials in guest",
				Summary: "Acquire credentials in the guest operating system",
			},
			Key: "vm.guest.AuthManager.acquireCredentials",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Release credentials in guest",
				Summary: "Release credentials in the guest operating system",
			},
			Key: "vm.guest.AuthManager.releaseCredentials",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create user",
				Summary: "Creates a local user account",
			},
			Key: "host.LocalAccountManager.createUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update user",
				Summary: "Updates a local user account",
			},
			Key: "host.LocalAccountManager.updateUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create group",
				Summary: "Creates a local group account",
			},
			Key: "host.LocalAccountManager.createGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete user",
				Summary: "Removes a local user account",
			},
			Key: "host.LocalAccountManager.removeUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove group",
				Summary: "Removes a local group account",
			},
			Key: "host.LocalAccountManager.removeGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Assign user to group",
				Summary: "Assign user to group",
			},
			Key: "host.LocalAccountManager.assignUserToGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unassign user from group",
				Summary: "Unassigns a user from a group",
			},
			Key: "host.LocalAccountManager.unassignUserFromGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add an image library",
				Summary: "Register an image library server with vCenter",
			},
			Key: "ImageLibraryManager.addLibrary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update image library",
				Summary: "Update image library information",
			},
			Key: "ImageLibraryManager.updateLibrary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove an image library",
				Summary: "Unregister an image library server from vCenter",
			},
			Key: "ImageLibraryManager.removeLibrary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import from image library",
				Summary: "Import files from the image library",
			},
			Key: "ImageLibraryManager.importLibraryMedia",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export to image library",
				Summary: "Export files to the image library",
			},
			Key: "ImageLibraryManager.exportMediaToLibrary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Publish to image library",
				Summary: "Publish files from datastore to image library",
			},
			Key: "ImageLibraryManager.publishMediaToLibrary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.ContentLibrary.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.ContentLibrary.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.ContentLibrary.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.ContentLibrary.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.ContentLibrary.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.ContentLibrary.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.ContentLibrary.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set EVC manager custom value",
				Summary: "Sets the value of a custom field for an Enhanced vMotion Compatibility manager",
			},
			Key: "cluster.TransitionalEVCManager.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure cluster EVC",
				Summary: "Enable/reconfigure Enhanced vMotion Compatibility for a cluster",
			},
			Key: "cluster.TransitionalEVCManager.configureEVC",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable cluster EVC",
				Summary: "Disable Enhanced vMotion Compatibility for a cluster",
			},
			Key: "cluster.TransitionalEVCManager.disableEVC",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate EVC mode for cluster",
				Summary: "Test the validity of configuring Enhanced vMotion Compatibility mode on the managed cluster",
			},
			Key: "cluster.TransitionalEVCManager.checkConfigureEVC",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate host for EVC cluster",
				Summary: "Tests the validity of adding a host into the Enhanced vMotion Compatibility cluster",
			},
			Key: "cluster.TransitionalEVCManager.checkAddHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "lookupVmOverheadMemory",
				Summary: "lookupVmOverheadMemory",
			},
			Key: "OverheadMemoryManager.lookupVmOverheadMemory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set event history latest page size",
				Summary: "Set the last page viewed size of event history",
			},
			Key: "event.EventHistoryCollector.setLatestPageSize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rewind event history",
				Summary: "Moves view to the oldest item of event history",
			},
			Key: "event.EventHistoryCollector.rewind",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset event history",
				Summary: "Moves view to the newest item of event history",
			},
			Key: "event.EventHistoryCollector.reset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove event history",
				Summary: "Removes the event history collector",
			},
			Key: "event.EventHistoryCollector.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read next event history",
				Summary: "Reads view from current position of event history, and then the position is moved to the next newer page",
			},
			Key: "event.EventHistoryCollector.readNext",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read previous event history",
				Summary: "Reads view from current position of event history and moves the position to the next older page",
			},
			Key: "event.EventHistoryCollector.readPrev",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set managed entity custom value",
				Summary: "Sets the value of a custom field of a managed entity",
			},
			Key: "ManagedEntity.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload managed entity",
				Summary: "Reload the entity state",
			},
			Key: "ManagedEntity.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename managed entity",
				Summary: "Rename this entity",
			},
			Key: "ManagedEntity.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove entity",
				Summary: "Deletes the entity and removes it from parent folder",
			},
			Key: "ManagedEntity.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the entity",
			},
			Key: "ManagedEntity.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the entity",
			},
			Key: "ManagedEntity.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "ManagedEntity.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set a custom value for EVC manager",
				Summary: "Sets a value in the custom field for Enhanced vMotion Compatibility manager",
			},
			Key: "cluster.EVCManager.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable/reconfigure EVC",
				Summary: "Enable/reconfigure Enhanced vMotion Compatibility in a cluster",
			},
			Key: "cluster.EVCManager.configureEvc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable cluster EVC",
				Summary: "Disable Enhanced vMotion Compatibility in a cluster",
			},
			Key: "cluster.EVCManager.disableEvc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate EVC configuration",
				Summary: "Validates the configuration of Enhanced vMotion Compatibility mode in the managed cluster",
			},
			Key: "cluster.EVCManager.checkConfigureEvc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate hosts in EVC",
				Summary: "Validates new hosts in the Enhanced vMotion Compatibility cluster",
			},
			Key: "cluster.EVCManager.checkAddHostEvc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve host profile description",
				Summary: "Retrieve host profile description",
			},
			Key: "profile.host.HostProfile.retrieveDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete host profile",
				Summary: "Delete host profile",
			},
			Key: "profile.host.HostProfile.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach host profile",
				Summary: "Attach host profile to host or cluster",
			},
			Key: "profile.host.HostProfile.associateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detach host profile",
				Summary: "Detach host profile from host or cluster",
			},
			Key: "profile.host.HostProfile.dissociateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compliance",
				Summary: "Check compliance of a host or cluster against a host profile",
			},
			Key: "profile.host.HostProfile.checkCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export host profile",
				Summary: "Export host profile to a file",
			},
			Key: "profile.host.HostProfile.exportProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update reference host",
				Summary: "Update reference host",
			},
			Key: "profile.host.HostProfile.updateReferenceHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update host profile",
				Summary: "Update host profile",
			},
			Key: "profile.host.HostProfile.update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "validate",
				Summary: "validate",
			},
			Key: "profile.host.HostProfile.validate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute profile",
				Summary: "Execute profile",
			},
			Key: "profile.host.HostProfile.execute",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a host profile",
				Summary: "Create a host profile",
			},
			Key: "profile.host.ProfileManager.createProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query policy metadata",
				Summary: "Query policy metadata",
			},
			Key: "profile.host.ProfileManager.queryPolicyMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find associated profile",
				Summary: "Find associated profile",
			},
			Key: "profile.host.ProfileManager.findAssociatedProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply host configuration",
				Summary: "Apply host configuration",
			},
			Key: "profile.host.ProfileManager.applyHostConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryMetadata",
				Summary: "queryMetadata",
			},
			Key: "profile.host.ProfileManager.queryMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate configuration task list for host profile",
				Summary: "Generates a list of configuration tasks to be performed when applying a host profile",
			},
			Key: "profile.host.ProfileManager.generateConfigTaskList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate task list",
				Summary: "Generate task list",
			},
			Key: "profile.host.ProfileManager.generateTaskList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile metadata",
				Summary: "Query profile metadata",
			},
			Key: "profile.host.ProfileManager.queryProfileMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query metadata for profile categories",
				Summary: "Retrieves the metadata for a set of profile categories",
			},
			Key: "profile.host.ProfileManager.queryProfileCategoryMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query metadata for profile components",
				Summary: "Retrieves the metadata for a set of profile components",
			},
			Key: "profile.host.ProfileManager.queryProfileComponentMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile structure",
				Summary: "Gets information about the structure of a profile",
			},
			Key: "profile.host.ProfileManager.queryProfileStructure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create default profile",
				Summary: "Create default profile",
			},
			Key: "profile.host.ProfileManager.createDefaultProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update host customizations",
				Summary: "Update host customizations for host",
			},
			Key: "profile.host.ProfileManager.updateAnswerFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate host customizations",
				Summary: "Validate host customizations for host",
			},
			Key: "profile.host.ProfileManager.validateAnswerFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve host customizations",
				Summary: "Returns the host customization data associated with a particular host",
			},
			Key: "profile.host.ProfileManager.retrieveAnswerFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveAnswerFileForProfile",
				Summary: "retrieveAnswerFileForProfile",
			},
			Key: "profile.host.ProfileManager.retrieveAnswerFileForProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export host customizations",
				Summary: "Export host customizations for host",
			},
			Key: "profile.host.ProfileManager.exportAnswerFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check host customizations status",
				Summary: "Check the status of the host customizations against associated profile",
			},
			Key: "profile.host.ProfileManager.checkAnswerFileStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query host customization status",
				Summary: "Returns the status of the host customization data associated with the specified hosts",
			},
			Key: "profile.host.ProfileManager.queryAnswerFileStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update host customizations",
				Summary: "Update host customizations",
			},
			Key: "profile.host.ProfileManager.updateHostCustomizations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "validateHostCustomizations",
				Summary: "validateHostCustomizations",
			},
			Key: "profile.host.ProfileManager.validateHostCustomizations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveHostCustomizations",
				Summary: "retrieveHostCustomizations",
			},
			Key: "profile.host.ProfileManager.retrieveHostCustomizations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveHostCustomizationsForProfile",
				Summary: "retrieveHostCustomizationsForProfile",
			},
			Key: "profile.host.ProfileManager.retrieveHostCustomizationsForProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export host customizations",
				Summary: "Export host customizations",
			},
			Key: "profile.host.ProfileManager.exportCustomizations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import host customizations",
				Summary: "Import host customizations",
			},
			Key: "profile.host.ProfileManager.importCustomizations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Pre-check Remediation",
				Summary: "Checks customization data and host state is valid for remediation",
			},
			Key: "profile.host.ProfileManager.generateHostConfigTaskSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Batch apply host configuration",
				Summary: "Batch apply host configuration",
			},
			Key: "profile.host.ProfileManager.applyEntitiesConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare validation of settings to be copied",
				Summary: "Generate differences between source and target host profile to validate settings to be copied",
			},
			Key: "profile.host.ProfileManager.validateComposition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Copy settings to host profiles",
				Summary: "Copy settings to host profiles",
			},
			Key: "profile.host.ProfileManager.compositeProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create inventory view",
				Summary: "Create a view for browsing the inventory and tracking changes to open folders",
			},
			Key: "view.ViewManager.createInventoryView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create container view",
				Summary: "Create a view for monitoring the contents of a single container",
			},
			Key: "view.ViewManager.createContainerView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create list view",
				Summary: "Create a view for getting updates",
			},
			Key: "view.ViewManager.createListView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create list view",
				Summary: "Create a list view from an existing view",
			},
			Key: "view.ViewManager.createListViewFromView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add key",
				Summary: "Add the specified key to the current host",
			},
			Key: "encryption.CryptoManager.addKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add keys",
				Summary: "Add the specified keys to the current host",
			},
			Key: "encryption.CryptoManager.addKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove key",
				Summary: "Remove the specified key from the current host",
			},
			Key: "encryption.CryptoManager.removeKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove keys",
				Summary: "Remove the specified keys from the current host",
			},
			Key: "encryption.CryptoManager.removeKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List all keys",
				Summary: "List all the keys registered on the current host",
			},
			Key: "encryption.CryptoManager.listKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.AntiAffinityGroup.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.AntiAffinityGroup.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.AntiAffinityGroup.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.AntiAffinityGroup.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.AntiAffinityGroup.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.AntiAffinityGroup.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.AntiAffinityGroup.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query supported switch specification",
				Summary: "Query supported switch specification",
			},
			Key: "dvs.DistributedVirtualSwitchManager.querySupportedSwitchSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query compatible hosts for a vSphere Distributed Switch specification",
				Summary: "Returns a list of hosts that are compatible with a given vSphere Distributed Switch specification",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryCompatibleHostForNewDvs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query compatible hosts for existing vSphere Distributed Switch",
				Summary: "Returns a list of hosts that are compatible with an existing vSphere Distributed Switch",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryCompatibleHostForExistingDvs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query compatible host specification",
				Summary: "Query compatible host specification",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryCompatibleHostSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query feature capabilities for vSphere Distributed Switch specification",
				Summary: "Queries feature capabilites available for a given vSphere Distributed Switch specification",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryFeatureCapability",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query switch by UUID",
				Summary: "Query switch by UUID",
			},
			Key: "dvs.DistributedVirtualSwitchManager.querySwitchByUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query configuration target",
				Summary: "Query configuration target",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryDvsConfigTarget",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compatibility of hosts against a vSphere Distributed Switch version",
				Summary: "Check compatibility of hosts against a vSphere Distributed Switch version",
			},
			Key: "dvs.DistributedVirtualSwitchManager.checkCompatibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update opaque data for set of entities",
				Summary: "Update opaque data for set of entities",
			},
			Key: "dvs.DistributedVirtualSwitchManager.updateOpaqueData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update opaque data for set of entities",
				Summary: "Update opaque data for set of entities",
			},
			Key: "dvs.DistributedVirtualSwitchManager.updateOpaqueDataEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fetch opaque data for set of entities",
				Summary: "Fetch opaque data for set of entities",
			},
			Key: "dvs.DistributedVirtualSwitchManager.fetchOpaqueData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fetch opaque data for set of entities",
				Summary: "Fetch opaque data for set of entities",
			},
			Key: "dvs.DistributedVirtualSwitchManager.fetchOpaqueDataEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute opaque command for set of entities",
				Summary: "Execute opaque command for set of entities",
			},
			Key: "dvs.DistributedVirtualSwitchManager.executeOpaqueCommand",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rectify vNetwork Distributed Switch host",
				Summary: "Rectify vNetwork Distributed Switch host",
			},
			Key: "dvs.DistributedVirtualSwitchManager.rectifyHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export configuration of the entity",
				Summary: "Export configuration of the entity",
			},
			Key: "dvs.DistributedVirtualSwitchManager.exportEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import configuration of the entity",
				Summary: "Import configuration of the entity",
			},
			Key: "dvs.DistributedVirtualSwitchManager.importEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Look up portgroup based on portgroup key",
				Summary: "Look up portgroup based on portgroup key",
			},
			Key: "dvs.DistributedVirtualSwitchManager.lookupPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query uplink team information",
				Summary: "Query uplink team information",
			},
			Key: "dvs.DistributedVirtualSwitchManager.QueryDvpgUplinkTeam",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryHostNetworkResource",
				Summary: "queryHostNetworkResource",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryHostNetworkResource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryVwirePort",
				Summary: "queryVwirePort",
			},
			Key: "dvs.DistributedVirtualSwitchManager.queryVwirePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compliance of host against profile",
				Summary: "Checks compliance of a host against a profile",
			},
			Key: "profile.host.profileEngine.ComplianceManager.checkHostCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query expression metadata",
				Summary: "Queries the metadata for the given expression names",
			},
			Key: "profile.host.profileEngine.ComplianceManager.queryExpressionMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get the default compliance from host configuration subprofiles",
				Summary: "Get the default compliance from host configuration subprofiles",
			},
			Key: "profile.host.profileEngine.ComplianceManager.getDefaultCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move file",
				Summary: "Move the file, folder, or disk from source datacenter to destination datacenter",
			},
			Key: "FileManager.move",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move file",
				Summary: "Move the source file or folder to destination datacenter",
			},
			Key: "FileManager.moveFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Copy file",
				Summary: "Copy the file, folder, or disk from source datacenter to destination datacenter",
			},
			Key: "FileManager.copy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Copy file",
				Summary: "Copy the source file or folder to destination datacenter",
			},
			Key: "FileManager.copyFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete file",
				Summary: "Delete the file, folder, or disk from source datacenter",
			},
			Key: "FileManager.delete",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete file",
				Summary: "Delete the source file or folder from the datastore",
			},
			Key: "FileManager.deleteFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Make Directory",
				Summary: "Create a directory using the specified name",
			},
			Key: "FileManager.makeDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change owner",
				Summary: "Change the owner of the specified file to the specified user",
			},
			Key: "FileManager.changeOwner",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.TagPolicyOption.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.TagPolicyOption.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.TagPolicyOption.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.TagPolicyOption.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.TagPolicyOption.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.TagPolicyOption.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.TagPolicyOption.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve cluster profile description",
				Summary: "Retrieve cluster profile description",
			},
			Key: "profile.cluster.ClusterProfile.retrieveDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete cluster profile",
				Summary: "Delete cluster profile",
			},
			Key: "profile.cluster.ClusterProfile.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach cluster profile",
				Summary: "Attach cluster profile to cluster",
			},
			Key: "profile.cluster.ClusterProfile.associateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detach cluster profile",
				Summary: "Detach cluster profile from cluster",
			},
			Key: "profile.cluster.ClusterProfile.dissociateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compliance",
				Summary: "Check compliance of a cluster against a cluster profile",
			},
			Key: "profile.cluster.ClusterProfile.checkCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export cluster profile",
				Summary: "Export cluster profile to a file",
			},
			Key: "profile.cluster.ClusterProfile.exportProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update cluster profile",
				Summary: "Update configuration of cluster profile",
			},
			Key: "profile.cluster.ClusterProfile.update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check",
				Summary: "Check for dependencies, conflicts, and obsolete updates",
			},
			Key: "host.PatchManager.Check",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Scan",
				Summary: "Scan the host for patch status",
			},
			Key: "host.PatchManager.Scan",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Scan",
				Summary: "Scan the host for patch status",
			},
			Key: "host.PatchManager.ScanV2",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stage",
				Summary: "Stage the updates to the host",
			},
			Key: "host.PatchManager.Stage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install",
				Summary: "Install the patch",
			},
			Key: "host.PatchManager.Install",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install",
				Summary: "Install the patch",
			},
			Key: "host.PatchManager.InstallV2",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Uninstall",
				Summary: "Uninstall the patch",
			},
			Key: "host.PatchManager.Uninstall",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query",
				Summary: "Query the host for installed bulletins",
			},
			Key: "host.PatchManager.Query",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query process information",
				Summary: "Retrieves information regarding processes",
			},
			Key: "host.SystemDebugManager.queryProcessInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure AutoStart Manager",
				Summary: "Changes the power on or power off sequence",
			},
			Key: "host.AutoStartManager.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Auto power On",
				Summary: "Powers On virtual machines according to the current AutoStart configuration",
			},
			Key: "host.AutoStartManager.autoPowerOn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Auto power Off",
				Summary: "Powers Off virtual machines according to the current AutoStart configuration",
			},
			Key: "host.AutoStartManager.autoPowerOff",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove managed object",
				Summary: "Remove the managed objects",
			},
			Key: "view.ManagedObjectView.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove list view",
				Summary: "Remove the list view object",
			},
			Key: "view.ListView.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Modify list view",
				Summary: "Modify the list view",
			},
			Key: "view.ListView.modify",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset list view",
				Summary: "Reset the list view",
			},
			Key: "view.ListView.reset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset view",
				Summary: "Resets a set of objects in a given view",
			},
			Key: "view.ListView.resetFromView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Creates a registry key",
				Summary: "Creates a registry key in the Windows guest operating system",
			},
			Key: "vm.guest.WindowsRegistryManager.createRegistryKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Lists all registry subkeys for a specified registry key",
				Summary: "Lists all registry subkeys for a specified registry key in the Windows guest operating system.",
			},
			Key: "vm.guest.WindowsRegistryManager.listRegistryKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deletes a registry key",
				Summary: "Deletes a registry key in the Windows guest operating system",
			},
			Key: "vm.guest.WindowsRegistryManager.deleteRegistryKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Sets and creates a registry value",
				Summary: "Sets and creates a registry value in the Windows guest operating system",
			},
			Key: "vm.guest.WindowsRegistryManager.setRegistryValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Lists all registry values for a specified registry key",
				Summary: "Lists all registry values for a specified registry key in the Windows guest operating system",
			},
			Key: "vm.guest.WindowsRegistryManager.listRegistryValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deletes a registry value",
				Summary: "Deletes a registry value in the Windows guest operating system",
			},
			Key: "vm.guest.WindowsRegistryManager.deleteRegistryValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register Fault Tolerant Secondary VM",
				Summary: "Registers a Secondary VM with a Fault Tolerant Primary VM",
			},
			Key: "host.FaultToleranceManager.registerSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister Fault Tolerant Secondary VM",
				Summary: "Unregister a Secondary VM from the associated Primary VM",
			},
			Key: "host.FaultToleranceManager.unregisterSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Make Primary VM",
				Summary: "Test Fault Tolerance failover by making a Secondary VM in a Fault Tolerance pair the Primary VM",
			},
			Key: "host.FaultToleranceManager.makePrimary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Make peer VM primary",
				Summary: "Makes the peer VM primary and terminates the local virtual machine",
			},
			Key: "host.FaultToleranceManager.goLivePeerVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop Fault Tolerant virtual machine",
				Summary: "Stop a specified virtual machine in a Fault Tolerant pair",
			},
			Key: "host.FaultToleranceManager.terminateFaultTolerantVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable Secondary VM",
				Summary: "Disable Fault Tolerance on a specified Secondary VM",
			},
			Key: "host.FaultToleranceManager.disableSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable Secondary VM",
				Summary: "Enable Fault Tolerance on a specified Secondary VM",
			},
			Key: "host.FaultToleranceManager.enableSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start Fault Tolerant Secondary VM",
				Summary: "Start Fault Tolerant Secondary VM on remote host",
			},
			Key: "host.FaultToleranceManager.startSecondaryOnRemoteHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister Fault Tolerance",
				Summary: "Unregister the Fault Tolerance service",
			},
			Key: "host.FaultToleranceManager.unregister",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set local VM component health",
				Summary: "Sets the component health information of the specified local virtual machine",
			},
			Key: "host.FaultToleranceManager.setLocalVMComponentHealth",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get peer VM component health",
				Summary: "Gets component health information of the FT peer of the specified local virtual machine",
			},
			Key: "host.FaultToleranceManager.getPeerVMComponentHealth",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set vCenter HA cluster mode",
				Summary: "Set vCenter HA cluster mode",
			},
			Key: "vcha.FailoverClusterManager.setClusterMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getClusterMode",
				Summary: "getClusterMode",
			},
			Key: "vcha.FailoverClusterManager.getClusterMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getClusterHealth",
				Summary: "getClusterHealth",
			},
			Key: "vcha.FailoverClusterManager.getClusterHealth",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate failover",
				Summary: "Initiate a failover from active vCenter Server node to the passive node",
			},
			Key: "vcha.FailoverClusterManager.initiateFailover",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "executeStep",
				Summary: "executeStep",
			},
			Key: "modularity.WorkflowStepHandler.executeStep",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "undoStep",
				Summary: "undoStep",
			},
			Key: "modularity.WorkflowStepHandler.undoStep",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "finalizeStep",
				Summary: "finalizeStep",
			},
			Key: "modularity.WorkflowStepHandler.finalizeStep",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.TagPolicy.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.TagPolicy.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.TagPolicy.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.TagPolicy.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.TagPolicy.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.TagPolicy.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.TagPolicy.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "CreateVRP",
				Summary: "CreateVRP",
			},
			Key: "VRPResourceManager.CreateVRP",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "UpdateVRP",
				Summary: "UpdateVRP",
			},
			Key: "VRPResourceManager.UpdateVRP",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "DeleteVRP",
				Summary: "DeleteVRP",
			},
			Key: "VRPResourceManager.DeleteVRP",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "DeployVM",
				Summary: "DeployVM",
			},
			Key: "VRPResourceManager.DeployVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "UndeployVM",
				Summary: "UndeployVM",
			},
			Key: "VRPResourceManager.UndeployVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "SetManagedByVDC",
				Summary: "SetManagedByVDC",
			},
			Key: "VRPResourceManager.SetManagedByVDC",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetAllVRPIds",
				Summary: "GetAllVRPIds",
			},
			Key: "VRPResourceManager.GetAllVRPIds",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetRPSettings",
				Summary: "GetRPSettings",
			},
			Key: "VRPResourceManager.GetRPSettings",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetVRPSettings",
				Summary: "GetVRPSettings",
			},
			Key: "VRPResourceManager.GetVRPSettings",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetVRPUsage",
				Summary: "GetVRPUsage",
			},
			Key: "VRPResourceManager.GetVRPUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetVRPofVM",
				Summary: "GetVRPofVM",
			},
			Key: "VRPResourceManager.GetVRPofVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "GetChildRPforHub",
				Summary: "GetChildRPforHub",
			},
			Key: "VRPResourceManager.GetChildRPforHub",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create directory",
				Summary: "Creates a top-level directory on the specified datastore",
			},
			Key: "DatastoreNamespaceManager.CreateDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete directory",
				Summary: "Deletes the specified top-level directory from the datastore",
			},
			Key: "DatastoreNamespaceManager.DeleteDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "ConvertNamespacePathToUuidPath",
				Summary: "ConvertNamespacePathToUuidPath",
			},
			Key: "DatastoreNamespaceManager.ConvertNamespacePathToUuidPath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.VirtualDatacenter.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.VirtualDatacenter.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.VirtualDatacenter.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.VirtualDatacenter.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.VirtualDatacenter.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.VirtualDatacenter.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.VirtualDatacenter.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve profile description",
				Summary: "Retrieve profile description",
			},
			Key: "profile.Profile.retrieveDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove profile",
				Summary: "Remove profile",
			},
			Key: "profile.Profile.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Associate entities",
				Summary: "Associate entities with the profile",
			},
			Key: "profile.Profile.associateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Dissociate entities",
				Summary: "Dissociate entities from the profile",
			},
			Key: "profile.Profile.dissociateEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compliance",
				Summary: "Check compliance against the profile",
			},
			Key: "profile.Profile.checkCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export profile",
				Summary: "Export profile to a file",
			},
			Key: "profile.Profile.exportProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getNetworkIpSettings",
				Summary: "getNetworkIpSettings",
			},
			Key: "vdcs.IpManager.getNetworkIpSettings",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "allocate",
				Summary: "allocate",
			},
			Key: "vdcs.IpManager.allocate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "release",
				Summary: "release",
			},
			Key: "vdcs.IpManager.release",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "releaseAll",
				Summary: "releaseAll",
			},
			Key: "vdcs.IpManager.releaseAll",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryAll",
				Summary: "queryAll",
			},
			Key: "vdcs.IpManager.queryAll",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set datastore cluster custom value",
				Summary: "Sets the value of a custom field of a datastore cluster",
			},
			Key: "StoragePod.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload datastore cluster",
				Summary: "Reloads the datastore cluster",
			},
			Key: "StoragePod.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename a datastore cluster",
				Summary: "Rename a datastore cluster",
			},
			Key: "StoragePod.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove a datastore cluster",
				Summary: "Remove a datastore cluster",
			},
			Key: "StoragePod.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tags to datastore cluster",
				Summary: "Adds a set of tags to a datastore cluster",
			},
			Key: "StoragePod.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tags from datastore cluster",
				Summary: "Removes a set of tags from a datastore cluster",
			},
			Key: "StoragePod.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "StoragePod.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create folder",
				Summary: "Creates a new folder",
			},
			Key: "StoragePod.createFolder",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move datastores into a datastore cluster",
				Summary: "Move datastores into a datastore cluster",
			},
			Key: "StoragePod.moveInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine",
				Summary: "Creates a new virtual machine",
			},
			Key: "StoragePod.createVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register virtual machine",
				Summary: "Adds an existing virtual machine to this datastore cluster",
			},
			Key: "StoragePod.registerVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create cluster",
				Summary: "Creates a new cluster compute-resource in this datastore cluster",
			},
			Key: "StoragePod.createCluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create cluster",
				Summary: "Creates a new cluster compute-resource in this datastore cluster",
			},
			Key: "StoragePod.createClusterEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add standalone host",
				Summary: "Creates a new single-host compute-resource",
			},
			Key: "StoragePod.addStandaloneHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add standalone host and enable lockdown mode",
				Summary: "Creates a new single-host compute-resource and enables lockdown mode on the host",
			},
			Key: "StoragePod.addStandaloneHostWithAdminDisabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create datacenter",
				Summary: "Create a new datacenter with the given name",
			},
			Key: "StoragePod.createDatacenter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister and delete",
				Summary: "Recursively deletes all child virtual machine folders and unregisters all virtual machines",
			},
			Key: "StoragePod.unregisterAndDestroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create vSphere Distributed Switch",
				Summary: "Creates a vSphere Distributed Switch",
			},
			Key: "StoragePod.createDistributedVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create datastore cluster",
				Summary: "Creates a new datastore cluster",
			},
			Key: "StoragePod.createStoragePod",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare to upgrade",
				Summary: "Deletes the content of the temporary directory on the host",
			},
			Key: "AgentManager.prepareToUpgrade",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade",
				Summary: "Validates and executes the installer/uninstaller executable uploaded to the temporary directory",
			},
			Key: "AgentManager.upgrade",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure host power management policy",
				Summary: "Configure host power management policy",
			},
			Key: "host.PowerSystem.configurePolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set network custom Value",
				Summary: "Sets the value of a custom field of a network",
			},
			Key: "Network.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload network",
				Summary: "Reload information about the network",
			},
			Key: "Network.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename network",
				Summary: "Rename network",
			},
			Key: "Network.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete network",
				Summary: "Deletes a network if it is not used by any host or virtual machine",
			},
			Key: "Network.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the network",
			},
			Key: "Network.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the network",
			},
			Key: "Network.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "Network.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove network",
				Summary: "Remove network",
			},
			Key: "Network.destroyNetwork",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve argument description for event type",
				Summary: "Retrieves the argument meta-data for a given event type",
			},
			Key: "event.EventManager.retrieveArgumentDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create event collector",
				Summary: "Creates an event collector to retrieve all server events based on a filter",
			},
			Key: "event.EventManager.createCollector",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Log user event",
				Summary: "Logs a user-defined event",
			},
			Key: "event.EventManager.logUserEvent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get events",
				Summary: "Provides the events selected by the specified filter",
			},
			Key: "event.EventManager.QueryEvent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query events by IDs",
				Summary: "Returns the events specified by a list of IDs",
			},
			Key: "event.EventManager.queryEventsById",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Post event",
				Summary: "Posts the specified event",
			},
			Key: "event.EventManager.postEvent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query latest events in event filter",
				Summary: "Query the latest events in the specified filter",
			},
			Key: "event.EventManager.queryLastEvent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual disk",
				Summary: "Create the disk, either a datastore path or a URL referring to the virtual disk",
			},
			Key: "VirtualDiskManager.createVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete virtual disk",
				Summary: "Delete the disk, either a datastore path or a URL referring to the virtual disk",
			},
			Key: "VirtualDiskManager.deleteVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk information",
				Summary: "Queries information about a virtual disk",
			},
			Key: "VirtualDiskManager.queryVirtualDiskInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move virtual disk",
				Summary: "Move the disk, either a datastore path or a URL referring to the virtual disk",
			},
			Key: "VirtualDiskManager.moveVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Copy virtual disk",
				Summary: "Copy the disk, either a datastore path or a URL referring to the virtual disk",
			},
			Key: "VirtualDiskManager.copyVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend virtual disk",
				Summary: "Expand the capacity of a virtual disk to the new capacity",
			},
			Key: "VirtualDiskManager.extendVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk fragmentation",
				Summary: "Return the percentage of fragmentation of the sparse virtual disk",
			},
			Key: "VirtualDiskManager.queryVirtualDiskFragmentation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Defragment virtual disk",
				Summary: "Defragment a sparse virtual disk",
			},
			Key: "VirtualDiskManager.defragmentVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Shrink virtual disk",
				Summary: "Shrink a sparse virtual disk",
			},
			Key: "VirtualDiskManager.shrinkVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Inflate virtual disk",
				Summary: "Inflate a sparse virtual disk up to the full size",
			},
			Key: "VirtualDiskManager.inflateVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Zero out virtual disk",
				Summary: "Explicitly zero out the virtual disk.",
			},
			Key: "VirtualDiskManager.eagerZeroVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fill virtual disk",
				Summary: "Overwrite all blocks of the virtual disk with zeros",
			},
			Key: "VirtualDiskManager.zeroFillVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Optimally eager zero the virtual disk",
				Summary: "Optimally eager zero a VMFS thick virtual disk.",
			},
			Key: "VirtualDiskManager.optimizeEagerZeroVirtualDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set virtual disk UUID",
				Summary: "Set the UUID for the disk, either a datastore path or a URL referring to the virtual disk",
			},
			Key: "VirtualDiskManager.setVirtualDiskUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk UUID",
				Summary: "Get the virtual disk SCSI inquiry page data",
			},
			Key: "VirtualDiskManager.queryVirtualDiskUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk geometry",
				Summary: "Get the disk geometry information for the virtual disk",
			},
			Key: "VirtualDiskManager.queryVirtualDiskGeometry",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reparent disks",
				Summary: "Reparent disks",
			},
			Key: "VirtualDiskManager.reparentDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a child disk",
				Summary: "Create a new disk and attach it to the end of disk chain specified",
			},
			Key: "VirtualDiskManager.createChildDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "revertToChildDisk",
				Summary: "revertToChildDisk",
			},
			Key: "VirtualDiskManager.revertToChildDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Consolidate disks",
				Summary: "Consolidate a list of disks to the parent most disk",
			},
			Key: "VirtualDiskManager.consolidateDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "importUnmanagedSnapshot",
				Summary: "importUnmanagedSnapshot",
			},
			Key: "VirtualDiskManager.importUnmanagedSnapshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "releaseManagedSnapshot",
				Summary: "releaseManagedSnapshot",
			},
			Key: "VirtualDiskManager.releaseManagedSnapshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "enableUPIT",
				Summary: "enableUPIT",
			},
			Key: "VirtualDiskManager.enableUPIT",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "disableUPIT",
				Summary: "disableUPIT",
			},
			Key: "VirtualDiskManager.disableUPIT",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryObjectInfo",
				Summary: "queryObjectInfo",
			},
			Key: "VirtualDiskManager.queryObjectInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryObjectTypes",
				Summary: "queryObjectTypes",
			},
			Key: "VirtualDiskManager.queryObjectTypes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a virtual disk object",
				Summary: "Create a virtual disk object",
			},
			Key: "vslm.host.VStorageObjectManager.createDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register a legacy disk to be a virtual disk object",
				Summary: "Register a legacy disk to be a virtual disk object",
			},
			Key: "vslm.host.VStorageObjectManager.registerDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend a virtual disk to the new capacity",
				Summary: "Extend a virtual disk to the new capacity",
			},
			Key: "vslm.host.VStorageObjectManager.extendDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Inflate a thin virtual disk",
				Summary: "Inflate a thin virtual disk",
			},
			Key: "vslm.host.VStorageObjectManager.inflateDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename a virtual storage object",
				Summary: "Rename a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.renameVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update storage policy on a virtual storage object",
				Summary: "Update storage policy on a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.updateVStorageObjectPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete a virtual storage object",
				Summary: "Delete a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.deleteVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve a virtual storage object",
				Summary: "Retrieve a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.retrieveVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveVStorageObjectState",
				Summary: "retrieveVStorageObjectState",
			},
			Key: "vslm.host.VStorageObjectManager.retrieveVStorageObjectState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List virtual storage objects on a datastore",
				Summary: "List virtual storage objects on a datastore",
			},
			Key: "vslm.host.VStorageObjectManager.listVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone a virtual storage object",
				Summary: "Clone a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.cloneVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Relocate a virtual storage object",
				Summary: "Relocate a virtual storage object",
			},
			Key: "vslm.host.VStorageObjectManager.relocateVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconcile datastore inventory",
				Summary: "Reconcile datastore inventory",
			},
			Key: "vslm.host.VStorageObjectManager.reconcileDatastoreInventory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Schedule reconcile datastore inventory",
				Summary: "Schedule reconcile datastore inventory",
			},
			Key: "vslm.host.VStorageObjectManager.scheduleReconcileDatastoreInventory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare vMotion send operation",
				Summary: "Prepare a vMotion send operation",
			},
			Key: "host.VMotionManager.prepareSource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare VMotion send operation asynchronously",
				Summary: "Prepares a VMotion send operation asynchronously",
			},
			Key: "host.VMotionManager.prepareSourceEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare vMotion receive operation",
				Summary: "Prepare a vMotion receive operation",
			},
			Key: "host.VMotionManager.prepareDestination",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare vMotion receive operation asynchronously",
				Summary: "Prepares a vMotion receive operation asynchronously",
			},
			Key: "host.VMotionManager.prepareDestinationEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate vMotion receive operation",
				Summary: "Initiate a vMotion receive operation",
			},
			Key: "host.VMotionManager.initiateDestination",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate vMotion send operation",
				Summary: "Initiate a vMotion send operation",
			},
			Key: "host.VMotionManager.initiateSource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate VMotion send operation",
				Summary: "Initiates a VMotion send operation",
			},
			Key: "host.VMotionManager.initiateSourceEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Complete vMotion source notification",
				Summary: "Tell the source that vMotion migration is complete (success or failure)",
			},
			Key: "host.VMotionManager.completeSource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Complete vMotion receive notification",
				Summary: "Tell the destination that vMotion migration is complete (success or failure)",
			},
			Key: "host.VMotionManager.completeDestination",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Commit vMotion destination upgrade",
				Summary: "Reparent the disks at destination and commit the redo logs at the end of a vMotion migration",
			},
			Key: "host.VMotionManager.upgradeDestination",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update VMotionManager memory mirror migrate flag",
				Summary: "Enables or disables VMotionManager memory mirror migrate",
			},
			Key: "host.VMotionManager.updateMemMirrorFlag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryMigrationIds",
				Summary: "queryMigrationIds",
			},
			Key: "host.VMotionManager.queryMigrationIds",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateHostSubSpecificationByFile",
				Summary: "updateHostSubSpecificationByFile",
			},
			Key: "profile.host.profileEngine.HostSpecificationAgent.updateHostSubSpecificationByFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateHostSubSpecificationByData",
				Summary: "updateHostSubSpecificationByData",
			},
			Key: "profile.host.profileEngine.HostSpecificationAgent.updateHostSubSpecificationByData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveHostSpecification",
				Summary: "retrieveHostSpecification",
			},
			Key: "profile.host.profileEngine.HostSpecificationAgent.retrieveHostSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "deleteHostSubSpecification",
				Summary: "deleteHostSubSpecification",
			},
			Key: "profile.host.profileEngine.HostSpecificationAgent.deleteHostSubSpecification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addKey",
				Summary: "addKey",
			},
			Key: "encryption.CryptoManagerKmip.addKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addKeys",
				Summary: "addKeys",
			},
			Key: "encryption.CryptoManagerKmip.addKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeKey",
				Summary: "removeKey",
			},
			Key: "encryption.CryptoManagerKmip.removeKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeKeys",
				Summary: "removeKeys",
			},
			Key: "encryption.CryptoManagerKmip.removeKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "listKeys",
				Summary: "listKeys",
			},
			Key: "encryption.CryptoManagerKmip.listKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "registerKmipServer",
				Summary: "registerKmipServer",
			},
			Key: "encryption.CryptoManagerKmip.registerKmipServer",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "markDefault",
				Summary: "markDefault",
			},
			Key: "encryption.CryptoManagerKmip.markDefault",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateKmipServer",
				Summary: "updateKmipServer",
			},
			Key: "encryption.CryptoManagerKmip.updateKmipServer",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeKmipServer",
				Summary: "removeKmipServer",
			},
			Key: "encryption.CryptoManagerKmip.removeKmipServer",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "listKmipServers",
				Summary: "listKmipServers",
			},
			Key: "encryption.CryptoManagerKmip.listKmipServers",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveKmipServersStatus",
				Summary: "retrieveKmipServersStatus",
			},
			Key: "encryption.CryptoManagerKmip.retrieveKmipServersStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateKey",
				Summary: "generateKey",
			},
			Key: "encryption.CryptoManagerKmip.generateKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveKmipServerCert",
				Summary: "retrieveKmipServerCert",
			},
			Key: "encryption.CryptoManagerKmip.retrieveKmipServerCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "uploadKmipServerCert",
				Summary: "uploadKmipServerCert",
			},
			Key: "encryption.CryptoManagerKmip.uploadKmipServerCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateSelfSignedClientCert",
				Summary: "generateSelfSignedClientCert",
			},
			Key: "encryption.CryptoManagerKmip.generateSelfSignedClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateClientCsr",
				Summary: "generateClientCsr",
			},
			Key: "encryption.CryptoManagerKmip.generateClientCsr",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveSelfSignedClientCert",
				Summary: "retrieveSelfSignedClientCert",
			},
			Key: "encryption.CryptoManagerKmip.retrieveSelfSignedClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveClientCsr",
				Summary: "retrieveClientCsr",
			},
			Key: "encryption.CryptoManagerKmip.retrieveClientCsr",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveClientCert",
				Summary: "retrieveClientCert",
			},
			Key: "encryption.CryptoManagerKmip.retrieveClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateSelfSignedClientCert",
				Summary: "updateSelfSignedClientCert",
			},
			Key: "encryption.CryptoManagerKmip.updateSelfSignedClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateKmsSignedCsrClientCert",
				Summary: "updateKmsSignedCsrClientCert",
			},
			Key: "encryption.CryptoManagerKmip.updateKmsSignedCsrClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "uploadClientCert",
				Summary: "uploadClientCert",
			},
			Key: "encryption.CryptoManagerKmip.uploadClientCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister extension",
				Summary: "Unregisters an extension",
			},
			Key: "ExtensionManager.unregisterExtension",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find extension",
				Summary: "Find an extension",
			},
			Key: "ExtensionManager.findExtension",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register extension",
				Summary: "Registers an extension",
			},
			Key: "ExtensionManager.registerExtension",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update extension",
				Summary: "Updates extension information",
			},
			Key: "ExtensionManager.updateExtension",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get server public key",
				Summary: "Get vCenter Server's public key",
			},
			Key: "ExtensionManager.getPublicKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set extension public key",
				Summary: "Set public key of the extension",
			},
			Key: "ExtensionManager.setPublicKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set extension certificate",
				Summary: "Update the stored authentication certificate for a specified extension",
			},
			Key: "ExtensionManager.setCertificate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update extension data",
				Summary: "Updates extension-specific data associated with an extension",
			},
			Key: "ExtensionManager.updateExtensionData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query extension data",
				Summary: "Retrieves extension-specific data associated with an extension",
			},
			Key: "ExtensionManager.queryExtensionData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query extension data keys",
				Summary: "Retrieves extension-specific data keys associated with an extension",
			},
			Key: "ExtensionManager.queryExtensionDataKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear extension data",
				Summary: "Clears extension-specific data associated with an extension",
			},
			Key: "ExtensionManager.clearExtensionData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query extension data usage",
				Summary: "Retrieves statistics about the amount of data being stored by extensions registered with vCenter Server",
			},
			Key: "ExtensionManager.queryExtensionDataUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query entities managed by extension",
				Summary: "Finds entities managed by an extension",
			},
			Key: "ExtensionManager.queryManagedBy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query statistics about IP allocation usage",
				Summary: "Query statistics about IP allocation usage, system-wide or for specified extensions",
			},
			Key: "ExtensionManager.queryExtensionIpAllocationUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable replication of virtual machine",
				Summary: "Enable replication of virtual machine",
			},
			Key: "HbrManager.enableReplication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable replication of virtual machine",
				Summary: "Disable replication of virtual machine",
			},
			Key: "HbrManager.disableReplication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure replication for virtual machine",
				Summary: "Reconfigure replication for virtual machine",
			},
			Key: "HbrManager.reconfigureReplication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve replication configuration of virtual machine",
				Summary: "Retrieve replication configuration of virtual machine",
			},
			Key: "HbrManager.retrieveReplicationConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Pause replication of virtual machine",
				Summary: "Pause replication of virtual machine",
			},
			Key: "HbrManager.pauseReplication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resume replication of virtual machine",
				Summary: "Resume replication of virtual machine",
			},
			Key: "HbrManager.resumeReplication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start a replication resynchronization for virtual machine",
				Summary: "Start a replication resynchronization for virtual machine",
			},
			Key: "HbrManager.fullSync",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start new replication instance for virtual machine",
				Summary: "Start extraction and transfer of a new replication instance for virtual machine",
			},
			Key: "HbrManager.createInstance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Replicate powered-off virtual machine",
				Summary: "Transfer a replication instance for powered-off virtual machine",
			},
			Key: "HbrManager.startOfflineInstance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop replication of powered-off virtual machine",
				Summary: "Stop replication of powered-off virtual machine",
			},
			Key: "HbrManager.stopOfflineInstance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual machine replication state",
				Summary: "Qureies the current state of a replicated virtual machine",
			},
			Key: "HbrManager.queryReplicationState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryReplicationCapabilities",
				Summary: "queryReplicationCapabilities",
			},
			Key: "HbrManager.queryReplicationCapabilities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set storage custom value",
				Summary: "Sets the value of a custom field of a host storage system",
			},
			Key: "host.StorageSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve disk partition information",
				Summary: "Gets the partition information for the disks named by the device names",
			},
			Key: "host.StorageSystem.retrieveDiskPartitionInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Compute disk partition information",
				Summary: "Computes the disk partition information given the desired disk layout",
			},
			Key: "host.StorageSystem.computeDiskPartitionInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Compute disk partition information for resize",
				Summary: "Compute disk partition information for resizing a partition",
			},
			Key: "host.StorageSystem.computeDiskPartitionInfoForResize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update disk partitions",
				Summary: "Change the partitions on the disk by supplying a partition specification and the device name",
			},
			Key: "host.StorageSystem.updateDiskPartitions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Format VMFS",
				Summary: "Formats a new VMFS on a disk partition",
			},
			Key: "host.StorageSystem.formatVmfs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mount VMFS volume",
				Summary: "Mounts an unmounted VMFS volume",
			},
			Key: "host.StorageSystem.mountVmfsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmount VMFS volume",
				Summary: "Unmount a mounted VMFS volume",
			},
			Key: "host.StorageSystem.unmountVmfsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmount VMFS volumes",
				Summary: "Unmounts one or more mounted VMFS volumes",
			},
			Key: "host.StorageSystem.unmountVmfsVolumeEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "mountVmfsVolumeEx",
				Summary: "mountVmfsVolumeEx",
			},
			Key: "host.StorageSystem.mountVmfsVolumeEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unmapVmfsVolumeEx",
				Summary: "unmapVmfsVolumeEx",
			},
			Key: "host.StorageSystem.unmapVmfsVolumeEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete state information for unmounted VMFS volume",
				Summary: "Removes the state information for a previously unmounted VMFS volume",
			},
			Key: "host.StorageSystem.deleteVmfsVolumeState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rescan VMFS",
				Summary: "Rescan for new VMFS volumes",
			},
			Key: "host.StorageSystem.rescanVmfs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend VMFS",
				Summary: "Extend a VMFS by attaching a disk partition",
			},
			Key: "host.StorageSystem.attachVmfsExtent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Expand VMFS extent",
				Summary: "Expand the capacity of the VMFS extent",
			},
			Key: "host.StorageSystem.expandVmfsExtent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade VMFS",
				Summary: "Upgrade the VMFS to the current VMFS version",
			},
			Key: "host.StorageSystem.upgradeVmfs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Relocate virtual machine disks",
				Summary: "Relocate the disks for all virtual machines into directories if stored in the ROOT",
			},
			Key: "host.StorageSystem.upgradeVmLayout",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query unbound VMFS volumes",
				Summary: "Query for the list of unbound VMFS volumes",
			},
			Key: "host.StorageSystem.queryUnresolvedVmfsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resolve VMFS volumes",
				Summary: "Resolve the detected copies of VMFS volumes",
			},
			Key: "host.StorageSystem.resolveMultipleUnresolvedVmfsVolumes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resolve VMFS volumes",
				Summary: "Resolves the detected copies of VMFS volumes",
			},
			Key: "host.StorageSystem.resolveMultipleUnresolvedVmfsVolumesEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmount force mounted VMFS",
				Summary: "Unmounts a force mounted VMFS volume",
			},
			Key: "host.StorageSystem.unmountForceMountedVmfsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rescan HBA",
				Summary: "Rescan a specific storage adapter for new storage devices",
			},
			Key: "host.StorageSystem.rescanHba",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rescan all HBAs",
				Summary: "Rescan all storage adapters for new storage devices",
			},
			Key: "host.StorageSystem.rescanAllHba",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change Software Internet SCSI Status",
				Summary: "Enables or disables Software Internet SCSI",
			},
			Key: "host.StorageSystem.updateSoftwareInternetScsiEnabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI discovery properties",
				Summary: "Updates the discovery properties for an Internet SCSI host bus adapter",
			},
			Key: "host.StorageSystem.updateInternetScsiDiscoveryProperties",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI authentication properties",
				Summary: "Updates the authentication properties for an Internet SCSI host bus adapter",
			},
			Key: "host.StorageSystem.updateInternetScsiAuthenticationProperties",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI digest properties",
				Summary: "Update the digest properties of an Internet SCSI host bus adapter or target",
			},
			Key: "host.StorageSystem.updateInternetScsiDigestProperties",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI advanced options",
				Summary: "Update the advanced options of an Internet SCSI host bus adapter or target",
			},
			Key: "host.StorageSystem.updateInternetScsiAdvancedOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI IP properties",
				Summary: "Updates the IP properties for an Internet SCSI host bus adapter",
			},
			Key: "host.StorageSystem.updateInternetScsiIPProperties",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI name",
				Summary: "Updates the name of an Internet SCSI host bus adapter",
			},
			Key: "host.StorageSystem.updateInternetScsiName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Internet SCSI alias",
				Summary: "Updates the alias of an Internet SCSI host bus adapter",
			},
			Key: "host.StorageSystem.updateInternetScsiAlias",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Internet SCSI send targets",
				Summary: "Adds send target entries to the host bus adapter discovery list",
			},
			Key: "host.StorageSystem.addInternetScsiSendTargets",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove Internet SCSI send targets",
				Summary: "Removes send target entries from the host bus adapter discovery list",
			},
			Key: "host.StorageSystem.removeInternetScsiSendTargets",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Internet SCSI static targets ",
				Summary: "Adds static target entries to the host bus adapter discovery list",
			},
			Key: "host.StorageSystem.addInternetScsiStaticTargets",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove Internet SCSI static targets",
				Summary: "Removes static target entries from the host bus adapter discovery list",
			},
			Key: "host.StorageSystem.removeInternetScsiStaticTargets",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable multiple path",
				Summary: "Enable a path for a logical unit",
			},
			Key: "host.StorageSystem.enableMultipathPath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable multiple path",
				Summary: "Disable a path for a logical unit",
			},
			Key: "host.StorageSystem.disableMultipathPath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set logical unit policy",
				Summary: "Set the multipath policy for a logical unit ",
			},
			Key: "host.StorageSystem.setMultipathLunPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query path selection policy options",
				Summary: "Queries the set of path selection policy options",
			},
			Key: "host.StorageSystem.queryPathSelectionPolicyOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query storage array type policy options",
				Summary: "Queries the set of storage array type policy options",
			},
			Key: "host.StorageSystem.queryStorageArrayTypePolicyOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update SCSI LUN display name",
				Summary: "Updates the display name of a SCSI LUN",
			},
			Key: "host.StorageSystem.updateScsiLunDisplayName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detach SCSI LUN",
				Summary: "Blocks I/O operations to the attached SCSI LUN",
			},
			Key: "host.StorageSystem.detachScsiLun",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detach SCSI LUNs",
				Summary: "Blocks I/O operations to one or more attached SCSI LUNs",
			},
			Key: "host.StorageSystem.detachScsiLunEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete state information for detached SCSI LUN",
				Summary: "Removes the state information for a previously detached SCSI LUN",
			},
			Key: "host.StorageSystem.deleteScsiLunState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach SCSI LUN",
				Summary: "Allow I/O issue to the specified detached SCSI LUN",
			},
			Key: "host.StorageSystem.attachScsiLun",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach SCSI LUNs",
				Summary: "Enables I/O operations to one or more detached SCSI LUNs",
			},
			Key: "host.StorageSystem.attachScsiLunEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh host storage system",
				Summary: "Refresh the storage information and settings to pick up any changes that have occurred",
			},
			Key: "host.StorageSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Discover FCOE storage",
				Summary: "Discovers new storage using FCOE",
			},
			Key: "host.StorageSystem.discoverFcoeHbas",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update FCOE HBA state",
				Summary: "Mark or unmark the specified FCOE HBA for removal from the host system",
			},
			Key: "host.StorageSystem.markForRemoval",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Format VFFS",
				Summary: "Formats a new VFFS on a SSD disk",
			},
			Key: "host.StorageSystem.formatVffs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend VFFS",
				Summary: "Extends a VFFS by attaching a SSD disk",
			},
			Key: "host.StorageSystem.extendVffs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete VFFS",
				Summary: "Deletes a VFFS from the host",
			},
			Key: "host.StorageSystem.destroyVffs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mounts VFFS volume",
				Summary: "Mounts an unmounted VFFS volume",
			},
			Key: "host.StorageSystem.mountVffsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmounts VFFS volume",
				Summary: "Unmounts a mounted VFFS volume",
			},
			Key: "host.StorageSystem.unmountVffsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete state information for unmounted VFFS volume",
				Summary: "Removes the state information for a previously unmounted VFFS volume",
			},
			Key: "host.StorageSystem.deleteVffsVolumeState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rescan VFFS",
				Summary: "Rescans for new VFFS volumes",
			},
			Key: "host.StorageSystem.rescanVffs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query available SSD disks",
				Summary: "Queries available SSD disks",
			},
			Key: "host.StorageSystem.queryAvailableSsds",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set NFS user",
				Summary: "Sets an NFS user",
			},
			Key: "host.StorageSystem.setNFSUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change NFS user password",
				Summary: "Changes the password of an NFS user",
			},
			Key: "host.StorageSystem.changeNFSUserPassword",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query NFS user",
				Summary: "Queries an NFS user",
			},
			Key: "host.StorageSystem.queryNFSUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear NFS user",
				Summary: "Deletes an NFS user",
			},
			Key: "host.StorageSystem.clearNFSUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Turn on disk locator LEDs",
				Summary: "Turns on one or more disk locator LEDs",
			},
			Key: "host.StorageSystem.turnDiskLocatorLedOn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Turn off locator LEDs",
				Summary: "Turns off one or more disk locator LEDs",
			},
			Key: "host.StorageSystem.turnDiskLocatorLedOff",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark the disk as a flash disk",
				Summary: "Marks the disk as a flash disk",
			},
			Key: "host.StorageSystem.markAsSsd",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark the disk as a HDD disk",
				Summary: "Marks the disk as a HDD disk",
			},
			Key: "host.StorageSystem.markAsNonSsd",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark the disk as a local disk",
				Summary: "Marks the disk as a local disk",
			},
			Key: "host.StorageSystem.markAsLocal",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark the disk as a remote disk",
				Summary: "Marks the disk as a remote disk",
			},
			Key: "host.StorageSystem.markAsNonLocal",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "QueryIoFilterProviderId",
				Summary: "QueryIoFilterProviderId",
			},
			Key: "host.StorageSystem.QueryIoFilterProviderId",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "FetchIoFilterSharedSecret",
				Summary: "FetchIoFilterSharedSecret",
			},
			Key: "host.StorageSystem.FetchIoFilterSharedSecret",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update VMFS unmap priority",
				Summary: "Updates the priority of VMFS space reclamation operation",
			},
			Key: "host.StorageSystem.updateVmfsUnmapPriority",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query VMFS config option",
				Summary: "Query VMFS config option",
			},
			Key: "host.StorageSystem.queryVmfsConfigOption",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Evaluate vMotion migration of VMs to hosts",
				Summary: "Checks whether the specified VMs can be migrated with vMotion to all the specified hosts",
			},
			Key: "vm.check.ProvisioningChecker.queryVMotionCompatibilityEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Evaluate migration of VM to destination",
				Summary: "Checks whether the VM can be migrated to the specified destination host, resource pool, and datastores",
			},
			Key: "vm.check.ProvisioningChecker.checkMigrate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Evaluate relocation of VM to destination",
				Summary: "Checks whether the VM can be relocated to the specified destination host, resource pool, and datastores",
			},
			Key: "vm.check.ProvisioningChecker.checkRelocate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Evaluate cloning VM to destination",
				Summary: "Checks whether the VM can be cloned to the specified destination host, resource pool, and datastores",
			},
			Key: "vm.check.ProvisioningChecker.checkClone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "checkInstantClone",
				Summary: "checkInstantClone",
			},
			Key: "vm.check.ProvisioningChecker.checkInstantClone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create cluster profile",
				Summary: "Create cluster profile",
			},
			Key: "profile.cluster.ProfileManager.createProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query policy metadata",
				Summary: "Query policy metadata",
			},
			Key: "profile.cluster.ProfileManager.queryPolicyMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find associated profile",
				Summary: "Find associated profile",
			},
			Key: "profile.cluster.ProfileManager.findAssociatedProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare a vCenter HA setup",
				Summary: "Prepare vCenter HA setup on the local vCenter Server",
			},
			Key: "vcha.FailoverClusterConfigurator.prepare",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy a vCenter HA cluster",
				Summary: "Deploy and configure vCenter HA on the local vCenter Server",
			},
			Key: "vcha.FailoverClusterConfigurator.deploy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure a vCenter HA cluster",
				Summary: "Configure vCenter HA on the local vCenter Server",
			},
			Key: "vcha.FailoverClusterConfigurator.configure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create passive node",
				Summary: "Create a passive node in a vCenter HA Cluster",
			},
			Key: "vcha.FailoverClusterConfigurator.createPassiveNode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create witness node",
				Summary: "Create a witness node in a vCenter HA Cluster",
			},
			Key: "vcha.FailoverClusterConfigurator.createWitnessNode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getConfig",
				Summary: "getConfig",
			},
			Key: "vcha.FailoverClusterConfigurator.getConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Destroy the vCenter HA cluster",
				Summary: "Destroy the vCenter HA cluster setup and remove all configuration files",
			},
			Key: "vcha.FailoverClusterConfigurator.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get current time",
				Summary: "Returns the current time on the server",
			},
			Key: "ServiceInstance.currentTime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve content",
				Summary: "Get the properties of the service instance",
			},
			Key: "ServiceInstance.retrieveContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve internal properties",
				Summary: "Retrieves the internal properties of the service instance",
			},
			Key: "ServiceInstance.retrieveInternalContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate migration",
				Summary: "Checks for errors and warnings of virtual machines migrated from one host to another",
			},
			Key: "ServiceInstance.validateMigration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vMotion compatibility",
				Summary: "Validates the vMotion compatibility of a set of hosts",
			},
			Key: "ServiceInstance.queryVMotionCompatibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve product components",
				Summary: "Component information for bundled products",
			},
			Key: "ServiceInstance.retrieveProductComponents",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create vSphere Distributed Switch",
				Summary: "Create vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.createDistributedVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove vSphere Distributed Switch",
				Summary: "Remove vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.removeDistributedVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure vSphere Distributed Switch",
				Summary: "Reconfigure vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.reconfigureDistributedVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update dvPort",
				Summary: "Update dvPort",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.updatePorts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete ports",
				Summary: "Delete ports",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.deletePorts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve port state",
				Summary: "Retrieve port state",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.fetchPortState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone port",
				Summary: "Clone port",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.clonePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve vSphere Distributed Switch configuration specification",
				Summary: "Retrieve vSphere Distributed Switch configuration specification",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.retrieveDvsConfigSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Distributed Port Groups",
				Summary: "Update Distributed Port Group",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.updateDVPortgroups",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve port group keys for vSphere Distributed Switch",
				Summary: "Retrieve the list of port group keys on a given vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.retrieveDVPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve distributed virtual port group specification",
				Summary: "Retrievs the configuration specification for distributed virtual port groups",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.retrieveDVPortgroupConfigSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Load port",
				Summary: "Load port",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.loadDVPort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve the list of port keys on the given vSphere Distributed Switch",
				Summary: "Retrieve the list of port keys on the given vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.retrieveDVPort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update dvPorts",
				Summary: "Update dvPort",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDVPort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Distributed Port Groups",
				Summary: "Update Distributed Port Group",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDVPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSphere Distributed Switch",
				Summary: "Update vSphere Distributed Switch",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDvs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSphere Distributed Switch list",
				Summary: "Update vSphere Distributed Switch list",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDvsList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Distributed Port Group list",
				Summary: "Update Distributed Port Group list",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDVPortgroupList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update dvPort list",
				Summary: "Update dvPort list",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.applyDVPortList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute opaque command",
				Summary: "Execute opaque command",
			},
			Key: "dvs.HostDistributedVirtualSwitchManager.executeOpaqueCommand",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create default host profile of specified type",
				Summary: "Creates a default host profile of the specified type",
			},
			Key: "profile.host.profileEngine.HostProfileManager.createDefaultProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile policy option metadata",
				Summary: "Gets the profile policy option metadata for the specified policy names",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryPolicyMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile metadata",
				Summary: "Gets the profile metadata for the specified profile names and profile types",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryProfileMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile category metadata",
				Summary: "Gets the profile category metadata for the specified category names",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryProfileCategoryMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile component metadata",
				Summary: "Gets the profile component metadata for the specified component names",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryProfileComponentMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute host profile manager engine",
				Summary: "Executes the host profile manager engine",
			},
			Key: "profile.host.profileEngine.HostProfileManager.execute",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Bookkeep host profile",
				Summary: "Bookkeep host profile",
			},
			Key: "profile.host.profileEngine.HostProfileManager.bookKeep",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve profile description",
				Summary: "Retrieves description of a profile",
			},
			Key: "profile.host.profileEngine.HostProfileManager.retrieveProfileDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update configuration tasks from host configuration",
				Summary: "Update configuration tasks from host configuration",
			},
			Key: "profile.host.profileEngine.HostProfileManager.updateTaskConfigSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateTaskList",
				Summary: "generateTaskList",
			},
			Key: "profile.host.profileEngine.HostProfileManager.generateTaskList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateHostConfigTaskSpec",
				Summary: "generateHostConfigTaskSpec",
			},
			Key: "profile.host.profileEngine.HostProfileManager.generateHostConfigTaskSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve profile from host configuration",
				Summary: "Retrieves a profile from the host's configuration",
			},
			Key: "profile.host.profileEngine.HostProfileManager.retrieveProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare host profile for export",
				Summary: "Prepares a host profile for export",
			},
			Key: "profile.host.profileEngine.HostProfileManager.prepareExport",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query user input policy options",
				Summary: "Gets a list of policy options that are set to require user inputs",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryUserInputPolicyOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query profile structure",
				Summary: "Gets information about the structure of a profile",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryProfileStructure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply host configuration",
				Summary: "Applies the specified host configuration to the host",
			},
			Key: "profile.host.profileEngine.HostProfileManager.applyHostConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query host profile manager state",
				Summary: "Gets the current state of the host profile manager and plug-ins on a host",
			},
			Key: "profile.host.profileEngine.HostProfileManager.queryState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check virtual machine's compatibility on host",
				Summary: "Checks whether a virtual machine is compatible on a host",
			},
			Key: "vm.check.CompatibilityChecker.checkCompatibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compatibility of a VM specification on a host",
				Summary: "Checks compatibility of a VM specification on a host",
			},
			Key: "vm.check.CompatibilityChecker.checkVMCompatibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query service list",
				Summary: "Location information that needs to match a service",
			},
			Key: "ServiceManager.queryServiceList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove inventory view",
				Summary: "Remove the inventory view object",
			},
			Key: "view.InventoryView.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Open inventory view folder",
				Summary: "Adds the child objects of a given managed entity to the view",
			},
			Key: "view.InventoryView.openFolder",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Close inventory view",
				Summary: "Notify the server that folders have been closed",
			},
			Key: "view.InventoryView.closeFolder",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure host update proxy",
				Summary: "Reconfigure host update proxy",
			},
			Key: "host.HostUpdateProxyManager.reconfigureHostUpdateProxy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve configuration of the host update proxy",
				Summary: "Retrieve configuration of the host update proxy",
			},
			Key: "host.HostUpdateProxyManager.retrieveHostUpdateProxyConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update VMCI access rights",
				Summary: "Updates VMCI (Virtual Machine Communication Interface) access rights for one or more virtual machines",
			},
			Key: "host.VmciAccessManager.updateAccess",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve VMCI service rights granted to virtual machine",
				Summary: "Retrieve VMCI (Virtual Machine Communication Interface) service rights granted to a VM",
			},
			Key: "host.VmciAccessManager.retrieveGrantedServices",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual machines with access to VMCI service",
				Summary: "Gets the VMs with granted access to a service",
			},
			Key: "host.VmciAccessManager.queryAccessToService",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "promoteDisks",
				Summary: "promoteDisks",
			},
			Key: "host.LowLevelProvisioningManager.promoteDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine",
				Summary: "Creates a virtual machine on disk",
			},
			Key: "host.LowLevelProvisioningManager.createVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete virtual machine",
				Summary: "Deletes a virtual machine on disk",
			},
			Key: "host.LowLevelProvisioningManager.deleteVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete virtual machine without deleting its virtual disks",
				Summary: "Deletes a virtual machine from its storage, all virtual machine files are deleted except its associated virtual disks",
			},
			Key: "host.LowLevelProvisioningManager.deleteVmExceptDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve virtual machine recovery information",
				Summary: "Retrieves virtual machine recovery information",
			},
			Key: "host.LowLevelProvisioningManager.retrieveVmRecoveryInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve last virtual machine migration status",
				Summary: "Retrieves the last virtual machine migration status if available",
			},
			Key: "host.LowLevelProvisioningManager.retrieveLastVmMigrationStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure virtual machine",
				Summary: "Reconfigures the virtual machine",
			},
			Key: "host.LowLevelProvisioningManager.reconfigVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload disks",
				Summary: "Reloads virtual disk information",
			},
			Key: "host.LowLevelProvisioningManager.reloadDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Consolidate disks",
				Summary: "Consolidates virtual disks",
			},
			Key: "host.LowLevelProvisioningManager.consolidateDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update snapshot layout information",
				Summary: "Updates the snapshot layout information of a virtual machine and reloads its snapshots",
			},
			Key: "host.LowLevelProvisioningManager.relayoutSnapshots",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reserve files for provisioning",
				Summary: "Reserves files or directories on a datastore to be used for a provisioning",
			},
			Key: "host.LowLevelProvisioningManager.reserveFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete files",
				Summary: "Deletes a list of files from a datastore",
			},
			Key: "host.LowLevelProvisioningManager.deleteFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extract NVRAM content",
				Summary: "Extracts the NVRAM content from a checkpoint file",
			},
			Key: "host.LowLevelProvisioningManager.extractNvramContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "setCustomValue",
				Summary: "setCustomValue",
			},
			Key: "external.ContentLibraryItem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "reload",
				Summary: "reload",
			},
			Key: "external.ContentLibraryItem.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rename",
				Summary: "rename",
			},
			Key: "external.ContentLibraryItem.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "destroy",
				Summary: "destroy",
			},
			Key: "external.ContentLibraryItem.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addTag",
				Summary: "addTag",
			},
			Key: "external.ContentLibraryItem.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeTag",
				Summary: "removeTag",
			},
			Key: "external.ContentLibraryItem.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "external.ContentLibraryItem.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete container view",
				Summary: "Remove a list view object from current contents of this view",
			},
			Key: "view.ContainerView.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set datastore custom value",
				Summary: "Sets the value of a custom field of a datastore",
			},
			Key: "Datastore.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload datastore",
				Summary: "Reload information about the datastore",
			},
			Key: "Datastore.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename datastore",
				Summary: "Renames a datastore",
			},
			Key: "Datastore.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove datastore",
				Summary: "Removes a datastore if it is not used by any host or virtual machine",
			},
			Key: "Datastore.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Tag",
				Summary: "Add a set of tags to the datastore",
			},
			Key: "Datastore.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the datastore",
			},
			Key: "Datastore.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "Datastore.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh datastore",
				Summary: "Refreshes free space on this datastore",
			},
			Key: "Datastore.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh storage information",
				Summary: "Refresh the storage information of the datastore",
			},
			Key: "Datastore.refreshStorageInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update virtual machine files",
				Summary: "Update virtual machine files on the datastore",
			},
			Key: "Datastore.updateVirtualMachineFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename datastore",
				Summary: "Rename the datastore",
			},
			Key: "Datastore.renameDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete datastore",
				Summary: "Delete datastore",
			},
			Key: "Datastore.destroyDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Replace embedded file paths",
				Summary: "Replace embedded file paths on the datastore",
			},
			Key: "Datastore.replaceEmbeddedFilePaths",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enter SDRS maintenance mode",
				Summary: "Virtual machine evacuation recommendations from the selected datastore are generated for SDRS maintenance mode",
			},
			Key: "Datastore.enterMaintenanceMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Exit SDRS maintenance mode",
				Summary: "Exit SDRS maintenance mode",
			},
			Key: "Datastore.exitMaintenanceMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get native clone capability",
				Summary: "Check if the datastore supports native clone",
			},
			Key: "Datastore.isNativeCloneCapable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cleanup locks",
				Summary: "Cleanup lock files on NFSV3 datastore",
			},
			Key: "Datastore.cleanupLocks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateVVolVirtualMachineFiles",
				Summary: "updateVVolVirtualMachineFiles",
			},
			Key: "Datastore.updateVVolVirtualMachineFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create scheduled task",
				Summary: "Create a scheduled task",
			},
			Key: "scheduler.ScheduledTaskManager.create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve scheduled task",
				Summary: "Available scheduled tasks defined on the entity",
			},
			Key: "scheduler.ScheduledTaskManager.retrieveEntityScheduledTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create scheduled task",
				Summary: "Create a scheduled task",
			},
			Key: "scheduler.ScheduledTaskManager.createObjectScheduledTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve scheduled task",
				Summary: "Available scheduled tasks defined on the object",
			},
			Key: "scheduler.ScheduledTaskManager.retrieveObjectScheduledTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add role",
				Summary: "Add a new role",
			},
			Key: "AuthorizationManager.addRole",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove role",
				Summary: "Remove a role",
			},
			Key: "AuthorizationManager.removeRole",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update role",
				Summary: "Update a role's name and/or privileges",
			},
			Key: "AuthorizationManager.updateRole",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reassign permissions",
				Summary: "Reassign all permissions of a role to another role",
			},
			Key: "AuthorizationManager.mergePermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get role permissions",
				Summary: "Gets all the permissions that use a particular role",
			},
			Key: "AuthorizationManager.retrieveRolePermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get entity permissions",
				Summary: "Get permissions defined on an entity",
			},
			Key: "AuthorizationManager.retrieveEntityPermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get permissions",
				Summary: "Get the permissions defined for all users",
			},
			Key: "AuthorizationManager.retrieveAllPermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrievePermissions",
				Summary: "retrievePermissions",
			},
			Key: "AuthorizationManager.retrievePermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set entity permission rules",
				Summary: "Define or update permission rules on an entity",
			},
			Key: "AuthorizationManager.setEntityPermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset entity permission rules",
				Summary: "Reset permission rules on an entity to the provided set",
			},
			Key: "AuthorizationManager.resetEntityPermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove entity permission",
				Summary: "Remove a permission rule from the entity",
			},
			Key: "AuthorizationManager.removeEntityPermission",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query disabled methods",
				Summary: "Get the list of source objects that have been disabled on the target entity",
			},
			Key: "AuthorizationManager.queryDisabledMethods",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable authorization methods",
				Summary: "Gets the set of method names to be disabled",
			},
			Key: "AuthorizationManager.disableMethods",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable authorization methods",
				Summary: "Gets the set of method names to be enabled",
			},
			Key: "AuthorizationManager.enableMethods",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check privileges on a managed entity",
				Summary: "Checks whether a session holds a set of privileges on a managed entity",
			},
			Key: "AuthorizationManager.hasPrivilegeOnEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check privileges on a set of managed entities",
				Summary: "Checks whether a session holds a set of privileges on a set of managed entities",
			},
			Key: "AuthorizationManager.hasPrivilegeOnEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "hasUserPrivilegeOnEntities",
				Summary: "hasUserPrivilegeOnEntities",
			},
			Key: "AuthorizationManager.hasUserPrivilegeOnEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "fetchUserPrivilegeOnEntities",
				Summary: "fetchUserPrivilegeOnEntities",
			},
			Key: "AuthorizationManager.fetchUserPrivilegeOnEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check method invocation privileges",
				Summary: "Checks whether a session holds a set of privileges required to invoke a specified method",
			},
			Key: "AuthorizationManager.checkMethodInvocation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query required permissions",
				Summary: "Get the permission requirements for the specified request",
			},
			Key: "AuthorizationManager.queryPermissions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "<internal>",
				Summary: "<internal>",
			},
			Key: "ServiceDirectory.queryServiceEndpointList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register service endpoint",
				Summary: "Registers a service endpoint",
			},
			Key: "ServiceDirectory.registerService",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister service endpoint",
				Summary: "Unregisters a service endpoint",
			},
			Key: "ServiceDirectory.unregisterService",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query options view",
				Summary: "Returns nodes in the option hierarchy",
			},
			Key: "option.OptionManager.queryView",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update option values",
				Summary: "Updates one or more properties",
			},
			Key: "option.OptionManager.updateValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "validate",
				Summary: "validate",
			},
			Key: "vdcs.NicManager.validate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "bind",
				Summary: "bind",
			},
			Key: "vdcs.NicManager.bind",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unbind",
				Summary: "unbind",
			},
			Key: "vdcs.NicManager.unbind",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Specification exists",
				Summary: "Check the existence of a specification",
			},
			Key: "CustomizationSpecManager.exists",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get specification",
				Summary: "Gets a specification",
			},
			Key: "CustomizationSpecManager.get",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create new specification",
				Summary: "Create a new specification",
			},
			Key: "CustomizationSpecManager.create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Overwrite specification",
				Summary: "Overwrite an existing specification",
			},
			Key: "CustomizationSpecManager.overwrite",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete specification",
				Summary: "Delete a specification",
			},
			Key: "CustomizationSpecManager.delete",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Duplicate specification",
				Summary: "Duplicate a specification",
			},
			Key: "CustomizationSpecManager.duplicate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename specification",
				Summary: "Rename a specification",
			},
			Key: "CustomizationSpecManager.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Convert specification item",
				Summary: "Convert a specification item to XML text",
			},
			Key: "CustomizationSpecManager.specItemToXml",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Convert XML item",
				Summary: "Convert an XML string to a specification item",
			},
			Key: "CustomizationSpecManager.xmlToSpecItem",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate required resources",
				Summary: "Validate that required resources are available on the server to customize a particular guest operating system",
			},
			Key: "CustomizationSpecManager.checkResources",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set cluster resource custom value",
				Summary: "Sets the value of a custom field for a cluster of objects as a unified compute-resource",
			},
			Key: "ClusterComputeResource.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload cluster",
				Summary: "Reloads the cluster",
			},
			Key: "ClusterComputeResource.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename cluster",
				Summary: "Rename the compute-resource",
			},
			Key: "ClusterComputeResource.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove cluster",
				Summary: "Deletes the cluster compute-resource and removes it from its parent folder (if any)",
			},
			Key: "ClusterComputeResource.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the cluster",
			},
			Key: "ClusterComputeResource.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Removes a set of tags from the cluster",
			},
			Key: "ClusterComputeResource.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "ClusterComputeResource.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure cluster",
				Summary: "Reconfigures a cluster",
			},
			Key: "ClusterComputeResource.reconfigureEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure cluster",
				Summary: "Reconfigures a cluster",
			},
			Key: "ClusterComputeResource.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply recommendation",
				Summary: "Applies a recommendation",
			},
			Key: "ClusterComputeResource.applyRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel recommendation",
				Summary: "Cancels a recommendation",
			},
			Key: "ClusterComputeResource.cancelRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Recommended power On hosts",
				Summary: "Get recommendations for a location to power on a specific virtual machine",
			},
			Key: "ClusterComputeResource.recommendHostsForVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add host",
				Summary: "Adds a new host to the cluster",
			},
			Key: "ClusterComputeResource.addHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add host and enable lockdown",
				Summary: "Adds a new host to the cluster and enables lockdown mode on the host",
			},
			Key: "ClusterComputeResource.addHostWithAdminDisabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move host into cluster",
				Summary: "Moves a set of existing hosts into the cluster",
			},
			Key: "ClusterComputeResource.moveInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move host into cluster",
				Summary: "Moves a host into the cluster",
			},
			Key: "ClusterComputeResource.moveHostInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh recommendations",
				Summary: "Refreshes the list of recommendations",
			},
			Key: "ClusterComputeResource.refreshRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve EVC",
				Summary: "Retrieve Enhanced vMotion Compatibility information for this cluster",
			},
			Key: "ClusterComputeResource.evcManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve transitional EVC manager",
				Summary: "Retrieve the transitional EVC manager for this cluster",
			},
			Key: "ClusterComputeResource.transitionalEVCManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve DAS advanced runtime information",
				Summary: "Retrieve DAS advanced runtime information for this cluster",
			},
			Key: "ClusterComputeResource.retrieveDasAdvancedRuntimeInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve vShpere HA data for cluster",
				Summary: "Retrieves HA data for a cluster",
			},
			Key: "ClusterComputeResource.retrieveDasData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check VM admission in vSphere HA cluster",
				Summary: "Checks if HA admission control allows a set of virtual machines to be powered on in the cluster",
			},
			Key: "ClusterComputeResource.checkDasAdmission",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check cluster for vSphere HA configuration",
				Summary: "Check how the specified HA config will affect the cluster state if high availability is enabled",
			},
			Key: "ClusterComputeResource.checkReconfigureDas",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "checkReconfigureDasVmcp",
				Summary: "checkReconfigureDasVmcp",
			},
			Key: "ClusterComputeResource.checkReconfigureDasVmcp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "DRS recommends hosts to evacuate",
				Summary: "DRS recommends hosts to evacuate",
			},
			Key: "ClusterComputeResource.enterMaintenanceMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find Fault Tolerance compatible hosts for placing secondary VM",
				Summary: "Find the set of Fault Tolerance compatible hosts for placing secondary of a given primary virtual machine",
			},
			Key: "ClusterComputeResource.queryFaultToleranceCompatibleHosts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find Fault Tolerance compatible datastores for a VM",
				Summary: "Find the set of Fault Tolerance compatible datastores for a given virtual machine",
			},
			Key: "ClusterComputeResource.queryFaultToleranceCompatibleDatastores",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Verify FaultToleranceConfigSpec",
				Summary: "Verify whether a given FaultToleranceConfigSpec satisfies the requirements for Fault Tolerance",
			},
			Key: "ClusterComputeResource.verifyFaultToleranceConfigSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check Fault Tolerance compatibility for VM",
				Summary: "Check whether a VM is compatible for turning on Fault Tolerance",
			},
			Key: "ClusterComputeResource.queryCompatibilityForFaultTolerance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Call DRS for cross vMotion placement recommendations",
				Summary: "Calls vSphere DRS for placement recommendations when migrating a VM across vCenter Server instances and virtual switches",
			},
			Key: "ClusterComputeResource.placeVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find rules for VM",
				Summary: "Locates all affinity and anti-affinity rules the specified VM participates in",
			},
			Key: "ClusterComputeResource.findRulesForVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "stampAllRulesWithUuid",
				Summary: "stampAllRulesWithUuid",
			},
			Key: "ClusterComputeResource.stampAllRulesWithUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getResourceUsage",
				Summary: "getResourceUsage",
			},
			Key: "ClusterComputeResource.getResourceUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryDrmDumpHistory",
				Summary: "queryDrmDumpHistory",
			},
			Key: "ClusterComputeResource.queryDrmDumpHistory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "generateDrmBundle",
				Summary: "generateDrmBundle",
			},
			Key: "ClusterComputeResource.generateDrmBundle",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "waitForChanges",
				Summary: "waitForChanges",
			},
			Key: "cdc.ChangeLogCollector.waitForChanges",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "initializeSequence",
				Summary: "initializeSequence",
			},
			Key: "cdc.ChangeLogCollector.initializeSequence",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "exchangeSequence",
				Summary: "exchangeSequence",
			},
			Key: "cdc.ChangeLogCollector.exchangeSequence",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate a certificate signing request",
				Summary: "Generates a certificate signing request (CSR) for the host",
			},
			Key: "host.CertificateManager.generateCertificateSigningRequest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate a certificate signing request using the specified Distinguished Name",
				Summary: "Generates a certificate signing request (CSR) for the host using the specified Distinguished Name",
			},
			Key: "host.CertificateManager.generateCertificateSigningRequestByDn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install a server certificate",
				Summary: "Installs a server certificate for the host",
			},
			Key: "host.CertificateManager.installServerCertificate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Replace CA certificates and certificate revocation lists",
				Summary: "Replaces the CA certificates and certificate revocation lists (CRLs) on the host",
			},
			Key: "host.CertificateManager.replaceCACertificatesAndCRLs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Notify services affected by SSL credentials change",
				Summary: "Notifies the host services affected by SSL credentials change",
			},
			Key: "host.CertificateManager.notifyAffectedServices",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List CA certificates",
				Summary: "Lists the CA certificates on the host",
			},
			Key: "host.CertificateManager.listCACertificates",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List CA certificate revocation lists",
				Summary: "Lists the CA certificate revocation lists (CRLs) on the host",
			},
			Key: "host.CertificateManager.listCACertificateRevocationLists",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set CPU scheduler system custom value",
				Summary: "Sets the value of a custom field of a host CPU scheduler",
			},
			Key: "host.CpuSchedulerSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable hyperthreading",
				Summary: "Enable hyperthreads as schedulable resources",
			},
			Key: "host.CpuSchedulerSystem.enableHyperThreading",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable hyperthreading",
				Summary: "Disable hyperthreads as schedulable resources",
			},
			Key: "host.CpuSchedulerSystem.disableHyperThreading",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Search datastore",
				Summary: "Returns the information for the files that match the given search criteria",
			},
			Key: "host.DatastoreBrowser.search",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Search datastore subfolders",
				Summary: "Returns the information for the files that match the given search criteria",
			},
			Key: "host.DatastoreBrowser.searchSubFolders",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete datastore file",
				Summary: "Deletes the specified files from the datastore",
			},
			Key: "host.DatastoreBrowser.deleteFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update configuration",
				Summary: "Update the date and time on the host",
			},
			Key: "host.DateTimeSystem.updateConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query available time zones",
				Summary: "Retrieves the list of available time zones on the host",
			},
			Key: "host.DateTimeSystem.queryAvailableTimeZones",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query date and time",
				Summary: "Get the current date and time on the host",
			},
			Key: "host.DateTimeSystem.queryDateTime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update date or time",
				Summary: "Update the date/time on the host",
			},
			Key: "host.DateTimeSystem.updateDateTime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh",
				Summary: "Refresh the date and time settings",
			},
			Key: "host.DateTimeSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire disk lease",
				Summary: "Acquire a lease for the files associated with the virtual disk referenced by the given datastore path",
			},
			Key: "host.DiskManager.acquireLease",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire lease extension",
				Summary: "Acquires a lease for the files associated with the virtual disk of a virtual machine",
			},
			Key: "host.DiskManager.acquireLeaseExt",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Renew all leases",
				Summary: "Resets the watchdog timer and confirms that all the locks for all the disks managed by this watchdog are still valid",
			},
			Key: "host.DiskManager.renewAllLeases",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set task custom value",
				Summary: "Sets the value of a custom field of a task",
			},
			Key: "Task.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel",
				Summary: "Cancels a running/queued task",
			},
			Key: "Task.cancel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update progress",
				Summary: "Update task progress",
			},
			Key: "Task.UpdateProgress",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set task state",
				Summary: "Sets task state",
			},
			Key: "Task.setState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update task description",
				Summary: "Updates task description with the current phase of the task",
			},
			Key: "Task.UpdateDescription",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Renew disk lease",
				Summary: "Renew a lease to prevent it from timing out",
			},
			Key: "host.DiskManager.Lease.renew",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Release disk lease",
				Summary: "End the lease if it is still active",
			},
			Key: "host.DiskManager.Lease.release",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Allocate blocks",
				Summary: "Prepare for writing to blocks",
			},
			Key: "host.DiskManager.Lease.allocateBlocks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear lazy zero",
				Summary: "Honor the contents of a block range",
			},
			Key: "host.DiskManager.Lease.clearLazyZero",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Map disk region",
				Summary: "Mapping a specified region of a virtual disk",
			},
			Key: "host.DiskManager.Lease.MapDiskRegion",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update ESX agent configuration",
				Summary: "Updates the ESX agent configuration of a host",
			},
			Key: "host.EsxAgentHostManager.updateConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset to factory default",
				Summary: "Reset the configuration to factory default",
			},
			Key: "host.FirmwareSystem.resetToFactoryDefaults",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Backup configuration",
				Summary: "Backup the configuration of the host",
			},
			Key: "host.FirmwareSystem.backupConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query configuration upload URL",
				Summary: "Host configuration must be uploaded for a restore operation",
			},
			Key: "host.FirmwareSystem.queryConfigUploadURL",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Restore configuration",
				Summary: "Restore configuration of the host",
			},
			Key: "host.FirmwareSystem.restoreConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Flush firmware configuration",
				Summary: "Writes the configuration of the firmware system to persistent storage",
			},
			Key: "host.FirmwareSystem.syncConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryQuantumMinutes",
				Summary: "queryQuantumMinutes",
			},
			Key: "host.FirmwareSystem.queryQuantumMinutes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "querySyncsPerQuantum",
				Summary: "querySyncsPerQuantum",
			},
			Key: "host.FirmwareSystem.querySyncsPerQuantum",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh hardware information",
				Summary: "Refresh hardware information",
			},
			Key: "host.HealthStatusSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset system health sensors",
				Summary: "Resets the state of the sensors of the IPMI subsystem",
			},
			Key: "host.HealthStatusSystem.resetSystemHealthInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear hardware IPMI System Event Log",
				Summary: "Clear hardware IPMI System Event Log",
			},
			Key: "host.HealthStatusSystem.clearSystemEventLog",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh hardware IPMI System Event Log",
				Summary: "Refresh hardware IPMI System Event Log",
			},
			Key: "host.HealthStatusSystem.FetchSystemEventLog",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve access entries",
				Summary: "Retrieves the access mode for each user or group with access permissions on the host",
			},
			Key: "host.HostAccessManager.retrieveAccessEntries",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change access mode",
				Summary: "Changes the access mode for a user or group on the host",
			},
			Key: "host.HostAccessManager.changeAccessMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve special DCUI access users",
				Summary: "Retrieves the list of users with special access to DCUI",
			},
			Key: "host.HostAccessManager.queryDcuiAccess",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update special DCUI access users",
				Summary: "Updates the list of users with special access to DCUI",
			},
			Key: "host.HostAccessManager.updateDcuiAccess",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve system users",
				Summary: "Retrieve the list of special system users on the host",
			},
			Key: "host.HostAccessManager.querySystemUsers",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update system users",
				Summary: "Updates the list of special system users on the host",
			},
			Key: "host.HostAccessManager.updateSystemUsers",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query lockdown exceptions",
				Summary: "Queries the current list of user exceptions for lockdown mode",
			},
			Key: "host.HostAccessManager.queryLockdownExceptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update lockdown exceptions",
				Summary: "Updates the current list of user exceptions for lockdown mode",
			},
			Key: "host.HostAccessManager.updateLockdownExceptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change lockdown mode",
				Summary: "Changes lockdown mode on the host",
			},
			Key: "host.HostAccessManager.changeLockdownMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get acceptance level for host image configuration",
				Summary: "Get acceptance level settings for host image configuration",
			},
			Key: "host.ImageConfigManager.queryHostAcceptanceLevel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query host image profile",
				Summary: "Queries the current host image profile information",
			},
			Key: "host.ImageConfigManager.queryHostImageProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update acceptance level",
				Summary: "Updates the acceptance level of a host",
			},
			Key: "host.ImageConfigManager.updateAcceptanceLevel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "fetchSoftwarePackages",
				Summary: "fetchSoftwarePackages",
			},
			Key: "host.ImageConfigManager.fetchSoftwarePackages",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "installDate",
				Summary: "installDate",
			},
			Key: "host.ImageConfigManager.installDate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query host kernel modules",
				Summary: "Retrieves information about the kernel modules on the host",
			},
			Key: "host.KernelModuleSystem.queryModules",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update kernel module option",
				Summary: "Specifies the options to be passed to the kernel module when loaded",
			},
			Key: "host.KernelModuleSystem.updateModuleOptionString",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query kernel module options",
				Summary: "Retrieves the options configured to be passed to a kernel module when loaded",
			},
			Key: "host.KernelModuleSystem.queryConfiguredModuleOptionString",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set memory manager custom value",
				Summary: "Sets the value of a custom field of a host memory manager system",
			},
			Key: "host.MemoryManagerSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set console memory reservation",
				Summary: "Set the configured service console memory reservation",
			},
			Key: "host.MemoryManagerSystem.reconfigureServiceConsoleReservation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure virtual machine reservation",
				Summary: "Updates the virtual machine reservation information",
			},
			Key: "host.MemoryManagerSystem.reconfigureVirtualMachineReservation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query proxy information",
				Summary: "Query the common message bus proxy service information",
			},
			Key: "host.MessageBusProxy.retrieveInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure proxy",
				Summary: "Configure the common message bus proxy service",
			},
			Key: "host.MessageBusProxy.configure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove proxy configuration",
				Summary: "Remove the common message proxy service configuration and disable the service",
			},
			Key: "host.MessageBusProxy.unconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start proxy",
				Summary: "Start the common message bus proxy service",
			},
			Key: "host.MessageBusProxy.start",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop proxy",
				Summary: "Stop the common message bus proxy service",
			},
			Key: "host.MessageBusProxy.stop",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload proxy",
				Summary: "Reload the common message bus proxy service and enable any configuration changes",
			},
			Key: "host.MessageBusProxy.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set service custom value",
				Summary: "Sets the value of a custom field of a host service system.",
			},
			Key: "host.ServiceSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update service activation policy",
				Summary: "Updates the activation policy of the service",
			},
			Key: "host.ServiceSystem.updatePolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start service",
				Summary: "Starts the service",
			},
			Key: "host.ServiceSystem.start",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop service",
				Summary: "Stops the service",
			},
			Key: "host.ServiceSystem.stop",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Restart service",
				Summary: "Restarts the service",
			},
			Key: "host.ServiceSystem.restart",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Uninstall service",
				Summary: "Uninstalls the service",
			},
			Key: "host.ServiceSystem.uninstall",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh service information",
				Summary: "Refresh the service information and settings to detect any changes made directly on the host",
			},
			Key: "host.ServiceSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure SNMP agent",
				Summary: "Reconfigure the SNMP agent",
			},
			Key: "host.SnmpSystem.reconfigureSnmpAgent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Send test notification",
				Summary: "Send test notification",
			},
			Key: "host.SnmpSystem.sendTestNotification",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure virtual flash resource",
				Summary: "Configures virtual flash resource on a list of SSD devices",
			},
			Key: "host.VFlashManager.configureVFlashResourceEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure virtual flash resource",
				Summary: "Configures virtual flash resource on a host",
			},
			Key: "host.VFlashManager.configureVFlashResource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove virtual flash resource",
				Summary: "Removes virtual flash resource from a host",
			},
			Key: "host.VFlashManager.removeVFlashResource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure virtual flash host swap cache",
				Summary: "Configures virtual flash host swap cache",
			},
			Key: "host.VFlashManager.configureHostVFlashCache",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve virtual flash module configuration options from a host",
				Summary: "Retrieves virtual flash module configuration options from a host",
			},
			Key: "host.VFlashManager.getVFlashModuleDefaultConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query disks for use in vSAN cluster",
				Summary: "Queries disk eligibility for use in the vSAN cluster",
			},
			Key: "host.VsanSystem.queryDisksForVsan",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add disks to vSAN",
				Summary: "Adds the selected disks to the vSAN cluster",
			},
			Key: "host.VsanSystem.addDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initialize disks in the vSAN cluster",
				Summary: "Initializes the selected disks to be used in the vSAN cluster",
			},
			Key: "host.VsanSystem.initializeDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove disk from vSAN",
				Summary: "Removes the disks that are used in the vSAN cluster",
			},
			Key: "host.VsanSystem.removeDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove disk group from vSAN",
				Summary: "Removes the selected disk group from the vSAN cluster",
			},
			Key: "host.VsanSystem.removeDiskMapping",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unmountDiskMapping",
				Summary: "unmountDiskMapping",
			},
			Key: "host.VsanSystem.unmountDiskMapping",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSAN configuration",
				Summary: "Updates the vSAN configuration for this host",
			},
			Key: "host.VsanSystem.update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve vSAN runtime information",
				Summary: "Retrieves the current vSAN runtime information for this host",
			},
			Key: "host.VsanSystem.queryHostStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Evacuate this host from vSAN cluster",
				Summary: "Evacuates the specified host from the vSAN cluster",
			},
			Key: "host.VsanSystem.evacuateNode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Recommission this host back to vSAN cluster",
				Summary: "Recommissions the host back to vSAN cluster",
			},
			Key: "host.VsanSystem.recommissionNode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve a ticket to register the vSAN VASA Provider",
				Summary: "Retrieves a ticket to register the VASA Provider for vSAN in the Storage Monitoring Service",
			},
			Key: "host.VsanSystem.fetchVsanSharedSecret",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Join Windows Domain",
				Summary: "Enables ActiveDirectory authentication on the host",
			},
			Key: "host.ActiveDirectoryAuthentication.joinDomain",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Join Windows Domain through vSphere Authentication Proxy service",
				Summary: "Enables Active Directory authentication on the host using a vSphere Authentication Proxy server",
			},
			Key: "host.ActiveDirectoryAuthentication.joinDomainWithCAM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import the certificate of vSphere Authentication Proxy server",
				Summary: "Import the certificate of vSphere Authentication Proxy server to ESXi's authentication store",
			},
			Key: "host.ActiveDirectoryAuthentication.importCertificateForCAM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Leave Windows Domain",
				Summary: "Disables ActiveDirectory authentication on the host",
			},
			Key: "host.ActiveDirectoryAuthentication.leaveCurrentDomain",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable Smart Card Authentication",
				Summary: "Enables smart card authentication of ESXi Direct Console UI users",
			},
			Key: "host.ActiveDirectoryAuthentication.enableSmartCardAuthentication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install a Smart Card Trust Anchor",
				Summary: "Installs a smart card trust anchor on the host",
			},
			Key: "host.ActiveDirectoryAuthentication.installSmartCardTrustAnchor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "replaceSmartCardTrustAnchors",
				Summary: "replaceSmartCardTrustAnchors",
			},
			Key: "host.ActiveDirectoryAuthentication.replaceSmartCardTrustAnchors",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove a Smart Card Trust Anchor",
				Summary: "Removes an installed smart card trust anchor from the host",
			},
			Key: "host.ActiveDirectoryAuthentication.removeSmartCardTrustAnchor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove Smart Card Trust Anchor",
				Summary: "Removes the installed smart card trust anchor from the host",
			},
			Key: "host.ActiveDirectoryAuthentication.removeSmartCardTrustAnchorByFingerprint",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List Smart Card Trust Anchors",
				Summary: "Lists the smart card trust anchors installed on the host",
			},
			Key: "host.ActiveDirectoryAuthentication.listSmartCardTrustAnchors",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable Smart Card Authentication",
				Summary: "Disables smart card authentication of ESXi Direct Console UI users",
			},
			Key: "host.ActiveDirectoryAuthentication.disableSmartCardAuthentication",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update local swap datastore",
				Summary: "Changes the datastore for virtual machine swap files",
			},
			Key: "host.DatastoreSystem.updateLocalSwapDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve disks for VMFS datastore",
				Summary: "Retrieves the list of disks that can be used to contain VMFS datastore extents",
			},
			Key: "host.DatastoreSystem.queryAvailableDisksForVmfs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query datastore create options",
				Summary: "Queries options for creating a new VMFS datastore for a disk",
			},
			Key: "host.DatastoreSystem.queryVmfsDatastoreCreateOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create VMFS datastore",
				Summary: "Creates a new VMFS datastore",
			},
			Key: "host.DatastoreSystem.createVmfsDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query datastore extend options",
				Summary: "Queries options for extending an existing VMFS datastore for a disk",
			},
			Key: "host.DatastoreSystem.queryVmfsDatastoreExtendOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query VMFS datastore expand options",
				Summary: "Query the options available for expanding the extents of a VMFS datastore",
			},
			Key: "host.DatastoreSystem.queryVmfsDatastoreExpandOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend datastore",
				Summary: "Extends an existing VMFS datastore",
			},
			Key: "host.DatastoreSystem.extendVmfsDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Expand VMFS datastore",
				Summary: "Expand the capacity of a VMFS datastore extent",
			},
			Key: "host.DatastoreSystem.expandVmfsDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "processVmfsDatastoreUpdate",
				Summary: "processVmfsDatastoreUpdate",
			},
			Key: "host.DatastoreSystem.processVmfsDatastoreUpdate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create NAS datastore",
				Summary: "Creates a new Network Attached Storage (NAS) datastore",
			},
			Key: "host.DatastoreSystem.createNasDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create local datastore",
				Summary: "Creates a new local datastore",
			},
			Key: "host.DatastoreSystem.createLocalDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Virtual Volume datastore",
				Summary: "Updates the Virtual Volume datastore configuration according to the provided settings",
			},
			Key: "host.DatastoreSystem.UpdateVvolDatastoreInternal",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Virtual Volume datastore",
				Summary: "Creates a datastore backed by a Virtual Volume storage container",
			},
			Key: "host.DatastoreSystem.createVvolDatastoreInternal",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Virtual Volume datastore",
				Summary: "Creates a Virtuial Volume datastore",
			},
			Key: "host.DatastoreSystem.createVvolDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove datastore",
				Summary: "Removes a datastore from a host",
			},
			Key: "host.DatastoreSystem.removeDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove datastores",
				Summary: "Removes one or more datastores from a host",
			},
			Key: "host.DatastoreSystem.removeDatastoreEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure datastore principal",
				Summary: "Configures datastore principal user for the host",
			},
			Key: "host.DatastoreSystem.configureDatastorePrincipal",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query unbound VMFS volumes",
				Summary: "Gets the list of unbound VMFS volumes",
			},
			Key: "host.DatastoreSystem.queryUnresolvedVmfsVolumes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resignature unresolved VMFS volume",
				Summary: "Resignature unresolved VMFS volume with new VMFS identifier",
			},
			Key: "host.DatastoreSystem.resignatureUnresolvedVmfsVolume",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "NotifyDatastore",
				Summary: "NotifyDatastore",
			},
			Key: "host.DatastoreSystem.NotifyDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check accessibility",
				Summary: "Check if the file objects for the specified virtual machine IDs are accessible",
			},
			Key: "host.DatastoreSystem.checkVmFileAccessibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set firewall custom value",
				Summary: "Sets the value of a custom field of a host firewall system",
			},
			Key: "host.FirewallSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update default firewall policy",
				Summary: "Updates the default firewall policy",
			},
			Key: "host.FirewallSystem.updateDefaultPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Open firewall ports",
				Summary: "Open the firewall ports belonging to the specified ruleset",
			},
			Key: "host.FirewallSystem.enableRuleset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Block firewall ports",
				Summary: "Block the firewall ports belonging to the specified ruleset",
			},
			Key: "host.FirewallSystem.disableRuleset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update allowed IP list of the firewall ruleset",
				Summary: "Update the allowed IP list of the specified ruleset",
			},
			Key: "host.FirewallSystem.updateRuleset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh firewall information",
				Summary: "Refresh the firewall information and settings to detect any changes made directly on the host",
			},
			Key: "host.FirewallSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set network custom value",
				Summary: "Sets the value of a custom field of a host network system",
			},
			Key: "host.NetworkSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network configuration",
				Summary: "Network configuration information",
			},
			Key: "host.NetworkSystem.updateNetworkConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update DNS configuration",
				Summary: "Update the DNS configuration for the host",
			},
			Key: "host.NetworkSystem.updateDnsConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update IP route configuration",
				Summary: "Update IP route configuration",
			},
			Key: "host.NetworkSystem.updateIpRouteConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update console IP route configuration",
				Summary: "Update console IP route configuration",
			},
			Key: "host.NetworkSystem.updateConsoleIpRouteConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update IP route table configuration",
				Summary: "Applies the IP route table configuration for the host",
			},
			Key: "host.NetworkSystem.updateIpRouteTableConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add virtual switch",
				Summary: "Add a new virtual switch to the system",
			},
			Key: "host.NetworkSystem.addVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove virtual switch",
				Summary: "Remove an existing virtual switch from the system",
			},
			Key: "host.NetworkSystem.removeVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update virtual switch",
				Summary: "Updates the properties of the virtual switch",
			},
			Key: "host.NetworkSystem.updateVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add port group",
				Summary: "Add a port group to the virtual switch",
			},
			Key: "host.NetworkSystem.addPortGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove port group",
				Summary: "Remove a port group from the virtual switch",
			},
			Key: "host.NetworkSystem.removePortGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure port group",
				Summary: "Reconfigure a port group on the virtual switch",
			},
			Key: "host.NetworkSystem.updatePortGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update physical NIC link speed",
				Summary: "Configure link speed and duplexity",
			},
			Key: "host.NetworkSystem.updatePhysicalNicLinkSpeed",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query network hint",
				Summary: "Request network hint information for a physical NIC",
			},
			Key: "host.NetworkSystem.queryNetworkHint",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add virtual NIC",
				Summary: "Add a virtual host or service console NIC",
			},
			Key: "host.NetworkSystem.addVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove virtual NIC",
				Summary: "Remove a virtual host or service console NIC",
			},
			Key: "host.NetworkSystem.removeVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update virtual NIC",
				Summary: "Configure virtual host or VMkernel NIC",
			},
			Key: "host.NetworkSystem.updateVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add service console virtual NIC",
				Summary: "Add a virtual service console NIC",
			},
			Key: "host.NetworkSystem.addServiceConsoleVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove service console virtual NIC",
				Summary: "Remove a virtual service console NIC",
			},
			Key: "host.NetworkSystem.removeServiceConsoleVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update service console virtual NIC",
				Summary: "Update IP configuration for a service console virtual NIC",
			},
			Key: "host.NetworkSystem.updateServiceConsoleVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Restart virtual network adapter interface",
				Summary: "Restart the service console virtual network adapter interface",
			},
			Key: "host.NetworkSystem.restartServiceConsoleVirtualNic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh network information",
				Summary: "Refresh the network information and settings to detect any changes that have occurred",
			},
			Key: "host.NetworkSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Invoke API call on host with transactionId",
				Summary: "Invoke API call on host with transactionId",
			},
			Key: "host.NetworkSystem.invokeHostTransactionCall",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Commit transaction to confirm that host is connected to vCenter Server",
				Summary: "Commit transaction to confirm that host is connected to vCenter Server",
			},
			Key: "host.NetworkSystem.commitTransaction",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "performHostOpaqueNetworkDataOperation",
				Summary: "performHostOpaqueNetworkDataOperation",
			},
			Key: "host.NetworkSystem.performHostOpaqueNetworkDataOperation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve available diagnostic partitions",
				Summary: "Retrieves a list of available diagnostic partitions",
			},
			Key: "host.DiagnosticSystem.queryAvailablePartition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change active diagnostic partition",
				Summary: "Changes the active diagnostic partition to a different partition",
			},
			Key: "host.DiagnosticSystem.selectActivePartition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve diagnostic partitionable disks",
				Summary: "Retrieves a list of disks that can be used to contain a diagnostic partition",
			},
			Key: "host.DiagnosticSystem.queryPartitionCreateOptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve diagnostic partition creation description",
				Summary: "Retrieves the diagnostic partition creation description for a disk",
			},
			Key: "host.DiagnosticSystem.queryPartitionCreateDesc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create diagnostic partition",
				Summary: "Creates a diagnostic partition according to the provided creation specification",
			},
			Key: "host.DiagnosticSystem.createDiagnosticPartition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set vApp custom value",
				Summary: "Sets the value of a custom field on a vApp",
			},
			Key: "VirtualApp.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload vApp",
				Summary: "Reload the vApp",
			},
			Key: "VirtualApp.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename vApp",
				Summary: "Rename the vApp",
			},
			Key: "VirtualApp.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete vApp",
				Summary: "Delete the vApp, including all child vApps and virtual machines",
			},
			Key: "VirtualApp.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the vApp",
			},
			Key: "VirtualApp.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the vApp",
			},
			Key: "VirtualApp.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "VirtualApp.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vApp resource configuration",
				Summary: "Updates the resource configuration for the vApp",
			},
			Key: "VirtualApp.updateConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move into vApp",
				Summary: "Moves a set of entities into this vApp",
			},
			Key: "VirtualApp.moveInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update child resource configuration",
				Summary: "Change resource configuration of a set of children of the vApp",
			},
			Key: "VirtualApp.updateChildResourceConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create resource pool",
				Summary: "Creates a new resource pool",
			},
			Key: "VirtualApp.createResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete vApp children",
				Summary: "Deletes all child resource pools recursively",
			},
			Key: "VirtualApp.destroyChildren",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create vApp",
				Summary: "Creates a child vApp of this vApp",
			},
			Key: "VirtualApp.createVApp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine",
				Summary: "Creates a virtual machine in this vApp",
			},
			Key: "VirtualApp.createVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register virtual machine",
				Summary: "Adds an existing virtual machine to this vApp",
			},
			Key: "VirtualApp.registerVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy OVF template",
				Summary: "Deploys a virtual machine or vApp",
			},
			Key: "VirtualApp.importVApp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query Virtual App resource configuration options",
				Summary: "Returns configuration options for a set of resources for a Virtual App",
			},
			Key: "VirtualApp.queryResourceConfigOption",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh Virtual App runtime information",
				Summary: "Refreshes the resource usage runtime information for a Virtual App",
			},
			Key: "VirtualApp.refreshRuntime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vApp Configuration",
				Summary: "Updates the vApp configuration",
			},
			Key: "VirtualApp.updateVAppConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update linked children",
				Summary: "Updates the list of linked children",
			},
			Key: "VirtualApp.updateLinkedChildren",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone vApp",
				Summary: "Clone the vApp, including all child entities",
			},
			Key: "VirtualApp.clone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF template",
				Summary: "Exports the vApp as an OVF template",
			},
			Key: "VirtualApp.exportVApp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start vApp",
				Summary: "Starts the vApp",
			},
			Key: "VirtualApp.powerOn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop vApp",
				Summary: "Stops the vApp",
			},
			Key: "VirtualApp.powerOff",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Suspend vApp",
				Summary: "Suspends the vApp",
			},
			Key: "VirtualApp.suspend",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister vApp",
				Summary: "Unregister all child virtual machines and remove the vApp",
			},
			Key: "VirtualApp.unregister",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set virtual NIC custom value",
				Summary: "Set the value of a custom filed of a host's virtual NIC manager",
			},
			Key: "host.VirtualNicManager.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query network configuration",
				Summary: "Gets the network configuration for the specified NIC type",
			},
			Key: "host.VirtualNicManager.queryNetConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Select virtual NIC",
				Summary: "Select the virtual NIC to be used for the specified NIC type",
			},
			Key: "host.VirtualNicManager.selectVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deselect virtual NIC",
				Summary: "Deselect the virtual NIC used for the specified NIC type",
			},
			Key: "host.VirtualNicManager.deselectVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Download overhead computation script",
				Summary: "Download overhead computation scheme script",
			},
			Key: "OverheadService.downloadScript",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Download host configuration",
				Summary: "Download host configuration consumed by overhead computation script",
			},
			Key: "OverheadService.downloadHostConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Download VM configuration",
				Summary: "Download VM configuration consumed by overhead computation script",
			},
			Key: "OverheadService.downloadVMXConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add an alias to the alias store in the guest",
				Summary: "Add an alias to the alias store in the guest operating system",
			},
			Key: "vm.guest.AliasManager.addAlias",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove an alias from the alias store in the guest",
				Summary: "Remove an alias from the alias store in the guest operating system",
			},
			Key: "vm.guest.AliasManager.removeAlias",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove all aliases associated with a SSO Server certificate from the guest",
				Summary: "Remove all aliases associated with a SSO Server certificate from the guest operating system",
			},
			Key: "vm.guest.AliasManager.removeAliasByCert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List all aliases for a user in the guest",
				Summary: "List all aliases for a user in the guest operating system",
			},
			Key: "vm.guest.AliasManager.listAliases",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List all mapped aliases in the guest",
				Summary: "List all mapped aliases in the guest operating system",
			},
			Key: "vm.guest.AliasManager.listMappedAliases",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a directory in the guest",
				Summary: "Create a directory in the guest operating system",
			},
			Key: "vm.guest.FileManager.makeDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete a file in the guest",
				Summary: "Delete a file in the guest operating system",
			},
			Key: "vm.guest.FileManager.deleteFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete a directory in the guest",
				Summary: "Delete a directory in the guest operating system",
			},
			Key: "vm.guest.FileManager.deleteDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move or rename a directory in the guest",
				Summary: "Move or rename a directory in the guest operating system",
			},
			Key: "vm.guest.FileManager.moveDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move or rename a file in the guest",
				Summary: "Move or rename a file in the guest operating system",
			},
			Key: "vm.guest.FileManager.moveFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a temporary file in the guest",
				Summary: "Create a temporary file in the guest operating system",
			},
			Key: "vm.guest.FileManager.createTemporaryFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a temporary directory in the guest",
				Summary: "Create a temporary directory in the guest operating system",
			},
			Key: "vm.guest.FileManager.createTemporaryDirectory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List files or directories in the guest",
				Summary: "List files or directories in the guest operating system",
			},
			Key: "vm.guest.FileManager.listFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Change the attributes of a file in the guest",
				Summary: "Change the attributes of a file in the guest operating system",
			},
			Key: "vm.guest.FileManager.changeFileAttributes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiates an operation to transfer a file from the guest",
				Summary: "Initiates an operation to transfer a file from the guest operating system",
			},
			Key: "vm.guest.FileManager.initiateFileTransferFromGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiates an operation to transfer a file to the guest",
				Summary: "Initiates an operation to transfer a file to the guest operating system",
			},
			Key: "vm.guest.FileManager.initiateFileTransferToGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start a program in the guest",
				Summary: "Start a program in the guest operating system",
			},
			Key: "vm.guest.ProcessManager.startProgram",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List processes in the guest",
				Summary: "List processes in the guest operating system",
			},
			Key: "vm.guest.ProcessManager.listProcesses",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Terminate a process in the guest",
				Summary: "Terminate a process in the guest operating system",
			},
			Key: "vm.guest.ProcessManager.terminateProcess",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read an environment variable in the guest",
				Summary: "Read an environment variable in the guest operating system",
			},
			Key: "vm.guest.ProcessManager.readEnvironmentVariable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove view",
				Summary: "Remove view",
			},
			Key: "view.View.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve associated License Data objects",
				Summary: "Retrieves all the associated License Data objects",
			},
			Key: "LicenseDataManager.queryEntityLicenseData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve license data associated with managed entity",
				Summary: "Retrieves the license data associated with a specified managed entity",
			},
			Key: "LicenseDataManager.queryAssociatedLicenseData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update entity license container",
				Summary: "Updates the license container associated with a specified managed entity",
			},
			Key: "LicenseDataManager.updateAssociatedLicenseData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply associated license data to managed entity",
				Summary: "Applies associated license data to a managed entity",
			},
			Key: "LicenseDataManager.applyAssociatedLicenseData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update assigned license",
				Summary: "Updates the license assigned to an entity",
			},
			Key: "LicenseAssignmentManager.updateAssignedLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove assigned license",
				Summary: "Removes an assignment of a license to an entity",
			},
			Key: "LicenseAssignmentManager.removeAssignedLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query assigned licenses",
				Summary: "Queries for all the licenses assigned to an entity or all entities",
			},
			Key: "LicenseAssignmentManager.queryAssignedLicenses",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check feature availability",
				Summary: "Checks if the corresponding features are licensed for a list of entities",
			},
			Key: "LicenseAssignmentManager.isFeatureAvailable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update in-use status of a licensed feature",
				Summary: "Updates in-use status of a licensed feature",
			},
			Key: "LicenseAssignmentManager.updateFeatureInUse",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register licenseable entity",
				Summary: "Registers a licenseable entity",
			},
			Key: "LicenseAssignmentManager.registerEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister licenseable entity",
				Summary: "Unregisters an existing licenseable entity and releases any serial numbers assigned to it.",
			},
			Key: "LicenseAssignmentManager.unregisterEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update license entity usage count",
				Summary: "Updates the usage count of a license entity",
			},
			Key: "LicenseAssignmentManager.updateUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upload license file",
				Summary: "Uploads a license file to vCenter Server",
			},
			Key: "LicenseAssignmentManager.uploadLicenseFile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryAssignedLicensesEx",
				Summary: "queryAssignedLicensesEx",
			},
			Key: "LicenseAssignmentManager.queryAssignedLicensesEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateEntity",
				Summary: "updateEntity",
			},
			Key: "LicenseAssignmentManager.updateEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateEntitiesProperties",
				Summary: "updateEntitiesProperties",
			},
			Key: "LicenseAssignmentManager.updateEntitiesProperties",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set backup agent custom value",
				Summary: "Set backup agent custom value",
			},
			Key: "vm.BackupAgent.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start virtual machine backup",
				Summary: "Start a backup operation inside the virtual machine guest",
			},
			Key: "vm.BackupAgent.startBackup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop virtual machine backup",
				Summary: "Stop a backup operation in a virtual machine",
			},
			Key: "vm.BackupAgent.abortBackup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Notify virtual machine snapshot completion",
				Summary: "Notify the virtual machine when a snapshot operation is complete",
			},
			Key: "vm.BackupAgent.notifySnapshotCompletion",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Wait for guest event",
				Summary: "Wait for an event delivered by the virtual machine guest",
			},
			Key: "vm.BackupAgent.waitForEvent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create namespace",
				Summary: "Create a virtual machine namespace",
			},
			Key: "vm.NamespaceManager.createNamespace",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete namespace",
				Summary: "Delete the virtual machine namespace",
			},
			Key: "vm.NamespaceManager.deleteNamespace",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete all namespaces",
				Summary: "Delete all namespaces associated with the virtual machine",
			},
			Key: "vm.NamespaceManager.deleteAllNamespaces",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update namespace",
				Summary: "Reconfigure the virtual machine namespace",
			},
			Key: "vm.NamespaceManager.updateNamespace",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query namespace",
				Summary: "Retrieve detailed information about the virtual machine namespace",
			},
			Key: "vm.NamespaceManager.queryNamespace",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List namespaces",
				Summary: "Retrieve the list of all namespaces for a virtual machine",
			},
			Key: "vm.NamespaceManager.listNamespaces",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Send event to the virtual machine",
				Summary: "Queue event for delivery to the agent in the virtual machine",
			},
			Key: "vm.NamespaceManager.sendEventToGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fetch events from the virtual machine",
				Summary: "Retrieve events sent by the agent in the virtual machine",
			},
			Key: "vm.NamespaceManager.fetchEventsFromGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update data",
				Summary: "Update key/value pairs accessible by the agent in the virtual machine",
			},
			Key: "vm.NamespaceManager.updateData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve data",
				Summary: "Retrieve key/value pairs set by the agent in the virtual machine",
			},
			Key: "vm.NamespaceManager.retrieveData",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Pause",
				Summary: "Pauses a virtual machine",
			},
			Key: "vm.PauseManager.pause",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unpause",
				Summary: "Unpauses a virtual machine",
			},
			Key: "vm.PauseManager.unpause",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Power on and pause",
				Summary: "Powers on a virtual machine and pauses it immediately",
			},
			Key: "vm.PauseManager.powerOnPaused",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure host cache performance enhancement",
				Summary: "Configures host cache by allocating space on a low latency device (usually a solid state drive) for enhanced system performance",
			},
			Key: "host.CacheConfigurationManager.configureCache",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query whether virtual NIC is used by iSCSI multi-pathing",
				Summary: "Query whether virtual NIC is used by iSCSI multi-pathing",
			},
			Key: "host.IscsiManager.queryVnicStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query whether physical NIC is used by iSCSI multi-pathing",
				Summary: "Query whether physical NIC is used by iSCSI multi-pathing",
			},
			Key: "host.IscsiManager.queryPnicStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query all the virtual NICs used by iSCSI multi-pathing",
				Summary: "Query all the virtual NICs used by iSCSI multi-pathing",
			},
			Key: "host.IscsiManager.queryBoundVnics",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query candidate virtual NICs that can be used for iSCSI multi-pathing",
				Summary: "Query candidate virtual NICs that can be used for iSCSI multi-pathing",
			},
			Key: "host.IscsiManager.queryCandidateNics",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add virtual NIC to iSCSI Adapter",
				Summary: "Add virtual NIC to iSCSI Adapter",
			},
			Key: "host.IscsiManager.bindVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove virtual NIC from iSCSI Adapter",
				Summary: "Remove virtual NIC from iSCSI Adapter",
			},
			Key: "host.IscsiManager.unbindVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query migration dependencies for migrating the physical and virtual NICs",
				Summary: "Query migration dependencies for migrating the physical and virtual NICs",
			},
			Key: "host.IscsiManager.queryMigrationDependencies",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set PCI passthrough system custom value",
				Summary: "Set PCI Passthrough system custom value",
			},
			Key: "host.PciPassthruSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh PCI passthrough device information",
				Summary: "Refresh the available PCI passthrough device information",
			},
			Key: "host.PciPassthruSystem.refresh",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update PCI passthrough configuration",
				Summary: "Update PCI passthrough device configuration",
			},
			Key: "host.PciPassthruSystem.updatePassthruConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query network protocol profiles",
				Summary: "Queries the list of network protocol profiles for a datacenter",
			},
			Key: "IpPoolManager.queryIpPools",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create network protocol profile",
				Summary: "Creates a new network protocol profile",
			},
			Key: "IpPoolManager.createIpPool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network protocol profile",
				Summary: "Updates a network protocol profile on a datacenter",
			},
			Key: "IpPoolManager.updateIpPool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Destroy network protocol profile",
				Summary: "Destroys a network protocol profile on the given datacenter",
			},
			Key: "IpPoolManager.destroyIpPool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Allocates an IPv4 address",
				Summary: "Allocates an IPv4 address from an IP pool",
			},
			Key: "IpPoolManager.allocateIpv4Address",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Allocates an IPv6 address",
				Summary: "Allocates an IPv6 address from an IP pool",
			},
			Key: "IpPoolManager.allocateIpv6Address",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Releases an IP allocation",
				Summary: "Releases an IP allocation back to an IP pool",
			},
			Key: "IpPoolManager.releaseIpAllocation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query IP allocations",
				Summary: "Query IP allocations by IP pool and extension key",
			},
			Key: "IpPoolManager.queryIPAllocations",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh the CA certificates on the host",
				Summary: "Refreshes the CA certificates on the host",
			},
			Key: "CertificateManager.refreshCACertificatesAndCRLs",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh the subject certificate on the host",
				Summary: "Refreshes the subject certificate on the host",
			},
			Key: "CertificateManager.refreshCertificates",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Revoke the subject certificate of a host",
				Summary: "Revokes the subject certificate of a host",
			},
			Key: "CertificateManager.revokeCertificates",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query entity provider summary",
				Summary: "Get information about the performance statistics that can be queried for a particular entity",
			},
			Key: "PerformanceManager.queryProviderSummary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query available metrics",
				Summary: "Gets available performance statistic metrics for the specified managed entity between begin and end times",
			},
			Key: "PerformanceManager.queryAvailableMetric",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query counter",
				Summary: "Get counter information for the list of counter IDs passed in",
			},
			Key: "PerformanceManager.queryCounter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query counter by level",
				Summary: "All performance data over 1 year old are deleted from the vCenter database",
			},
			Key: "PerformanceManager.queryCounterByLevel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query performance statistics",
				Summary: "Gets the performance statistics for the entity",
			},
			Key: "PerformanceManager.queryStats",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get composite statistics",
				Summary: "Get performance statistics for the entity and the breakdown across its child entities",
			},
			Key: "PerformanceManager.queryCompositeStats",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Summarizes performance statistics",
				Summary: "Summarizes performance statistics at the specified interval",
			},
			Key: "PerformanceManager.summarizeStats",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create historical interval",
				Summary: "Add a new historical interval configuration",
			},
			Key: "PerformanceManager.createHistoricalInterval",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove historical interval",
				Summary: "Remove a historical interval configuration",
			},
			Key: "PerformanceManager.removeHistoricalInterval",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update historical interval",
				Summary: "Update a historical interval configuration if it exists",
			},
			Key: "PerformanceManager.updateHistoricalInterval",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update counter level mapping",
				Summary: "Update counter to level mapping",
			},
			Key: "PerformanceManager.updateCounterLevelMapping",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset counter level mapping",
				Summary: "Reset counter to level mapping to the default values",
			},
			Key: "PerformanceManager.resetCounterLevelMapping",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query internal performance counters",
				Summary: "Queries all internal counters, supported by this performance manager",
			},
			Key: "PerformanceManager.queryPerfCounterInt",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable performance counters",
				Summary: "Enable a counter or a set of counters in the counters collection of this performance manager",
			},
			Key: "PerformanceManager.enableStat",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable performance counters",
				Summary: "Exclude a counter or a set of counters from the counters collection of this performance manager",
			},
			Key: "PerformanceManager.disableStat",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "registerProvider",
				Summary: "registerProvider",
			},
			Key: "ExternalStatsManager.registerProvider",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unregisterProvider",
				Summary: "unregisterProvider",
			},
			Key: "ExternalStatsManager.unregisterProvider",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "isRegistered",
				Summary: "isRegistered",
			},
			Key: "ExternalStatsManager.isRegistered",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getRegisteredProviders",
				Summary: "getRegisteredProviders",
			},
			Key: "ExternalStatsManager.getRegisteredProviders",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getEnabledClusters",
				Summary: "getEnabledClusters",
			},
			Key: "ExternalStatsManager.getEnabledClusters",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "updateStats",
				Summary: "updateStats",
			},
			Key: "ExternalStatsManager.updateStats",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create task collector",
				Summary: "Creates a task collector to retrieve all tasks that have executed on the server based on a filter",
			},
			Key: "TaskManager.createCollector",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create task",
				Summary: "Create a task",
			},
			Key: "TaskManager.createTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "createTaskWithEntityName",
				Summary: "createTaskWithEntityName",
			},
			Key: "TaskManager.createTaskWithEntityName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set host custom value",
				Summary: "Sets the value of a custom field of an host",
			},
			Key: "HostSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload host system",
				Summary: "Reloads the host system",
			},
			Key: "HostSystem.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename host",
				Summary: "Rename this host",
			},
			Key: "HostSystem.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove host",
				Summary: "Removes the host",
			},
			Key: "HostSystem.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the host",
			},
			Key: "HostSystem.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the host",
			},
			Key: "HostSystem.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "HostSystem.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query TPM attestation information",
				Summary: "Provides details of the secure boot and TPM status",
			},
			Key: "HostSystem.queryTpmAttestationReport",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query connection information",
				Summary: "Connection information about a host",
			},
			Key: "HostSystem.queryConnectionInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve internal host capabilities",
				Summary: "Retrieves vCenter Server-specific internal host capabilities",
			},
			Key: "HostSystem.retrieveInternalCapability",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "<internal>",
				Summary: "<internal>",
			},
			Key: "HostSystem.retrieveInternalConfigManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update system resources",
				Summary: "Update the configuration of the system resource hierarchy",
			},
			Key: "HostSystem.updateSystemResources",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update system swap configuration",
				Summary: "Update the configuration of the system swap",
			},
			Key: "HostSystem.updateSystemSwapConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconnect host",
				Summary: "Reconnects to a host",
			},
			Key: "HostSystem.reconnect",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disconnect host",
				Summary: "Disconnects from a host",
			},
			Key: "HostSystem.disconnect",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enter maintenance mode",
				Summary: "Puts the host in maintenance mode",
			},
			Key: "HostSystem.enterMaintenanceMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Exit maintenance mode",
				Summary: "Disables maintenance mode",
			},
			Key: "HostSystem.exitMaintenanceMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate host reboot",
				Summary: "Initiates a host reboot",
			},
			Key: "HostSystem.reboot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate host shutdown",
				Summary: "Initiates a host shutdown",
			},
			Key: "HostSystem.shutdown",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enter standby mode",
				Summary: "Puts this host into standby mode",
			},
			Key: "HostSystem.enterStandbyMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Exit standby mode",
				Summary: "Brings this host out of standby mode",
			},
			Key: "HostSystem.exitStandbyMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query host overhead",
				Summary: "Determines the amount of memory overhead necessary to power on a virtual machine with the specified characteristics",
			},
			Key: "HostSystem.queryOverhead",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query memory overhead",
				Summary: "Query memory overhead",
			},
			Key: "HostSystem.queryOverheadEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure vSphere HA host",
				Summary: "Reconfigures the host for vSphere HA",
			},
			Key: "HostSystem.reconfigureDAS",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve Patch Manager",
				Summary: "Retrieves a reference to Patch Manager",
			},
			Key: "HostSystem.retrievePatchManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update host system flags",
				Summary: "Update the flags of the host system",
			},
			Key: "HostSystem.updateFlags",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Send Wake-on-LAN packet",
				Summary: "Send Wake-on-LAN packets to the physical NICs specified",
			},
			Key: "HostSystem.sendWakeOnLanPacket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable lockdown mode",
				Summary: "Enable lockdown mode on this host",
			},
			Key: "HostSystem.disableAdmin",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable lockdown mode",
				Summary: "Disable lockdown mode on this host",
			},
			Key: "HostSystem.enableAdmin",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable lockdown mode",
				Summary: "Enable lockdown mode on this host",
			},
			Key: "HostSystem.enterLockdownMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable lockdown mode",
				Summary: "Disable lockdown mode on this host",
			},
			Key: "HostSystem.exitLockdownMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update management server IP",
				Summary: "Update information about the vCenter Server managing this host",
			},
			Key: "HostSystem.updateManagementServerIp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire CIM service",
				Summary: "Establish a remote connection to a CIM interface",
			},
			Key: "HostSystem.acquireCimServicesTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update IPMI or ILO information used by DPM",
				Summary: "Update IPMI or ILO information for this host used by DPM",
			},
			Key: "HostSystem.updateIpmi",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update SSL thumbprint registry",
				Summary: "Updates the SSL thumbprint registry on the host",
			},
			Key: "HostSystem.updateSslThumbprintInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve host hardware uptime",
				Summary: "Retrieves the hardware uptime for the host in seconds",
			},
			Key: "HostSystem.retrieveHardwareUptime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve Dynamic Type Manager",
				Summary: "Retrieves a reference to Dynamic Type Manager",
			},
			Key: "HostSystem.retrieveDynamicTypeManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve Managed Method Executer",
				Summary: "Retrieves a referemce to Managed Method Executer",
			},
			Key: "HostSystem.retrieveManagedMethodExecuter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual machine memory overhead",
				Summary: "Query memory overhead for a virtual machine power on",
			},
			Key: "HostSystem.queryOverheadEx2",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Test EVC mode",
				Summary: "Test an EVC mode on a host",
			},
			Key: "HostSystem.testEvcMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply EVC mode",
				Summary: "Applies an EVC mode to a host",
			},
			Key: "HostSystem.applyEvcMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check whether the certificate is trusted by vCenter Server",
				Summary: "Checks whether the certificate matches the host certificate that vCenter Server trusts",
			},
			Key: "HostSystem.checkCertificateTrusted",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Prepare host",
				Summary: "Prepare host for encryption",
			},
			Key: "HostSystem.prepareCrypto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable encryption",
				Summary: "Enable encryption on the current host",
			},
			Key: "HostSystem.enableCrypto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure the host key",
				Summary: "Configure the encryption key on the current host",
			},
			Key: "HostSystem.configureCryptoKey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch set custom value",
				Summary: "vSphere Distributed Switch set custom value",
			},
			Key: "DistributedVirtualSwitch.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch reload",
				Summary: "vSphere Distributed Switch reload",
			},
			Key: "DistributedVirtualSwitch.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename vSphere Distributed Switch",
				Summary: "Rename vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete vSphere Distributed Switch",
				Summary: "Delete vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch add tag",
				Summary: "vSphere Distributed Switch add tag",
			},
			Key: "DistributedVirtualSwitch.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch remove tag",
				Summary: "vSphere Distributed Switch remove tag",
			},
			Key: "DistributedVirtualSwitch.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "DistributedVirtualSwitch.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve dvPort keys",
				Summary: "Retrieve dvPort keys",
			},
			Key: "DistributedVirtualSwitch.fetchPortKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve dvPorts",
				Summary: "Retrieve dvPorts",
			},
			Key: "DistributedVirtualSwitch.fetchPorts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query vSphere Distributed Switch used virtual LAN ID",
				Summary: "Query vSphere Distributed Switch used virtual LAN ID",
			},
			Key: "DistributedVirtualSwitch.queryUsedVlanId",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure vSphere Distributed Switch",
				Summary: "Reconfigure vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch product specification operation",
				Summary: "vSphere Distributed Switch product specification operation",
			},
			Key: "DistributedVirtualSwitch.performProductSpecOperation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Merge vSphere Distributed Switches",
				Summary: "Merge vSphere Distributed Switches",
			},
			Key: "DistributedVirtualSwitch.merge",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Distributed Port Group",
				Summary: "Add Distributed Port Group",
			},
			Key: "DistributedVirtualSwitch.addPortgroups",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move dvPorts",
				Summary: "Move dvPorts",
			},
			Key: "DistributedVirtualSwitch.movePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSphere Distributed Switch capability",
				Summary: "Update vSphere Distributed Switch capability",
			},
			Key: "DistributedVirtualSwitch.updateCapability",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure dvPort",
				Summary: "Reconfigure dvPort",
			},
			Key: "DistributedVirtualSwitch.reconfigurePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh dvPort state",
				Summary: "Refresh dvPort state",
			},
			Key: "DistributedVirtualSwitch.refreshPortState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rectify host in vSphere Distributed Switch",
				Summary: "Rectify host in vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.rectifyHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network resource pools on vSphere Distributed Switch",
				Summary: "Update network resource pools on vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.updateNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add network resource pools on vSphere Distributed Switch",
				Summary: "Add network resource pools on vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.addNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove network resource pools on vSphere Distributed Switch",
				Summary: "Remove network resource pools on vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.removeNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure a network resource pool on a distributed switch",
				Summary: "Reconfigures the network resource pool on a distributed switch",
			},
			Key: "DistributedVirtualSwitch.reconfigureVmVnicNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network I/O control on vSphere Distributed Switch",
				Summary: "Update network I/O control on vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.enableNetworkResourceManagement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get vSphere Distributed Switch configuration spec to rollback",
				Summary: "Get vSphere Distributed Switch configuration spec to rollback",
			},
			Key: "DistributedVirtualSwitch.rollback",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Distributed Port Group",
				Summary: "Add Distributed Port Group",
			},
			Key: "DistributedVirtualSwitch.addPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update health check configuration on vSphere Distributed Switch",
				Summary: "Update health check configuration on vSphere Distributed Switch",
			},
			Key: "DistributedVirtualSwitch.updateHealthCheckConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Look up portgroup based on portgroup key",
				Summary: "Look up portgroup based on portgroup key",
			},
			Key: "DistributedVirtualSwitch.lookupPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Annotate OVF section tree",
				Summary: "Annotates the given OVF section tree with configuration choices for this OVF consumer",
			},
			Key: "OvfConsumer.annotateOst",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate instantiation OVF section tree",
				Summary: "Validates that this OVF consumer can accept an instantiation OVF section tree",
			},
			Key: "OvfConsumer.validateInstantiationOst",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Request registration of OVF section tree nodes",
				Summary: "Notifies the OVF consumer that the specified OVF section tree nodes should be registered",
			},
			Key: "OvfConsumer.registerEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Request managed entities unregistration from OVF consumer",
				Summary: "Notifies the OVF consumer that the specified managed entities should be unregistered",
			},
			Key: "OvfConsumer.unregisterEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Notify OVF consumer for cloned entities",
				Summary: "Notifies the OVF consumer that the specified entities have been cloned",
			},
			Key: "OvfConsumer.cloneEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Populate entity OVF section tree",
				Summary: "Create OVF sections for the given managed entities and populate the entity OVF section tree",
			},
			Key: "OvfConsumer.populateEntityOst",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve public OVF environment sections for virtual machine ",
				Summary: "Retrieves the public OVF environment sections that this OVF consumer has for a given virtual machine",
			},
			Key: "OvfConsumer.retrievePublicOvfEnvironmentSections",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Notify OVF consumer for virtual machine power on",
				Summary: "Notifies the OVF consumer that a virtual machine is about to be powered on",
			},
			Key: "OvfConsumer.notifyPowerOn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set snapshot custom value",
				Summary: "Sets the value of a custom field of a virtual machine snapshot",
			},
			Key: "vm.Snapshot.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Revert snapshot",
				Summary: "Change the execution state of the virtual machine to the state of this snapshot",
			},
			Key: "vm.Snapshot.revert",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove snapshot",
				Summary: "Remove snapshot and delete its associated storage",
			},
			Key: "vm.Snapshot.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename snapshot",
				Summary: "Rename the snapshot",
			},
			Key: "vm.Snapshot.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Linked Clone",
				Summary: "Create a linked clone from this snapshot",
			},
			Key: "vm.Snapshot.createLinkedClone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF template",
				Summary: "Export the snapshot as an OVF template",
			},
			Key: "vm.Snapshot.exportSnapshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check compliance",
				Summary: "Check compliance of host or cluster against a profile",
			},
			Key: "profile.ComplianceManager.checkCompliance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query compliance status",
				Summary: "Query compliance status",
			},
			Key: "profile.ComplianceManager.queryComplianceStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryEntitiesByComplianceStatus",
				Summary: "queryEntitiesByComplianceStatus",
			},
			Key: "profile.ComplianceManager.queryEntitiesByComplianceStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear compliance history",
				Summary: "Clear historical compliance data",
			},
			Key: "profile.ComplianceManager.clearComplianceStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query expression metadata",
				Summary: "Query expression metadata",
			},
			Key: "profile.ComplianceManager.queryExpressionMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create alarm",
				Summary: "Create a new alarm",
			},
			Key: "alarm.AlarmManager.create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve alarm",
				Summary: "Get available alarms defined on the entity",
			},
			Key: "alarm.AlarmManager.getAlarm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get alarm actions enabled",
				Summary: "Checks if alarm actions are enabled for an entity",
			},
			Key: "alarm.AlarmManager.getAlarmActionsEnabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set alarm actions enabled",
				Summary: "Enables or disables firing alarm actions for an entity",
			},
			Key: "alarm.AlarmManager.setAlarmActionsEnabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get alarm state",
				Summary: "The state of instantiated alarms on the entity",
			},
			Key: "alarm.AlarmManager.getAlarmState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acknowledge alarm",
				Summary: "Stops alarm actions from firing until the alarm next triggers on an entity",
			},
			Key: "alarm.AlarmManager.acknowledgeAlarm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set alarm status",
				Summary: "Sets the status of an alarm for an entity",
			},
			Key: "alarm.AlarmManager.setAlarmStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "clearTriggeredAlarms",
				Summary: "clearTriggeredAlarms",
			},
			Key: "alarm.AlarmManager.clearTriggeredAlarms",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "testSMTPSetup",
				Summary: "testSMTPSetup",
			},
			Key: "alarm.AlarmManager.testSMTPSetup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create private alarm on managed entity",
				Summary: "Creates a Private (trigger-only) Alarm on a managed entity",
			},
			Key: "alarm.AlarmManager.createPrivateAlarm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query private alarms on managed entity",
				Summary: "Retrieves all of the Private (trigger-only) Alarms defined on the specified managed entity",
			},
			Key: "alarm.AlarmManager.queryPrivateAlarms",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Sync triggered alarms list",
				Summary: "Retrieves the full list of currently-triggered Alarms, as a list of triggers",
			},
			Key: "alarm.AlarmManager.syncTriggeredAlarms",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve queued-up alarm triggers",
				Summary: "Retrieves any queued-up alarm triggers representing Alarm state changes since the last time this method was called",
			},
			Key: "alarm.AlarmManager.retrieveTriggers",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update the VASA provider state",
				Summary: "Updates the VASA provider state for the specified datastores",
			},
			Key: "VasaVvolManager.updateVasaProviderState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Virtual Volume datastore",
				Summary: "Creates a new Virtual Volume datastore",
			},
			Key: "VasaVvolManager.createVVolDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove Virtual Volume datastore",
				Summary: "Remove Virtual Volume datastore from specified hosts",
			},
			Key: "VasaVvolManager.removeVVolDatastore",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update the VASA client context",
				Summary: "Updates the VASA client context on the host",
			},
			Key: "VasaVvolManager.updateVasaClientContext",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "fetchRelocatedMACAddress",
				Summary: "fetchRelocatedMACAddress",
			},
			Key: "NetworkManager.fetchRelocatedMACAddress",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check MAC addresses in use",
				Summary: "Checks the MAC addresses used by this vCenter Server instance",
			},
			Key: "NetworkManager.checkIfMACAddressInUse",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reclaim MAC addresses",
				Summary: "Reclaims the MAC addresses that are not used by remote vCenter Server instances",
			},
			Key: "NetworkManager.reclaimMAC",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create new identity binding",
				Summary: "Creates a new identity binding between the host and vCenter Server",
			},
			Key: "host.TpmManager.requestIdentity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Verify authenticity of credential",
				Summary: "Verifies the authenticity and correctness of the supplied attestation credential",
			},
			Key: "host.TpmManager.verifyCredential",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate integrity report",
				Summary: "Generates an integrity report for the selected components",
			},
			Key: "host.TpmManager.generateReport",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Distributed Port Group set custom value",
				Summary: "Distributed Port Group set custom value",
			},
			Key: "dvs.DistributedVirtualPortgroup.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload Distributed Port Group",
				Summary: "Reload Distributed Port Group",
			},
			Key: "dvs.DistributedVirtualPortgroup.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename Distributed Port Group",
				Summary: "Rename Distributed Port Group",
			},
			Key: "dvs.DistributedVirtualPortgroup.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete Distributed Port Group",
				Summary: "Delete Distributed Port Group",
			},
			Key: "dvs.DistributedVirtualPortgroup.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag to Distributed Port Group",
				Summary: "Add tag to Distributed Port Group",
			},
			Key: "dvs.DistributedVirtualPortgroup.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Distributed Port Group remove tag",
				Summary: "Distributed Port Group remove tag",
			},
			Key: "dvs.DistributedVirtualPortgroup.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "dvs.DistributedVirtualPortgroup.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Distributed Port Group delete network",
				Summary: "Distributed Port Group delete network",
			},
			Key: "dvs.DistributedVirtualPortgroup.destroyNetwork",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure Distributed Port Group",
				Summary: "Reconfigure Distributed Port Group",
			},
			Key: "dvs.DistributedVirtualPortgroup.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get Distributed Port Group configuration spec to rollback",
				Summary: "Get Distributed Port Group configuration spec to rollback",
			},
			Key: "dvs.DistributedVirtualPortgroup.rollback",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set alarm custom value",
				Summary: "Sets the value of a custom field of an alarm",
			},
			Key: "alarm.Alarm.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove alarm",
				Summary: "Remove the alarm",
			},
			Key: "alarm.Alarm.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure alarm",
				Summary: "Reconfigure the alarm",
			},
			Key: "alarm.Alarm.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set compute-resource custom value",
				Summary: "Sets the value of a custom field for a unified compute resource",
			},
			Key: "ComputeResource.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload resource",
				Summary: "Reloads the resource",
			},
			Key: "ComputeResource.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename compute-resource",
				Summary: "Rename the compute-resource",
			},
			Key: "ComputeResource.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove host",
				Summary: "Removes the host resource",
			},
			Key: "ComputeResource.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to this object",
			},
			Key: "ComputeResource.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Removes a set of tags from this object",
			},
			Key: "ComputeResource.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "ComputeResource.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure compute-resource",
				Summary: "Reconfigures a compute-resource",
			},
			Key: "ComputeResource.reconfigureEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set latest page size",
				Summary: "Set the last page viewed size and contain at most maxCount items in the page",
			},
			Key: "HistoryCollector.setLatestPageSize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rewind",
				Summary: "Move the scroll position to the oldest item",
			},
			Key: "HistoryCollector.rewind",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset",
				Summary: "Move the scroll position to the item just above the last page viewed",
			},
			Key: "HistoryCollector.reset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove collector",
				Summary: "Remove the collector from server",
			},
			Key: "HistoryCollector.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update specific metadata",
				Summary: "Update specific metadata for the given owner and list of virtual machine IDs",
			},
			Key: "vm.MetadataManager.updateMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve specific metadata",
				Summary: "Retrieve specific metadata for the given owner and list of virtual machine IDs",
			},
			Key: "vm.MetadataManager.retrieveMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve all metadata",
				Summary: "Retrieve all metadata for the given owner and datastore",
			},
			Key: "vm.MetadataManager.retrieveAllMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clear metadata",
				Summary: "Clear all metadata for the given owner and datastore",
			},
			Key: "vm.MetadataManager.clearMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query latest statistics for a virtual machine",
				Summary: "Queries the latest values of performance statistics of a virtual machine",
			},
			Key: "InternalStatsCollector.queryLatestVmStats",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch set custom value",
				Summary: "vSphere Distributed Switch set custom value",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload vSphere Distributed Switch",
				Summary: "Reload vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename vSphere Distributed Switch",
				Summary: "Rename vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove vSphere Distributed Switch",
				Summary: "Remove vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch add tag",
				Summary: "vSphere Distributed Switch add tag",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch remove tag",
				Summary: "vSphere Distributed Switch remove tag",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve dvPort keys",
				Summary: "Retrieve dvPort keys",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.fetchPortKeys",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve dvPorts",
				Summary: "Retrieve dvPorts",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.fetchPorts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query used virtual LAN ID",
				Summary: "Query used virtual LAN ID",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.queryUsedVlanId",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure vSphere Distributed Switch",
				Summary: "Reconfigure vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSphere Distributed Switch product specification operation",
				Summary: "vSphere Distributed Switch product specification operation",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.performProductSpecOperation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Merge vSphere Distributed Switch",
				Summary: "Merge vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.merge",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Distributed Port Groups",
				Summary: "Add Distributed Port Groups",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.addPortgroups",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move dvPort",
				Summary: "Move dvPort",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.movePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSphere Distributed Switch capability",
				Summary: "Update vSphere Distributed Switch capability",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.updateCapability",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure dvPort",
				Summary: "Reconfigure dvPort",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.reconfigurePort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh dvPort state",
				Summary: "Refresh dvPort state",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.refreshPortState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rectify vSphere Distributed Switch host",
				Summary: "Rectify vSphere Distributed Switch host",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.rectifyHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network resource pools on vSphere Distributed Switch",
				Summary: "Update network resource pools on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.updateNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add network resource pools on vSphere Distributed Switch",
				Summary: "Add network resource pools on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.addNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove network resource pools on vSphere Distributed Switch",
				Summary: "Remove network resource pools on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.removeNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure a network resource pool on a distributed switch",
				Summary: "Reconfigures a network resource pool on a distributed switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.reconfigureVmVnicNetworkResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update network I/O control on vSphere Distributed Switch",
				Summary: "Update network I/O control on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.enableNetworkResourceManagement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get vSphere Distributed Switch configuration spec to rollback",
				Summary: "Get vSphere Distributed Switch configuration spec to rollback",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.rollback",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Distributed Port Group",
				Summary: "Add Distributed Port Group",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.addPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update health check configuration on vSphere Distributed Switch",
				Summary: "Update health check configuration on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.updateHealthCheckConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Look up portgroup based on portgroup key",
				Summary: "Look up portgroup based on portgroup key",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.lookupPortgroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Link Aggregation Control Protocol groups on vSphere Distributed Switch",
				Summary: "Update Link Aggregation Control Protocol groups on vSphere Distributed Switch",
			},
			Key: "dvs.VmwareDistributedVirtualSwitch.updateLacpGroupConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a virtual disk object",
				Summary: "Create a virtual disk object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.createDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register a legacy disk to be a virtual disk object",
				Summary: "Register a legacy disk to be a virtual disk object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.registerDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extend a virtual disk to the new capacity",
				Summary: "Extend a virtual disk to the new capacity",
			},
			Key: "vslm.vcenter.VStorageObjectManager.extendDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Inflate a thin virtual disk",
				Summary: "Inflate a thin virtual disk",
			},
			Key: "vslm.vcenter.VStorageObjectManager.inflateDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename a virtual storage object",
				Summary: "Rename a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.renameVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update storage policy on a virtual storage object",
				Summary: "Update storage policy on a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.updateVStorageObjectPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete a virtual storage object",
				Summary: "Delete a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.deleteVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve a virtual storage object",
				Summary: "Retrieve a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.retrieveVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveVStorageObjectState",
				Summary: "retrieveVStorageObjectState",
			},
			Key: "vslm.vcenter.VStorageObjectManager.retrieveVStorageObjectState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "List virtual storage objects on a datastore",
				Summary: "List virtual storage objects on a datastore",
			},
			Key: "vslm.vcenter.VStorageObjectManager.listVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone a virtual storage object",
				Summary: "Clone a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.cloneVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Relocate a virtual storage object",
				Summary: "Relocate a virtual storage object",
			},
			Key: "vslm.vcenter.VStorageObjectManager.relocateVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "attachTagToVStorageObject",
				Summary: "attachTagToVStorageObject",
			},
			Key: "vslm.vcenter.VStorageObjectManager.attachTagToVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "detachTagFromVStorageObject",
				Summary: "detachTagFromVStorageObject",
			},
			Key: "vslm.vcenter.VStorageObjectManager.detachTagFromVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "listVStorageObjectsAttachedToTag",
				Summary: "listVStorageObjectsAttachedToTag",
			},
			Key: "vslm.vcenter.VStorageObjectManager.listVStorageObjectsAttachedToTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "listTagsAttachedToVStorageObject",
				Summary: "listTagsAttachedToVStorageObject",
			},
			Key: "vslm.vcenter.VStorageObjectManager.listTagsAttachedToVStorageObject",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconcile datastore inventory",
				Summary: "Reconcile datastore inventory",
			},
			Key: "vslm.vcenter.VStorageObjectManager.reconcileDatastoreInventory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Schedule reconcile datastore inventory",
				Summary: "Schedule reconcile datastore inventory",
			},
			Key: "vslm.vcenter.VStorageObjectManager.scheduleReconcileDatastoreInventory",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check group membership",
				Summary: "Check whether a user is a member of a given list of groups",
			},
			Key: "UserDirectory.checkGroupMembership",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get user groups",
				Summary: "Searches for users and groups",
			},
			Key: "UserDirectory.retrieveUserGroups",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create profile",
				Summary: "Create profile",
			},
			Key: "profile.ProfileManager.createProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query policy metadata",
				Summary: "Query policy metadata",
			},
			Key: "profile.ProfileManager.queryPolicyMetadata",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find associated profile",
				Summary: "Find associated profile",
			},
			Key: "profile.ProfileManager.findAssociatedProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate host for OVF package compatibility",
				Summary: "Validates if a host is compatible with the requirements in an OVF package",
			},
			Key: "OvfManager.validateHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Parse OVF descriptor",
				Summary: "Parses and validates an OVF descriptor",
			},
			Key: "OvfManager.parseDescriptor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Convert OVF descriptor",
				Summary: "Convert OVF descriptor to entity specification",
			},
			Key: "OvfManager.createImportSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create an OVF descriptor",
				Summary: "Creates an OVF descriptor from either a VM or vApp",
			},
			Key: "OvfManager.createDescriptor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Parse OVF Descriptor at URL",
				Summary: "Parses and validates an OVF descriptor at a given URL",
			},
			Key: "OvfManager.parseDescriptorAtUrl",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy OVF template",
				Summary: "Deploys an OVF template from a URL",
			},
			Key: "OvfManager.importOvfAtUrl",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export as OVF template",
				Summary: "Uploads OVF template to a remote server",
			},
			Key: "OvfManager.exportOvfToUrl",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update global message",
				Summary: "Updates the system global message",
			},
			Key: "SessionManager.updateMessage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login by token",
				Summary: "Logs on to the server through token representing principal identity",
			},
			Key: "SessionManager.loginByToken",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login",
				Summary: "Create a login session",
			},
			Key: "SessionManager.login",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login by SSPI",
				Summary: "Log on to the server using SSPI passthrough authentication",
			},
			Key: "SessionManager.loginBySSPI",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login by SSL thumbprint",
				Summary: "Log on to the server using SSL thumbprint authentication",
			},
			Key: "SessionManager.loginBySSLThumbprint",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login by session ticket",
				Summary: "Log on to the server using a session ticket",
			},
			Key: "SessionManager.loginBySessionTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire session ticket",
				Summary: "Acquire a ticket for authenticating to a remote service",
			},
			Key: "SessionManager.acquireSessionTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Logout",
				Summary: "Logout and end the current session",
			},
			Key: "SessionManager.logout",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire local ticket",
				Summary: "Acquire one-time ticket for authenticating server-local client",
			},
			Key: "SessionManager.acquireLocalTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire generic service ticket",
				Summary: "Acquire a one-time credential that may be used to make the specified request",
			},
			Key: "SessionManager.acquireGenericServiceTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Terminate session",
				Summary: "Logout and end the provided list of sessions",
			},
			Key: "SessionManager.terminate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set locale",
				Summary: "Set the session locale for determining the languages used for messages and formatting data",
			},
			Key: "SessionManager.setLocale",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login extension",
				Summary: "Creates a privileged login session for an extension",
			},
			Key: "SessionManager.loginExtension",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login extension",
				Summary: "Invalid subject name",
			},
			Key: "SessionManager.loginExtensionBySubjectName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Login extension by certificate",
				Summary: "Login extension by certificate",
			},
			Key: "SessionManager.loginExtensionByCertificate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Impersonate user",
				Summary: "Convert session to impersonate specified user",
			},
			Key: "SessionManager.impersonateUser",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Session active query",
				Summary: "Validates that a currently active session exists",
			},
			Key: "SessionManager.sessionIsActive",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire clone ticket",
				Summary: "Acquire a session-specific ticket string that can be used to clone the current session",
			},
			Key: "SessionManager.acquireCloneTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone session",
				Summary: "Clone the specified session and associate it with the current connection",
			},
			Key: "SessionManager.cloneSession",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Open remote disk for read/write",
				Summary: "Opens a disk on a virtual machine for read/write access",
			},
			Key: "NfcService.randomAccessOpen",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Open remote disk for read",
				Summary: "Opens a disk on a virtual machine for read access",
			},
			Key: "NfcService.randomAccessOpenReadonly",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "randomAccessFileOpen",
				Summary: "randomAccessFileOpen",
			},
			Key: "NfcService.randomAccessFileOpen",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read virtual machine files",
				Summary: "Read files associated with a virtual machine",
			},
			Key: "NfcService.getVmFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Write virtual machine files",
				Summary: "Write files associated with a virtual machine",
			},
			Key: "NfcService.putVmFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Manipulate file paths",
				Summary: "Permission to manipulate file paths",
			},
			Key: "NfcService.fileManagement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Manipulate system-related file paths",
				Summary: "Permission to manipulate all system related file paths",
			},
			Key: "NfcService.systemManagement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "getServerNfcLibVersion",
				Summary: "getServerNfcLibVersion",
			},
			Key: "NfcService.getServerNfcLibVersion",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "registerProvider",
				Summary: "registerProvider",
			},
			Key: "HealthUpdateManager.registerProvider",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unregisterProvider",
				Summary: "unregisterProvider",
			},
			Key: "HealthUpdateManager.unregisterProvider",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryProviderList",
				Summary: "queryProviderList",
			},
			Key: "HealthUpdateManager.queryProviderList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "hasProvider",
				Summary: "hasProvider",
			},
			Key: "HealthUpdateManager.hasProvider",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryProviderName",
				Summary: "queryProviderName",
			},
			Key: "HealthUpdateManager.queryProviderName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryHealthUpdateInfos",
				Summary: "queryHealthUpdateInfos",
			},
			Key: "HealthUpdateManager.queryHealthUpdateInfos",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addMonitoredEntities",
				Summary: "addMonitoredEntities",
			},
			Key: "HealthUpdateManager.addMonitoredEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeMonitoredEntities",
				Summary: "removeMonitoredEntities",
			},
			Key: "HealthUpdateManager.removeMonitoredEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryMonitoredEntities",
				Summary: "queryMonitoredEntities",
			},
			Key: "HealthUpdateManager.queryMonitoredEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "hasMonitoredEntity",
				Summary: "hasMonitoredEntity",
			},
			Key: "HealthUpdateManager.hasMonitoredEntity",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryUnmonitoredHosts",
				Summary: "queryUnmonitoredHosts",
			},
			Key: "HealthUpdateManager.queryUnmonitoredHosts",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "postHealthUpdates",
				Summary: "postHealthUpdates",
			},
			Key: "HealthUpdateManager.postHealthUpdates",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryHealthUpdates",
				Summary: "queryHealthUpdates",
			},
			Key: "HealthUpdateManager.queryHealthUpdates",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addFilter",
				Summary: "addFilter",
			},
			Key: "HealthUpdateManager.addFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryFilterList",
				Summary: "queryFilterList",
			},
			Key: "HealthUpdateManager.queryFilterList",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryFilterName",
				Summary: "queryFilterName",
			},
			Key: "HealthUpdateManager.queryFilterName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryFilterInfoIds",
				Summary: "queryFilterInfoIds",
			},
			Key: "HealthUpdateManager.queryFilterInfoIds",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryFilterEntities",
				Summary: "queryFilterEntities",
			},
			Key: "HealthUpdateManager.queryFilterEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "addFilterEntities",
				Summary: "addFilterEntities",
			},
			Key: "HealthUpdateManager.addFilterEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeFilterEntities",
				Summary: "removeFilterEntities",
			},
			Key: "HealthUpdateManager.removeFilterEntities",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "removeFilter",
				Summary: "removeFilter",
			},
			Key: "HealthUpdateManager.removeFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set vMotion custom value",
				Summary: "Sets the value of a custom field of a host vMotion system",
			},
			Key: "host.VMotionSystem.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update IP configuration",
				Summary: "Update the IP configuration of the vMotion virtual NIC",
			},
			Key: "host.VMotionSystem.updateIpConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Select vMotion virtual NIC",
				Summary: "Select the virtual NIC to be used for vMotion",
			},
			Key: "host.VMotionSystem.selectVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deselect vMotion virtual NIC",
				Summary: "Deselect the virtual NIC to be used for vMotion",
			},
			Key: "host.VMotionSystem.deselectVnic",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add custom field",
				Summary: "Creates a new custom property",
			},
			Key: "CustomFieldsManager.addFieldDefinition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove custom field",
				Summary: "Removes a custom property",
			},
			Key: "CustomFieldsManager.removeFieldDefinition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename custom property",
				Summary: "Renames a custom property",
			},
			Key: "CustomFieldsManager.renameFieldDefinition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set custom field",
				Summary: "Assigns a value to a custom property",
			},
			Key: "CustomFieldsManager.setField",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get ManagedEntities",
				Summary: "Get the list of ManagedEntities that the name is a Substring of the custom field name and the value is a Substring of the field value.",
			},
			Key: "CustomFieldsManager.getEntitiesWithCustomFieldAndValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomFields",
				Summary: "retrieveCustomFields",
			},
			Key: "CustomFieldsManager.retrieveCustomFields",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure virtual disk digest",
				Summary: "Controls the configuration of the digests for the virtual disks",
			},
			Key: "CbrcManager.configureDigest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Recompute virtual disk digest",
				Summary: "Recomputes the digest for the given virtual disks, if necessary",
			},
			Key: "CbrcManager.recomputeDigest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk digest configuration",
				Summary: "Returns the current configuration of the digest for the given digest-enabled virtual disks",
			},
			Key: "CbrcManager.queryDigestInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual disk digest runtime information",
				Summary: "Returns the status of runtime digest usage for the given digest-enabled virtual disks",
			},
			Key: "CbrcManager.queryDigestRuntimeInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get diagnostic files",
				Summary: "Gets the list of diagnostic files for a given system",
			},
			Key: "DiagnosticManager.queryDescriptions",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Browse diagnostic manager",
				Summary: "Returns part of a log file",
			},
			Key: "DiagnosticManager.browse",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Generate system logs bundles",
				Summary: "Instructs the server to generate system logs bundles",
			},
			Key: "DiagnosticManager.generateLogBundles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query file hash",
				Summary: "Queries file integrity information",
			},
			Key: "DiagnosticManager.queryFileHash",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure workload model calculation parameters for datastore",
				Summary: "Configures calculation parameters used for computation of workload model for a datastore",
			},
			Key: "DrsStatsManager.configureWorkloadCharacterization",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query current workload model calculation parameters",
				Summary: "Queries a host for the current workload model calculation parameters",
			},
			Key: "DrsStatsManager.queryWorkloadCharacterization",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure datastore correlation detector",
				Summary: "Configures datastore correlation detector with datastore to datastore cluster mappings",
			},
			Key: "DrsStatsManager.configureCorrelationDetector",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query datastore correlation result",
				Summary: "Queries correlation detector for a list of datastores correlated to a given datastore",
			},
			Key: "DrsStatsManager.queryCorrelationResult",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update agent virtual machine information",
				Summary: "Updates agent virtual machine information",
			},
			Key: "EsxAgentConfigManager.updateAgentVmInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query agent virtual machine information",
				Summary: "Returns the state for each of the specified agent virtual machines",
			},
			Key: "EsxAgentConfigManager.queryAgentVmInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update compute resource agent information",
				Summary: "Updates the number of required agent virtual machines for one or more compute resources",
			},
			Key: "EsxAgentConfigManager.updateComputeResourceAgentInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query compute resource agent information",
				Summary: "Retrieves the agent information for one or more compute resources",
			},
			Key: "EsxAgentConfigManager.queryComputeResourceAgentInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set extensible custom value",
				Summary: "Sets the value of a custom field of an extensible managed object",
			},
			Key: "ExtensibleManagedObject.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get lease download manifest",
				Summary: "Gets the download manifest for this lease",
			},
			Key: "HttpNfcLease.getManifest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Complete the lease",
				Summary: "The lease completed successfully",
			},
			Key: "HttpNfcLease.complete",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "End the lease",
				Summary: "The lease has ended",
			},
			Key: "HttpNfcLease.abort",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update lease progress",
				Summary: "Updates lease progress",
			},
			Key: "HttpNfcLease.progress",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install IO Filter",
				Summary: "Installs an IO Filter on a compute resource",
			},
			Key: "IoFilterManager.installIoFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Uninstall IO Filter",
				Summary: "Uninstalls an IO Filter from a compute resource",
			},
			Key: "IoFilterManager.uninstallIoFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade IO Filter",
				Summary: "Upgrades an IO Filter on a compute resource",
			},
			Key: "IoFilterManager.upgradeIoFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query IO Filter installation issues",
				Summary: "Queries IO Filter installation issues on a compute resource",
			},
			Key: "IoFilterManager.queryIssue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryIoFilterInfo",
				Summary: "queryIoFilterInfo",
			},
			Key: "IoFilterManager.queryIoFilterInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resolve IO Filter installation errors on host",
				Summary: "Resolves IO Filter installation errors on a host",
			},
			Key: "IoFilterManager.resolveInstallationErrorsOnHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resolve IO Filter installation errors on cluster",
				Summary: "Resolves IO Filter installation errors on a cluster",
			},
			Key: "IoFilterManager.resolveInstallationErrorsOnCluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query information about virtual disks using IO Filter",
				Summary: "Queries information about virtual disks that use an IO Filter installed on a compute resource",
			},
			Key: "IoFilterManager.queryDisksUsingFilter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update IO Filter policy",
				Summary: "Updates the policy to IO Filter mapping in vCenter Server",
			},
			Key: "IoFilterManager.updateIoFilterPolicy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query supported features",
				Summary: "Searches the current license source for licenses available from this system",
			},
			Key: "LicenseManager.querySupportedFeatures",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query license source",
				Summary: "Searches the current license source for licenses available for each feature known to this system",
			},
			Key: "LicenseManager.querySourceAvailability",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query license usage",
				Summary: "Returns the list of features and the number of licenses that have been reserved",
			},
			Key: "LicenseManager.queryUsage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set product edition",
				Summary: "Defines the product edition",
			},
			Key: "LicenseManager.setEdition",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check feature",
				Summary: "Checks if a feature is enabled",
			},
			Key: "LicenseManager.checkFeature",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable license",
				Summary: "Enable a feature that is marked as user-configurable",
			},
			Key: "LicenseManager.enable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable license",
				Summary: "Release licenses for a user-configurable feature",
			},
			Key: "LicenseManager.disable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure license source",
				Summary: "Allows reconfiguration of the License Manager license source",
			},
			Key: "LicenseManager.configureSource",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Installing license",
				Summary: "Installing license",
			},
			Key: "LicenseManager.updateLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add license",
				Summary: "Adds a new license to the license inventory",
			},
			Key: "LicenseManager.addLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove license",
				Summary: "Removes a license from the license inventory",
			},
			Key: "LicenseManager.removeLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Decode license",
				Summary: "Decodes the license to return the properties of that license key",
			},
			Key: "LicenseManager.decodeLicense",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update license label",
				Summary: "Update a license's label",
			},
			Key: "LicenseManager.updateLabel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove license label",
				Summary: "Removes a license's label",
			},
			Key: "LicenseManager.removeLabel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get License Data Manager",
				Summary: "Gets the License Data Manager",
			},
			Key: "LicenseManager.queryLicenseDataManager",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Activate remote hard enforcement",
				Summary: "Activates the remote hard enforcement",
			},
			Key: "LicenseManager.activateRemoteHardEnforcement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add end point",
				Summary: "Add a service whose connections are to be proxied",
			},
			Key: "ProxyService.addEndpoint",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove end point",
				Summary: "End point to be detached",
			},
			Key: "ProxyService.removeEndpoint",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Estimate database size",
				Summary: "Estimates the database size required to store VirtualCenter data",
			},
			Key: "ResourcePlanningManager.estimateDatabaseSize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find entity by UUID",
				Summary: "Finds a virtual machine or host by UUID",
			},
			Key: "SearchIndex.findByUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find virtual machine by datastore path",
				Summary: "Finds a virtual machine by its location on a datastore",
			},
			Key: "SearchIndex.findByDatastorePath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find entity by DNS",
				Summary: "Finds a virtual machine or host by its DNS name",
			},
			Key: "SearchIndex.findByDnsName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find entity by IP",
				Summary: "Finds a virtual machine or host by IP address",
			},
			Key: "SearchIndex.findByIp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find entity by inventory path",
				Summary: "Finds a virtual machine or host based on its location in the inventory",
			},
			Key: "SearchIndex.findByInventoryPath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find folder child",
				Summary: "Finds an immediate child of a folder",
			},
			Key: "SearchIndex.findChild",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find by UUID",
				Summary: "Find entities based on their UUID",
			},
			Key: "SearchIndex.findAllByUuid",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find by DNS name",
				Summary: "Find by DNS name",
			},
			Key: "SearchIndex.findAllByDnsName",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Find by IP address",
				Summary: "Find entities based on their IP address",
			},
			Key: "SearchIndex.findAllByIp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "findAllInstantCloneParentInGroup",
				Summary: "findAllInstantCloneParentInGroup",
			},
			Key: "SearchIndex.findAllInstantCloneParentInGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "findAllInstantCloneChildrenOfGroup",
				Summary: "findAllInstantCloneChildrenOfGroup",
			},
			Key: "SearchIndex.findAllInstantCloneChildrenOfGroup",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute client service",
				Summary: "Execute the client service",
			},
			Key: "SimpleCommand.Execute",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure Storage I/O Control on datastore",
				Summary: "Configure Storage I/O Control on datastore",
			},
			Key: "StorageResourceManager.ConfigureDatastoreIORM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure Storage I/O Control on datastore",
				Summary: "Configure Storage I/O Control on datastore",
			},
			Key: "StorageResourceManager.ConfigureDatastoreIORMOnHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query Storage I/O Control configuration options",
				Summary: "Query Storage I/O Control configuration options",
			},
			Key: "StorageResourceManager.QueryIORMConfigOption",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get storage I/O resource management device model",
				Summary: "Returns the device model computed for a given datastore by storage DRS",
			},
			Key: "StorageResourceManager.GetStorageIORMDeviceModel",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query datastore performance summary",
				Summary: "Query datastore performance metrics in summary form",
			},
			Key: "StorageResourceManager.queryDatastorePerformanceSummary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply a Storage DRS recommendation",
				Summary: "Apply a Storage DRS recommendation",
			},
			Key: "StorageResourceManager.applyRecommendationToPod",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply Storage DRS recommendations",
				Summary: "Apply Storage DRS recommendations",
			},
			Key: "StorageResourceManager.applyRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel storage DRS recommendation",
				Summary: "Cancels a storage DRS recommendation",
			},
			Key: "StorageResourceManager.cancelRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh storage DRS recommendation",
				Summary: "Refreshes the storage DRS recommendations on the specified datastore cluster",
			},
			Key: "StorageResourceManager.refreshRecommendation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "refreshRecommendationsForPod",
				Summary: "refreshRecommendationsForPod",
			},
			Key: "StorageResourceManager.refreshRecommendationsForPod",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure Storage DRS",
				Summary: "Configure Storage DRS on a datastore cluster",
			},
			Key: "StorageResourceManager.configureStorageDrsForPod",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Invoke storage DRS for placement recommendations",
				Summary: "Invokes storage DRS for placement recommendations",
			},
			Key: "StorageResourceManager.recommendDatastores",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "rankForPlacement",
				Summary: "rankForPlacement",
			},
			Key: "StorageResourceManager.rankForPlacement",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryStorageStatisticsByProfile",
				Summary: "queryStorageStatisticsByProfile",
			},
			Key: "StorageResourceManager.queryStorageStatisticsByProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set latest page size",
				Summary: "Set the last page viewed size and contain at most maxCount items in the page",
			},
			Key: "TaskHistoryCollector.setLatestPageSize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rewind",
				Summary: "Move the scroll position to the oldest item",
			},
			Key: "TaskHistoryCollector.rewind",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset",
				Summary: "Move the scroll position to the item just above the last page viewed",
			},
			Key: "TaskHistoryCollector.reset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove collector",
				Summary: "Remove the collector from server",
			},
			Key: "TaskHistoryCollector.remove",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read next",
				Summary: "The scroll position is moved to the next new page after the read",
			},
			Key: "TaskHistoryCollector.readNext",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Read previous",
				Summary: "The scroll position is moved to the next older page after the read",
			},
			Key: "TaskHistoryCollector.readPrev",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "performUpgradePreflightCheck",
				Summary: "performUpgradePreflightCheck",
			},
			Key: "VsanUpgradeSystem.performUpgradePreflightCheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryUpgradeStatus",
				Summary: "queryUpgradeStatus",
			},
			Key: "VsanUpgradeSystem.queryUpgradeStatus",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "performUpgrade",
				Summary: "performUpgrade",
			},
			Key: "VsanUpgradeSystem.performUpgrade",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set a custom property to an opaque network",
				Summary: "Sets the value of a custom field of an opaque network",
			},
			Key: "OpaqueNetwork.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload an opaque network",
				Summary: "Reloads the information about the opaque network",
			},
			Key: "OpaqueNetwork.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename an opaque network",
				Summary: "Renames an opaque network",
			},
			Key: "OpaqueNetwork.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete opaque network",
				Summary: "Deletes an opaque network if it is not used by any host or virtual machine",
			},
			Key: "OpaqueNetwork.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add a tag to an opaque network",
				Summary: "Adds a set of tags to the opaque network",
			},
			Key: "OpaqueNetwork.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove a tag from an opaque network",
				Summary: "Removes a set of tags from the opaque network",
			},
			Key: "OpaqueNetwork.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "OpaqueNetwork.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove an opaque network",
				Summary: "Removes an opaque network",
			},
			Key: "OpaqueNetwork.destroyNetwork",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set resource pool custom value",
				Summary: "Sets the value of a custom field of a resource pool of physical resources",
			},
			Key: "ResourcePool.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload resource pool",
				Summary: "Reload the resource pool",
			},
			Key: "ResourcePool.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename resource pool",
				Summary: "Rename the resource pool",
			},
			Key: "ResourcePool.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete resource pool",
				Summary: "Delete the resource pool, which also deletes its contents and removes it from its parent folder (if any)",
			},
			Key: "ResourcePool.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the resource pool",
			},
			Key: "ResourcePool.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the resource pool",
			},
			Key: "ResourcePool.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "ResourcePool.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update resource pool configuration",
				Summary: "Updates the resource pool configuration",
			},
			Key: "ResourcePool.updateConfig",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move into resource pool",
				Summary: "Moves a set of resource pools or virtual machines into this pool",
			},
			Key: "ResourcePool.moveInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update child resource configuration",
				Summary: "Change the resource configuration of a set of children of the resource pool",
			},
			Key: "ResourcePool.updateChildResourceConfiguration",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create resource pool",
				Summary: "Creates a new resource pool",
			},
			Key: "ResourcePool.createResourcePool",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete resource pool children",
				Summary: "Removes all child resource pools recursively",
			},
			Key: "ResourcePool.destroyChildren",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create vApp",
				Summary: "Creates a child vApp of this resource pool",
			},
			Key: "ResourcePool.createVApp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine",
				Summary: "Creates a virtual machine in this resource pool",
			},
			Key: "ResourcePool.createVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register virtual machine",
				Summary: "Adds an existing virtual machine to this resource pool",
			},
			Key: "ResourcePool.registerVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy OVF template",
				Summary: "Deploys a virtual machine or vApp",
			},
			Key: "ResourcePool.importVApp",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query resource pool resource configuration options",
				Summary: "Returns configuration options for a set of resources for a resource pool",
			},
			Key: "ResourcePool.queryResourceConfigOption",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh resource runtime information",
				Summary: "Refreshes the resource usage runtime information",
			},
			Key: "ResourcePool.refreshRuntime",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set virtual machine custom value",
				Summary: "Sets the value of a custom field of a virtual machine",
			},
			Key: "VirtualMachine.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload virtual machine",
				Summary: "Reloads the virtual machine",
			},
			Key: "VirtualMachine.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename virtual machine",
				Summary: "Rename the virtual machine",
			},
			Key: "VirtualMachine.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete virtual machine",
				Summary: "Delete this virtual machine. Deleting this virtual machine also deletes its contents and removes it from its parent folder (if any).",
			},
			Key: "VirtualMachine.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add Tag",
				Summary: "Add a set of tags to the virtual machine",
			},
			Key: "VirtualMachine.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the virtual machine",
			},
			Key: "VirtualMachine.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "VirtualMachine.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Refresh virtual machine storage information",
				Summary: "Refresh storage information for the virtual machine",
			},
			Key: "VirtualMachine.refreshStorageInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve virtual machine backup agent",
				Summary: "Retrieves the backup agent for the virtual machine",
			},
			Key: "VirtualMachine.retrieveBackupAgent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine snapshot",
				Summary: "Create a new snapshot of this virtual machine",
			},
			Key: "VirtualMachine.createSnapshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine snapshot",
				Summary: "Create a new snapshot of this virtual machine",
			},
			Key: "VirtualMachine.createSnapshotEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Revert to current snapshot",
				Summary: "Reverts the virtual machine to the current snapshot",
			},
			Key: "VirtualMachine.revertToCurrentSnapshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove all snapshots",
				Summary: "Remove all the snapshots associated with this virtual machine",
			},
			Key: "VirtualMachine.removeAllSnapshots",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Consolidate virtual machine disk files",
				Summary: "Consolidate disk files of this virtual machine",
			},
			Key: "VirtualMachine.consolidateDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Estimate virtual machine disks consolidation space requirement",
				Summary: "Estimate the temporary space required to consolidate disk files.",
			},
			Key: "VirtualMachine.estimateStorageRequirementForConsolidate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure virtual machine",
				Summary: "Reconfigure this virtual machine",
			},
			Key: "VirtualMachine.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade VM compatibility",
				Summary: "Upgrade virtual machine compatibility to the latest version",
			},
			Key: "VirtualMachine.upgradeVirtualHardware",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Extract OVF environment",
				Summary: "Returns the XML document that represents the OVF environment",
			},
			Key: "VirtualMachine.extractOvfEnvironment",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Power On virtual machine",
				Summary: "Power On this virtual machine",
			},
			Key: "VirtualMachine.powerOn",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Power Off virtual machine",
				Summary: "Power Off this virtual machine",
			},
			Key: "VirtualMachine.powerOff",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Suspend virtual machine",
				Summary: "Suspend virtual machine",
			},
			Key: "VirtualMachine.suspend",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset virtual machine",
				Summary: "Reset this virtual machine",
			},
			Key: "VirtualMachine.reset",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate guest OS shutdown",
				Summary: "Issues a command to the guest operating system to perform a clean shutdown of all services",
			},
			Key: "VirtualMachine.shutdownGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate guest OS reboot",
				Summary: "Issues a command to the guest operating system asking it to perform a reboot",
			},
			Key: "VirtualMachine.rebootGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiate guest OS standby",
				Summary: "Issues a command to the guest operating system to prepare for a suspend operation",
			},
			Key: "VirtualMachine.standbyGuest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Answer virtual machine question",
				Summary: "Respond to a question that is blocking this virtual machine",
			},
			Key: "VirtualMachine.answer",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Customize virtual machine guest OS",
				Summary: "Customize a virtual machine's guest operating system",
			},
			Key: "VirtualMachine.customize",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check customization specification",
				Summary: "Check the customization specification against the virtual machine configuration",
			},
			Key: "VirtualMachine.checkCustomizationSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Migrate virtual machine",
				Summary: "Migrate a virtual machine's execution to a specific resource pool or host",
			},
			Key: "VirtualMachine.migrate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Relocate virtual machine",
				Summary: "Relocate the virtual machine to a specific location",
			},
			Key: "VirtualMachine.relocate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone virtual machine",
				Summary: "Creates a clone of this virtual machine",
			},
			Key: "VirtualMachine.clone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "instantClone",
				Summary: "instantClone",
			},
			Key: "VirtualMachine.instantClone",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveInstantCloneChildren",
				Summary: "retrieveInstantCloneChildren",
			},
			Key: "VirtualMachine.retrieveInstantCloneChildren",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveInstantCloneParent",
				Summary: "retrieveInstantCloneParent",
			},
			Key: "VirtualMachine.retrieveInstantCloneParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "markAsInstantCloneParent",
				Summary: "markAsInstantCloneParent",
			},
			Key: "VirtualMachine.markAsInstantCloneParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "unmarkAsInstantCloneParent",
				Summary: "unmarkAsInstantCloneParent",
			},
			Key: "VirtualMachine.unmarkAsInstantCloneParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "createForkChild",
				Summary: "createForkChild",
			},
			Key: "VirtualMachine.createForkChild",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "enableForkParent",
				Summary: "enableForkParent",
			},
			Key: "VirtualMachine.enableForkParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "disableForkParent",
				Summary: "disableForkParent",
			},
			Key: "VirtualMachine.disableForkParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveForkChildren",
				Summary: "retrieveForkChildren",
			},
			Key: "VirtualMachine.retrieveForkChildren",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveForkParent",
				Summary: "retrieveForkParent",
			},
			Key: "VirtualMachine.retrieveForkParent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF template",
				Summary: "Exports the virtual machine as an OVF template",
			},
			Key: "VirtualMachine.exportVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark virtual machine as template",
				Summary: "Virtual machine is marked as a template",
			},
			Key: "VirtualMachine.markAsTemplate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mark as virtual machine",
				Summary: "Reassociate a virtual machine with a host or resource pool",
			},
			Key: "VirtualMachine.markAsVirtualMachine",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister virtual machine",
				Summary: "Removes this virtual machine from the inventory without removing any of the virtual machine files on disk",
			},
			Key: "VirtualMachine.unregister",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reset guest OS information",
				Summary: "Clears cached guest OS information",
			},
			Key: "VirtualMachine.resetGuestInformation",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiated VMware Tools Installer Mount",
				Summary: "Mounts the tools CD installer as a CD-ROM for the guest",
			},
			Key: "VirtualMachine.mountToolsInstaller",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Connect VMware Tools CD",
				Summary: "Connects the VMware Tools CD image to the guest",
			},
			Key: "VirtualMachine.mountToolsInstallerImage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmount tools installer",
				Summary: "Unmounts the tools installer",
			},
			Key: "VirtualMachine.unmountToolsInstaller",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiated VMware Tools install or upgrade",
				Summary: "Issues a command to the guest operating system to install VMware Tools or upgrade to the latest revision",
			},
			Key: "VirtualMachine.upgradeTools",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initiated VMware Tools upgrade",
				Summary: "Upgrades VMware Tools in the virtual machine from specified CD image",
			},
			Key: "VirtualMachine.upgradeToolsFromImage",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire virtual machine Mouse Keyboard Screen Ticket",
				Summary: "Establishing a Mouse Keyboard Screen Ticket",
			},
			Key: "VirtualMachine.acquireMksTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Acquire virtual machine service ticket",
				Summary: "Establishing a specific remote virtual machine connection ticket",
			},
			Key: "VirtualMachine.acquireTicket",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set console window screen resolution",
				Summary: "Sets the console window's resolution as specified",
			},
			Key: "VirtualMachine.setScreenResolution",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Defragment all disks",
				Summary: "Defragment all virtual disks attached to this virtual machine",
			},
			Key: "VirtualMachine.defragmentAllDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Turn On Fault Tolerance",
				Summary: "Secondary VM created",
			},
			Key: "VirtualMachine.createSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Turn On Fault Tolerance",
				Summary: "Creates a secondary VM",
			},
			Key: "VirtualMachine.createSecondaryEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Turn Off Fault Tolerance",
				Summary: "Remove all secondaries for this virtual machine and turn off Fault Tolerance",
			},
			Key: "VirtualMachine.turnOffFaultTolerance",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Test failover",
				Summary: "Test Fault Tolerance failover by making a Secondary VM in a Fault Tolerance pair the Primary VM",
			},
			Key: "VirtualMachine.makePrimary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Test restarting Secondary VM",
				Summary: "Test restart Secondary VM by stopping a Secondary VM in the Fault Tolerance pair",
			},
			Key: "VirtualMachine.terminateFaultTolerantVM",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Suspend Fault Tolerance",
				Summary: "Suspend Fault Tolerance on this virtual machine",
			},
			Key: "VirtualMachine.disableSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Resume Fault Tolerance",
				Summary: "Resume Fault Tolerance on this virtual machine",
			},
			Key: "VirtualMachine.enableSecondary",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set virtual machine display topology",
				Summary: "Set the display topology for the virtual machine",
			},
			Key: "VirtualMachine.setDisplayTopology",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start recording",
				Summary: "Start a recording session on this virtual machine",
			},
			Key: "VirtualMachine.startRecording",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop recording",
				Summary: "Stop a currently active recording session on this virtual machine",
			},
			Key: "VirtualMachine.stopRecording",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start replaying",
				Summary: "Start a replay session on this virtual machine",
			},
			Key: "VirtualMachine.startReplaying",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop replaying",
				Summary: "Stop a replay session on this virtual machine",
			},
			Key: "VirtualMachine.stopReplaying",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Promote virtual machine disks",
				Summary: "Promote disks of the virtual machine that have delta disk backings",
			},
			Key: "VirtualMachine.promoteDisks",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Take virtual machine screenshot",
				Summary: "Take a screenshot of a virtual machine's guest OS console",
			},
			Key: "VirtualMachine.createScreenshot",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Put USB HID scan codes",
				Summary: "Injects a sequence of USB HID scan codes into the keyboard",
			},
			Key: "VirtualMachine.putUsbScanCodes",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query virtual machine disk changes",
				Summary: "Query for changes to the virtual machine's disks since a given point in the past",
			},
			Key: "VirtualMachine.queryChangedDiskAreas",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query unowned virtual machine files",
				Summary: "Query files of the virtual machine not owned by the datastore principal user",
			},
			Key: "VirtualMachine.queryUnownedFiles",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload virtual machine from new configuration",
				Summary: "Reloads the virtual machine from a new configuration file",
			},
			Key: "VirtualMachine.reloadFromPath",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query Virtual Machine Fault Tolerance Compatibility",
				Summary: "Check if virtual machine is compatible for Fault Tolerance",
			},
			Key: "VirtualMachine.queryFaultToleranceCompatibility",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryFaultToleranceCompatibilityEx",
				Summary: "queryFaultToleranceCompatibilityEx",
			},
			Key: "VirtualMachine.queryFaultToleranceCompatibilityEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Suspend and resume the virtual machine",
				Summary: "Suspend and resume the virtual machine",
			},
			Key: "VirtualMachine.invokeFSR",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Hard stop virtual machine",
				Summary: "Hard stop virtual machine",
			},
			Key: "VirtualMachine.terminate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get native clone capability",
				Summary: "Check if native clone is supported on the virtual machine",
			},
			Key: "VirtualMachine.isNativeSnapshotCapable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure quorum file path prefix",
				Summary: "Configures the quorum file path prefix for the virtual machine",
			},
			Key: "VirtualMachine.configureQuorumFilePathPrefix",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Retrieve quorum file path prefix",
				Summary: "Retrieves the quorum file path prefix for the virtual machine",
			},
			Key: "VirtualMachine.retrieveQuorumFilePathPrefix",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Inject OVF Environment into virtual machine",
				Summary: "Specifies the OVF Environments to be injected into and returned for a virtual machine",
			},
			Key: "VirtualMachine.injectOvfEnvironment",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Wipe a Flex-SE virtual disk",
				Summary: "Wipes a Flex-SE virtual disk",
			},
			Key: "VirtualMachine.wipeDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Shrink a Flex-SE virtual disk",
				Summary: "Shrinks a Flex-SE virtual disk",
			},
			Key: "VirtualMachine.shrinkDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Send NMI",
				Summary: "Sends a non-maskable interrupt (NMI) to the virtual machine",
			},
			Key: "VirtualMachine.sendNMI",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload virtual machine",
				Summary: "Reloads the virtual machine",
			},
			Key: "VirtualMachine.reloadEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach a virtual disk",
				Summary: "Attach an existing virtual disk to the virtual machine",
			},
			Key: "VirtualMachine.attachDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detach a virtual disk",
				Summary: "Detach a virtual disk from the virtual machine",
			},
			Key: "VirtualMachine.detachDisk",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply EVC Mode",
				Summary: "Apply EVC Mode to a virtual machine",
			},
			Key: "VirtualMachine.applyEvcMode",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set datacenter custom value",
				Summary: "Sets the value of a custom field of a datacenter",
			},
			Key: "Datacenter.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload datacenter",
				Summary: "Reloads the datacenter",
			},
			Key: "Datacenter.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename datacenter",
				Summary: "Rename the datacenter",
			},
			Key: "Datacenter.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove datacenter",
				Summary: "Deletes the datacenter and removes it from its parent folder (if any)",
			},
			Key: "Datacenter.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the datacenter",
			},
			Key: "Datacenter.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the datacenter",
			},
			Key: "Datacenter.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "Datacenter.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query connection information",
				Summary: "Gets information of a host that can be used in the connection wizard",
			},
			Key: "Datacenter.queryConnectionInfo",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "queryConnectionInfoViaSpec",
				Summary: "queryConnectionInfoViaSpec",
			},
			Key: "Datacenter.queryConnectionInfoViaSpec",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Initialize powering On",
				Summary: "Initialize tasks for powering on virtual machines",
			},
			Key: "Datacenter.powerOnVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Query configuration option descriptor",
				Summary: "Retrieve the list of configuration option keys available in this datacenter",
			},
			Key: "Datacenter.queryConfigOptionDescriptor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure datacenter",
				Summary: "Reconfigures the datacenter",
			},
			Key: "Datacenter.reconfigure",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set folder custom value",
				Summary: "Sets the value of a custom field of a folder",
			},
			Key: "Folder.setCustomValue",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reload folder",
				Summary: "Reloads the folder",
			},
			Key: "Folder.reload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rename folder",
				Summary: "Rename the folder",
			},
			Key: "Folder.rename",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete folder",
				Summary: "Delete this object, deleting its contents and removing it from its parent folder (if any)",
			},
			Key: "Folder.destroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add tag",
				Summary: "Add a set of tags to the folder",
			},
			Key: "Folder.addTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove tag",
				Summary: "Remove a set of tags from the folder",
			},
			Key: "Folder.removeTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "retrieveCustomValues",
				Summary: "retrieveCustomValues",
			},
			Key: "Folder.retrieveCustomValues",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create folder",
				Summary: "Creates a new folder",
			},
			Key: "Folder.createFolder",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Move entities",
				Summary: "Moves a set of managed entities into this folder",
			},
			Key: "Folder.moveInto",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create virtual machine",
				Summary: "Create a new virtual machine",
			},
			Key: "Folder.createVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Register virtual machine",
				Summary: "Adds an existing virtual machine to the folder",
			},
			Key: "Folder.registerVm",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create cluster",
				Summary: "Create a new cluster compute-resource in this folder",
			},
			Key: "Folder.createCluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create cluster",
				Summary: "Create a new cluster compute-resource in this folder",
			},
			Key: "Folder.createClusterEx",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add standalone host",
				Summary: "Create a new single-host compute-resource",
			},
			Key: "Folder.addStandaloneHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add standalone host and enable lockdown",
				Summary: "Create a new single-host compute-resource and enable lockdown mode on the host",
			},
			Key: "Folder.addStandaloneHostWithAdminDisabled",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create datacenter",
				Summary: "Create a new datacenter with the given name",
			},
			Key: "Folder.createDatacenter",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unregister and Delete",
				Summary: "Recursively deletes all child virtual machine folders and unregisters all virtual machines",
			},
			Key: "Folder.unregisterAndDestroy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a vSphere Distributed Switch",
				Summary: "Create a vSphere Distributed Switch",
			},
			Key: "Folder.createDistributedVirtualSwitch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create a datastore cluster",
				Summary: "Create a datastore cluster",
			},
			Key: "Folder.createStoragePod",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Get boot devices",
				Summary: "Get available boot devices for the host system",
			},
			Key: "host.BootDeviceSystem.queryBootDevices",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update boot device",
				Summary: "Update the boot device on the host system",
			},
			Key: "host.BootDeviceSystem.updateBootDevice",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configuring vSphere HA",
				Summary: "Configuring vSphere HA",
			},
			Key: "DasConfig.ConfigureHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unconfiguring vSphere HA",
				Summary: "Unconfiguring vSphere HA",
			},
			Key: "DasConfig.UnconfigureHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Migrate virtual machine",
				Summary: "Migrates a virtual machine from one host to another",
			},
			Key: "Drm.ExecuteVMotionLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Power On virtual machine",
				Summary: "Power on this virtual machine",
			},
			Key: "Drm.ExecuteVmPowerOnLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enter standby mode",
				Summary: "Puts this host into standby mode",
			},
			Key: "Drm.EnterStandbyLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Exit standby mode",
				Summary: "Brings this host out of standby mode",
			},
			Key: "Drm.ExitStandbyLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Power On virtual machine",
				Summary: "Power On this virtual machine",
			},
			Key: "Datacenter.ExecuteVmPowerOnLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade vCenter Agent",
				Summary: "Upgrade the vCenter Agent",
			},
			Key: "Upgrade.UpgradeAgent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade vCenter Agents on cluster hosts",
				Summary: "Upgrade the vCenter Agents on all cluster hosts",
			},
			Key: "ClusterUpgrade.UpgradeAgent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy OVF template",
				Summary: "Deploys a virtual machine or vApp",
			},
			Key: "ResourcePool.ImportVAppLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set cluster suspended state",
				Summary: "Set suspended state of the cluster",
			},
			Key: "ClusterComputeResource.setSuspendedState",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF template",
				Summary: "Exports the virtual machine as an OVF template",
			},
			Key: "VirtualMachine.ExportVmLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF template",
				Summary: "Exports the vApp as an OVF template",
			},
			Key: "VirtualApp.ExportVAppLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Start Fault Tolerance Secondary VM",
				Summary: "Start Secondary VM as the Primary VM is powered on",
			},
			Key: "FaultTolerance.PowerOnSecondaryLRO",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Execute Storage vMotion for Storage DRS",
				Summary: "Execute Storage vMotion migrations for Storage DRS",
			},
			Key: "Drm.ExecuteStorageVmotionLro",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply recommendations for SDRS maintenance mode",
				Summary: "Apply recommendations to enter into SDRS maintenance mode",
			},
			Key: "Drm.ExecuteMaintenanceRecommendationsLro",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enter SDRS maintenance mode monitor task",
				Summary: "Task that monitors the SDRS maintenance mode activity",
			},
			Key: "Drm.TrackEnterMaintenanceLro",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "ResetSensor",
				Summary: "ResetSensor",
			},
			Key: "com.vmware.hardwarehealth.ResetSensor",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "ResetSelLog",
				Summary: "ResetSelLog",
			},
			Key: "com.vmware.hardwarehealth.ResetSelLog",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "RefreshHost",
				Summary: "RefreshHost",
			},
			Key: "com.vmware.hardwarehealth.RefreshHost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "install",
				Summary: "install",
			},
			Key: "eam.agent.install",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "uninstall",
				Summary: "uninstall",
			},
			Key: "eam.agent.uninstall",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "upgrade",
				Summary: "upgrade",
			},
			Key: "eam.agent.upgrade",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Bulk Remediation",
				Summary: "Remediating hosts in bulk",
			},
			Key: "com.vmware.rbd.bulkRemediateMapping",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Rule",
				Summary: "Creating rule in Auto Deploy server",
			},
			Key: "com.vmware.rbd.CreateRuleWithTransform",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Apply Image Profile",
				Summary: "Applying Image profile to a host",
			},
			Key: "com.vmware.rbd.ApplyImageProfile",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Edit Rule",
				Summary: "Editing Auto Deploy rule",
			},
			Key: "com.vmware.rbd.UpdateSpecWithTransform",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Repair Cache",
				Summary: "Repairing Deploy cache in Auto Deploy server",
			},
			Key: "com.vmware.rbd.RepairDeployCache",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Bulk Compliance Check",
				Summary: "Compliance checking hosts in bulk",
			},
			Key: "com.vmware.rbd.bulkMappingComplianceCheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Mount an ISO Library Item as a Virtual CD-ROM",
				Summary: "Mount",
			},
			Key: "com.vmware.vcenter.iso.Image.Mount",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Unmount a Virtual CD-ROM mounted with ISO backing",
				Summary: "Unmount",
			},
			Key: "com.vmware.vcenter.iso.Image.Unmount",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import OVF package",
				Summary: "Create",
			},
			Key: "com.vmware.ovfs.ImportSession.Create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Export OVF package",
				Summary: "Create",
			},
			Key: "com.vmware.ovfs.ExportSession.Create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Deploy OVF package from Content Library to Resource Pool",
				Summary: "instantiate",
			},
			Key: "com.vmware.ovfs.LibraryItem.instantiate",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Clone to OVF package in Content Library from Virtual Machine or Virtual Appliance",
				Summary: "capture",
			},
			Key: "com.vmware.ovfs.LibraryItem.capture",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Parse OVF package in Content Library",
				Summary: "parse",
			},
			Key: "com.vmware.ovfs.LibraryItem.parse",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Scrub Database after Restore",
				Summary: "Scrub",
			},
			Key: "com.vmware.content.Scrub",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Library",
				Summary: "Create",
			},
			Key: "com.vmware.content.Library.Create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Library",
				Summary: "Update",
			},
			Key: "com.vmware.content.Library.Update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete Library",
				Summary: "Delete",
			},
			Key: "com.vmware.content.Library.Delete",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete Library Content",
				Summary: "DeleteContent",
			},
			Key: "com.vmware.content.Library.DeleteContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Sync Library",
				Summary: "Sync",
			},
			Key: "com.vmware.content.Library.Sync",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Validate Library Content against the Storage Backing After Restore",
				Summary: "Scrub",
			},
			Key: "com.vmware.content.Library.Scrub",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Library Item",
				Summary: "Create",
			},
			Key: "com.vmware.content.LibraryItem.Create",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Library Item",
				Summary: "Update",
			},
			Key: "com.vmware.content.LibraryItem.Update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update Library Item Backing",
				Summary: "UpdateBackings",
			},
			Key: "com.vmware.content.LibraryItem.UpdateBackings",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete Library Item",
				Summary: "Delete",
			},
			Key: "com.vmware.content.LibraryItem.Delete",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Delete Library Item Content",
				Summary: "DeleteContent",
			},
			Key: "com.vmware.content.LibraryItem.DeleteContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "DeleteFileContent",
				Summary: "DeleteFileContent",
			},
			Key: "com.vmware.content.LibraryItem.DeleteFileContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upload Files to a Library Item",
				Summary: "UploadContent",
			},
			Key: "com.vmware.content.LibraryItem.UploadContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fetch Content of a Library Item",
				Summary: "FetchContent",
			},
			Key: "com.vmware.content.LibraryItem.FetchContent",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Copy Library Item",
				Summary: "Copy",
			},
			Key: "com.vmware.content.LibraryItem.Copy",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Sync Library Item",
				Summary: "Sync",
			},
			Key: "com.vmware.content.LibraryItem.Sync",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Waiting For Upload",
				Summary: "WaitForUpload",
			},
			Key: "com.vmware.content.LibraryItem.WaitForUpload",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Setting Library Item Tag",
				Summary: "SetTag",
			},
			Key: "com.vmware.content.LibraryItem.SetTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Removing Library Item Tag",
				Summary: "RemoveTag",
			},
			Key: "com.vmware.content.LibraryItem.RemoveTag",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install vSAN iSCSI target service",
				Summary: "Install vSAN iSCSI target service",
			},
			Key: "com.vmware.vsan.iscsi.tasks.installVibTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create Home Object and set vSAN iSCSI target service",
				Summary: "Create Home Object and set vSAN iSCSI target service",
			},
			Key: "com.vmware.vsan.iscsi.tasks.settingTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable vSAN iSCSI target service in cluster",
				Summary: "Enable vSAN iSCSI target service in cluster",
			},
			Key: "com.vmware.vsan.iscsi.tasks.enable",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Edit vSAN iSCSI target service in cluster",
				Summary: "Edit vSAN iSCSI target service in cluster",
			},
			Key: "com.vmware.vsan.iscsi.tasks.edit",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add a new iSCSI target",
				Summary: "Add a new iSCSI target",
			},
			Key: "com.vmware.vsan.iscsi.tasks.addTarget",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Edit the iSCSI target",
				Summary: "Edit the iSCSI target",
			},
			Key: "com.vmware.vsan.iscsi.tasks.editTarget",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove the iSCSI target",
				Summary: "Remove the iSCSI target",
			},
			Key: "com.vmware.vsan.iscsi.tasks.removeTarget",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add a new iSCSI LUN",
				Summary: "Add a new iSCSI LUN",
			},
			Key: "com.vmware.vsan.iscsi.tasks.addLUN",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Edit the iSCSI LUN",
				Summary: "Edit the iSCSI LUN",
			},
			Key: "com.vmware.vsan.iscsi.tasks.editLUN",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove the iSCSI LUN",
				Summary: "Remove the iSCSI LUN",
			},
			Key: "com.vmware.vsan.iscsi.tasks.removeLUN",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "VMDK Load Test",
				Summary: "VMDK Load Test",
			},
			Key: "com.vmware.vsan.health.tasks.runvmdkloadtest",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install vSAN health ESX extension",
				Summary: "Install vSAN health ESX extension",
			},
			Key: "com.vmware.vsan.health.tasks.health.preparecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Uninstall vSAN health ESX extension",
				Summary: "Uninstall vSAN health ESX extension",
			},
			Key: "com.vmware.vsan.health.tasks.health.uninstallcluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Install vSAN sizing ESX extension",
				Summary: "Install vSAN sizing ESX extension",
			},
			Key: "com.vmware.vsan.health.tasks.sizing.preparecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Uninstall vSAN sizing ESX extension",
				Summary: "Uninstall vSAN sizing ESX extension",
			},
			Key: "com.vmware.vsan.health.tasks.sizing.uninstallcluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "preparecluster",
				Summary: "preparecluster",
			},
			Key: "com.vmware.vsan.health.tasks.perfsvc.preparecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "uninstallcluster",
				Summary: "uninstallcluster",
			},
			Key: "com.vmware.vsan.health.tasks.perfsvc.uninstallcluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Fix vSAN Cluster Object Immediately",
				Summary: "Fix vSAN Cluster Object Immediately",
			},
			Key: "com.vmware.vsan.health.tasks.repairclusterobjects",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Rebalance vSAN Cluster",
				Summary: "Rebalance vSAN Cluster",
			},
			Key: "com.vmware.vsan.health.tasks.rebalancecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stop Rebalance vSAN Cluster",
				Summary: "Stop Rebalance vSAN Cluster",
			},
			Key: "com.vmware.vsan.health.tasks.stoprebalancecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upgrade vSAN disk format",
				Summary: "Upgrade vSAN disk format",
			},
			Key: "com.vmware.vsan.health.tasks.upgrade",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach vSAN support bundle to SR",
				Summary: "Attach vSAN support bundle to SR",
			},
			Key: "com.vmware.vsan.health.tasks.attachtosr",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Attach vSAN support bundle to PR",
				Summary: "Attach vSAN support bundle to PR",
			},
			Key: "com.vmware.vsan.health.tasks.attachtopr",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Download file from URL",
				Summary: "Download file from URL",
			},
			Key: "com.vmware.vsan.health.tasks.downloadfromurl",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Online check of vSAN health",
				Summary: "Online check of vSAN health",
			},
			Key: "com.vmware.vsan.health.tasks.performonlinehealthcheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remediate vSAN cluster",
				Summary: "Remediate vSAN cluster",
			},
			Key: "com.vmware.vsan.clustermgmt.tasks.remediatevsancluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remediate vSAN configurations",
				Summary: "Remediate vSAN configurations",
			},
			Key: "com.vmware.vsan.clustermgmt.tasks.remediatevc",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Enable vSAN performance service",
				Summary: "Enable vSAN performance service",
			},
			Key: "com.vmware.vsan.perfsvc.tasks.createstatsdb",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Disable vSAN performance service",
				Summary: "Disable vSAN performance service",
			},
			Key: "com.vmware.vsan.perfsvc.tasks.deletestatsdb",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Gathering data for performance diagnosis",
				Summary: "Gathering data for performance diagnosis",
			},
			Key: "com.vmware.vsan.perfsvc.tasks.runqueryfordiagnosis",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSAN: Update Software/Driver/Firmware",
				Summary: "vSAN: Update Software/Driver/Firmware",
			},
			Key: "com.vmware.vsan.patch.tasks.patch",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSAN: Migrate VSS to VDS",
				Summary: "vSAN: Migrate VSS to VDS",
			},
			Key: "com.vmware.vsan.vds.tasks.migratevss",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Create disk group on vSAN",
				Summary: "Create disk group on vSAN",
			},
			Key: "com.vmware.vsan.diskmgmt.tasks.initializediskmappings",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Add witness host",
				Summary: "Add witness host to a stretched cluster",
			},
			Key: "com.vmware.vsan.stretchedcluster.tasks.addwitnesshost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Replace witness host",
				Summary: "Replace witness host for a stretched cluster",
			},
			Key: "com.vmware.vsan.stretchedcluster.tasks.replacewitnesshost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remove witness host",
				Summary: "Remove witness host from a stretched cluster",
			},
			Key: "com.vmware.vsan.stretchedcluster.tasks.removewitnesshost",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Convert to a stretched cluster",
				Summary: "Convert the given configuration to a streched cluster",
			},
			Key: "com.vmware.vsan.stretchedcluster.tasks.convert2stretchedcluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Set preferred fault domain",
				Summary: "Set preferred fault domain for a stretched cluster",
			},
			Key: "com.vmware.vsan.stretchedcluster.tasks.setpreferredfaultdomain",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Convert disk format for vSAN",
				Summary: "Convert disk format for vSAN",
			},
			Key: "com.vmware.vsan.diskconvertion.tasks.conversion",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Reconfigure vSAN cluster",
				Summary: "Reconfigure vSAN cluster",
			},
			Key: "com.vmware.vsan.clustermgmt.tasks.reconfigurecluster",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Regenerate new keys for encrypted vSAN cluster",
				Summary: "Regenerate new keys for encrypted vSAN cluster",
			},
			Key: "com.vmware.vsan.clustermgmt.tasks.rekey",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "vSAN operation precheck",
				Summary: "vSAN operation precheck",
			},
			Key: "com.vmware.vsan.clustermgmt.tasks.precheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Update vSAN configuration",
				Summary: "Updates the vSAN configuration for this host",
			},
			Key: "com.vmware.vsan.vsansystem.tasks.update",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Scan vSAN Objects",
				Summary: "Scan vSAN Objects for issues",
			},
			Key: "com.vmware.vsan.diskmgmt.tasks.objectscan",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Perform Convert disk format precheck",
				Summary: "Perform Convert disk format precheck for issues that could be encountered",
			},
			Key: "com.vmware.vsan.diskconvertion.tasks.precheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Perform compliance resource check task",
				Summary: "Perform compliance resource check task",
			},
			Key: "com.vmware.vsan.prechecker.tasks.complianceresourcecheck",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Download patch definitions",
				Summary: "Download patch definitions",
			},
			Key: "com.vmware.vcIntegrity.SigUpdateTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Check new notifications",
				Summary: "Check new notifications",
			},
			Key: "com.vmware.vcIntegrity.CheckNotificationTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Scan entity",
				Summary: "Scan an entity",
			},
			Key: "com.vmware.vcIntegrity.ScanTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remediate entity",
				Summary: "Remediate an entity",
			},
			Key: "com.vmware.vcIntegrity.RemediateTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Stage patches to entity",
				Summary: "Stage patches to an entity",
			},
			Key: "com.vmware.vcIntegrity.StageTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Discover virtual appliance",
				Summary: "Discover virtual appliance",
			},
			Key: "com.vmware.vcIntegrity.VaDiscoveryTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Detect Update Manager Guest Agent",
				Summary: "Detect Update Manager Guest Agent installataion on Linux VMs",
			},
			Key: "com.vmware.vcIntegrity.DetectLinuxGATask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel detecting Update Manager GuestAgent",
				Summary: "Cancel detecting Update Manager GuestAgent installataion on Linux VMs",
			},
			Key: "com.vmware.vcIntegrity.CancelDetectLinuxGATask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel download of patch definitions",
				Summary: "Cancel download of patch definitions",
			},
			Key: "com.vmware.vcIntegrity.CancelSigUpdateTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel scanning entity",
				Summary: "Cancel scanning an entity",
			},
			Key: "com.vmware.vcIntegrity.CancelScanTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel remediating entity",
				Summary: "Cancel remediating an entity",
			},
			Key: "com.vmware.vcIntegrity.CancelRemediateTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel discovering virtual appliance",
				Summary: "Cancel discovering a virtual appliance",
			},
			Key: "com.vmware.vcIntegrity.CancelVaDiscoveryTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Configure VMware Tools upgrade setting",
				Summary: "Configure VMware Tools upgrade setting",
			},
			Key: "com.vmware.vcIntegrity.ConfigureToolsUpgradeTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Import ESXi image",
				Summary: "Import ESXi image",
			},
			Key: "com.vmware.vcIntegrity.ImportRelease",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Upload offline patches",
				Summary: "Upload offline patches",
			},
			Key: "com.vmware.vcIntegrity.DownloadOfflinePatchTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Confirm importing offline patches",
				Summary: "Confirm importing offline host patches",
			},
			Key: "com.vmware.vcIntegrity.ConfirmOfflinePatchTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Cancel importing offline patches",
				Summary: "Cancel importing offline host patches",
			},
			Key: "com.vmware.vcIntegrity.CancelOfflinePatchTask",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Remediation pre-check",
				Summary: "Remediation pre-check",
			},
			Key: "com.vmware.vcIntegrity.RemediatePrecheckTask",
		},
	},
	State: []types.BaseElementDescription{
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Queued",
				Summary: "Task is queued",
			},
			Key: "queued",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Running",
				Summary: "Task is in progress",
			},
			Key: "running",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Success",
				Summary: "Task completed successfully",
			},
			Key: "success",
		},
		&types.ElementDescription{
			Description: types.Description{
				Label:   "Error",
				Summary: "Task completed with a failure",
			},
			Key: "error",
		},
	},
	Reason: []types.BaseTypeDescription{
		&types.TypeDescription{
			Description: types.Description{
				Label:   "Scheduled task",
				Summary: "Task started by a scheduled task",
			},
			Key: "TaskReasonSchedule",
		},
		&types.TypeDescription{
			Description: types.Description{
				Label:   "User task",
				Summary: "Task started by a specific user",
			},
			Key: "TaskReasonUser",
		},
		&types.TypeDescription{
			Description: types.Description{
				Label:   "System task",
				Summary: "Task started by the server",
			},
			Key: "TaskReasonSystem",
		},
		&types.TypeDescription{
			Description: types.Description{
				Label:   "Alarm task",
				Summary: "Task started by an alarm",
			},
			Key: "TaskReasonAlarm",
		},
	},
}
