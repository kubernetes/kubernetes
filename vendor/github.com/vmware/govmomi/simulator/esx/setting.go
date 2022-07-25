/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package esx

import "github.com/vmware/govmomi/vim25/types"

// HardwareVersion is the default VirtualMachine.Config.Version
var HardwareVersion = "vmx-13"

// Setting is captured from ESX's HostSystem.configManager.advancedOption
// Capture method:
//   govc object.collect -s -dump $(govc object.collect -s HostSystem:ha-host configManager.advancedOption) setting
var Setting = []types.BaseOptionValue{
	// This list is currently pruned to include a single option for testing
	&types.OptionValue{
		Key:   "Config.HostAgent.log.level",
		Value: "info",
	},
}
