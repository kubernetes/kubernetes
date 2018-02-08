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

package vpx

import "github.com/vmware/govmomi/vim25/types"

// Setting is captured from VC's ServiceContent.OptionManager.setting
var Setting = []types.BaseOptionValue{
	// This list is currently pruned to include sso options only with sso.enabled set to false
	&types.OptionValue{
		Key:   "config.vpxd.sso.sts.uri",
		Value: "https://127.0.0.1/sts/STSService/vsphere.local",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.solutionUser.privateKey",
		Value: "/etc/vmware-vpx/ssl/vcsoluser.key",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.solutionUser.name",
		Value: "vpxd-b643d01c-928f-469b-96a5-d571d762a78e@vsphere.local",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.solutionUser.certificate",
		Value: "/etc/vmware-vpx/ssl/vcsoluser.crt",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.groupcheck.uri",
		Value: "https://127.0.0.1/sso-adminserver/sdk/vsphere.local",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.enabled",
		Value: "false",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.default.isGroup",
		Value: "false",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.default.admin",
		Value: "Administrator@vsphere.local",
	},
	&types.OptionValue{
		Key:   "config.vpxd.sso.admin.uri",
		Value: "https://127.0.0.1/sso-adminserver/sdk/vsphere.local",
	},
}
