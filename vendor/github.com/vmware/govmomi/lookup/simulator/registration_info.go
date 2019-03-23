/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"github.com/google/uuid"
	"github.com/vmware/govmomi/lookup"
	"github.com/vmware/govmomi/lookup/types"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/vim25"
)

var (
	siteID = "vcsim"
)

// registrationInfo returns a ServiceRegistration populated with vcsim's OptionManager settings.
// The complete list can be captured using: govc sso.service.ls -dump
func registrationInfo() []types.LookupServiceRegistrationInfo {
	vc := simulator.Map.Get(vim25.ServiceInstance).(*simulator.ServiceInstance)
	setting := simulator.Map.OptionManager().Setting
	opts := make(map[string]string, len(setting))

	for _, o := range setting {
		opt := o.GetOptionValue()
		if val, ok := opt.Value.(string); ok {
			opts[opt.Key] = val
		}
	}

	trust := []string{opts["vcsim.server.cert"]}
	sdk := opts["vcsim.server.url"] + vim25.Path
	admin := opts["config.vpxd.sso.default.admin"]
	owner := opts["config.vpxd.sso.solutionUser.name"]
	instance := opts["VirtualCenter.InstanceName"]

	// Real PSC has 30+ services by default, we just provide a few that are useful for vmomi interaction..
	return []types.LookupServiceRegistrationInfo{
		{
			LookupServiceRegistrationCommonServiceInfo: types.LookupServiceRegistrationCommonServiceInfo{
				LookupServiceRegistrationMutableServiceInfo: types.LookupServiceRegistrationMutableServiceInfo{
					ServiceVersion: lookup.Version,
					ServiceEndpoints: []types.LookupServiceRegistrationEndpoint{
						{
							Url: opts["config.vpxd.sso.sts.uri"],
							EndpointType: types.LookupServiceRegistrationEndpointType{
								Protocol: "wsTrust",
								Type:     "com.vmware.cis.cs.identity.sso",
							},
							SslTrust: trust,
						},
					},
				},
				OwnerId: admin,
				ServiceType: types.LookupServiceRegistrationServiceType{
					Product: "com.vmware.cis",
					Type:    "sso:sts",
				},
			},
			ServiceId: siteID + ":" + uuid.New().String(),
			SiteId:    siteID,
		},
		{
			LookupServiceRegistrationCommonServiceInfo: types.LookupServiceRegistrationCommonServiceInfo{
				LookupServiceRegistrationMutableServiceInfo: types.LookupServiceRegistrationMutableServiceInfo{
					ServiceVersion: vim25.Version,
					ServiceEndpoints: []types.LookupServiceRegistrationEndpoint{
						{
							Url: sdk,
							EndpointType: types.LookupServiceRegistrationEndpointType{
								Protocol: "vmomi",
								Type:     "com.vmware.vim",
							},
							SslTrust: trust,
							EndpointAttributes: []types.LookupServiceRegistrationAttribute{
								{
									Key:   "cis.common.ep.localurl",
									Value: sdk,
								},
							},
						},
					},
					ServiceAttributes: []types.LookupServiceRegistrationAttribute{
						{
							Key:   "com.vmware.cis.cm.GroupInternalId",
							Value: "com.vmware.vim.vcenter",
						},
						{
							Key:   "com.vmware.vim.vcenter.instanceName",
							Value: instance,
						},
						{
							Key:   "com.vmware.cis.cm.ControlScript",
							Value: "service-control-default-vmon",
						},
						{
							Key:   "com.vmware.cis.cm.HostId",
							Value: uuid.New().String(),
						},
					},
					ServiceNameResourceKey:        "AboutInfo.vpx.name",
					ServiceDescriptionResourceKey: "AboutInfo.vpx.name",
				},
				OwnerId: owner,
				ServiceType: types.LookupServiceRegistrationServiceType{
					Product: "com.vmware.cis",
					Type:    "vcenterserver",
				},
				NodeId: uuid.New().String(),
			},
			ServiceId: vc.Content.About.InstanceUuid,
			SiteId:    siteID,
		},
	}
}
