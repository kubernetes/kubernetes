/*
Copyright (c) 2019 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"sync/atomic"
	"time"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

var DefaultCustomizationSpec = []types.CustomizationSpecItem{
	types.CustomizationSpecItem{
		Info: types.CustomizationSpecInfo{
			Name:           "vcsim-linux",
			Description:    "",
			Type:           "Linux",
			ChangeVersion:  "1569965707",
			LastUpdateTime: types.NewTime(time.Now()),
		},
		Spec: types.CustomizationSpec{
			Options: &types.CustomizationLinuxOptions{},
			Identity: &types.CustomizationLinuxPrep{
				CustomizationIdentitySettings: types.CustomizationIdentitySettings{},
				HostName:                      &types.CustomizationVirtualMachineName{},
				Domain:                        "eng.vmware.com",
				TimeZone:                      "Pacific/Apia",
				HwClockUTC:                    types.NewBool(true),
			},
			GlobalIPSettings: types.CustomizationGlobalIPSettings{
				DnsSuffixList: nil,
				DnsServerList: []string{"127.0.1.1"},
			},
			NicSettingMap: []types.CustomizationAdapterMapping{
				{
					MacAddress: "",
					Adapter: types.CustomizationIPSettings{
						Ip:            &types.CustomizationDhcpIpGenerator{},
						SubnetMask:    "",
						Gateway:       nil,
						IpV6Spec:      (*types.CustomizationIPSettingsIpV6AddressSpec)(nil),
						DnsServerList: nil,
						DnsDomain:     "",
						PrimaryWINS:   "",
						SecondaryWINS: "",
						NetBIOS:       "",
					},
				},
			},
			EncryptionKey: nil,
		},
	},
	types.CustomizationSpecItem{
		Info: types.CustomizationSpecInfo{
			Name:           "vcsim-linux-static",
			Description:    "",
			Type:           "Linux",
			ChangeVersion:  "1569969598",
			LastUpdateTime: types.NewTime(time.Now()),
		},
		Spec: types.CustomizationSpec{
			Options: &types.CustomizationLinuxOptions{},
			Identity: &types.CustomizationLinuxPrep{
				CustomizationIdentitySettings: types.CustomizationIdentitySettings{},
				HostName: &types.CustomizationPrefixName{
					CustomizationName: types.CustomizationName{},
					Base:              "vcsim",
				},
				Domain:     "eng.vmware.com",
				TimeZone:   "Africa/Cairo",
				HwClockUTC: types.NewBool(true),
			},
			GlobalIPSettings: types.CustomizationGlobalIPSettings{
				DnsSuffixList: nil,
				DnsServerList: []string{"127.0.1.1"},
			},
			NicSettingMap: []types.CustomizationAdapterMapping{
				{
					MacAddress: "",
					Adapter: types.CustomizationIPSettings{
						Ip:            &types.CustomizationUnknownIpGenerator{},
						SubnetMask:    "255.255.255.0",
						Gateway:       []string{"10.0.0.1"},
						IpV6Spec:      (*types.CustomizationIPSettingsIpV6AddressSpec)(nil),
						DnsServerList: nil,
						DnsDomain:     "",
						PrimaryWINS:   "",
						SecondaryWINS: "",
						NetBIOS:       "",
					},
				},
			},
			EncryptionKey: nil,
		},
	},
	types.CustomizationSpecItem{
		Info: types.CustomizationSpecInfo{
			Name:           "vcsim-windows-static",
			Description:    "",
			Type:           "Windows",
			ChangeVersion:  "1569978029",
			LastUpdateTime: types.NewTime(time.Now()),
		},
		Spec: types.CustomizationSpec{
			Options: &types.CustomizationWinOptions{
				CustomizationOptions: types.CustomizationOptions{},
				ChangeSID:            true,
				DeleteAccounts:       false,
				Reboot:               "",
			},
			Identity: &types.CustomizationSysprep{
				CustomizationIdentitySettings: types.CustomizationIdentitySettings{},
				GuiUnattended: types.CustomizationGuiUnattended{
					Password:       (*types.CustomizationPassword)(nil),
					TimeZone:       2,
					AutoLogon:      false,
					AutoLogonCount: 1,
				},
				UserData: types.CustomizationUserData{
					FullName:     "vcsim",
					OrgName:      "VMware",
					ComputerName: &types.CustomizationVirtualMachineName{},
					ProductId:    "",
				},
				GuiRunOnce: (*types.CustomizationGuiRunOnce)(nil),
				Identification: types.CustomizationIdentification{
					JoinWorkgroup:       "WORKGROUP",
					JoinDomain:          "",
					DomainAdmin:         "",
					DomainAdminPassword: (*types.CustomizationPassword)(nil),
				},
				LicenseFilePrintData: &types.CustomizationLicenseFilePrintData{
					AutoMode:  "perServer",
					AutoUsers: 5,
				},
			},
			GlobalIPSettings: types.CustomizationGlobalIPSettings{},
			NicSettingMap: []types.CustomizationAdapterMapping{
				{
					MacAddress: "",
					Adapter: types.CustomizationIPSettings{
						Ip:            &types.CustomizationUnknownIpGenerator{},
						SubnetMask:    "255.255.255.0",
						Gateway:       []string{"10.0.0.1"},
						IpV6Spec:      (*types.CustomizationIPSettingsIpV6AddressSpec)(nil),
						DnsServerList: nil,
						DnsDomain:     "",
						PrimaryWINS:   "",
						SecondaryWINS: "",
						NetBIOS:       "",
					},
				},
			},
			EncryptionKey: []uint8{0x30},
		},
	},
	types.CustomizationSpecItem{
		Info: types.CustomizationSpecInfo{
			Name:           "vcsim-windows-domain",
			Description:    "",
			Type:           "Windows",
			ChangeVersion:  "1569970234",
			LastUpdateTime: types.NewTime(time.Now()),
		},
		Spec: types.CustomizationSpec{
			Options: &types.CustomizationWinOptions{
				CustomizationOptions: types.CustomizationOptions{},
				ChangeSID:            true,
				DeleteAccounts:       false,
				Reboot:               "",
			},
			Identity: &types.CustomizationSysprep{
				CustomizationIdentitySettings: types.CustomizationIdentitySettings{},
				GuiUnattended: types.CustomizationGuiUnattended{
					Password: &types.CustomizationPassword{
						Value:     "3Gs...==",
						PlainText: false,
					},
					TimeZone:       15,
					AutoLogon:      false,
					AutoLogonCount: 1,
				},
				UserData: types.CustomizationUserData{
					FullName:     "dougm",
					OrgName:      "VMware",
					ComputerName: &types.CustomizationVirtualMachineName{},
					ProductId:    "",
				},
				GuiRunOnce: (*types.CustomizationGuiRunOnce)(nil),
				Identification: types.CustomizationIdentification{
					JoinWorkgroup: "",
					JoinDomain:    "DOMAIN",
					DomainAdmin:   "vcsim",
					DomainAdminPassword: &types.CustomizationPassword{
						Value:     "H3g...==",
						PlainText: false,
					},
				},
				LicenseFilePrintData: &types.CustomizationLicenseFilePrintData{
					AutoMode:  "perServer",
					AutoUsers: 5,
				},
			},
			GlobalIPSettings: types.CustomizationGlobalIPSettings{},
			NicSettingMap: []types.CustomizationAdapterMapping{
				{
					MacAddress: "",
					Adapter: types.CustomizationIPSettings{
						Ip:            &types.CustomizationUnknownIpGenerator{},
						SubnetMask:    "255.255.255.0",
						Gateway:       []string{"10.0.0.1"},
						IpV6Spec:      (*types.CustomizationIPSettingsIpV6AddressSpec)(nil),
						DnsServerList: nil,
						DnsDomain:     "",
						PrimaryWINS:   "",
						SecondaryWINS: "",
						NetBIOS:       "",
					},
				},
			},
			EncryptionKey: []uint8{0x30},
		},
	},
}

type CustomizationSpecManager struct {
	mo.CustomizationSpecManager

	items []types.CustomizationSpecItem
}

func (m *CustomizationSpecManager) init(r *Registry) {
	m.items = DefaultCustomizationSpec
}

var customizeNameCounter uint64

func customizeName(vm *VirtualMachine, base types.BaseCustomizationName) string {
	n := atomic.AddUint64(&customizeNameCounter, 1)

	switch name := base.(type) {
	case *types.CustomizationPrefixName:
		return fmt.Sprintf("%s-%d", name.Base, n)
	case *types.CustomizationCustomName:
		return fmt.Sprintf("%s-%d", name.Argument, n)
	case *types.CustomizationFixedName:
		return name.Name
	case *types.CustomizationUnknownName:
		return ""
	case *types.CustomizationVirtualMachineName:
		return fmt.Sprintf("%s-%d", vm.Name, n)
	default:
		return ""
	}
}

func (m *CustomizationSpecManager) DoesCustomizationSpecExist(ctx *Context, req *types.DoesCustomizationSpecExist) soap.HasFault {
	exists := false

	for _, item := range m.items {
		if item.Info.Name == req.Name {
			exists = true
			break
		}
	}

	return &methods.DoesCustomizationSpecExistBody{
		Res: &types.DoesCustomizationSpecExistResponse{
			Returnval: exists,
		},
	}
}

func (m *CustomizationSpecManager) GetCustomizationSpec(ctx *Context, req *types.GetCustomizationSpec) soap.HasFault {
	body := new(methods.GetCustomizationSpecBody)

	for _, item := range m.items {
		if item.Info.Name == req.Name {
			body.Res = &types.GetCustomizationSpecResponse{
				Returnval: item,
			}
			return body
		}
	}

	body.Fault_ = Fault("", new(types.NotFound))

	return body
}

func (m *CustomizationSpecManager) CreateCustomizationSpec(ctx *Context, req *types.CreateCustomizationSpec) soap.HasFault {
	body := new(methods.CreateCustomizationSpecBody)

	for _, item := range m.items {
		if item.Info.Name == req.Item.Info.Name {
			body.Fault_ = Fault("", &types.AlreadyExists{Name: req.Item.Info.Name})
			return body
		}
	}

	m.items = append(m.items, req.Item)
	body.Res = new(types.CreateCustomizationSpecResponse)

	return body
}

func (m *CustomizationSpecManager) OverwriteCustomizationSpec(ctx *Context, req *types.OverwriteCustomizationSpec) soap.HasFault {
	body := new(methods.OverwriteCustomizationSpecBody)

	for i, item := range m.items {
		if item.Info.Name == req.Item.Info.Name {
			m.items[i] = req.Item
			body.Res = new(types.OverwriteCustomizationSpecResponse)
			return body
		}
	}

	body.Fault_ = Fault("", new(types.NotFound))

	return body
}

func (m *CustomizationSpecManager) Get() mo.Reference {
	clone := *m

	for i := range clone.items {
		clone.Info = append(clone.Info, clone.items[i].Info)
	}

	return &clone
}
