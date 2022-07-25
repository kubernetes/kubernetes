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
// Copyright 2017 VMware, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package simulator

import (
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// EvalLicense is the default license
var EvalLicense = types.LicenseManagerLicenseInfo{
	LicenseKey: "00000-00000-00000-00000-00000",
	EditionKey: "eval",
	Name:       "Evaluation Mode",
	Properties: []types.KeyAnyValue{
		{
			Key: "feature",
			Value: types.KeyValue{
				Key:   "serialuri:2",
				Value: "Remote virtual Serial Port Concentrator",
			},
		},
		{
			Key: "feature",
			Value: types.KeyValue{
				Key:   "dvs",
				Value: "vSphere Distributed Switch",
			},
		},
	},
}

type LicenseManager struct {
	mo.LicenseManager
}

func NewLicenseManager(ref types.ManagedObjectReference) object.Reference {
	m := &LicenseManager{}
	m.Self = ref
	m.Licenses = []types.LicenseManagerLicenseInfo{EvalLicense}

	if Map.IsVPX() {
		am := Map.Put(&LicenseAssignmentManager{}).Reference()
		m.LicenseAssignmentManager = &am
	}

	return m
}

func (m *LicenseManager) AddLicense(req *types.AddLicense) soap.HasFault {
	body := &methods.AddLicenseBody{
		Res: &types.AddLicenseResponse{},
	}

	for _, license := range m.Licenses {
		if license.LicenseKey == req.LicenseKey {
			body.Res.Returnval = licenseInfo(license.LicenseKey, license.Labels)
			return body
		}
	}

	m.Licenses = append(m.Licenses, types.LicenseManagerLicenseInfo{
		LicenseKey: req.LicenseKey,
		Labels:     req.Labels,
	})

	body.Res.Returnval = licenseInfo(req.LicenseKey, req.Labels)

	return body
}

func (m *LicenseManager) RemoveLicense(req *types.RemoveLicense) soap.HasFault {
	body := &methods.RemoveLicenseBody{
		Res: &types.RemoveLicenseResponse{},
	}

	for i, license := range m.Licenses {
		if req.LicenseKey == license.LicenseKey {
			m.Licenses = append(m.Licenses[:i], m.Licenses[i+1:]...)
			return body
		}
	}
	return body
}

func (m *LicenseManager) UpdateLicenseLabel(req *types.UpdateLicenseLabel) soap.HasFault {
	body := &methods.UpdateLicenseLabelBody{}

	for i := range m.Licenses {
		license := &m.Licenses[i]

		if req.LicenseKey != license.LicenseKey {
			continue
		}

		body.Res = new(types.UpdateLicenseLabelResponse)

		for j := range license.Labels {
			label := &license.Labels[j]

			if label.Key == req.LabelKey {
				if req.LabelValue == "" {
					license.Labels = append(license.Labels[:i], license.Labels[i+1:]...)
				} else {
					label.Value = req.LabelValue
				}
				return body
			}
		}

		license.Labels = append(license.Labels, types.KeyValue{
			Key:   req.LabelKey,
			Value: req.LabelValue,
		})

		return body
	}

	body.Fault_ = Fault("", &types.InvalidArgument{InvalidProperty: "licenseKey"})
	return body
}

type LicenseAssignmentManager struct {
	mo.LicenseAssignmentManager
}

func (m *LicenseAssignmentManager) QueryAssignedLicenses(req *types.QueryAssignedLicenses) soap.HasFault {
	body := &methods.QueryAssignedLicensesBody{
		Res: &types.QueryAssignedLicensesResponse{},
	}

	// EntityId can be a HostSystem or the vCenter InstanceUuid
	if req.EntityId != "" {
		if req.EntityId != Map.content().About.InstanceUuid {
			id := types.ManagedObjectReference{
				Type:  "HostSystem",
				Value: req.EntityId,
			}

			if Map.Get(id) == nil {
				return body
			}
		}
	}

	body.Res.Returnval = []types.LicenseAssignmentManagerLicenseAssignment{
		{
			EntityId:        req.EntityId,
			AssignedLicense: EvalLicense,
		},
	}

	return body
}

func licenseInfo(key string, labels []types.KeyValue) types.LicenseManagerLicenseInfo {
	info := EvalLicense

	info.LicenseKey = key
	info.Labels = labels

	return info
}
