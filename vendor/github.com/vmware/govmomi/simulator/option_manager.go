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

package simulator

import (
	"strings"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type OptionManager struct {
	mo.OptionManager
}

func NewOptionManager(ref *types.ManagedObjectReference, setting []types.BaseOptionValue) object.Reference {
	s := &OptionManager{}
	if ref != nil {
		s.Self = *ref
	}
	s.Setting = setting
	return s
}

func (m *OptionManager) QueryOptions(req *types.QueryOptions) soap.HasFault {
	body := &methods.QueryOptionsBody{}
	res := &types.QueryOptionsResponse{}

	for _, opt := range m.Setting {
		if strings.HasPrefix(opt.GetOptionValue().Key, req.Name) {
			res.Returnval = append(res.Returnval, opt)
		}
	}

	if len(res.Returnval) == 0 {
		body.Fault_ = Fault("", &types.InvalidName{Name: req.Name})
	} else {
		body.Res = res
	}

	return body
}

func (m *OptionManager) find(key string) *types.OptionValue {
	for _, opt := range m.Setting {
		setting := opt.GetOptionValue()
		if setting.Key == key {
			return setting
		}
	}
	return nil
}

func (m *OptionManager) UpdateOptions(req *types.UpdateOptions) soap.HasFault {
	body := new(methods.UpdateOptionsBody)

	for _, change := range req.ChangedValue {
		setting := change.GetOptionValue()

		// We don't currently include the entire list of default settings for ESX and vCenter,
		// this prefix is currently used to test the failure path.
		// Real vCenter seems to only allow new options if Key has a "config." prefix.
		// TODO: consider behaving the same, which would require including 2 long lists of options in vpx.Setting and esx.Setting
		if strings.HasPrefix(setting.Key, "ENOENT.") {
			body.Fault_ = Fault("", &types.InvalidName{Name: setting.Key})
			return body
		}

		opt := m.find(setting.Key)
		if opt != nil {
			// This is an existing option.
			opt.Value = setting.Value
			continue
		}

		m.Setting = append(m.Setting, change)
	}

	body.Res = new(types.UpdateOptionsResponse)
	return body
}
