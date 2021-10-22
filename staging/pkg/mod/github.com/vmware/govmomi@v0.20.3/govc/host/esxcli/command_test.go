/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package esxcli

import (
	"reflect"
	"testing"

	"github.com/vmware/govmomi/internal"
	"github.com/vmware/govmomi/vim25/types"
)

func TestSystemSettingsAdvancedSetCommand(t *testing.T) {
	c := NewCommand([]string{"system", "settings", "advanced", "set", "-o", "/Net/GuestIPHack", "-i", "1"})

	tests := []struct {
		f      func() string
		expect string
	}{
		{c.Name, "set"},
		{c.Namespace, "system.settings.advanced"},
		{c.Method, "vim.EsxCLI.system.settings.advanced.set"},
		{c.Moid, "ha-cli-handler-system-settings-advanced"},
	}

	for _, test := range tests {
		actual := test.f()
		if actual != test.expect {
			t.Errorf("%s != %s", actual, test.expect)
		}
	}

	params := []CommandInfoParam{
		{
			CommandInfoItem: CommandInfoItem{Name: "default", DisplayName: "default", Help: "Reset the option to its default value."},
			Aliases:         []string{"-d", "--default"},
			Flag:            true,
		},
		{
			CommandInfoItem: CommandInfoItem{Name: "intvalue", DisplayName: "int-value", Help: "If the option is an integer value use this option."},
			Aliases:         []string{"-i", "--int-value"},
			Flag:            false,
		},
		{
			CommandInfoItem: CommandInfoItem{Name: "option", DisplayName: "option", Help: "The name of the option to set the value of. Example: \"/Misc/HostName\""},
			Aliases:         []string{"-o", "--option"},
			Flag:            false,
		},
		{
			CommandInfoItem: CommandInfoItem{Name: "stringvalue", DisplayName: "string-value", Help: "If the option is a string use this option."},
			Aliases:         []string{"-s", "--string-value"},
			Flag:            false,
		},
	}

	args, err := c.Parse(params)
	if err != nil {
		t.Fatal(err)
	}

	expect := []internal.ReflectManagedMethodExecuterSoapArgument{
		{
			DynamicData: types.DynamicData{},
			Name:        "intvalue",
			Val:         "<intvalue>1</intvalue>",
		},
		{
			DynamicData: types.DynamicData{},
			Name:        "option",
			Val:         "<option>/Net/GuestIPHack</option>",
		},
	}

	if !reflect.DeepEqual(args, expect) {
		t.Errorf("%s != %s", args, expect)
	}
}

func TestNetworkVmListCommand(t *testing.T) {
	c := NewCommand([]string{"network", "vm", "list"})

	tests := []struct {
		f      func() string
		expect string
	}{
		{c.Name, "list"},
		{c.Namespace, "network.vm"},
		{c.Method, "vim.EsxCLI.network.vm.list"},
		{c.Moid, "ha-cli-handler-network-vm"},
	}

	for _, test := range tests {
		actual := test.f()
		if actual != test.expect {
			t.Errorf("%s != %s", actual, test.expect)
		}
	}

	var params []CommandInfoParam

	args, err := c.Parse(params)
	if err != nil {
		t.Fatal(err)
	}

	expect := []internal.ReflectManagedMethodExecuterSoapArgument{}

	if !reflect.DeepEqual(args, expect) {
		t.Errorf("%s != %s", args, expect)
	}
}
