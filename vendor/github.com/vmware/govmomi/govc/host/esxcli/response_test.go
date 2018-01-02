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
	"os"
	"reflect"
	"testing"

	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
)

func load(name string) *Response {
	f, err := os.Open(name)
	if err != nil {
		panic(err)
	}

	defer f.Close()

	var b Response

	dec := xml.NewDecoder(f)
	dec.TypeFunc = types.TypeFunc()
	if err := dec.Decode(&b); err != nil {
		panic(err)
	}

	return &b
}

func TestSystemHostnameGetResponse(t *testing.T) {
	res := load("fixtures/system_hostname_get.xml")

	expect := []Values{
		{
			"DomainName":               {"localdomain"},
			"FullyQualifiedDomainName": {"esxbox.localdomain"},
			"HostName":                 {"esxbox"},
		},
	}

	if !reflect.DeepEqual(res.Values, expect) {
		t.Errorf("%s != %s", res.Values, expect)
	}
}

func TestNetworkVmList(t *testing.T) {
	res := load("fixtures/network_vm_list.xml")

	expect := []Values{
		{
			"Name":     {"foo"},
			"Networks": {"VM Network", "dougm"},
			"NumPorts": {"2"},
			"WorldID":  {"98842"},
		},
		{
			"Name":     {"bar"},
			"Networks": {"VM Network"},
			"NumPorts": {"1"},
			"WorldID":  {"236235"},
		},
	}

	if !reflect.DeepEqual(res.Values, expect) {
		t.Errorf("%s != %s", res.Values, expect)
	}
}

func TestNetworkVmPortList(t *testing.T) {
	r := load("fixtures/network_vm_port_list.xml")

	expect := []Values{
		{
			"IPAddress":    {"192.168.247.149"},
			"MACAddress":   {"00:0c:29:12:b2:cf"},
			"PortID":       {"33554438"},
			"Portgroup":    {"VM Network"},
			"TeamUplink":   {"vmnic0"},
			"UplinkPortID": {"33554434"},
			"vSwitch":      {"vSwitch0"},
			"DVPortID":     {""},
		},
	}

	if !reflect.DeepEqual(r.Values, expect) {
		t.Errorf("%s != %s", r.Values, expect)
	}
}
