/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package object

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/vmware/govmomi/test"
	"github.com/vmware/govmomi/vim25/mo"
)

func TestSearch(t *testing.T) {
	c := test.NewAuthenticatedClient(t)
	s := NewSearchIndex(c)

	ref, err := s.FindChild(context.Background(), NewRootFolder(c), "ha-datacenter")
	if err != nil {
		t.Fatal(err)
	}

	dc, ok := ref.(*Datacenter)
	if !ok {
		t.Errorf("Expected Datacenter: %#v", ref)
	}

	folders, err := dc.Folders(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	ref, err = s.FindChild(context.Background(), folders.DatastoreFolder, "datastore1")
	if err != nil {
		t.Fatal(err)
	}

	_, ok = ref.(*Datastore)
	if !ok {
		t.Errorf("Expected Datastore: %#v", ref)
	}

	ref, err = s.FindByInventoryPath(context.Background(), "/ha-datacenter/network/VM Network")
	if err != nil {
		t.Fatal(err)
	}

	_, ok = ref.(*Network)
	if !ok {
		t.Errorf("Expected Network: %#v", ref)
	}

	crs, err := folders.HostFolder.Children(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(crs) != 0 {
		var cr mo.ComputeResource
		ref = crs[0]
		err = s.Properties(context.Background(), ref.Reference(), []string{"host"}, &cr)
		if err != nil {
			t.Fatal(err)
		}

		var host mo.HostSystem
		ref = NewHostSystem(c, cr.Host[0])
		err = s.Properties(context.Background(), ref.Reference(), []string{"name", "hardware", "config"}, &host)
		if err != nil {
			t.Fatal(err)
		}

		dnsConfig := host.Config.Network.DnsConfig.GetHostDnsConfig()
		dnsName := fmt.Sprintf("%s.%s", dnsConfig.HostName, dnsConfig.DomainName)
		shost, err := s.FindByDnsName(context.Background(), dc, dnsName, false)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ref, shost) {
			t.Errorf("%#v != %#v\n", ref, shost)
		}

		shost, err = s.FindByUuid(context.Background(), dc, host.Hardware.SystemInfo.Uuid, false, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ref, shost) {
			t.Errorf("%#v != %#v\n", ref, shost)
		}
	}

	vms, err := folders.VmFolder.Children(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(vms) != 0 {
		var vm mo.VirtualMachine
		ref = vms[0]
		err = s.Properties(context.Background(), ref.Reference(), []string{"config", "guest"}, &vm)
		if err != nil {
			t.Fatal(err)
		}
		svm, err := s.FindByDatastorePath(context.Background(), dc, vm.Config.Files.VmPathName)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ref, svm) {
			t.Errorf("%#v != %#v\n", ref, svm)
		}

		svm, err = s.FindByUuid(context.Background(), dc, vm.Config.Uuid, true, nil)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ref, svm) {
			t.Errorf("%#v != %#v\n", ref, svm)
		}

		if vm.Guest.HostName != "" {
			svm, err := s.FindByDnsName(context.Background(), dc, vm.Guest.HostName, true)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(ref, svm) {
				t.Errorf("%#v != %#v\n", ref, svm)
			}
		}

		if vm.Guest.IpAddress != "" {
			svm, err := s.FindByIp(context.Background(), dc, vm.Guest.IpAddress, true)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(ref, svm) {
				t.Errorf("%#v != %#v\n", ref, svm)
			}
		}
	}
}
