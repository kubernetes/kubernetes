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

package toolbox

import (
	"net"
	"reflect"
	"testing"
)

func TestDefaultGuestNicProto(t *testing.T) {
	p := DefaultGuestNicInfo()

	info := p.V3

	for _, nic := range info.Nics {
		if len(nic.MacAddress) == 0 {
			continue
		}
		_, err := net.ParseMAC(nic.MacAddress)
		if err != nil {
			t.Errorf("invalid MAC %s: %s", nic.MacAddress, err)
		}
	}

	b, err := EncodeXDR(p)
	if err != nil {
		t.Fatal(err)
	}

	var dp GuestNicInfo
	err = DecodeXDR(b, &dp)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(p, &dp) {
		t.Error("decode mismatch")
	}
}

func TestMaxGuestNic(t *testing.T) {
	p := DefaultGuestNicInfo()

	maxNics = len(p.V3.Nics)

	a, _ := net.Interfaces()
	a = append(a, a...) // double the number of interfaces returned
	netInterfaces = func() ([]net.Interface, error) {
		return a, nil
	}

	p = DefaultGuestNicInfo()

	l := len(p.V3.Nics)
	if l != maxNics {
		t.Errorf("Nics=%d", l)
	}
}
