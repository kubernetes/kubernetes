/*
Copyright 2014 Google Inc. All rights reserved.

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

package cloudprovider

import (
	"net"
	"regexp"
)

type FakeCloud struct {
	Exists   bool
	Err      error
	Calls    []string
	IP       net.IP
	Machines []string
}

func (f *FakeCloud) addCall(desc string) {
	f.Calls = append(f.Calls, desc)
}

func (f *FakeCloud) ClearCalls() {
	f.Calls = []string{}
}

func (f *FakeCloud) TCPLoadBalancer() (TCPLoadBalancer, bool) {
	return f, true
}

func (f *FakeCloud) Instances() (Instances, bool) {
	return f, true
}

func (f *FakeCloud) TCPLoadBalancerExists(name, region string) (bool, error) {
	return f.Exists, f.Err
}

func (f *FakeCloud) CreateTCPLoadBalancer(name, region string, port int, hosts []string) error {
	f.addCall("create")
	return f.Err
}

func (f *FakeCloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	f.addCall("update")
	return f.Err
}

func (f *FakeCloud) DeleteTCPLoadBalancer(name, region string) error {
	f.addCall("delete")
	return f.Err
}

func (f *FakeCloud) IPAddress(instance string) (net.IP, error) {
	f.addCall("ip-address")
	return f.IP, f.Err
}

func (f *FakeCloud) List(filter string) ([]string, error) {
	f.addCall("list")
	result := []string{}
	for _, machine := range f.Machines {
		if match, _ := regexp.MatchString(filter, machine); match {
			result = append(result, machine)
		}
	}
	return result, f.Err
}
