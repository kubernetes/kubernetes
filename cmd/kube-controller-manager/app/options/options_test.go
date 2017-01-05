/*
Copyright 2014 The Kubernetes Authors.

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

package options

import (
	"testing"

	"github.com/spf13/pflag"
)

func TestAddFlagsFlag(t *testing.T) {
	// TODO: This tests four flags (master,cluster-name,cluster-cidr,service-cluster-ip-range) for now.
	// Expand the test to include other flags as well.
	f := pflag.NewFlagSet("addflagstest", pflag.ContinueOnError)
	s := NewCMServer()
	s.AddFlags(f)
	if s.ClusterName != "kubernetes" {
		t.Errorf("CMServer ClusterName expected kubernetes , got : %s", s.ClusterName)
	}

	args := []string{
		"--master=10.45.1.1:8080",
		"--cluster-name=mykubernetes",
		"--cluster-cidr=172.17.0.1/16",
		"--service-cluster-ip-range=192.168.1.1/16",
	}
	f.Parse(args)
	if s.Master != "10.45.1.1:8080" || s.ClusterName != "mykubernetes" ||
		s.ClusterCIDR != "172.17.0.1/16" || s.ServiceCIDR != "192.168.1.1/16" {
		t.Errorf("s.Master expected to be 10.45.1.1:8080, s.ClusterName expected to be mykubernetes, " +
			"s.ClusterCIDR expected to be 172.17.0.1/16 and s.ServiceCIDR expected to be 192.168.1.1/16")
	}
}
