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

package ovirt

import (
	"io"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/cloudprovider"
)

func TestOVirtCloudConfiguration(t *testing.T) {
	config1 := (io.Reader)(nil)

	_, err1 := cloudprovider.GetCloudProvider("ovirt", config1)
	if err1 == nil {
		t.Fatalf("An error is expected when the configuration is missing")
	}

	config2 := strings.NewReader("")

	_, err2 := cloudprovider.GetCloudProvider("ovirt", config2)
	if err2 == nil {
		t.Fatalf("An error is expected when the configuration is empty")
	}

	config3 := strings.NewReader(`
[connection]
	`)

	_, err3 := cloudprovider.GetCloudProvider("ovirt", config3)
	if err3 == nil {
		t.Fatalf("An error is expected when the uri is missing")
	}

	config4 := strings.NewReader(`
[connection]
uri = https://localhost:8443/ovirt-engine/api
`)

	_, err4 := cloudprovider.GetCloudProvider("ovirt", config4)
	if err4 != nil {
		t.Fatalf("Unexpected error creating the provider: %s", err4)
	}
}

func TestOVirtCloudXmlParsing(t *testing.T) {
	body1 := (io.Reader)(nil)

	_, err1 := getInstancesFromXml(body1)
	if err1 == nil {
		t.Fatalf("An error is expected when body is missing")
	}

	body2 := strings.NewReader("")

	_, err2 := getInstancesFromXml(body2)
	if err2 == nil {
		t.Fatalf("An error is expected when body is empty")
	}

	body3 := strings.NewReader(`
<vms>
  <vm></vm>
</vms>
`)

	instances3, err3 := getInstancesFromXml(body3)
	if err3 != nil {
		t.Fatalf("Unexpected error listing instances: %s", err3)
	}
	if len(instances3) > 0 {
		t.Fatalf("Unexpected number of instance(s): %d", len(instances3))
	}

	body4 := strings.NewReader(`
<vms>
  <vm>
    <status><state>Up</state></status>
    <guest_info><fqdn>host1</fqdn></guest_info>
  </vm>
  <vm>
    <!-- empty -->
  </vm>
  <vm>
    <status><state>Up</state></status>
  </vm>
  <vm>
    <status><state>Down</state></status>
    <guest_info><fqdn>host2</fqdn></guest_info>
  </vm>
  <vm>
    <status><state>Up</state></status>
    <guest_info><fqdn>host3</fqdn></guest_info>
  </vm>
</vms>
`)

	instances4, err4 := getInstancesFromXml(body4)
	if err4 != nil {
		t.Fatalf("Unexpected error listing instances: %s", err4)
	}
	if len(instances4) != 2 {
		t.Fatalf("Unexpected number of instance(s): %d", len(instances4))
	}

	names := instances4.ListSortedNames()
	if names[0] != "host1" || names[1] != "host3" {
		t.Fatalf("Unexpected instance(s): %s", instances4)
	}
}
