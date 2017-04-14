/*
Copyright 2016 The Kubernetes Authors.

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

package coredns

import (
	"fmt"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"strings"
)

// Compile time check for interface adherence
var _ dnsprovider.Zones = Zones{}

type Zones struct {
	intf     *Interface
	zoneList []Zone
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	var zoneList []dnsprovider.Zone
	for _, zone := range zones.zoneList {
		zoneList = append(zoneList, zone)
	}
	return zoneList, nil
}

func (zones Zones) Get(dnsZoneName string, dnsZoneID string) (dnsprovider.Zone, error) {
	dnsZones, err := zones.List()
	if err != nil {
		return nil, err
	}

	var matches []dnsprovider.Zone
	for _, zone := range dnsZones {
		if strings.TrimSuffix(dnsZoneName, ".") == strings.TrimSuffix(zone.Name(), ".") {
			matches = append(matches, zone)
		}
	}

	if len(matches) == 0 {
		return nil, nil
	}
	if len(matches) > 1 {
		return nil, fmt.Errorf("DNS zone %s is ambiguous (please specify zoneID).", dnsZoneName)
	}
	return matches[0], nil
}

func (zones Zones) Add(zone dnsprovider.Zone) (dnsprovider.Zone, error) {
	return &Zone{}, fmt.Errorf("OperationNotSupported")
}

func (zones Zones) Remove(zone dnsprovider.Zone) error {
	return fmt.Errorf("OperationNotSupported")
}
func (zones Zones) New(name string) (dnsprovider.Zone, error) {
	return &Zone{}, fmt.Errorf("OperationNotSupported")
}
