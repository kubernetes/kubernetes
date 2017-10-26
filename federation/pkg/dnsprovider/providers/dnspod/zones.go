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

package dnspod

import (
	"strconv"

	dns "github.com/decker502/dnspod-go"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adeherence
var _ dnsprovider.Zones = Zones{}

type Zones struct {
	client *dns.Client
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	domains, _, err := zones.client.Domains.List()
	if err != nil {
		return nil, err
	}
	zoneList := make([]dnsprovider.Zone, len(domains))
	for i, domain := range domains {
		zoneList[i] = &Zone{domain, &zones}
	}
	return zoneList, nil
}

func (zones Zones) Add(zone dnsprovider.Zone) (dnsprovider.Zone, error) {
	newDomain := dns.Domain{Name: zone.Name()}
	domain, _, err := zones.client.Domains.Create(newDomain)
	if err != nil {
		return nil, err
	}
	return &Zone{domain, &zones}, nil
}

func (zones Zones) Remove(zone dnsprovider.Zone) error {
	id, err := strconv.Atoi(zone.ID())
	if err != nil {
		return err
	}
	_, err = zones.client.Domains.Delete(id)
	if err != nil {
		return err
	}
	return nil
}

func (zones Zones) New(name string) (dnsprovider.Zone, error) {
	newDomain := dns.Domain{Name: name}
	return &Zone{newDomain, &zones}, nil
}
