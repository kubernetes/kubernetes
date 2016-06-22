/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package route53

import (
	"github.com/aws/aws-sdk-go/service/route53"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/pkg/util"
)

// Compile time check for interface adeherence
var _ dnsprovider.Zones = Zones{}

type Zones struct {
	interface_ *Interface
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	input := route53.ListHostedZonesInput{}
	response, err := zones.interface_.service.ListHostedZones(&input)
	if err != nil {
		return []dnsprovider.Zone{}, err
	}
	hostedZones := response.HostedZones
	// TODO: Handle result truncation
	// https://docs.aws.amazon.com/sdk-for-go/api/service/route53/Route53.html#ListHostedZones-instance_method
	zoneList := make([]dnsprovider.Zone, len(hostedZones))
	for i, zone := range hostedZones {
		zoneList[i] = &Zone{zone, &zones}
	}
	return zoneList, nil
}

func (zones Zones) Add(zone dnsprovider.Zone) (dnsprovider.Zone, error) {
	dnsName := zone.Name()
	callerReference := string(util.NewUUID())
	input := route53.CreateHostedZoneInput{Name: &dnsName, CallerReference: &callerReference}
	output, err := zones.interface_.service.CreateHostedZone(&input)
	if err != nil {
		return nil, err
	}
	return &Zone{output.HostedZone, &zones}, nil
}

func (zones Zones) Remove(zone dnsprovider.Zone) error {
	zoneId := zone.(*Zone).impl.Id
	input := route53.DeleteHostedZoneInput{Id: zoneId}
	_, err := zones.interface_.service.DeleteHostedZone(&input)
	if err != nil {
		return err
	}
	return nil
}
func (zones Zones) New(name string) (dnsprovider.Zone, error) {
	id := string(util.NewUUID())
	managedZone := route53.HostedZone{Id: &id, Name: &name}
	return &Zone{&managedZone, &zones}, nil
}
