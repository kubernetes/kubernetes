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

package route53

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/route53"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"strings"
)

// Compile time check for interface adherence
var _ dnsprovider.Zones = Zones{}

type Zones struct {
	interface_ *Interface
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	var zoneList []dnsprovider.Zone

	input := route53.ListHostedZonesInput{}
	err := zones.interface_.service.ListHostedZonesPages(&input, func(page *route53.ListHostedZonesOutput, lastPage bool) bool {
		for _, zone := range page.HostedZones {
			zoneList = append(zoneList, &Zone{zone, &zones})
		}
		return true
	})
	if err != nil {
		return []dnsprovider.Zone{}, err
	}
	return zoneList, nil
}

func (zones Zones) Get(dnsZoneName string, dnsZoneID string) (dnsprovider.Zone, error) {
	if len(dnsZoneID) > 0 {
		input := route53.GetHostedZoneInput{
			Id: aws.String(dnsZoneID),
		}
		output, err := zones.interface_.service.GetHostedZone(&input)
		if err != nil {
			return nil, err
		}
		return &Zone{output.HostedZone, &zones}, nil
	} else {
		input := route53.ListHostedZonesByNameInput{
			DNSName:  aws.String(dnsZoneName),
			MaxItems: aws.String("2"), // get 2 items to make sure there is no duplicate zones with same DNS name
		}
		output, err := zones.interface_.service.ListHostedZonesByName(&input)
		if err != nil {
			return nil, err
		}

		var matches []*route53.HostedZone
		for _, hostedZone := range output.HostedZones {
			if strings.TrimSuffix(dnsZoneName, ".") == strings.TrimSuffix(aws.StringValue(hostedZone.Name), ".") {
				matches = append(matches, hostedZone)
			}
		}

		if len(matches) == 0 {
			return nil, nil
		}
		if len(matches) > 1 {
			return nil, fmt.Errorf("DNS zone %s is ambiguous (please specify zoneID).", dnsZoneName)
		}
		return &Zone{matches[0], &zones}, nil
	}
}

func (zones Zones) Add(zone dnsprovider.Zone) (dnsprovider.Zone, error) {
	dnsName := zone.Name()
	callerReference := string(uuid.NewUUID())
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
	id := string(uuid.NewUUID())
	managedZone := route53.HostedZone{Id: &id, Name: &name}
	return &Zone{&managedZone, &zones}, nil
}
