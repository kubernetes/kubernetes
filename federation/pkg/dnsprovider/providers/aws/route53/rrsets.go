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
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/route53"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

// Compile time check for interface adeherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone *Zone
}

func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	input := route53.ListResourceRecordSetsInput{
		HostedZoneId: rrsets.zone.impl.Id,
	}
	response, err := rrsets.zone.zones.interface_.service.ListResourceRecordSets(&input)
	// TODO: Handle truncated responses
	if err != nil {
		return nil, err
	}
	list := make([]dnsprovider.ResourceRecordSet, len(response.ResourceRecordSets))
	for i, rrset := range response.ResourceRecordSets {
		list[i] = &ResourceRecordSet{rrset, &rrsets}
	}
	return list, nil
}

func (rrsets ResourceRecordSets) Add(rrset dnsprovider.ResourceRecordSet) (dnsprovider.ResourceRecordSet, error) {
	service := rrsets.zone.zones.interface_.service
	input := getChangeResourceRecordSetsInput("CREATE", rrset.Name(), string(rrset.Type()), *rrset.(ResourceRecordSet).rrsets.zone.impl.Id, rrset.Rrdatas(), rrset.Ttl())
	_, err := service.ChangeResourceRecordSets(input)
	if err != nil {
		// Cast err to awserr.Error to get the Code and
		// Message from an error.
		return nil, err
	}
	return ResourceRecordSet{input.ChangeBatch.Changes[0].ResourceRecordSet, &rrsets}, nil
}

func (rrsets ResourceRecordSets) Remove(rrset dnsprovider.ResourceRecordSet) error {
	input := getChangeResourceRecordSetsInput("DELETE", rrset.Name(), string(rrset.Type()), *rrset.(ResourceRecordSet).rrsets.zone.impl.Id, rrset.Rrdatas(), rrset.Ttl())
	_, err := rrsets.zone.zones.interface_.service.ChangeResourceRecordSets(input)
	if err != nil {
		// Cast err to awserr.Error to get the Code and
		// Message from an error.
		return err
	}
	return nil
}

func getChangeResourceRecordSetsInput(action, name, type_, hostedZoneId string, rrdatas []string, ttl int64) *route53.ChangeResourceRecordSetsInput {
	input := &route53.ChangeResourceRecordSetsInput{
		ChangeBatch: &route53.ChangeBatch{ // Required
			Changes: []*route53.Change{ // Required
				{ // Required
					Action: aws.String(action), // Required
					ResourceRecordSet: &route53.ResourceRecordSet{ // Required
						Name: aws.String(name),  // Required
						Type: aws.String(type_), // Required
						/*
							AliasTarget: &route53.AliasTarget{
								DNSName:              aws.String("DNSName"),    // Required
								EvaluateTargetHealth: aws.Bool(true),           // Required
								HostedZoneId:         aws.String("ResourceId"), // Required
							},
							Failover: aws.String("ResourceRecordSetFailover"),
							GeoLocation: &route53.GeoLocation{
								ContinentCode:   aws.String("GeoLocationContinentCode"),
								CountryCode:     aws.String("GeoLocationCountryCode"),
								SubdivisionCode: aws.String("GeoLocationSubdivisionCode"),
							},
							HealthCheckId: aws.String("HealthCheckId"),
							Region:        aws.String("ResourceRecordSetRegion"),
						*/
						ResourceRecords: []*route53.ResourceRecord{
							{ // Required
								Value: aws.String(rrdatas[0]), // Required
							},
							// More values...
						},
						/*
							SetIdentifier: aws.String("ResourceRecordSetIdentifier"),

						*/
						TTL: aws.Int64(ttl),
						/*
							TrafficPolicyInstanceId: aws.String("TrafficPolicyInstanceId"),
							Weight:                  aws.Int64(1),
						*/
					},
				},
				// More values...
			},
		},
		HostedZoneId: aws.String(hostedZoneId), // Required
	}
	return input
}

func (rrsets ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	rrstypeStr := string(rrstype)
	return ResourceRecordSet{
		&route53.ResourceRecordSet{
			Name: &name,
			Type: &rrstypeStr,
			TTL:  &ttl,
			ResourceRecords: []*route53.ResourceRecord{
				{
					Value: &rrdatas[0],
				},
			},
		}, // TODO: Add remaining rrdatas
		&rrsets,
	}
}
