/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/route53"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

const (
	regionName = "us-west-2"
)

type (
	Interface struct {
		dnsService *route53.Route53
	}

	Zones struct {
		Interface
	}

	Zone struct {
		Zones
		name string // e.g. "google.com"
		id   string // Internal identifier used against cloud API
	}

	ResourceRecordSets struct {
		zone Zone
	}

	ResourceRecordSet struct {
		rrsets                  ResourceRecordSets
		name                    string
		ttl                     int64
		rrsType                 rrstype.RrsType
		rrDatas                 []string
		failover                string
		healthCheckId           string
		regionName              string
		setIdentifier           string
		trafficPolicyInstanceId string
		weight                  int64
		/* aliasTargets[] */ // TODO: quinton
		/* geoLocation */ // TODO: quinton
	}
)

var (
	// Must satisfy dnsprovider interfaces - compile time check
	_ dnsprovider.Interface          = Interface{}
	_ dnsprovider.Zones              = Zones{}
	_ dnsprovider.Zone               = Zone{}
	_ dnsprovider.ResourceRecordSets = ResourceRecordSets{}
	_ dnsprovider.ResourceRecordSet  = ResourceRecordSet{}
)

func NewInterface() Interface {
	dnsService := route53.New(session.New(&aws.Config{Region: aws.String("us-west-2")}))
	if dnsService == nil {
		glog.Errorf("Failed to get DNS client for AWS Route53.  No details of error provided.")
	}
	glog.Infof("Successfully got DNS client for AWS Route53: %v\n", *dnsService)

	return Interface{dnsService: dnsService}
}

func (iface Interface) Zones() (dnsprovider.Zones, bool) {
	return Zones{iface}, true
}

func (zones Zones) List() ([]dnsprovider.Zone, error) {
	var zlist []dnsprovider.Zone
	input := route53.ListHostedZonesInput{}
	output, err := zones.dnsService.ListHostedZones(&input)
	if err != nil {
		glog.Errorf("Failed to list hosted zones: %v", err)
		return zlist, err
	}
	zlist = make([]dnsprovider.Zone, len(output.HostedZones))
	for i, z := range output.HostedZones {
		zlist[i] = Zone{zones, *z.Name, *z.Id}
	}
	return zlist, err
}

func (zone Zone) Name() string {
	return zone.name
}

func (zone Zone) ResourceRecordSets() (dnsprovider.ResourceRecordSets, bool) {
	return ResourceRecordSets{zone}, true
}

func (rrs ResourceRecordSets) performAction(r dnsprovider.ResourceRecordSet, action string) (dnsprovider.ResourceRecordSet, error) {
	input := &route53.ChangeResourceRecordSetsInput{
		ChangeBatch: &route53.ChangeBatch{ // Required
			Changes: []*route53.Change{ // Required
				{ // Required
					Action: aws.String(action), // Required
					ResourceRecordSet: &route53.ResourceRecordSet{ // Required
						Name: aws.String(r.Name()),         // Required
						Type: aws.String(string(r.Type())), // Required
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
								Value: aws.String(r.Rrdatas()[0]), // Required
							},
							// More values...
						},
						/*
							SetIdentifier: aws.String("ResourceRecordSetIdentifier"),

						*/
						TTL: aws.Int64(r.Ttl()),
						/*
							TrafficPolicyInstanceId: aws.String("TrafficPolicyInstanceId"),
							Weight:                  aws.Int64(1),
						*/
					},
				},
				// More values...
			},
			Comment: aws.String("Ubernetes"), // TODO: quinton - client should provide this, and it should contain e.g. the service name, cluster, ID etc.
		},
		HostedZoneId: aws.String(rrs.zone.id), // Required
	}
	_, err := rrs.zone.dnsService.ChangeResourceRecordSets(input)
	if err != nil {
		// Print the error, cast err to awserr.Error to get the Code and
		// Message from an error.
		glog.Errorf("Failed to %s resource record %v: %v\n", action, r, err.Error())
		return ResourceRecordSet{}, err
	}
	data := []string{r.Rrdatas()[0]} // TODO!  quinton
	return ResourceRecordSet{rrsets: rrs, name: r.Name(), ttl: r.Ttl(), rrsType: r.Type(), rrDatas: data}, nil
}

func (rrs ResourceRecordSets) Add(r dnsprovider.ResourceRecordSet) (dnsprovider.ResourceRecordSet, error) {
	return rrs.performAction(r, "CREATE")
}

func (rrs ResourceRecordSets) Remove(r dnsprovider.ResourceRecordSet) error {
	_, err := rrs.performAction(r, "DELETE")
	return err
}

func (rrs ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	var returnVal []dnsprovider.ResourceRecordSet
	input := &route53.ListResourceRecordSetsInput{
		HostedZoneId: aws.String(rrs.zone.id), // Required
		/*
			MaxItems:              aws.String("PageMaxItems"),
			StartRecordIdentifier: aws.String("ResourceRecordSetIdentifier"),
			StartRecordName:       aws.String("DNSName"),
			StartRecordType:       aws.String("RRType"),
		*/
	}
	output, err := rrs.zone.dnsService.ListResourceRecordSets(input)

	if err != nil {
		return returnVal, err
	}

	returnVal = make([]dnsprovider.ResourceRecordSet, len(output.ResourceRecordSets))
	for i, rs := range output.ResourceRecordSets {
		returnVal[i] = ResourceRecordSet{rrsets: rrs, name: *rs.Name, ttl: *rs.TTL, rrsType: rrstype.RrsType(*rs.Type), rrDatas: []string{*rs.ResourceRecords[0].Value}}
	}
	return returnVal, nil
}

func (rs ResourceRecordSet) Name() string {
	return rs.name
}

func (rs ResourceRecordSet) Ttl() int64 {
	return rs.ttl
}

func (rs ResourceRecordSet) Type() rrstype.RrsType {
	return rrstype.RrsType(rs.rrsType)
}

func (rs ResourceRecordSet) Rrdatas() []string {
	return rs.rrDatas
	// return []string{} //TODO: Add stuff here.
}

func NewResourceRecordSet(rss ResourceRecordSets, name string, ttl int64, rrsType rrstype.RrsType, rrDatas []string) ResourceRecordSet {
	return ResourceRecordSet{rrsets: rss, name: name, ttl: ttl, rrsType: rrsType, rrDatas: rrDatas}
}
