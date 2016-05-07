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

package clouddns

import (
	"fmt"
	"net/http"

	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
	dns "google.golang.org/api/dns/v1"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

const (
	projectName = "federation0-cluster00"
)

var (
	// Implementations must satisfy interfaces - compile time check
	_ dnsprovider.Interface = ifaceImpl{}
	_ iface                 = ifaceImpl{}
	_ dnsprovider.Zones     = zonesImpl{}
	_ zones                 = zonesImpl{}
	_ dnsprovider.Zone      = zoneImpl{}
	_ zone                  = zoneImpl{}
	// _ managedZonesService            = zonesImpl{}
	_ dnsprovider.ResourceRecordSets = resourceRecordSetsImpl{}
	_ resourceRecordSets             = resourceRecordSetsImpl{}
	_ dnsprovider.ResourceRecordSet  = resourceRecordSetImpl{}
	_ resourceRecordSet              = resourceRecordSetImpl{}
	_ changesService                 = changesServiceImpl{}
	// _ managedZonesService            = managedZonesServiceImpl{}
	// _ managedZone                    = managedZoneImpl{}
	// _ resourceRecordSetsService      = resourceRecordSetsServiceImpl{}
	// _ dnsService                     = dnsServiceImpl{}
)

type (
	changesService interface {
		Do(project string, zone string, additions, deletions []dnsprovider.ResourceRecordSet) error
	}

	changesServiceImpl struct {
		service dns.ChangesService
	}

	/*
		managedZonesService interface {
			// List(project string) *managedZonesListCall
			// List(project string) ([]managedZone, error)
			//	List() ([]managedZone, error)
		}
	*/

	managedZonesServiceImpl struct {
		service dns.ManagedZonesService
	}

	managedZone interface {
		dnsName() string
		id() uint64
		name() string
	}

	managedZoneImpl struct {
		dns.ManagedZone
	}

	/*
		managedZonesListResponse interface {
			kind() string
			managedZones() []managedZone
		}

		managedZonesListResponseImpl struct {
			dns.ManagedZonesListResponse
		}

		managedZonesListCall interface {
			Do(opts ...googleapi.CallOption) (managedZonesListResponse, error)
		}

		managedZonesListCallImpl struct {
			impl dns.ManagedZonesListCall
		}
	*/
	projectsService interface {
		// Not currently needed or implemented.
	}

	resourceRecordSetsService interface {
		List( /*project, */ zone string) ([]dnsprovider.ResourceRecordSet, error)
	}

	resourceRecordSetsServiceImpl struct {
		impl dns.ResourceRecordSetsService
	}

	resourceRecordSets interface {
		dnsprovider.ResourceRecordSets
		zone() zone
		//		rrsets() []resourceRecordSet
	}

	resourceRecordSetsImpl struct {
		_zone zoneImpl
		impl  dns.ResourceRecordSetsService
		// _rrsets []resourceRecordSetImpl
	}

	resourceRecordSet interface {
		dnsprovider.ResourceRecordSet
		ResourceRecordSets() []resourceRecordSet
	}

	resourceRecordSetImpl struct {
		_rrsets  resourceRecordSetsImpl
		_name    string
		_ttl     int64
		_rrsType rrstype.RrsType
		_rrDatas []string
	}

	dnsServiceImpl struct {
		service dns.Service
	}

	dnsService interface {
		dnsprovider.Interface
		// BasePath() string
		// UserAgent() string
		Changes() changesService
		// ManagedZones() zones
		// Projects() projectsService // Not yet needed
		resourceRecordSets() resourceRecordSets // resourceRecordSetsService
	}

	iface interface {
		dnsService
		dnsprovider.Interface
		service() dnsService
	}

	ifaceImpl struct {
		_service dns.Service
	}

	zonesImpl struct {
		_iface ifaceImpl
		impl   dns.ManagedZonesService
	}

	zones interface {
		// managedZonesService // New!
		dnsprovider.Zones
		list() ([]zone, error)
		iface() iface
	}

	zone interface {
		dnsprovider.Zone
		managedZone
		zones() zones
		// name() string // e.g. "google.com"
	}

	zoneImpl struct {
		_zones zonesImpl
		impl   dns.ManagedZone
		// _name  string // e.g. "google.com"
		// _id    string // Internal identifier used against cloud API
	}
)

func (s changesServiceImpl) Do(project string, zone string, additions, deletions []dnsprovider.ResourceRecordSet) error {
	pAdditions := make([]*dns.ResourceRecordSet, len(additions))
	pDeletions := make([]*dns.ResourceRecordSet, len(deletions))
	for i, addition := range additions {
		pAdditions[i] = &dns.ResourceRecordSet{Name: addition.Name(), Rrdatas: addition.Rrdatas(), Ttl: addition.Ttl(), Type: string(addition.Type())}
	}
	for i, deletion := range deletions {
		pDeletions[i] = &dns.ResourceRecordSet{Name: deletion.Name(), Rrdatas: deletion.Rrdatas(), Ttl: deletion.Ttl(), Type: string(deletion.Type())}
	}
	call := s.service.Create(project, zone, &dns.Change{Additions: pAdditions, Deletions: pDeletions})
	_, err := call.Do() // TODO: Don't throw away call status tracking info
	if err != nil {
		return err
	}
	return nil
}

/*
func (impl managedZonesListResponseImpl) kind() string {
	return impl.Kind
}

func (impl managedZonesListResponseImpl) managedZones() []managedZone {
	var zs []managedZone = make([]managedZone, len(impl.ManagedZones))
	for i, z := range impl.ManagedZones {
		zs[i] = managedZoneImpl{*z}
	}
	return zs
}
*/

/*
func (impl managedZoneImpl) dnsName() string {
	return impl.DnsName
}

func (impl managedZoneImpl) id() uint64 {
	return impl.Id
}

func (impl managedZoneImpl) name() string {
	return impl.Name
}
*/
func (zone zoneImpl) dnsName() string {
	return zone.impl.DnsName
}

func (zone zoneImpl) id() uint64 {
	return zone.impl.Id
}

func (zone zoneImpl) name() string {
	return zone.impl.Name
}

/*
func (s managedZonesServiceImpl) List(project string) *managedZonesListCall {
	var m managedZonesListCall = managedZonesListCallImpl{*s.service.List(project)}
	return &m
}
*/

/*
func (s managedZonesServiceImpl) List() ([]managedZone, error) {
	call := s.service.List(projectName)
	result, err := call.Do()
	if err != nil {
		return make([]managedZone, 0), err
	} else {
		zones := make([]managedZone, len(result.ManagedZones))
		for i, z := range result.ManagedZones {
			zones[i] = managedZoneImpl{*z}
		}
		return zones, nil
	}
}
*/

func (s dnsServiceImpl) Changes() changesService {
	var svc changesService = changesServiceImpl{*s.service.Changes}
	return svc
}

/*
func (s dnsServiceImpl) ManagedZones() managedZonesService {
	var zoneService managedZonesService = managedZonesServiceImpl{*s.service.ManagedZones}
	return zoneService
}
*/

func (s dnsServiceImpl) Projects() projectsService {
	return s.service.Projects
}

/*
func (s dnsServiceImpl) ResourceRecordSets() resourceRecordSetsService {
	return resourceRecordSetsServiceImpl{*s.service.ResourceRecordSets}
}
*/
func (s dnsServiceImpl) resourceRecordSets() resourceRecordSets {
	return resourceRecordSetsImpl{zoneImpl{}, *s.service.ResourceRecordSets} // TODO!! Empty zoneImpl
}

/*
func (m managedZonesListCallImpl) Do(opts ...googleapi.CallOption) (managedZonesListResponse, error) {
	var resp managedZonesListResponse
	r, err := m.impl.Do(opts...)
	resp = managedZonesListResponseImpl{*r}
	return resp, err
}
*/
func NewInterface() iface {
	oauthClient, err := getOauthClient()
	if err != nil {
		glog.Errorf("Failed to get OAuth Client: %v", err)
	}
	service, err := dns.New(oauthClient)
	if err != nil {
		glog.Errorf("Failed to get Cloud DNS client: %v", err)
	}
	glog.Infof("Successfully got DNS service: %v\n", service)
	return ifaceImpl{*service}
}

func (i ifaceImpl) service() dnsService {
	var srvc dnsService = dnsServiceImpl{i._service}
	return srvc
}

func (i ifaceImpl) Zones() (dnsprovider.Zones, bool) {
	return zonesImpl{i}, true
}

func (i ifaceImpl) ManagedZones() (zones, bool) {
	return zonesImpl{i, *i._service.ManagedZones}, true
}

func (i ifaceImpl) Changes() changesService {
	return changesServiceImpl{*i._service.Changes}
}

/*
func (i ifaceImpl) Projects() []string {
	return projectsServiceImpl{*i._service.Projects}
}
*/
/*
func (zones zonesImpl) List() ([]dnsprovider.Zone, error) {
	// response, err := (*(zones.iface().service().ManagedZones().List(projectName))).Do()
	zs, err := zones.iface().service().ManagedZones().List(projectName)
	if err != nil {
		glog.Errorf("Failed to list managed zones for project %s: %v", projectName, err)
		return nil, err
	}
	// zlist := make([]dnsprovider.Zone, len(response.managedZones()))
	zlist := make([]dnsprovider.Zone, len(zs))
	// for i, z := range response.managedZones() {
	for i, z := range zs {
		zlist[i] = zoneImpl{zones, z.dnsName(), z.name()}
	}
	return zlist, nil
}
*/
func (zones zonesImpl) List() ([]dnsprovider.Zone, error) {
	zoneArry, err := zones.list()
	providerZones := make([]dnsprovider.Zone, len(zoneArry))
	for i, z := range zoneArry {
		providerZones[i] = z
	}
	return providerZones, err
}

func (zones zonesImpl) list() ([]zone, error) {
	// response, err := (*(zones.iface().service().ManagedZones().List(projectName))).Do()
	// zs, err := zones.iface().service().ManagedZones().List()
	call := zones.impl.List(projectName)
	response, err := call.Do()
	if err != nil {
		glog.Errorf("Failed to list managed zones for project %s: %v", projectName, err)
		return nil, err
	}
	// zlist := make([]dnsprovider.Zone, len(response.managedZones()))
	zlist := make([]zone, len(response.ManagedZones))
	for i, z := range response.ManagedZones {
		// for i, z := range zs {
		zlist[i] = zoneImpl{zones, *z}
	}
	return zlist, nil
}

func (zones zonesImpl) iface() iface {
	return zones._iface
}

func (zone zoneImpl) Name() string {
	return zone._name
}

func (zone zoneImpl) name() string {
	return zone._name
}

func (zone zoneImpl) id() string {
	return zone._id
}

func (zone zoneImpl) zones() zones {
	return zone._zones
}

func (i zoneImpl) ResourceRecordSets() (dnsprovider.ResourceRecordSets, bool) {
	// var z zone = i
	return resourceRecordSetsImpl{i}, true
}

func (rrs resourceRecordSetsImpl) zone() zone {
	return rrs._zone
}

/*
func (rrs resourceRecordSetsImpl) rrsets() []resourceRecordSet {
	r, err := rrs.List()
	if err != nil {
		// TODO: Deal with it
		return []resourceRecordSet{}
	}

	returnVal := make([]resourceRecordSet, len(r))
	for i, val := range r {
		returnVal[i] = val.(resourceRecordSet)
	}
	return returnVal
}
*/

func (rrs resourceRecordSetsImpl) Add(r dnsprovider.ResourceRecordSet) (dnsprovider.ResourceRecordSet, error) {
	recordSet := resourceRecordSetImpl{_name: r.Name(), _ttl: r.Ttl(), _rrsType: r.Type(), _rrDatas: r.Rrdatas()}
	additions := []dnsprovider.ResourceRecordSet{recordSet}
	ch := rrs.zone().zones().iface().service().Changes()
	err := ch.Do(projectName, rrs._zone._id, additions, []dnsprovider.ResourceRecordSet{})
	if err != nil {
		return nil, err
	}
	glog.Infof("Successfully added record %v to zone %s", r, rrs.zone().Name())
	return recordSet, nil
}

func (rrs resourceRecordSetsImpl) Remove(r dnsprovider.ResourceRecordSet) error { // TODO: Largely a duplicate of Add above.  Refactor!
	var change dns.Change
	recordSet := resourceRecordSetImpl{_name: r.Name(), _ttl: r.Ttl(), _rrsType: r.Type(), _rrDatas: r.Rrdatas()}
	deletions := []dnsprovider.ResourceRecordSet{recordSet}
	ch := rrs.zone().zones().iface().service().Changes()
	err := ch.Do(projectName, rrs._zone._id, []dnsprovider.ResourceRecordSet{}, deletions)
	if err != nil {
		glog.Errorf("Failed to execute change %v on zone %s: %v", change, rrs.zone().Name(), err)
		return err
	}
	glog.Infof("Successfully applied change %v to zone %s", change, rrs.zone().Name())
	// data := []string{r.Rrdatas()[0]}
	return nil
}

func (rrs resourceRecordSetsImpl) New(name string, rrdatas []string, ttl int64, rrstype rrstype.RrsType) dnsprovider.ResourceRecordSet {
	return resourceRecordSetImpl{_rrsets: rrs, _name: name, _rrDatas: rrdatas, _ttl: ttl, _rrsType: rrstype}
}

func (rrs resourceRecordSetsImpl) List() ([]dnsprovider.ResourceRecordSet, error) {
	return rrs.zone().zones().iface().service().ResourceRecordSets().List(projectName, rrs.zone().id() /* rrs.zone.Name() */)
}

func (rrs resourceRecordSetsServiceImpl) List(zone string) ([]dnsprovider.ResourceRecordSet, error) {
	call := rrs.impl.List(projectName, zone)
	response, err := call.Do()
	if err != nil {
		glog.Errorf("Failed to list resource records for domain %s in project %s: %v", zone, projectName, err)
		return nil, err
	}
	result := make([]dnsprovider.ResourceRecordSet, len(response.Rrsets))
	for i, rs := range response.Rrsets {
		result[i] = resourceRecordSetImpl{resourceRecordSetsImpl{}, rs.Name, rs.Ttl, rrstype.RrsType(rs.Type), rs.Rrdatas}
		// TODO!! Use the correct thing for first field!
	}
	return result, nil
}

func (rs resourceRecordSetImpl) Name() string {
	return rs._name
}

/*
func (rs resourceRecordSetImpl) name() string {
	return rs._name
}
*/

func (rs resourceRecordSetImpl) Ttl() int64 {
	return rs._ttl
}

/*
func (rs resourceRecordSetImpl) ttl() int64 {
	return rs._ttl
}
*/

func (rs resourceRecordSetImpl) Type() rrstype.RrsType {
	return rrstype.RrsType(rs._rrsType)
}

/*
func (rs resourceRecordSetImpl) rrsType() rrstype.RrsType {
	return rs.Type()
}
*/

func (rs resourceRecordSetImpl) Rrdatas() []string {
	return rs._rrDatas
}

/*
func (rs resourceRecordSetImpl) rrDatas() []string {
	return rs._rrDatas
}
*/
func (rs resourceRecordSetImpl) ResourceRecordSets() []resourceRecordSet {
	return rs._rrsets.rrsets()
}

func NewResourceRecordSet(rss resourceRecordSets, name string, ttl int64, rrsType rrstype.RrsType, rrDatas []string) resourceRecordSet {
	return resourceRecordSetImpl{rss.(resourceRecordSetsImpl), name, ttl, rrsType, rrDatas}
}

func getOauthClient() (*http.Client, error) {
	tokenSource, err := google.DefaultTokenSource(
		oauth2.NoContext,
		compute.CloudPlatformScope,
		compute.ComputeScope)
	if err != nil {
		return nil, err
	}
	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	if client != nil {
		return client, nil
	} else {
		return nil, fmt.Errorf("Failed to get OAuth client.  No details provided.")
	}
}
