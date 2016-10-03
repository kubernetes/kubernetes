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
	"encoding/json"
	"fmt"
	etcdc "github.com/coreos/etcd/client"
	"github.com/golang/glog"
	dnsmsg "github.com/miekg/coredns/middleware/etcd/msg"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
	"net"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordSets = ResourceRecordSets{}

type ResourceRecordSets struct {
	zone *Zone
}

func (rrsets ResourceRecordSets) List() ([]dnsprovider.ResourceRecordSet, error) {
	var list []dnsprovider.ResourceRecordSet
	return list, fmt.Errorf("OperationNotSupported")
}

func (rrsets ResourceRecordSets) Get(name string) (dnsprovider.ResourceRecordSet, error) {
	getOpts := &etcdc.GetOptions{
		Recursive: true,
	}
	etcdPathPrefix := rrsets.zone.zones.intf.etcdPathPrefix
	response, err := rrsets.zone.zones.intf.etcdKeysAPI.Get(context.Background(), dnsmsg.Path(name, etcdPathPrefix), getOpts)
	if err != nil {
		if etcdc.IsKeyNotFound(err) {
			glog.V(2).Infof("Subdomain %q does not exist", name)
			return nil, nil
		}
		return nil, fmt.Errorf("Failed to get service from etcd, err: %v", err)
	}
	if emptyResponse(response) {
		glog.V(2).Infof("Subdomain %q does not exist in etcd", name)
		return nil, nil
	}

	rrset := ResourceRecordSet{name: name, rrdatas: []string{}, rrsets: &rrsets}
	found := false
	for _, node := range response.Node.Nodes {
		found = true
		service := dnsmsg.Service{}
		err = json.Unmarshal([]byte(node.Value), &service)
		if err != nil {
			return nil, fmt.Errorf("Failed to unmarshall json data, err: %v", err)
		}

		// assuming all rrdatas in a rrset will have same type
		ip := net.ParseIP(service.Host)
		switch {
		case ip == nil:
			rrset.rrsType = rrstype.CNAME
		case ip.To4() != nil:
			rrset.rrsType = rrstype.A
		}
		rrset.rrdatas = append(rrset.rrdatas, service.Host)
		rrset.ttl = int64(service.TTL)
	}

	if !found {
		return nil, nil
	}

	return rrset, nil
}

func (rrsets ResourceRecordSets) StartChangeset() dnsprovider.ResourceRecordChangeset {
	return &ResourceRecordChangeset{
		zone:   rrsets.zone,
		rrsets: &rrsets,
	}
}

func (rrsets ResourceRecordSets) New(name string, rrdatas []string, ttl int64, rrsType rrstype.RrsType) dnsprovider.ResourceRecordSet {
	return ResourceRecordSet{
		name:    name,
		rrdatas: rrdatas,
		ttl:     ttl,
		rrsType: rrsType,
		rrsets:  &rrsets,
	}
}

func emptyResponse(resp *etcdc.Response) bool {
	return resp == nil || resp.Node == nil || (len(resp.Node.Value) == 0 && len(resp.Node.Nodes) == 0)
}
