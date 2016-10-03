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
	dnsmsg "github.com/miekg/coredns/middleware/etcd/msg"
	"golang.org/x/net/context"
	"hash/fnv"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// Compile time check for interface adherence
var _ dnsprovider.ResourceRecordChangeset = &ResourceRecordChangeset{}

type ChangeSetType string

const (
	ADDITION = ChangeSetType("ADDITION")
	DELETION = ChangeSetType("DELETION")
)

type ChangeSet struct {
	cstype ChangeSetType
	rrset  dnsprovider.ResourceRecordSet
}

type ResourceRecordChangeset struct {
	zone   *Zone
	rrsets *ResourceRecordSets

	changeset []ChangeSet
}

func (c *ResourceRecordChangeset) Add(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.changeset = append(c.changeset, ChangeSet{cstype: ADDITION, rrset: rrset})
	return c
}

func (c *ResourceRecordChangeset) Remove(rrset dnsprovider.ResourceRecordSet) dnsprovider.ResourceRecordChangeset {
	c.changeset = append(c.changeset, ChangeSet{cstype: DELETION, rrset: rrset})
	return c
}

func (c *ResourceRecordChangeset) Apply() error {
	ctx := context.Background()
	etcdPathPrefix := c.zone.zones.intf.etcdPathPrefix
	getOpts := &etcdc.GetOptions{}
	setOpts := &etcdc.SetOptions{}
	deleteOpts := &etcdc.DeleteOptions{
		Recursive: true,
	}

	for _, changeset := range c.changeset {
		switch changeset.cstype {
		case ADDITION:
			for _, rrdata := range changeset.rrset.Rrdatas() {
				b, err := json.Marshal(&dnsmsg.Service{Host: rrdata, TTL: uint32(changeset.rrset.Ttl()), Group: changeset.rrset.Name()})
				if err != nil {
					return err
				}
				recordValue := string(b)
				recordLabel := getHash(rrdata)
				recordKey := buildDNSNameString(changeset.rrset.Name(), recordLabel)

				response, err := c.zone.zones.intf.etcdKeysAPI.Get(ctx, dnsmsg.Path(recordKey, etcdPathPrefix), getOpts)
				if err == nil && response != nil {
					return fmt.Errorf("Key already exist, key: %v", recordKey)
				}

				_, err = c.zone.zones.intf.etcdKeysAPI.Set(ctx, dnsmsg.Path(recordKey, etcdPathPrefix), recordValue, setOpts)
				if err != nil {
					return err
				}
			}
		case DELETION:
			for _, rrdata := range changeset.rrset.Rrdatas() {
				recordLabel := getHash(rrdata)
				recordKey := buildDNSNameString(changeset.rrset.Name(), recordLabel)
				_, err := c.zone.zones.intf.etcdKeysAPI.Delete(ctx, dnsmsg.Path(recordKey, etcdPathPrefix), deleteOpts)
				if err != nil {
					return err
				}
			}
			// TODO: We need to cleanup empty dirs in etcd
		}
	}
	return nil
}

func getHash(text string) string {
	h := fnv.New32a()
	h.Write([]byte(text))
	return fmt.Sprintf("%x", h.Sum32())
}

func buildDNSNameString(labels ...string) string {
	var res string
	for _, label := range labels {
		if res == "" {
			res = label
		} else {
			res = fmt.Sprintf("%s.%s", label, res)
		}
	}
	return res
}
