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

package main

import (
	"flag"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

var (
	etcdAddress   = flag.String("etcd-address", "", "Etcd address")
	ttlKeysPrefix = flag.String("ttl-keys-prefix", "", "Prefix for TTL keys")
	leaseDuration = flag.Duration("lease-duration", time.Hour, "Lease duration (seconds granularity)")
)

func main() {
	flag.Parse()

	if *etcdAddress == "" {
		glog.Fatalf("--etcd-address flag is required")
	}
	client, err := clientv3.New(clientv3.Config{Endpoints: []string{*etcdAddress}})
	if err != nil {
		glog.Fatalf("Error while creating etcd client: %v", err)
	}

	// Make sure that ttlKeysPrefix is ended with "/" so that we only get children "directories".
	if !strings.HasSuffix(*ttlKeysPrefix, "/") {
		*ttlKeysPrefix += "/"
	}
	ctx := context.Background()

	objectsResp, err := client.KV.Get(ctx, *ttlKeysPrefix, clientv3.WithPrefix())
	if err != nil {
		glog.Fatalf("Error while getting objects to attach to the lease")
	}

	lease, err := client.Lease.Grant(ctx, int64(*leaseDuration/time.Second))
	if err != nil {
		glog.Fatalf("Error while creating lease: %v", err)
	}
	glog.Infof("Lease with TTL: %v created", lease.TTL)

	glog.Infof("Attaching lease to %d entries", len(objectsResp.Kvs))
	for _, kv := range objectsResp.Kvs {
		_, err := client.KV.Put(ctx, string(kv.Key), string(kv.Value), clientv3.WithLease(lease.ID))
		if err != nil {
			glog.Errorf("Error while attaching lease to: %s", string(kv.Key))
		}
	}
}
