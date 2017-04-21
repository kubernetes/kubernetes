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
	"fmt"
	"strings"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

type etcdFlags struct {
	etcdAddress   *string
	ttlKeysPrefix *string
	leaseDuration *time.Duration
	certFile      *string
	keyFile       *string
	caFile        *string
}

func generateClientConfig(flags *etcdFlags) (*clientv3.Config, error) {
	if *(flags.etcdAddress) == "" {
		return nil, fmt.Errorf("--etcd-address flag is required")
	}

	c := &clientv3.Config{
		Endpoints: []string{*(flags.etcdAddress)},
	}

	var cfgtls *transport.TLSInfo
	tlsinfo := transport.TLSInfo{}
	if *(flags.certFile) != "" {
		tlsinfo.CertFile = *(flags.certFile)
		cfgtls = &tlsinfo
	}

	if *(flags.keyFile) != "" {
		tlsinfo.KeyFile = *(flags.keyFile)
		cfgtls = &tlsinfo
	}

	if *(flags.caFile) != "" {
		tlsinfo.CAFile = *(flags.caFile)
		cfgtls = &tlsinfo
	}

	if cfgtls != nil {
		clientTLS, err := cfgtls.ClientConfig()
		if err != nil {
			return nil, fmt.Errorf("Error while creating etcd client: %v", err)
		}
		c.TLS = clientTLS
	}
	return c, nil
}

func main() {

	flags := &etcdFlags{
		etcdAddress:   flag.String("etcd-address", "", "Etcd address"),
		ttlKeysPrefix: flag.String("ttl-keys-prefix", "", "Prefix for TTL keys"),
		leaseDuration: flag.Duration("lease-duration", time.Hour, "Lease duration (seconds granularity)"),
		certFile:      flag.String("cert", "", "identify secure client using this TLS certificate file"),
		keyFile:       flag.String("key", "", "identify secure client using this TLS key file"),
		caFile:        flag.String("cacert", "", "verify certificates of TLS-enabled secure servers using this CA bundle"),
	}

	flag.Parse()

	c, err := generateClientConfig(flags)
	if err != nil {
		glog.Fatalf(err.Error())
	}

	client, err := clientv3.New(*c)
	if err != nil {
		glog.Fatalf("Error while creating etcd client: %v", err)
	}

	// Make sure that ttlKeysPrefix is ended with "/" so that we only get children "directories".
	if !strings.HasSuffix(*(flags.ttlKeysPrefix), "/") {
		*(flags.ttlKeysPrefix) += "/"
	}
	ctx := context.Background()

	objectsResp, err := client.KV.Get(ctx, *(flags.ttlKeysPrefix), clientv3.WithPrefix())
	if err != nil {
		glog.Fatalf("Error while getting objects to attach to the lease")
	}

	lease, err := client.Lease.Grant(ctx, int64(*(flags.leaseDuration)/time.Second))
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
