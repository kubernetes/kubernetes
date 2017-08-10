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

// Package coredns is the implementation of pkg/dnsprovider interface for CoreDNS
package coredns

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	etcdc "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/golang/glog"
	"gopkg.in/gcfg.v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
)

// "coredns" should be used to use this DNS provider
const (
	ProviderName = "coredns"
)

// Config to override defaults
type Config struct {
	Global struct {
		EtcdEndpoints    string `gcfg:"etcd-endpoints"`
		CertFile         string `gcfg:"etcd-cert-file"`
		KeyFile          string `gcfg:"etcd-key-file"`
		CAFile           string `gcfg:"etcd-ca-file"`
		DNSZones         string `gcfg:"zones"`
		CoreDNSEndpoints string `gcfg:"coredns-endpoints"`
	}
}

func init() {
	dnsprovider.RegisterDnsProvider(ProviderName, func(config io.Reader) (dnsprovider.Interface, error) {
		return newCoreDNSProviderInterface(config)
	})
}

func newTransportForETCD2(certFile, keyFile, caFile string) (*http.Transport, error) {
	var cfg *tls.Config
	if len(certFile) == 0 && len(keyFile) == 0 && len(caFile) == 0 {
		cfg = nil
	} else {
		info := transport.TLSInfo{
			CertFile: certFile,
			KeyFile:  keyFile,
			CAFile:   caFile,
		}
		var err error
		cfg, err = info.ClientConfig()
		if err != nil {
			return nil, fmt.Errorf("error creating tls config: %v", err)
		}
	}
	// Copied from etcd.DefaultTransport declaration.
	// TODO: Determine if transport needs optimization
	tr := utilnet.SetTransportDefaults(&http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     cfg,
	})
	return tr, nil
}

// newCoreDnsProviderInterface creates a new instance of an CoreDNS DNS Interface.
func newCoreDNSProviderInterface(config io.Reader) (*Interface, error) {
	etcdEndpoints := "http://federation-dns-server-etcd:2379"
	etcdPathPrefix := "skydns"
	dnsZones := ""
	var certFile, keyFile, caFile string

	// Possibly override defaults with config below
	if config != nil {
		var cfg Config
		if err := gcfg.ReadInto(&cfg, config); err != nil {
			glog.Errorf("Couldn't read config: %v", err)
			return nil, err
		}
		etcdEndpoints = cfg.Global.EtcdEndpoints
		dnsZones = cfg.Global.DNSZones
		certFile = cfg.Global.CertFile
		caFile = cfg.Global.CAFile
		keyFile = cfg.Global.KeyFile
	}
	glog.Infof("Using CoreDNS DNS provider")

	if dnsZones == "" {
		return nil, fmt.Errorf("Need to provide at least one DNS Zone")
	}
	glog.Infof("Creating etcd transport with %s, %s, %s", certFile, keyFile, caFile)
	etcdTransport, err := newTransportForETCD2(certFile, keyFile, caFile)
	if err != nil {
		return nil, fmt.Errorf("error creating transport for etcd: %v", err)
	}

	etcdCfg := etcdc.Config{
		Endpoints: strings.Split(etcdEndpoints, ","),
		Transport: etcdTransport,
	}

	c, err := etcdc.New(etcdCfg)
	if err != nil {
		return nil, fmt.Errorf("Create etcd client from the config failed")
	}
	etcdKeysAPI := etcdc.NewKeysAPI(c)

	intf := newInterfaceWithStub(etcdKeysAPI)
	intf.etcdPathPrefix = etcdPathPrefix
	zoneList := strings.Split(dnsZones, ",")

	intf.zones = Zones{intf: intf}
	for index, zoneName := range zoneList {
		zone := Zone{domain: zoneName, id: strconv.Itoa(index), zones: &intf.zones}
		intf.zones.zoneList = append(intf.zones.zoneList, zone)
	}

	return intf, nil
}
