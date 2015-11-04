// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package subnet

import (
	"fmt"
	"log"
	"path"
	"sync"
	"time"

	etcd "github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/client"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
)

type Registry interface {
	getNetworkConfig(ctx context.Context, network string) (*etcd.Response, error)
	getSubnets(ctx context.Context, network string) (*etcd.Response, error)
	createSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error)
	updateSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error)
	deleteSubnet(ctx context.Context, network, sn string) (*etcd.Response, error)
	watch(ctx context.Context, path string, since uint64) (*etcd.Response, error)
	getNetworks(ctx context.Context) (*etcd.Response, error)
}

type EtcdConfig struct {
	Endpoints []string
	Keyfile   string
	Certfile  string
	CAFile    string
	Prefix    string
}

type etcdSubnetRegistry struct {
	mux     sync.Mutex
	cli     etcd.KeysAPI
	etcdCfg *EtcdConfig
}

func newEtcdClient(c *EtcdConfig) (etcd.KeysAPI, error) {
	tlsInfo := transport.TLSInfo{
		CertFile: c.Certfile,
		KeyFile:  c.Keyfile,
		CAFile:   c.CAFile,
	}

	t, err := transport.NewTransport(tlsInfo)
	if err != nil {
		return nil, err
	}

	cli, err := etcd.New(etcd.Config{
		Endpoints: c.Endpoints,
		Transport: t,
	})
	if err != nil {
		return nil, err
	}

	return etcd.NewKeysAPI(cli), nil
}

func newEtcdSubnetRegistry(config *EtcdConfig) (Registry, error) {
	r := &etcdSubnetRegistry{
		etcdCfg: config,
	}

	var err error
	r.cli, err = newEtcdClient(config)
	if err != nil {
		return nil, err
	}

	return r, nil
}

func (esr *etcdSubnetRegistry) getNetworkConfig(ctx context.Context, network string) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, network, "config")
	resp, err := esr.client().Get(ctx, key, nil)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (esr *etcdSubnetRegistry) getSubnets(ctx context.Context, network string) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, network, "subnets")
	return esr.client().Get(ctx, key, &etcd.GetOptions{Recursive: true})
}

func (esr *etcdSubnetRegistry) createSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, network, "subnets", sn)
	opts := &etcd.SetOptions{
		PrevExist: etcd.PrevNoExist,
		TTL:       ttl,
	}
	resp, err := esr.client().Set(ctx, key, data, opts)
	if err != nil {
		return nil, err
	}

	ensureExpiration(resp, ttl)
	return resp, nil
}

func (esr *etcdSubnetRegistry) updateSubnet(ctx context.Context, network, sn, data string, ttl time.Duration) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, network, "subnets", sn)
	resp, err := esr.client().Set(ctx, key, data, &etcd.SetOptions{TTL: ttl})
	if err != nil {
		return nil, err
	}

	ensureExpiration(resp, ttl)
	return resp, nil
}

func (esr *etcdSubnetRegistry) deleteSubnet(ctx context.Context, network, sn string) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, network, "subnets", sn)
	return esr.client().Delete(ctx, key, nil)
}

func (esr *etcdSubnetRegistry) watch(ctx context.Context, subpath string, since uint64) (*etcd.Response, error) {
	key := path.Join(esr.etcdCfg.Prefix, subpath)
	opts := &etcd.WatcherOptions{
		AfterIndex: since,
		Recursive:  true,
	}
	e, err := esr.client().Watcher(key, opts).Next(ctx)
	return e, err
}

func (esr *etcdSubnetRegistry) getNetworks(ctx context.Context) (*etcd.Response, error) {
	return esr.client().Get(ctx, esr.etcdCfg.Prefix, &etcd.GetOptions{Recursive: true})
}

func (esr *etcdSubnetRegistry) client() etcd.KeysAPI {
	esr.mux.Lock()
	defer esr.mux.Unlock()
	return esr.cli
}

func (esr *etcdSubnetRegistry) resetClient() {
	esr.mux.Lock()
	defer esr.mux.Unlock()

	var err error
	esr.cli, err = newEtcdClient(esr.etcdCfg)
	if err != nil {
		panic(fmt.Errorf("resetClient: error recreating etcd client: %v", err))
	}
}

func ensureExpiration(resp *etcd.Response, ttl time.Duration) {
	if resp.Node.Expiration == nil {
		// should not be but calc it ourselves in this case
		log.Printf("Expiration field missing on etcd response, calculating locally")
		exp := time.Now().Add(time.Duration(ttl) * time.Second)
		resp.Node.Expiration = &exp
	}
}
