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

package factory

import (
	"net"
	"net/http"
	"time"

	etcd2client "github.com/coreos/etcd/client"
	"github.com/coreos/etcd/pkg/transport"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func newETCD2Storage(c storagebackend.Config) (storage.Interface, DestroyFunc, error) {
	tr, err := newTransportForETCD2(c.CertFile, c.KeyFile, c.CAFile)
	if err != nil {
		return nil, nil, err
	}
	client, err := newETCD2Client(tr, c.ServerList)
	if err != nil {
		return nil, nil, err
	}
	s := etcd.NewEtcdStorage(client, c.Codec, c.Prefix, c.Quorum, c.DeserializationCacheSize, c.Copier, etcd.IdentityTransformer)
	return s, tr.CloseIdleConnections, nil
}

func newETCD2Client(tr *http.Transport, serverList []string) (etcd2client.Client, error) {
	cli, err := etcd2client.New(etcd2client.Config{
		Endpoints: serverList,
		Transport: tr,
	})
	if err != nil {
		return nil, err
	}

	return cli, nil
}

func newTransportForETCD2(certFile, keyFile, caFile string) (*http.Transport, error) {
	info := transport.TLSInfo{
		CertFile: certFile,
		KeyFile:  keyFile,
		CAFile:   caFile,
	}
	cfg, err := info.ClientConfig()
	if err != nil {
		return nil, err
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
		MaxIdleConnsPerHost: 500,
		TLSClientConfig:     cfg,
	})
	return tr, nil
}
