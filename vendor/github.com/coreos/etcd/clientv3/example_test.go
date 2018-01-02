// Copyright 2016 The etcd Authors
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

package clientv3_test

import (
	"log"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/net/context"
)

var (
	dialTimeout    = 5 * time.Second
	requestTimeout = 1 * time.Second
	endpoints      = []string{"localhost:2379", "localhost:22379", "localhost:32379"}
)

func Example() {
	var plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "clientv3")
	clientv3.SetLogger(plog)

	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close() // make sure to close the client

	_, err = cli.Put(context.TODO(), "foo", "bar")
	if err != nil {
		log.Fatal(err)
	}
}

func ExampleConfig_withTLS() {
	tlsInfo := transport.TLSInfo{
		CertFile:      "/tmp/test-certs/test-name-1.pem",
		KeyFile:       "/tmp/test-certs/test-name-1-key.pem",
		TrustedCAFile: "/tmp/test-certs/trusted-ca.pem",
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		log.Fatal(err)
	}
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
		TLS:         tlsConfig,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close() // make sure to close the client

	_, err = cli.Put(context.TODO(), "foo", "bar")
	if err != nil {
		log.Fatal(err)
	}
}
