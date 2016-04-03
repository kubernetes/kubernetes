// Copyright 2016 CoreOS, Inc.
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
	"fmt"
	"log"

	"github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

func ExampleLease_create() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	// minimum lease TTL is 5-second
	resp, err := cli.Create(context.TODO(), 5)
	if err != nil {
		log.Fatal(err)
	}

	// after 5 seconds, the key 'foo' will be removed
	_, err = cli.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
	if err != nil {
		log.Fatal(err)
	}
}

func ExampleLease_revoke() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.Create(context.TODO(), 5)
	if err != nil {
		log.Fatal(err)
	}

	_, err = cli.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
	if err != nil {
		log.Fatal(err)
	}

	// revoking lease expires the key attached to its lease ID
	_, err = cli.Revoke(context.TODO(), clientv3.LeaseID(resp.ID))
	if err != nil {
		log.Fatal(err)
	}

	gresp, err := cli.Get(context.TODO(), "foo")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("number of keys:", len(gresp.Kvs))
	// number of keys: 0
}

func ExampleLease_keepAlive() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.Create(context.TODO(), 5)
	if err != nil {
		log.Fatal(err)
	}

	_, err = cli.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
	if err != nil {
		log.Fatal(err)
	}

	// the key 'foo' will be kept forever
	_, err = cli.KeepAlive(context.TODO(), clientv3.LeaseID(resp.ID))
	if err != nil {
		log.Fatal(err)
	}
}

func ExampleLease_keepAliveOnce() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.Create(context.TODO(), 5)
	if err != nil {
		log.Fatal(err)
	}

	_, err = cli.Put(context.TODO(), "foo", "bar", clientv3.WithLease(clientv3.LeaseID(resp.ID)))
	if err != nil {
		log.Fatal(err)
	}

	// to renew the lease only once
	_, err = cli.KeepAliveOnce(context.TODO(), clientv3.LeaseID(resp.ID))
	if err != nil {
		log.Fatal(err)
	}
}
