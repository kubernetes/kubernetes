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
	"fmt"
	"log"

	"golang.org/x/net/context"

	"github.com/coreos/etcd/clientv3"
)

func ExampleMaintenance_status() {
	for _, ep := range endpoints {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   []string{ep},
			DialTimeout: dialTimeout,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer cli.Close()

		// resp, err := cli.Status(context.Background(), ep)
		//
		// or
		//
		mapi := clientv3.NewMaintenance(cli)
		resp, err := mapi.Status(context.Background(), ep)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("endpoint: %s / IsLeader: %v\n", ep, resp.Header.MemberId == resp.Leader)
	}
	// endpoint: localhost:2379 / IsLeader: false
	// endpoint: localhost:22379 / IsLeader: false
	// endpoint: localhost:32379 / IsLeader: true
}

func ExampleMaintenance_defragment() {
	for _, ep := range endpoints {
		cli, err := clientv3.New(clientv3.Config{
			Endpoints:   []string{ep},
			DialTimeout: dialTimeout,
		})
		if err != nil {
			log.Fatal(err)
		}
		defer cli.Close()

		if _, err = cli.Defragment(context.TODO(), ep); err != nil {
			log.Fatal(err)
		}
	}
}
