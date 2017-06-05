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

	"github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

func ExampleCluster_memberList() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.MemberList(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("members:", len(resp.Members))
	// Output: members: 3
}

func ExampleCluster_memberAdd() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints[:2],
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	peerURLs := endpoints[2:]
	mresp, err := cli.MemberAdd(context.Background(), peerURLs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("added member.PeerURLs:", mresp.Member.PeerURLs)
	// added member.PeerURLs: [http://localhost:32380]
}

func ExampleCluster_memberRemove() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints[1:],
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.MemberList(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	_, err = cli.MemberRemove(context.Background(), resp.Members[0].ID)
	if err != nil {
		log.Fatal(err)
	}
}

func ExampleCluster_memberUpdate() {
	cli, err := clientv3.New(clientv3.Config{
		Endpoints:   endpoints,
		DialTimeout: dialTimeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer cli.Close()

	resp, err := cli.MemberList(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	peerURLs := []string{"http://localhost:12380"}
	_, err = cli.MemberUpdate(context.Background(), resp.Members[0].ID, peerURLs)
	if err != nil {
		log.Fatal(err)
	}
}
