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

package command

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	v3 "github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	spb "github.com/coreos/etcd/storage/storagepb"
	"github.com/olekukonko/tablewriter"
)

type printer interface {
	Del(v3.DeleteResponse)
	Get(v3.GetResponse)
	Put(v3.PutResponse)
	Txn(v3.TxnResponse)
	Watch(v3.WatchResponse)

	MemberList(v3.MemberListResponse)
}

func NewPrinter(printerType string, isHex bool) printer {
	switch printerType {
	case "simple":
		return &simplePrinter{isHex: isHex}
	case "json":
		return &jsonPrinter{}
	case "protobuf":
		return &pbPrinter{}
	}
	return nil
}

type simplePrinter struct {
	isHex bool
}

func (s *simplePrinter) Del(v3.DeleteResponse) {
	// TODO: add number of key removed into the response of delete.
	// TODO: print out the number of removed keys.
	fmt.Println(0)
}

func (s *simplePrinter) Get(resp v3.GetResponse) {
	for _, kv := range resp.Kvs {
		printKV(s.isHex, kv)
	}
}

func (s *simplePrinter) Put(r v3.PutResponse) { fmt.Println("OK") }

func (s *simplePrinter) Txn(resp v3.TxnResponse) {
	if resp.Succeeded {
		fmt.Println("SUCCESS")
	} else {
		fmt.Println("FAILURE")
	}

	for _, r := range resp.Responses {
		fmt.Println("")
		switch v := r.Response.(type) {
		case *pb.ResponseUnion_ResponseDeleteRange:
			s.Del((v3.DeleteResponse)(*v.ResponseDeleteRange))
		case *pb.ResponseUnion_ResponsePut:
			s.Put((v3.PutResponse)(*v.ResponsePut))
		case *pb.ResponseUnion_ResponseRange:
			s.Get(((v3.GetResponse)(*v.ResponseRange)))
		default:
			fmt.Printf("unexpected response %+v\n", r)
		}
	}
}

func (s *simplePrinter) Watch(resp v3.WatchResponse) {
	for _, e := range resp.Events {
		fmt.Println(e.Type)
		printKV(s.isHex, e.Kv)
	}
}

func (s *simplePrinter) MemberList(resp v3.MemberListResponse) {
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"ID", "Status", "Name", "Peer Addrs", "Client Addrs", "Is Leader"})

	for _, m := range resp.Members {
		status := "started"
		if len(m.Name) == 0 {
			status = "unstarted"
		}

		table.Append([]string{
			fmt.Sprintf("%x", m.ID),
			status,
			m.Name,
			strings.Join(m.PeerURLs, ","),
			strings.Join(m.ClientURLs, ","),
			fmt.Sprint(m.IsLeader),
		})
	}

	table.Render()
}

type jsonPrinter struct{}

func (p *jsonPrinter) Del(r v3.DeleteResponse) { printJSON(r) }
func (p *jsonPrinter) Get(r v3.GetResponse) {
	for _, kv := range r.Kvs {
		printJSON(kv)
	}
}
func (p *jsonPrinter) Put(r v3.PutResponse)               { printJSON(r) }
func (p *jsonPrinter) Txn(r v3.TxnResponse)               { printJSON(r) }
func (p *jsonPrinter) Watch(r v3.WatchResponse)           { printJSON(r) }
func (p *jsonPrinter) MemberList(r v3.MemberListResponse) { printJSON(r) }

func printJSON(v interface{}) {
	b, err := json.Marshal(v)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return
	}
	fmt.Println(string(b))
}

type pbPrinter struct{}

type pbMarshal interface {
	Marshal() ([]byte, error)
}

func (p *pbPrinter) Del(r v3.DeleteResponse) {
	printPB((*pb.DeleteRangeResponse)(&r))
}

func (p *pbPrinter) Get(r v3.GetResponse) {
	printPB((*pb.RangeResponse)(&r))
}

func (p *pbPrinter) Put(r v3.PutResponse) {
	printPB((*pb.PutResponse)(&r))
}

func (p *pbPrinter) Txn(r v3.TxnResponse) {
	printPB((*pb.TxnResponse)(&r))
}

func (p *pbPrinter) Watch(r v3.WatchResponse) {
	for _, ev := range r.Events {
		printPB((*spb.Event)(ev))
	}
}

func (pb *pbPrinter) MemberList(r v3.MemberListResponse) {
	ExitWithError(ExitBadFeature, errors.New("only support simple or json as output format"))
}

func printPB(m pbMarshal) {
	b, err := m.Marshal()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		return
	}
	fmt.Printf(string(b))
}
