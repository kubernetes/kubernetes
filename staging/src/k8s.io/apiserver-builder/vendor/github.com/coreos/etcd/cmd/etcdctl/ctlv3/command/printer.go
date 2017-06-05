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

package command

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"

	v3 "github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	spb "github.com/coreos/etcd/mvcc/mvccpb"
	"github.com/dustin/go-humanize"
	"github.com/olekukonko/tablewriter"
)

type printer interface {
	Del(v3.DeleteResponse)
	Get(v3.GetResponse)
	Put(v3.PutResponse)
	Txn(v3.TxnResponse)
	Watch(v3.WatchResponse)

	MemberList(v3.MemberListResponse)

	EndpointStatus([]epStatus)

	Alarm(v3.AlarmResponse)
	DBStatus(dbstatus)
}

func NewPrinter(printerType string, isHex bool) printer {
	switch printerType {
	case "simple":
		return &simplePrinter{isHex: isHex}
	case "json":
		return &jsonPrinter{}
	case "protobuf":
		return &pbPrinter{}
	case "table":
		return &tablePrinter{}
	}
	return nil
}

func makeMemberListTable(r v3.MemberListResponse) (hdr []string, rows [][]string) {
	hdr = []string{"ID", "Status", "Name", "Peer Addrs", "Client Addrs"}
	for _, m := range r.Members {
		status := "started"
		if len(m.Name) == 0 {
			status = "unstarted"
		}
		rows = append(rows, []string{
			fmt.Sprintf("%x", m.ID),
			status,
			m.Name,
			strings.Join(m.PeerURLs, ","),
			strings.Join(m.ClientURLs, ","),
		})
	}
	return
}

func makeEndpointStatusTable(statusList []epStatus) (hdr []string, rows [][]string) {
	hdr = []string{"endpoint", "ID", "version", "db size", "is leader", "raft term", "raft index"}
	for _, status := range statusList {
		rows = append(rows, []string{
			fmt.Sprint(status.Ep),
			fmt.Sprintf("%x", status.Resp.Header.MemberId),
			fmt.Sprint(status.Resp.Version),
			fmt.Sprint(humanize.Bytes(uint64(status.Resp.DbSize))),
			fmt.Sprint(status.Resp.Leader == status.Resp.Header.MemberId),
			fmt.Sprint(status.Resp.RaftTerm),
			fmt.Sprint(status.Resp.RaftIndex),
		})
	}
	return
}

func makeDBStatusTable(ds dbstatus) (hdr []string, rows [][]string) {
	hdr = []string{"hash", "revision", "total keys", "total size"}
	rows = append(rows, []string{
		fmt.Sprintf("%x", ds.Hash),
		fmt.Sprint(ds.Revision),
		fmt.Sprint(ds.TotalKey),
		humanize.Bytes(uint64(ds.TotalSize)),
	})
	return
}

type simplePrinter struct {
	isHex bool
}

func (s *simplePrinter) Del(resp v3.DeleteResponse) {
	fmt.Println(resp.Deleted)
	for _, kv := range resp.PrevKvs {
		printKV(s.isHex, kv)
	}
}

func (s *simplePrinter) Get(resp v3.GetResponse) {
	for _, kv := range resp.Kvs {
		printKV(s.isHex, kv)
	}
}

func (s *simplePrinter) Put(r v3.PutResponse) {
	fmt.Println("OK")
	if r.PrevKv != nil {
		printKV(s.isHex, r.PrevKv)
	}
}

func (s *simplePrinter) Txn(resp v3.TxnResponse) {
	if resp.Succeeded {
		fmt.Println("SUCCESS")
	} else {
		fmt.Println("FAILURE")
	}

	for _, r := range resp.Responses {
		fmt.Println("")
		switch v := r.Response.(type) {
		case *pb.ResponseOp_ResponseDeleteRange:
			s.Del((v3.DeleteResponse)(*v.ResponseDeleteRange))
		case *pb.ResponseOp_ResponsePut:
			s.Put((v3.PutResponse)(*v.ResponsePut))
		case *pb.ResponseOp_ResponseRange:
			s.Get(((v3.GetResponse)(*v.ResponseRange)))
		default:
			fmt.Printf("unexpected response %+v\n", r)
		}
	}
}

func (s *simplePrinter) Watch(resp v3.WatchResponse) {
	for _, e := range resp.Events {
		fmt.Println(e.Type)
		if e.PrevKv != nil {
			printKV(s.isHex, e.PrevKv)
		}
		printKV(s.isHex, e.Kv)
	}
}

func (s *simplePrinter) Alarm(resp v3.AlarmResponse) {
	for _, e := range resp.Alarms {
		fmt.Printf("%+v\n", e)
	}
}

func (s *simplePrinter) MemberList(resp v3.MemberListResponse) {
	_, rows := makeMemberListTable(resp)
	for _, row := range rows {
		fmt.Println(strings.Join(row, ", "))
	}
}

func (s *simplePrinter) EndpointStatus(statusList []epStatus) {
	_, rows := makeEndpointStatusTable(statusList)
	for _, row := range rows {
		fmt.Println(strings.Join(row, ", "))
	}
}

func (s *simplePrinter) DBStatus(ds dbstatus) {
	_, rows := makeDBStatusTable(ds)
	for _, row := range rows {
		fmt.Println(strings.Join(row, ", "))
	}
}

type tablePrinter struct{}

func (tp *tablePrinter) Del(r v3.DeleteResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) Get(r v3.GetResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) Put(r v3.PutResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) Txn(r v3.TxnResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) Watch(r v3.WatchResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) Alarm(r v3.AlarmResponse) {
	ExitWithError(ExitBadFeature, errors.New("table is not supported as output format"))
}
func (tp *tablePrinter) MemberList(r v3.MemberListResponse) {
	hdr, rows := makeMemberListTable(r)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(hdr)
	for _, row := range rows {
		table.Append(row)
	}
	table.Render()
}
func (tp *tablePrinter) EndpointStatus(r []epStatus) {
	hdr, rows := makeEndpointStatusTable(r)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(hdr)
	for _, row := range rows {
		table.Append(row)
	}
	table.Render()
}
func (tp *tablePrinter) DBStatus(r dbstatus) {
	hdr, rows := makeDBStatusTable(r)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader(hdr)
	for _, row := range rows {
		table.Append(row)
	}
	table.Render()
}

type jsonPrinter struct{}

func (p *jsonPrinter) Del(r v3.DeleteResponse)            { printJSON(r) }
func (p *jsonPrinter) Get(r v3.GetResponse)               { printJSON(r) }
func (p *jsonPrinter) Put(r v3.PutResponse)               { printJSON(r) }
func (p *jsonPrinter) Txn(r v3.TxnResponse)               { printJSON(r) }
func (p *jsonPrinter) Watch(r v3.WatchResponse)           { printJSON(r) }
func (p *jsonPrinter) Alarm(r v3.AlarmResponse)           { printJSON(r) }
func (p *jsonPrinter) MemberList(r v3.MemberListResponse) { printJSON(r) }
func (p *jsonPrinter) EndpointStatus(r []epStatus)        { printJSON(r) }
func (p *jsonPrinter) DBStatus(r dbstatus)                { printJSON(r) }

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

func (p *pbPrinter) Alarm(r v3.AlarmResponse) {
	printPB((*pb.AlarmResponse)(&r))
}

func (p *pbPrinter) MemberList(r v3.MemberListResponse) {
	printPB((*pb.MemberListResponse)(&r))
}

func (p *pbPrinter) EndpointStatus(statusList []epStatus) {
	ExitWithError(ExitBadFeature, errors.New("only support simple or json as output format"))
}

func (p *pbPrinter) DBStatus(r dbstatus) {
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
