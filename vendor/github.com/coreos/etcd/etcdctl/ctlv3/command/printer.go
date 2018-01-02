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
	"errors"
	"fmt"
	"strings"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/dustin/go-humanize"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type printer interface {
	Del(v3.DeleteResponse)
	Get(v3.GetResponse)
	Put(v3.PutResponse)
	Txn(v3.TxnResponse)
	Watch(v3.WatchResponse)

	Grant(r v3.LeaseGrantResponse)
	Revoke(id v3.LeaseID, r v3.LeaseRevokeResponse)
	KeepAlive(r v3.LeaseKeepAliveResponse)
	TimeToLive(r v3.LeaseTimeToLiveResponse, keys bool)

	MemberAdd(v3.MemberAddResponse)
	MemberRemove(id uint64, r v3.MemberRemoveResponse)
	MemberUpdate(id uint64, r v3.MemberUpdateResponse)
	MemberList(v3.MemberListResponse)

	EndpointStatus([]epStatus)

	Alarm(v3.AlarmResponse)
	DBStatus(dbstatus)

	RoleAdd(role string, r v3.AuthRoleAddResponse)
	RoleGet(role string, r v3.AuthRoleGetResponse)
	RoleDelete(role string, r v3.AuthRoleDeleteResponse)
	RoleList(v3.AuthRoleListResponse)
	RoleGrantPermission(role string, r v3.AuthRoleGrantPermissionResponse)
	RoleRevokePermission(role string, key string, end string, r v3.AuthRoleRevokePermissionResponse)

	UserAdd(user string, r v3.AuthUserAddResponse)
	UserGet(user string, r v3.AuthUserGetResponse)
	UserList(r v3.AuthUserListResponse)
	UserChangePassword(v3.AuthUserChangePasswordResponse)
	UserGrantRole(user string, role string, r v3.AuthUserGrantRoleResponse)
	UserRevokeRole(user string, role string, r v3.AuthUserRevokeRoleResponse)
	UserDelete(user string, r v3.AuthUserDeleteResponse)
}

func NewPrinter(printerType string, isHex bool) printer {
	switch printerType {
	case "simple":
		return &simplePrinter{isHex: isHex}
	case "fields":
		return &fieldsPrinter{newPrinterUnsupported("fields")}
	case "json":
		return newJSONPrinter()
	case "protobuf":
		return newPBPrinter()
	case "table":
		return &tablePrinter{newPrinterUnsupported("table")}
	}
	return nil
}

type printerRPC struct {
	printer
	p func(interface{})
}

func (p *printerRPC) Del(r v3.DeleteResponse)  { p.p((*pb.DeleteRangeResponse)(&r)) }
func (p *printerRPC) Get(r v3.GetResponse)     { p.p((*pb.RangeResponse)(&r)) }
func (p *printerRPC) Put(r v3.PutResponse)     { p.p((*pb.PutResponse)(&r)) }
func (p *printerRPC) Txn(r v3.TxnResponse)     { p.p((*pb.TxnResponse)(&r)) }
func (p *printerRPC) Watch(r v3.WatchResponse) { p.p(&r) }

func (p *printerRPC) Grant(r v3.LeaseGrantResponse)                      { p.p(r) }
func (p *printerRPC) Revoke(id v3.LeaseID, r v3.LeaseRevokeResponse)     { p.p(r) }
func (p *printerRPC) KeepAlive(r v3.LeaseKeepAliveResponse)              { p.p(r) }
func (p *printerRPC) TimeToLive(r v3.LeaseTimeToLiveResponse, keys bool) { p.p(&r) }

func (p *printerRPC) MemberAdd(r v3.MemberAddResponse) { p.p((*pb.MemberAddResponse)(&r)) }
func (p *printerRPC) MemberRemove(id uint64, r v3.MemberRemoveResponse) {
	p.p((*pb.MemberRemoveResponse)(&r))
}
func (p *printerRPC) MemberUpdate(id uint64, r v3.MemberUpdateResponse) {
	p.p((*pb.MemberUpdateResponse)(&r))
}
func (p *printerRPC) MemberList(r v3.MemberListResponse) { p.p((*pb.MemberListResponse)(&r)) }
func (p *printerRPC) Alarm(r v3.AlarmResponse)           { p.p((*pb.AlarmResponse)(&r)) }

func (p *printerRPC) RoleAdd(_ string, r v3.AuthRoleAddResponse) { p.p((*pb.AuthRoleAddResponse)(&r)) }
func (p *printerRPC) RoleGet(_ string, r v3.AuthRoleGetResponse) { p.p((*pb.AuthRoleGetResponse)(&r)) }
func (p *printerRPC) RoleDelete(_ string, r v3.AuthRoleDeleteResponse) {
	p.p((*pb.AuthRoleDeleteResponse)(&r))
}
func (p *printerRPC) RoleList(r v3.AuthRoleListResponse) { p.p((*pb.AuthRoleListResponse)(&r)) }
func (p *printerRPC) RoleGrantPermission(_ string, r v3.AuthRoleGrantPermissionResponse) {
	p.p((*pb.AuthRoleGrantPermissionResponse)(&r))
}
func (p *printerRPC) RoleRevokePermission(_ string, _ string, _ string, r v3.AuthRoleRevokePermissionResponse) {
	p.p((*pb.AuthRoleRevokePermissionResponse)(&r))
}
func (p *printerRPC) UserAdd(_ string, r v3.AuthUserAddResponse) { p.p((*pb.AuthUserAddResponse)(&r)) }
func (p *printerRPC) UserGet(_ string, r v3.AuthUserGetResponse) { p.p((*pb.AuthUserGetResponse)(&r)) }
func (p *printerRPC) UserList(r v3.AuthUserListResponse)         { p.p((*pb.AuthUserListResponse)(&r)) }
func (p *printerRPC) UserChangePassword(r v3.AuthUserChangePasswordResponse) {
	p.p((*pb.AuthUserChangePasswordResponse)(&r))
}
func (p *printerRPC) UserGrantRole(_ string, _ string, r v3.AuthUserGrantRoleResponse) {
	p.p((*pb.AuthUserGrantRoleResponse)(&r))
}
func (p *printerRPC) UserRevokeRole(_ string, _ string, r v3.AuthUserRevokeRoleResponse) {
	p.p((*pb.AuthUserRevokeRoleResponse)(&r))
}
func (p *printerRPC) UserDelete(_ string, r v3.AuthUserDeleteResponse) {
	p.p((*pb.AuthUserDeleteResponse)(&r))
}

type printerUnsupported struct{ printerRPC }

func newPrinterUnsupported(n string) printer {
	f := func(interface{}) {
		ExitWithError(ExitBadFeature, errors.New(n+" not supported as output format"))
	}
	return &printerUnsupported{printerRPC{nil, f}}
}

func (p *printerUnsupported) EndpointStatus([]epStatus) { p.p(nil) }
func (p *printerUnsupported) DBStatus(dbstatus)         { p.p(nil) }

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
