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
	"fmt"
	"strings"

	v3 "github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type simplePrinter struct {
	isHex     bool
	valueOnly bool
}

func (s *simplePrinter) Del(resp v3.DeleteResponse) {
	fmt.Println(resp.Deleted)
	for _, kv := range resp.PrevKvs {
		printKV(s.isHex, s.valueOnly, kv)
	}
}

func (s *simplePrinter) Get(resp v3.GetResponse) {
	for _, kv := range resp.Kvs {
		printKV(s.isHex, s.valueOnly, kv)
	}
}

func (s *simplePrinter) Put(r v3.PutResponse) {
	fmt.Println("OK")
	if r.PrevKv != nil {
		printKV(s.isHex, s.valueOnly, r.PrevKv)
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
			printKV(s.isHex, s.valueOnly, e.PrevKv)
		}
		printKV(s.isHex, s.valueOnly, e.Kv)
	}
}

func (s *simplePrinter) Grant(resp v3.LeaseGrantResponse) {
	fmt.Printf("lease %016x granted with TTL(%ds)\n", resp.ID, resp.TTL)
}

func (p *simplePrinter) Revoke(id v3.LeaseID, r v3.LeaseRevokeResponse) {
	fmt.Printf("lease %016x revoked\n", id)
}

func (p *simplePrinter) KeepAlive(resp v3.LeaseKeepAliveResponse) {
	fmt.Printf("lease %016x keepalived with TTL(%d)\n", resp.ID, resp.TTL)
}

func (s *simplePrinter) TimeToLive(resp v3.LeaseTimeToLiveResponse, keys bool) {
	txt := fmt.Sprintf("lease %016x granted with TTL(%ds), remaining(%ds)", resp.ID, resp.GrantedTTL, resp.TTL)
	if keys {
		ks := make([]string, len(resp.Keys))
		for i := range resp.Keys {
			ks[i] = string(resp.Keys[i])
		}
		txt += fmt.Sprintf(", attached keys(%v)", ks)
	}
	fmt.Println(txt)
}

func (s *simplePrinter) Alarm(resp v3.AlarmResponse) {
	for _, e := range resp.Alarms {
		fmt.Printf("%+v\n", e)
	}
}

func (s *simplePrinter) MemberAdd(r v3.MemberAddResponse) {
	fmt.Printf("Member %16x added to cluster %16x\n", r.Member.ID, r.Header.ClusterId)
}

func (s *simplePrinter) MemberRemove(id uint64, r v3.MemberRemoveResponse) {
	fmt.Printf("Member %16x removed from cluster %16x\n", id, r.Header.ClusterId)
}

func (s *simplePrinter) MemberUpdate(id uint64, r v3.MemberUpdateResponse) {
	fmt.Printf("Member %16x updated in cluster %16x\n", id, r.Header.ClusterId)
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

func (s *simplePrinter) RoleAdd(role string, r v3.AuthRoleAddResponse) {
	fmt.Printf("Role %s created\n", role)
}

func (s *simplePrinter) RoleGet(role string, r v3.AuthRoleGetResponse) {
	fmt.Printf("Role %s\n", role)
	fmt.Println("KV Read:")

	printRange := func(perm *v3.Permission) {
		sKey := string(perm.Key)
		sRangeEnd := string(perm.RangeEnd)
		fmt.Printf("\t[%s, %s)", sKey, sRangeEnd)
		if strings.Compare(v3.GetPrefixRangeEnd(sKey), sRangeEnd) == 0 {
			fmt.Printf(" (prefix %s)", sKey)
		}
		fmt.Printf("\n")
	}

	for _, perm := range r.Perm {
		if perm.PermType == v3.PermRead || perm.PermType == v3.PermReadWrite {
			if len(perm.RangeEnd) == 0 {
				fmt.Printf("\t%s\n", string(perm.Key))
			} else {
				printRange((*v3.Permission)(perm))
			}
		}
	}
	fmt.Println("KV Write:")
	for _, perm := range r.Perm {
		if perm.PermType == v3.PermWrite || perm.PermType == v3.PermReadWrite {
			if len(perm.RangeEnd) == 0 {
				fmt.Printf("\t%s\n", string(perm.Key))
			} else {
				printRange((*v3.Permission)(perm))
			}
		}
	}
}

func (s *simplePrinter) RoleList(r v3.AuthRoleListResponse) {
	for _, role := range r.Roles {
		fmt.Printf("%s\n", role)
	}
}

func (s *simplePrinter) RoleDelete(role string, r v3.AuthRoleDeleteResponse) {
	fmt.Printf("Role %s deleted\n", role)
}

func (s *simplePrinter) RoleGrantPermission(role string, r v3.AuthRoleGrantPermissionResponse) {
	fmt.Printf("Role %s updated\n", role)
}

func (s *simplePrinter) RoleRevokePermission(role string, key string, end string, r v3.AuthRoleRevokePermissionResponse) {
	if len(end) == 0 {
		fmt.Printf("Permission of key %s is revoked from role %s\n", key, role)
		return
	}
	fmt.Printf("Permission of range [%s, %s) is revoked from role %s\n", key, end, role)
}

func (s *simplePrinter) UserAdd(name string, r v3.AuthUserAddResponse) {
	fmt.Printf("User %s created\n", name)
}

func (s *simplePrinter) UserGet(name string, r v3.AuthUserGetResponse) {
	fmt.Printf("User: %s\n", name)
	fmt.Printf("Roles:")
	for _, role := range r.Roles {
		fmt.Printf(" %s", role)
	}
	fmt.Printf("\n")
}

func (s *simplePrinter) UserChangePassword(v3.AuthUserChangePasswordResponse) {
	fmt.Println("Password updated")
}

func (s *simplePrinter) UserGrantRole(user string, role string, r v3.AuthUserGrantRoleResponse) {
	fmt.Printf("Role %s is granted to user %s\n", role, user)
}

func (s *simplePrinter) UserRevokeRole(user string, role string, r v3.AuthUserRevokeRoleResponse) {
	fmt.Printf("Role %s is revoked from user %s\n", role, user)
}

func (s *simplePrinter) UserDelete(user string, r v3.AuthUserDeleteResponse) {
	fmt.Printf("User %s deleted\n", user)
}

func (s *simplePrinter) UserList(r v3.AuthUserListResponse) {
	for _, user := range r.Users {
		fmt.Printf("%s\n", user)
	}
}
