// Copyright 2018 Google LLC. All Rights Reserved.
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

package nftables

import (
	"encoding/binary"
	"fmt"

	"github.com/google/nftables/binaryutil"
	"github.com/google/nftables/expr"
	"github.com/google/nftables/internal/parseexprfunc"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

var (
	newObjHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWOBJ)
	delObjHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELOBJ)
)

type ObjType uint32

// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=be0bae0ad31b0adb506f96de083f52a2bd0d4fbf#n1612
const (
	ObjTypeCounter   ObjType = unix.NFT_OBJECT_COUNTER
	ObjTypeQuota     ObjType = unix.NFT_OBJECT_QUOTA
	ObjTypeCtHelper  ObjType = unix.NFT_OBJECT_CT_HELPER
	ObjTypeLimit     ObjType = unix.NFT_OBJECT_LIMIT
	ObjTypeConnLimit ObjType = unix.NFT_OBJECT_CONNLIMIT
	ObjTypeTunnel    ObjType = unix.NFT_OBJECT_TUNNEL
	ObjTypeCtTimeout ObjType = unix.NFT_OBJECT_CT_TIMEOUT
	ObjTypeSecMark   ObjType = unix.NFT_OBJECT_SECMARK
	ObjTypeCtExpect  ObjType = unix.NFT_OBJECT_CT_EXPECT
	ObjTypeSynProxy  ObjType = unix.NFT_OBJECT_SYNPROXY
)

var objByObjTypeMagic = map[ObjType]string{
	ObjTypeCounter:   "counter",
	ObjTypeQuota:     "quota",
	ObjTypeLimit:     "limit",
	ObjTypeConnLimit: "connlimit",
	ObjTypeCtHelper:  "cthelper",
	ObjTypeTunnel:    "tunnel", // not implemented in expr
	ObjTypeCtTimeout: "cttimeout",
	ObjTypeSecMark:   "secmark",
	ObjTypeCtExpect:  "ctexpect",
	ObjTypeSynProxy:  "synproxy",
}

// Obj represents a netfilter stateful object. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Stateful_objects
type Obj interface {
	table() *Table
	family() TableFamily
	data() expr.Any
	name() string
	objType() ObjType
}

// NamedObj represents nftables stateful object attributes
// Corresponds to netfilter nft_object_attributes as per
// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=116e95aa7b6358c917de8c69f6f173874030b46b#n1626
type NamedObj struct {
	Table *Table
	Name  string
	Type  ObjType
	Obj   expr.Any
}

func (o *NamedObj) table() *Table {
	return o.Table
}

func (o *NamedObj) family() TableFamily {
	return o.Table.Family
}

func (o *NamedObj) data() expr.Any {
	return o.Obj
}

func (o *NamedObj) name() string {
	return o.Name
}

func (o *NamedObj) objType() ObjType {
	return o.Type
}

// AddObject adds the specified Obj. Alias of AddObj.
func (cc *Conn) AddObject(o Obj) Obj {
	return cc.AddObj(o)
}

// AddObj adds the specified Obj. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Stateful_objects
func (cc *Conn) AddObj(o Obj) Obj {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data, err := expr.MarshalExprData(byte(o.family()), o.data())
	if err != nil {
		cc.setErr(err)
		return nil
	}

	attrs := []netlink.Attribute{
		{Type: unix.NFTA_OBJ_TABLE, Data: []byte(o.table().Name + "\x00")},
		{Type: unix.NFTA_OBJ_NAME, Data: []byte(o.name() + "\x00")},
		{Type: unix.NFTA_OBJ_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(o.objType()))},
	}
	if len(data) > 0 {
		attrs = append(attrs, netlink.Attribute{Type: unix.NLA_F_NESTED | unix.NFTA_OBJ_DATA, Data: data})
	}

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWOBJ),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(o.family()), 0), cc.marshalAttr(attrs)...),
	})
	return o
}

// DeleteObject deletes the specified Obj
func (cc *Conn) DeleteObject(o Obj) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	attrs := []netlink.Attribute{
		{Type: unix.NFTA_OBJ_TABLE, Data: []byte(o.table().Name + "\x00")},
		{Type: unix.NFTA_OBJ_NAME, Data: []byte(o.name() + "\x00")},
		{Type: unix.NFTA_OBJ_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(o.objType()))},
	}
	data := cc.marshalAttr(attrs)
	data = append(data, cc.marshalAttr([]netlink.Attribute{{Type: unix.NLA_F_NESTED | unix.NFTA_OBJ_DATA}})...)

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELOBJ),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(o.family()), 0), data...),
	})
}

// GetObj is a legacy method that return all Obj that belongs
// to the same table as the given one
// This function returns the same concrete type as passed,
// e.g. QuotaObj, CounterObj or NamedObj. Prefer using the more
// generic NamedObj over the legacy QuotaObj and CounterObj types.
func (cc *Conn) GetObj(o Obj) ([]Obj, error) {
	return cc.getObjWithLegacyType(nil, o.table(), unix.NFT_MSG_GETOBJ, cc.useLegacyObjType(o))
}

// GetObjReset is a legacy method that reset all Obj that belongs
// the same table as the given one
// This function returns the same concrete type as passed,
// e.g. QuotaObj, CounterObj or NamedObj. Prefer using the more
// generic NamedObj over the legacy QuotaObj and CounterObj types.
func (cc *Conn) GetObjReset(o Obj) ([]Obj, error) {
	return cc.getObjWithLegacyType(nil, o.table(), unix.NFT_MSG_GETOBJ_RESET, cc.useLegacyObjType(o))
}

// GetObject gets the specified Object
// This function returns the same concrete type as passed,
// e.g. QuotaObj, CounterObj or NamedObj. Prefer using the more
// generic NamedObj over the legacy QuotaObj and CounterObj types.
func (cc *Conn) GetObject(o Obj) (Obj, error) {
	objs, err := cc.getObj(o, o.table(), unix.NFT_MSG_GETOBJ)

	if len(objs) == 0 {
		return nil, err
	}

	return objs[0], err
}

// GetObjects get all the Obj that belongs to the given table
// This function will always return legacy QuotaObj/CounterObj
// types for backwards compatibility
func (cc *Conn) GetObjects(t *Table) ([]Obj, error) {
	return cc.getObj(nil, t, unix.NFT_MSG_GETOBJ)
}

// GetNamedObjects get all the Obj that belongs to the given table
// This function always return NamedObj types
func (cc *Conn) GetNamedObjects(t *Table) ([]Obj, error) {
	return cc.getObjWithLegacyType(nil, t, unix.NFT_MSG_GETOBJ, false)
}

// ResetObject reset the given Obj
// This function returns the same concrete type as passed,
// e.g. QuotaObj, CounterObj or NamedObj. Prefer using the more
// generic NamedObj over the legacy QuotaObj and CounterObj types.
func (cc *Conn) ResetObject(o Obj) (Obj, error) {
	objs, err := cc.getObj(o, o.table(), unix.NFT_MSG_GETOBJ_RESET)

	if len(objs) == 0 {
		return nil, err
	}

	return objs[0], err
}

// ResetObjects reset all the Obj that belongs to the given table
// This function will always return legacy QuotaObj/CounterObj
// types for backwards compatibility
func (cc *Conn) ResetObjects(t *Table) ([]Obj, error) {
	return cc.getObj(nil, t, unix.NFT_MSG_GETOBJ_RESET)
}

// ResetNamedObjects reset all the Obj that belongs to the given table
// This function always return NamedObj types
func (cc *Conn) ResetNamedObjects(t *Table) ([]Obj, error) {
	return cc.getObjWithLegacyType(nil, t, unix.NFT_MSG_GETOBJ_RESET, false)
}

func objFromMsg(msg netlink.Message, returnLegacyType bool) (Obj, error) {
	if got, want1, want2 := msg.Header.Type, newObjHeaderType, delObjHeaderType; got != want1 && got != want2 {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v or %v", got, want1, want2)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian
	var (
		table      *Table
		name       string
		objectType uint32
	)
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_OBJ_TABLE:
			table = &Table{Name: ad.String(), Family: TableFamily(msg.Data[0])}
		case unix.NFTA_OBJ_NAME:
			name = ad.String()
		case unix.NFTA_OBJ_TYPE:
			objectType = ad.Uint32()
		case unix.NFTA_OBJ_DATA:
			if returnLegacyType {
				return objDataFromMsgLegacy(ad, table, name, objectType)
			}

			o := NamedObj{
				Table: table,
				Name:  name,
				Type:  ObjType(objectType),
			}

			objs, err := parseexprfunc.ParseExprBytesFromNameFunc(byte(o.family()), ad, objByObjTypeMagic[o.Type])
			if err != nil {
				return nil, err
			}
			if len(objs) == 0 {
				return nil, fmt.Errorf("objFromMsg: objs is empty for obj %v", o)
			}
			exprs := make([]expr.Any, len(objs))
			for i := range exprs {
				exprs[i] = objs[i].(expr.Any)
			}

			o.Obj = exprs[0]
			return &o, ad.Err()
		}
	}
	if err := ad.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("malformed stateful object")
}

func objDataFromMsgLegacy(ad *netlink.AttributeDecoder, table *Table, name string, objectType uint32) (Obj, error) {
	switch objectType {
	case unix.NFT_OBJECT_COUNTER:
		o := CounterObj{
			Table: table,
			Name:  name,
		}

		ad.Do(func(b []byte) error {
			ad, err := netlink.NewAttributeDecoder(b)
			if err != nil {
				return err
			}
			ad.ByteOrder = binary.BigEndian
			return o.unmarshal(ad)
		})
		return &o, ad.Err()
	case unix.NFT_OBJECT_QUOTA:
		o := QuotaObj{
			Table: table,
			Name:  name,
		}

		ad.Do(func(b []byte) error {
			ad, err := netlink.NewAttributeDecoder(b)
			if err != nil {
				return err
			}
			ad.ByteOrder = binary.BigEndian
			return o.unmarshal(ad)
		})
		return &o, ad.Err()
	}
	if err := ad.Err(); err != nil {
		return nil, err
	}
	return nil, fmt.Errorf("malformed stateful object")
}

func (cc *Conn) getObj(o Obj, t *Table, msgType uint16) ([]Obj, error) {
	return cc.getObjWithLegacyType(o, t, msgType, cc.useLegacyObjType(o))
}

func (cc *Conn) getObjWithLegacyType(o Obj, t *Table, msgType uint16, returnLegacyObjType bool) ([]Obj, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	var data []byte
	var flags netlink.HeaderFlags

	if o != nil {
		attrs := []netlink.Attribute{
			{Type: unix.NFTA_OBJ_TABLE, Data: []byte(o.table().Name + "\x00")},
			{Type: unix.NFTA_OBJ_NAME, Data: []byte(o.name() + "\x00")},
			{Type: unix.NFTA_OBJ_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(o.objType()))},
		}
		data = cc.marshalAttr(attrs)
	} else {
		flags = netlink.Dump
		data, err = netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_RULE_TABLE, Data: []byte(t.Name + "\x00")},
		})
	}
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | msgType),
			Flags: netlink.Request | netlink.Acknowledge | flags,
		},
		Data: append(extraHeader(uint8(t.Family), 0), data...),
	}

	if _, err := conn.SendMessages([]netlink.Message{message}); err != nil {
		return nil, fmt.Errorf("SendMessages: %v", err)
	}

	reply, err := receiveAckAware(conn, message.Header.Flags)
	if err != nil {
		return nil, fmt.Errorf("receiveAckAware: %v", err)
	}
	var objs []Obj
	for _, msg := range reply {
		o, err := objFromMsg(msg, returnLegacyObjType)
		if err != nil {
			return nil, err
		}
		objs = append(objs, o)
	}

	return objs, nil
}

func (cc *Conn) useLegacyObjType(o Obj) bool {
	useLegacyType := true
	if o != nil {
		switch o.(type) {
		case *NamedObj:
			useLegacyType = false
		}
	}
	return useLegacyType
}
