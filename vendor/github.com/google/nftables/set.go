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
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/nftables/binaryutil"
	"github.com/google/nftables/expr"
	"github.com/google/nftables/internal/parseexprfunc"
	"github.com/google/nftables/userdata"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

// SetConcatTypeBits defines concatination bits, originally defined in
// https://git.netfilter.org/iptables/tree/iptables/nft.c?id=26753888720d8e7eb422ae4311348347f5a05cb4#n1002
const (
	SetConcatTypeBits = 6
	SetConcatTypeMask = (1 << SetConcatTypeBits) - 1
	// below consts added because not found in go unix package
	// https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=d1289bff58e1878c3162f574c603da993e29b113#n306
	NFT_SET_CONCAT = 0x80
	// https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=d1289bff58e1878c3162f574c603da993e29b113#n330
	NFTA_SET_DESC_CONCAT = 2
	// https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=d1289bff58e1878c3162f574c603da993e29b113#n428
	NFTA_SET_ELEM_KEY_END = 10
	// https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=d1289bff58e1878c3162f574c603da993e29b113#n429
	NFTA_SET_ELEM_EXPRESSIONS = 0x11
)

var allocSetID uint32

// SetDatatype represents a datatype declared by nft.
type SetDatatype struct {
	Name  string
	Bytes uint32

	// nftMagic represents the magic value that nft uses for
	// certain types (ie: IP addresses). We populate SET_KEY_TYPE
	// identically, so `nft list ...` commands produce correct output.
	nftMagic uint32
}

// GetNFTMagic returns a custom datatype based on user's parameters
func (s *SetDatatype) GetNFTMagic() uint32 {
	return s.nftMagic
}

// SetNFTMagic returns a custom datatype based on user's parameters
func (s *SetDatatype) SetNFTMagic(nftMagic uint32) {
	s.nftMagic = nftMagic
}

// NFT datatypes. See: https://git.netfilter.org/nftables/tree/include/datatype.h
var (
	TypeInvalid     = SetDatatype{Name: "invalid", nftMagic: 0}
	TypeVerdict     = SetDatatype{Name: "verdict", Bytes: 0, nftMagic: 1}
	TypeNFProto     = SetDatatype{Name: "nf_proto", Bytes: 1, nftMagic: 2}
	TypeBitmask     = SetDatatype{Name: "bitmask", Bytes: 0, nftMagic: 3}
	TypeInteger     = SetDatatype{Name: "integer", Bytes: 4, nftMagic: 4}
	TypeString      = SetDatatype{Name: "string", Bytes: 0, nftMagic: 5}
	TypeLLAddr      = SetDatatype{Name: "ll_addr", Bytes: 0, nftMagic: 6}
	TypeIPAddr      = SetDatatype{Name: "ipv4_addr", Bytes: 4, nftMagic: 7}
	TypeIP6Addr     = SetDatatype{Name: "ipv6_addr", Bytes: 16, nftMagic: 8}
	TypeEtherAddr   = SetDatatype{Name: "ether_addr", Bytes: 6, nftMagic: 9}
	TypeEtherType   = SetDatatype{Name: "ether_type", Bytes: 2, nftMagic: 10}
	TypeARPOp       = SetDatatype{Name: "arp_op", Bytes: 2, nftMagic: 11}
	TypeInetProto   = SetDatatype{Name: "inet_proto", Bytes: 1, nftMagic: 12}
	TypeInetService = SetDatatype{Name: "inet_service", Bytes: 2, nftMagic: 13}
	TypeICMPType    = SetDatatype{Name: "icmp_type", Bytes: 1, nftMagic: 14}
	TypeTCPFlag     = SetDatatype{Name: "tcp_flag", Bytes: 1, nftMagic: 15}
	TypeDCCPPktType = SetDatatype{Name: "dccp_pkttype", Bytes: 1, nftMagic: 16}
	TypeMHType      = SetDatatype{Name: "mh_type", Bytes: 1, nftMagic: 17}
	TypeTime        = SetDatatype{Name: "time", Bytes: 8, nftMagic: 18}
	TypeMark        = SetDatatype{Name: "mark", Bytes: 4, nftMagic: 19}
	TypeIFIndex     = SetDatatype{Name: "iface_index", Bytes: 4, nftMagic: 20}
	TypeARPHRD      = SetDatatype{Name: "iface_type", Bytes: 2, nftMagic: 21}
	TypeRealm       = SetDatatype{Name: "realm", Bytes: 4, nftMagic: 22}
	TypeClassID     = SetDatatype{Name: "classid", Bytes: 4, nftMagic: 23}
	TypeUID         = SetDatatype{Name: "uid", Bytes: sizeOfUIDT, nftMagic: 24}
	TypeGID         = SetDatatype{Name: "gid", Bytes: sizeOfGIDT, nftMagic: 25}
	TypeCTState     = SetDatatype{Name: "ct_state", Bytes: 4, nftMagic: 26}
	TypeCTDir       = SetDatatype{Name: "ct_dir", Bytes: 1, nftMagic: 27}
	TypeCTStatus    = SetDatatype{Name: "ct_status", Bytes: 4, nftMagic: 28}
	TypeICMP6Type   = SetDatatype{Name: "icmpv6_type", Bytes: 1, nftMagic: 29}
	TypeCTLabel     = SetDatatype{Name: "ct_label", Bytes: ctLabelBitSize / 8, nftMagic: 30}
	TypePktType     = SetDatatype{Name: "pkt_type", Bytes: 1, nftMagic: 31}
	TypeICMPCode    = SetDatatype{Name: "icmp_code", Bytes: 1, nftMagic: 32}
	TypeICMPV6Code  = SetDatatype{Name: "icmpv6_code", Bytes: 1, nftMagic: 33}
	TypeICMPXCode   = SetDatatype{Name: "icmpx_code", Bytes: 1, nftMagic: 34}
	TypeDevGroup    = SetDatatype{Name: "devgroup", Bytes: 4, nftMagic: 35}
	TypeDSCP        = SetDatatype{Name: "dscp", Bytes: 1, nftMagic: 36}
	TypeECN         = SetDatatype{Name: "ecn", Bytes: 1, nftMagic: 37}
	TypeFIBAddr     = SetDatatype{Name: "fib_addrtype", Bytes: 4, nftMagic: 38}
	TypeBoolean     = SetDatatype{Name: "boolean", Bytes: 1, nftMagic: 39}
	TypeCTEventBit  = SetDatatype{Name: "ct_event", Bytes: 4, nftMagic: 40}
	TypeIFName      = SetDatatype{Name: "ifname", Bytes: ifNameSize, nftMagic: 41}
	TypeIGMPType    = SetDatatype{Name: "igmp_type", Bytes: 1, nftMagic: 42}
	TypeTimeDate    = SetDatatype{Name: "time", Bytes: 8, nftMagic: 43}
	TypeTimeHour    = SetDatatype{Name: "hour", Bytes: 4, nftMagic: 44}
	TypeTimeDay     = SetDatatype{Name: "day", Bytes: 1, nftMagic: 45}
	TypeCGroupV2    = SetDatatype{Name: "cgroupsv2", Bytes: 8, nftMagic: 46}

	nftDatatypes = []SetDatatype{
		TypeVerdict,
		TypeNFProto,
		TypeBitmask,
		TypeInteger,
		TypeString,
		TypeLLAddr,
		TypeIPAddr,
		TypeIP6Addr,
		TypeEtherAddr,
		TypeEtherType,
		TypeARPOp,
		TypeInetProto,
		TypeInetService,
		TypeICMPType,
		TypeTCPFlag,
		TypeDCCPPktType,
		TypeMHType,
		TypeTime,
		TypeMark,
		TypeIFIndex,
		TypeARPHRD,
		TypeRealm,
		TypeClassID,
		TypeUID,
		TypeGID,
		TypeCTState,
		TypeCTDir,
		TypeCTStatus,
		TypeICMP6Type,
		TypeCTLabel,
		TypePktType,
		TypeICMPCode,
		TypeICMPV6Code,
		TypeICMPXCode,
		TypeDevGroup,
		TypeDSCP,
		TypeECN,
		TypeFIBAddr,
		TypeBoolean,
		TypeCTEventBit,
		TypeIFName,
		TypeIGMPType,
		TypeTimeDate,
		TypeTimeHour,
		TypeTimeDay,
		TypeCGroupV2,
	}

	// ctLabelBitSize is defined in https://git.netfilter.org/nftables/tree/src/ct.c.
	ctLabelBitSize uint32 = 128

	// ifNameSize is called IFNAMSIZ in linux/if.h.
	ifNameSize uint32 = 16

	// bits/typesizes.h
	sizeOfUIDT uint32 = 4
	sizeOfGIDT uint32 = 4
)

var nftDatatypesByName map[string]SetDatatype
var nftDatatypesByMagic map[uint32]SetDatatype

// Create maps for efficient datatype lookup.
func init() {
	nftDatatypesByName = make(map[string]SetDatatype, len(nftDatatypes))
	nftDatatypesByMagic = make(map[uint32]SetDatatype, len(nftDatatypes))
	for _, dt := range nftDatatypes {
		nftDatatypesByName[dt.Name] = dt
		nftDatatypesByMagic[dt.nftMagic] = dt
	}
}

// ErrTooManyTypes is the error returned by ConcatSetType, if nftMagic would overflow.
var ErrTooManyTypes = errors.New("too many types to concat")

// MustConcatSetType does the same as ConcatSetType, but panics instead of an
// error. It simplifies safe initialization of global variables.
func MustConcatSetType(types ...SetDatatype) SetDatatype {
	t, err := ConcatSetType(types...)
	if err != nil {
		panic(err)
	}
	return t
}

// ConcatSetType constructs a new SetDatatype which consists of a concatenation
// of the passed types. It returns ErrTooManyTypes, if nftMagic would overflow
// (more than 5 types).
func ConcatSetType(types ...SetDatatype) (SetDatatype, error) {
	if len(types) > 32/SetConcatTypeBits {
		return SetDatatype{}, ErrTooManyTypes
	}

	var magic, bytes uint32
	names := make([]string, len(types))
	for i, t := range types {
		bytes += t.Bytes
		// concatenated types pad the length to multiples of the register size (4 bytes)
		// see https://git.netfilter.org/nftables/tree/src/datatype.c?id=488356b895024d0944b20feb1f930558726e0877#n1162
		if t.Bytes%4 != 0 {
			bytes += 4 - (t.Bytes % 4)
		}
		names[i] = t.Name

		magic <<= SetConcatTypeBits
		magic |= t.nftMagic & SetConcatTypeMask
	}
	return SetDatatype{Name: strings.Join(names, " . "), Bytes: bytes, nftMagic: magic}, nil
}

// ConcatSetTypeElements uses the ConcatSetType name to calculate and  return
// a list of base types which were used to construct the concatenated type
func ConcatSetTypeElements(t SetDatatype) []SetDatatype {
	names := strings.Split(t.Name, " . ")
	types := make([]SetDatatype, len(names))
	for i, n := range names {
		types[i] = nftDatatypesByName[n]
	}
	return types
}

// Set represents an nftables set. Anonymous sets are only valid within the
// context of a single batch.
type Set struct {
	Table      *Table
	ID         uint32
	Name       string
	Anonymous  bool
	Constant   bool
	Interval   bool
	AutoMerge  bool
	IsMap      bool
	HasTimeout bool
	Counter    bool
	// Can be updated per evaluation path, per `nft list ruleset`
	// indicates that set contains "flags dynamic"
	// https://git.netfilter.org/libnftnl/tree/include/linux/netfilter/nf_tables.h?id=84d12cfacf8ddd857a09435f3d982ab6250d250c#n298
	Dynamic bool
	// Indicates that the set contains a concatenation
	// https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=d1289bff58e1878c3162f574c603da993e29b113#n306
	Concatenation bool
	Timeout       time.Duration
	KeyType       SetDatatype
	DataType      SetDatatype
	// Either host (binaryutil.NativeEndian) or big (binaryutil.BigEndian) endian as per
	// https://git.netfilter.org/nftables/tree/include/datatype.h?id=d486c9e626405e829221b82d7355558005b26d8a#n109
	KeyByteOrder binaryutil.ByteOrder
	Comment      string
	// Indicates that the set has "size" specifier
	Size uint32
}

// SetElement represents a data point within a set.
type SetElement struct {
	Key []byte
	Val []byte
	// Field used for definition of ending interval value in concatenated types
	// https://git.netfilter.org/libnftnl/tree/include/set_elem.h?id=e2514c0eff4da7e8e0aabd410f7b7d0b7564c880#n11
	KeyEnd      []byte
	IntervalEnd bool
	// To support vmap, a caller must be able to pass Verdict type of data.
	// If IsMap is true and VerdictData is not nil, then Val of SetElement will be ignored
	// and VerdictData will be wrapped into Attribute data.
	VerdictData *expr.Verdict
	// To support aging of set elements
	Timeout time.Duration

	// Life left of the "timeout" elements
	Expires time.Duration

	Counter *expr.Counter
	Comment string
}

func (s *SetElement) decode(fam byte) func(b []byte) error {
	return func(b []byte) error {
		ad, err := netlink.NewAttributeDecoder(b)
		if err != nil {
			return fmt.Errorf("failed to create nested attribute decoder: %v", err)
		}
		ad.ByteOrder = binary.BigEndian

		for ad.Next() {
			switch ad.Type() {
			case unix.NFTA_SET_ELEM_KEY:
				s.Key, err = decodeElement(ad.Bytes())
				if err != nil {
					return err
				}
			case NFTA_SET_ELEM_KEY_END:
				s.KeyEnd, err = decodeElement(ad.Bytes())
				if err != nil {
					return err
				}
			case unix.NFTA_SET_ELEM_DATA:
				s.Val, err = decodeElement(ad.Bytes())
				if err != nil {
					return err
				}
			case unix.NFTA_SET_ELEM_FLAGS:
				flags := ad.Uint32()
				s.IntervalEnd = (flags & unix.NFT_SET_ELEM_INTERVAL_END) != 0
			case unix.NFTA_SET_ELEM_TIMEOUT:
				s.Timeout = time.Millisecond * time.Duration(ad.Uint64())
			case unix.NFTA_SET_ELEM_EXPIRATION:
				s.Expires = time.Millisecond * time.Duration(ad.Uint64())
			case unix.NFTA_SET_ELEM_USERDATA:
				userData := ad.Bytes()
				// Try to extract comment from userdata if present
				if comment, ok := userdata.GetString(userData, userdata.NFTNL_UDATA_SET_ELEM_COMMENT); ok {
					s.Comment = comment
				}
			case unix.NFTA_SET_ELEM_EXPR:
				elems, err := parseexprfunc.ParseExprBytesFunc(fam, ad)
				if err != nil {
					return err
				}

				for _, elem := range elems {
					switch item := elem.(type) {
					case *expr.Counter:
						s.Counter = item
					}
				}
			}
		}
		return ad.Err()
	}
}

func decodeElement(d []byte) ([]byte, error) {
	ad, err := netlink.NewAttributeDecoder(d)
	if err != nil {
		return nil, fmt.Errorf("failed to create nested attribute decoder: %v", err)
	}
	ad.ByteOrder = binary.BigEndian
	var b []byte
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_SET_ELEM_KEY:
			fallthrough
		case unix.NFTA_SET_ELEM_DATA:
			b = ad.Bytes()
		}
	}
	if err := ad.Err(); err != nil {
		return nil, err
	}
	return b, nil
}

// SetAddElements applies data points to an nftables set.
func (cc *Conn) SetAddElements(s *Set, vals []SetElement) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if s.Anonymous {
		return errors.New("anonymous sets cannot be updated")
	}

	elements, err := s.makeElemList(vals, s.ID)
	if err != nil {
		return err
	}
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWSETELEM),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), cc.marshalAttr(elements)...),
	})

	return nil
}

func (s *Set) makeElemList(vals []SetElement, id uint32) ([]netlink.Attribute, error) {
	var elements []netlink.Attribute

	for i, v := range vals {
		item := make([]netlink.Attribute, 0)
		var flags uint32
		if v.IntervalEnd {
			flags |= unix.NFT_SET_ELEM_INTERVAL_END
			item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_FLAGS | unix.NLA_F_NESTED, Data: binaryutil.BigEndian.PutUint32(flags)})
		}

		encodedKey, err := netlink.MarshalAttributes([]netlink.Attribute{{Type: unix.NFTA_DATA_VALUE, Data: v.Key}})
		if err != nil {
			return nil, fmt.Errorf("marshal key %d: %v", i, err)
		}

		item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_KEY | unix.NLA_F_NESTED, Data: encodedKey})
		if len(v.KeyEnd) > 0 {
			encodedKeyEnd, err := netlink.MarshalAttributes([]netlink.Attribute{{Type: unix.NFTA_DATA_VALUE, Data: v.KeyEnd}})
			if err != nil {
				return nil, fmt.Errorf("marshal key end %d: %v", i, err)
			}
			item = append(item, netlink.Attribute{Type: NFTA_SET_ELEM_KEY_END | unix.NLA_F_NESTED, Data: encodedKeyEnd})
		}
		if s.HasTimeout && v.Timeout != 0 {
			// Set has Timeout flag set, which means an individual element can specify its own timeout.
			item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_TIMEOUT, Data: binaryutil.BigEndian.PutUint64(uint64(v.Timeout.Milliseconds()))})
		}
		// The following switch statement deal with 3 different types of elements.
		// 1. v is an element of vmap
		// 2. v is an element of a regular map
		// 3. v is an element of a regular set (default)
		switch {
		case v.VerdictData != nil:
			// Since VerdictData is not nil, v is vmap element, need to add to the attributes
			encodedVal := []byte{}
			encodedKind, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_DATA_VALUE, Data: binaryutil.BigEndian.PutUint32(uint32(v.VerdictData.Kind))},
			})
			if err != nil {
				return nil, fmt.Errorf("marshal item %d: %v", i, err)
			}
			encodedVal = append(encodedVal, encodedKind...)
			if len(v.VerdictData.Chain) != 0 {
				encodedChain, err := netlink.MarshalAttributes([]netlink.Attribute{
					{Type: unix.NFTA_SET_ELEM_DATA, Data: []byte(v.VerdictData.Chain + "\x00")},
				})
				if err != nil {
					return nil, fmt.Errorf("marshal item %d: %v", i, err)
				}
				encodedVal = append(encodedVal, encodedChain...)
			}
			encodedVerdict, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_SET_ELEM_DATA | unix.NLA_F_NESTED, Data: encodedVal}})
			if err != nil {
				return nil, fmt.Errorf("marshal item %d: %v", i, err)
			}
			item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_DATA | unix.NLA_F_NESTED, Data: encodedVerdict})
		case len(v.Val) > 0:
			// Since v.Val's length is not 0 then, v is a regular map element, need to add to the attributes
			encodedVal, err := netlink.MarshalAttributes([]netlink.Attribute{{Type: unix.NFTA_DATA_VALUE, Data: v.Val}})
			if err != nil {
				return nil, fmt.Errorf("marshal item %d: %v", i, err)
			}

			item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_DATA | unix.NLA_F_NESTED, Data: encodedVal})
		default:
			// If niether of previous cases matche, it means 'e' is an element of a regular Set, no need to add to the attributes
		}

		// Add comment to userdata if present
		if len(v.Comment) > 0 {
			userData := userdata.AppendString(nil, userdata.NFTNL_UDATA_SET_ELEM_COMMENT, v.Comment)
			item = append(item, netlink.Attribute{Type: unix.NFTA_SET_ELEM_USERDATA, Data: userData})
		}

		encodedItem, err := netlink.MarshalAttributes(item)
		if err != nil {
			return nil, fmt.Errorf("marshal item %d: %v", i, err)
		}
		elements = append(elements, netlink.Attribute{Type: uint16(i+1) | unix.NLA_F_NESTED, Data: encodedItem})
	}

	encodedElem, err := netlink.MarshalAttributes(elements)
	if err != nil {
		return nil, fmt.Errorf("marshal elements: %v", err)
	}

	return []netlink.Attribute{
		{Type: unix.NFTA_SET_NAME, Data: []byte(s.Name + "\x00")},
		{Type: unix.NFTA_LOOKUP_SET_ID, Data: binaryutil.BigEndian.PutUint32(id)},
		{Type: unix.NFTA_SET_TABLE, Data: []byte(s.Table.Name + "\x00")},
		{Type: unix.NFTA_SET_ELEM_LIST_ELEMENTS | unix.NLA_F_NESTED, Data: encodedElem},
	}, nil
}

// AddSet adds the specified Set.
func (cc *Conn) AddSet(s *Set, vals []SetElement) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	// Based on nft implementation & linux source.
	// Link: https://github.com/torvalds/linux/blob/49a57857aeea06ca831043acbb0fa5e0f50602fd/net/netfilter/nf_tables_api.c#L3395
	// Another reference: https://git.netfilter.org/nftables/tree/src

	if s.Anonymous && !s.Constant {
		return errors.New("anonymous structs must be constant")
	}

	if s.ID == 0 {
		allocSetID++
		s.ID = allocSetID
		if s.Anonymous {
			s.Name = "__set%d"
			if s.IsMap {
				s.Name = "__map%d"
			}
		}
	}

	var flags uint32
	if s.Anonymous {
		flags |= unix.NFT_SET_ANONYMOUS
	}
	if s.Constant {
		flags |= unix.NFT_SET_CONSTANT
	}
	if s.Interval {
		flags |= unix.NFT_SET_INTERVAL
	}
	if s.IsMap {
		flags |= unix.NFT_SET_MAP
	}
	if s.HasTimeout {
		flags |= unix.NFT_SET_TIMEOUT
	}
	if s.Dynamic {
		flags |= unix.NFT_SET_EVAL
	}
	if s.Concatenation {
		flags |= NFT_SET_CONCAT
	}
	tableInfo := []netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(s.Table.Name + "\x00")},
		{Type: unix.NFTA_SET_NAME, Data: []byte(s.Name + "\x00")},
		{Type: unix.NFTA_SET_FLAGS, Data: binaryutil.BigEndian.PutUint32(flags)},
		{Type: unix.NFTA_SET_KEY_TYPE, Data: binaryutil.BigEndian.PutUint32(s.KeyType.nftMagic)},
		{Type: unix.NFTA_SET_KEY_LEN, Data: binaryutil.BigEndian.PutUint32(s.KeyType.Bytes)},
		{Type: unix.NFTA_SET_ID, Data: binaryutil.BigEndian.PutUint32(s.ID)},
	}
	if s.IsMap {
		// Check if it is vmap case
		if s.DataType.nftMagic == 1 {
			// For Verdict data type, the expected magic is 0xfffff0
			tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NFTA_SET_DATA_TYPE, Data: binaryutil.BigEndian.PutUint32(uint32(unix.NFT_DATA_VERDICT))},
				netlink.Attribute{Type: unix.NFTA_SET_DATA_LEN, Data: binaryutil.BigEndian.PutUint32(s.DataType.Bytes)})
		} else {
			tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NFTA_SET_DATA_TYPE, Data: binaryutil.BigEndian.PutUint32(s.DataType.nftMagic)},
				netlink.Attribute{Type: unix.NFTA_SET_DATA_LEN, Data: binaryutil.BigEndian.PutUint32(s.DataType.Bytes)})
		}
	}
	if s.HasTimeout && s.Timeout != 0 {
		// If Set's global timeout is specified, add it to set's attributes
		tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NFTA_SET_TIMEOUT, Data: binaryutil.BigEndian.PutUint64(uint64(s.Timeout.Milliseconds()))})
	}
	if s.Constant {
		// nft cli tool adds the number of elements to set/map's descriptor
		// It make sense to do only if a set or map are constant, otherwise skip NFTA_SET_DESC attribute
		numberOfElements, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_DATA_VALUE, Data: binaryutil.BigEndian.PutUint32(uint32(len(vals)))},
		})
		if err != nil {
			return fmt.Errorf("fail to marshal number of elements %d: %v", len(vals), err)
		}
		tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NLA_F_NESTED | unix.NFTA_SET_DESC, Data: numberOfElements})
	}

	var descBytes []byte

	if s.Size > 0 {
		// Marshal set size description
		descSizeBytes, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_SET_DESC_SIZE, Data: binaryutil.BigEndian.PutUint32(s.Size)},
		})
		if err != nil {
			return fmt.Errorf("fail to marshal set size description: %w", err)
		}

		descBytes = append(descBytes, descSizeBytes...)
	}

	if s.Concatenation {
		// Length of concatenated types is a must, otherwise segfaults when executing nft list ruleset
		var concatDefinition []byte
		elements := ConcatSetTypeElements(s.KeyType)
		for i, v := range elements {
			// Marshal base type size value
			valData, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_DATA_VALUE, Data: binaryutil.BigEndian.PutUint32(v.Bytes)},
			})
			if err != nil {
				return fmt.Errorf("fail to marshal element key size %d: %v", i, err)
			}
			// Marshal base type size description
			descSize, err := netlink.MarshalAttributes([]netlink.Attribute{
				{Type: unix.NFTA_SET_DESC_SIZE, Data: valData},
			})
			if err != nil {
				return fmt.Errorf("fail to marshal base type size description: %w", err)
			}
			concatDefinition = append(concatDefinition, descSize...)
		}
		// Marshal all base type descriptions into concatenation size description
		concatBytes, err := netlink.MarshalAttributes([]netlink.Attribute{{Type: unix.NLA_F_NESTED | NFTA_SET_DESC_CONCAT, Data: concatDefinition}})
		if err != nil {
			return fmt.Errorf("fail to marshal concat definition %v", err)
		}

		descBytes = append(descBytes, concatBytes...)
	}

	if len(descBytes) > 0 {
		// Marshal set description
		tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NLA_F_NESTED | unix.NFTA_SET_DESC, Data: descBytes})
	}

	// https://git.netfilter.org/libnftnl/tree/include/udata.h#n17
	var userData []byte

	if s.Anonymous || s.Constant || s.Interval || s.KeyByteOrder == binaryutil.BigEndian {
		// Semantically useless - kept for binary compatability with nft
		userData = userdata.AppendUint32(userData, userdata.NFTNL_UDATA_SET_KEYBYTEORDER, 2)
	} else if s.KeyByteOrder == binaryutil.NativeEndian {
		// Per https://git.netfilter.org/nftables/tree/src/mnl.c?id=187c6d01d35722618c2711bbc49262c286472c8f#n1165
		userData = userdata.AppendUint32(userData, userdata.NFTNL_UDATA_SET_KEYBYTEORDER, 1)
	}

	if s.Interval && s.AutoMerge {
		// https://git.netfilter.org/nftables/tree/src/mnl.c?id=187c6d01d35722618c2711bbc49262c286472c8f#n1174
		userData = userdata.AppendUint32(userData, userdata.NFTNL_UDATA_SET_MERGE_ELEMENTS, 1)
	}

	if len(s.Comment) != 0 {
		userData = userdata.AppendString(userData, userdata.NFTNL_UDATA_SET_COMMENT, s.Comment)
	}

	if len(userData) > 0 {
		tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NFTA_SET_USERDATA, Data: userData})
	}

	if s.Counter {
		data, err := netlink.MarshalAttributes([]netlink.Attribute{
			{Type: unix.NFTA_LIST_ELEM, Data: []byte("counter\x00")},
			{Type: unix.NFTA_SET_ELEM_PAD | unix.NFTA_SET_ELEM_DATA, Data: []byte{}},
		})
		if err != nil {
			return err
		}
		tableInfo = append(tableInfo, netlink.Attribute{Type: unix.NLA_F_NESTED | NFTA_SET_ELEM_EXPRESSIONS, Data: data})
	}

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWSET),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), cc.marshalAttr(tableInfo)...),
	})

	// Set the values of the set if initial values were provided.
	if len(vals) > 0 {
		hdrType := unix.NFT_MSG_NEWSETELEM
		elements, err := s.makeElemList(vals, s.ID)
		if err != nil {
			return err
		}
		cc.messages = append(cc.messages, netlink.Message{
			Header: netlink.Header{
				Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | hdrType),
				Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
			},
			Data: append(extraHeader(uint8(s.Table.Family), 0), cc.marshalAttr(elements)...),
		})
	}

	return nil
}

// DelSet deletes a specific set, along with all elements it contains.
func (cc *Conn) DelSet(s *Set) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(s.Table.Name + "\x00")},
		{Type: unix.NFTA_SET_NAME, Data: []byte(s.Name + "\x00")},
	})
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELSET),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), data...),
	})
}

// SetDeleteElements deletes data points from an nftables set.
func (cc *Conn) SetDeleteElements(s *Set, vals []SetElement) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if s.Anonymous {
		return errors.New("anonymous sets cannot be updated")
	}

	elements, err := s.makeElemList(vals, s.ID)
	if err != nil {
		return err
	}
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELSETELEM),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Create,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), cc.marshalAttr(elements)...),
	})

	return nil
}

// FlushSet deletes all data points from an nftables set.
func (cc *Conn) FlushSet(s *Set) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(s.Table.Name + "\x00")},
		{Type: unix.NFTA_SET_NAME, Data: []byte(s.Name + "\x00")},
	})
	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELSETELEM),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), data...),
	})
}

var (
	newSetHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWSET)
	delSetHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELSET)
)

func setsFromMsg(msg netlink.Message) (*Set, error) {
	if got, want1, want2 := msg.Header.Type, newSetHeaderType, delSetHeaderType; got != want1 && got != want2 {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v or %v", got, want1, want2)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian

	var set Set
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_SET_NAME:
			set.Name = ad.String()
		case unix.NFTA_SET_ID:
			set.ID = binary.BigEndian.Uint32(ad.Bytes())
		case unix.NFTA_SET_TIMEOUT:
			set.Timeout = time.Duration(time.Millisecond * time.Duration(binary.BigEndian.Uint64(ad.Bytes())))
			set.HasTimeout = true
		case unix.NFTA_SET_FLAGS:
			flags := ad.Uint32()
			set.Constant = (flags & unix.NFT_SET_CONSTANT) != 0
			set.Anonymous = (flags & unix.NFT_SET_ANONYMOUS) != 0
			set.Interval = (flags & unix.NFT_SET_INTERVAL) != 0
			set.IsMap = (flags & unix.NFT_SET_MAP) != 0
			set.HasTimeout = (flags & unix.NFT_SET_TIMEOUT) != 0
			set.Dynamic = (flags & unix.NFT_SET_EVAL) != 0
			set.Concatenation = (flags & NFT_SET_CONCAT) != 0
		case unix.NFTA_SET_KEY_TYPE:
			nftMagic := ad.Uint32()
			dt, err := parseSetDatatype(nftMagic)
			if err != nil {
				return nil, fmt.Errorf("could not determine data type: %w", err)
			}
			set.KeyType = dt
		case unix.NFTA_SET_KEY_LEN:
			set.KeyType.Bytes = binary.BigEndian.Uint32(ad.Bytes())
		case unix.NFTA_SET_DATA_TYPE:
			nftMagic := ad.Uint32()
			// Special case for the data type verdict, in the message it is stored as 0xffffff00 but it is defined as 1
			if nftMagic == 0xffffff00 {
				set.KeyType = TypeVerdict
				break
			}
			dt, err := parseSetDatatype(nftMagic)
			if err != nil {
				return nil, fmt.Errorf("could not determine data type: %w", err)
			}
			set.DataType = dt
		case unix.NFTA_SET_DATA_LEN:
			set.DataType.Bytes = binary.BigEndian.Uint32(ad.Bytes())
		case unix.NFTA_SET_USERDATA:
			data := ad.Bytes()
			value, ok := userdata.GetUint32(data, userdata.NFTNL_UDATA_SET_MERGE_ELEMENTS)
			set.AutoMerge = ok && value == 1
		case unix.NFTA_SET_DESC:
			nestedAD, err := netlink.NewAttributeDecoder(ad.Bytes())
			if err != nil {
				return nil, fmt.Errorf("nested NewAttributeDecoder() failed: %w", err)
			}
			for nestedAD.Next() {
				switch nestedAD.Type() {
				case unix.NFTA_SET_DESC_SIZE:
					set.Size = binary.BigEndian.Uint32(nestedAD.Bytes())
				}
			}
			if nestedAD.Err() != nil {
				return nil, fmt.Errorf("decoding set description: %w", nestedAD.Err())
			}
		}
	}
	return &set, nil
}

func parseSetDatatype(magic uint32) (SetDatatype, error) {
	types := make([]SetDatatype, 0, 32/SetConcatTypeBits)
	for magic != 0 {
		t := magic & SetConcatTypeMask
		magic = magic >> SetConcatTypeBits
		dt, ok := nftDatatypesByMagic[t]
		if !ok {
			return TypeInvalid, fmt.Errorf("could not determine data type %+v", dt)
		}
		// Because we start with the last type, we insert the later types at the front.
		types = append([]SetDatatype{dt}, types...)
	}

	dt, err := ConcatSetType(types...)
	if err != nil {
		return TypeInvalid, fmt.Errorf("could not create data type: %w", err)
	}
	return dt, nil
}

var (
	newElemHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWSETELEM)
	delElemHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELSETELEM)
)

func elementsFromMsg(fam byte, msg netlink.Message) ([]SetElement, error) {
	if got, want1, want2 := msg.Header.Type, newElemHeaderType, delElemHeaderType; got != want1 && got != want2 {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v or %v", got, want1, want2)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian

	var elements []SetElement
	for ad.Next() {
		b := ad.Bytes()
		if ad.Type() == unix.NFTA_SET_ELEM_LIST_ELEMENTS {
			ad, err := netlink.NewAttributeDecoder(b)

			if err != nil {
				return nil, err
			}
			ad.ByteOrder = binary.BigEndian

			for ad.Next() {
				var elem SetElement
				switch ad.Type() {
				case unix.NFTA_LIST_ELEM:
					ad.Do(elem.decode(fam))
				}

				elements = append(elements, elem)
			}
		}
	}
	return elements, nil
}

// GetSets returns the sets in the specified table.
func (cc *Conn) GetSets(t *Table) ([]*Set, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	data, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(t.Name + "\x00")},
	})
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETSET),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Dump,
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
	var sets []*Set
	for _, msg := range reply {
		s, err := setsFromMsg(msg)
		if err != nil {
			return nil, err
		}
		s.Table = &Table{Name: t.Name, Use: t.Use, Flags: t.Flags, Family: t.Family}
		sets = append(sets, s)
	}

	return sets, nil
}

// GetSetByName returns the set in the specified table if matching name is found.
func (cc *Conn) GetSetByName(t *Table, name string) (*Set, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	data, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(t.Name + "\x00")},
		{Type: unix.NFTA_SET_NAME, Data: []byte(name + "\x00")},
	})
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETSET),
			Flags: netlink.Request | netlink.Acknowledge,
		},
		Data: append(extraHeader(uint8(t.Family), 0), data...),
	}

	if _, err := conn.SendMessages([]netlink.Message{message}); err != nil {
		return nil, fmt.Errorf("SendMessages: %w", err)
	}

	reply, err := receiveAckAware(conn, message.Header.Flags)
	if err != nil {
		return nil, fmt.Errorf("receiveAckAware: %w", err)
	}

	if len(reply) != 1 {
		return nil, fmt.Errorf("receiveAckAware: expected to receive 1 message but got %d", len(reply))
	}
	rs, err := setsFromMsg(reply[0])
	if err != nil {
		return nil, err
	}
	rs.Table = &Table{Name: t.Name, Use: t.Use, Flags: t.Flags, Family: t.Family}

	return rs, nil
}

// GetSetElements returns the elements in the specified set.
func (cc *Conn) GetSetElements(s *Set) ([]SetElement, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	data, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_SET_TABLE, Data: []byte(s.Table.Name + "\x00")},
		{Type: unix.NFTA_SET_NAME, Data: []byte(s.Name + "\x00")},
	})
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETSETELEM),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Dump,
		},
		Data: append(extraHeader(uint8(s.Table.Family), 0), data...),
	}

	if _, err := conn.SendMessages([]netlink.Message{message}); err != nil {
		return nil, fmt.Errorf("SendMessages: %v", err)
	}

	reply, err := receiveAckAware(conn, message.Header.Flags)
	if err != nil {
		return nil, fmt.Errorf("receiveAckAware: %v", err)
	}
	var elems []SetElement
	for _, msg := range reply {
		s, err := elementsFromMsg(uint8(s.Table.Family), msg)
		if err != nil {
			return nil, err
		}
		elems = append(elems, s...)
	}

	return elems, nil
}
