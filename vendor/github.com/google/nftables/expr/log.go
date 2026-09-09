// Copyright 2019 Google LLC. All Rights Reserved.
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

package expr

import (
	"encoding/binary"

	"github.com/google/nftables/binaryutil"
	"github.com/mdlayher/netlink"
	"golang.org/x/sys/unix"
)

type LogLevel uint32

const (
	// See https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_tables.h?id=5b364657a35f4e4cd5d220ba2a45303d729c8eca#n1226
	LogLevelEmerg LogLevel = iota
	LogLevelAlert
	LogLevelCrit
	LogLevelErr
	LogLevelWarning
	LogLevelNotice
	LogLevelInfo
	LogLevelDebug
	LogLevelAudit
)

type LogFlags uint32

const (
	// See https://git.netfilter.org/nftables/tree/include/linux/netfilter/nf_log.h?id=5b364657a35f4e4cd5d220ba2a45303d729c8eca
	LogFlagsTCPSeq LogFlags = 0x01 << iota
	LogFlagsTCPOpt
	LogFlagsIPOpt
	LogFlagsUID
	LogFlagsNFLog
	LogFlagsMACDecode
	LogFlagsMask LogFlags = 0x2f
)

// Log defines type for NFT logging
// See https://git.netfilter.org/libnftnl/tree/src/expr/log.c?id=09456c720e9c00eecc08e41ac6b7c291b3821ee5#n25
type Log struct {
	Level LogLevel
	// Refers to log flags (flags all, flags ip options, ...)
	Flags LogFlags
	// Equivalent to expression flags.
	// Indicates that an option is set by setting a bit
	// on index referred by the NFTA_LOG_* value.
	// See https://cs.opensource.google/go/x/sys/+/3681064d:unix/ztypes_linux.go;l=2126;drc=3681064d51587c1db0324b3d5c23c2ddbcff6e8f
	Key        uint32
	Snaplen    uint32
	Group      uint16
	QThreshold uint16
	// Log prefix string content
	Data []byte
}

func (e *Log) marshal(fam byte) ([]byte, error) {
	data, err := e.marshalData(fam)
	if err != nil {
		return nil, err
	}

	return netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_EXPR_NAME, Data: []byte("log\x00")},
		{Type: unix.NLA_F_NESTED | unix.NFTA_EXPR_DATA, Data: data},
	})
}

func (e *Log) marshalData(fam byte) ([]byte, error) {
	// Per https://git.netfilter.org/libnftnl/tree/src/expr/log.c?id=09456c720e9c00eecc08e41ac6b7c291b3821ee5#n129
	attrs := make([]netlink.Attribute, 0)
	if e.Key&(1<<unix.NFTA_LOG_GROUP) != 0 {
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_GROUP,
			Data: binaryutil.BigEndian.PutUint16(e.Group),
		})
	}
	if e.Key&(1<<unix.NFTA_LOG_PREFIX) != 0 {
		prefix := append(e.Data, '\x00')
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_PREFIX,
			Data: prefix,
		})
	}
	if e.Key&(1<<unix.NFTA_LOG_SNAPLEN) != 0 {
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_SNAPLEN,
			Data: binaryutil.BigEndian.PutUint32(e.Snaplen),
		})
	}
	if e.Key&(1<<unix.NFTA_LOG_QTHRESHOLD) != 0 {
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_QTHRESHOLD,
			Data: binaryutil.BigEndian.PutUint16(e.QThreshold),
		})
	}
	if e.Key&(1<<unix.NFTA_LOG_LEVEL) != 0 {
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_LEVEL,
			Data: binaryutil.BigEndian.PutUint32(uint32(e.Level)),
		})
	}
	if e.Key&(1<<unix.NFTA_LOG_FLAGS) != 0 {
		attrs = append(attrs, netlink.Attribute{
			Type: unix.NFTA_LOG_FLAGS,
			Data: binaryutil.BigEndian.PutUint32(uint32(e.Flags)),
		})
	}

	return netlink.MarshalAttributes(attrs)
}

func (e *Log) unmarshal(fam byte, data []byte) error {
	ad, err := netlink.NewAttributeDecoder(data)
	if err != nil {
		return err
	}

	ad.ByteOrder = binary.BigEndian
	for ad.Next() {
		e.Key |= 1 << uint32(ad.Type())
		data := ad.Bytes()
		switch ad.Type() {
		case unix.NFTA_LOG_GROUP:
			e.Group = binaryutil.BigEndian.Uint16(data)
		case unix.NFTA_LOG_PREFIX:
			// Getting rid of \x00 at the end of string
			e.Data = data[:len(data)-1]
		case unix.NFTA_LOG_SNAPLEN:
			e.Snaplen = binaryutil.BigEndian.Uint32(data)
		case unix.NFTA_LOG_QTHRESHOLD:
			e.QThreshold = binaryutil.BigEndian.Uint16(data)
		case unix.NFTA_LOG_LEVEL:
			e.Level = LogLevel(binaryutil.BigEndian.Uint32(data))
		case unix.NFTA_LOG_FLAGS:
			e.Flags = LogFlags(binaryutil.BigEndian.Uint32(data))
		}
	}
	return ad.Err()
}
