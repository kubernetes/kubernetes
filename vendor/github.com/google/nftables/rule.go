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
	newRuleHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_NEWRULE)
	delRuleHeaderType = netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_DELRULE)
)

type ruleOperation uint32

// Possible PayloadOperationType values.
const (
	operationAdd ruleOperation = iota
	operationInsert
	operationReplace
)

// A Rule does something with a packet. See also
// https://wiki.nftables.org/wiki-nftables/index.php/Simple_rule_management
type Rule struct {
	Table    *Table
	Chain    *Chain
	Position uint64
	Handle   uint64
	// The list of possible flags are specified by nftnl_rule_attr, see
	// https://git.netfilter.org/libnftnl/tree/include/libnftnl/rule.h#n21
	// Current nftables go implementation supports only
	// NFTNL_RULE_POSITION flag for setting rule at position 0
	Flags    uint32
	Exprs    []expr.Any
	UserData []byte
}

// GetRule returns the rules in the specified table and chain.
//
// Deprecated: use GetRules instead.
func (cc *Conn) GetRule(t *Table, c *Chain) ([]*Rule, error) {
	return cc.GetRules(t, c)
}

// GetRules returns the rules in the specified table and chain.
func (cc *Conn) GetRules(t *Table, c *Chain) ([]*Rule, error) {
	conn, closer, err := cc.netlinkConn()
	if err != nil {
		return nil, err
	}
	defer func() { _ = closer() }()

	data, err := netlink.MarshalAttributes([]netlink.Attribute{
		{Type: unix.NFTA_RULE_TABLE, Data: []byte(t.Name + "\x00")},
		{Type: unix.NFTA_RULE_CHAIN, Data: []byte(c.Name + "\x00")},
	})
	if err != nil {
		return nil, err
	}

	message := netlink.Message{
		Header: netlink.Header{
			Type:  netlink.HeaderType((unix.NFNL_SUBSYS_NFTABLES << 8) | unix.NFT_MSG_GETRULE),
			Flags: netlink.Request | netlink.Acknowledge | netlink.Dump | unix.NLM_F_ECHO,
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
	var rules []*Rule
	for _, msg := range reply {
		r, err := ruleFromMsg(t.Family, msg)
		if err != nil {
			return nil, err
		}
		rules = append(rules, r)
	}

	return rules, nil
}

// AddRule adds the specified Rule
func (cc *Conn) newRule(r *Rule, op ruleOperation) *Rule {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	exprAttrs := make([]netlink.Attribute, len(r.Exprs))
	for idx, expr := range r.Exprs {
		exprAttrs[idx] = netlink.Attribute{
			Type: unix.NLA_F_NESTED | unix.NFTA_LIST_ELEM,
			Data: cc.marshalExpr(byte(r.Table.Family), expr),
		}
	}

	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_RULE_TABLE, Data: []byte(r.Table.Name + "\x00")},
		{Type: unix.NFTA_RULE_CHAIN, Data: []byte(r.Chain.Name + "\x00")},
	})

	if r.Handle != 0 {
		data = append(data, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NFTA_RULE_HANDLE, Data: binaryutil.BigEndian.PutUint64(r.Handle)},
		})...)
	}

	data = append(data, cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NLA_F_NESTED | unix.NFTA_RULE_EXPRESSIONS, Data: cc.marshalAttr(exprAttrs)},
	})...)

	if compatPolicy, err := getCompatPolicy(r.Exprs); err != nil {
		cc.setErr(err)
	} else if compatPolicy != nil {
		data = append(data, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NLA_F_NESTED | unix.NFTA_RULE_COMPAT, Data: cc.marshalAttr([]netlink.Attribute{
				{Type: unix.NFTA_RULE_COMPAT_PROTO, Data: binaryutil.BigEndian.PutUint32(compatPolicy.Proto)},
				{Type: unix.NFTA_RULE_COMPAT_FLAGS, Data: binaryutil.BigEndian.PutUint32(compatPolicy.Flag & nft_RULE_COMPAT_F_MASK)},
			})},
		})...)
	}

	msgData := []byte{}

	msgData = append(msgData, data...)
	var flags netlink.HeaderFlags
	if r.UserData != nil {
		msgData = append(msgData, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NFTA_RULE_USERDATA, Data: r.UserData},
		})...)
	}

	switch op {
	case operationAdd:
		flags = netlink.Request | netlink.Acknowledge | netlink.Create | unix.NLM_F_ECHO | unix.NLM_F_APPEND
	case operationInsert:
		flags = netlink.Request | netlink.Acknowledge | netlink.Create | unix.NLM_F_ECHO
	case operationReplace:
		flags = netlink.Request | netlink.Acknowledge | netlink.Replace | unix.NLM_F_ECHO | unix.NLM_F_REPLACE
	}

	if r.Position != 0 || (r.Flags&(1<<unix.NFTA_RULE_POSITION)) != 0 {
		msgData = append(msgData, cc.marshalAttr([]netlink.Attribute{
			{Type: unix.NFTA_RULE_POSITION, Data: binaryutil.BigEndian.PutUint64(r.Position)},
		})...)
	}

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  newRuleHeaderType,
			Flags: flags,
		},
		Data: append(extraHeader(uint8(r.Table.Family), 0), msgData...),
	})

	return r
}

func (cc *Conn) ReplaceRule(r *Rule) *Rule {
	return cc.newRule(r, operationReplace)
}

func (cc *Conn) AddRule(r *Rule) *Rule {
	if r.Handle != 0 {
		return cc.newRule(r, operationReplace)
	}

	return cc.newRule(r, operationAdd)
}

func (cc *Conn) InsertRule(r *Rule) *Rule {
	if r.Handle != 0 {
		return cc.newRule(r, operationReplace)
	}

	return cc.newRule(r, operationInsert)
}

// DelRule deletes the specified Rule, rule's handle cannot be 0
func (cc *Conn) DelRule(r *Rule) error {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	data := cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_RULE_TABLE, Data: []byte(r.Table.Name + "\x00")},
		{Type: unix.NFTA_RULE_CHAIN, Data: []byte(r.Chain.Name + "\x00")},
	})
	if r.Handle == 0 {
		return fmt.Errorf("rule's handle cannot be 0")
	}
	data = append(data, cc.marshalAttr([]netlink.Attribute{
		{Type: unix.NFTA_RULE_HANDLE, Data: binaryutil.BigEndian.PutUint64(r.Handle)},
	})...)
	flags := netlink.Request | netlink.Acknowledge

	cc.messages = append(cc.messages, netlink.Message{
		Header: netlink.Header{
			Type:  delRuleHeaderType,
			Flags: flags,
		},
		Data: append(extraHeader(uint8(r.Table.Family), 0), data...),
	})

	return nil
}

func ruleFromMsg(fam TableFamily, msg netlink.Message) (*Rule, error) {
	if got, want1, want2 := msg.Header.Type, newRuleHeaderType, delRuleHeaderType; got != want1 && got != want2 {
		return nil, fmt.Errorf("unexpected header type: got %v, want %v or %v", msg.Header.Type, want1, want2)
	}
	ad, err := netlink.NewAttributeDecoder(msg.Data[4:])
	if err != nil {
		return nil, err
	}
	ad.ByteOrder = binary.BigEndian
	var r Rule
	for ad.Next() {
		switch ad.Type() {
		case unix.NFTA_RULE_TABLE:
			r.Table = &Table{
				Name:   ad.String(),
				Family: fam,
			}
		case unix.NFTA_RULE_CHAIN:
			r.Chain = &Chain{Name: ad.String()}
		case unix.NFTA_RULE_EXPRESSIONS:
			ad.Do(func(b []byte) error {
				exprs, err := parseexprfunc.ParseExprMsgFunc(byte(fam), b)
				if err != nil {
					return err
				}
				r.Exprs = make([]expr.Any, len(exprs))
				for i := range exprs {
					r.Exprs[i] = exprs[i].(expr.Any)
				}
				return nil
			})
		case unix.NFTA_RULE_POSITION:
			r.Position = ad.Uint64()
		case unix.NFTA_RULE_HANDLE:
			r.Handle = ad.Uint64()
		case unix.NFTA_RULE_USERDATA:
			r.UserData = ad.Bytes()
		}
	}
	return &r, ad.Err()
}
