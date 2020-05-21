// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// File contains Compare functionality
//
// https://tools.ietf.org/html/rfc4511
//
// CompareRequest ::= [APPLICATION 14] SEQUENCE {
//              entry           LDAPDN,
//              ava             AttributeValueAssertion }
//
// AttributeValueAssertion ::= SEQUENCE {
//              attributeDesc   AttributeDescription,
//              assertionValue  AssertionValue }
//
// AttributeDescription ::= LDAPString
//                         -- Constrained to <attributedescription>
//                         -- [RFC4512]
//
// AttributeValue ::= OCTET STRING
//

package ldap

import (
	"errors"
	"fmt"

	"gopkg.in/asn1-ber.v1"
)

// Compare checks to see if the attribute of the dn matches value. Returns true if it does otherwise
// false with any error that occurs if any.
func (l *Conn) Compare(dn, attribute, value string) (bool, error) {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))

	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationCompareRequest, nil, "Compare Request")
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, dn, "DN"))

	ava := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "AttributeValueAssertion")
	ava.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, attribute, "AttributeDesc"))
	ava.AppendChild(ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagOctetString, value, "AssertionValue"))
	request.AppendChild(ava)
	packet.AppendChild(request)

	l.Debug.PrintPacket(packet)

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return false, err
	}
	defer l.finishMessage(msgCtx)

	l.Debug.Printf("%d: waiting for response", msgCtx.id)
	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return false, NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return false, err
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			return false, err
		}
		ber.PrintPacket(packet)
	}

	if packet.Children[1].Tag == ApplicationCompareResponse {
		resultCode, resultDescription := getLDAPResultCode(packet)
		if resultCode == LDAPResultCompareTrue {
			return true, nil
		} else if resultCode == LDAPResultCompareFalse {
			return false, nil
		} else {
			return false, NewError(resultCode, errors.New(resultDescription))
		}
	}
	return false, fmt.Errorf("Unexpected Response: %d", packet.Children[1].Tag)
}
