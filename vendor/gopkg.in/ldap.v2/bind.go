// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ldap

import (
	"errors"

	"gopkg.in/asn1-ber.v1"
)

// SimpleBindRequest represents a username/password bind operation
type SimpleBindRequest struct {
	// Username is the name of the Directory object that the client wishes to bind as
	Username string
	// Password is the credentials to bind with
	Password string
	// Controls are optional controls to send with the bind request
	Controls []Control
}

// SimpleBindResult contains the response from the server
type SimpleBindResult struct {
	Controls []Control
}

// NewSimpleBindRequest returns a bind request
func NewSimpleBindRequest(username string, password string, controls []Control) *SimpleBindRequest {
	return &SimpleBindRequest{
		Username: username,
		Password: password,
		Controls: controls,
	}
}

func (bindRequest *SimpleBindRequest) encode() *ber.Packet {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, bindRequest.Username, "User Name"))
	request.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, bindRequest.Password, "Password"))

	request.AppendChild(encodeControls(bindRequest.Controls))

	return request
}

// SimpleBind performs the simple bind operation defined in the given request
func (l *Conn) SimpleBind(simpleBindRequest *SimpleBindRequest) (*SimpleBindResult, error) {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	encodedBindRequest := simpleBindRequest.encode()
	packet.AppendChild(encodedBindRequest)

	if l.Debug {
		ber.PrintPacket(packet)
	}

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return nil, NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return nil, err
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			return nil, err
		}
		ber.PrintPacket(packet)
	}

	result := &SimpleBindResult{
		Controls: make([]Control, 0),
	}

	if len(packet.Children) == 3 {
		for _, child := range packet.Children[2].Children {
			result.Controls = append(result.Controls, DecodeControl(child))
		}
	}

	resultCode, resultDescription := getLDAPResultCode(packet)
	if resultCode != 0 {
		return result, NewError(resultCode, errors.New(resultDescription))
	}

	return result, nil
}

// Bind performs a bind with the given username and password
func (l *Conn) Bind(username, password string) error {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	bindRequest := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	bindRequest.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	bindRequest.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, username, "User Name"))
	bindRequest.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, password, "Password"))
	packet.AppendChild(bindRequest)

	if l.Debug {
		ber.PrintPacket(packet)
	}

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return err
	}
	defer l.finishMessage(msgCtx)

	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return err
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			return err
		}
		ber.PrintPacket(packet)
	}

	resultCode, resultDescription := getLDAPResultCode(packet)
	if resultCode != 0 {
		return NewError(resultCode, errors.New(resultDescription))
	}

	return nil
}
