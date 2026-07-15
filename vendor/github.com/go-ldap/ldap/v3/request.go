package ldap

import (
	"errors"

	ber "github.com/go-asn1-ber/asn1-ber"
)

var (
	errRespChanClosed = errors.New("ldap: response channel closed")
	errCouldNotRetMsg = errors.New("ldap: could not retrieve message")
	// ErrNilConnection is returned if doRequest is called with a nil connection.
	ErrNilConnection = errors.New("ldap: conn is nil, expected net.Conn")
)

type request interface {
	appendTo(*ber.Packet) error
}

type requestFunc func(*ber.Packet) error

func (f requestFunc) appendTo(p *ber.Packet) error {
	return f(p)
}

func (l *Conn) doRequest(req request) (*messageContext, error) {
	if l == nil || l.conn == nil {
		return nil, ErrNilConnection
	}

	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	if err := req.appendTo(packet); err != nil {
		return nil, err
	}

	if l.Debug {
		l.Debug.PrintPacket(packet)
	}

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return nil, err
	}
	l.Debug.Printf("%d: returning", msgCtx.id)
	return msgCtx, nil
}

func (l *Conn) readPacket(msgCtx *messageContext) (*ber.Packet, error) {
	l.Debug.Printf("%d: waiting for response", msgCtx.id)
	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return nil, NewError(ErrorNetwork, errRespChanClosed)
	}
	packet, err := packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return nil, err
	}

	if packet == nil {
		return nil, NewError(ErrorNetwork, errCouldNotRetMsg)
	}

	if l.Debug {
		if err = addLDAPDescriptions(packet); err != nil {
			return nil, err
		}
		l.Debug.PrintPacket(packet)
	}
	return packet, nil
}

func getReferral(err error, packet *ber.Packet) (referral string) {
	if !IsErrorWithCode(err, LDAPResultReferral) {
		return ""
	}

	if len(packet.Children) < 2 {
		return ""
	}

	// The packet Tag itself (of child 2) is generally a ber.TagObjectDescriptor with referrals however OpenLDAP
	// seemingly returns a ber.Tag.GeneralizedTime. Every currently tested LDAP server which returns referrals returns
	// an ASN.1 BER packet with the Type of ber.TypeConstructed and Class of ber.ClassApplication however. Thus this
	// check expressly checks these fields instead.
	//
	// Related Issues:
	//   - https://github.com/authelia/authelia/issues/4199 (downstream)
	if len(packet.Children[1].Children) == 0 || (packet.Children[1].TagType != ber.TypeConstructed || packet.Children[1].ClassType != ber.ClassApplication) {
		return ""
	}

	var ok bool

	for _, child := range packet.Children[1].Children {
		// The referral URI itself should be contained within a child which has a Tag of ber.BitString or
		// ber.TagPrintableString, and the Type of ber.TypeConstructed and the Class of ClassContext. As soon as any of
		// these conditions is not true  we can skip this child.
		if (child.Tag != ber.TagBitString && child.Tag != ber.TagPrintableString) || child.TagType != ber.TypeConstructed || child.ClassType != ber.ClassContext {
			continue
		}

		if referral, ok = child.Children[0].Value.(string); ok {
			return referral
		}
	}

	return ""
}
