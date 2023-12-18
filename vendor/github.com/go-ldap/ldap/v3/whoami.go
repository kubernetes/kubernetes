package ldap

// This file contains the "Who Am I?" extended operation as specified in rfc 4532
//
// https://tools.ietf.org/html/rfc4532

import (
	"errors"
	"fmt"

	ber "github.com/go-asn1-ber/asn1-ber"
)

type whoAmIRequest bool

// WhoAmIResult is returned by the WhoAmI() call
type WhoAmIResult struct {
	AuthzID string
}

func (r whoAmIRequest) encode() (*ber.Packet, error) {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationExtendedRequest, nil, "Who Am I? Extended Operation")
	request.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, ControlTypeWhoAmI, "Extended Request Name: Who Am I? OID"))
	return request, nil
}

// WhoAmI returns the authzId the server thinks we are, you may pass controls
// like a Proxied Authorization control
func (l *Conn) WhoAmI(controls []Control) (*WhoAmIResult, error) {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	req := whoAmIRequest(true)
	encodedWhoAmIRequest, err := req.encode()
	if err != nil {
		return nil, err
	}
	packet.AppendChild(encodedWhoAmIRequest)

	if len(controls) != 0 {
		packet.AppendChild(encodeControls(controls))
	}

	l.Debug.PrintPacket(packet)

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	result := &WhoAmIResult{}

	l.Debug.Printf("%d: waiting for response", msgCtx.id)
	packetResponse, ok := <-msgCtx.responses
	if !ok {
		return nil, NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if err != nil {
		return nil, err
	}

	if packet == nil {
		return nil, NewError(ErrorNetwork, errors.New("ldap: could not retrieve message"))
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			return nil, err
		}
		ber.PrintPacket(packet)
	}

	if packet.Children[1].Tag == ApplicationExtendedResponse {
		if err := GetLDAPError(packet); err != nil {
			return nil, err
		}
	} else {
		return nil, NewError(ErrorUnexpectedResponse, fmt.Errorf("Unexpected Response: %d", packet.Children[1].Tag))
	}

	extendedResponse := packet.Children[1]
	for _, child := range extendedResponse.Children {
		if child.Tag == 11 {
			result.AuthzID = ber.DecodeString(child.Data.Bytes())
		}
	}

	return result, nil
}
