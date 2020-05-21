//
// https://tools.ietf.org/html/rfc4511
//
// DelRequest ::= [APPLICATION 10] LDAPDN

package ldap

import (
	"errors"
	"log"

	"gopkg.in/asn1-ber.v1"
)

// DelRequest implements an LDAP deletion request
type DelRequest struct {
	// DN is the name of the directory entry to delete
	DN string
	// Controls hold optional controls to send with the request
	Controls []Control
}

func (d DelRequest) encode() *ber.Packet {
	request := ber.Encode(ber.ClassApplication, ber.TypePrimitive, ApplicationDelRequest, d.DN, "Del Request")
	request.Data.Write([]byte(d.DN))
	return request
}

// NewDelRequest creates a delete request for the given DN and controls
func NewDelRequest(DN string,
	Controls []Control) *DelRequest {
	return &DelRequest{
		DN:       DN,
		Controls: Controls,
	}
}

// Del executes the given delete request
func (l *Conn) Del(delRequest *DelRequest) error {
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))
	packet.AppendChild(delRequest.encode())
	if delRequest.Controls != nil {
		packet.AppendChild(encodeControls(delRequest.Controls))
	}

	l.Debug.PrintPacket(packet)

	msgCtx, err := l.sendMessage(packet)
	if err != nil {
		return err
	}
	defer l.finishMessage(msgCtx)

	l.Debug.Printf("%d: waiting for response", msgCtx.id)
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

	if packet.Children[1].Tag == ApplicationDelResponse {
		resultCode, resultDescription := getLDAPResultCode(packet)
		if resultCode != 0 {
			return NewError(resultCode, errors.New(resultDescription))
		}
	} else {
		log.Printf("Unexpected Response: %d", packet.Children[1].Tag)
	}

	l.Debug.Printf("%d: returning", msgCtx.id)
	return nil
}
