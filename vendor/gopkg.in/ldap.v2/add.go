//
// https://tools.ietf.org/html/rfc4511
//
// AddRequest ::= [APPLICATION 8] SEQUENCE {
//      entry           LDAPDN,
//      attributes      AttributeList }
//
// AttributeList ::= SEQUENCE OF attribute Attribute

package ldap

import (
	"errors"
	"log"

	"gopkg.in/asn1-ber.v1"
)

type Attribute struct {
	attrType string
	attrVals []string
}

func (a *Attribute) encode() *ber.Packet {
	seq := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Attribute")
	seq.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, a.attrType, "Type"))
	set := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSet, nil, "AttributeValue")
	for _, value := range a.attrVals {
		set.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, value, "Vals"))
	}
	seq.AppendChild(set)
	return seq
}

type AddRequest struct {
	dn         string
	attributes []Attribute
}

func (a AddRequest) encode() *ber.Packet {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationAddRequest, nil, "Add Request")
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, a.dn, "DN"))
	attributes := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Attributes")
	for _, attribute := range a.attributes {
		attributes.AppendChild(attribute.encode())
	}
	request.AppendChild(attributes)
	return request
}

func (a *AddRequest) Attribute(attrType string, attrVals []string) {
	a.attributes = append(a.attributes, Attribute{attrType: attrType, attrVals: attrVals})
}

func NewAddRequest(dn string) *AddRequest {
	return &AddRequest{
		dn: dn,
	}

}

func (l *Conn) Add(addRequest *AddRequest) error {
	messageID := l.nextMessageID()
	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, messageID, "MessageID"))
	packet.AppendChild(addRequest.encode())

	l.Debug.PrintPacket(packet)

	channel, err := l.sendMessage(packet)
	if err != nil {
		return err
	}
	if channel == nil {
		return NewError(ErrorNetwork, errors.New("ldap: could not send message"))
	}
	defer l.finishMessage(messageID)

	l.Debug.Printf("%d: waiting for response", messageID)
	packetResponse, ok := <-channel
	if !ok {
		return NewError(ErrorNetwork, errors.New("ldap: channel closed"))
	}
	packet, err = packetResponse.ReadPacket()
	l.Debug.Printf("%d: got response %p", messageID, packet)
	if err != nil {
		return err
	}

	if l.Debug {
		if err := addLDAPDescriptions(packet); err != nil {
			return err
		}
		ber.PrintPacket(packet)
	}

	if packet.Children[1].Tag == ApplicationAddResponse {
		resultCode, resultDescription := getLDAPResultCode(packet)
		if resultCode != 0 {
			return NewError(resultCode, errors.New(resultDescription))
		}
	} else {
		log.Printf("Unexpected Response: %d", packet.Children[1].Tag)
	}

	l.Debug.Printf("%d: returning", messageID)
	return nil
}
