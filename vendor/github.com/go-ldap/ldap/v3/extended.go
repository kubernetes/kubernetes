package ldap

import (
	"fmt"
	ber "github.com/go-asn1-ber/asn1-ber"
)

// ExtendedRequest represents an extended request to send to the server
// See: https://www.rfc-editor.org/rfc/rfc4511#section-4.12
type ExtendedRequest struct {
	// ExtendedRequest ::= [APPLICATION 23] SEQUENCE {
	// 	requestName      [0] LDAPOID,
	// 	requestValue     [1] OCTET STRING OPTIONAL }

	Name     string
	Value    *ber.Packet
	Controls []Control
}

// NewExtendedRequest returns a new ExtendedRequest. The value can be
// nil depending on the type of request
func NewExtendedRequest(name string, value *ber.Packet) *ExtendedRequest {
	return &ExtendedRequest{
		Name:  name,
		Value: value,
	}
}

func (er ExtendedRequest) appendTo(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationExtendedRequest, nil, "Extended Request")
	pkt.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, ber.TagEOC, er.Name, "Extended Request Name"))
	if er.Value != nil {
		pkt.AppendChild(er.Value)
	}
	envelope.AppendChild(pkt)
	if len(er.Controls) > 0 {
		envelope.AppendChild(encodeControls(er.Controls))
	}
	return nil
}

// ExtendedResponse represents the response from the directory server
// after sending an extended request
// See: https://www.rfc-editor.org/rfc/rfc4511#section-4.12
type ExtendedResponse struct {
	// ExtendedResponse ::= [APPLICATION 24] SEQUENCE {
	//   COMPONENTS OF LDAPResult,
	//   responseName     [10] LDAPOID OPTIONAL,
	//   responseValue    [11] OCTET STRING OPTIONAL }

	Name     string
	Value    *ber.Packet
	Controls []Control
}

// Extended performs an extended request. The resulting
// ExtendedResponse may return a value in the form of a *ber.Packet
func (l *Conn) Extended(er *ExtendedRequest) (*ExtendedResponse, error) {
	msgCtx, err := l.doRequest(er)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return nil, err
	}
	if err = GetLDAPError(packet); err != nil {
		return nil, err
	}

	if len(packet.Children[1].Children) < 4 {
		return nil, fmt.Errorf(
			"ldap: malformed extended response: expected 4 children, got %d",
			len(packet.Children),
		)
	}

	response := &ExtendedResponse{
		Name:     packet.Children[1].Children[3].Data.String(),
		Controls: make([]Control, 0),
	}

	if len(packet.Children) == 3 {
		for _, child := range packet.Children[2].Children {
			decodedChild, decodeErr := DecodeControl(child)
			if decodeErr != nil {
				return nil, fmt.Errorf("failed to decode child control: %s", decodeErr)
			}
			response.Controls = append(response.Controls, decodedChild)
		}
	}

	if len(packet.Children[1].Children) == 5 {
		response.Value = packet.Children[1].Children[4]
	}

	return response, nil
}
