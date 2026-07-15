package ldap

import (
	"fmt"
	ber "github.com/go-asn1-ber/asn1-ber"
)

// DelRequest implements an LDAP deletion request
type DelRequest struct {
	// DN is the name of the directory entry to delete
	DN string
	// Controls hold optional controls to send with the request
	Controls []Control
}

func (req *DelRequest) appendTo(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypePrimitive, ApplicationDelRequest, req.DN, "Del Request")
	pkt.Data.Write([]byte(req.DN))

	envelope.AppendChild(pkt)
	if len(req.Controls) > 0 {
		envelope.AppendChild(encodeControls(req.Controls))
	}

	return nil
}

// NewDelRequest creates a delete request for the given DN and controls
func NewDelRequest(DN string, Controls []Control) *DelRequest {
	return &DelRequest{
		DN:       DN,
		Controls: Controls,
	}
}

// Del executes the given delete request
func (l *Conn) Del(delRequest *DelRequest) error {
	msgCtx, err := l.doRequest(delRequest)
	if err != nil {
		return err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return err
	}

	if packet.Children[1].Tag == ApplicationDelResponse {
		err := GetLDAPError(packet)
		if err != nil {
			return err
		}
	} else {
		return fmt.Errorf("ldap: unexpected response: %d", packet.Children[1].Tag)
	}

	return nil
}
