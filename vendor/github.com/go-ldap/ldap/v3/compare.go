package ldap

import (
	"fmt"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// CompareRequest represents an LDAP CompareRequest operation.
type CompareRequest struct {
	DN        string
	Attribute string
	Value     string
}

func (req *CompareRequest) appendTo(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationCompareRequest, nil, "Compare Request")
	pkt.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, req.DN, "DN"))

	ava := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "AttributeValueAssertion")
	ava.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, req.Attribute, "AttributeDesc"))
	ava.AppendChild(ber.Encode(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, req.Value, "AssertionValue"))

	pkt.AppendChild(ava)

	envelope.AppendChild(pkt)

	return nil
}

// Compare checks to see if the attribute of the dn matches value. Returns true if it does otherwise
// false with any error that occurs if any.
func (l *Conn) Compare(dn, attribute, value string) (bool, error) {
	msgCtx, err := l.doRequest(&CompareRequest{
		DN:        dn,
		Attribute: attribute,
		Value:     value,
	})
	if err != nil {
		return false, err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return false, err
	}

	if packet.Children[1].Tag == ApplicationCompareResponse {
		err := GetLDAPError(packet)

		switch {
		case IsErrorWithCode(err, LDAPResultCompareTrue):
			return true, nil
		case IsErrorWithCode(err, LDAPResultCompareFalse):
			return false, nil
		default:
			return false, err
		}
	}
	return false, fmt.Errorf("unexpected Response: %d", packet.Children[1].Tag)
}
