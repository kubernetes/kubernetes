package ldap

import (
	"errors"

	ber "github.com/go-asn1-ber/asn1-ber"
)

// ErrConnUnbound is returned when Unbind is called on an already closing connection.
var ErrConnUnbound = NewError(ErrorNetwork, errors.New("ldap: connection is closed"))

type unbindRequest struct{}

func (unbindRequest) appendTo(envelope *ber.Packet) error {
	envelope.AppendChild(ber.Encode(ber.ClassApplication, ber.TypePrimitive, ApplicationUnbindRequest, nil, ApplicationMap[ApplicationUnbindRequest]))
	return nil
}

// Unbind will perform an unbind request. The Unbind operation
// should be thought of as the "quit" operation.
// See https://datatracker.ietf.org/doc/html/rfc4511#section-4.3
func (l *Conn) Unbind() error {
	if l.IsClosing() {
		return ErrConnUnbound
	}

	_, err := l.doRequest(unbindRequest{})
	if err != nil {
		return err
	}

	// Sending an unbindRequest will make the connection unusable.
	// Pending requests will fail with:
	// LDAP Result Code 200 "Network Error": ldap: response channel closed
	l.Close()

	return nil
}
