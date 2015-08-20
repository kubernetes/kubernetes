// This file contains the password modify extended operation as specified in rfc 3062
//
// https://tools.ietf.org/html/rfc3062
//

package ldap

import (
	"errors"
	"fmt"

	"gopkg.in/asn1-ber.v1"
)

const (
	passwordModifyOID = "1.3.6.1.4.1.4203.1.11.1"
)

type PasswordModifyRequest struct {
	UserIdentity string
	OldPassword  string
	NewPassword  string
}

type PasswordModifyResult struct {
	GeneratedPassword string
}

func (r *PasswordModifyRequest) encode() (*ber.Packet, error) {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationExtendedRequest, nil, "Password Modify Extended Operation")
	request.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, passwordModifyOID, "Extended Request Name: Password Modify OID"))
	extendedRequestValue := ber.Encode(ber.ClassContext, ber.TypePrimitive, 1, nil, "Extended Request Value: Password Modify Request")
	passwordModifyRequestValue := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "Password Modify Request")
	if r.UserIdentity != "" {
		passwordModifyRequestValue.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, r.UserIdentity, "User Identity"))
	}
	if r.OldPassword != "" {
		passwordModifyRequestValue.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 1, r.OldPassword, "Old Password"))
	}
	if r.NewPassword != "" {
		passwordModifyRequestValue.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 2, r.NewPassword, "New Password"))
	}

	extendedRequestValue.AppendChild(passwordModifyRequestValue)
	request.AppendChild(extendedRequestValue)

	return request, nil
}

// Create a new PasswordModifyRequest
//
// According to the RFC 3602:
// userIdentity is a string representing the user associated with the request.
// This string may or may not be an LDAPDN (RFC 2253).
// If userIdentity is empty then the operation will act on the user associated
// with the session.
//
// oldPassword is the current user's password, it can be empty or it can be
// needed depending on the session user access rights (usually an administrator
// can change a user's password without knowing the current one) and the
// password policy (see pwdSafeModify password policy's attribute)
//
// newPassword is the desired user's password. If empty the server can return
// an error or generate a new password that will be available in the
// PasswordModifyResult.GeneratedPassword
//
func NewPasswordModifyRequest(userIdentity string, oldPassword string, newPassword string) *PasswordModifyRequest {
	return &PasswordModifyRequest{
		UserIdentity: userIdentity,
		OldPassword:  oldPassword,
		NewPassword:  newPassword,
	}
}

func (l *Conn) PasswordModify(passwordModifyRequest *PasswordModifyRequest) (*PasswordModifyResult, error) {
	messageID := l.nextMessageID()

	packet := ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
	packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, messageID, "MessageID"))

	encodedPasswordModifyRequest, err := passwordModifyRequest.encode()
	if err != nil {
		return nil, err
	}
	packet.AppendChild(encodedPasswordModifyRequest)

	l.Debug.PrintPacket(packet)

	channel, err := l.sendMessage(packet)
	if err != nil {
		return nil, err
	}
	if channel == nil {
		return nil, NewError(ErrorNetwork, errors.New("ldap: could not send message"))
	}
	defer l.finishMessage(messageID)

	result := &PasswordModifyResult{}

	l.Debug.Printf("%d: waiting for response", messageID)
	packet = <-channel
	l.Debug.Printf("%d: got response %p", messageID, packet)

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
		resultCode, resultDescription := getLDAPResultCode(packet)
		if resultCode != 0 {
			return nil, NewError(resultCode, errors.New(resultDescription))
		}
	} else {
		return nil, NewError(ErrorUnexpectedResponse, fmt.Errorf("Unexpected Response: %d", packet.Children[1].Tag))
	}

	extendedResponse := packet.Children[1]
	for _, child := range extendedResponse.Children {
		if child.Tag == 11 {
			passwordModifyReponseValue := ber.DecodePacket(child.Data.Bytes())
			if len(passwordModifyReponseValue.Children) == 1 {
				if passwordModifyReponseValue.Children[0].Tag == 0 {
					result.GeneratedPassword = ber.DecodeString(passwordModifyReponseValue.Children[0].Data.Bytes())
				}
			}
		}
	}

	return result, nil
}
