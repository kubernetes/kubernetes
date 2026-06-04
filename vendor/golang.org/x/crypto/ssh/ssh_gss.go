// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"encoding/asn1"
	"errors"
)

var krb5OID []byte

func init() {
	krb5OID, _ = asn1.Marshal(krb5Mesh)
}

// GSSAPIClient provides the API to plug-in GSSAPI authentication for client logins.
type GSSAPIClient interface {
	// InitSecContext initiates the establishment of a security context for GSS-API between the
	// ssh client and ssh server. Initially the token parameter should be specified as nil.
	// The routine may return a outputToken which should be transferred to
	// the ssh server, where the ssh server will present it to
	// AcceptSecContext. If no token need be sent, InitSecContext will indicate this by setting
	// needContinue to false. To complete the context
	// establishment, one or more reply tokens may be required from the ssh
	// server;if so, InitSecContext will return a needContinue which is true.
	// In this case, InitSecContext should be called again when the
	// reply token is received from the ssh server, passing the reply
	// token to InitSecContext via the token parameters.
	// See RFC 2743 section 2.2.1 and RFC 4462 section 3.4.
	InitSecContext(target string, token []byte, isGSSDelegCreds bool) (outputToken []byte, needContinue bool, err error)
	// GetMIC generates a cryptographic MIC for the SSH2 message, and places
	// the MIC in a token for transfer to the ssh server.
	// The contents of the MIC field are obtained by calling GSS_GetMIC()
	// over the following, using the GSS-API context that was just
	// established:
	//  string    session identifier
	//  byte      SSH_MSG_USERAUTH_REQUEST
	//  string    user name
	//  string    service
	//  string    "gssapi-with-mic"
	// See RFC 2743 section 2.3.1 and RFC 4462 3.5.
	GetMIC(micFiled []byte) ([]byte, error)
	// Whenever possible, it should be possible for
	// DeleteSecContext() calls to be successfully processed even
	// if other calls cannot succeed, thereby enabling context-related
	// resources to be released.
	// In addition to deleting established security contexts,
	// gss_delete_sec_context must also be able to delete "half-built"
	// security contexts resulting from an incomplete sequence of
	// InitSecContext()/AcceptSecContext() calls.
	// See RFC 2743 section 2.2.3.
	DeleteSecContext() error
}

// GSSAPIServer provides the API to plug in GSSAPI authentication for server logins.
type GSSAPIServer interface {
	// AcceptSecContext allows a remotely initiated security context between the application
	// and a remote peer to be established by the ssh client. The routine may return a
	// outputToken which should be transferred to the ssh client,
	// where the ssh client will present it to InitSecContext.
	// If no token need be sent, AcceptSecContext will indicate this
	// by setting the needContinue to false. To
	// complete the context establishment, one or more reply tokens may be
	// required from the ssh client. if so, AcceptSecContext
	// will return a needContinue which is true, in which case it
	// should be called again when the reply token is received from the ssh
	// client, passing the token to AcceptSecContext via the
	// token parameters.
	// The srcName return value is the authenticated username.
	// See RFC 2743 section 2.2.2 and RFC 4462 section 3.4.
	AcceptSecContext(token []byte) (outputToken []byte, srcName string, needContinue bool, err error)
	// VerifyMIC verifies that a cryptographic MIC, contained in the token parameter,
	// fits the supplied message is received from the ssh client.
	// See RFC 2743 section 2.3.2.
	VerifyMIC(micField []byte, micToken []byte) error
	// Whenever possible, it should be possible for
	// DeleteSecContext() calls to be successfully processed even
	// if other calls cannot succeed, thereby enabling context-related
	// resources to be released.
	// In addition to deleting established security contexts,
	// gss_delete_sec_context must also be able to delete "half-built"
	// security contexts resulting from an incomplete sequence of
	// InitSecContext()/AcceptSecContext() calls.
	// See RFC 2743 section 2.2.3.
	DeleteSecContext() error
}

var (
	// OpenSSH supports Kerberos V5 mechanism only for GSS-API authentication,
	// so we also support the krb5 mechanism only.
	// See RFC 1964 section 1.
	krb5Mesh = asn1.ObjectIdentifier{1, 2, 840, 113554, 1, 2, 2}
)

// The GSS-API authentication method is initiated when the client sends an SSH_MSG_USERAUTH_REQUEST
// See RFC 4462 section 3.2.
type userAuthRequestGSSAPI struct {
	N    uint32
	OIDS []asn1.ObjectIdentifier
}

func parseGSSAPIPayload(payload []byte) (*userAuthRequestGSSAPI, error) {
	n, rest, ok := parseUint32(payload)
	if !ok {
		return nil, errors.New("parse uint32 failed")
	}
	// Each ASN.1 encoded OID must have a minimum
	// of 2 bytes; 64 maximum mechanisms is an
	// arbitrary, but reasonable ceiling.
	const maxMechs = 64
	if n > maxMechs || int(n)*2 > len(rest) {
		return nil, errors.New("invalid mechanism count")
	}
	s := &userAuthRequestGSSAPI{
		N:    n,
		OIDS: make([]asn1.ObjectIdentifier, n),
	}
	for i := 0; i < int(n); i++ {
		var (
			desiredMech []byte
			err         error
		)
		desiredMech, rest, ok = parseString(rest)
		if !ok {
			return nil, errors.New("parse string failed")
		}
		if rest, err = asn1.Unmarshal(desiredMech, &s.OIDS[i]); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// See RFC 4462 section 3.6.
func buildMIC(sessionID string, username string, service string, authMethod string) []byte {
	out := make([]byte, 0, 0)
	out = appendString(out, sessionID)
	out = append(out, msgUserAuthRequest)
	out = appendString(out, username)
	out = appendString(out, service)
	out = appendString(out, authMethod)
	return out
}
