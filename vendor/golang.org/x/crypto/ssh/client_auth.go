// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strings"
)

type authResult int

const (
	authFailure authResult = iota
	authPartialSuccess
	authSuccess
)

// clientAuthenticate authenticates with the remote server. See RFC 4252.
func (c *connection) clientAuthenticate(config *ClientConfig) error {
	// initiate user auth session
	if err := c.transport.writePacket(Marshal(&serviceRequestMsg{serviceUserAuth})); err != nil {
		return err
	}
	packet, err := c.transport.readPacket()
	if err != nil {
		return err
	}
	// The server may choose to send a SSH_MSG_EXT_INFO at this point (if we
	// advertised willingness to receive one, which we always do) or not. See
	// RFC 8308, Section 2.4.
	extensions := make(map[string][]byte)
	if len(packet) > 0 && packet[0] == msgExtInfo {
		var extInfo extInfoMsg
		if err := Unmarshal(packet, &extInfo); err != nil {
			return err
		}
		payload := extInfo.Payload
		for i := uint32(0); i < extInfo.NumExtensions; i++ {
			name, rest, ok := parseString(payload)
			if !ok {
				return parseError(msgExtInfo)
			}
			value, rest, ok := parseString(rest)
			if !ok {
				return parseError(msgExtInfo)
			}
			extensions[string(name)] = value
			payload = rest
		}
		packet, err = c.transport.readPacket()
		if err != nil {
			return err
		}
	}
	var serviceAccept serviceAcceptMsg
	if err := Unmarshal(packet, &serviceAccept); err != nil {
		return err
	}

	// during the authentication phase the client first attempts the "none" method
	// then any untried methods suggested by the server.
	var tried []string
	var lastMethods []string

	sessionID := c.transport.getSessionID()
	for auth := AuthMethod(new(noneAuth)); auth != nil; {
		ok, methods, err := auth.auth(sessionID, config.User, c.transport, config.Rand, extensions)
		if err != nil {
			// On disconnect, return error immediately
			if _, ok := err.(*disconnectMsg); ok {
				return err
			}
			// We return the error later if there is no other method left to
			// try.
			ok = authFailure
		}
		if ok == authSuccess {
			// success
			return nil
		} else if ok == authFailure {
			if m := auth.method(); !contains(tried, m) {
				tried = append(tried, m)
			}
		}
		if methods == nil {
			methods = lastMethods
		}
		lastMethods = methods

		auth = nil

	findNext:
		for _, a := range config.Auth {
			candidateMethod := a.method()
			if contains(tried, candidateMethod) {
				continue
			}
			for _, meth := range methods {
				if meth == candidateMethod {
					auth = a
					break findNext
				}
			}
		}

		if auth == nil && err != nil {
			// We have an error and there are no other authentication methods to
			// try, so we return it.
			return err
		}
	}
	return fmt.Errorf("ssh: unable to authenticate, attempted methods %v, no supported methods remain", tried)
}

func contains(list []string, e string) bool {
	for _, s := range list {
		if s == e {
			return true
		}
	}
	return false
}

// An AuthMethod represents an instance of an RFC 4252 authentication method.
type AuthMethod interface {
	// auth authenticates user over transport t.
	// Returns true if authentication is successful.
	// If authentication is not successful, a []string of alternative
	// method names is returned. If the slice is nil, it will be ignored
	// and the previous set of possible methods will be reused.
	auth(session []byte, user string, p packetConn, rand io.Reader, extensions map[string][]byte) (authResult, []string, error)

	// method returns the RFC 4252 method name.
	method() string
}

// "none" authentication, RFC 4252 section 5.2.
type noneAuth int

func (n *noneAuth) auth(session []byte, user string, c packetConn, rand io.Reader, _ map[string][]byte) (authResult, []string, error) {
	if err := c.writePacket(Marshal(&userAuthRequestMsg{
		User:    user,
		Service: serviceSSH,
		Method:  "none",
	})); err != nil {
		return authFailure, nil, err
	}

	return handleAuthResponse(c)
}

func (n *noneAuth) method() string {
	return "none"
}

// passwordCallback is an AuthMethod that fetches the password through
// a function call, e.g. by prompting the user.
type passwordCallback func() (password string, err error)

func (cb passwordCallback) auth(session []byte, user string, c packetConn, rand io.Reader, _ map[string][]byte) (authResult, []string, error) {
	type passwordAuthMsg struct {
		User     string `sshtype:"50"`
		Service  string
		Method   string
		Reply    bool
		Password string
	}

	pw, err := cb()
	// REVIEW NOTE: is there a need to support skipping a password attempt?
	// The program may only find out that the user doesn't have a password
	// when prompting.
	if err != nil {
		return authFailure, nil, err
	}

	if err := c.writePacket(Marshal(&passwordAuthMsg{
		User:     user,
		Service:  serviceSSH,
		Method:   cb.method(),
		Reply:    false,
		Password: pw,
	})); err != nil {
		return authFailure, nil, err
	}

	return handleAuthResponse(c)
}

func (cb passwordCallback) method() string {
	return "password"
}

// Password returns an AuthMethod using the given password.
func Password(secret string) AuthMethod {
	return passwordCallback(func() (string, error) { return secret, nil })
}

// PasswordCallback returns an AuthMethod that uses a callback for
// fetching a password.
func PasswordCallback(prompt func() (secret string, err error)) AuthMethod {
	return passwordCallback(prompt)
}

type publickeyAuthMsg struct {
	User    string `sshtype:"50"`
	Service string
	Method  string
	// HasSig indicates to the receiver packet that the auth request is signed and
	// should be used for authentication of the request.
	HasSig   bool
	Algoname string
	PubKey   []byte
	// Sig is tagged with "rest" so Marshal will exclude it during
	// validateKey
	Sig []byte `ssh:"rest"`
}

// publicKeyCallback is an AuthMethod that uses a set of key
// pairs for authentication.
type publicKeyCallback func() ([]Signer, error)

func (cb publicKeyCallback) method() string {
	return "publickey"
}

func pickSignatureAlgorithm(signer Signer, extensions map[string][]byte) (MultiAlgorithmSigner, string, error) {
	var as MultiAlgorithmSigner
	keyFormat := signer.PublicKey().Type()

	// If the signer implements MultiAlgorithmSigner we use the algorithms it
	// support, if it implements AlgorithmSigner we assume it supports all
	// algorithms, otherwise only the key format one.
	switch s := signer.(type) {
	case MultiAlgorithmSigner:
		as = s
	case AlgorithmSigner:
		as = &multiAlgorithmSigner{
			AlgorithmSigner:     s,
			supportedAlgorithms: algorithmsForKeyFormat(underlyingAlgo(keyFormat)),
		}
	default:
		as = &multiAlgorithmSigner{
			AlgorithmSigner:     algorithmSignerWrapper{signer},
			supportedAlgorithms: []string{underlyingAlgo(keyFormat)},
		}
	}

	getFallbackAlgo := func() (string, error) {
		// Fallback to use if there is no "server-sig-algs" extension or a
		// common algorithm cannot be found. We use the public key format if the
		// MultiAlgorithmSigner supports it, otherwise we return an error.
		if !contains(as.Algorithms(), underlyingAlgo(keyFormat)) {
			return "", fmt.Errorf("ssh: no common public key signature algorithm, server only supports %q for key type %q, signer only supports %v",
				underlyingAlgo(keyFormat), keyFormat, as.Algorithms())
		}
		return keyFormat, nil
	}

	extPayload, ok := extensions["server-sig-algs"]
	if !ok {
		// If there is no "server-sig-algs" extension use the fallback
		// algorithm.
		algo, err := getFallbackAlgo()
		return as, algo, err
	}

	// The server-sig-algs extension only carries underlying signature
	// algorithm, but we are trying to select a protocol-level public key
	// algorithm, which might be a certificate type. Extend the list of server
	// supported algorithms to include the corresponding certificate algorithms.
	serverAlgos := strings.Split(string(extPayload), ",")
	for _, algo := range serverAlgos {
		if certAlgo, ok := certificateAlgo(algo); ok {
			serverAlgos = append(serverAlgos, certAlgo)
		}
	}

	// Filter algorithms based on those supported by MultiAlgorithmSigner.
	var keyAlgos []string
	for _, algo := range algorithmsForKeyFormat(keyFormat) {
		if contains(as.Algorithms(), underlyingAlgo(algo)) {
			keyAlgos = append(keyAlgos, algo)
		}
	}

	algo, err := findCommon("public key signature algorithm", keyAlgos, serverAlgos)
	if err != nil {
		// If there is no overlap, return the fallback algorithm to support
		// servers that fail to list all supported algorithms.
		algo, err := getFallbackAlgo()
		return as, algo, err
	}
	return as, algo, nil
}

func (cb publicKeyCallback) auth(session []byte, user string, c packetConn, rand io.Reader, extensions map[string][]byte) (authResult, []string, error) {
	// Authentication is performed by sending an enquiry to test if a key is
	// acceptable to the remote. If the key is acceptable, the client will
	// attempt to authenticate with the valid key.  If not the client will repeat
	// the process with the remaining keys.

	signers, err := cb()
	if err != nil {
		return authFailure, nil, err
	}
	var methods []string
	var errSigAlgo error

	origSignersLen := len(signers)
	for idx := 0; idx < len(signers); idx++ {
		signer := signers[idx]
		pub := signer.PublicKey()
		as, algo, err := pickSignatureAlgorithm(signer, extensions)
		if err != nil && errSigAlgo == nil {
			// If we cannot negotiate a signature algorithm store the first
			// error so we can return it to provide a more meaningful message if
			// no other signers work.
			errSigAlgo = err
			continue
		}
		ok, err := validateKey(pub, algo, user, c)
		if err != nil {
			return authFailure, nil, err
		}
		// OpenSSH 7.2-7.7 advertises support for rsa-sha2-256 and rsa-sha2-512
		// in the "server-sig-algs" extension but doesn't support these
		// algorithms for certificate authentication, so if the server rejects
		// the key try to use the obtained algorithm as if "server-sig-algs" had
		// not been implemented if supported from the algorithm signer.
		if !ok && idx < origSignersLen && isRSACert(algo) && algo != CertAlgoRSAv01 {
			if contains(as.Algorithms(), KeyAlgoRSA) {
				// We retry using the compat algorithm after all signers have
				// been tried normally.
				signers = append(signers, &multiAlgorithmSigner{
					AlgorithmSigner:     as,
					supportedAlgorithms: []string{KeyAlgoRSA},
				})
			}
		}
		if !ok {
			continue
		}

		pubKey := pub.Marshal()
		data := buildDataSignedForAuth(session, userAuthRequestMsg{
			User:    user,
			Service: serviceSSH,
			Method:  cb.method(),
		}, algo, pubKey)
		sign, err := as.SignWithAlgorithm(rand, data, underlyingAlgo(algo))
		if err != nil {
			return authFailure, nil, err
		}

		// manually wrap the serialized signature in a string
		s := Marshal(sign)
		sig := make([]byte, stringLength(len(s)))
		marshalString(sig, s)
		msg := publickeyAuthMsg{
			User:     user,
			Service:  serviceSSH,
			Method:   cb.method(),
			HasSig:   true,
			Algoname: algo,
			PubKey:   pubKey,
			Sig:      sig,
		}
		p := Marshal(&msg)
		if err := c.writePacket(p); err != nil {
			return authFailure, nil, err
		}
		var success authResult
		success, methods, err = handleAuthResponse(c)
		if err != nil {
			return authFailure, nil, err
		}

		// If authentication succeeds or the list of available methods does not
		// contain the "publickey" method, do not attempt to authenticate with any
		// other keys.  According to RFC 4252 Section 7, the latter can occur when
		// additional authentication methods are required.
		if success == authSuccess || !contains(methods, cb.method()) {
			return success, methods, err
		}
	}

	return authFailure, methods, errSigAlgo
}

// validateKey validates the key provided is acceptable to the server.
func validateKey(key PublicKey, algo string, user string, c packetConn) (bool, error) {
	pubKey := key.Marshal()
	msg := publickeyAuthMsg{
		User:     user,
		Service:  serviceSSH,
		Method:   "publickey",
		HasSig:   false,
		Algoname: algo,
		PubKey:   pubKey,
	}
	if err := c.writePacket(Marshal(&msg)); err != nil {
		return false, err
	}

	return confirmKeyAck(key, c)
}

func confirmKeyAck(key PublicKey, c packetConn) (bool, error) {
	pubKey := key.Marshal()

	for {
		packet, err := c.readPacket()
		if err != nil {
			return false, err
		}
		switch packet[0] {
		case msgUserAuthBanner:
			if err := handleBannerResponse(c, packet); err != nil {
				return false, err
			}
		case msgUserAuthPubKeyOk:
			var msg userAuthPubKeyOkMsg
			if err := Unmarshal(packet, &msg); err != nil {
				return false, err
			}
			// According to RFC 4252 Section 7 the algorithm in
			// SSH_MSG_USERAUTH_PK_OK should match that of the request but some
			// servers send the key type instead. OpenSSH allows any algorithm
			// that matches the public key, so we do the same.
			// https://github.com/openssh/openssh-portable/blob/86bdd385/sshconnect2.c#L709
			if !contains(algorithmsForKeyFormat(key.Type()), msg.Algo) {
				return false, nil
			}
			if !bytes.Equal(msg.PubKey, pubKey) {
				return false, nil
			}
			return true, nil
		case msgUserAuthFailure:
			return false, nil
		default:
			return false, unexpectedMessageError(msgUserAuthPubKeyOk, packet[0])
		}
	}
}

// PublicKeys returns an AuthMethod that uses the given key
// pairs.
func PublicKeys(signers ...Signer) AuthMethod {
	return publicKeyCallback(func() ([]Signer, error) { return signers, nil })
}

// PublicKeysCallback returns an AuthMethod that runs the given
// function to obtain a list of key pairs.
func PublicKeysCallback(getSigners func() (signers []Signer, err error)) AuthMethod {
	return publicKeyCallback(getSigners)
}

// handleAuthResponse returns whether the preceding authentication request succeeded
// along with a list of remaining authentication methods to try next and
// an error if an unexpected response was received.
func handleAuthResponse(c packetConn) (authResult, []string, error) {
	gotMsgExtInfo := false
	for {
		packet, err := c.readPacket()
		if err != nil {
			return authFailure, nil, err
		}

		switch packet[0] {
		case msgUserAuthBanner:
			if err := handleBannerResponse(c, packet); err != nil {
				return authFailure, nil, err
			}
		case msgExtInfo:
			// Ignore post-authentication RFC 8308 extensions, once.
			if gotMsgExtInfo {
				return authFailure, nil, unexpectedMessageError(msgUserAuthSuccess, packet[0])
			}
			gotMsgExtInfo = true
		case msgUserAuthFailure:
			var msg userAuthFailureMsg
			if err := Unmarshal(packet, &msg); err != nil {
				return authFailure, nil, err
			}
			if msg.PartialSuccess {
				return authPartialSuccess, msg.Methods, nil
			}
			return authFailure, msg.Methods, nil
		case msgUserAuthSuccess:
			return authSuccess, nil, nil
		default:
			return authFailure, nil, unexpectedMessageError(msgUserAuthSuccess, packet[0])
		}
	}
}

func handleBannerResponse(c packetConn, packet []byte) error {
	var msg userAuthBannerMsg
	if err := Unmarshal(packet, &msg); err != nil {
		return err
	}

	transport, ok := c.(*handshakeTransport)
	if !ok {
		return nil
	}

	if transport.bannerCallback != nil {
		return transport.bannerCallback(msg.Message)
	}

	return nil
}

// KeyboardInteractiveChallenge should print questions, optionally
// disabling echoing (e.g. for passwords), and return all the answers.
// Challenge may be called multiple times in a single session. After
// successful authentication, the server may send a challenge with no
// questions, for which the name and instruction messages should be
// printed.  RFC 4256 section 3.3 details how the UI should behave for
// both CLI and GUI environments.
type KeyboardInteractiveChallenge func(name, instruction string, questions []string, echos []bool) (answers []string, err error)

// KeyboardInteractive returns an AuthMethod using a prompt/response
// sequence controlled by the server.
func KeyboardInteractive(challenge KeyboardInteractiveChallenge) AuthMethod {
	return challenge
}

func (cb KeyboardInteractiveChallenge) method() string {
	return "keyboard-interactive"
}

func (cb KeyboardInteractiveChallenge) auth(session []byte, user string, c packetConn, rand io.Reader, _ map[string][]byte) (authResult, []string, error) {
	type initiateMsg struct {
		User       string `sshtype:"50"`
		Service    string
		Method     string
		Language   string
		Submethods string
	}

	if err := c.writePacket(Marshal(&initiateMsg{
		User:    user,
		Service: serviceSSH,
		Method:  "keyboard-interactive",
	})); err != nil {
		return authFailure, nil, err
	}

	gotMsgExtInfo := false
	gotUserAuthInfoRequest := false
	for {
		packet, err := c.readPacket()
		if err != nil {
			return authFailure, nil, err
		}

		// like handleAuthResponse, but with less options.
		switch packet[0] {
		case msgUserAuthBanner:
			if err := handleBannerResponse(c, packet); err != nil {
				return authFailure, nil, err
			}
			continue
		case msgExtInfo:
			// Ignore post-authentication RFC 8308 extensions, once.
			if gotMsgExtInfo {
				return authFailure, nil, unexpectedMessageError(msgUserAuthInfoRequest, packet[0])
			}
			gotMsgExtInfo = true
			continue
		case msgUserAuthInfoRequest:
			// OK
		case msgUserAuthFailure:
			var msg userAuthFailureMsg
			if err := Unmarshal(packet, &msg); err != nil {
				return authFailure, nil, err
			}
			if msg.PartialSuccess {
				return authPartialSuccess, msg.Methods, nil
			}
			if !gotUserAuthInfoRequest {
				return authFailure, msg.Methods, unexpectedMessageError(msgUserAuthInfoRequest, packet[0])
			}
			return authFailure, msg.Methods, nil
		case msgUserAuthSuccess:
			return authSuccess, nil, nil
		default:
			return authFailure, nil, unexpectedMessageError(msgUserAuthInfoRequest, packet[0])
		}

		var msg userAuthInfoRequestMsg
		if err := Unmarshal(packet, &msg); err != nil {
			return authFailure, nil, err
		}
		gotUserAuthInfoRequest = true

		// Manually unpack the prompt/echo pairs.
		rest := msg.Prompts
		var prompts []string
		var echos []bool
		for i := 0; i < int(msg.NumPrompts); i++ {
			prompt, r, ok := parseString(rest)
			if !ok || len(r) == 0 {
				return authFailure, nil, errors.New("ssh: prompt format error")
			}
			prompts = append(prompts, string(prompt))
			echos = append(echos, r[0] != 0)
			rest = r[1:]
		}

		if len(rest) != 0 {
			return authFailure, nil, errors.New("ssh: extra data following keyboard-interactive pairs")
		}

		answers, err := cb(msg.Name, msg.Instruction, prompts, echos)
		if err != nil {
			return authFailure, nil, err
		}

		if len(answers) != len(prompts) {
			return authFailure, nil, fmt.Errorf("ssh: incorrect number of answers from keyboard-interactive callback %d (expected %d)", len(answers), len(prompts))
		}
		responseLength := 1 + 4
		for _, a := range answers {
			responseLength += stringLength(len(a))
		}
		serialized := make([]byte, responseLength)
		p := serialized
		p[0] = msgUserAuthInfoResponse
		p = p[1:]
		p = marshalUint32(p, uint32(len(answers)))
		for _, a := range answers {
			p = marshalString(p, []byte(a))
		}

		if err := c.writePacket(serialized); err != nil {
			return authFailure, nil, err
		}
	}
}

type retryableAuthMethod struct {
	authMethod AuthMethod
	maxTries   int
}

func (r *retryableAuthMethod) auth(session []byte, user string, c packetConn, rand io.Reader, extensions map[string][]byte) (ok authResult, methods []string, err error) {
	for i := 0; r.maxTries <= 0 || i < r.maxTries; i++ {
		ok, methods, err = r.authMethod.auth(session, user, c, rand, extensions)
		if ok != authFailure || err != nil { // either success, partial success or error terminate
			return ok, methods, err
		}
	}
	return ok, methods, err
}

func (r *retryableAuthMethod) method() string {
	return r.authMethod.method()
}

// RetryableAuthMethod is a decorator for other auth methods enabling them to
// be retried up to maxTries before considering that AuthMethod itself failed.
// If maxTries is <= 0, will retry indefinitely
//
// This is useful for interactive clients using challenge/response type
// authentication (e.g. Keyboard-Interactive, Password, etc) where the user
// could mistype their response resulting in the server issuing a
// SSH_MSG_USERAUTH_FAILURE (rfc4252 #8 [password] and rfc4256 #3.4
// [keyboard-interactive]); Without this decorator, the non-retryable
// AuthMethod would be removed from future consideration, and never tried again
// (and so the user would never be able to retry their entry).
func RetryableAuthMethod(auth AuthMethod, maxTries int) AuthMethod {
	return &retryableAuthMethod{authMethod: auth, maxTries: maxTries}
}

// GSSAPIWithMICAuthMethod is an AuthMethod with "gssapi-with-mic" authentication.
// See RFC 4462 section 3
// gssAPIClient is implementation of the GSSAPIClient interface, see the definition of the interface for details.
// target is the server host you want to log in to.
func GSSAPIWithMICAuthMethod(gssAPIClient GSSAPIClient, target string) AuthMethod {
	if gssAPIClient == nil {
		panic("gss-api client must be not nil with enable gssapi-with-mic")
	}
	return &gssAPIWithMICCallback{gssAPIClient: gssAPIClient, target: target}
}

type gssAPIWithMICCallback struct {
	gssAPIClient GSSAPIClient
	target       string
}

func (g *gssAPIWithMICCallback) auth(session []byte, user string, c packetConn, rand io.Reader, _ map[string][]byte) (authResult, []string, error) {
	m := &userAuthRequestMsg{
		User:    user,
		Service: serviceSSH,
		Method:  g.method(),
	}
	// The GSS-API authentication method is initiated when the client sends an SSH_MSG_USERAUTH_REQUEST.
	// See RFC 4462 section 3.2.
	m.Payload = appendU32(m.Payload, 1)
	m.Payload = appendString(m.Payload, string(krb5OID))
	if err := c.writePacket(Marshal(m)); err != nil {
		return authFailure, nil, err
	}
	// The server responds to the SSH_MSG_USERAUTH_REQUEST with either an
	// SSH_MSG_USERAUTH_FAILURE if none of the mechanisms are supported or
	// with an SSH_MSG_USERAUTH_GSSAPI_RESPONSE.
	// See RFC 4462 section 3.3.
	// OpenSSH supports Kerberos V5 mechanism only for GSS-API authentication,so I don't want to check
	// selected mech if it is valid.
	packet, err := c.readPacket()
	if err != nil {
		return authFailure, nil, err
	}
	userAuthGSSAPIResp := &userAuthGSSAPIResponse{}
	if err := Unmarshal(packet, userAuthGSSAPIResp); err != nil {
		return authFailure, nil, err
	}
	// Start the loop into the exchange token.
	// See RFC 4462 section 3.4.
	var token []byte
	defer g.gssAPIClient.DeleteSecContext()
	for {
		// Initiates the establishment of a security context between the application and a remote peer.
		nextToken, needContinue, err := g.gssAPIClient.InitSecContext("host@"+g.target, token, false)
		if err != nil {
			return authFailure, nil, err
		}
		if len(nextToken) > 0 {
			if err := c.writePacket(Marshal(&userAuthGSSAPIToken{
				Token: nextToken,
			})); err != nil {
				return authFailure, nil, err
			}
		}
		if !needContinue {
			break
		}
		packet, err = c.readPacket()
		if err != nil {
			return authFailure, nil, err
		}
		switch packet[0] {
		case msgUserAuthFailure:
			var msg userAuthFailureMsg
			if err := Unmarshal(packet, &msg); err != nil {
				return authFailure, nil, err
			}
			if msg.PartialSuccess {
				return authPartialSuccess, msg.Methods, nil
			}
			return authFailure, msg.Methods, nil
		case msgUserAuthGSSAPIError:
			userAuthGSSAPIErrorResp := &userAuthGSSAPIError{}
			if err := Unmarshal(packet, userAuthGSSAPIErrorResp); err != nil {
				return authFailure, nil, err
			}
			return authFailure, nil, fmt.Errorf("GSS-API Error:\n"+
				"Major Status: %d\n"+
				"Minor Status: %d\n"+
				"Error Message: %s\n", userAuthGSSAPIErrorResp.MajorStatus, userAuthGSSAPIErrorResp.MinorStatus,
				userAuthGSSAPIErrorResp.Message)
		case msgUserAuthGSSAPIToken:
			userAuthGSSAPITokenReq := &userAuthGSSAPIToken{}
			if err := Unmarshal(packet, userAuthGSSAPITokenReq); err != nil {
				return authFailure, nil, err
			}
			token = userAuthGSSAPITokenReq.Token
		}
	}
	// Binding Encryption Keys.
	// See RFC 4462 section 3.5.
	micField := buildMIC(string(session), user, "ssh-connection", "gssapi-with-mic")
	micToken, err := g.gssAPIClient.GetMIC(micField)
	if err != nil {
		return authFailure, nil, err
	}
	if err := c.writePacket(Marshal(&userAuthGSSAPIMIC{
		MIC: micToken,
	})); err != nil {
		return authFailure, nil, err
	}
	return handleAuthResponse(c)
}

func (g *gssAPIWithMICCallback) method() string {
	return "gssapi-with-mic"
}
