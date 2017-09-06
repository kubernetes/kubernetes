// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
)

// The Permissions type holds fine-grained permissions that are
// specific to a user or a specific authentication method for a user.
// The Permissions value for a successful authentication attempt is
// available in ServerConn, so it can be used to pass information from
// the user-authentication phase to the application layer.
type Permissions struct {
	// CriticalOptions indicate restrictions to the default
	// permissions, and are typically used in conjunction with
	// user certificates. The standard for SSH certificates
	// defines "force-command" (only allow the given command to
	// execute) and "source-address" (only allow connections from
	// the given address). The SSH package currently only enforces
	// the "source-address" critical option. It is up to server
	// implementations to enforce other critical options, such as
	// "force-command", by checking them after the SSH handshake
	// is successful. In general, SSH servers should reject
	// connections that specify critical options that are unknown
	// or not supported.
	CriticalOptions map[string]string

	// Extensions are extra functionality that the server may
	// offer on authenticated connections. Lack of support for an
	// extension does not preclude authenticating a user. Common
	// extensions are "permit-agent-forwarding",
	// "permit-X11-forwarding". The Go SSH library currently does
	// not act on any extension, and it is up to server
	// implementations to honor them. Extensions can be used to
	// pass data from the authentication callbacks to the server
	// application layer.
	Extensions map[string]string
}

// ServerConfig holds server specific configuration data.
type ServerConfig struct {
	// Config contains configuration shared between client and server.
	Config

	hostKeys []Signer

	// NoClientAuth is true if clients are allowed to connect without
	// authenticating.
	NoClientAuth bool

	// MaxAuthTries specifies the maximum number of authentication attempts
	// permitted per connection. If set to a negative number, the number of
	// attempts are unlimited. If set to zero, the number of attempts are limited
	// to 6.
	MaxAuthTries int

	// PasswordCallback, if non-nil, is called when a user
	// attempts to authenticate using a password.
	PasswordCallback func(conn ConnMetadata, password []byte) (*Permissions, error)

	// PublicKeyCallback, if non-nil, is called when a client
	// offers a public key for authentication. It must return a nil error
	// if the given public key can be used to authenticate the
	// given user. For example, see CertChecker.Authenticate. A
	// call to this function does not guarantee that the key
	// offered is in fact used to authenticate. To record any data
	// depending on the public key, store it inside a
	// Permissions.Extensions entry.
	PublicKeyCallback func(conn ConnMetadata, key PublicKey) (*Permissions, error)

	// KeyboardInteractiveCallback, if non-nil, is called when
	// keyboard-interactive authentication is selected (RFC
	// 4256). The client object's Challenge function should be
	// used to query the user. The callback may offer multiple
	// Challenge rounds. To avoid information leaks, the client
	// should be presented a challenge even if the user is
	// unknown.
	KeyboardInteractiveCallback func(conn ConnMetadata, client KeyboardInteractiveChallenge) (*Permissions, error)

	// AuthLogCallback, if non-nil, is called to log all authentication
	// attempts.
	AuthLogCallback func(conn ConnMetadata, method string, err error)

	// ServerVersion is the version identification string to announce in
	// the public handshake.
	// If empty, a reasonable default is used.
	// Note that RFC 4253 section 4.2 requires that this string start with
	// "SSH-2.0-".
	ServerVersion string
}

// AddHostKey adds a private key as a host key. If an existing host
// key exists with the same algorithm, it is overwritten. Each server
// config must have at least one host key.
func (s *ServerConfig) AddHostKey(key Signer) {
	for i, k := range s.hostKeys {
		if k.PublicKey().Type() == key.PublicKey().Type() {
			s.hostKeys[i] = key
			return
		}
	}

	s.hostKeys = append(s.hostKeys, key)
}

// cachedPubKey contains the results of querying whether a public key is
// acceptable for a user.
type cachedPubKey struct {
	user       string
	pubKeyData []byte
	result     error
	perms      *Permissions
}

const maxCachedPubKeys = 16

// pubKeyCache caches tests for public keys.  Since SSH clients
// will query whether a public key is acceptable before attempting to
// authenticate with it, we end up with duplicate queries for public
// key validity.  The cache only applies to a single ServerConn.
type pubKeyCache struct {
	keys []cachedPubKey
}

// get returns the result for a given user/algo/key tuple.
func (c *pubKeyCache) get(user string, pubKeyData []byte) (cachedPubKey, bool) {
	for _, k := range c.keys {
		if k.user == user && bytes.Equal(k.pubKeyData, pubKeyData) {
			return k, true
		}
	}
	return cachedPubKey{}, false
}

// add adds the given tuple to the cache.
func (c *pubKeyCache) add(candidate cachedPubKey) {
	if len(c.keys) < maxCachedPubKeys {
		c.keys = append(c.keys, candidate)
	}
}

// ServerConn is an authenticated SSH connection, as seen from the
// server
type ServerConn struct {
	Conn

	// If the succeeding authentication callback returned a
	// non-nil Permissions pointer, it is stored here.
	Permissions *Permissions
}

// NewServerConn starts a new SSH server with c as the underlying
// transport.  It starts with a handshake and, if the handshake is
// unsuccessful, it closes the connection and returns an error.  The
// Request and NewChannel channels must be serviced, or the connection
// will hang.
func NewServerConn(c net.Conn, config *ServerConfig) (*ServerConn, <-chan NewChannel, <-chan *Request, error) {
	fullConf := *config
	fullConf.SetDefaults()
	if fullConf.MaxAuthTries == 0 {
		fullConf.MaxAuthTries = 6
	}

	s := &connection{
		sshConn: sshConn{conn: c},
	}
	perms, err := s.serverHandshake(&fullConf)
	if err != nil {
		c.Close()
		return nil, nil, nil, err
	}
	return &ServerConn{s, perms}, s.mux.incomingChannels, s.mux.incomingRequests, nil
}

// signAndMarshal signs the data with the appropriate algorithm,
// and serializes the result in SSH wire format.
func signAndMarshal(k Signer, rand io.Reader, data []byte) ([]byte, error) {
	sig, err := k.Sign(rand, data)
	if err != nil {
		return nil, err
	}

	return Marshal(sig), nil
}

// handshake performs key exchange and user authentication.
func (s *connection) serverHandshake(config *ServerConfig) (*Permissions, error) {
	if len(config.hostKeys) == 0 {
		return nil, errors.New("ssh: server has no host keys")
	}

	if !config.NoClientAuth && config.PasswordCallback == nil && config.PublicKeyCallback == nil && config.KeyboardInteractiveCallback == nil {
		return nil, errors.New("ssh: no authentication methods configured but NoClientAuth is also false")
	}

	if config.ServerVersion != "" {
		s.serverVersion = []byte(config.ServerVersion)
	} else {
		s.serverVersion = []byte(packageVersion)
	}
	var err error
	s.clientVersion, err = exchangeVersions(s.sshConn.conn, s.serverVersion)
	if err != nil {
		return nil, err
	}

	tr := newTransport(s.sshConn.conn, config.Rand, false /* not client */)
	s.transport = newServerTransport(tr, s.clientVersion, s.serverVersion, config)

	if err := s.transport.waitSession(); err != nil {
		return nil, err
	}

	// We just did the key change, so the session ID is established.
	s.sessionID = s.transport.getSessionID()

	var packet []byte
	if packet, err = s.transport.readPacket(); err != nil {
		return nil, err
	}

	var serviceRequest serviceRequestMsg
	if err = Unmarshal(packet, &serviceRequest); err != nil {
		return nil, err
	}
	if serviceRequest.Service != serviceUserAuth {
		return nil, errors.New("ssh: requested service '" + serviceRequest.Service + "' before authenticating")
	}
	serviceAccept := serviceAcceptMsg{
		Service: serviceUserAuth,
	}
	if err := s.transport.writePacket(Marshal(&serviceAccept)); err != nil {
		return nil, err
	}

	perms, err := s.serverAuthenticate(config)
	if err != nil {
		return nil, err
	}
	s.mux = newMux(s.transport)
	return perms, err
}

func isAcceptableAlgo(algo string) bool {
	switch algo {
	case KeyAlgoRSA, KeyAlgoDSA, KeyAlgoECDSA256, KeyAlgoECDSA384, KeyAlgoECDSA521, KeyAlgoED25519,
		CertAlgoRSAv01, CertAlgoDSAv01, CertAlgoECDSA256v01, CertAlgoECDSA384v01, CertAlgoECDSA521v01:
		return true
	}
	return false
}

func checkSourceAddress(addr net.Addr, sourceAddrs string) error {
	if addr == nil {
		return errors.New("ssh: no address known for client, but source-address match required")
	}

	tcpAddr, ok := addr.(*net.TCPAddr)
	if !ok {
		return fmt.Errorf("ssh: remote address %v is not an TCP address when checking source-address match", addr)
	}

	for _, sourceAddr := range strings.Split(sourceAddrs, ",") {
		if allowedIP := net.ParseIP(sourceAddr); allowedIP != nil {
			if allowedIP.Equal(tcpAddr.IP) {
				return nil
			}
		} else {
			_, ipNet, err := net.ParseCIDR(sourceAddr)
			if err != nil {
				return fmt.Errorf("ssh: error parsing source-address restriction %q: %v", sourceAddr, err)
			}

			if ipNet.Contains(tcpAddr.IP) {
				return nil
			}
		}
	}

	return fmt.Errorf("ssh: remote address %v is not allowed because of source-address restriction", addr)
}

// ServerAuthError implements the error interface. It appends any authentication
// errors that may occur, and is returned if all of the authentication methods
// provided by the user failed to authenticate.
type ServerAuthError struct {
	// Errors contains authentication errors returned by the authentication
	// callback methods.
	Errors []error
}

func (l ServerAuthError) Error() string {
	var errs []string
	for _, err := range l.Errors {
		errs = append(errs, err.Error())
	}
	return "[" + strings.Join(errs, ", ") + "]"
}

func (s *connection) serverAuthenticate(config *ServerConfig) (*Permissions, error) {
	sessionID := s.transport.getSessionID()
	var cache pubKeyCache
	var perms *Permissions

	authFailures := 0
	var authErrs []error

userAuthLoop:
	for {
		if authFailures >= config.MaxAuthTries && config.MaxAuthTries > 0 {
			discMsg := &disconnectMsg{
				Reason:  2,
				Message: "too many authentication failures",
			}

			if err := s.transport.writePacket(Marshal(discMsg)); err != nil {
				return nil, err
			}

			return nil, discMsg
		}

		var userAuthReq userAuthRequestMsg
		if packet, err := s.transport.readPacket(); err != nil {
			if err == io.EOF {
				return nil, &ServerAuthError{Errors: authErrs}
			}
			return nil, err
		} else if err = Unmarshal(packet, &userAuthReq); err != nil {
			return nil, err
		}

		if userAuthReq.Service != serviceSSH {
			return nil, errors.New("ssh: client attempted to negotiate for unknown service: " + userAuthReq.Service)
		}

		s.user = userAuthReq.User
		perms = nil
		authErr := errors.New("no auth passed yet")

		switch userAuthReq.Method {
		case "none":
			if config.NoClientAuth {
				authErr = nil
			}

			// allow initial attempt of 'none' without penalty
			if authFailures == 0 {
				authFailures--
			}
		case "password":
			if config.PasswordCallback == nil {
				authErr = errors.New("ssh: password auth not configured")
				break
			}
			payload := userAuthReq.Payload
			if len(payload) < 1 || payload[0] != 0 {
				return nil, parseError(msgUserAuthRequest)
			}
			payload = payload[1:]
			password, payload, ok := parseString(payload)
			if !ok || len(payload) > 0 {
				return nil, parseError(msgUserAuthRequest)
			}

			perms, authErr = config.PasswordCallback(s, password)
		case "keyboard-interactive":
			if config.KeyboardInteractiveCallback == nil {
				authErr = errors.New("ssh: keyboard-interactive auth not configubred")
				break
			}

			prompter := &sshClientKeyboardInteractive{s}
			perms, authErr = config.KeyboardInteractiveCallback(s, prompter.Challenge)
		case "publickey":
			if config.PublicKeyCallback == nil {
				authErr = errors.New("ssh: publickey auth not configured")
				break
			}
			payload := userAuthReq.Payload
			if len(payload) < 1 {
				return nil, parseError(msgUserAuthRequest)
			}
			isQuery := payload[0] == 0
			payload = payload[1:]
			algoBytes, payload, ok := parseString(payload)
			if !ok {
				return nil, parseError(msgUserAuthRequest)
			}
			algo := string(algoBytes)
			if !isAcceptableAlgo(algo) {
				authErr = fmt.Errorf("ssh: algorithm %q not accepted", algo)
				break
			}

			pubKeyData, payload, ok := parseString(payload)
			if !ok {
				return nil, parseError(msgUserAuthRequest)
			}

			pubKey, err := ParsePublicKey(pubKeyData)
			if err != nil {
				return nil, err
			}

			candidate, ok := cache.get(s.user, pubKeyData)
			if !ok {
				candidate.user = s.user
				candidate.pubKeyData = pubKeyData
				candidate.perms, candidate.result = config.PublicKeyCallback(s, pubKey)
				if candidate.result == nil && candidate.perms != nil && candidate.perms.CriticalOptions != nil && candidate.perms.CriticalOptions[sourceAddressCriticalOption] != "" {
					candidate.result = checkSourceAddress(
						s.RemoteAddr(),
						candidate.perms.CriticalOptions[sourceAddressCriticalOption])
				}
				cache.add(candidate)
			}

			if isQuery {
				// The client can query if the given public key
				// would be okay.

				if len(payload) > 0 {
					return nil, parseError(msgUserAuthRequest)
				}

				if candidate.result == nil {
					okMsg := userAuthPubKeyOkMsg{
						Algo:   algo,
						PubKey: pubKeyData,
					}
					if err = s.transport.writePacket(Marshal(&okMsg)); err != nil {
						return nil, err
					}
					continue userAuthLoop
				}
				authErr = candidate.result
			} else {
				sig, payload, ok := parseSignature(payload)
				if !ok || len(payload) > 0 {
					return nil, parseError(msgUserAuthRequest)
				}
				// Ensure the public key algo and signature algo
				// are supported.  Compare the private key
				// algorithm name that corresponds to algo with
				// sig.Format.  This is usually the same, but
				// for certs, the names differ.
				if !isAcceptableAlgo(sig.Format) {
					break
				}
				signedData := buildDataSignedForAuth(sessionID, userAuthReq, algoBytes, pubKeyData)

				if err := pubKey.Verify(signedData, sig); err != nil {
					return nil, err
				}

				authErr = candidate.result
				perms = candidate.perms
			}
		default:
			authErr = fmt.Errorf("ssh: unknown method %q", userAuthReq.Method)
		}

		authErrs = append(authErrs, authErr)

		if config.AuthLogCallback != nil {
			config.AuthLogCallback(s, userAuthReq.Method, authErr)
		}

		if authErr == nil {
			break userAuthLoop
		}

		authFailures++

		var failureMsg userAuthFailureMsg
		if config.PasswordCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "password")
		}
		if config.PublicKeyCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "publickey")
		}
		if config.KeyboardInteractiveCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "keyboard-interactive")
		}

		if len(failureMsg.Methods) == 0 {
			return nil, errors.New("ssh: no authentication methods configured but NoClientAuth is also false")
		}

		if err := s.transport.writePacket(Marshal(&failureMsg)); err != nil {
			return nil, err
		}
	}

	if err := s.transport.writePacket([]byte{msgUserAuthSuccess}); err != nil {
		return nil, err
	}
	return perms, nil
}

// sshClientKeyboardInteractive implements a ClientKeyboardInteractive by
// asking the client on the other side of a ServerConn.
type sshClientKeyboardInteractive struct {
	*connection
}

func (c *sshClientKeyboardInteractive) Challenge(user, instruction string, questions []string, echos []bool) (answers []string, err error) {
	if len(questions) != len(echos) {
		return nil, errors.New("ssh: echos and questions must have equal length")
	}

	var prompts []byte
	for i := range questions {
		prompts = appendString(prompts, questions[i])
		prompts = appendBool(prompts, echos[i])
	}

	if err := c.transport.writePacket(Marshal(&userAuthInfoRequestMsg{
		Instruction: instruction,
		NumPrompts:  uint32(len(questions)),
		Prompts:     prompts,
	})); err != nil {
		return nil, err
	}

	packet, err := c.transport.readPacket()
	if err != nil {
		return nil, err
	}
	if packet[0] != msgUserAuthInfoResponse {
		return nil, unexpectedMessageError(msgUserAuthInfoResponse, packet[0])
	}
	packet = packet[1:]

	n, packet, ok := parseUint32(packet)
	if !ok || int(n) != len(questions) {
		return nil, parseError(msgUserAuthInfoResponse)
	}

	for i := uint32(0); i < n; i++ {
		ans, rest, ok := parseString(packet)
		if !ok {
			return nil, parseError(msgUserAuthInfoResponse)
		}

		answers = append(answers, string(ans))
		packet = rest
	}
	if len(packet) != 0 {
		return nil, errors.New("ssh: junk at end of message")
	}

	return answers, nil
}
