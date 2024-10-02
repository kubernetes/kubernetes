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

type GSSAPIWithMICConfig struct {
	// AllowLogin, must be set, is called when gssapi-with-mic
	// authentication is selected (RFC 4462 section 3). The srcName is from the
	// results of the GSS-API authentication. The format is username@DOMAIN.
	// GSSAPI just guarantees to the server who the user is, but not if they can log in, and with what permissions.
	// This callback is called after the user identity is established with GSSAPI to decide if the user can login with
	// which permissions. If the user is allowed to login, it should return a nil error.
	AllowLogin func(conn ConnMetadata, srcName string) (*Permissions, error)

	// Server must be set. It's the implementation
	// of the GSSAPIServer interface. See GSSAPIServer interface for details.
	Server GSSAPIServer
}

// ServerConfig holds server specific configuration data.
type ServerConfig struct {
	// Config contains configuration shared between client and server.
	Config

	// PublicKeyAuthAlgorithms specifies the supported client public key
	// authentication algorithms. Note that this should not include certificate
	// types since those use the underlying algorithm. This list is sent to the
	// client if it supports the server-sig-algs extension. Order is irrelevant.
	// If unspecified then a default set of algorithms is used.
	PublicKeyAuthAlgorithms []string

	hostKeys []Signer

	// NoClientAuth is true if clients are allowed to connect without
	// authenticating.
	// To determine NoClientAuth at runtime, set NoClientAuth to true
	// and the optional NoClientAuthCallback to a non-nil value.
	NoClientAuth bool

	// NoClientAuthCallback, if non-nil, is called when a user
	// attempts to authenticate with auth method "none".
	// NoClientAuth must also be set to true for this be used, or
	// this func is unused.
	NoClientAuthCallback func(ConnMetadata) (*Permissions, error)

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

	// BannerCallback, if present, is called and the return string is sent to
	// the client after key exchange completed but before authentication.
	BannerCallback func(conn ConnMetadata) string

	// GSSAPIWithMICConfig includes gssapi server and callback, which if both non-nil, is used
	// when gssapi-with-mic authentication is selected (RFC 4462 section 3).
	GSSAPIWithMICConfig *GSSAPIWithMICConfig
}

// AddHostKey adds a private key as a host key. If an existing host
// key exists with the same public key format, it is replaced. Each server
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
//
// The returned error may be of type *ServerAuthError for
// authentication errors.
func NewServerConn(c net.Conn, config *ServerConfig) (*ServerConn, <-chan NewChannel, <-chan *Request, error) {
	fullConf := *config
	fullConf.SetDefaults()
	if fullConf.MaxAuthTries == 0 {
		fullConf.MaxAuthTries = 6
	}
	if len(fullConf.PublicKeyAuthAlgorithms) == 0 {
		fullConf.PublicKeyAuthAlgorithms = supportedPubKeyAuthAlgos
	} else {
		for _, algo := range fullConf.PublicKeyAuthAlgorithms {
			if !contains(supportedPubKeyAuthAlgos, algo) {
				c.Close()
				return nil, nil, nil, fmt.Errorf("ssh: unsupported public key authentication algorithm %s", algo)
			}
		}
	}
	// Check if the config contains any unsupported key exchanges
	for _, kex := range fullConf.KeyExchanges {
		if _, ok := serverForbiddenKexAlgos[kex]; ok {
			c.Close()
			return nil, nil, nil, fmt.Errorf("ssh: unsupported key exchange %s for server", kex)
		}
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
// and serializes the result in SSH wire format. algo is the negotiate
// algorithm and may be a certificate type.
func signAndMarshal(k AlgorithmSigner, rand io.Reader, data []byte, algo string) ([]byte, error) {
	sig, err := k.SignWithAlgorithm(rand, data, underlyingAlgo(algo))
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

	if !config.NoClientAuth && config.PasswordCallback == nil && config.PublicKeyCallback == nil &&
		config.KeyboardInteractiveCallback == nil && (config.GSSAPIWithMICConfig == nil ||
		config.GSSAPIWithMICConfig.AllowLogin == nil || config.GSSAPIWithMICConfig.Server == nil) {
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

func gssExchangeToken(gssapiConfig *GSSAPIWithMICConfig, token []byte, s *connection,
	sessionID []byte, userAuthReq userAuthRequestMsg) (authErr error, perms *Permissions, err error) {
	gssAPIServer := gssapiConfig.Server
	defer gssAPIServer.DeleteSecContext()
	var srcName string
	for {
		var (
			outToken     []byte
			needContinue bool
		)
		outToken, srcName, needContinue, err = gssAPIServer.AcceptSecContext(token)
		if err != nil {
			return err, nil, nil
		}
		if len(outToken) != 0 {
			if err := s.transport.writePacket(Marshal(&userAuthGSSAPIToken{
				Token: outToken,
			})); err != nil {
				return nil, nil, err
			}
		}
		if !needContinue {
			break
		}
		packet, err := s.transport.readPacket()
		if err != nil {
			return nil, nil, err
		}
		userAuthGSSAPITokenReq := &userAuthGSSAPIToken{}
		if err := Unmarshal(packet, userAuthGSSAPITokenReq); err != nil {
			return nil, nil, err
		}
		token = userAuthGSSAPITokenReq.Token
	}
	packet, err := s.transport.readPacket()
	if err != nil {
		return nil, nil, err
	}
	userAuthGSSAPIMICReq := &userAuthGSSAPIMIC{}
	if err := Unmarshal(packet, userAuthGSSAPIMICReq); err != nil {
		return nil, nil, err
	}
	mic := buildMIC(string(sessionID), userAuthReq.User, userAuthReq.Service, userAuthReq.Method)
	if err := gssAPIServer.VerifyMIC(mic, userAuthGSSAPIMICReq.MIC); err != nil {
		return err, nil, nil
	}
	perms, authErr = gssapiConfig.AllowLogin(s, srcName)
	return authErr, perms, nil
}

// isAlgoCompatible checks if the signature format is compatible with the
// selected algorithm taking into account edge cases that occur with old
// clients.
func isAlgoCompatible(algo, sigFormat string) bool {
	// Compatibility for old clients.
	//
	// For certificate authentication with OpenSSH 7.2-7.7 signature format can
	// be rsa-sha2-256 or rsa-sha2-512 for the algorithm
	// ssh-rsa-cert-v01@openssh.com.
	//
	// With gpg-agent < 2.2.6 the algorithm can be rsa-sha2-256 or rsa-sha2-512
	// for signature format ssh-rsa.
	if isRSA(algo) && isRSA(sigFormat) {
		return true
	}
	// Standard case: the underlying algorithm must match the signature format.
	return underlyingAlgo(algo) == sigFormat
}

// ServerAuthError represents server authentication errors and is
// sometimes returned by NewServerConn. It appends any authentication
// errors that may occur, and is returned if all of the authentication
// methods provided by the user failed to authenticate.
type ServerAuthError struct {
	// Errors contains authentication errors returned by the authentication
	// callback methods. The first entry is typically ErrNoAuth.
	Errors []error
}

func (l ServerAuthError) Error() string {
	var errs []string
	for _, err := range l.Errors {
		errs = append(errs, err.Error())
	}
	return "[" + strings.Join(errs, ", ") + "]"
}

// ServerAuthCallbacks defines server-side authentication callbacks.
type ServerAuthCallbacks struct {
	// PasswordCallback behaves like [ServerConfig.PasswordCallback].
	PasswordCallback func(conn ConnMetadata, password []byte) (*Permissions, error)

	// PublicKeyCallback behaves like [ServerConfig.PublicKeyCallback].
	PublicKeyCallback func(conn ConnMetadata, key PublicKey) (*Permissions, error)

	// KeyboardInteractiveCallback behaves like [ServerConfig.KeyboardInteractiveCallback].
	KeyboardInteractiveCallback func(conn ConnMetadata, client KeyboardInteractiveChallenge) (*Permissions, error)

	// GSSAPIWithMICConfig behaves like [ServerConfig.GSSAPIWithMICConfig].
	GSSAPIWithMICConfig *GSSAPIWithMICConfig
}

// PartialSuccessError can be returned by any of the [ServerConfig]
// authentication callbacks to indicate to the client that authentication has
// partially succeeded, but further steps are required.
type PartialSuccessError struct {
	// Next defines the authentication callbacks to apply to further steps. The
	// available methods communicated to the client are based on the non-nil
	// ServerAuthCallbacks fields.
	Next ServerAuthCallbacks
}

func (p *PartialSuccessError) Error() string {
	return "ssh: authenticated with partial success"
}

// ErrNoAuth is the error value returned if no
// authentication method has been passed yet. This happens as a normal
// part of the authentication loop, since the client first tries
// 'none' authentication to discover available methods.
// It is returned in ServerAuthError.Errors from NewServerConn.
var ErrNoAuth = errors.New("ssh: no auth passed yet")

// BannerError is an error that can be returned by authentication handlers in
// ServerConfig to send a banner message to the client.
type BannerError struct {
	Err     error
	Message string
}

func (b *BannerError) Unwrap() error {
	return b.Err
}

func (b *BannerError) Error() string {
	if b.Err == nil {
		return b.Message
	}
	return b.Err.Error()
}

func (s *connection) serverAuthenticate(config *ServerConfig) (*Permissions, error) {
	sessionID := s.transport.getSessionID()
	var cache pubKeyCache
	var perms *Permissions

	authFailures := 0
	noneAuthCount := 0
	var authErrs []error
	var displayedBanner bool
	partialSuccessReturned := false
	// Set the initial authentication callbacks from the config. They can be
	// changed if a PartialSuccessError is returned.
	authConfig := ServerAuthCallbacks{
		PasswordCallback:            config.PasswordCallback,
		PublicKeyCallback:           config.PublicKeyCallback,
		KeyboardInteractiveCallback: config.KeyboardInteractiveCallback,
		GSSAPIWithMICConfig:         config.GSSAPIWithMICConfig,
	}

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

		if s.user != userAuthReq.User && partialSuccessReturned {
			return nil, fmt.Errorf("ssh: client changed the user after a partial success authentication, previous user %q, current user %q",
				s.user, userAuthReq.User)
		}

		s.user = userAuthReq.User

		if !displayedBanner && config.BannerCallback != nil {
			displayedBanner = true
			msg := config.BannerCallback(s)
			if msg != "" {
				bannerMsg := &userAuthBannerMsg{
					Message: msg,
				}
				if err := s.transport.writePacket(Marshal(bannerMsg)); err != nil {
					return nil, err
				}
			}
		}

		perms = nil
		authErr := ErrNoAuth

		switch userAuthReq.Method {
		case "none":
			noneAuthCount++
			// We don't allow none authentication after a partial success
			// response.
			if config.NoClientAuth && !partialSuccessReturned {
				if config.NoClientAuthCallback != nil {
					perms, authErr = config.NoClientAuthCallback(s)
				} else {
					authErr = nil
				}
			}
		case "password":
			if authConfig.PasswordCallback == nil {
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

			perms, authErr = authConfig.PasswordCallback(s, password)
		case "keyboard-interactive":
			if authConfig.KeyboardInteractiveCallback == nil {
				authErr = errors.New("ssh: keyboard-interactive auth not configured")
				break
			}

			prompter := &sshClientKeyboardInteractive{s}
			perms, authErr = authConfig.KeyboardInteractiveCallback(s, prompter.Challenge)
		case "publickey":
			if authConfig.PublicKeyCallback == nil {
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
			if !contains(config.PublicKeyAuthAlgorithms, underlyingAlgo(algo)) {
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
				candidate.perms, candidate.result = authConfig.PublicKeyCallback(s, pubKey)
				_, isPartialSuccessError := candidate.result.(*PartialSuccessError)

				if (candidate.result == nil || isPartialSuccessError) &&
					candidate.perms != nil &&
					candidate.perms.CriticalOptions != nil &&
					candidate.perms.CriticalOptions[sourceAddressCriticalOption] != "" {
					if err := checkSourceAddress(
						s.RemoteAddr(),
						candidate.perms.CriticalOptions[sourceAddressCriticalOption]); err != nil {
						candidate.result = err
					}
				}
				cache.add(candidate)
			}

			if isQuery {
				// The client can query if the given public key
				// would be okay.

				if len(payload) > 0 {
					return nil, parseError(msgUserAuthRequest)
				}
				_, isPartialSuccessError := candidate.result.(*PartialSuccessError)
				if candidate.result == nil || isPartialSuccessError {
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
				// Ensure the declared public key algo is compatible with the
				// decoded one. This check will ensure we don't accept e.g.
				// ssh-rsa-cert-v01@openssh.com algorithm with ssh-rsa public
				// key type. The algorithm and public key type must be
				// consistent: both must be certificate algorithms, or neither.
				if !contains(algorithmsForKeyFormat(pubKey.Type()), algo) {
					authErr = fmt.Errorf("ssh: public key type %q not compatible with selected algorithm %q",
						pubKey.Type(), algo)
					break
				}
				// Ensure the public key algo and signature algo
				// are supported.  Compare the private key
				// algorithm name that corresponds to algo with
				// sig.Format.  This is usually the same, but
				// for certs, the names differ.
				if !contains(config.PublicKeyAuthAlgorithms, sig.Format) {
					authErr = fmt.Errorf("ssh: algorithm %q not accepted", sig.Format)
					break
				}
				if !isAlgoCompatible(algo, sig.Format) {
					authErr = fmt.Errorf("ssh: signature %q not compatible with selected algorithm %q", sig.Format, algo)
					break
				}

				signedData := buildDataSignedForAuth(sessionID, userAuthReq, algo, pubKeyData)

				if err := pubKey.Verify(signedData, sig); err != nil {
					return nil, err
				}

				authErr = candidate.result
				perms = candidate.perms
			}
		case "gssapi-with-mic":
			if authConfig.GSSAPIWithMICConfig == nil {
				authErr = errors.New("ssh: gssapi-with-mic auth not configured")
				break
			}
			gssapiConfig := authConfig.GSSAPIWithMICConfig
			userAuthRequestGSSAPI, err := parseGSSAPIPayload(userAuthReq.Payload)
			if err != nil {
				return nil, parseError(msgUserAuthRequest)
			}
			// OpenSSH supports Kerberos V5 mechanism only for GSS-API authentication.
			if userAuthRequestGSSAPI.N == 0 {
				authErr = fmt.Errorf("ssh: Mechanism negotiation is not supported")
				break
			}
			var i uint32
			present := false
			for i = 0; i < userAuthRequestGSSAPI.N; i++ {
				if userAuthRequestGSSAPI.OIDS[i].Equal(krb5Mesh) {
					present = true
					break
				}
			}
			if !present {
				authErr = fmt.Errorf("ssh: GSSAPI authentication must use the Kerberos V5 mechanism")
				break
			}
			// Initial server response, see RFC 4462 section 3.3.
			if err := s.transport.writePacket(Marshal(&userAuthGSSAPIResponse{
				SupportMech: krb5OID,
			})); err != nil {
				return nil, err
			}
			// Exchange token, see RFC 4462 section 3.4.
			packet, err := s.transport.readPacket()
			if err != nil {
				return nil, err
			}
			userAuthGSSAPITokenReq := &userAuthGSSAPIToken{}
			if err := Unmarshal(packet, userAuthGSSAPITokenReq); err != nil {
				return nil, err
			}
			authErr, perms, err = gssExchangeToken(gssapiConfig, userAuthGSSAPITokenReq.Token, s, sessionID,
				userAuthReq)
			if err != nil {
				return nil, err
			}
		default:
			authErr = fmt.Errorf("ssh: unknown method %q", userAuthReq.Method)
		}

		authErrs = append(authErrs, authErr)

		if config.AuthLogCallback != nil {
			config.AuthLogCallback(s, userAuthReq.Method, authErr)
		}

		var bannerErr *BannerError
		if errors.As(authErr, &bannerErr) {
			if bannerErr.Message != "" {
				bannerMsg := &userAuthBannerMsg{
					Message: bannerErr.Message,
				}
				if err := s.transport.writePacket(Marshal(bannerMsg)); err != nil {
					return nil, err
				}
			}
		}

		if authErr == nil {
			break userAuthLoop
		}

		var failureMsg userAuthFailureMsg

		if partialSuccess, ok := authErr.(*PartialSuccessError); ok {
			// After a partial success error we don't allow changing the user
			// name and execute the NoClientAuthCallback.
			partialSuccessReturned = true

			// In case a partial success is returned, the server may send
			// a new set of authentication methods.
			authConfig = partialSuccess.Next

			// Reset pubkey cache, as the new PublicKeyCallback might
			// accept a different set of public keys.
			cache = pubKeyCache{}

			// Send back a partial success message to the user.
			failureMsg.PartialSuccess = true
		} else {
			// Allow initial attempt of 'none' without penalty.
			if authFailures > 0 || userAuthReq.Method != "none" || noneAuthCount != 1 {
				authFailures++
			}
			if config.MaxAuthTries > 0 && authFailures >= config.MaxAuthTries {
				// If we have hit the max attempts, don't bother sending the
				// final SSH_MSG_USERAUTH_FAILURE message, since there are
				// no more authentication methods which can be attempted,
				// and this message may cause the client to re-attempt
				// authentication while we send the disconnect message.
				// Continue, and trigger the disconnect at the start of
				// the loop.
				//
				// The SSH specification is somewhat confusing about this,
				// RFC 4252 Section 5.1 requires each authentication failure
				// be responded to with a respective SSH_MSG_USERAUTH_FAILURE
				// message, but Section 4 says the server should disconnect
				// after some number of attempts, but it isn't explicit which
				// message should take precedence (i.e. should there be a failure
				// message than a disconnect message, or if we are going to
				// disconnect, should we only send that message.)
				//
				// Either way, OpenSSH disconnects immediately after the last
				// failed authentication attempt, and given they are typically
				// considered the golden implementation it seems reasonable
				// to match that behavior.
				continue
			}
		}

		if authConfig.PasswordCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "password")
		}
		if authConfig.PublicKeyCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "publickey")
		}
		if authConfig.KeyboardInteractiveCallback != nil {
			failureMsg.Methods = append(failureMsg.Methods, "keyboard-interactive")
		}
		if authConfig.GSSAPIWithMICConfig != nil && authConfig.GSSAPIWithMICConfig.Server != nil &&
			authConfig.GSSAPIWithMICConfig.AllowLogin != nil {
			failureMsg.Methods = append(failureMsg.Methods, "gssapi-with-mic")
		}

		if len(failureMsg.Methods) == 0 {
			return nil, errors.New("ssh: no authentication methods available")
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

func (c *sshClientKeyboardInteractive) Challenge(name, instruction string, questions []string, echos []bool) (answers []string, err error) {
	if len(questions) != len(echos) {
		return nil, errors.New("ssh: echos and questions must have equal length")
	}

	var prompts []byte
	for i := range questions {
		prompts = appendString(prompts, questions[i])
		prompts = appendBool(prompts, echos[i])
	}

	if err := c.transport.writePacket(Marshal(&userAuthInfoRequestMsg{
		Name:        name,
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
