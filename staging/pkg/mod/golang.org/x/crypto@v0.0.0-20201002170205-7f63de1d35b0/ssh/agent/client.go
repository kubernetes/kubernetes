// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package agent implements the ssh-agent protocol, and provides both
// a client and a server. The client can talk to a standard ssh-agent
// that uses UNIX sockets, and one could implement an alternative
// ssh-agent process using the sample server.
//
// References:
//  [PROTOCOL.agent]: https://tools.ietf.org/html/draft-miller-ssh-agent-00
package agent // import "golang.org/x/crypto/ssh/agent"

import (
	"bytes"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/big"
	"sync"

	"crypto"
	"golang.org/x/crypto/ed25519"
	"golang.org/x/crypto/ssh"
)

// SignatureFlags represent additional flags that can be passed to the signature
// requests an defined in [PROTOCOL.agent] section 4.5.1.
type SignatureFlags uint32

// SignatureFlag values as defined in [PROTOCOL.agent] section 5.3.
const (
	SignatureFlagReserved SignatureFlags = 1 << iota
	SignatureFlagRsaSha256
	SignatureFlagRsaSha512
)

// Agent represents the capabilities of an ssh-agent.
type Agent interface {
	// List returns the identities known to the agent.
	List() ([]*Key, error)

	// Sign has the agent sign the data using a protocol 2 key as defined
	// in [PROTOCOL.agent] section 2.6.2.
	Sign(key ssh.PublicKey, data []byte) (*ssh.Signature, error)

	// Add adds a private key to the agent.
	Add(key AddedKey) error

	// Remove removes all identities with the given public key.
	Remove(key ssh.PublicKey) error

	// RemoveAll removes all identities.
	RemoveAll() error

	// Lock locks the agent. Sign and Remove will fail, and List will empty an empty list.
	Lock(passphrase []byte) error

	// Unlock undoes the effect of Lock
	Unlock(passphrase []byte) error

	// Signers returns signers for all the known keys.
	Signers() ([]ssh.Signer, error)
}

type ExtendedAgent interface {
	Agent

	// SignWithFlags signs like Sign, but allows for additional flags to be sent/received
	SignWithFlags(key ssh.PublicKey, data []byte, flags SignatureFlags) (*ssh.Signature, error)

	// Extension processes a custom extension request. Standard-compliant agents are not
	// required to support any extensions, but this method allows agents to implement
	// vendor-specific methods or add experimental features. See [PROTOCOL.agent] section 4.7.
	// If agent extensions are unsupported entirely this method MUST return an
	// ErrExtensionUnsupported error. Similarly, if just the specific extensionType in
	// the request is unsupported by the agent then ErrExtensionUnsupported MUST be
	// returned.
	//
	// In the case of success, since [PROTOCOL.agent] section 4.7 specifies that the contents
	// of the response are unspecified (including the type of the message), the complete
	// response will be returned as a []byte slice, including the "type" byte of the message.
	Extension(extensionType string, contents []byte) ([]byte, error)
}

// ConstraintExtension describes an optional constraint defined by users.
type ConstraintExtension struct {
	// ExtensionName consist of a UTF-8 string suffixed by the
	// implementation domain following the naming scheme defined
	// in Section 4.2 of [RFC4251], e.g.  "foo@example.com".
	ExtensionName string
	// ExtensionDetails contains the actual content of the extended
	// constraint.
	ExtensionDetails []byte
}

// AddedKey describes an SSH key to be added to an Agent.
type AddedKey struct {
	// PrivateKey must be a *rsa.PrivateKey, *dsa.PrivateKey,
	// ed25519.PrivateKey or *ecdsa.PrivateKey, which will be inserted into the
	// agent.
	PrivateKey interface{}
	// Certificate, if not nil, is communicated to the agent and will be
	// stored with the key.
	Certificate *ssh.Certificate
	// Comment is an optional, free-form string.
	Comment string
	// LifetimeSecs, if not zero, is the number of seconds that the
	// agent will store the key for.
	LifetimeSecs uint32
	// ConfirmBeforeUse, if true, requests that the agent confirm with the
	// user before each use of this key.
	ConfirmBeforeUse bool
	// ConstraintExtensions are the experimental or private-use constraints
	// defined by users.
	ConstraintExtensions []ConstraintExtension
}

// See [PROTOCOL.agent], section 3.
const (
	agentRequestV1Identities   = 1
	agentRemoveAllV1Identities = 9

	// 3.2 Requests from client to agent for protocol 2 key operations
	agentAddIdentity         = 17
	agentRemoveIdentity      = 18
	agentRemoveAllIdentities = 19
	agentAddIDConstrained    = 25

	// 3.3 Key-type independent requests from client to agent
	agentAddSmartcardKey            = 20
	agentRemoveSmartcardKey         = 21
	agentLock                       = 22
	agentUnlock                     = 23
	agentAddSmartcardKeyConstrained = 26

	// 3.7 Key constraint identifiers
	agentConstrainLifetime  = 1
	agentConstrainConfirm   = 2
	agentConstrainExtension = 3
)

// maxAgentResponseBytes is the maximum agent reply size that is accepted. This
// is a sanity check, not a limit in the spec.
const maxAgentResponseBytes = 16 << 20

// Agent messages:
// These structures mirror the wire format of the corresponding ssh agent
// messages found in [PROTOCOL.agent].

// 3.4 Generic replies from agent to client
const agentFailure = 5

type failureAgentMsg struct{}

const agentSuccess = 6

type successAgentMsg struct{}

// See [PROTOCOL.agent], section 2.5.2.
const agentRequestIdentities = 11

type requestIdentitiesAgentMsg struct{}

// See [PROTOCOL.agent], section 2.5.2.
const agentIdentitiesAnswer = 12

type identitiesAnswerAgentMsg struct {
	NumKeys uint32 `sshtype:"12"`
	Keys    []byte `ssh:"rest"`
}

// See [PROTOCOL.agent], section 2.6.2.
const agentSignRequest = 13

type signRequestAgentMsg struct {
	KeyBlob []byte `sshtype:"13"`
	Data    []byte
	Flags   uint32
}

// See [PROTOCOL.agent], section 2.6.2.

// 3.6 Replies from agent to client for protocol 2 key operations
const agentSignResponse = 14

type signResponseAgentMsg struct {
	SigBlob []byte `sshtype:"14"`
}

type publicKey struct {
	Format string
	Rest   []byte `ssh:"rest"`
}

// 3.7 Key constraint identifiers
type constrainLifetimeAgentMsg struct {
	LifetimeSecs uint32 `sshtype:"1"`
}

type constrainExtensionAgentMsg struct {
	ExtensionName    string `sshtype:"3"`
	ExtensionDetails []byte

	// Rest is a field used for parsing, not part of message
	Rest []byte `ssh:"rest"`
}

// See [PROTOCOL.agent], section 4.7
const agentExtension = 27
const agentExtensionFailure = 28

// ErrExtensionUnsupported indicates that an extension defined in
// [PROTOCOL.agent] section 4.7 is unsupported by the agent. Specifically this
// error indicates that the agent returned a standard SSH_AGENT_FAILURE message
// as the result of a SSH_AGENTC_EXTENSION request. Note that the protocol
// specification (and therefore this error) does not distinguish between a
// specific extension being unsupported and extensions being unsupported entirely.
var ErrExtensionUnsupported = errors.New("agent: extension unsupported")

type extensionAgentMsg struct {
	ExtensionType string `sshtype:"27"`
	Contents      []byte
}

// Key represents a protocol 2 public key as defined in
// [PROTOCOL.agent], section 2.5.2.
type Key struct {
	Format  string
	Blob    []byte
	Comment string
}

func clientErr(err error) error {
	return fmt.Errorf("agent: client error: %v", err)
}

// String returns the storage form of an agent key with the format, base64
// encoded serialized key, and the comment if it is not empty.
func (k *Key) String() string {
	s := string(k.Format) + " " + base64.StdEncoding.EncodeToString(k.Blob)

	if k.Comment != "" {
		s += " " + k.Comment
	}

	return s
}

// Type returns the public key type.
func (k *Key) Type() string {
	return k.Format
}

// Marshal returns key blob to satisfy the ssh.PublicKey interface.
func (k *Key) Marshal() []byte {
	return k.Blob
}

// Verify satisfies the ssh.PublicKey interface.
func (k *Key) Verify(data []byte, sig *ssh.Signature) error {
	pubKey, err := ssh.ParsePublicKey(k.Blob)
	if err != nil {
		return fmt.Errorf("agent: bad public key: %v", err)
	}
	return pubKey.Verify(data, sig)
}

type wireKey struct {
	Format string
	Rest   []byte `ssh:"rest"`
}

func parseKey(in []byte) (out *Key, rest []byte, err error) {
	var record struct {
		Blob    []byte
		Comment string
		Rest    []byte `ssh:"rest"`
	}

	if err := ssh.Unmarshal(in, &record); err != nil {
		return nil, nil, err
	}

	var wk wireKey
	if err := ssh.Unmarshal(record.Blob, &wk); err != nil {
		return nil, nil, err
	}

	return &Key{
		Format:  wk.Format,
		Blob:    record.Blob,
		Comment: record.Comment,
	}, record.Rest, nil
}

// client is a client for an ssh-agent process.
type client struct {
	// conn is typically a *net.UnixConn
	conn io.ReadWriter
	// mu is used to prevent concurrent access to the agent
	mu sync.Mutex
}

// NewClient returns an Agent that talks to an ssh-agent process over
// the given connection.
func NewClient(rw io.ReadWriter) ExtendedAgent {
	return &client{conn: rw}
}

// call sends an RPC to the agent. On success, the reply is
// unmarshaled into reply and replyType is set to the first byte of
// the reply, which contains the type of the message.
func (c *client) call(req []byte) (reply interface{}, err error) {
	buf, err := c.callRaw(req)
	if err != nil {
		return nil, err
	}
	reply, err = unmarshal(buf)
	if err != nil {
		return nil, clientErr(err)
	}
	return reply, nil
}

// callRaw sends an RPC to the agent. On success, the raw
// bytes of the response are returned; no unmarshalling is
// performed on the response.
func (c *client) callRaw(req []byte) (reply []byte, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	msg := make([]byte, 4+len(req))
	binary.BigEndian.PutUint32(msg, uint32(len(req)))
	copy(msg[4:], req)
	if _, err = c.conn.Write(msg); err != nil {
		return nil, clientErr(err)
	}

	var respSizeBuf [4]byte
	if _, err = io.ReadFull(c.conn, respSizeBuf[:]); err != nil {
		return nil, clientErr(err)
	}
	respSize := binary.BigEndian.Uint32(respSizeBuf[:])
	if respSize > maxAgentResponseBytes {
		return nil, clientErr(errors.New("response too large"))
	}

	buf := make([]byte, respSize)
	if _, err = io.ReadFull(c.conn, buf); err != nil {
		return nil, clientErr(err)
	}
	return buf, nil
}

func (c *client) simpleCall(req []byte) error {
	resp, err := c.call(req)
	if err != nil {
		return err
	}
	if _, ok := resp.(*successAgentMsg); ok {
		return nil
	}
	return errors.New("agent: failure")
}

func (c *client) RemoveAll() error {
	return c.simpleCall([]byte{agentRemoveAllIdentities})
}

func (c *client) Remove(key ssh.PublicKey) error {
	req := ssh.Marshal(&agentRemoveIdentityMsg{
		KeyBlob: key.Marshal(),
	})
	return c.simpleCall(req)
}

func (c *client) Lock(passphrase []byte) error {
	req := ssh.Marshal(&agentLockMsg{
		Passphrase: passphrase,
	})
	return c.simpleCall(req)
}

func (c *client) Unlock(passphrase []byte) error {
	req := ssh.Marshal(&agentUnlockMsg{
		Passphrase: passphrase,
	})
	return c.simpleCall(req)
}

// List returns the identities known to the agent.
func (c *client) List() ([]*Key, error) {
	// see [PROTOCOL.agent] section 2.5.2.
	req := []byte{agentRequestIdentities}

	msg, err := c.call(req)
	if err != nil {
		return nil, err
	}

	switch msg := msg.(type) {
	case *identitiesAnswerAgentMsg:
		if msg.NumKeys > maxAgentResponseBytes/8 {
			return nil, errors.New("agent: too many keys in agent reply")
		}
		keys := make([]*Key, msg.NumKeys)
		data := msg.Keys
		for i := uint32(0); i < msg.NumKeys; i++ {
			var key *Key
			var err error
			if key, data, err = parseKey(data); err != nil {
				return nil, err
			}
			keys[i] = key
		}
		return keys, nil
	case *failureAgentMsg:
		return nil, errors.New("agent: failed to list keys")
	}
	panic("unreachable")
}

// Sign has the agent sign the data using a protocol 2 key as defined
// in [PROTOCOL.agent] section 2.6.2.
func (c *client) Sign(key ssh.PublicKey, data []byte) (*ssh.Signature, error) {
	return c.SignWithFlags(key, data, 0)
}

func (c *client) SignWithFlags(key ssh.PublicKey, data []byte, flags SignatureFlags) (*ssh.Signature, error) {
	req := ssh.Marshal(signRequestAgentMsg{
		KeyBlob: key.Marshal(),
		Data:    data,
		Flags:   uint32(flags),
	})

	msg, err := c.call(req)
	if err != nil {
		return nil, err
	}

	switch msg := msg.(type) {
	case *signResponseAgentMsg:
		var sig ssh.Signature
		if err := ssh.Unmarshal(msg.SigBlob, &sig); err != nil {
			return nil, err
		}

		return &sig, nil
	case *failureAgentMsg:
		return nil, errors.New("agent: failed to sign challenge")
	}
	panic("unreachable")
}

// unmarshal parses an agent message in packet, returning the parsed
// form and the message type of packet.
func unmarshal(packet []byte) (interface{}, error) {
	if len(packet) < 1 {
		return nil, errors.New("agent: empty packet")
	}
	var msg interface{}
	switch packet[0] {
	case agentFailure:
		return new(failureAgentMsg), nil
	case agentSuccess:
		return new(successAgentMsg), nil
	case agentIdentitiesAnswer:
		msg = new(identitiesAnswerAgentMsg)
	case agentSignResponse:
		msg = new(signResponseAgentMsg)
	case agentV1IdentitiesAnswer:
		msg = new(agentV1IdentityMsg)
	default:
		return nil, fmt.Errorf("agent: unknown type tag %d", packet[0])
	}
	if err := ssh.Unmarshal(packet, msg); err != nil {
		return nil, err
	}
	return msg, nil
}

type rsaKeyMsg struct {
	Type        string `sshtype:"17|25"`
	N           *big.Int
	E           *big.Int
	D           *big.Int
	Iqmp        *big.Int // IQMP = Inverse Q Mod P
	P           *big.Int
	Q           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type dsaKeyMsg struct {
	Type        string `sshtype:"17|25"`
	P           *big.Int
	Q           *big.Int
	G           *big.Int
	Y           *big.Int
	X           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type ecdsaKeyMsg struct {
	Type        string `sshtype:"17|25"`
	Curve       string
	KeyBytes    []byte
	D           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type ed25519KeyMsg struct {
	Type        string `sshtype:"17|25"`
	Pub         []byte
	Priv        []byte
	Comments    string
	Constraints []byte `ssh:"rest"`
}

// Insert adds a private key to the agent.
func (c *client) insertKey(s interface{}, comment string, constraints []byte) error {
	var req []byte
	switch k := s.(type) {
	case *rsa.PrivateKey:
		if len(k.Primes) != 2 {
			return fmt.Errorf("agent: unsupported RSA key with %d primes", len(k.Primes))
		}
		k.Precompute()
		req = ssh.Marshal(rsaKeyMsg{
			Type:        ssh.KeyAlgoRSA,
			N:           k.N,
			E:           big.NewInt(int64(k.E)),
			D:           k.D,
			Iqmp:        k.Precomputed.Qinv,
			P:           k.Primes[0],
			Q:           k.Primes[1],
			Comments:    comment,
			Constraints: constraints,
		})
	case *dsa.PrivateKey:
		req = ssh.Marshal(dsaKeyMsg{
			Type:        ssh.KeyAlgoDSA,
			P:           k.P,
			Q:           k.Q,
			G:           k.G,
			Y:           k.Y,
			X:           k.X,
			Comments:    comment,
			Constraints: constraints,
		})
	case *ecdsa.PrivateKey:
		nistID := fmt.Sprintf("nistp%d", k.Params().BitSize)
		req = ssh.Marshal(ecdsaKeyMsg{
			Type:        "ecdsa-sha2-" + nistID,
			Curve:       nistID,
			KeyBytes:    elliptic.Marshal(k.Curve, k.X, k.Y),
			D:           k.D,
			Comments:    comment,
			Constraints: constraints,
		})
	case ed25519.PrivateKey:
		req = ssh.Marshal(ed25519KeyMsg{
			Type:        ssh.KeyAlgoED25519,
			Pub:         []byte(k)[32:],
			Priv:        []byte(k),
			Comments:    comment,
			Constraints: constraints,
		})
	// This function originally supported only *ed25519.PrivateKey, however the
	// general idiom is to pass ed25519.PrivateKey by value, not by pointer.
	// We still support the pointer variant for backwards compatibility.
	case *ed25519.PrivateKey:
		req = ssh.Marshal(ed25519KeyMsg{
			Type:        ssh.KeyAlgoED25519,
			Pub:         []byte(*k)[32:],
			Priv:        []byte(*k),
			Comments:    comment,
			Constraints: constraints,
		})
	default:
		return fmt.Errorf("agent: unsupported key type %T", s)
	}

	// if constraints are present then the message type needs to be changed.
	if len(constraints) != 0 {
		req[0] = agentAddIDConstrained
	}

	resp, err := c.call(req)
	if err != nil {
		return err
	}
	if _, ok := resp.(*successAgentMsg); ok {
		return nil
	}
	return errors.New("agent: failure")
}

type rsaCertMsg struct {
	Type        string `sshtype:"17|25"`
	CertBytes   []byte
	D           *big.Int
	Iqmp        *big.Int // IQMP = Inverse Q Mod P
	P           *big.Int
	Q           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type dsaCertMsg struct {
	Type        string `sshtype:"17|25"`
	CertBytes   []byte
	X           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type ecdsaCertMsg struct {
	Type        string `sshtype:"17|25"`
	CertBytes   []byte
	D           *big.Int
	Comments    string
	Constraints []byte `ssh:"rest"`
}

type ed25519CertMsg struct {
	Type        string `sshtype:"17|25"`
	CertBytes   []byte
	Pub         []byte
	Priv        []byte
	Comments    string
	Constraints []byte `ssh:"rest"`
}

// Add adds a private key to the agent. If a certificate is given,
// that certificate is added instead as public key.
func (c *client) Add(key AddedKey) error {
	var constraints []byte

	if secs := key.LifetimeSecs; secs != 0 {
		constraints = append(constraints, ssh.Marshal(constrainLifetimeAgentMsg{secs})...)
	}

	if key.ConfirmBeforeUse {
		constraints = append(constraints, agentConstrainConfirm)
	}

	cert := key.Certificate
	if cert == nil {
		return c.insertKey(key.PrivateKey, key.Comment, constraints)
	}
	return c.insertCert(key.PrivateKey, cert, key.Comment, constraints)
}

func (c *client) insertCert(s interface{}, cert *ssh.Certificate, comment string, constraints []byte) error {
	var req []byte
	switch k := s.(type) {
	case *rsa.PrivateKey:
		if len(k.Primes) != 2 {
			return fmt.Errorf("agent: unsupported RSA key with %d primes", len(k.Primes))
		}
		k.Precompute()
		req = ssh.Marshal(rsaCertMsg{
			Type:        cert.Type(),
			CertBytes:   cert.Marshal(),
			D:           k.D,
			Iqmp:        k.Precomputed.Qinv,
			P:           k.Primes[0],
			Q:           k.Primes[1],
			Comments:    comment,
			Constraints: constraints,
		})
	case *dsa.PrivateKey:
		req = ssh.Marshal(dsaCertMsg{
			Type:        cert.Type(),
			CertBytes:   cert.Marshal(),
			X:           k.X,
			Comments:    comment,
			Constraints: constraints,
		})
	case *ecdsa.PrivateKey:
		req = ssh.Marshal(ecdsaCertMsg{
			Type:        cert.Type(),
			CertBytes:   cert.Marshal(),
			D:           k.D,
			Comments:    comment,
			Constraints: constraints,
		})
	case ed25519.PrivateKey:
		req = ssh.Marshal(ed25519CertMsg{
			Type:        cert.Type(),
			CertBytes:   cert.Marshal(),
			Pub:         []byte(k)[32:],
			Priv:        []byte(k),
			Comments:    comment,
			Constraints: constraints,
		})
	// This function originally supported only *ed25519.PrivateKey, however the
	// general idiom is to pass ed25519.PrivateKey by value, not by pointer.
	// We still support the pointer variant for backwards compatibility.
	case *ed25519.PrivateKey:
		req = ssh.Marshal(ed25519CertMsg{
			Type:        cert.Type(),
			CertBytes:   cert.Marshal(),
			Pub:         []byte(*k)[32:],
			Priv:        []byte(*k),
			Comments:    comment,
			Constraints: constraints,
		})
	default:
		return fmt.Errorf("agent: unsupported key type %T", s)
	}

	// if constraints are present then the message type needs to be changed.
	if len(constraints) != 0 {
		req[0] = agentAddIDConstrained
	}

	signer, err := ssh.NewSignerFromKey(s)
	if err != nil {
		return err
	}
	if bytes.Compare(cert.Key.Marshal(), signer.PublicKey().Marshal()) != 0 {
		return errors.New("agent: signer and cert have different public key")
	}

	resp, err := c.call(req)
	if err != nil {
		return err
	}
	if _, ok := resp.(*successAgentMsg); ok {
		return nil
	}
	return errors.New("agent: failure")
}

// Signers provides a callback for client authentication.
func (c *client) Signers() ([]ssh.Signer, error) {
	keys, err := c.List()
	if err != nil {
		return nil, err
	}

	var result []ssh.Signer
	for _, k := range keys {
		result = append(result, &agentKeyringSigner{c, k})
	}
	return result, nil
}

type agentKeyringSigner struct {
	agent *client
	pub   ssh.PublicKey
}

func (s *agentKeyringSigner) PublicKey() ssh.PublicKey {
	return s.pub
}

func (s *agentKeyringSigner) Sign(rand io.Reader, data []byte) (*ssh.Signature, error) {
	// The agent has its own entropy source, so the rand argument is ignored.
	return s.agent.Sign(s.pub, data)
}

func (s *agentKeyringSigner) SignWithOpts(rand io.Reader, data []byte, opts crypto.SignerOpts) (*ssh.Signature, error) {
	var flags SignatureFlags
	if opts != nil {
		switch opts.HashFunc() {
		case crypto.SHA256:
			flags = SignatureFlagRsaSha256
		case crypto.SHA512:
			flags = SignatureFlagRsaSha512
		}
	}
	return s.agent.SignWithFlags(s.pub, data, flags)
}

// Calls an extension method. It is up to the agent implementation as to whether or not
// any particular extension is supported and may always return an error. Because the
// type of the response is up to the implementation, this returns the bytes of the
// response and does not attempt any type of unmarshalling.
func (c *client) Extension(extensionType string, contents []byte) ([]byte, error) {
	req := ssh.Marshal(extensionAgentMsg{
		ExtensionType: extensionType,
		Contents:      contents,
	})
	buf, err := c.callRaw(req)
	if err != nil {
		return nil, err
	}
	if len(buf) == 0 {
		return nil, errors.New("agent: failure; empty response")
	}
	// [PROTOCOL.agent] section 4.7 indicates that an SSH_AGENT_FAILURE message
	// represents an agent that does not support the extension
	if buf[0] == agentFailure {
		return nil, ErrExtensionUnsupported
	}
	if buf[0] == agentExtensionFailure {
		return nil, errors.New("agent: generic extension failure")
	}

	return buf, nil
}
