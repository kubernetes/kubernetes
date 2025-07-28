// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"sort"
	"time"
)

// Certificate algorithm names from [PROTOCOL.certkeys]. These values can appear
// in Certificate.Type, PublicKey.Type, and ClientConfig.HostKeyAlgorithms.
// Unlike key algorithm names, these are not passed to AlgorithmSigner nor
// returned by MultiAlgorithmSigner and don't appear in the Signature.Format
// field.
const (
	CertAlgoRSAv01 = "ssh-rsa-cert-v01@openssh.com"
	// Deprecated: DSA is only supported at insecure key sizes, and was removed
	// from major implementations.
	CertAlgoDSAv01 = InsecureCertAlgoDSAv01
	// Deprecated: DSA is only supported at insecure key sizes, and was removed
	// from major implementations.
	InsecureCertAlgoDSAv01 = "ssh-dss-cert-v01@openssh.com"
	CertAlgoECDSA256v01    = "ecdsa-sha2-nistp256-cert-v01@openssh.com"
	CertAlgoECDSA384v01    = "ecdsa-sha2-nistp384-cert-v01@openssh.com"
	CertAlgoECDSA521v01    = "ecdsa-sha2-nistp521-cert-v01@openssh.com"
	CertAlgoSKECDSA256v01  = "sk-ecdsa-sha2-nistp256-cert-v01@openssh.com"
	CertAlgoED25519v01     = "ssh-ed25519-cert-v01@openssh.com"
	CertAlgoSKED25519v01   = "sk-ssh-ed25519-cert-v01@openssh.com"

	// CertAlgoRSASHA256v01 and CertAlgoRSASHA512v01 can't appear as a
	// Certificate.Type (or PublicKey.Type), but only in
	// ClientConfig.HostKeyAlgorithms.
	CertAlgoRSASHA256v01 = "rsa-sha2-256-cert-v01@openssh.com"
	CertAlgoRSASHA512v01 = "rsa-sha2-512-cert-v01@openssh.com"
)

const (
	// Deprecated: use CertAlgoRSAv01.
	CertSigAlgoRSAv01 = CertAlgoRSAv01
	// Deprecated: use CertAlgoRSASHA256v01.
	CertSigAlgoRSASHA2256v01 = CertAlgoRSASHA256v01
	// Deprecated: use CertAlgoRSASHA512v01.
	CertSigAlgoRSASHA2512v01 = CertAlgoRSASHA512v01
)

// Certificate types distinguish between host and user
// certificates. The values can be set in the CertType field of
// Certificate.
const (
	UserCert = 1
	HostCert = 2
)

// Signature represents a cryptographic signature.
type Signature struct {
	Format string
	Blob   []byte
	Rest   []byte `ssh:"rest"`
}

// CertTimeInfinity can be used for OpenSSHCertV01.ValidBefore to indicate that
// a certificate does not expire.
const CertTimeInfinity = 1<<64 - 1

// An Certificate represents an OpenSSH certificate as defined in
// [PROTOCOL.certkeys]?rev=1.8. The Certificate type implements the
// PublicKey interface, so it can be unmarshaled using
// ParsePublicKey.
type Certificate struct {
	Nonce           []byte
	Key             PublicKey
	Serial          uint64
	CertType        uint32
	KeyId           string
	ValidPrincipals []string
	ValidAfter      uint64
	ValidBefore     uint64
	Permissions
	Reserved     []byte
	SignatureKey PublicKey
	Signature    *Signature
}

// genericCertData holds the key-independent part of the certificate data.
// Overall, certificates contain an nonce, public key fields and
// key-independent fields.
type genericCertData struct {
	Serial          uint64
	CertType        uint32
	KeyId           string
	ValidPrincipals []byte
	ValidAfter      uint64
	ValidBefore     uint64
	CriticalOptions []byte
	Extensions      []byte
	Reserved        []byte
	SignatureKey    []byte
	Signature       []byte
}

func marshalStringList(namelist []string) []byte {
	var to []byte
	for _, name := range namelist {
		s := struct{ N string }{name}
		to = append(to, Marshal(&s)...)
	}
	return to
}

type optionsTuple struct {
	Key   string
	Value []byte
}

type optionsTupleValue struct {
	Value string
}

// serialize a map of critical options or extensions
// issue #10569 - per [PROTOCOL.certkeys] and SSH implementation,
// we need two length prefixes for a non-empty string value
func marshalTuples(tups map[string]string) []byte {
	keys := make([]string, 0, len(tups))
	for key := range tups {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var ret []byte
	for _, key := range keys {
		s := optionsTuple{Key: key}
		if value := tups[key]; len(value) > 0 {
			s.Value = Marshal(&optionsTupleValue{value})
		}
		ret = append(ret, Marshal(&s)...)
	}
	return ret
}

// issue #10569 - per [PROTOCOL.certkeys] and SSH implementation,
// we need two length prefixes for a non-empty option value
func parseTuples(in []byte) (map[string]string, error) {
	tups := map[string]string{}
	var lastKey string
	var haveLastKey bool

	for len(in) > 0 {
		var key, val, extra []byte
		var ok bool

		if key, in, ok = parseString(in); !ok {
			return nil, errShortRead
		}
		keyStr := string(key)
		// according to [PROTOCOL.certkeys], the names must be in
		// lexical order.
		if haveLastKey && keyStr <= lastKey {
			return nil, fmt.Errorf("ssh: certificate options are not in lexical order")
		}
		lastKey, haveLastKey = keyStr, true
		// the next field is a data field, which if non-empty has a string embedded
		if val, in, ok = parseString(in); !ok {
			return nil, errShortRead
		}
		if len(val) > 0 {
			val, extra, ok = parseString(val)
			if !ok {
				return nil, errShortRead
			}
			if len(extra) > 0 {
				return nil, fmt.Errorf("ssh: unexpected trailing data after certificate option value")
			}
			tups[keyStr] = string(val)
		} else {
			tups[keyStr] = ""
		}
	}
	return tups, nil
}

func parseCert(in []byte, privAlgo string) (*Certificate, error) {
	nonce, rest, ok := parseString(in)
	if !ok {
		return nil, errShortRead
	}

	key, rest, err := parsePubKey(rest, privAlgo)
	if err != nil {
		return nil, err
	}

	var g genericCertData
	if err := Unmarshal(rest, &g); err != nil {
		return nil, err
	}

	c := &Certificate{
		Nonce:       nonce,
		Key:         key,
		Serial:      g.Serial,
		CertType:    g.CertType,
		KeyId:       g.KeyId,
		ValidAfter:  g.ValidAfter,
		ValidBefore: g.ValidBefore,
	}

	for principals := g.ValidPrincipals; len(principals) > 0; {
		principal, rest, ok := parseString(principals)
		if !ok {
			return nil, errShortRead
		}
		c.ValidPrincipals = append(c.ValidPrincipals, string(principal))
		principals = rest
	}

	c.CriticalOptions, err = parseTuples(g.CriticalOptions)
	if err != nil {
		return nil, err
	}
	c.Extensions, err = parseTuples(g.Extensions)
	if err != nil {
		return nil, err
	}
	c.Reserved = g.Reserved
	k, err := ParsePublicKey(g.SignatureKey)
	if err != nil {
		return nil, err
	}

	c.SignatureKey = k
	c.Signature, rest, ok = parseSignatureBody(g.Signature)
	if !ok || len(rest) > 0 {
		return nil, errors.New("ssh: signature parse error")
	}

	return c, nil
}

type openSSHCertSigner struct {
	pub    *Certificate
	signer Signer
}

type algorithmOpenSSHCertSigner struct {
	*openSSHCertSigner
	algorithmSigner AlgorithmSigner
}

// NewCertSigner returns a Signer that signs with the given Certificate, whose
// private key is held by signer. It returns an error if the public key in cert
// doesn't match the key used by signer.
func NewCertSigner(cert *Certificate, signer Signer) (Signer, error) {
	if !bytes.Equal(cert.Key.Marshal(), signer.PublicKey().Marshal()) {
		return nil, errors.New("ssh: signer and cert have different public key")
	}

	switch s := signer.(type) {
	case MultiAlgorithmSigner:
		return &multiAlgorithmSigner{
			AlgorithmSigner: &algorithmOpenSSHCertSigner{
				&openSSHCertSigner{cert, signer}, s},
			supportedAlgorithms: s.Algorithms(),
		}, nil
	case AlgorithmSigner:
		return &algorithmOpenSSHCertSigner{
			&openSSHCertSigner{cert, signer}, s}, nil
	default:
		return &openSSHCertSigner{cert, signer}, nil
	}
}

func (s *openSSHCertSigner) Sign(rand io.Reader, data []byte) (*Signature, error) {
	return s.signer.Sign(rand, data)
}

func (s *openSSHCertSigner) PublicKey() PublicKey {
	return s.pub
}

func (s *algorithmOpenSSHCertSigner) SignWithAlgorithm(rand io.Reader, data []byte, algorithm string) (*Signature, error) {
	return s.algorithmSigner.SignWithAlgorithm(rand, data, algorithm)
}

const sourceAddressCriticalOption = "source-address"

// CertChecker does the work of verifying a certificate. Its methods
// can be plugged into ClientConfig.HostKeyCallback and
// ServerConfig.PublicKeyCallback. For the CertChecker to work,
// minimally, the IsAuthority callback should be set.
type CertChecker struct {
	// SupportedCriticalOptions lists the CriticalOptions that the
	// server application layer understands. These are only used
	// for user certificates.
	SupportedCriticalOptions []string

	// IsUserAuthority should return true if the key is recognized as an
	// authority for the given user certificate. This allows for
	// certificates to be signed by other certificates. This must be set
	// if this CertChecker will be checking user certificates.
	IsUserAuthority func(auth PublicKey) bool

	// IsHostAuthority should report whether the key is recognized as
	// an authority for this host. This allows for certificates to be
	// signed by other keys, and for those other keys to only be valid
	// signers for particular hostnames. This must be set if this
	// CertChecker will be checking host certificates.
	IsHostAuthority func(auth PublicKey, address string) bool

	// Clock is used for verifying time stamps. If nil, time.Now
	// is used.
	Clock func() time.Time

	// UserKeyFallback is called when CertChecker.Authenticate encounters a
	// public key that is not a certificate. It must implement validation
	// of user keys or else, if nil, all such keys are rejected.
	UserKeyFallback func(conn ConnMetadata, key PublicKey) (*Permissions, error)

	// HostKeyFallback is called when CertChecker.CheckHostKey encounters a
	// public key that is not a certificate. It must implement host key
	// validation or else, if nil, all such keys are rejected.
	HostKeyFallback HostKeyCallback

	// IsRevoked is called for each certificate so that revocation checking
	// can be implemented. It should return true if the given certificate
	// is revoked and false otherwise. If nil, no certificates are
	// considered to have been revoked.
	IsRevoked func(cert *Certificate) bool
}

// CheckHostKey checks a host key certificate. This method can be
// plugged into ClientConfig.HostKeyCallback.
func (c *CertChecker) CheckHostKey(addr string, remote net.Addr, key PublicKey) error {
	cert, ok := key.(*Certificate)
	if !ok {
		if c.HostKeyFallback != nil {
			return c.HostKeyFallback(addr, remote, key)
		}
		return errors.New("ssh: non-certificate host key")
	}
	if cert.CertType != HostCert {
		return fmt.Errorf("ssh: certificate presented as a host key has type %d", cert.CertType)
	}
	if !c.IsHostAuthority(cert.SignatureKey, addr) {
		return fmt.Errorf("ssh: no authorities for hostname: %v", addr)
	}

	hostname, _, err := net.SplitHostPort(addr)
	if err != nil {
		return err
	}

	// Pass hostname only as principal for host certificates (consistent with OpenSSH)
	return c.CheckCert(hostname, cert)
}

// Authenticate checks a user certificate. Authenticate can be used as
// a value for ServerConfig.PublicKeyCallback.
func (c *CertChecker) Authenticate(conn ConnMetadata, pubKey PublicKey) (*Permissions, error) {
	cert, ok := pubKey.(*Certificate)
	if !ok {
		if c.UserKeyFallback != nil {
			return c.UserKeyFallback(conn, pubKey)
		}
		return nil, errors.New("ssh: normal key pairs not accepted")
	}

	if cert.CertType != UserCert {
		return nil, fmt.Errorf("ssh: cert has type %d", cert.CertType)
	}
	if !c.IsUserAuthority(cert.SignatureKey) {
		return nil, fmt.Errorf("ssh: certificate signed by unrecognized authority")
	}

	if err := c.CheckCert(conn.User(), cert); err != nil {
		return nil, err
	}

	return &cert.Permissions, nil
}

// CheckCert checks CriticalOptions, ValidPrincipals, revocation, timestamp and
// the signature of the certificate.
func (c *CertChecker) CheckCert(principal string, cert *Certificate) error {
	if c.IsRevoked != nil && c.IsRevoked(cert) {
		return fmt.Errorf("ssh: certificate serial %d revoked", cert.Serial)
	}

	for opt := range cert.CriticalOptions {
		// sourceAddressCriticalOption will be enforced by
		// serverAuthenticate
		if opt == sourceAddressCriticalOption {
			continue
		}

		found := false
		for _, supp := range c.SupportedCriticalOptions {
			if supp == opt {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("ssh: unsupported critical option %q in certificate", opt)
		}
	}

	if len(cert.ValidPrincipals) > 0 {
		// By default, certs are valid for all users/hosts.
		found := false
		for _, p := range cert.ValidPrincipals {
			if p == principal {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("ssh: principal %q not in the set of valid principals for given certificate: %q", principal, cert.ValidPrincipals)
		}
	}

	clock := c.Clock
	if clock == nil {
		clock = time.Now
	}

	unixNow := clock().Unix()
	if after := int64(cert.ValidAfter); after < 0 || unixNow < int64(cert.ValidAfter) {
		return fmt.Errorf("ssh: cert is not yet valid")
	}
	if before := int64(cert.ValidBefore); cert.ValidBefore != uint64(CertTimeInfinity) && (unixNow >= before || before < 0) {
		return fmt.Errorf("ssh: cert has expired")
	}
	if err := cert.SignatureKey.Verify(cert.bytesForSigning(), cert.Signature); err != nil {
		return fmt.Errorf("ssh: certificate signature does not verify")
	}

	return nil
}

// SignCert signs the certificate with an authority, setting the Nonce,
// SignatureKey, and Signature fields. If the authority implements the
// MultiAlgorithmSigner interface the first algorithm in the list is used. This
// is useful if you want to sign with a specific algorithm.
func (c *Certificate) SignCert(rand io.Reader, authority Signer) error {
	c.Nonce = make([]byte, 32)
	if _, err := io.ReadFull(rand, c.Nonce); err != nil {
		return err
	}
	c.SignatureKey = authority.PublicKey()

	if v, ok := authority.(MultiAlgorithmSigner); ok {
		if len(v.Algorithms()) == 0 {
			return errors.New("the provided authority has no signature algorithm")
		}
		// Use the first algorithm in the list.
		sig, err := v.SignWithAlgorithm(rand, c.bytesForSigning(), v.Algorithms()[0])
		if err != nil {
			return err
		}
		c.Signature = sig
		return nil
	} else if v, ok := authority.(AlgorithmSigner); ok && v.PublicKey().Type() == KeyAlgoRSA {
		// Default to KeyAlgoRSASHA512 for ssh-rsa signers.
		// TODO: consider using KeyAlgoRSASHA256 as default.
		sig, err := v.SignWithAlgorithm(rand, c.bytesForSigning(), KeyAlgoRSASHA512)
		if err != nil {
			return err
		}
		c.Signature = sig
		return nil
	}

	sig, err := authority.Sign(rand, c.bytesForSigning())
	if err != nil {
		return err
	}
	c.Signature = sig
	return nil
}

// certKeyAlgoNames is a mapping from known certificate algorithm names to the
// corresponding public key signature algorithm.
//
// This map must be kept in sync with the one in agent/client.go.
var certKeyAlgoNames = map[string]string{
	CertAlgoRSAv01:         KeyAlgoRSA,
	CertAlgoRSASHA256v01:   KeyAlgoRSASHA256,
	CertAlgoRSASHA512v01:   KeyAlgoRSASHA512,
	InsecureCertAlgoDSAv01: InsecureKeyAlgoDSA,
	CertAlgoECDSA256v01:    KeyAlgoECDSA256,
	CertAlgoECDSA384v01:    KeyAlgoECDSA384,
	CertAlgoECDSA521v01:    KeyAlgoECDSA521,
	CertAlgoSKECDSA256v01:  KeyAlgoSKECDSA256,
	CertAlgoED25519v01:     KeyAlgoED25519,
	CertAlgoSKED25519v01:   KeyAlgoSKED25519,
}

// underlyingAlgo returns the signature algorithm associated with algo (which is
// an advertised or negotiated public key or host key algorithm). These are
// usually the same, except for certificate algorithms.
func underlyingAlgo(algo string) string {
	if a, ok := certKeyAlgoNames[algo]; ok {
		return a
	}
	return algo
}

// certificateAlgo returns the certificate algorithms that uses the provided
// underlying signature algorithm.
func certificateAlgo(algo string) (certAlgo string, ok bool) {
	for certName, algoName := range certKeyAlgoNames {
		if algoName == algo {
			return certName, true
		}
	}
	return "", false
}

func (cert *Certificate) bytesForSigning() []byte {
	c2 := *cert
	c2.Signature = nil
	out := c2.Marshal()
	// Drop trailing signature length.
	return out[:len(out)-4]
}

// Marshal serializes c into OpenSSH's wire format. It is part of the
// PublicKey interface.
func (c *Certificate) Marshal() []byte {
	generic := genericCertData{
		Serial:          c.Serial,
		CertType:        c.CertType,
		KeyId:           c.KeyId,
		ValidPrincipals: marshalStringList(c.ValidPrincipals),
		ValidAfter:      uint64(c.ValidAfter),
		ValidBefore:     uint64(c.ValidBefore),
		CriticalOptions: marshalTuples(c.CriticalOptions),
		Extensions:      marshalTuples(c.Extensions),
		Reserved:        c.Reserved,
		SignatureKey:    c.SignatureKey.Marshal(),
	}
	if c.Signature != nil {
		generic.Signature = Marshal(c.Signature)
	}
	genericBytes := Marshal(&generic)
	keyBytes := c.Key.Marshal()
	_, keyBytes, _ = parseString(keyBytes)
	prefix := Marshal(&struct {
		Name  string
		Nonce []byte
		Key   []byte `ssh:"rest"`
	}{c.Type(), c.Nonce, keyBytes})

	result := make([]byte, 0, len(prefix)+len(genericBytes))
	result = append(result, prefix...)
	result = append(result, genericBytes...)
	return result
}

// Type returns the certificate algorithm name. It is part of the PublicKey interface.
func (c *Certificate) Type() string {
	certName, ok := certificateAlgo(c.Key.Type())
	if !ok {
		panic("unknown certificate type for key type " + c.Key.Type())
	}
	return certName
}

// Verify verifies a signature against the certificate's public
// key. It is part of the PublicKey interface.
func (c *Certificate) Verify(data []byte, sig *Signature) error {
	return c.Key.Verify(data, sig)
}

func parseSignatureBody(in []byte) (out *Signature, rest []byte, ok bool) {
	format, in, ok := parseString(in)
	if !ok {
		return
	}

	out = &Signature{
		Format: string(format),
	}

	if out.Blob, in, ok = parseString(in); !ok {
		return
	}

	switch out.Format {
	case KeyAlgoSKECDSA256, CertAlgoSKECDSA256v01, KeyAlgoSKED25519, CertAlgoSKED25519v01:
		out.Rest = in
		return out, nil, ok
	}

	return out, in, ok
}

func parseSignature(in []byte) (out *Signature, rest []byte, ok bool) {
	sigBytes, rest, ok := parseString(in)
	if !ok {
		return
	}

	out, trailing, ok := parseSignatureBody(sigBytes)
	if !ok || len(trailing) > 0 {
		return nil, nil, false
	}
	return
}
