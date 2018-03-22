// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"bytes"
	"crypto"
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/md5"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/asn1"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strings"

	"golang.org/x/crypto/ed25519"
)

// These constants represent the algorithm names for key types supported by this
// package.
const (
	KeyAlgoRSA      = "ssh-rsa"
	KeyAlgoDSA      = "ssh-dss"
	KeyAlgoECDSA256 = "ecdsa-sha2-nistp256"
	KeyAlgoECDSA384 = "ecdsa-sha2-nistp384"
	KeyAlgoECDSA521 = "ecdsa-sha2-nistp521"
	KeyAlgoED25519  = "ssh-ed25519"
)

// parsePubKey parses a public key of the given algorithm.
// Use ParsePublicKey for keys with prepended algorithm.
func parsePubKey(in []byte, algo string) (pubKey PublicKey, rest []byte, err error) {
	switch algo {
	case KeyAlgoRSA:
		return parseRSA(in)
	case KeyAlgoDSA:
		return parseDSA(in)
	case KeyAlgoECDSA256, KeyAlgoECDSA384, KeyAlgoECDSA521:
		return parseECDSA(in)
	case KeyAlgoED25519:
		return parseED25519(in)
	case CertAlgoRSAv01, CertAlgoDSAv01, CertAlgoECDSA256v01, CertAlgoECDSA384v01, CertAlgoECDSA521v01, CertAlgoED25519v01:
		cert, err := parseCert(in, certToPrivAlgo(algo))
		if err != nil {
			return nil, nil, err
		}
		return cert, nil, nil
	}
	return nil, nil, fmt.Errorf("ssh: unknown key algorithm: %v", algo)
}

// parseAuthorizedKey parses a public key in OpenSSH authorized_keys format
// (see sshd(8) manual page) once the options and key type fields have been
// removed.
func parseAuthorizedKey(in []byte) (out PublicKey, comment string, err error) {
	in = bytes.TrimSpace(in)

	i := bytes.IndexAny(in, " \t")
	if i == -1 {
		i = len(in)
	}
	base64Key := in[:i]

	key := make([]byte, base64.StdEncoding.DecodedLen(len(base64Key)))
	n, err := base64.StdEncoding.Decode(key, base64Key)
	if err != nil {
		return nil, "", err
	}
	key = key[:n]
	out, err = ParsePublicKey(key)
	if err != nil {
		return nil, "", err
	}
	comment = string(bytes.TrimSpace(in[i:]))
	return out, comment, nil
}

// ParseKnownHosts parses an entry in the format of the known_hosts file.
//
// The known_hosts format is documented in the sshd(8) manual page. This
// function will parse a single entry from in. On successful return, marker
// will contain the optional marker value (i.e. "cert-authority" or "revoked")
// or else be empty, hosts will contain the hosts that this entry matches,
// pubKey will contain the public key and comment will contain any trailing
// comment at the end of the line. See the sshd(8) manual page for the various
// forms that a host string can take.
//
// The unparsed remainder of the input will be returned in rest. This function
// can be called repeatedly to parse multiple entries.
//
// If no entries were found in the input then err will be io.EOF. Otherwise a
// non-nil err value indicates a parse error.
func ParseKnownHosts(in []byte) (marker string, hosts []string, pubKey PublicKey, comment string, rest []byte, err error) {
	for len(in) > 0 {
		end := bytes.IndexByte(in, '\n')
		if end != -1 {
			rest = in[end+1:]
			in = in[:end]
		} else {
			rest = nil
		}

		end = bytes.IndexByte(in, '\r')
		if end != -1 {
			in = in[:end]
		}

		in = bytes.TrimSpace(in)
		if len(in) == 0 || in[0] == '#' {
			in = rest
			continue
		}

		i := bytes.IndexAny(in, " \t")
		if i == -1 {
			in = rest
			continue
		}

		// Strip out the beginning of the known_host key.
		// This is either an optional marker or a (set of) hostname(s).
		keyFields := bytes.Fields(in)
		if len(keyFields) < 3 || len(keyFields) > 5 {
			return "", nil, nil, "", nil, errors.New("ssh: invalid entry in known_hosts data")
		}

		// keyFields[0] is either "@cert-authority", "@revoked" or a comma separated
		// list of hosts
		marker := ""
		if keyFields[0][0] == '@' {
			marker = string(keyFields[0][1:])
			keyFields = keyFields[1:]
		}

		hosts := string(keyFields[0])
		// keyFields[1] contains the key type (e.g. “ssh-rsa”).
		// However, that information is duplicated inside the
		// base64-encoded key and so is ignored here.

		key := bytes.Join(keyFields[2:], []byte(" "))
		if pubKey, comment, err = parseAuthorizedKey(key); err != nil {
			return "", nil, nil, "", nil, err
		}

		return marker, strings.Split(hosts, ","), pubKey, comment, rest, nil
	}

	return "", nil, nil, "", nil, io.EOF
}

// ParseAuthorizedKeys parses a public key from an authorized_keys
// file used in OpenSSH according to the sshd(8) manual page.
func ParseAuthorizedKey(in []byte) (out PublicKey, comment string, options []string, rest []byte, err error) {
	for len(in) > 0 {
		end := bytes.IndexByte(in, '\n')
		if end != -1 {
			rest = in[end+1:]
			in = in[:end]
		} else {
			rest = nil
		}

		end = bytes.IndexByte(in, '\r')
		if end != -1 {
			in = in[:end]
		}

		in = bytes.TrimSpace(in)
		if len(in) == 0 || in[0] == '#' {
			in = rest
			continue
		}

		i := bytes.IndexAny(in, " \t")
		if i == -1 {
			in = rest
			continue
		}

		if out, comment, err = parseAuthorizedKey(in[i:]); err == nil {
			return out, comment, options, rest, nil
		}

		// No key type recognised. Maybe there's an options field at
		// the beginning.
		var b byte
		inQuote := false
		var candidateOptions []string
		optionStart := 0
		for i, b = range in {
			isEnd := !inQuote && (b == ' ' || b == '\t')
			if (b == ',' && !inQuote) || isEnd {
				if i-optionStart > 0 {
					candidateOptions = append(candidateOptions, string(in[optionStart:i]))
				}
				optionStart = i + 1
			}
			if isEnd {
				break
			}
			if b == '"' && (i == 0 || (i > 0 && in[i-1] != '\\')) {
				inQuote = !inQuote
			}
		}
		for i < len(in) && (in[i] == ' ' || in[i] == '\t') {
			i++
		}
		if i == len(in) {
			// Invalid line: unmatched quote
			in = rest
			continue
		}

		in = in[i:]
		i = bytes.IndexAny(in, " \t")
		if i == -1 {
			in = rest
			continue
		}

		if out, comment, err = parseAuthorizedKey(in[i:]); err == nil {
			options = candidateOptions
			return out, comment, options, rest, nil
		}

		in = rest
		continue
	}

	return nil, "", nil, nil, errors.New("ssh: no key found")
}

// ParsePublicKey parses an SSH public key formatted for use in
// the SSH wire protocol according to RFC 4253, section 6.6.
func ParsePublicKey(in []byte) (out PublicKey, err error) {
	algo, in, ok := parseString(in)
	if !ok {
		return nil, errShortRead
	}
	var rest []byte
	out, rest, err = parsePubKey(in, string(algo))
	if len(rest) > 0 {
		return nil, errors.New("ssh: trailing junk in public key")
	}

	return out, err
}

// MarshalAuthorizedKey serializes key for inclusion in an OpenSSH
// authorized_keys file. The return value ends with newline.
func MarshalAuthorizedKey(key PublicKey) []byte {
	b := &bytes.Buffer{}
	b.WriteString(key.Type())
	b.WriteByte(' ')
	e := base64.NewEncoder(base64.StdEncoding, b)
	e.Write(key.Marshal())
	e.Close()
	b.WriteByte('\n')
	return b.Bytes()
}

// PublicKey is an abstraction of different types of public keys.
type PublicKey interface {
	// Type returns the key's type, e.g. "ssh-rsa".
	Type() string

	// Marshal returns the serialized key data in SSH wire format,
	// with the name prefix.
	Marshal() []byte

	// Verify that sig is a signature on the given data using this
	// key. This function will hash the data appropriately first.
	Verify(data []byte, sig *Signature) error
}

// CryptoPublicKey, if implemented by a PublicKey,
// returns the underlying crypto.PublicKey form of the key.
type CryptoPublicKey interface {
	CryptoPublicKey() crypto.PublicKey
}

// A Signer can create signatures that verify against a public key.
type Signer interface {
	// PublicKey returns an associated PublicKey instance.
	PublicKey() PublicKey

	// Sign returns raw signature for the given data. This method
	// will apply the hash specified for the keytype to the data.
	Sign(rand io.Reader, data []byte) (*Signature, error)
}

type rsaPublicKey rsa.PublicKey

func (r *rsaPublicKey) Type() string {
	return "ssh-rsa"
}

// parseRSA parses an RSA key according to RFC 4253, section 6.6.
func parseRSA(in []byte) (out PublicKey, rest []byte, err error) {
	var w struct {
		E    *big.Int
		N    *big.Int
		Rest []byte `ssh:"rest"`
	}
	if err := Unmarshal(in, &w); err != nil {
		return nil, nil, err
	}

	if w.E.BitLen() > 24 {
		return nil, nil, errors.New("ssh: exponent too large")
	}
	e := w.E.Int64()
	if e < 3 || e&1 == 0 {
		return nil, nil, errors.New("ssh: incorrect exponent")
	}

	var key rsa.PublicKey
	key.E = int(e)
	key.N = w.N
	return (*rsaPublicKey)(&key), w.Rest, nil
}

func (r *rsaPublicKey) Marshal() []byte {
	e := new(big.Int).SetInt64(int64(r.E))
	// RSA publickey struct layout should match the struct used by
	// parseRSACert in the x/crypto/ssh/agent package.
	wirekey := struct {
		Name string
		E    *big.Int
		N    *big.Int
	}{
		KeyAlgoRSA,
		e,
		r.N,
	}
	return Marshal(&wirekey)
}

func (r *rsaPublicKey) Verify(data []byte, sig *Signature) error {
	if sig.Format != r.Type() {
		return fmt.Errorf("ssh: signature type %s for key type %s", sig.Format, r.Type())
	}
	h := crypto.SHA1.New()
	h.Write(data)
	digest := h.Sum(nil)
	return rsa.VerifyPKCS1v15((*rsa.PublicKey)(r), crypto.SHA1, digest, sig.Blob)
}

func (r *rsaPublicKey) CryptoPublicKey() crypto.PublicKey {
	return (*rsa.PublicKey)(r)
}

type dsaPublicKey dsa.PublicKey

func (k *dsaPublicKey) Type() string {
	return "ssh-dss"
}

func checkDSAParams(param *dsa.Parameters) error {
	// SSH specifies FIPS 186-2, which only provided a single size
	// (1024 bits) DSA key. FIPS 186-3 allows for larger key
	// sizes, which would confuse SSH.
	if l := param.P.BitLen(); l != 1024 {
		return fmt.Errorf("ssh: unsupported DSA key size %d", l)
	}

	return nil
}

// parseDSA parses an DSA key according to RFC 4253, section 6.6.
func parseDSA(in []byte) (out PublicKey, rest []byte, err error) {
	var w struct {
		P, Q, G, Y *big.Int
		Rest       []byte `ssh:"rest"`
	}
	if err := Unmarshal(in, &w); err != nil {
		return nil, nil, err
	}

	param := dsa.Parameters{
		P: w.P,
		Q: w.Q,
		G: w.G,
	}
	if err := checkDSAParams(&param); err != nil {
		return nil, nil, err
	}

	key := &dsaPublicKey{
		Parameters: param,
		Y:          w.Y,
	}
	return key, w.Rest, nil
}

func (k *dsaPublicKey) Marshal() []byte {
	// DSA publickey struct layout should match the struct used by
	// parseDSACert in the x/crypto/ssh/agent package.
	w := struct {
		Name       string
		P, Q, G, Y *big.Int
	}{
		k.Type(),
		k.P,
		k.Q,
		k.G,
		k.Y,
	}

	return Marshal(&w)
}

func (k *dsaPublicKey) Verify(data []byte, sig *Signature) error {
	if sig.Format != k.Type() {
		return fmt.Errorf("ssh: signature type %s for key type %s", sig.Format, k.Type())
	}
	h := crypto.SHA1.New()
	h.Write(data)
	digest := h.Sum(nil)

	// Per RFC 4253, section 6.6,
	// The value for 'dss_signature_blob' is encoded as a string containing
	// r, followed by s (which are 160-bit integers, without lengths or
	// padding, unsigned, and in network byte order).
	// For DSS purposes, sig.Blob should be exactly 40 bytes in length.
	if len(sig.Blob) != 40 {
		return errors.New("ssh: DSA signature parse error")
	}
	r := new(big.Int).SetBytes(sig.Blob[:20])
	s := new(big.Int).SetBytes(sig.Blob[20:])
	if dsa.Verify((*dsa.PublicKey)(k), digest, r, s) {
		return nil
	}
	return errors.New("ssh: signature did not verify")
}

func (k *dsaPublicKey) CryptoPublicKey() crypto.PublicKey {
	return (*dsa.PublicKey)(k)
}

type dsaPrivateKey struct {
	*dsa.PrivateKey
}

func (k *dsaPrivateKey) PublicKey() PublicKey {
	return (*dsaPublicKey)(&k.PrivateKey.PublicKey)
}

func (k *dsaPrivateKey) Sign(rand io.Reader, data []byte) (*Signature, error) {
	h := crypto.SHA1.New()
	h.Write(data)
	digest := h.Sum(nil)
	r, s, err := dsa.Sign(rand, k.PrivateKey, digest)
	if err != nil {
		return nil, err
	}

	sig := make([]byte, 40)
	rb := r.Bytes()
	sb := s.Bytes()

	copy(sig[20-len(rb):20], rb)
	copy(sig[40-len(sb):], sb)

	return &Signature{
		Format: k.PublicKey().Type(),
		Blob:   sig,
	}, nil
}

type ecdsaPublicKey ecdsa.PublicKey

func (k *ecdsaPublicKey) Type() string {
	return "ecdsa-sha2-" + k.nistID()
}

func (k *ecdsaPublicKey) nistID() string {
	switch k.Params().BitSize {
	case 256:
		return "nistp256"
	case 384:
		return "nistp384"
	case 521:
		return "nistp521"
	}
	panic("ssh: unsupported ecdsa key size")
}

type ed25519PublicKey ed25519.PublicKey

func (k ed25519PublicKey) Type() string {
	return KeyAlgoED25519
}

func parseED25519(in []byte) (out PublicKey, rest []byte, err error) {
	var w struct {
		KeyBytes []byte
		Rest     []byte `ssh:"rest"`
	}

	if err := Unmarshal(in, &w); err != nil {
		return nil, nil, err
	}

	key := ed25519.PublicKey(w.KeyBytes)

	return (ed25519PublicKey)(key), w.Rest, nil
}

func (k ed25519PublicKey) Marshal() []byte {
	w := struct {
		Name     string
		KeyBytes []byte
	}{
		KeyAlgoED25519,
		[]byte(k),
	}
	return Marshal(&w)
}

func (k ed25519PublicKey) Verify(b []byte, sig *Signature) error {
	if sig.Format != k.Type() {
		return fmt.Errorf("ssh: signature type %s for key type %s", sig.Format, k.Type())
	}

	edKey := (ed25519.PublicKey)(k)
	if ok := ed25519.Verify(edKey, b, sig.Blob); !ok {
		return errors.New("ssh: signature did not verify")
	}

	return nil
}

func (k ed25519PublicKey) CryptoPublicKey() crypto.PublicKey {
	return ed25519.PublicKey(k)
}

func supportedEllipticCurve(curve elliptic.Curve) bool {
	return curve == elliptic.P256() || curve == elliptic.P384() || curve == elliptic.P521()
}

// ecHash returns the hash to match the given elliptic curve, see RFC
// 5656, section 6.2.1
func ecHash(curve elliptic.Curve) crypto.Hash {
	bitSize := curve.Params().BitSize
	switch {
	case bitSize <= 256:
		return crypto.SHA256
	case bitSize <= 384:
		return crypto.SHA384
	}
	return crypto.SHA512
}

// parseECDSA parses an ECDSA key according to RFC 5656, section 3.1.
func parseECDSA(in []byte) (out PublicKey, rest []byte, err error) {
	var w struct {
		Curve    string
		KeyBytes []byte
		Rest     []byte `ssh:"rest"`
	}

	if err := Unmarshal(in, &w); err != nil {
		return nil, nil, err
	}

	key := new(ecdsa.PublicKey)

	switch w.Curve {
	case "nistp256":
		key.Curve = elliptic.P256()
	case "nistp384":
		key.Curve = elliptic.P384()
	case "nistp521":
		key.Curve = elliptic.P521()
	default:
		return nil, nil, errors.New("ssh: unsupported curve")
	}

	key.X, key.Y = elliptic.Unmarshal(key.Curve, w.KeyBytes)
	if key.X == nil || key.Y == nil {
		return nil, nil, errors.New("ssh: invalid curve point")
	}
	return (*ecdsaPublicKey)(key), w.Rest, nil
}

func (k *ecdsaPublicKey) Marshal() []byte {
	// See RFC 5656, section 3.1.
	keyBytes := elliptic.Marshal(k.Curve, k.X, k.Y)
	// ECDSA publickey struct layout should match the struct used by
	// parseECDSACert in the x/crypto/ssh/agent package.
	w := struct {
		Name string
		ID   string
		Key  []byte
	}{
		k.Type(),
		k.nistID(),
		keyBytes,
	}

	return Marshal(&w)
}

func (k *ecdsaPublicKey) Verify(data []byte, sig *Signature) error {
	if sig.Format != k.Type() {
		return fmt.Errorf("ssh: signature type %s for key type %s", sig.Format, k.Type())
	}

	h := ecHash(k.Curve).New()
	h.Write(data)
	digest := h.Sum(nil)

	// Per RFC 5656, section 3.1.2,
	// The ecdsa_signature_blob value has the following specific encoding:
	//    mpint    r
	//    mpint    s
	var ecSig struct {
		R *big.Int
		S *big.Int
	}

	if err := Unmarshal(sig.Blob, &ecSig); err != nil {
		return err
	}

	if ecdsa.Verify((*ecdsa.PublicKey)(k), digest, ecSig.R, ecSig.S) {
		return nil
	}
	return errors.New("ssh: signature did not verify")
}

func (k *ecdsaPublicKey) CryptoPublicKey() crypto.PublicKey {
	return (*ecdsa.PublicKey)(k)
}

// NewSignerFromKey takes an *rsa.PrivateKey, *dsa.PrivateKey,
// *ecdsa.PrivateKey or any other crypto.Signer and returns a
// corresponding Signer instance. ECDSA keys must use P-256, P-384 or
// P-521. DSA keys must use parameter size L1024N160.
func NewSignerFromKey(key interface{}) (Signer, error) {
	switch key := key.(type) {
	case crypto.Signer:
		return NewSignerFromSigner(key)
	case *dsa.PrivateKey:
		return newDSAPrivateKey(key)
	default:
		return nil, fmt.Errorf("ssh: unsupported key type %T", key)
	}
}

func newDSAPrivateKey(key *dsa.PrivateKey) (Signer, error) {
	if err := checkDSAParams(&key.PublicKey.Parameters); err != nil {
		return nil, err
	}

	return &dsaPrivateKey{key}, nil
}

type wrappedSigner struct {
	signer crypto.Signer
	pubKey PublicKey
}

// NewSignerFromSigner takes any crypto.Signer implementation and
// returns a corresponding Signer interface. This can be used, for
// example, with keys kept in hardware modules.
func NewSignerFromSigner(signer crypto.Signer) (Signer, error) {
	pubKey, err := NewPublicKey(signer.Public())
	if err != nil {
		return nil, err
	}

	return &wrappedSigner{signer, pubKey}, nil
}

func (s *wrappedSigner) PublicKey() PublicKey {
	return s.pubKey
}

func (s *wrappedSigner) Sign(rand io.Reader, data []byte) (*Signature, error) {
	var hashFunc crypto.Hash

	switch key := s.pubKey.(type) {
	case *rsaPublicKey, *dsaPublicKey:
		hashFunc = crypto.SHA1
	case *ecdsaPublicKey:
		hashFunc = ecHash(key.Curve)
	case ed25519PublicKey:
	default:
		return nil, fmt.Errorf("ssh: unsupported key type %T", key)
	}

	var digest []byte
	if hashFunc != 0 {
		h := hashFunc.New()
		h.Write(data)
		digest = h.Sum(nil)
	} else {
		digest = data
	}

	signature, err := s.signer.Sign(rand, digest, hashFunc)
	if err != nil {
		return nil, err
	}

	// crypto.Signer.Sign is expected to return an ASN.1-encoded signature
	// for ECDSA and DSA, but that's not the encoding expected by SSH, so
	// re-encode.
	switch s.pubKey.(type) {
	case *ecdsaPublicKey, *dsaPublicKey:
		type asn1Signature struct {
			R, S *big.Int
		}
		asn1Sig := new(asn1Signature)
		_, err := asn1.Unmarshal(signature, asn1Sig)
		if err != nil {
			return nil, err
		}

		switch s.pubKey.(type) {
		case *ecdsaPublicKey:
			signature = Marshal(asn1Sig)

		case *dsaPublicKey:
			signature = make([]byte, 40)
			r := asn1Sig.R.Bytes()
			s := asn1Sig.S.Bytes()
			copy(signature[20-len(r):20], r)
			copy(signature[40-len(s):40], s)
		}
	}

	return &Signature{
		Format: s.pubKey.Type(),
		Blob:   signature,
	}, nil
}

// NewPublicKey takes an *rsa.PublicKey, *dsa.PublicKey, *ecdsa.PublicKey,
// or ed25519.PublicKey returns a corresponding PublicKey instance.
// ECDSA keys must use P-256, P-384 or P-521.
func NewPublicKey(key interface{}) (PublicKey, error) {
	switch key := key.(type) {
	case *rsa.PublicKey:
		return (*rsaPublicKey)(key), nil
	case *ecdsa.PublicKey:
		if !supportedEllipticCurve(key.Curve) {
			return nil, errors.New("ssh: only P-256, P-384 and P-521 EC keys are supported")
		}
		return (*ecdsaPublicKey)(key), nil
	case *dsa.PublicKey:
		return (*dsaPublicKey)(key), nil
	case ed25519.PublicKey:
		return (ed25519PublicKey)(key), nil
	default:
		return nil, fmt.Errorf("ssh: unsupported key type %T", key)
	}
}

// ParsePrivateKey returns a Signer from a PEM encoded private key. It supports
// the same keys as ParseRawPrivateKey.
func ParsePrivateKey(pemBytes []byte) (Signer, error) {
	key, err := ParseRawPrivateKey(pemBytes)
	if err != nil {
		return nil, err
	}

	return NewSignerFromKey(key)
}

// ParsePrivateKeyWithPassphrase returns a Signer from a PEM encoded private
// key and passphrase. It supports the same keys as
// ParseRawPrivateKeyWithPassphrase.
func ParsePrivateKeyWithPassphrase(pemBytes, passPhrase []byte) (Signer, error) {
	key, err := ParseRawPrivateKeyWithPassphrase(pemBytes, passPhrase)
	if err != nil {
		return nil, err
	}

	return NewSignerFromKey(key)
}

// encryptedBlock tells whether a private key is
// encrypted by examining its Proc-Type header
// for a mention of ENCRYPTED
// according to RFC 1421 Section 4.6.1.1.
func encryptedBlock(block *pem.Block) bool {
	return strings.Contains(block.Headers["Proc-Type"], "ENCRYPTED")
}

// ParseRawPrivateKey returns a private key from a PEM encoded private key. It
// supports RSA (PKCS#1), DSA (OpenSSL), and ECDSA private keys.
func ParseRawPrivateKey(pemBytes []byte) (interface{}, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, errors.New("ssh: no key found")
	}

	if encryptedBlock(block) {
		return nil, errors.New("ssh: cannot decode encrypted private keys")
	}

	switch block.Type {
	case "RSA PRIVATE KEY":
		return x509.ParsePKCS1PrivateKey(block.Bytes)
	case "EC PRIVATE KEY":
		return x509.ParseECPrivateKey(block.Bytes)
	case "DSA PRIVATE KEY":
		return ParseDSAPrivateKey(block.Bytes)
	case "OPENSSH PRIVATE KEY":
		return parseOpenSSHPrivateKey(block.Bytes)
	default:
		return nil, fmt.Errorf("ssh: unsupported key type %q", block.Type)
	}
}

// ParseRawPrivateKeyWithPassphrase returns a private key decrypted with
// passphrase from a PEM encoded private key. If wrong passphrase, return
// x509.IncorrectPasswordError.
func ParseRawPrivateKeyWithPassphrase(pemBytes, passPhrase []byte) (interface{}, error) {
	block, _ := pem.Decode(pemBytes)
	if block == nil {
		return nil, errors.New("ssh: no key found")
	}
	buf := block.Bytes

	if encryptedBlock(block) {
		if x509.IsEncryptedPEMBlock(block) {
			var err error
			buf, err = x509.DecryptPEMBlock(block, passPhrase)
			if err != nil {
				if err == x509.IncorrectPasswordError {
					return nil, err
				}
				return nil, fmt.Errorf("ssh: cannot decode encrypted private keys: %v", err)
			}
		}
	}

	switch block.Type {
	case "RSA PRIVATE KEY":
		return x509.ParsePKCS1PrivateKey(buf)
	case "EC PRIVATE KEY":
		return x509.ParseECPrivateKey(buf)
	case "DSA PRIVATE KEY":
		return ParseDSAPrivateKey(buf)
	case "OPENSSH PRIVATE KEY":
		return parseOpenSSHPrivateKey(buf)
	default:
		return nil, fmt.Errorf("ssh: unsupported key type %q", block.Type)
	}
}

// ParseDSAPrivateKey returns a DSA private key from its ASN.1 DER encoding, as
// specified by the OpenSSL DSA man page.
func ParseDSAPrivateKey(der []byte) (*dsa.PrivateKey, error) {
	var k struct {
		Version int
		P       *big.Int
		Q       *big.Int
		G       *big.Int
		Pub     *big.Int
		Priv    *big.Int
	}
	rest, err := asn1.Unmarshal(der, &k)
	if err != nil {
		return nil, errors.New("ssh: failed to parse DSA key: " + err.Error())
	}
	if len(rest) > 0 {
		return nil, errors.New("ssh: garbage after DSA key")
	}

	return &dsa.PrivateKey{
		PublicKey: dsa.PublicKey{
			Parameters: dsa.Parameters{
				P: k.P,
				Q: k.Q,
				G: k.G,
			},
			Y: k.Pub,
		},
		X: k.Priv,
	}, nil
}

// Implemented based on the documentation at
// https://github.com/openssh/openssh-portable/blob/master/PROTOCOL.key
func parseOpenSSHPrivateKey(key []byte) (crypto.PrivateKey, error) {
	magic := append([]byte("openssh-key-v1"), 0)
	if !bytes.Equal(magic, key[0:len(magic)]) {
		return nil, errors.New("ssh: invalid openssh private key format")
	}
	remaining := key[len(magic):]

	var w struct {
		CipherName   string
		KdfName      string
		KdfOpts      string
		NumKeys      uint32
		PubKey       []byte
		PrivKeyBlock []byte
	}

	if err := Unmarshal(remaining, &w); err != nil {
		return nil, err
	}

	if w.KdfName != "none" || w.CipherName != "none" {
		return nil, errors.New("ssh: cannot decode encrypted private keys")
	}

	pk1 := struct {
		Check1  uint32
		Check2  uint32
		Keytype string
		Rest    []byte `ssh:"rest"`
	}{}

	if err := Unmarshal(w.PrivKeyBlock, &pk1); err != nil {
		return nil, err
	}

	if pk1.Check1 != pk1.Check2 {
		return nil, errors.New("ssh: checkint mismatch")
	}

	// we only handle ed25519 and rsa keys currently
	switch pk1.Keytype {
	case KeyAlgoRSA:
		// https://github.com/openssh/openssh-portable/blob/master/sshkey.c#L2760-L2773
		key := struct {
			N       *big.Int
			E       *big.Int
			D       *big.Int
			Iqmp    *big.Int
			P       *big.Int
			Q       *big.Int
			Comment string
			Pad     []byte `ssh:"rest"`
		}{}

		if err := Unmarshal(pk1.Rest, &key); err != nil {
			return nil, err
		}

		for i, b := range key.Pad {
			if int(b) != i+1 {
				return nil, errors.New("ssh: padding not as expected")
			}
		}

		pk := &rsa.PrivateKey{
			PublicKey: rsa.PublicKey{
				N: key.N,
				E: int(key.E.Int64()),
			},
			D:      key.D,
			Primes: []*big.Int{key.P, key.Q},
		}

		if err := pk.Validate(); err != nil {
			return nil, err
		}

		pk.Precompute()

		return pk, nil
	case KeyAlgoED25519:
		key := struct {
			Pub     []byte
			Priv    []byte
			Comment string
			Pad     []byte `ssh:"rest"`
		}{}

		if err := Unmarshal(pk1.Rest, &key); err != nil {
			return nil, err
		}

		if len(key.Priv) != ed25519.PrivateKeySize {
			return nil, errors.New("ssh: private key unexpected length")
		}

		for i, b := range key.Pad {
			if int(b) != i+1 {
				return nil, errors.New("ssh: padding not as expected")
			}
		}

		pk := ed25519.PrivateKey(make([]byte, ed25519.PrivateKeySize))
		copy(pk, key.Priv)
		return &pk, nil
	default:
		return nil, errors.New("ssh: unhandled key type")
	}
}

// FingerprintLegacyMD5 returns the user presentation of the key's
// fingerprint as described by RFC 4716 section 4.
func FingerprintLegacyMD5(pubKey PublicKey) string {
	md5sum := md5.Sum(pubKey.Marshal())
	hexarray := make([]string, len(md5sum))
	for i, c := range md5sum {
		hexarray[i] = hex.EncodeToString([]byte{c})
	}
	return strings.Join(hexarray, ":")
}

// FingerprintSHA256 returns the user presentation of the key's
// fingerprint as unpadded base64 encoded sha256 hash.
// This format was introduced from OpenSSH 6.8.
// https://www.openssh.com/txt/release-6.8
// https://tools.ietf.org/html/rfc4648#section-3.2 (unpadded base64 encoding)
func FingerprintSHA256(pubKey PublicKey) string {
	sha256sum := sha256.Sum256(pubKey.Marshal())
	hash := base64.RawStdEncoding.EncodeToString(sha256sum[:])
	return "SHA256:" + hash
}
