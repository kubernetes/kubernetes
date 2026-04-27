// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

import (
	"crypto"
	"crypto/fips140"
	"crypto/rand"
	"fmt"
	"io"
	"math"
	"slices"
	"sync"

	_ "crypto/sha1"
	_ "crypto/sha256"
	_ "crypto/sha512"
)

// These are string constants in the SSH protocol.
const (
	compressionNone = "none"
	serviceUserAuth = "ssh-userauth"
	serviceSSH      = "ssh-connection"
)

// The ciphers currently or previously implemented by this library, to use in
// [Config.Ciphers]. For a list, see the [Algorithms.Ciphers] returned by
// [SupportedAlgorithms] or [InsecureAlgorithms].
const (
	CipherAES128GCM            = "aes128-gcm@openssh.com"
	CipherAES256GCM            = "aes256-gcm@openssh.com"
	CipherChaCha20Poly1305     = "chacha20-poly1305@openssh.com"
	CipherAES128CTR            = "aes128-ctr"
	CipherAES192CTR            = "aes192-ctr"
	CipherAES256CTR            = "aes256-ctr"
	InsecureCipherAES128CBC    = "aes128-cbc"
	InsecureCipherTripleDESCBC = "3des-cbc"
	InsecureCipherRC4          = "arcfour"
	InsecureCipherRC4128       = "arcfour128"
	InsecureCipherRC4256       = "arcfour256"
)

// The key exchanges currently or previously implemented by this library, to use
// in [Config.KeyExchanges]. For a list, see the
// [Algorithms.KeyExchanges] returned by [SupportedAlgorithms] or
// [InsecureAlgorithms].
const (
	InsecureKeyExchangeDH1SHA1   = "diffie-hellman-group1-sha1"
	InsecureKeyExchangeDH14SHA1  = "diffie-hellman-group14-sha1"
	KeyExchangeDH14SHA256        = "diffie-hellman-group14-sha256"
	KeyExchangeDH16SHA512        = "diffie-hellman-group16-sha512"
	KeyExchangeECDHP256          = "ecdh-sha2-nistp256"
	KeyExchangeECDHP384          = "ecdh-sha2-nistp384"
	KeyExchangeECDHP521          = "ecdh-sha2-nistp521"
	KeyExchangeCurve25519        = "curve25519-sha256"
	InsecureKeyExchangeDHGEXSHA1 = "diffie-hellman-group-exchange-sha1"
	KeyExchangeDHGEXSHA256       = "diffie-hellman-group-exchange-sha256"
	// KeyExchangeMLKEM768X25519 is supported from Go 1.24.
	KeyExchangeMLKEM768X25519 = "mlkem768x25519-sha256"

	// An alias for KeyExchangeCurve25519SHA256. This kex ID will be added if
	// KeyExchangeCurve25519SHA256 is requested for backward compatibility with
	// OpenSSH versions up to 7.2.
	keyExchangeCurve25519LibSSH = "curve25519-sha256@libssh.org"
)

// The message authentication code (MAC) currently or previously implemented by
// this library, to use in [Config.MACs]. For a list, see the
// [Algorithms.MACs] returned by [SupportedAlgorithms] or
// [InsecureAlgorithms].
const (
	HMACSHA256ETM      = "hmac-sha2-256-etm@openssh.com"
	HMACSHA512ETM      = "hmac-sha2-512-etm@openssh.com"
	HMACSHA256         = "hmac-sha2-256"
	HMACSHA512         = "hmac-sha2-512"
	HMACSHA1           = "hmac-sha1"
	InsecureHMACSHA196 = "hmac-sha1-96"
)

var (
	// supportedKexAlgos specifies key-exchange algorithms implemented by this
	// package in preference order, excluding those with security issues.
	supportedKexAlgos = []string{
		KeyExchangeMLKEM768X25519,
		KeyExchangeCurve25519,
		KeyExchangeECDHP256,
		KeyExchangeECDHP384,
		KeyExchangeECDHP521,
		KeyExchangeDH14SHA256,
		KeyExchangeDH16SHA512,
		KeyExchangeDHGEXSHA256,
	}
	// defaultKexAlgos specifies the default preference for key-exchange
	// algorithms in preference order.
	defaultKexAlgos = []string{
		KeyExchangeMLKEM768X25519,
		KeyExchangeCurve25519,
		KeyExchangeECDHP256,
		KeyExchangeECDHP384,
		KeyExchangeECDHP521,
		KeyExchangeDH14SHA256,
		InsecureKeyExchangeDH14SHA1,
	}
	// insecureKexAlgos specifies key-exchange algorithms implemented by this
	// package and which have security issues.
	insecureKexAlgos = []string{
		InsecureKeyExchangeDH14SHA1,
		InsecureKeyExchangeDH1SHA1,
		InsecureKeyExchangeDHGEXSHA1,
	}
	// supportedCiphers specifies cipher algorithms implemented by this package
	// in preference order, excluding those with security issues.
	supportedCiphers = []string{
		CipherAES128GCM,
		CipherAES256GCM,
		CipherChaCha20Poly1305,
		CipherAES128CTR,
		CipherAES192CTR,
		CipherAES256CTR,
	}
	// defaultCiphers specifies the default preference for ciphers algorithms
	// in preference order.
	defaultCiphers = supportedCiphers
	// insecureCiphers specifies cipher algorithms implemented by this
	// package and which have security issues.
	insecureCiphers = []string{
		InsecureCipherAES128CBC,
		InsecureCipherTripleDESCBC,
		InsecureCipherRC4256,
		InsecureCipherRC4128,
		InsecureCipherRC4,
	}
	// supportedMACs specifies MAC algorithms implemented by this package in
	// preference order, excluding those with security issues.
	supportedMACs = []string{
		HMACSHA256ETM,
		HMACSHA512ETM,
		HMACSHA256,
		HMACSHA512,
		HMACSHA1,
	}
	// defaultMACs specifies the default preference for MAC algorithms in
	// preference order.
	defaultMACs = []string{
		HMACSHA256ETM,
		HMACSHA512ETM,
		HMACSHA256,
		HMACSHA512,
		HMACSHA1,
		InsecureHMACSHA196,
	}
	// insecureMACs specifies MAC algorithms implemented by this
	// package and which have security issues.
	insecureMACs = []string{
		InsecureHMACSHA196,
	}
	// supportedHostKeyAlgos specifies the supported host-key algorithms (i.e.
	// methods of authenticating servers) implemented by this package in
	// preference order, excluding those with security issues.
	supportedHostKeyAlgos = []string{
		CertAlgoRSASHA256v01,
		CertAlgoRSASHA512v01,
		CertAlgoECDSA256v01,
		CertAlgoECDSA384v01,
		CertAlgoECDSA521v01,
		CertAlgoED25519v01,
		KeyAlgoRSASHA256,
		KeyAlgoRSASHA512,
		KeyAlgoECDSA256,
		KeyAlgoECDSA384,
		KeyAlgoECDSA521,
		KeyAlgoED25519,
	}
	// defaultHostKeyAlgos specifies the default preference for host-key
	// algorithms in preference order.
	defaultHostKeyAlgos = []string{
		CertAlgoRSASHA256v01,
		CertAlgoRSASHA512v01,
		CertAlgoRSAv01,
		InsecureCertAlgoDSAv01,
		CertAlgoECDSA256v01,
		CertAlgoECDSA384v01,
		CertAlgoECDSA521v01,
		CertAlgoED25519v01,
		KeyAlgoECDSA256,
		KeyAlgoECDSA384,
		KeyAlgoECDSA521,
		KeyAlgoRSASHA256,
		KeyAlgoRSASHA512,
		KeyAlgoRSA,
		InsecureKeyAlgoDSA,
		KeyAlgoED25519,
	}
	// insecureHostKeyAlgos specifies host-key algorithms implemented by this
	// package and which have security issues.
	insecureHostKeyAlgos = []string{
		KeyAlgoRSA,
		InsecureKeyAlgoDSA,
		CertAlgoRSAv01,
		InsecureCertAlgoDSAv01,
	}
	// supportedPubKeyAuthAlgos specifies the supported client public key
	// authentication algorithms. Note that this doesn't include certificate
	// types since those use the underlying algorithm. Order is irrelevant.
	supportedPubKeyAuthAlgos = []string{
		KeyAlgoED25519,
		KeyAlgoSKED25519,
		KeyAlgoSKECDSA256,
		KeyAlgoECDSA256,
		KeyAlgoECDSA384,
		KeyAlgoECDSA521,
		KeyAlgoRSASHA256,
		KeyAlgoRSASHA512,
	}

	// defaultPubKeyAuthAlgos specifies the preferred client public key
	// authentication algorithms. This list is sent to the client if it supports
	// the server-sig-algs extension. Order is irrelevant.
	defaultPubKeyAuthAlgos = []string{
		KeyAlgoED25519,
		KeyAlgoSKED25519,
		KeyAlgoSKECDSA256,
		KeyAlgoECDSA256,
		KeyAlgoECDSA384,
		KeyAlgoECDSA521,
		KeyAlgoRSASHA256,
		KeyAlgoRSASHA512,
		KeyAlgoRSA,
		InsecureKeyAlgoDSA,
	}
	// insecurePubKeyAuthAlgos specifies client public key authentication
	// algorithms implemented by this package and which have security issues.
	insecurePubKeyAuthAlgos = []string{
		KeyAlgoRSA,
		InsecureKeyAlgoDSA,
	}
)

// NegotiatedAlgorithms defines algorithms negotiated between client and server.
type NegotiatedAlgorithms struct {
	KeyExchange string
	HostKey     string
	Read        DirectionAlgorithms
	Write       DirectionAlgorithms
}

// Algorithms defines a set of algorithms that can be configured in the client
// or server config for negotiation during a handshake.
type Algorithms struct {
	KeyExchanges   []string
	Ciphers        []string
	MACs           []string
	HostKeys       []string
	PublicKeyAuths []string
}

func init() {
	if fips140.Enabled() {
		defaultHostKeyAlgos = slices.DeleteFunc(defaultHostKeyAlgos, func(algo string) bool {
			_, err := hashFunc(underlyingAlgo(algo))
			return err != nil
		})
		defaultPubKeyAuthAlgos = slices.DeleteFunc(defaultPubKeyAuthAlgos, func(algo string) bool {
			_, err := hashFunc(underlyingAlgo(algo))
			return err != nil
		})
	}
}

func hashFunc(format string) (crypto.Hash, error) {
	switch format {
	case KeyAlgoRSASHA256, KeyAlgoECDSA256, KeyAlgoSKED25519, KeyAlgoSKECDSA256:
		return crypto.SHA256, nil
	case KeyAlgoECDSA384:
		return crypto.SHA384, nil
	case KeyAlgoRSASHA512, KeyAlgoECDSA521:
		return crypto.SHA512, nil
	case KeyAlgoED25519:
		// KeyAlgoED25519 doesn't pre-hash.
		return 0, nil
	case KeyAlgoRSA, InsecureKeyAlgoDSA:
		if fips140.Enabled() {
			return 0, fmt.Errorf("ssh: hash algorithm for format %q not allowed in FIPS 140 mode", format)
		}
		return crypto.SHA1, nil
	default:
		return 0, fmt.Errorf("ssh: hash algorithm for format %q not mapped", format)
	}
}

// SupportedAlgorithms returns algorithms currently implemented by this package,
// excluding those with security issues, which are returned by
// InsecureAlgorithms. The algorithms listed here are in preference order.
func SupportedAlgorithms() Algorithms {
	return Algorithms{
		Ciphers:        slices.Clone(supportedCiphers),
		MACs:           slices.Clone(supportedMACs),
		KeyExchanges:   slices.Clone(supportedKexAlgos),
		HostKeys:       slices.Clone(supportedHostKeyAlgos),
		PublicKeyAuths: slices.Clone(supportedPubKeyAuthAlgos),
	}
}

// InsecureAlgorithms returns algorithms currently implemented by this package
// and which have security issues.
func InsecureAlgorithms() Algorithms {
	return Algorithms{
		KeyExchanges:   slices.Clone(insecureKexAlgos),
		Ciphers:        slices.Clone(insecureCiphers),
		MACs:           slices.Clone(insecureMACs),
		HostKeys:       slices.Clone(insecureHostKeyAlgos),
		PublicKeyAuths: slices.Clone(insecurePubKeyAuthAlgos),
	}
}

var supportedCompressions = []string{compressionNone}

// algorithmsForKeyFormat returns the supported signature algorithms for a given
// public key format (PublicKey.Type), in order of preference. See RFC 8332,
// Section 2. See also the note in sendKexInit on backwards compatibility.
func algorithmsForKeyFormat(keyFormat string) []string {
	switch keyFormat {
	case KeyAlgoRSA:
		return []string{KeyAlgoRSASHA256, KeyAlgoRSASHA512, KeyAlgoRSA}
	case CertAlgoRSAv01:
		return []string{CertAlgoRSASHA256v01, CertAlgoRSASHA512v01, CertAlgoRSAv01}
	default:
		return []string{keyFormat}
	}
}

// keyFormatForAlgorithm returns the key format corresponding to the given
// signature algorithm. It returns an empty string if the signature algorithm is
// invalid or unsupported.
func keyFormatForAlgorithm(sigAlgo string) string {
	switch sigAlgo {
	case KeyAlgoRSA, KeyAlgoRSASHA256, KeyAlgoRSASHA512:
		return KeyAlgoRSA
	case CertAlgoRSAv01, CertAlgoRSASHA256v01, CertAlgoRSASHA512v01:
		return CertAlgoRSAv01
	case KeyAlgoED25519,
		KeyAlgoSKED25519,
		KeyAlgoSKECDSA256,
		KeyAlgoECDSA256,
		KeyAlgoECDSA384,
		KeyAlgoECDSA521,
		InsecureKeyAlgoDSA,
		InsecureCertAlgoDSAv01,
		CertAlgoECDSA256v01,
		CertAlgoECDSA384v01,
		CertAlgoECDSA521v01,
		CertAlgoSKECDSA256v01,
		CertAlgoED25519v01,
		CertAlgoSKED25519v01:
		return sigAlgo
	default:
		return ""
	}
}

// isRSA returns whether algo is a supported RSA algorithm, including certificate
// algorithms.
func isRSA(algo string) bool {
	algos := algorithmsForKeyFormat(KeyAlgoRSA)
	return slices.Contains(algos, underlyingAlgo(algo))
}

func isRSACert(algo string) bool {
	_, ok := certKeyAlgoNames[algo]
	if !ok {
		return false
	}
	return isRSA(algo)
}

// unexpectedMessageError results when the SSH message that we received didn't
// match what we wanted.
func unexpectedMessageError(expected, got uint8) error {
	return fmt.Errorf("ssh: unexpected message type %d (expected %d)", got, expected)
}

// parseError results from a malformed SSH message.
func parseError(tag uint8) error {
	return fmt.Errorf("ssh: parse error in message type %d", tag)
}

func findCommon(what string, client []string, server []string, isClient bool) (string, error) {
	for _, c := range client {
		for _, s := range server {
			if c == s {
				return c, nil
			}
		}
	}
	err := &AlgorithmNegotiationError{
		What: what,
	}
	if isClient {
		err.SupportedAlgorithms = client
		err.RequestedAlgorithms = server
	} else {
		err.SupportedAlgorithms = server
		err.RequestedAlgorithms = client
	}
	return "", err
}

// AlgorithmNegotiationError defines the error returned if the client and the
// server cannot agree on an algorithm for key exchange, host key, cipher, MAC.
type AlgorithmNegotiationError struct {
	What string
	// RequestedAlgorithms lists the algorithms supported by the peer.
	RequestedAlgorithms []string
	// SupportedAlgorithms lists the algorithms supported on our side.
	SupportedAlgorithms []string
}

func (a *AlgorithmNegotiationError) Error() string {
	return fmt.Sprintf("ssh: no common algorithm for %s; we offered: %v, peer offered: %v",
		a.What, a.SupportedAlgorithms, a.RequestedAlgorithms)
}

// DirectionAlgorithms defines the algorithms negotiated in one direction
// (either read or write).
type DirectionAlgorithms struct {
	Cipher      string
	MAC         string
	compression string
}

// rekeyBytes returns a rekeying intervals in bytes.
func (a *DirectionAlgorithms) rekeyBytes() int64 {
	// According to RFC 4344 block ciphers should rekey after
	// 2^(BLOCKSIZE/4) blocks. For all AES flavors BLOCKSIZE is
	// 128.
	switch a.Cipher {
	case CipherAES128CTR, CipherAES192CTR, CipherAES256CTR, CipherAES128GCM, CipherAES256GCM, InsecureCipherAES128CBC:
		return 16 * (1 << 32)

	}

	// For others, stick with RFC 4253 recommendation to rekey after 1 Gb of data.
	return 1 << 30
}

var aeadCiphers = map[string]bool{
	CipherAES128GCM:        true,
	CipherAES256GCM:        true,
	CipherChaCha20Poly1305: true,
}

func findAgreedAlgorithms(isClient bool, clientKexInit, serverKexInit *kexInitMsg) (algs *NegotiatedAlgorithms, err error) {
	result := &NegotiatedAlgorithms{}

	result.KeyExchange, err = findCommon("key exchange", clientKexInit.KexAlgos, serverKexInit.KexAlgos, isClient)
	if err != nil {
		return
	}

	result.HostKey, err = findCommon("host key", clientKexInit.ServerHostKeyAlgos, serverKexInit.ServerHostKeyAlgos, isClient)
	if err != nil {
		return
	}

	stoc, ctos := &result.Write, &result.Read
	if isClient {
		ctos, stoc = stoc, ctos
	}

	ctos.Cipher, err = findCommon("client to server cipher", clientKexInit.CiphersClientServer, serverKexInit.CiphersClientServer, isClient)
	if err != nil {
		return
	}

	stoc.Cipher, err = findCommon("server to client cipher", clientKexInit.CiphersServerClient, serverKexInit.CiphersServerClient, isClient)
	if err != nil {
		return
	}

	if !aeadCiphers[ctos.Cipher] {
		ctos.MAC, err = findCommon("client to server MAC", clientKexInit.MACsClientServer, serverKexInit.MACsClientServer, isClient)
		if err != nil {
			return
		}
	}

	if !aeadCiphers[stoc.Cipher] {
		stoc.MAC, err = findCommon("server to client MAC", clientKexInit.MACsServerClient, serverKexInit.MACsServerClient, isClient)
		if err != nil {
			return
		}
	}

	ctos.compression, err = findCommon("client to server compression", clientKexInit.CompressionClientServer, serverKexInit.CompressionClientServer, isClient)
	if err != nil {
		return
	}

	stoc.compression, err = findCommon("server to client compression", clientKexInit.CompressionServerClient, serverKexInit.CompressionServerClient, isClient)
	if err != nil {
		return
	}

	return result, nil
}

// If rekeythreshold is too small, we can't make any progress sending
// stuff.
const minRekeyThreshold uint64 = 256

// Config contains configuration data common to both ServerConfig and
// ClientConfig.
type Config struct {
	// Rand provides the source of entropy for cryptographic
	// primitives. If Rand is nil, the cryptographic random reader
	// in package crypto/rand will be used.
	Rand io.Reader

	// The maximum number of bytes sent or received after which a
	// new key is negotiated. It must be at least 256. If
	// unspecified, a size suitable for the chosen cipher is used.
	RekeyThreshold uint64

	// The allowed key exchanges algorithms. If unspecified then a default set
	// of algorithms is used. Unsupported values are silently ignored.
	KeyExchanges []string

	// The allowed cipher algorithms. If unspecified then a sensible default is
	// used. Unsupported values are silently ignored.
	Ciphers []string

	// The allowed MAC algorithms. If unspecified then a sensible default is
	// used. Unsupported values are silently ignored.
	MACs []string
}

// SetDefaults sets sensible values for unset fields in config. This is
// exported for testing: Configs passed to SSH functions are copied and have
// default values set automatically.
func (c *Config) SetDefaults() {
	if c.Rand == nil {
		c.Rand = rand.Reader
	}
	if c.Ciphers == nil {
		c.Ciphers = defaultCiphers
	}
	var ciphers []string
	for _, c := range c.Ciphers {
		if cipherModes[c] != nil {
			// Ignore the cipher if we have no cipherModes definition.
			ciphers = append(ciphers, c)
		}
	}
	c.Ciphers = ciphers

	if c.KeyExchanges == nil {
		c.KeyExchanges = defaultKexAlgos
	}
	var kexs []string
	for _, k := range c.KeyExchanges {
		if kexAlgoMap[k] != nil {
			// Ignore the KEX if we have no kexAlgoMap definition.
			kexs = append(kexs, k)
			if k == KeyExchangeCurve25519 && !slices.Contains(c.KeyExchanges, keyExchangeCurve25519LibSSH) {
				kexs = append(kexs, keyExchangeCurve25519LibSSH)
			}
		}
	}
	c.KeyExchanges = kexs

	if c.MACs == nil {
		c.MACs = defaultMACs
	}
	var macs []string
	for _, m := range c.MACs {
		if macModes[m] != nil {
			// Ignore the MAC if we have no macModes definition.
			macs = append(macs, m)
		}
	}
	c.MACs = macs

	if c.RekeyThreshold == 0 {
		// cipher specific default
	} else if c.RekeyThreshold < minRekeyThreshold {
		c.RekeyThreshold = minRekeyThreshold
	} else if c.RekeyThreshold >= math.MaxInt64 {
		// Avoid weirdness if somebody uses -1 as a threshold.
		c.RekeyThreshold = math.MaxInt64
	}
}

// buildDataSignedForAuth returns the data that is signed in order to prove
// possession of a private key. See RFC 4252, section 7. algo is the advertised
// algorithm, and may be a certificate type.
func buildDataSignedForAuth(sessionID []byte, req userAuthRequestMsg, algo string, pubKey []byte) []byte {
	data := struct {
		Session []byte
		Type    byte
		User    string
		Service string
		Method  string
		Sign    bool
		Algo    string
		PubKey  []byte
	}{
		sessionID,
		msgUserAuthRequest,
		req.User,
		req.Service,
		req.Method,
		true,
		algo,
		pubKey,
	}
	return Marshal(data)
}

func appendU16(buf []byte, n uint16) []byte {
	return append(buf, byte(n>>8), byte(n))
}

func appendU32(buf []byte, n uint32) []byte {
	return append(buf, byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
}

func appendU64(buf []byte, n uint64) []byte {
	return append(buf,
		byte(n>>56), byte(n>>48), byte(n>>40), byte(n>>32),
		byte(n>>24), byte(n>>16), byte(n>>8), byte(n))
}

func appendInt(buf []byte, n int) []byte {
	return appendU32(buf, uint32(n))
}

func appendString(buf []byte, s string) []byte {
	buf = appendU32(buf, uint32(len(s)))
	buf = append(buf, s...)
	return buf
}

func appendBool(buf []byte, b bool) []byte {
	if b {
		return append(buf, 1)
	}
	return append(buf, 0)
}

// newCond is a helper to hide the fact that there is no usable zero
// value for sync.Cond.
func newCond() *sync.Cond { return sync.NewCond(new(sync.Mutex)) }

// window represents the buffer available to clients
// wishing to write to a channel.
type window struct {
	*sync.Cond
	win          uint32 // RFC 4254 5.2 says the window size can grow to 2^32-1
	writeWaiters int
	closed       bool
}

// add adds win to the amount of window available
// for consumers.
func (w *window) add(win uint32) bool {
	// a zero sized window adjust is a noop.
	if win == 0 {
		return true
	}
	w.L.Lock()
	if w.win+win < win {
		w.L.Unlock()
		return false
	}
	w.win += win
	// It is unusual that multiple goroutines would be attempting to reserve
	// window space, but not guaranteed. Use broadcast to notify all waiters
	// that additional window is available.
	w.Broadcast()
	w.L.Unlock()
	return true
}

// close sets the window to closed, so all reservations fail
// immediately.
func (w *window) close() {
	w.L.Lock()
	w.closed = true
	w.Broadcast()
	w.L.Unlock()
}

// reserve reserves win from the available window capacity.
// If no capacity remains, reserve will block. reserve may
// return less than requested.
func (w *window) reserve(win uint32) (uint32, error) {
	var err error
	w.L.Lock()
	w.writeWaiters++
	w.Broadcast()
	for w.win == 0 && !w.closed {
		w.Wait()
	}
	w.writeWaiters--
	if w.win < win {
		win = w.win
	}
	w.win -= win
	if w.closed {
		err = io.EOF
	}
	w.L.Unlock()
	return win, err
}

// waitWriterBlocked waits until some goroutine is blocked for further
// writes. It is used in tests only.
func (w *window) waitWriterBlocked() {
	w.Cond.L.Lock()
	for w.writeWaiters == 0 {
		w.Cond.Wait()
	}
	w.Cond.L.Unlock()
}
