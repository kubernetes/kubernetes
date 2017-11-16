package libtrust

import (
	"bytes"
	"crypto"
	"crypto/elliptic"
	"crypto/tls"
	"crypto/x509"
	"encoding/base32"
	"encoding/base64"
	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// LoadOrCreateTrustKey will load a PrivateKey from the specified path
func LoadOrCreateTrustKey(trustKeyPath string) (PrivateKey, error) {
	if err := os.MkdirAll(filepath.Dir(trustKeyPath), 0700); err != nil {
		return nil, err
	}

	trustKey, err := LoadKeyFile(trustKeyPath)
	if err == ErrKeyFileDoesNotExist {
		trustKey, err = GenerateECP256PrivateKey()
		if err != nil {
			return nil, fmt.Errorf("error generating key: %s", err)
		}

		if err := SaveKey(trustKeyPath, trustKey); err != nil {
			return nil, fmt.Errorf("error saving key file: %s", err)
		}

		dir, file := filepath.Split(trustKeyPath)
		if err := SavePublicKey(filepath.Join(dir, "public-"+file), trustKey.PublicKey()); err != nil {
			return nil, fmt.Errorf("error saving public key file: %s", err)
		}
	} else if err != nil {
		return nil, fmt.Errorf("error loading key file: %s", err)
	}
	return trustKey, nil
}

// NewIdentityAuthTLSClientConfig returns a tls.Config configured to use identity
// based authentication from the specified dockerUrl, the rootConfigPath and
// the server name to which it is connecting.
// If trustUnknownHosts is true it will automatically add the host to the
// known-hosts.json in rootConfigPath.
func NewIdentityAuthTLSClientConfig(dockerUrl string, trustUnknownHosts bool, rootConfigPath string, serverName string) (*tls.Config, error) {
	tlsConfig := newTLSConfig()

	trustKeyPath := filepath.Join(rootConfigPath, "key.json")
	knownHostsPath := filepath.Join(rootConfigPath, "known-hosts.json")

	u, err := url.Parse(dockerUrl)
	if err != nil {
		return nil, fmt.Errorf("unable to parse machine url")
	}

	if u.Scheme == "unix" {
		return nil, nil
	}

	addr := u.Host
	proto := "tcp"

	trustKey, err := LoadOrCreateTrustKey(trustKeyPath)
	if err != nil {
		return nil, fmt.Errorf("unable to load trust key: %s", err)
	}

	knownHosts, err := LoadKeySetFile(knownHostsPath)
	if err != nil {
		return nil, fmt.Errorf("could not load trusted hosts file: %s", err)
	}

	allowedHosts, err := FilterByHosts(knownHosts, addr, false)
	if err != nil {
		return nil, fmt.Errorf("error filtering hosts: %s", err)
	}

	certPool, err := GenerateCACertPool(trustKey, allowedHosts)
	if err != nil {
		return nil, fmt.Errorf("Could not create CA pool: %s", err)
	}

	tlsConfig.ServerName = serverName
	tlsConfig.RootCAs = certPool

	x509Cert, err := GenerateSelfSignedClientCert(trustKey)
	if err != nil {
		return nil, fmt.Errorf("certificate generation error: %s", err)
	}

	tlsConfig.Certificates = []tls.Certificate{{
		Certificate: [][]byte{x509Cert.Raw},
		PrivateKey:  trustKey.CryptoPrivateKey(),
		Leaf:        x509Cert,
	}}

	tlsConfig.InsecureSkipVerify = true

	testConn, err := tls.Dial(proto, addr, tlsConfig)
	if err != nil {
		return nil, fmt.Errorf("tls Handshake error: %s", err)
	}

	opts := x509.VerifyOptions{
		Roots:         tlsConfig.RootCAs,
		CurrentTime:   time.Now(),
		DNSName:       tlsConfig.ServerName,
		Intermediates: x509.NewCertPool(),
	}

	certs := testConn.ConnectionState().PeerCertificates
	for i, cert := range certs {
		if i == 0 {
			continue
		}
		opts.Intermediates.AddCert(cert)
	}

	if _, err := certs[0].Verify(opts); err != nil {
		if _, ok := err.(x509.UnknownAuthorityError); ok {
			if trustUnknownHosts {
				pubKey, err := FromCryptoPublicKey(certs[0].PublicKey)
				if err != nil {
					return nil, fmt.Errorf("error extracting public key from cert: %s", err)
				}

				pubKey.AddExtendedField("hosts", []string{addr})

				if err := AddKeySetFile(knownHostsPath, pubKey); err != nil {
					return nil, fmt.Errorf("error adding machine to known hosts: %s", err)
				}
			} else {
				return nil, fmt.Errorf("unable to connect.  unknown host: %s", addr)
			}
		}
	}

	testConn.Close()
	tlsConfig.InsecureSkipVerify = false

	return tlsConfig, nil
}

// joseBase64UrlEncode encodes the given data using the standard base64 url
// encoding format but with all trailing '=' characters ommitted in accordance
// with the jose specification.
// http://tools.ietf.org/html/draft-ietf-jose-json-web-signature-31#section-2
func joseBase64UrlEncode(b []byte) string {
	return strings.TrimRight(base64.URLEncoding.EncodeToString(b), "=")
}

// joseBase64UrlDecode decodes the given string using the standard base64 url
// decoder but first adds the appropriate number of trailing '=' characters in
// accordance with the jose specification.
// http://tools.ietf.org/html/draft-ietf-jose-json-web-signature-31#section-2
func joseBase64UrlDecode(s string) ([]byte, error) {
	s = strings.Replace(s, "\n", "", -1)
	s = strings.Replace(s, " ", "", -1)
	switch len(s) % 4 {
	case 0:
	case 2:
		s += "=="
	case 3:
		s += "="
	default:
		return nil, errors.New("illegal base64url string")
	}
	return base64.URLEncoding.DecodeString(s)
}

func keyIDEncode(b []byte) string {
	s := strings.TrimRight(base32.StdEncoding.EncodeToString(b), "=")
	var buf bytes.Buffer
	var i int
	for i = 0; i < len(s)/4-1; i++ {
		start := i * 4
		end := start + 4
		buf.WriteString(s[start:end] + ":")
	}
	buf.WriteString(s[i*4:])
	return buf.String()
}

func keyIDFromCryptoKey(pubKey PublicKey) string {
	// Generate and return a 'libtrust' fingerprint of the public key.
	// For an RSA key this should be:
	//   SHA256(DER encoded ASN1)
	// Then truncated to 240 bits and encoded into 12 base32 groups like so:
	//   ABCD:EFGH:IJKL:MNOP:QRST:UVWX:YZ23:4567:ABCD:EFGH:IJKL:MNOP
	derBytes, err := x509.MarshalPKIXPublicKey(pubKey.CryptoPublicKey())
	if err != nil {
		return ""
	}
	hasher := crypto.SHA256.New()
	hasher.Write(derBytes)
	return keyIDEncode(hasher.Sum(nil)[:30])
}

func stringFromMap(m map[string]interface{}, key string) (string, error) {
	val, ok := m[key]
	if !ok {
		return "", fmt.Errorf("%q value not specified", key)
	}

	str, ok := val.(string)
	if !ok {
		return "", fmt.Errorf("%q value must be a string", key)
	}
	delete(m, key)

	return str, nil
}

func parseECCoordinate(cB64Url string, curve elliptic.Curve) (*big.Int, error) {
	curveByteLen := (curve.Params().BitSize + 7) >> 3

	cBytes, err := joseBase64UrlDecode(cB64Url)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 URL encoding: %s", err)
	}
	cByteLength := len(cBytes)
	if cByteLength != curveByteLen {
		return nil, fmt.Errorf("invalid number of octets: got %d, should be %d", cByteLength, curveByteLen)
	}
	return new(big.Int).SetBytes(cBytes), nil
}

func parseECPrivateParam(dB64Url string, curve elliptic.Curve) (*big.Int, error) {
	dBytes, err := joseBase64UrlDecode(dB64Url)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 URL encoding: %s", err)
	}

	// The length of this octet string MUST be ceiling(log-base-2(n)/8)
	// octets (where n is the order of the curve). This is because the private
	// key d must be in the interval [1, n-1] so the bitlength of d should be
	// no larger than the bitlength of n-1. The easiest way to find the octet
	// length is to take bitlength(n-1), add 7 to force a carry, and shift this
	// bit sequence right by 3, which is essentially dividing by 8 and adding
	// 1 if there is any remainder. Thus, the private key value d should be
	// output to (bitlength(n-1)+7)>>3 octets.
	n := curve.Params().N
	octetLength := (new(big.Int).Sub(n, big.NewInt(1)).BitLen() + 7) >> 3
	dByteLength := len(dBytes)

	if dByteLength != octetLength {
		return nil, fmt.Errorf("invalid number of octets: got %d, should be %d", dByteLength, octetLength)
	}

	return new(big.Int).SetBytes(dBytes), nil
}

func parseRSAModulusParam(nB64Url string) (*big.Int, error) {
	nBytes, err := joseBase64UrlDecode(nB64Url)
	if err != nil {
		return nil, fmt.Errorf("invalid base64 URL encoding: %s", err)
	}

	return new(big.Int).SetBytes(nBytes), nil
}

func serializeRSAPublicExponentParam(e int) []byte {
	// We MUST use the minimum number of octets to represent E.
	// E is supposed to be 65537 for performance and security reasons
	// and is what golang's rsa package generates, but it might be
	// different if imported from some other generator.
	buf := make([]byte, 4)
	binary.BigEndian.PutUint32(buf, uint32(e))
	var i int
	for i = 0; i < 8; i++ {
		if buf[i] != 0 {
			break
		}
	}
	return buf[i:]
}

func parseRSAPublicExponentParam(eB64Url string) (int, error) {
	eBytes, err := joseBase64UrlDecode(eB64Url)
	if err != nil {
		return 0, fmt.Errorf("invalid base64 URL encoding: %s", err)
	}
	// Only the minimum number of bytes were used to represent E, but
	// binary.BigEndian.Uint32 expects at least 4 bytes, so we need
	// to add zero padding if necassary.
	byteLen := len(eBytes)
	buf := make([]byte, 4-byteLen, 4)
	eBytes = append(buf, eBytes...)

	return int(binary.BigEndian.Uint32(eBytes)), nil
}

func parseRSAPrivateKeyParamFromMap(m map[string]interface{}, key string) (*big.Int, error) {
	b64Url, err := stringFromMap(m, key)
	if err != nil {
		return nil, err
	}

	paramBytes, err := joseBase64UrlDecode(b64Url)
	if err != nil {
		return nil, fmt.Errorf("invaled base64 URL encoding: %s", err)
	}

	return new(big.Int).SetBytes(paramBytes), nil
}

func createPemBlock(name string, derBytes []byte, headers map[string]interface{}) (*pem.Block, error) {
	pemBlock := &pem.Block{Type: name, Bytes: derBytes, Headers: map[string]string{}}
	for k, v := range headers {
		switch val := v.(type) {
		case string:
			pemBlock.Headers[k] = val
		case []string:
			if k == "hosts" {
				pemBlock.Headers[k] = strings.Join(val, ",")
			} else {
				// Return error, non-encodable type
			}
		default:
			// Return error, non-encodable type
		}
	}

	return pemBlock, nil
}

func pubKeyFromPEMBlock(pemBlock *pem.Block) (PublicKey, error) {
	cryptoPublicKey, err := x509.ParsePKIXPublicKey(pemBlock.Bytes)
	if err != nil {
		return nil, fmt.Errorf("unable to decode Public Key PEM data: %s", err)
	}

	pubKey, err := FromCryptoPublicKey(cryptoPublicKey)
	if err != nil {
		return nil, err
	}

	addPEMHeadersToKey(pemBlock, pubKey)

	return pubKey, nil
}

func addPEMHeadersToKey(pemBlock *pem.Block, pubKey PublicKey) {
	for key, value := range pemBlock.Headers {
		var safeVal interface{}
		if key == "hosts" {
			safeVal = strings.Split(value, ",")
		} else {
			safeVal = value
		}
		pubKey.AddExtendedField(key, safeVal)
	}
}
