package dns

import (
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rsa"
	"io"
	"math/big"
	"strings"
)

func (k *DNSKEY) NewPrivateKey(s string) (PrivateKey, error) {
	if s[len(s)-1] != '\n' { // We need a closing newline
		return k.ReadPrivateKey(strings.NewReader(s+"\n"), "")
	}
	return k.ReadPrivateKey(strings.NewReader(s), "")
}

// ReadPrivateKey reads a private key from the io.Reader q. The string file is
// only used in error reporting.
// The public key must be
// known, because some cryptographic algorithms embed the public inside the privatekey.
func (k *DNSKEY) ReadPrivateKey(q io.Reader, file string) (PrivateKey, error) {
	m, e := parseKey(q, file)
	if m == nil {
		return nil, e
	}
	if _, ok := m["private-key-format"]; !ok {
		return nil, ErrPrivKey
	}
	if m["private-key-format"] != "v1.2" && m["private-key-format"] != "v1.3" {
		return nil, ErrPrivKey
	}
	// TODO(mg): check if the pubkey matches the private key
	switch m["algorithm"] {
	case "3 (DSA)":
		p, e := readPrivateKeyDSA(m)
		if e != nil {
			return nil, e
		}
		if !k.setPublicKeyInPrivate(p) {
			return nil, ErrKey
		}
		return p, e
	case "1 (RSAMD5)":
		fallthrough
	case "5 (RSASHA1)":
		fallthrough
	case "7 (RSASHA1NSEC3SHA1)":
		fallthrough
	case "8 (RSASHA256)":
		fallthrough
	case "10 (RSASHA512)":
		p, e := readPrivateKeyRSA(m)
		if e != nil {
			return nil, e
		}
		if !k.setPublicKeyInPrivate(p) {
			return nil, ErrKey
		}
		return p, e
	case "12 (ECC-GOST)":
		p, e := readPrivateKeyGOST(m)
		if e != nil {
			return nil, e
		}
		// setPublicKeyInPrivate(p)
		return p, e
	case "13 (ECDSAP256SHA256)":
		fallthrough
	case "14 (ECDSAP384SHA384)":
		p, e := readPrivateKeyECDSA(m)
		if e != nil {
			return nil, e
		}
		if !k.setPublicKeyInPrivate(p) {
			return nil, ErrKey
		}
		return p, e
	}
	return nil, ErrPrivKey
}

// Read a private key (file) string and create a public key. Return the private key.
func readPrivateKeyRSA(m map[string]string) (PrivateKey, error) {
	p := new(rsa.PrivateKey)
	p.Primes = []*big.Int{nil, nil}
	for k, v := range m {
		switch k {
		case "modulus", "publicexponent", "privateexponent", "prime1", "prime2":
			v1, err := fromBase64([]byte(v))
			if err != nil {
				return nil, err
			}
			switch k {
			case "modulus":
				p.PublicKey.N = big.NewInt(0)
				p.PublicKey.N.SetBytes(v1)
			case "publicexponent":
				i := big.NewInt(0)
				i.SetBytes(v1)
				p.PublicKey.E = int(i.Int64()) // int64 should be large enough
			case "privateexponent":
				p.D = big.NewInt(0)
				p.D.SetBytes(v1)
			case "prime1":
				p.Primes[0] = big.NewInt(0)
				p.Primes[0].SetBytes(v1)
			case "prime2":
				p.Primes[1] = big.NewInt(0)
				p.Primes[1].SetBytes(v1)
			}
		case "exponent1", "exponent2", "coefficient":
			// not used in Go (yet)
		case "created", "publish", "activate":
			// not used in Go (yet)
		}
	}
	return p, nil
}

func readPrivateKeyDSA(m map[string]string) (PrivateKey, error) {
	p := new(dsa.PrivateKey)
	p.X = big.NewInt(0)
	for k, v := range m {
		switch k {
		case "private_value(x)":
			v1, err := fromBase64([]byte(v))
			if err != nil {
				return nil, err
			}
			p.X.SetBytes(v1)
		case "created", "publish", "activate":
			/* not used in Go (yet) */
		}
	}
	return p, nil
}

func readPrivateKeyECDSA(m map[string]string) (PrivateKey, error) {
	p := new(ecdsa.PrivateKey)
	p.D = big.NewInt(0)
	// TODO: validate that the required flags are present
	for k, v := range m {
		switch k {
		case "privatekey":
			v1, err := fromBase64([]byte(v))
			if err != nil {
				return nil, err
			}
			p.D.SetBytes(v1)
		case "created", "publish", "activate":
			/* not used in Go (yet) */
		}
	}
	return p, nil
}

func readPrivateKeyGOST(m map[string]string) (PrivateKey, error) {
	// TODO(miek)
	return nil, nil
}

// parseKey reads a private key from r. It returns a map[string]string,
// with the key-value pairs, or an error when the file is not correct.
func parseKey(r io.Reader, file string) (map[string]string, error) {
	s := scanInit(r)
	m := make(map[string]string)
	c := make(chan lex)
	k := ""
	// Start the lexer
	go klexer(s, c)
	for l := range c {
		// It should alternate
		switch l.value {
		case _KEY:
			k = l.token
		case _VALUE:
			if k == "" {
				return nil, &ParseError{file, "no private key seen", l}
			}
			//println("Setting", strings.ToLower(k), "to", l.token, "b")
			m[strings.ToLower(k)] = l.token
			k = ""
		}
	}
	return m, nil
}

// klexer scans the sourcefile and returns tokens on the channel c.
func klexer(s *scan, c chan lex) {
	var l lex
	str := "" // Hold the current read text
	commt := false
	key := true
	x, err := s.tokenText()
	defer close(c)
	for err == nil {
		l.column = s.position.Column
		l.line = s.position.Line
		switch x {
		case ':':
			if commt {
				break
			}
			l.token = str
			if key {
				l.value = _KEY
				c <- l
				// Next token is a space, eat it
				s.tokenText()
				key = false
				str = ""
			} else {
				l.value = _VALUE
			}
		case ';':
			commt = true
		case '\n':
			if commt {
				// Reset a comment
				commt = false
			}
			l.value = _VALUE
			l.token = str
			c <- l
			str = ""
			commt = false
			key = true
		default:
			if commt {
				break
			}
			str += string(x)
		}
		x, err = s.tokenText()
	}
	if len(str) > 0 {
		// Send remainder
		l.token = str
		l.value = _VALUE
		c <- l
	}
}
