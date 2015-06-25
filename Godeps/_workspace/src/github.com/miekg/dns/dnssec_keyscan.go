package dns

import (
	"crypto/dsa"
	"crypto/ecdsa"
	"crypto/rsa"
	"io"
	"math/big"
	"strconv"
	"strings"
)

// NewPrivateKey returns a PrivateKey by parsing the string s.
// s should be in the same form of the BIND private key files.
func (k *DNSKEY) NewPrivateKey(s string) (PrivateKey, error) {
	if s[len(s)-1] != '\n' { // We need a closing newline
		return k.ReadPrivateKey(strings.NewReader(s+"\n"), "")
	}
	return k.ReadPrivateKey(strings.NewReader(s), "")
}

// ReadPrivateKey reads a private key from the io.Reader q. The string file is
// only used in error reporting.
// The public key must be known, because some cryptographic algorithms embed
// the public inside the privatekey.
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
	algo, err := strconv.Atoi(strings.SplitN(m["algorithm"], " ", 2)[0])
	if err != nil {
		return nil, ErrPrivKey
	}
	switch uint8(algo) {
	case DSA:
		priv, e := readPrivateKeyDSA(m)
		if e != nil {
			return nil, e
		}
		pub := k.publicKeyDSA()
		if pub == nil {
			return nil, ErrKey
		}
		priv.PublicKey = *pub
		return (*DSAPrivateKey)(priv), e
	case RSAMD5:
		fallthrough
	case RSASHA1:
		fallthrough
	case RSASHA1NSEC3SHA1:
		fallthrough
	case RSASHA256:
		fallthrough
	case RSASHA512:
		priv, e := readPrivateKeyRSA(m)
		if e != nil {
			return nil, e
		}
		pub := k.publicKeyRSA()
		if pub == nil {
			return nil, ErrKey
		}
		priv.PublicKey = *pub
		return (*RSAPrivateKey)(priv), e
	case ECCGOST:
		return nil, ErrPrivKey
	case ECDSAP256SHA256:
		fallthrough
	case ECDSAP384SHA384:
		priv, e := readPrivateKeyECDSA(m)
		if e != nil {
			return nil, e
		}
		pub := k.publicKeyECDSA()
		if pub == nil {
			return nil, ErrKey
		}
		priv.PublicKey = *pub
		return (*ECDSAPrivateKey)(priv), e
	default:
		return nil, ErrPrivKey
	}
}

// Read a private key (file) string and create a public key. Return the private key.
func readPrivateKeyRSA(m map[string]string) (*rsa.PrivateKey, error) {
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

func readPrivateKeyDSA(m map[string]string) (*dsa.PrivateKey, error) {
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

func readPrivateKeyECDSA(m map[string]string) (*ecdsa.PrivateKey, error) {
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
		case zKey:
			k = l.token
		case zValue:
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
				l.value = zKey
				c <- l
				// Next token is a space, eat it
				s.tokenText()
				key = false
				str = ""
			} else {
				l.value = zValue
			}
		case ';':
			commt = true
		case '\n':
			if commt {
				// Reset a comment
				commt = false
			}
			l.value = zValue
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
		l.value = zValue
		c <- l
	}
}
