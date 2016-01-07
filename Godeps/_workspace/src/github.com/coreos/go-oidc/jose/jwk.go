package jose

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math/big"
	"strings"
)

// JSON Web Key
// https://tools.ietf.org/html/draft-ietf-jose-json-web-key-36#page-5
type JWK struct {
	ID       string
	Type     string
	Alg      string
	Use      string
	Exponent int
	Modulus  *big.Int
	Secret   []byte
}

type jwkJSON struct {
	ID       string `json:"kid"`
	Type     string `json:"kty"`
	Alg      string `json:"alg"`
	Use      string `json:"use"`
	Exponent string `json:"e"`
	Modulus  string `json:"n"`
}

func (j *JWK) MarshalJSON() ([]byte, error) {
	t := jwkJSON{
		ID:       j.ID,
		Type:     j.Type,
		Alg:      j.Alg,
		Use:      j.Use,
		Exponent: encodeExponent(j.Exponent),
		Modulus:  encodeModulus(j.Modulus),
	}

	return json.Marshal(&t)
}

func (j *JWK) UnmarshalJSON(data []byte) error {
	var t jwkJSON
	err := json.Unmarshal(data, &t)
	if err != nil {
		return err
	}

	e, err := decodeExponent(t.Exponent)
	if err != nil {
		return err
	}

	n, err := decodeModulus(t.Modulus)
	if err != nil {
		return err
	}

	j.ID = t.ID
	j.Type = t.Type
	j.Alg = t.Alg
	j.Use = t.Use
	j.Exponent = e
	j.Modulus = n

	return nil
}

func decodeExponent(e string) (int, error) {
	decE, err := decodeBase64URLPaddingOptional(e)
	if err != nil {
		return 0, err
	}
	var eBytes []byte
	if len(decE) < 8 {
		eBytes = make([]byte, 8-len(decE), 8)
		eBytes = append(eBytes, decE...)
	} else {
		eBytes = decE
	}
	eReader := bytes.NewReader(eBytes)
	var E uint64
	err = binary.Read(eReader, binary.BigEndian, &E)
	if err != nil {
		return 0, err
	}
	return int(E), nil
}

func encodeExponent(e int) string {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, uint64(e))
	var idx int
	for ; idx < 8; idx++ {
		if b[idx] != 0x0 {
			break
		}
	}
	return base64.URLEncoding.EncodeToString(b[idx:])
}

// Turns a URL encoded modulus of a key into a big int.
func decodeModulus(n string) (*big.Int, error) {
	decN, err := decodeBase64URLPaddingOptional(n)
	if err != nil {
		return nil, err
	}
	N := big.NewInt(0)
	N.SetBytes(decN)
	return N, nil
}

func encodeModulus(n *big.Int) string {
	return base64.URLEncoding.EncodeToString(n.Bytes())
}

// decodeBase64URLPaddingOptional decodes Base64 whether there is padding or not.
// The stdlib version currently doesn't handle this.
// We can get rid of this is if this bug:
//   https://github.com/golang/go/issues/4237
// ever closes.
func decodeBase64URLPaddingOptional(e string) ([]byte, error) {
	if m := len(e) % 4; m != 0 {
		e += strings.Repeat("=", 4-m)
	}
	return base64.URLEncoding.DecodeString(e)
}
