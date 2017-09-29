package libtrust

import (
	"bytes"
	"crypto"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"
	"unicode"
)

var (
	// ErrInvalidSignContent is used when the content to be signed is invalid.
	ErrInvalidSignContent = errors.New("invalid sign content")

	// ErrInvalidJSONContent is used when invalid json is encountered.
	ErrInvalidJSONContent = errors.New("invalid json content")

	// ErrMissingSignatureKey is used when the specified signature key
	// does not exist in the JSON content.
	ErrMissingSignatureKey = errors.New("missing signature key")
)

type jsHeader struct {
	JWK       PublicKey `json:"jwk,omitempty"`
	Algorithm string    `json:"alg"`
	Chain     []string  `json:"x5c,omitempty"`
}

type jsSignature struct {
	Header    jsHeader `json:"header"`
	Signature string   `json:"signature"`
	Protected string   `json:"protected,omitempty"`
}

type jsSignaturesSorted []jsSignature

func (jsbkid jsSignaturesSorted) Swap(i, j int) { jsbkid[i], jsbkid[j] = jsbkid[j], jsbkid[i] }
func (jsbkid jsSignaturesSorted) Len() int      { return len(jsbkid) }

func (jsbkid jsSignaturesSorted) Less(i, j int) bool {
	ki, kj := jsbkid[i].Header.JWK.KeyID(), jsbkid[j].Header.JWK.KeyID()
	si, sj := jsbkid[i].Signature, jsbkid[j].Signature

	if ki == kj {
		return si < sj
	}

	return ki < kj
}

type signKey struct {
	PrivateKey
	Chain []*x509.Certificate
}

// JSONSignature represents a signature of a json object.
type JSONSignature struct {
	payload      string
	signatures   []jsSignature
	indent       string
	formatLength int
	formatTail   []byte
}

func newJSONSignature() *JSONSignature {
	return &JSONSignature{
		signatures: make([]jsSignature, 0, 1),
	}
}

// Payload returns the encoded payload of the signature. This
// payload should not be signed directly
func (js *JSONSignature) Payload() ([]byte, error) {
	return joseBase64UrlDecode(js.payload)
}

func (js *JSONSignature) protectedHeader() (string, error) {
	protected := map[string]interface{}{
		"formatLength": js.formatLength,
		"formatTail":   joseBase64UrlEncode(js.formatTail),
		"time":         time.Now().UTC().Format(time.RFC3339),
	}
	protectedBytes, err := json.Marshal(protected)
	if err != nil {
		return "", err
	}

	return joseBase64UrlEncode(protectedBytes), nil
}

func (js *JSONSignature) signBytes(protectedHeader string) ([]byte, error) {
	buf := make([]byte, len(js.payload)+len(protectedHeader)+1)
	copy(buf, protectedHeader)
	buf[len(protectedHeader)] = '.'
	copy(buf[len(protectedHeader)+1:], js.payload)
	return buf, nil
}

// Sign adds a signature using the given private key.
func (js *JSONSignature) Sign(key PrivateKey) error {
	protected, err := js.protectedHeader()
	if err != nil {
		return err
	}
	signBytes, err := js.signBytes(protected)
	if err != nil {
		return err
	}
	sigBytes, algorithm, err := key.Sign(bytes.NewReader(signBytes), crypto.SHA256)
	if err != nil {
		return err
	}

	js.signatures = append(js.signatures, jsSignature{
		Header: jsHeader{
			JWK:       key.PublicKey(),
			Algorithm: algorithm,
		},
		Signature: joseBase64UrlEncode(sigBytes),
		Protected: protected,
	})

	return nil
}

// SignWithChain adds a signature using the given private key
// and setting the x509 chain. The public key of the first element
// in the chain must be the public key corresponding with the sign key.
func (js *JSONSignature) SignWithChain(key PrivateKey, chain []*x509.Certificate) error {
	// Ensure key.Chain[0] is public key for key
	//key.Chain.PublicKey
	//key.PublicKey().CryptoPublicKey()

	// Verify chain
	protected, err := js.protectedHeader()
	if err != nil {
		return err
	}
	signBytes, err := js.signBytes(protected)
	if err != nil {
		return err
	}
	sigBytes, algorithm, err := key.Sign(bytes.NewReader(signBytes), crypto.SHA256)
	if err != nil {
		return err
	}

	header := jsHeader{
		Chain:     make([]string, len(chain)),
		Algorithm: algorithm,
	}

	for i, cert := range chain {
		header.Chain[i] = base64.StdEncoding.EncodeToString(cert.Raw)
	}

	js.signatures = append(js.signatures, jsSignature{
		Header:    header,
		Signature: joseBase64UrlEncode(sigBytes),
		Protected: protected,
	})

	return nil
}

// Verify verifies all the signatures and returns the list of
// public keys used to sign. Any x509 chains are not checked.
func (js *JSONSignature) Verify() ([]PublicKey, error) {
	keys := make([]PublicKey, len(js.signatures))
	for i, signature := range js.signatures {
		signBytes, err := js.signBytes(signature.Protected)
		if err != nil {
			return nil, err
		}
		var publicKey PublicKey
		if len(signature.Header.Chain) > 0 {
			certBytes, err := base64.StdEncoding.DecodeString(signature.Header.Chain[0])
			if err != nil {
				return nil, err
			}
			cert, err := x509.ParseCertificate(certBytes)
			if err != nil {
				return nil, err
			}
			publicKey, err = FromCryptoPublicKey(cert.PublicKey)
			if err != nil {
				return nil, err
			}
		} else if signature.Header.JWK != nil {
			publicKey = signature.Header.JWK
		} else {
			return nil, errors.New("missing public key")
		}

		sigBytes, err := joseBase64UrlDecode(signature.Signature)
		if err != nil {
			return nil, err
		}

		err = publicKey.Verify(bytes.NewReader(signBytes), signature.Header.Algorithm, sigBytes)
		if err != nil {
			return nil, err
		}

		keys[i] = publicKey
	}
	return keys, nil
}

// VerifyChains verifies all the signatures and the chains associated
// with each signature and returns the list of verified chains.
// Signatures without an x509 chain are not checked.
func (js *JSONSignature) VerifyChains(ca *x509.CertPool) ([][]*x509.Certificate, error) {
	chains := make([][]*x509.Certificate, 0, len(js.signatures))
	for _, signature := range js.signatures {
		signBytes, err := js.signBytes(signature.Protected)
		if err != nil {
			return nil, err
		}
		var publicKey PublicKey
		if len(signature.Header.Chain) > 0 {
			certBytes, err := base64.StdEncoding.DecodeString(signature.Header.Chain[0])
			if err != nil {
				return nil, err
			}
			cert, err := x509.ParseCertificate(certBytes)
			if err != nil {
				return nil, err
			}
			publicKey, err = FromCryptoPublicKey(cert.PublicKey)
			if err != nil {
				return nil, err
			}
			intermediates := x509.NewCertPool()
			if len(signature.Header.Chain) > 1 {
				intermediateChain := signature.Header.Chain[1:]
				for i := range intermediateChain {
					certBytes, err := base64.StdEncoding.DecodeString(intermediateChain[i])
					if err != nil {
						return nil, err
					}
					intermediate, err := x509.ParseCertificate(certBytes)
					if err != nil {
						return nil, err
					}
					intermediates.AddCert(intermediate)
				}
			}

			verifyOptions := x509.VerifyOptions{
				Intermediates: intermediates,
				Roots:         ca,
			}

			verifiedChains, err := cert.Verify(verifyOptions)
			if err != nil {
				return nil, err
			}
			chains = append(chains, verifiedChains...)

			sigBytes, err := joseBase64UrlDecode(signature.Signature)
			if err != nil {
				return nil, err
			}

			err = publicKey.Verify(bytes.NewReader(signBytes), signature.Header.Algorithm, sigBytes)
			if err != nil {
				return nil, err
			}
		}

	}
	return chains, nil
}

// JWS returns JSON serialized JWS according to
// http://tools.ietf.org/html/draft-ietf-jose-json-web-signature-31#section-7.2
func (js *JSONSignature) JWS() ([]byte, error) {
	if len(js.signatures) == 0 {
		return nil, errors.New("missing signature")
	}

	sort.Sort(jsSignaturesSorted(js.signatures))

	jsonMap := map[string]interface{}{
		"payload":    js.payload,
		"signatures": js.signatures,
	}

	return json.MarshalIndent(jsonMap, "", "   ")
}

func notSpace(r rune) bool {
	return !unicode.IsSpace(r)
}

func detectJSONIndent(jsonContent []byte) (indent string) {
	if len(jsonContent) > 2 && jsonContent[0] == '{' && jsonContent[1] == '\n' {
		quoteIndex := bytes.IndexRune(jsonContent[1:], '"')
		if quoteIndex > 0 {
			indent = string(jsonContent[2 : quoteIndex+1])
		}
	}
	return
}

type jsParsedHeader struct {
	JWK       json.RawMessage `json:"jwk"`
	Algorithm string          `json:"alg"`
	Chain     []string        `json:"x5c"`
}

type jsParsedSignature struct {
	Header    jsParsedHeader `json:"header"`
	Signature string         `json:"signature"`
	Protected string         `json:"protected"`
}

// ParseJWS parses a JWS serialized JSON object into a Json Signature.
func ParseJWS(content []byte) (*JSONSignature, error) {
	type jsParsed struct {
		Payload    string              `json:"payload"`
		Signatures []jsParsedSignature `json:"signatures"`
	}
	parsed := &jsParsed{}
	err := json.Unmarshal(content, parsed)
	if err != nil {
		return nil, err
	}
	if len(parsed.Signatures) == 0 {
		return nil, errors.New("missing signatures")
	}
	payload, err := joseBase64UrlDecode(parsed.Payload)
	if err != nil {
		return nil, err
	}

	js, err := NewJSONSignature(payload)
	if err != nil {
		return nil, err
	}
	js.signatures = make([]jsSignature, len(parsed.Signatures))
	for i, signature := range parsed.Signatures {
		header := jsHeader{
			Algorithm: signature.Header.Algorithm,
		}
		if signature.Header.Chain != nil {
			header.Chain = signature.Header.Chain
		}
		if signature.Header.JWK != nil {
			publicKey, err := UnmarshalPublicKeyJWK([]byte(signature.Header.JWK))
			if err != nil {
				return nil, err
			}
			header.JWK = publicKey
		}
		js.signatures[i] = jsSignature{
			Header:    header,
			Signature: signature.Signature,
			Protected: signature.Protected,
		}
	}

	return js, nil
}

// NewJSONSignature returns a new unsigned JWS from a json byte array.
// JSONSignature will need to be signed before serializing or storing.
// Optionally, one or more signatures can be provided as byte buffers,
// containing serialized JWS signatures, to assemble a fully signed JWS
// package. It is the callers responsibility to ensure uniqueness of the
// provided signatures.
func NewJSONSignature(content []byte, signatures ...[]byte) (*JSONSignature, error) {
	var dataMap map[string]interface{}
	err := json.Unmarshal(content, &dataMap)
	if err != nil {
		return nil, err
	}

	js := newJSONSignature()
	js.indent = detectJSONIndent(content)

	js.payload = joseBase64UrlEncode(content)

	// Find trailing } and whitespace, put in protected header
	closeIndex := bytes.LastIndexFunc(content, notSpace)
	if content[closeIndex] != '}' {
		return nil, ErrInvalidJSONContent
	}
	lastRuneIndex := bytes.LastIndexFunc(content[:closeIndex], notSpace)
	if content[lastRuneIndex] == ',' {
		return nil, ErrInvalidJSONContent
	}
	js.formatLength = lastRuneIndex + 1
	js.formatTail = content[js.formatLength:]

	if len(signatures) > 0 {
		for _, signature := range signatures {
			var parsedJSig jsParsedSignature

			if err := json.Unmarshal(signature, &parsedJSig); err != nil {
				return nil, err
			}

			// TODO(stevvooe): A lot of the code below is repeated in
			// ParseJWS. It will require more refactoring to fix that.
			jsig := jsSignature{
				Header: jsHeader{
					Algorithm: parsedJSig.Header.Algorithm,
				},
				Signature: parsedJSig.Signature,
				Protected: parsedJSig.Protected,
			}

			if parsedJSig.Header.Chain != nil {
				jsig.Header.Chain = parsedJSig.Header.Chain
			}

			if parsedJSig.Header.JWK != nil {
				publicKey, err := UnmarshalPublicKeyJWK([]byte(parsedJSig.Header.JWK))
				if err != nil {
					return nil, err
				}
				jsig.Header.JWK = publicKey
			}

			js.signatures = append(js.signatures, jsig)
		}
	}

	return js, nil
}

// NewJSONSignatureFromMap returns a new unsigned JSONSignature from a map or
// struct. JWS will need to be signed before serializing or storing.
func NewJSONSignatureFromMap(content interface{}) (*JSONSignature, error) {
	switch content.(type) {
	case map[string]interface{}:
	case struct{}:
	default:
		return nil, errors.New("invalid data type")
	}

	js := newJSONSignature()
	js.indent = "   "

	payload, err := json.MarshalIndent(content, "", js.indent)
	if err != nil {
		return nil, err
	}
	js.payload = joseBase64UrlEncode(payload)

	// Remove '\n}' from formatted section, put in protected header
	js.formatLength = len(payload) - 2
	js.formatTail = payload[js.formatLength:]

	return js, nil
}

func readIntFromMap(key string, m map[string]interface{}) (int, bool) {
	value, ok := m[key]
	if !ok {
		return 0, false
	}
	switch v := value.(type) {
	case int:
		return v, true
	case float64:
		return int(v), true
	default:
		return 0, false
	}
}

func readStringFromMap(key string, m map[string]interface{}) (v string, ok bool) {
	value, ok := m[key]
	if !ok {
		return "", false
	}
	v, ok = value.(string)
	return
}

// ParsePrettySignature parses a formatted signature into a
// JSON signature. If the signatures are missing the format information
// an error is thrown. The formatted signature must be created by
// the same method as format signature.
func ParsePrettySignature(content []byte, signatureKey string) (*JSONSignature, error) {
	var contentMap map[string]json.RawMessage
	err := json.Unmarshal(content, &contentMap)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling content: %s", err)
	}
	sigMessage, ok := contentMap[signatureKey]
	if !ok {
		return nil, ErrMissingSignatureKey
	}

	var signatureBlocks []jsParsedSignature
	err = json.Unmarshal([]byte(sigMessage), &signatureBlocks)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling signatures: %s", err)
	}

	js := newJSONSignature()
	js.signatures = make([]jsSignature, len(signatureBlocks))

	for i, signatureBlock := range signatureBlocks {
		protectedBytes, err := joseBase64UrlDecode(signatureBlock.Protected)
		if err != nil {
			return nil, fmt.Errorf("base64 decode error: %s", err)
		}
		var protectedHeader map[string]interface{}
		err = json.Unmarshal(protectedBytes, &protectedHeader)
		if err != nil {
			return nil, fmt.Errorf("error unmarshalling protected header: %s", err)
		}

		formatLength, ok := readIntFromMap("formatLength", protectedHeader)
		if !ok {
			return nil, errors.New("missing formatted length")
		}
		encodedTail, ok := readStringFromMap("formatTail", protectedHeader)
		if !ok {
			return nil, errors.New("missing formatted tail")
		}
		formatTail, err := joseBase64UrlDecode(encodedTail)
		if err != nil {
			return nil, fmt.Errorf("base64 decode error on tail: %s", err)
		}
		if js.formatLength == 0 {
			js.formatLength = formatLength
		} else if js.formatLength != formatLength {
			return nil, errors.New("conflicting format length")
		}
		if len(js.formatTail) == 0 {
			js.formatTail = formatTail
		} else if bytes.Compare(js.formatTail, formatTail) != 0 {
			return nil, errors.New("conflicting format tail")
		}

		header := jsHeader{
			Algorithm: signatureBlock.Header.Algorithm,
			Chain:     signatureBlock.Header.Chain,
		}
		if signatureBlock.Header.JWK != nil {
			publicKey, err := UnmarshalPublicKeyJWK([]byte(signatureBlock.Header.JWK))
			if err != nil {
				return nil, fmt.Errorf("error unmarshalling public key: %s", err)
			}
			header.JWK = publicKey
		}
		js.signatures[i] = jsSignature{
			Header:    header,
			Signature: signatureBlock.Signature,
			Protected: signatureBlock.Protected,
		}
	}
	if js.formatLength > len(content) {
		return nil, errors.New("invalid format length")
	}
	formatted := make([]byte, js.formatLength+len(js.formatTail))
	copy(formatted, content[:js.formatLength])
	copy(formatted[js.formatLength:], js.formatTail)
	js.indent = detectJSONIndent(formatted)
	js.payload = joseBase64UrlEncode(formatted)

	return js, nil
}

// PrettySignature formats a json signature into an easy to read
// single json serialized object.
func (js *JSONSignature) PrettySignature(signatureKey string) ([]byte, error) {
	if len(js.signatures) == 0 {
		return nil, errors.New("no signatures")
	}
	payload, err := joseBase64UrlDecode(js.payload)
	if err != nil {
		return nil, err
	}
	payload = payload[:js.formatLength]

	sort.Sort(jsSignaturesSorted(js.signatures))

	var marshalled []byte
	var marshallErr error
	if js.indent != "" {
		marshalled, marshallErr = json.MarshalIndent(js.signatures, js.indent, js.indent)
	} else {
		marshalled, marshallErr = json.Marshal(js.signatures)
	}
	if marshallErr != nil {
		return nil, marshallErr
	}

	buf := bytes.NewBuffer(make([]byte, 0, len(payload)+len(marshalled)+34))
	buf.Write(payload)
	buf.WriteByte(',')
	if js.indent != "" {
		buf.WriteByte('\n')
		buf.WriteString(js.indent)
		buf.WriteByte('"')
		buf.WriteString(signatureKey)
		buf.WriteString("\": ")
		buf.Write(marshalled)
		buf.WriteByte('\n')
	} else {
		buf.WriteByte('"')
		buf.WriteString(signatureKey)
		buf.WriteString("\":")
		buf.Write(marshalled)
	}
	buf.WriteByte('}')

	return buf.Bytes(), nil
}

// Signatures provides the signatures on this JWS as opaque blobs, sorted by
// keyID. These blobs can be stored and reassembled with payloads. Internally,
// they are simply marshaled json web signatures but implementations should
// not rely on this.
func (js *JSONSignature) Signatures() ([][]byte, error) {
	sort.Sort(jsSignaturesSorted(js.signatures))

	var sb [][]byte
	for _, jsig := range js.signatures {
		p, err := json.Marshal(jsig)
		if err != nil {
			return nil, err
		}

		sb = append(sb, p)
	}

	return sb, nil
}

// Merge combines the signatures from one or more other signatures into the
// method receiver. If the payloads differ for any argument, an error will be
// returned and the receiver will not be modified.
func (js *JSONSignature) Merge(others ...*JSONSignature) error {
	merged := js.signatures
	for _, other := range others {
		if js.payload != other.payload {
			return fmt.Errorf("payloads differ from merge target")
		}
		merged = append(merged, other.signatures...)
	}

	js.signatures = merged
	return nil
}
