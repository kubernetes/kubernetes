package ldap

import (
	"bytes"
	"crypto/md5"
	enchex "encoding/hex"
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"strings"

	"github.com/Azure/go-ntlmssp"
	ber "github.com/go-asn1-ber/asn1-ber"
)

// SimpleBindRequest represents a username/password bind operation
type SimpleBindRequest struct {
	// Username is the name of the Directory object that the client wishes to bind as
	Username string
	// Password is the credentials to bind with
	Password string
	// Controls are optional controls to send with the bind request
	Controls []Control
	// AllowEmptyPassword sets whether the client allows binding with an empty password
	// (normally used for unauthenticated bind).
	AllowEmptyPassword bool
}

// SimpleBindResult contains the response from the server
type SimpleBindResult struct {
	Controls []Control
}

// NewSimpleBindRequest returns a bind request
func NewSimpleBindRequest(username string, password string, controls []Control) *SimpleBindRequest {
	return &SimpleBindRequest{
		Username:           username,
		Password:           password,
		Controls:           controls,
		AllowEmptyPassword: false,
	}
}

func (req *SimpleBindRequest) appendTo(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	pkt.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, req.Username, "User Name"))
	pkt.AppendChild(ber.NewString(ber.ClassContext, ber.TypePrimitive, 0, req.Password, "Password"))

	envelope.AppendChild(pkt)
	if len(req.Controls) > 0 {
		envelope.AppendChild(encodeControls(req.Controls))
	}

	return nil
}

// SimpleBind performs the simple bind operation defined in the given request
func (l *Conn) SimpleBind(simpleBindRequest *SimpleBindRequest) (*SimpleBindResult, error) {
	if simpleBindRequest.Password == "" && !simpleBindRequest.AllowEmptyPassword {
		return nil, NewError(ErrorEmptyPassword, errors.New("ldap: empty password not allowed by the client"))
	}

	msgCtx, err := l.doRequest(simpleBindRequest)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return nil, err
	}

	result := &SimpleBindResult{
		Controls: make([]Control, 0),
	}

	if len(packet.Children) == 3 {
		for _, child := range packet.Children[2].Children {
			decodedChild, decodeErr := DecodeControl(child)
			if decodeErr != nil {
				return nil, fmt.Errorf("failed to decode child control: %s", decodeErr)
			}
			result.Controls = append(result.Controls, decodedChild)
		}
	}

	err = GetLDAPError(packet)
	return result, err
}

// Bind performs a bind with the given username and password.
//
// It does not allow unauthenticated bind (i.e. empty password). Use the UnauthenticatedBind method
// for that.
func (l *Conn) Bind(username, password string) error {
	req := &SimpleBindRequest{
		Username:           username,
		Password:           password,
		AllowEmptyPassword: false,
	}
	_, err := l.SimpleBind(req)
	return err
}

// UnauthenticatedBind performs an unauthenticated bind.
//
// A username may be provided for trace (e.g. logging) purpose only, but it is normally not
// authenticated or otherwise validated by the LDAP server.
//
// See https://tools.ietf.org/html/rfc4513#section-5.1.2 .
// See https://tools.ietf.org/html/rfc4513#section-6.3.1 .
func (l *Conn) UnauthenticatedBind(username string) error {
	req := &SimpleBindRequest{
		Username:           username,
		Password:           "",
		AllowEmptyPassword: true,
	}
	_, err := l.SimpleBind(req)
	return err
}

// DigestMD5BindRequest represents a digest-md5 bind operation
type DigestMD5BindRequest struct {
	Host string
	// Username is the name of the Directory object that the client wishes to bind as
	Username string
	// Password is the credentials to bind with
	Password string
	// Controls are optional controls to send with the bind request
	Controls []Control
}

func (req *DigestMD5BindRequest) appendTo(envelope *ber.Packet) error {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "User Name"))

	auth := ber.Encode(ber.ClassContext, ber.TypeConstructed, 3, "", "authentication")
	auth.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "DIGEST-MD5", "SASL Mech"))
	request.AppendChild(auth)
	envelope.AppendChild(request)
	if len(req.Controls) > 0 {
		envelope.AppendChild(encodeControls(req.Controls))
	}
	return nil
}

// DigestMD5BindResult contains the response from the server
type DigestMD5BindResult struct {
	Controls []Control
}

// MD5Bind performs a digest-md5 bind with the given host, username and password.
func (l *Conn) MD5Bind(host, username, password string) error {
	req := &DigestMD5BindRequest{
		Host:     host,
		Username: username,
		Password: password,
	}
	_, err := l.DigestMD5Bind(req)
	return err
}

// DigestMD5Bind performs the digest-md5 bind operation defined in the given request
func (l *Conn) DigestMD5Bind(digestMD5BindRequest *DigestMD5BindRequest) (*DigestMD5BindResult, error) {
	if digestMD5BindRequest.Password == "" {
		return nil, NewError(ErrorEmptyPassword, errors.New("ldap: empty password not allowed by the client"))
	}

	msgCtx, err := l.doRequest(digestMD5BindRequest)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return nil, err
	}
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if l.Debug {
		if err = addLDAPDescriptions(packet); err != nil {
			return nil, err
		}
		ber.PrintPacket(packet)
	}

	result := &DigestMD5BindResult{
		Controls: make([]Control, 0),
	}
	var params map[string]string
	if len(packet.Children) == 2 {
		if len(packet.Children[1].Children) == 4 {
			child := packet.Children[1].Children[0]
			if child.Tag != ber.TagEnumerated {
				return result, GetLDAPError(packet)
			}
			if child.Value.(int64) != 14 {
				return result, GetLDAPError(packet)
			}
			child = packet.Children[1].Children[3]
			if child.Tag != ber.TagObjectDescriptor {
				return result, GetLDAPError(packet)
			}
			if child.Data == nil {
				return result, GetLDAPError(packet)
			}
			data, _ := ioutil.ReadAll(child.Data)
			params, err = parseParams(string(data))
			if err != nil {
				return result, fmt.Errorf("parsing digest-challenge: %s", err)
			}
		}
	}

	if params != nil {
		resp := computeResponse(
			params,
			"ldap/"+strings.ToLower(digestMD5BindRequest.Host),
			digestMD5BindRequest.Username,
			digestMD5BindRequest.Password,
		)
		packet = ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
		packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))

		request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
		request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
		request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "User Name"))

		auth := ber.Encode(ber.ClassContext, ber.TypeConstructed, 3, "", "authentication")
		auth.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "DIGEST-MD5", "SASL Mech"))
		auth.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, resp, "Credentials"))
		request.AppendChild(auth)
		packet.AppendChild(request)
		msgCtx, err = l.sendMessage(packet)
		if err != nil {
			return nil, fmt.Errorf("send message: %s", err)
		}
		defer l.finishMessage(msgCtx)
		packetResponse, ok := <-msgCtx.responses
		if !ok {
			return nil, NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
		}
		packet, err = packetResponse.ReadPacket()
		l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
		if err != nil {
			return nil, fmt.Errorf("read packet: %s", err)
		}
	}

	err = GetLDAPError(packet)
	return result, err
}

func parseParams(str string) (map[string]string, error) {
	m := make(map[string]string)
	var key, value string
	var state int
	for i := 0; i <= len(str); i++ {
		switch state {
		case 0: //reading key
			if i == len(str) {
				return nil, fmt.Errorf("syntax error on %d", i)
			}
			if str[i] != '=' {
				key += string(str[i])
				continue
			}
			state = 1
		case 1: //reading value
			if i == len(str) {
				m[key] = value
				break
			}
			switch str[i] {
			case ',':
				m[key] = value
				state = 0
				key = ""
				value = ""
			case '"':
				if value != "" {
					return nil, fmt.Errorf("syntax error on %d", i)
				}
				state = 2
			default:
				value += string(str[i])
			}
		case 2: //inside quotes
			if i == len(str) {
				return nil, fmt.Errorf("syntax error on %d", i)
			}
			if str[i] != '"' {
				value += string(str[i])
			} else {
				state = 1
			}
		}
	}
	return m, nil
}

func computeResponse(params map[string]string, uri, username, password string) string {
	nc := "00000001"
	qop := "auth"
	cnonce := enchex.EncodeToString(randomBytes(16))
	x := username + ":" + params["realm"] + ":" + password
	y := md5Hash([]byte(x))

	a1 := bytes.NewBuffer(y)
	a1.WriteString(":" + params["nonce"] + ":" + cnonce)
	if len(params["authzid"]) > 0 {
		a1.WriteString(":" + params["authzid"])
	}
	a2 := bytes.NewBuffer([]byte("AUTHENTICATE"))
	a2.WriteString(":" + uri)
	ha1 := enchex.EncodeToString(md5Hash(a1.Bytes()))
	ha2 := enchex.EncodeToString(md5Hash(a2.Bytes()))

	kd := ha1
	kd += ":" + params["nonce"]
	kd += ":" + nc
	kd += ":" + cnonce
	kd += ":" + qop
	kd += ":" + ha2
	resp := enchex.EncodeToString(md5Hash([]byte(kd)))
	return fmt.Sprintf(
		`username="%s",realm="%s",nonce="%s",cnonce="%s",nc=00000001,qop=%s,digest-uri="%s",response=%s`,
		username,
		params["realm"],
		params["nonce"],
		cnonce,
		qop,
		uri,
		resp,
	)
}

func md5Hash(b []byte) []byte {
	hasher := md5.New()
	hasher.Write(b)
	return hasher.Sum(nil)
}

func randomBytes(len int) []byte {
	b := make([]byte, len)
	for i := 0; i < len; i++ {
		b[i] = byte(rand.Intn(256))
	}
	return b
}

var externalBindRequest = requestFunc(func(envelope *ber.Packet) error {
	pkt := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	pkt.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	pkt.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "User Name"))

	saslAuth := ber.Encode(ber.ClassContext, ber.TypeConstructed, 3, "", "authentication")
	saslAuth.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "EXTERNAL", "SASL Mech"))
	saslAuth.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "SASL Cred"))

	pkt.AppendChild(saslAuth)

	envelope.AppendChild(pkt)

	return nil
})

// ExternalBind performs SASL/EXTERNAL authentication.
//
// Use ldap.DialURL("ldapi://") to connect to the Unix socket before ExternalBind.
//
// See https://tools.ietf.org/html/rfc4422#appendix-A
func (l *Conn) ExternalBind() error {
	msgCtx, err := l.doRequest(externalBindRequest)
	if err != nil {
		return err
	}
	defer l.finishMessage(msgCtx)

	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return err
	}

	return GetLDAPError(packet)
}

// NTLMBind performs an NTLMSSP bind leveraging https://github.com/Azure/go-ntlmssp

// NTLMBindRequest represents an NTLMSSP bind operation
type NTLMBindRequest struct {
	// Domain is the AD Domain to authenticate too. If not specified, it will be grabbed from the NTLMSSP Challenge
	Domain string
	// Username is the name of the Directory object that the client wishes to bind as
	Username string
	// Password is the credentials to bind with
	Password string
	// Hash is the hex NTLM hash to bind with. Password or hash must be provided
	Hash string
	// Controls are optional controls to send with the bind request
	Controls []Control
}

func (req *NTLMBindRequest) appendTo(envelope *ber.Packet) error {
	request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
	request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
	request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "User Name"))

	// generate an NTLMSSP Negotiation message for the  specified domain (it can be blank)
	negMessage, err := ntlmssp.NewNegotiateMessage(req.Domain, "")
	if err != nil {
		return fmt.Errorf("err creating negmessage: %s", err)
	}

	// append the generated NTLMSSP message as a TagEnumerated BER value
	auth := ber.Encode(ber.ClassContext, ber.TypePrimitive, ber.TagEnumerated, negMessage, "authentication")
	request.AppendChild(auth)
	envelope.AppendChild(request)
	if len(req.Controls) > 0 {
		envelope.AppendChild(encodeControls(req.Controls))
	}
	return nil
}

// NTLMBindResult contains the response from the server
type NTLMBindResult struct {
	Controls []Control
}

// NTLMBind performs an NTLMSSP Bind with the given domain, username and password
func (l *Conn) NTLMBind(domain, username, password string) error {
	req := &NTLMBindRequest{
		Domain:   domain,
		Username: username,
		Password: password,
	}
	_, err := l.NTLMChallengeBind(req)
	return err
}

// NTLMBindWithHash performs an NTLM Bind with an NTLM hash instead of plaintext password (pass-the-hash)
func (l *Conn) NTLMBindWithHash(domain, username, hash string) error {
	req := &NTLMBindRequest{
		Domain:   domain,
		Username: username,
		Hash:     hash,
	}
	_, err := l.NTLMChallengeBind(req)
	return err
}

// NTLMChallengeBind performs the NTLMSSP bind operation defined in the given request
func (l *Conn) NTLMChallengeBind(ntlmBindRequest *NTLMBindRequest) (*NTLMBindResult, error) {
	if ntlmBindRequest.Password == "" && ntlmBindRequest.Hash == "" {
		return nil, NewError(ErrorEmptyPassword, errors.New("ldap: empty password not allowed by the client"))
	}

	msgCtx, err := l.doRequest(ntlmBindRequest)
	if err != nil {
		return nil, err
	}
	defer l.finishMessage(msgCtx)
	packet, err := l.readPacket(msgCtx)
	if err != nil {
		return nil, err
	}
	l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
	if l.Debug {
		if err = addLDAPDescriptions(packet); err != nil {
			return nil, err
		}
		ber.PrintPacket(packet)
	}
	result := &NTLMBindResult{
		Controls: make([]Control, 0),
	}
	var ntlmsspChallenge []byte

	// now find the NTLM Response Message
	if len(packet.Children) == 2 {
		if len(packet.Children[1].Children) == 3 {
			child := packet.Children[1].Children[1]
			ntlmsspChallenge = child.ByteValue
			// Check to make sure we got the right message. It will always start with NTLMSSP
			if len(ntlmsspChallenge) < 7 || !bytes.Equal(ntlmsspChallenge[:7], []byte("NTLMSSP")) {
				return result, GetLDAPError(packet)
			}
			l.Debug.Printf("%d: found ntlmssp challenge", msgCtx.id)
		}
	}
	if ntlmsspChallenge != nil {
		var err error
		var responseMessage []byte
		// generate a response message to the challenge with the given Username/Password if password is provided
		if ntlmBindRequest.Password != "" {
			responseMessage, err = ntlmssp.ProcessChallenge(ntlmsspChallenge, ntlmBindRequest.Username, ntlmBindRequest.Password)
		} else if ntlmBindRequest.Hash != "" {
			responseMessage, err = ntlmssp.ProcessChallengeWithHash(ntlmsspChallenge, ntlmBindRequest.Username, ntlmBindRequest.Hash)
		} else {
			err = fmt.Errorf("need a password or hash to generate reply")
		}
		if err != nil {
			return result, fmt.Errorf("parsing ntlm-challenge: %s", err)
		}
		packet = ber.Encode(ber.ClassUniversal, ber.TypeConstructed, ber.TagSequence, nil, "LDAP Request")
		packet.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, l.nextMessageID(), "MessageID"))

		request := ber.Encode(ber.ClassApplication, ber.TypeConstructed, ApplicationBindRequest, nil, "Bind Request")
		request.AppendChild(ber.NewInteger(ber.ClassUniversal, ber.TypePrimitive, ber.TagInteger, 3, "Version"))
		request.AppendChild(ber.NewString(ber.ClassUniversal, ber.TypePrimitive, ber.TagOctetString, "", "User Name"))

		// append the challenge response message as a TagEmbeddedPDV BER value
		auth := ber.Encode(ber.ClassContext, ber.TypePrimitive, ber.TagEmbeddedPDV, responseMessage, "authentication")

		request.AppendChild(auth)
		packet.AppendChild(request)
		msgCtx, err = l.sendMessage(packet)
		if err != nil {
			return nil, fmt.Errorf("send message: %s", err)
		}
		defer l.finishMessage(msgCtx)
		packetResponse, ok := <-msgCtx.responses
		if !ok {
			return nil, NewError(ErrorNetwork, errors.New("ldap: response channel closed"))
		}
		packet, err = packetResponse.ReadPacket()
		l.Debug.Printf("%d: got response %p", msgCtx.id, packet)
		if err != nil {
			return nil, fmt.Errorf("read packet: %s", err)
		}

	}

	err = GetLDAPError(packet)
	return result, err
}
