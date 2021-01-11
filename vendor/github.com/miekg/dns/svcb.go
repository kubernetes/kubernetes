package dns

import (
	"bytes"
	"encoding/binary"
	"errors"
	"net"
	"sort"
	"strconv"
	"strings"
)

type SVCBKey uint16

// Keys defined in draft-ietf-dnsop-svcb-https-01 Section 12.3.2.
const (
	SVCB_MANDATORY       SVCBKey = 0
	SVCB_ALPN            SVCBKey = 1
	SVCB_NO_DEFAULT_ALPN SVCBKey = 2
	SVCB_PORT            SVCBKey = 3
	SVCB_IPV4HINT        SVCBKey = 4
	SVCB_ECHCONFIG       SVCBKey = 5
	SVCB_IPV6HINT        SVCBKey = 6
	svcb_RESERVED        SVCBKey = 65535
)

var svcbKeyToStringMap = map[SVCBKey]string{
	SVCB_MANDATORY:       "mandatory",
	SVCB_ALPN:            "alpn",
	SVCB_NO_DEFAULT_ALPN: "no-default-alpn",
	SVCB_PORT:            "port",
	SVCB_IPV4HINT:        "ipv4hint",
	SVCB_ECHCONFIG:       "echconfig",
	SVCB_IPV6HINT:        "ipv6hint",
}

var svcbStringToKeyMap = reverseSVCBKeyMap(svcbKeyToStringMap)

func reverseSVCBKeyMap(m map[SVCBKey]string) map[string]SVCBKey {
	n := make(map[string]SVCBKey, len(m))
	for u, s := range m {
		n[s] = u
	}
	return n
}

// String takes the numerical code of an SVCB key and returns its name.
// Returns an empty string for reserved keys.
// Accepts unassigned keys as well as experimental/private keys.
func (key SVCBKey) String() string {
	if x := svcbKeyToStringMap[key]; x != "" {
		return x
	}
	if key == svcb_RESERVED {
		return ""
	}
	return "key" + strconv.FormatUint(uint64(key), 10)
}

// svcbStringToKey returns the numerical code of an SVCB key.
// Returns svcb_RESERVED for reserved/invalid keys.
// Accepts unassigned keys as well as experimental/private keys.
func svcbStringToKey(s string) SVCBKey {
	if strings.HasPrefix(s, "key") {
		a, err := strconv.ParseUint(s[3:], 10, 16)
		// no leading zeros
		// key shouldn't be registered
		if err != nil || a == 65535 || s[3] == '0' || svcbKeyToStringMap[SVCBKey(a)] != "" {
			return svcb_RESERVED
		}
		return SVCBKey(a)
	}
	if key, ok := svcbStringToKeyMap[s]; ok {
		return key
	}
	return svcb_RESERVED
}

func (rr *SVCB) parse(c *zlexer, o string) *ParseError {
	l, _ := c.Next()
	i, e := strconv.ParseUint(l.token, 10, 16)
	if e != nil || l.err {
		return &ParseError{l.token, "bad SVCB priority", l}
	}
	rr.Priority = uint16(i)

	c.Next()        // zBlank
	l, _ = c.Next() // zString
	rr.Target = l.token

	name, nameOk := toAbsoluteName(l.token, o)
	if l.err || !nameOk {
		return &ParseError{l.token, "bad SVCB Target", l}
	}
	rr.Target = name

	// Values (if any)
	l, _ = c.Next()
	var xs []SVCBKeyValue
	// Helps require whitespace between pairs.
	// Prevents key1000="a"key1001=...
	canHaveNextKey := true
	for l.value != zNewline && l.value != zEOF {
		switch l.value {
		case zString:
			if !canHaveNextKey {
				// The key we can now read was probably meant to be
				// a part of the last value.
				return &ParseError{l.token, "bad SVCB value quotation", l}
			}

			// In key=value pairs, value does not have to be quoted unless value
			// contains whitespace. And keys don't need to have values.
			// Similarly, keys with an equality signs after them don't need values.
			// l.token includes at least up to the first equality sign.
			idx := strings.IndexByte(l.token, '=')
			var key, value string
			if idx < 0 {
				// Key with no value and no equality sign
				key = l.token
			} else if idx == 0 {
				return &ParseError{l.token, "bad SVCB key", l}
			} else {
				key, value = l.token[:idx], l.token[idx+1:]

				if value == "" {
					// We have a key and an equality sign. Maybe we have nothing
					// after "=" or we have a double quote.
					l, _ = c.Next()
					if l.value == zQuote {
						// Only needed when value ends with double quotes.
						// Any value starting with zQuote ends with it.
						canHaveNextKey = false

						l, _ = c.Next()
						switch l.value {
						case zString:
							// We have a value in double quotes.
							value = l.token
							l, _ = c.Next()
							if l.value != zQuote {
								return &ParseError{l.token, "SVCB unterminated value", l}
							}
						case zQuote:
							// There's nothing in double quotes.
						default:
							return &ParseError{l.token, "bad SVCB value", l}
						}
					}
				}
			}
			kv := makeSVCBKeyValue(svcbStringToKey(key))
			if kv == nil {
				return &ParseError{l.token, "bad SVCB key", l}
			}
			if err := kv.parse(value); err != nil {
				return &ParseError{l.token, err.Error(), l}
			}
			xs = append(xs, kv)
		case zQuote:
			return &ParseError{l.token, "SVCB key can't contain double quotes", l}
		case zBlank:
			canHaveNextKey = true
		default:
			return &ParseError{l.token, "bad SVCB values", l}
		}
		l, _ = c.Next()
	}
	rr.Value = xs
	if rr.Priority == 0 && len(xs) > 0 {
		return &ParseError{l.token, "SVCB aliasform can't have values", l}
	}
	return nil
}

// makeSVCBKeyValue returns an SVCBKeyValue struct with the key or nil for reserved keys.
func makeSVCBKeyValue(key SVCBKey) SVCBKeyValue {
	switch key {
	case SVCB_MANDATORY:
		return new(SVCBMandatory)
	case SVCB_ALPN:
		return new(SVCBAlpn)
	case SVCB_NO_DEFAULT_ALPN:
		return new(SVCBNoDefaultAlpn)
	case SVCB_PORT:
		return new(SVCBPort)
	case SVCB_IPV4HINT:
		return new(SVCBIPv4Hint)
	case SVCB_ECHCONFIG:
		return new(SVCBECHConfig)
	case SVCB_IPV6HINT:
		return new(SVCBIPv6Hint)
	case svcb_RESERVED:
		return nil
	default:
		e := new(SVCBLocal)
		e.KeyCode = key
		return e
	}
}

// SVCB RR. See RFC xxxx (https://tools.ietf.org/html/draft-ietf-dnsop-svcb-https-01).
type SVCB struct {
	Hdr      RR_Header
	Priority uint16
	Target   string         `dns:"domain-name"`
	Value    []SVCBKeyValue `dns:"pairs"` // Value must be empty if Priority is non-zero.
}

// HTTPS RR. Everything valid for SVCB applies to HTTPS as well.
// Except that the HTTPS record is intended for use with the HTTP and HTTPS protocols.
type HTTPS struct {
	SVCB
}

func (rr *HTTPS) String() string {
	return rr.SVCB.String()
}

func (rr *HTTPS) parse(c *zlexer, o string) *ParseError {
	return rr.SVCB.parse(c, o)
}

// SVCBKeyValue defines a key=value pair for the SVCB RR type.
// An SVCB RR can have multiple SVCBKeyValues appended to it.
type SVCBKeyValue interface {
	Key() SVCBKey          // Key returns the numerical key code.
	pack() ([]byte, error) // pack returns the encoded value.
	unpack([]byte) error   // unpack sets the value.
	String() string        // String returns the string representation of the value.
	parse(string) error    // parse sets the value to the given string representation of the value.
	copy() SVCBKeyValue    // copy returns a deep-copy of the pair.
	len() int              // len returns the length of value in the wire format.
}

// SVCBMandatory pair adds to required keys that must be interpreted for the RR
// to be functional.
// Basic use pattern for creating a mandatory option:
//
//	s := &dns.SVCB{Hdr: dns.RR_Header{Name: ".", Rrtype: dns.TypeSVCB, Class: dns.ClassINET}}
//	e := new(dns.SVCBMandatory)
//	e.Code = []uint16{65403}
//	s.Value = append(s.Value, e)
type SVCBMandatory struct {
	Code []SVCBKey // Must not include mandatory
}

func (*SVCBMandatory) Key() SVCBKey { return SVCB_MANDATORY }

func (s *SVCBMandatory) String() string {
	str := make([]string, len(s.Code))
	for i, e := range s.Code {
		str[i] = e.String()
	}
	return strings.Join(str, ",")
}

func (s *SVCBMandatory) pack() ([]byte, error) {
	codes := append([]SVCBKey(nil), s.Code...)
	sort.Slice(codes, func(i, j int) bool {
		return codes[i] < codes[j]
	})
	b := make([]byte, 2*len(codes))
	for i, e := range codes {
		binary.BigEndian.PutUint16(b[2*i:], uint16(e))
	}
	return b, nil
}

func (s *SVCBMandatory) unpack(b []byte) error {
	if len(b)%2 != 0 {
		return errors.New("dns: svcbmandatory: value length is not a multiple of 2")
	}
	codes := make([]SVCBKey, 0, len(b)/2)
	for i := 0; i < len(b); i += 2 {
		// We assume strictly increasing order.
		codes = append(codes, SVCBKey(binary.BigEndian.Uint16(b[i:])))
	}
	s.Code = codes
	return nil
}

func (s *SVCBMandatory) parse(b string) error {
	str := strings.Split(b, ",")
	codes := make([]SVCBKey, 0, len(str))
	for _, e := range str {
		codes = append(codes, svcbStringToKey(e))
	}
	s.Code = codes
	return nil
}

func (s *SVCBMandatory) len() int {
	return 2 * len(s.Code)
}

func (s *SVCBMandatory) copy() SVCBKeyValue {
	return &SVCBMandatory{
		append([]SVCBKey(nil), s.Code...),
	}
}

// SVCBAlpn pair is used to list supported connection protocols.
// Protocol ids can be found at:
// https://www.iana.org/assignments/tls-extensiontype-values/tls-extensiontype-values.xhtml#alpn-protocol-ids
// Basic use pattern for creating an alpn option:
//
//	h := new(dns.HTTPS)
//	h.Hdr = dns.RR_Header{Name: ".", Rrtype: dns.TypeHTTPS, Class: dns.ClassINET}
//	e := new(dns.SVCBAlpn)
//	e.Alpn = []string{"h2", "http/1.1"}
//	h.Value = append(o.Value, e)
type SVCBAlpn struct {
	Alpn []string
}

func (*SVCBAlpn) Key() SVCBKey     { return SVCB_ALPN }
func (s *SVCBAlpn) String() string { return strings.Join(s.Alpn, ",") }

func (s *SVCBAlpn) pack() ([]byte, error) {
	// Liberally estimate the size of an alpn as 10 octets
	b := make([]byte, 0, 10*len(s.Alpn))
	for _, e := range s.Alpn {
		if len(e) == 0 {
			return nil, errors.New("dns: svcbalpn: empty alpn-id")
		}
		if len(e) > 255 {
			return nil, errors.New("dns: svcbalpn: alpn-id too long")
		}
		b = append(b, byte(len(e)))
		b = append(b, e...)
	}
	return b, nil
}

func (s *SVCBAlpn) unpack(b []byte) error {
	// Estimate the size of the smallest alpn as 4 bytes
	alpn := make([]string, 0, len(b)/4)
	for i := 0; i < len(b); {
		length := int(b[i])
		i++
		if i+length > len(b) {
			return errors.New("dns: svcbalpn: alpn array overflowing")
		}
		alpn = append(alpn, string(b[i:i+length]))
		i += length
	}
	s.Alpn = alpn
	return nil
}

func (s *SVCBAlpn) parse(b string) error {
	s.Alpn = strings.Split(b, ",")
	return nil
}

func (s *SVCBAlpn) len() int {
	var l int
	for _, e := range s.Alpn {
		l += 1 + len(e)
	}
	return l
}

func (s *SVCBAlpn) copy() SVCBKeyValue {
	return &SVCBAlpn{
		append([]string(nil), s.Alpn...),
	}
}

// SVCBNoDefaultAlpn pair signifies no support for default connection protocols.
// Basic use pattern for creating a no-default-alpn option:
//
//	s := &dns.SVCB{Hdr: dns.RR_Header{Name: ".", Rrtype: dns.TypeSVCB, Class: dns.ClassINET}}
//	e := new(dns.SVCBNoDefaultAlpn)
//	s.Value = append(s.Value, e)
type SVCBNoDefaultAlpn struct{}

func (*SVCBNoDefaultAlpn) Key() SVCBKey          { return SVCB_NO_DEFAULT_ALPN }
func (*SVCBNoDefaultAlpn) copy() SVCBKeyValue    { return &SVCBNoDefaultAlpn{} }
func (*SVCBNoDefaultAlpn) pack() ([]byte, error) { return []byte{}, nil }
func (*SVCBNoDefaultAlpn) String() string        { return "" }
func (*SVCBNoDefaultAlpn) len() int              { return 0 }

func (*SVCBNoDefaultAlpn) unpack(b []byte) error {
	if len(b) != 0 {
		return errors.New("dns: svcbnodefaultalpn: no_default_alpn must have no value")
	}
	return nil
}

func (*SVCBNoDefaultAlpn) parse(b string) error {
	if len(b) != 0 {
		return errors.New("dns: svcbnodefaultalpn: no_default_alpn must have no value")
	}
	return nil
}

// SVCBPort pair defines the port for connection.
// Basic use pattern for creating a port option:
//
//	s := &dns.SVCB{Hdr: dns.RR_Header{Name: ".", Rrtype: dns.TypeSVCB, Class: dns.ClassINET}}
//	e := new(dns.SVCBPort)
//	e.Port = 80
//	s.Value = append(s.Value, e)
type SVCBPort struct {
	Port uint16
}

func (*SVCBPort) Key() SVCBKey         { return SVCB_PORT }
func (*SVCBPort) len() int             { return 2 }
func (s *SVCBPort) String() string     { return strconv.FormatUint(uint64(s.Port), 10) }
func (s *SVCBPort) copy() SVCBKeyValue { return &SVCBPort{s.Port} }

func (s *SVCBPort) unpack(b []byte) error {
	if len(b) != 2 {
		return errors.New("dns: svcbport: port length is not exactly 2 octets")
	}
	s.Port = binary.BigEndian.Uint16(b)
	return nil
}

func (s *SVCBPort) pack() ([]byte, error) {
	b := make([]byte, 2)
	binary.BigEndian.PutUint16(b, s.Port)
	return b, nil
}

func (s *SVCBPort) parse(b string) error {
	port, err := strconv.ParseUint(b, 10, 16)
	if err != nil {
		return errors.New("dns: svcbport: port out of range")
	}
	s.Port = uint16(port)
	return nil
}

// SVCBIPv4Hint pair suggests an IPv4 address which may be used to open connections
// if A and AAAA record responses for SVCB's Target domain haven't been received.
// In that case, optionally, A and AAAA requests can be made, after which the connection
// to the hinted IP address may be terminated and a new connection may be opened.
// Basic use pattern for creating an ipv4hint option:
//
//	h := new(dns.HTTPS)
//	h.Hdr = dns.RR_Header{Name: ".", Rrtype: dns.TypeHTTPS, Class: dns.ClassINET}
//	e := new(dns.SVCBIPv4Hint)
//	e.Hint = []net.IP{net.IPv4(1,1,1,1).To4()}
//
//  Or
//
//	e.Hint = []net.IP{net.ParseIP("1.1.1.1").To4()}
//	h.Value = append(h.Value, e)
type SVCBIPv4Hint struct {
	Hint []net.IP
}

func (*SVCBIPv4Hint) Key() SVCBKey { return SVCB_IPV4HINT }
func (s *SVCBIPv4Hint) len() int   { return 4 * len(s.Hint) }

func (s *SVCBIPv4Hint) pack() ([]byte, error) {
	b := make([]byte, 0, 4*len(s.Hint))
	for _, e := range s.Hint {
		x := e.To4()
		if x == nil {
			return nil, errors.New("dns: svcbipv4hint: expected ipv4, hint is ipv6")
		}
		b = append(b, x...)
	}
	return b, nil
}

func (s *SVCBIPv4Hint) unpack(b []byte) error {
	if len(b) == 0 || len(b)%4 != 0 {
		return errors.New("dns: svcbipv4hint: ipv4 address byte array length is not a multiple of 4")
	}
	x := make([]net.IP, 0, len(b)/4)
	for i := 0; i < len(b); i += 4 {
		x = append(x, net.IP(b[i:i+4]))
	}
	s.Hint = x
	return nil
}

func (s *SVCBIPv4Hint) String() string {
	str := make([]string, len(s.Hint))
	for i, e := range s.Hint {
		x := e.To4()
		if x == nil {
			return "<nil>"
		}
		str[i] = x.String()
	}
	return strings.Join(str, ",")
}

func (s *SVCBIPv4Hint) parse(b string) error {
	if strings.Contains(b, ":") {
		return errors.New("dns: svcbipv4hint: expected ipv4, got ipv6")
	}
	str := strings.Split(b, ",")
	dst := make([]net.IP, len(str))
	for i, e := range str {
		ip := net.ParseIP(e).To4()
		if ip == nil {
			return errors.New("dns: svcbipv4hint: bad ip")
		}
		dst[i] = ip
	}
	s.Hint = dst
	return nil
}

func (s *SVCBIPv4Hint) copy() SVCBKeyValue {
	return &SVCBIPv4Hint{
		append([]net.IP(nil), s.Hint...),
	}
}

// SVCBECHConfig pair contains the ECHConfig structure defined in draft-ietf-tls-esni [RFC xxxx].
// Basic use pattern for creating an echconfig option:
//
//	h := new(dns.HTTPS)
//	h.Hdr = dns.RR_Header{Name: ".", Rrtype: dns.TypeHTTPS, Class: dns.ClassINET}
//	e := new(dns.SVCBECHConfig)
//	e.ECH = []byte{0xfe, 0x08, ...}
//	h.Value = append(h.Value, e)
type SVCBECHConfig struct {
	ECH []byte
}

func (*SVCBECHConfig) Key() SVCBKey     { return SVCB_ECHCONFIG }
func (s *SVCBECHConfig) String() string { return toBase64(s.ECH) }
func (s *SVCBECHConfig) len() int       { return len(s.ECH) }

func (s *SVCBECHConfig) pack() ([]byte, error) {
	return append([]byte(nil), s.ECH...), nil
}

func (s *SVCBECHConfig) copy() SVCBKeyValue {
	return &SVCBECHConfig{
		append([]byte(nil), s.ECH...),
	}
}

func (s *SVCBECHConfig) unpack(b []byte) error {
	s.ECH = append([]byte(nil), b...)
	return nil
}
func (s *SVCBECHConfig) parse(b string) error {
	x, err := fromBase64([]byte(b))
	if err != nil {
		return errors.New("dns: svcbechconfig: bad base64 echconfig")
	}
	s.ECH = x
	return nil
}

// SVCBIPv6Hint pair suggests an IPv6 address which may be used to open connections
// if A and AAAA record responses for SVCB's Target domain haven't been received.
// In that case, optionally, A and AAAA requests can be made, after which the
// connection to the hinted IP address may be terminated and a new connection may be opened.
// Basic use pattern for creating an ipv6hint option:
//
//	h := new(dns.HTTPS)
//	h.Hdr = dns.RR_Header{Name: ".", Rrtype: dns.TypeHTTPS, Class: dns.ClassINET}
//	e := new(dns.SVCBIPv6Hint)
//	e.Hint = []net.IP{net.ParseIP("2001:db8::1")}
//	h.Value = append(h.Value, e)
type SVCBIPv6Hint struct {
	Hint []net.IP
}

func (*SVCBIPv6Hint) Key() SVCBKey { return SVCB_IPV6HINT }
func (s *SVCBIPv6Hint) len() int   { return 16 * len(s.Hint) }

func (s *SVCBIPv6Hint) pack() ([]byte, error) {
	b := make([]byte, 0, 16*len(s.Hint))
	for _, e := range s.Hint {
		if len(e) != net.IPv6len || e.To4() != nil {
			return nil, errors.New("dns: svcbipv6hint: expected ipv6, hint is ipv4")
		}
		b = append(b, e...)
	}
	return b, nil
}

func (s *SVCBIPv6Hint) unpack(b []byte) error {
	if len(b) == 0 || len(b)%16 != 0 {
		return errors.New("dns: svcbipv6hint: ipv6 address byte array length not a multiple of 16")
	}
	x := make([]net.IP, 0, len(b)/16)
	for i := 0; i < len(b); i += 16 {
		ip := net.IP(b[i : i+16])
		if ip.To4() != nil {
			return errors.New("dns: svcbipv6hint: expected ipv6, got ipv4")
		}
		x = append(x, ip)
	}
	s.Hint = x
	return nil
}

func (s *SVCBIPv6Hint) String() string {
	str := make([]string, len(s.Hint))
	for i, e := range s.Hint {
		if x := e.To4(); x != nil {
			return "<nil>"
		}
		str[i] = e.String()
	}
	return strings.Join(str, ",")
}

func (s *SVCBIPv6Hint) parse(b string) error {
	if strings.Contains(b, ".") {
		return errors.New("dns: svcbipv6hint: expected ipv6, got ipv4")
	}
	str := strings.Split(b, ",")
	dst := make([]net.IP, len(str))
	for i, e := range str {
		ip := net.ParseIP(e)
		if ip == nil {
			return errors.New("dns: svcbipv6hint: bad ip")
		}
		dst[i] = ip
	}
	s.Hint = dst
	return nil
}

func (s *SVCBIPv6Hint) copy() SVCBKeyValue {
	return &SVCBIPv6Hint{
		append([]net.IP(nil), s.Hint...),
	}
}

// SVCBLocal pair is intended for experimental/private use. The key is recommended
// to be in the range [SVCB_PRIVATE_LOWER, SVCB_PRIVATE_UPPER].
// Basic use pattern for creating a keyNNNNN option:
//
//	h := new(dns.HTTPS)
//	h.Hdr = dns.RR_Header{Name: ".", Rrtype: dns.TypeHTTPS, Class: dns.ClassINET}
//	e := new(dns.SVCBLocal)
//	e.KeyCode = 65400
//	e.Data = []byte("abc")
//	h.Value = append(h.Value, e)
type SVCBLocal struct {
	KeyCode SVCBKey // Never 65535 or any assigned keys.
	Data    []byte  // All byte sequences are allowed.
}

func (s *SVCBLocal) Key() SVCBKey          { return s.KeyCode }
func (s *SVCBLocal) pack() ([]byte, error) { return append([]byte(nil), s.Data...), nil }
func (s *SVCBLocal) len() int              { return len(s.Data) }

func (s *SVCBLocal) unpack(b []byte) error {
	s.Data = append([]byte(nil), b...)
	return nil
}

func (s *SVCBLocal) String() string {
	var str strings.Builder
	str.Grow(4 * len(s.Data))
	for _, e := range s.Data {
		if ' ' <= e && e <= '~' {
			switch e {
			case '"', ';', ' ', '\\':
				str.WriteByte('\\')
				str.WriteByte(e)
			default:
				str.WriteByte(e)
			}
		} else {
			str.WriteString(escapeByte(e))
		}
	}
	return str.String()
}

func (s *SVCBLocal) parse(b string) error {
	data := make([]byte, 0, len(b))
	for i := 0; i < len(b); {
		if b[i] != '\\' {
			data = append(data, b[i])
			i++
			continue
		}
		if i+1 == len(b) {
			return errors.New("dns: svcblocal: svcb private/experimental key escape unterminated")
		}
		if isDigit(b[i+1]) {
			if i+3 < len(b) && isDigit(b[i+2]) && isDigit(b[i+3]) {
				a, err := strconv.ParseUint(b[i+1:i+4], 10, 8)
				if err == nil {
					i += 4
					data = append(data, byte(a))
					continue
				}
			}
			return errors.New("dns: svcblocal: svcb private/experimental key bad escaped octet")
		} else {
			data = append(data, b[i+1])
			i += 2
		}
	}
	s.Data = data
	return nil
}

func (s *SVCBLocal) copy() SVCBKeyValue {
	return &SVCBLocal{s.KeyCode,
		append([]byte(nil), s.Data...),
	}
}

func (rr *SVCB) String() string {
	s := rr.Hdr.String() +
		strconv.Itoa(int(rr.Priority)) + " " +
		sprintName(rr.Target)
	for _, e := range rr.Value {
		s += " " + e.Key().String() + "=\"" + e.String() + "\""
	}
	return s
}

// areSVCBPairArraysEqual checks if SVCBKeyValue arrays are equal after sorting their
// copies. arrA and arrB have equal lengths, otherwise zduplicate.go wouldn't call this function.
func areSVCBPairArraysEqual(a []SVCBKeyValue, b []SVCBKeyValue) bool {
	a = append([]SVCBKeyValue(nil), a...)
	b = append([]SVCBKeyValue(nil), b...)
	sort.Slice(a, func(i, j int) bool { return a[i].Key() < a[j].Key() })
	sort.Slice(b, func(i, j int) bool { return b[i].Key() < b[j].Key() })
	for i, e := range a {
		if e.Key() != b[i].Key() {
			return false
		}
		b1, err1 := e.pack()
		b2, err2 := b[i].pack()
		if err1 != nil || err2 != nil || !bytes.Equal(b1, b2) {
			return false
		}
	}
	return true
}
