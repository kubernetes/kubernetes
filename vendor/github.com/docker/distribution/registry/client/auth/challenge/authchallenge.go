package challenge

import (
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// Challenge carries information from a WWW-Authenticate response header.
// See RFC 2617.
type Challenge struct {
	// Scheme is the auth-scheme according to RFC 2617
	Scheme string

	// Parameters are the auth-params according to RFC 2617
	Parameters map[string]string
}

// Manager manages the challenges for endpoints.
// The challenges are pulled out of HTTP responses. Only
// responses which expect challenges should be added to
// the manager, since a non-unauthorized request will be
// viewed as not requiring challenges.
type Manager interface {
	// GetChallenges returns the challenges for the given
	// endpoint URL.
	GetChallenges(endpoint url.URL) ([]Challenge, error)

	// AddResponse adds the response to the challenge
	// manager. The challenges will be parsed out of
	// the WWW-Authenicate headers and added to the
	// URL which was produced the response. If the
	// response was authorized, any challenges for the
	// endpoint will be cleared.
	AddResponse(resp *http.Response) error
}

// NewSimpleManager returns an instance of
// Manger which only maps endpoints to challenges
// based on the responses which have been added the
// manager. The simple manager will make no attempt to
// perform requests on the endpoints or cache the responses
// to a backend.
func NewSimpleManager() Manager {
	return &simpleManager{
		Challanges: make(map[string][]Challenge),
	}
}

type simpleManager struct {
	sync.RWMutex
	Challanges map[string][]Challenge
}

func normalizeURL(endpoint *url.URL) {
	endpoint.Host = strings.ToLower(endpoint.Host)
	endpoint.Host = canonicalAddr(endpoint)
}

func (m *simpleManager) GetChallenges(endpoint url.URL) ([]Challenge, error) {
	normalizeURL(&endpoint)

	m.RLock()
	defer m.RUnlock()
	challenges := m.Challanges[endpoint.String()]
	return challenges, nil
}

func (m *simpleManager) AddResponse(resp *http.Response) error {
	challenges := ResponseChallenges(resp)
	if resp.Request == nil {
		return fmt.Errorf("missing request reference")
	}
	urlCopy := url.URL{
		Path:   resp.Request.URL.Path,
		Host:   resp.Request.URL.Host,
		Scheme: resp.Request.URL.Scheme,
	}
	normalizeURL(&urlCopy)

	m.Lock()
	defer m.Unlock()
	m.Challanges[urlCopy.String()] = challenges
	return nil
}

// Octet types from RFC 2616.
type octetType byte

var octetTypes [256]octetType

const (
	isToken octetType = 1 << iota
	isSpace
)

func init() {
	// OCTET      = <any 8-bit sequence of data>
	// CHAR       = <any US-ASCII character (octets 0 - 127)>
	// CTL        = <any US-ASCII control character (octets 0 - 31) and DEL (127)>
	// CR         = <US-ASCII CR, carriage return (13)>
	// LF         = <US-ASCII LF, linefeed (10)>
	// SP         = <US-ASCII SP, space (32)>
	// HT         = <US-ASCII HT, horizontal-tab (9)>
	// <">        = <US-ASCII double-quote mark (34)>
	// CRLF       = CR LF
	// LWS        = [CRLF] 1*( SP | HT )
	// TEXT       = <any OCTET except CTLs, but including LWS>
	// separators = "(" | ")" | "<" | ">" | "@" | "," | ";" | ":" | "\" | <">
	//              | "/" | "[" | "]" | "?" | "=" | "{" | "}" | SP | HT
	// token      = 1*<any CHAR except CTLs or separators>
	// qdtext     = <any TEXT except <">>

	for c := 0; c < 256; c++ {
		var t octetType
		isCtl := c <= 31 || c == 127
		isChar := 0 <= c && c <= 127
		isSeparator := strings.IndexRune(" \t\"(),/:;<=>?@[]\\{}", rune(c)) >= 0
		if strings.IndexRune(" \t\r\n", rune(c)) >= 0 {
			t |= isSpace
		}
		if isChar && !isCtl && !isSeparator {
			t |= isToken
		}
		octetTypes[c] = t
	}
}

// ResponseChallenges returns a list of authorization challenges
// for the given http Response. Challenges are only checked if
// the response status code was a 401.
func ResponseChallenges(resp *http.Response) []Challenge {
	if resp.StatusCode == http.StatusUnauthorized {
		// Parse the WWW-Authenticate Header and store the challenges
		// on this endpoint object.
		return parseAuthHeader(resp.Header)
	}

	return nil
}

func parseAuthHeader(header http.Header) []Challenge {
	challenges := []Challenge{}
	for _, h := range header[http.CanonicalHeaderKey("WWW-Authenticate")] {
		v, p := parseValueAndParams(h)
		if v != "" {
			challenges = append(challenges, Challenge{Scheme: v, Parameters: p})
		}
	}
	return challenges
}

func parseValueAndParams(header string) (value string, params map[string]string) {
	params = make(map[string]string)
	value, s := expectToken(header)
	if value == "" {
		return
	}
	value = strings.ToLower(value)
	s = "," + skipSpace(s)
	for strings.HasPrefix(s, ",") {
		var pkey string
		pkey, s = expectToken(skipSpace(s[1:]))
		if pkey == "" {
			return
		}
		if !strings.HasPrefix(s, "=") {
			return
		}
		var pvalue string
		pvalue, s = expectTokenOrQuoted(s[1:])
		if pvalue == "" {
			return
		}
		pkey = strings.ToLower(pkey)
		params[pkey] = pvalue
		s = skipSpace(s)
	}
	return
}

func skipSpace(s string) (rest string) {
	i := 0
	for ; i < len(s); i++ {
		if octetTypes[s[i]]&isSpace == 0 {
			break
		}
	}
	return s[i:]
}

func expectToken(s string) (token, rest string) {
	i := 0
	for ; i < len(s); i++ {
		if octetTypes[s[i]]&isToken == 0 {
			break
		}
	}
	return s[:i], s[i:]
}

func expectTokenOrQuoted(s string) (value string, rest string) {
	if !strings.HasPrefix(s, "\"") {
		return expectToken(s)
	}
	s = s[1:]
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			return s[:i], s[i+1:]
		case '\\':
			p := make([]byte, len(s)-1)
			j := copy(p, s[:i])
			escape := true
			for i = i + 1; i < len(s); i++ {
				b := s[i]
				switch {
				case escape:
					escape = false
					p[j] = b
					j++
				case b == '\\':
					escape = true
				case b == '"':
					return string(p[:j]), s[i+1:]
				default:
					p[j] = b
					j++
				}
			}
			return "", ""
		}
	}
	return "", ""
}
