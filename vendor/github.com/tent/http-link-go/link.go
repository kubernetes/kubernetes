// Package link implements parsing and serialization of Link header values as
// defined in RFC 5988.
package link

import (
	"bytes"
	"errors"
	"sort"
	"unicode"
)

type Link struct {
	URI    string
	Rel    string
	Params map[string]string
}

// Format serializes a slice of Links into a header value. It does not currently
// implement RFC 2231 handling of non-ASCII character encoding and language
// information.
func Format(links []Link) string {
	buf := &bytes.Buffer{}
	for i, link := range links {
		if i > 0 {
			buf.Write([]byte(", "))
		}
		buf.WriteByte('<')
		buf.WriteString(link.URI)
		buf.WriteByte('>')

		writeParam(buf, "rel", link.Rel)

		keys := make([]string, 0, len(link.Params))
		for k := range link.Params {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			writeParam(buf, k, link.Params[k])
		}
	}

	return buf.String()
}

func writeParam(buf *bytes.Buffer, key, value string) {
	buf.Write([]byte("; "))
	buf.WriteString(key)
	buf.Write([]byte(`="`))
	buf.WriteString(value)
	buf.WriteByte('"')
}

// Parse parses a Link header value into a slice of Links. It does not currently
// implement RFC 2231 handling of non-ASCII character encoding and language
// information.
func Parse(l string) ([]Link, error) {
	v := []byte(l)
	v = bytes.TrimSpace(v)
	if len(v) == 0 {
		return nil, nil
	}

	links := make([]Link, 0, 1)
	for len(v) > 0 {
		if v[0] != '<' {
			return nil, errors.New("link: does not start with <")
		}
		lend := bytes.IndexByte(v, '>')
		if lend == -1 {
			return nil, errors.New("link: does not contain ending >")
		}

		params := make(map[string]string)
		link := Link{URI: string(v[1:lend]), Params: params}
		links = append(links, link)

		// trim off parsed url
		v = v[lend+1:]
		if len(v) == 0 {
			break
		}
		v = bytes.TrimLeftFunc(v, unicode.IsSpace)

		for len(v) > 0 {
			if v[0] != ';' && v[0] != ',' {
				return nil, errors.New(`link: expected ";" or "'", got "` + string(v[0:1]) + `"`)
			}
			var next bool
			if v[0] == ',' {
				next = true
			}
			v = bytes.TrimLeftFunc(v[1:], unicode.IsSpace)
			if next || len(v) == 0 {
				break
			}
			var key, value []byte
			key, value, v = consumeParam(v)
			if key == nil || value == nil {
				return nil, errors.New("link: malformed param")
			}
			if k := string(key); k == "rel" {
				if links[len(links)-1].Rel == "" {
					links[len(links)-1].Rel = string(value)
				}
			} else {
				params[k] = string(value)
			}
			v = bytes.TrimLeftFunc(v, unicode.IsSpace)
		}
	}

	return links, nil
}

func isTokenChar(r rune) bool {
	return r > 0x20 && r < 0x7f && r != '"' && r != ',' && r != '=' && r != ';'
}

func isNotTokenChar(r rune) bool { return !isTokenChar(r) }

func consumeToken(v []byte) (token, rest []byte) {
	notPos := bytes.IndexFunc(v, isNotTokenChar)
	if notPos == -1 {
		return v, nil
	}
	if notPos == 0 {
		return nil, v
	}
	return v[0:notPos], v[notPos:]
}

func consumeValue(v []byte) (value, rest []byte) {
	if v[0] != '"' {
		return nil, v
	}

	rest = v[1:]
	buffer := &bytes.Buffer{}
	var nextIsLiteral bool
	for idx, r := range string(rest) {
		switch {
		case nextIsLiteral:
			buffer.WriteRune(r)
			nextIsLiteral = false
		case r == '"':
			return buffer.Bytes(), rest[idx+1:]
		case r == '\\':
			nextIsLiteral = true
		case r != '\r' && r != '\n':
			buffer.WriteRune(r)
		default:
			return nil, v
		}
	}
	return nil, v
}

func consumeParam(v []byte) (param, value, rest []byte) {
	param, rest = consumeToken(v)
	param = bytes.ToLower(param)
	if param == nil {
		return nil, nil, v
	}

	rest = bytes.TrimLeftFunc(rest, unicode.IsSpace)
	if len(rest) == 0 || rest[0] != '=' {
		return nil, nil, v
	}
	rest = rest[1:] // consume equals sign
	rest = bytes.TrimLeftFunc(rest, unicode.IsSpace)
	if len(rest) == 0 {
		return nil, nil, v
	}
	if rest[0] != '"' {
		value, rest = consumeToken(rest)
	} else {
		value, rest = consumeValue(rest)
	}
	if value == nil {
		return nil, nil, v
	}
	return param, value, rest
}
