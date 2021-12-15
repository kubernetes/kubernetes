package sprig

import (
	"encoding/base32"
	"encoding/base64"
	"fmt"
	"reflect"
	"strconv"
	"strings"
)

func base64encode(v string) string {
	return base64.StdEncoding.EncodeToString([]byte(v))
}

func base64decode(v string) string {
	data, err := base64.StdEncoding.DecodeString(v)
	if err != nil {
		return err.Error()
	}
	return string(data)
}

func base32encode(v string) string {
	return base32.StdEncoding.EncodeToString([]byte(v))
}

func base32decode(v string) string {
	data, err := base32.StdEncoding.DecodeString(v)
	if err != nil {
		return err.Error()
	}
	return string(data)
}

func quote(str ...interface{}) string {
	out := make([]string, 0, len(str))
	for _, s := range str {
		if s != nil {
			out = append(out, fmt.Sprintf("%q", strval(s)))
		}
	}
	return strings.Join(out, " ")
}

func squote(str ...interface{}) string {
	out := make([]string, 0, len(str))
	for _, s := range str {
		if s != nil {
			out = append(out, fmt.Sprintf("'%v'", s))
		}
	}
	return strings.Join(out, " ")
}

func cat(v ...interface{}) string {
	v = removeNilElements(v)
	r := strings.TrimSpace(strings.Repeat("%v ", len(v)))
	return fmt.Sprintf(r, v...)
}

func indent(spaces int, v string) string {
	pad := strings.Repeat(" ", spaces)
	return pad + strings.Replace(v, "\n", "\n"+pad, -1)
}

func nindent(spaces int, v string) string {
	return "\n" + indent(spaces, v)
}

func replace(old, new, src string) string {
	return strings.Replace(src, old, new, -1)
}

func plural(one, many string, count int) string {
	if count == 1 {
		return one
	}
	return many
}

func strslice(v interface{}) []string {
	switch v := v.(type) {
	case []string:
		return v
	case []interface{}:
		b := make([]string, 0, len(v))
		for _, s := range v {
			if s != nil {
				b = append(b, strval(s))
			}
		}
		return b
	default:
		val := reflect.ValueOf(v)
		switch val.Kind() {
		case reflect.Array, reflect.Slice:
			l := val.Len()
			b := make([]string, 0, l)
			for i := 0; i < l; i++ {
				value := val.Index(i).Interface()
				if value != nil {
					b = append(b, strval(value))
				}
			}
			return b
		default:
			if v == nil {
				return []string{}
			}

			return []string{strval(v)}
		}
	}
}

func removeNilElements(v []interface{}) []interface{} {
	newSlice := make([]interface{}, 0, len(v))
	for _, i := range v {
		if i != nil {
			newSlice = append(newSlice, i)
		}
	}
	return newSlice
}

func strval(v interface{}) string {
	switch v := v.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	case error:
		return v.Error()
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

func trunc(c int, s string) string {
	if c < 0 && len(s)+c > 0 {
		return s[len(s)+c:]
	}
	if c >= 0 && len(s) > c {
		return s[:c]
	}
	return s
}

func join(sep string, v interface{}) string {
	return strings.Join(strslice(v), sep)
}

func split(sep, orig string) map[string]string {
	parts := strings.Split(orig, sep)
	res := make(map[string]string, len(parts))
	for i, v := range parts {
		res["_"+strconv.Itoa(i)] = v
	}
	return res
}

func splitn(sep string, n int, orig string) map[string]string {
	parts := strings.SplitN(orig, sep, n)
	res := make(map[string]string, len(parts))
	for i, v := range parts {
		res["_"+strconv.Itoa(i)] = v
	}
	return res
}

// substring creates a substring of the given string.
//
// If start is < 0, this calls string[:end].
//
// If start is >= 0 and end < 0 or end bigger than s length, this calls string[start:]
//
// Otherwise, this calls string[start, end].
func substring(start, end int, s string) string {
	if start < 0 {
		return s[:end]
	}
	if end < 0 || end > len(s) {
		return s[start:]
	}
	return s[start:end]
}
