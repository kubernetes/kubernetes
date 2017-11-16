package escape

import "strings"

var (
	Codes = map[byte][]byte{
		',': []byte(`\,`),
		'"': []byte(`\"`),
		' ': []byte(`\ `),
		'=': []byte(`\=`),
	}

	codesStr = map[string]string{}
)

func init() {
	for k, v := range Codes {
		codesStr[string(k)] = string(v)
	}
}

func UnescapeString(in string) string {
	if strings.IndexByte(in, '\\') == -1 {
		return in
	}

	for b, esc := range codesStr {
		in = strings.Replace(in, esc, b, -1)
	}
	return in
}

func String(in string) string {
	for b, esc := range codesStr {
		in = strings.Replace(in, b, esc, -1)
	}
	return in
}
