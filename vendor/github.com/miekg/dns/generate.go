package dns

import (
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Parse the $GENERATE statement as used in BIND9 zones.
// See http://www.zytrax.com/books/dns/ch8/generate.html for instance.
// We are called after '$GENERATE '. After which we expect:
// * the range (12-24/2)
// * lhs (ownername)
// * [[ttl][class]]
// * type
// * rhs (rdata)
// But we are lazy here, only the range is parsed *all* occurrences
// of $ after that are interpreted.
func (zp *ZoneParser) generate(l lex) (RR, bool) {
	token := l.token
	step := int64(1)
	if i := strings.IndexByte(token, '/'); i >= 0 {
		if i+1 == len(token) {
			return zp.setParseError("bad step in $GENERATE range", l)
		}

		s, err := strconv.ParseInt(token[i+1:], 10, 64)
		if err != nil || s <= 0 {
			return zp.setParseError("bad step in $GENERATE range", l)
		}

		step = s
		token = token[:i]
	}

	sx := strings.SplitN(token, "-", 2)
	if len(sx) != 2 {
		return zp.setParseError("bad start-stop in $GENERATE range", l)
	}

	start, err := strconv.ParseInt(sx[0], 10, 64)
	if err != nil {
		return zp.setParseError("bad start in $GENERATE range", l)
	}

	end, err := strconv.ParseInt(sx[1], 10, 64)
	if err != nil {
		return zp.setParseError("bad stop in $GENERATE range", l)
	}
	if end < 0 || start < 0 || end < start || (end-start)/step > 65535 {
		return zp.setParseError("bad range in $GENERATE range", l)
	}

	// _BLANK
	l, ok := zp.c.Next()
	if !ok || l.value != zBlank {
		return zp.setParseError("garbage after $GENERATE range", l)
	}

	// Create a complete new string, which we then parse again.
	var s string
	for l, ok := zp.c.Next(); ok; l, ok = zp.c.Next() {
		if l.err {
			return zp.setParseError("bad data in $GENERATE directive", l)
		}
		if l.value == zNewline {
			break
		}

		s += l.token
	}

	r := &generateReader{
		s: s,

		cur:   int(start),
		start: int(start),
		end:   int(end),
		step:  int(step),

		file: zp.file,
		lex:  &l,
	}
	zp.sub = NewZoneParser(r, zp.origin, zp.file)
	zp.sub.includeDepth, zp.sub.includeAllowed = zp.includeDepth, zp.includeAllowed
	zp.sub.generateDisallowed = true
	zp.sub.SetDefaultTTL(defaultTtl)
	return zp.subNext()
}

type generateReader struct {
	s  string
	si int

	cur   int
	start int
	end   int
	step  int

	mod bytes.Buffer

	escape bool

	eof bool

	file string
	lex  *lex
}

func (r *generateReader) parseError(msg string, end int) *ParseError {
	r.eof = true // Make errors sticky.

	l := *r.lex
	l.token = r.s[r.si-1 : end]
	l.column += r.si // l.column starts one zBLANK before r.s

	return &ParseError{r.file, msg, l}
}

func (r *generateReader) Read(p []byte) (int, error) {
	// NewZLexer, through NewZoneParser, should use ReadByte and
	// not end up here.

	panic("not implemented")
}

func (r *generateReader) ReadByte() (byte, error) {
	if r.eof {
		return 0, io.EOF
	}
	if r.mod.Len() > 0 {
		return r.mod.ReadByte()
	}

	if r.si >= len(r.s) {
		r.si = 0
		r.cur += r.step

		r.eof = r.cur > r.end || r.cur < 0
		return '\n', nil
	}

	si := r.si
	r.si++

	switch r.s[si] {
	case '\\':
		if r.escape {
			r.escape = false
			return '\\', nil
		}

		r.escape = true
		return r.ReadByte()
	case '$':
		if r.escape {
			r.escape = false
			return '$', nil
		}

		mod := "%d"

		if si >= len(r.s)-1 {
			// End of the string
			fmt.Fprintf(&r.mod, mod, r.cur)
			return r.mod.ReadByte()
		}

		if r.s[si+1] == '$' {
			r.si++
			return '$', nil
		}

		var offset int

		// Search for { and }
		if r.s[si+1] == '{' {
			// Modifier block
			sep := strings.Index(r.s[si+2:], "}")
			if sep < 0 {
				return 0, r.parseError("bad modifier in $GENERATE", len(r.s))
			}

			var errMsg string
			mod, offset, errMsg = modToPrintf(r.s[si+2 : si+2+sep])
			if errMsg != "" {
				return 0, r.parseError(errMsg, si+3+sep)
			}
			if r.start+offset < 0 || int64(r.end) + int64(offset) > 1<<31-1 {
				return 0, r.parseError("bad offset in $GENERATE", si+3+sep)
			}

			r.si += 2 + sep // Jump to it
		}

		fmt.Fprintf(&r.mod, mod, r.cur+offset)
		return r.mod.ReadByte()
	default:
		if r.escape { // Pretty useless here
			r.escape = false
			return r.ReadByte()
		}

		return r.s[si], nil
	}
}

// Convert a $GENERATE modifier 0,0,d to something Printf can deal with.
func modToPrintf(s string) (string, int, string) {
	// Modifier is { offset [ ,width [ ,base ] ] } - provide default
	// values for optional width and type, if necessary.
	var offStr, widthStr, base string
	switch xs := strings.Split(s, ","); len(xs) {
	case 1:
		offStr, widthStr, base = xs[0], "0", "d"
	case 2:
		offStr, widthStr, base = xs[0], xs[1], "d"
	case 3:
		offStr, widthStr, base = xs[0], xs[1], xs[2]
	default:
		return "", 0, "bad modifier in $GENERATE"
	}

	switch base {
	case "o", "d", "x", "X":
	default:
		return "", 0, "bad base in $GENERATE"
	}

	offset, err := strconv.ParseInt(offStr, 10, 64)
	if err != nil {
		return "", 0, "bad offset in $GENERATE"
	}

	width, err := strconv.ParseInt(widthStr, 10, 64)
	if err != nil || width < 0 || width > 255 {
		return "", 0, "bad width in $GENERATE"
	}

	if width == 0 {
		return "%" + base, int(offset), ""
	}

	return "%0" + widthStr + base, int(offset), ""
}
