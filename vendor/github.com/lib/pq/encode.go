package pq

import (
	"bytes"
	"database/sql/driver"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/lib/pq/oid"
)

func binaryEncode(parameterStatus *parameterStatus, x interface{}) []byte {
	switch v := x.(type) {
	case []byte:
		return v
	default:
		return encode(parameterStatus, x, oid.T_unknown)
	}
}

func encode(parameterStatus *parameterStatus, x interface{}, pgtypOid oid.Oid) []byte {
	switch v := x.(type) {
	case int64:
		return strconv.AppendInt(nil, v, 10)
	case float64:
		return strconv.AppendFloat(nil, v, 'f', -1, 64)
	case []byte:
		if pgtypOid == oid.T_bytea {
			return encodeBytea(parameterStatus.serverVersion, v)
		}

		return v
	case string:
		if pgtypOid == oid.T_bytea {
			return encodeBytea(parameterStatus.serverVersion, []byte(v))
		}

		return []byte(v)
	case bool:
		return strconv.AppendBool(nil, v)
	case time.Time:
		return formatTs(v)

	default:
		errorf("encode: unknown type for %T", v)
	}

	panic("not reached")
}

func decode(parameterStatus *parameterStatus, s []byte, typ oid.Oid, f format) interface{} {
	switch f {
	case formatBinary:
		return binaryDecode(parameterStatus, s, typ)
	case formatText:
		return textDecode(parameterStatus, s, typ)
	default:
		panic("not reached")
	}
}

func binaryDecode(parameterStatus *parameterStatus, s []byte, typ oid.Oid) interface{} {
	switch typ {
	case oid.T_bytea:
		return s
	case oid.T_int8:
		return int64(binary.BigEndian.Uint64(s))
	case oid.T_int4:
		return int64(int32(binary.BigEndian.Uint32(s)))
	case oid.T_int2:
		return int64(int16(binary.BigEndian.Uint16(s)))
	case oid.T_uuid:
		b, err := decodeUUIDBinary(s)
		if err != nil {
			panic(err)
		}
		return b

	default:
		errorf("don't know how to decode binary parameter of type %d", uint32(typ))
	}

	panic("not reached")
}

func textDecode(parameterStatus *parameterStatus, s []byte, typ oid.Oid) interface{} {
	switch typ {
	case oid.T_char, oid.T_varchar, oid.T_text:
		return string(s)
	case oid.T_bytea:
		b, err := parseBytea(s)
		if err != nil {
			errorf("%s", err)
		}
		return b
	case oid.T_timestamptz:
		return parseTs(parameterStatus.currentLocation, string(s))
	case oid.T_timestamp, oid.T_date:
		return parseTs(nil, string(s))
	case oid.T_time:
		return mustParse("15:04:05", typ, s)
	case oid.T_timetz:
		return mustParse("15:04:05-07", typ, s)
	case oid.T_bool:
		return s[0] == 't'
	case oid.T_int8, oid.T_int4, oid.T_int2:
		i, err := strconv.ParseInt(string(s), 10, 64)
		if err != nil {
			errorf("%s", err)
		}
		return i
	case oid.T_float4, oid.T_float8:
		bits := 64
		if typ == oid.T_float4 {
			bits = 32
		}
		f, err := strconv.ParseFloat(string(s), bits)
		if err != nil {
			errorf("%s", err)
		}
		return f
	}

	return s
}

// appendEncodedText encodes item in text format as required by COPY
// and appends to buf
func appendEncodedText(parameterStatus *parameterStatus, buf []byte, x interface{}) []byte {
	switch v := x.(type) {
	case int64:
		return strconv.AppendInt(buf, v, 10)
	case float64:
		return strconv.AppendFloat(buf, v, 'f', -1, 64)
	case []byte:
		encodedBytea := encodeBytea(parameterStatus.serverVersion, v)
		return appendEscapedText(buf, string(encodedBytea))
	case string:
		return appendEscapedText(buf, v)
	case bool:
		return strconv.AppendBool(buf, v)
	case time.Time:
		return append(buf, formatTs(v)...)
	case nil:
		return append(buf, "\\N"...)
	default:
		errorf("encode: unknown type for %T", v)
	}

	panic("not reached")
}

func appendEscapedText(buf []byte, text string) []byte {
	escapeNeeded := false
	startPos := 0
	var c byte

	// check if we need to escape
	for i := 0; i < len(text); i++ {
		c = text[i]
		if c == '\\' || c == '\n' || c == '\r' || c == '\t' {
			escapeNeeded = true
			startPos = i
			break
		}
	}
	if !escapeNeeded {
		return append(buf, text...)
	}

	// copy till first char to escape, iterate the rest
	result := append(buf, text[:startPos]...)
	for i := startPos; i < len(text); i++ {
		c = text[i]
		switch c {
		case '\\':
			result = append(result, '\\', '\\')
		case '\n':
			result = append(result, '\\', 'n')
		case '\r':
			result = append(result, '\\', 'r')
		case '\t':
			result = append(result, '\\', 't')
		default:
			result = append(result, c)
		}
	}
	return result
}

func mustParse(f string, typ oid.Oid, s []byte) time.Time {
	str := string(s)

	// check for a 30-minute-offset timezone
	if (typ == oid.T_timestamptz || typ == oid.T_timetz) &&
		str[len(str)-3] == ':' {
		f += ":00"
	}
	t, err := time.Parse(f, str)
	if err != nil {
		errorf("decode: %s", err)
	}
	return t
}

var errInvalidTimestamp = errors.New("invalid timestamp")

type timestampParser struct {
	err error
}

func (p *timestampParser) expect(str string, char byte, pos int) {
	if p.err != nil {
		return
	}
	if pos+1 > len(str) {
		p.err = errInvalidTimestamp
		return
	}
	if c := str[pos]; c != char && p.err == nil {
		p.err = fmt.Errorf("expected '%v' at position %v; got '%v'", char, pos, c)
	}
}

func (p *timestampParser) mustAtoi(str string, begin int, end int) int {
	if p.err != nil {
		return 0
	}
	if begin < 0 || end < 0 || begin > end || end > len(str) {
		p.err = errInvalidTimestamp
		return 0
	}
	result, err := strconv.Atoi(str[begin:end])
	if err != nil {
		if p.err == nil {
			p.err = fmt.Errorf("expected number; got '%v'", str)
		}
		return 0
	}
	return result
}

// The location cache caches the time zones typically used by the client.
type locationCache struct {
	cache map[int]*time.Location
	lock  sync.Mutex
}

// All connections share the same list of timezones. Benchmarking shows that
// about 5% speed could be gained by putting the cache in the connection and
// losing the mutex, at the cost of a small amount of memory and a somewhat
// significant increase in code complexity.
var globalLocationCache = newLocationCache()

func newLocationCache() *locationCache {
	return &locationCache{cache: make(map[int]*time.Location)}
}

// Returns the cached timezone for the specified offset, creating and caching
// it if necessary.
func (c *locationCache) getLocation(offset int) *time.Location {
	c.lock.Lock()
	defer c.lock.Unlock()

	location, ok := c.cache[offset]
	if !ok {
		location = time.FixedZone("", offset)
		c.cache[offset] = location
	}

	return location
}

var infinityTsEnabled = false
var infinityTsNegative time.Time
var infinityTsPositive time.Time

const (
	infinityTsEnabledAlready        = "pq: infinity timestamp enabled already"
	infinityTsNegativeMustBeSmaller = "pq: infinity timestamp: negative value must be smaller (before) than positive"
)

// EnableInfinityTs controls the handling of Postgres' "-infinity" and
// "infinity" "timestamp"s.
//
// If EnableInfinityTs is not called, "-infinity" and "infinity" will return
// []byte("-infinity") and []byte("infinity") respectively, and potentially
// cause error "sql: Scan error on column index 0: unsupported driver -> Scan
// pair: []uint8 -> *time.Time", when scanning into a time.Time value.
//
// Once EnableInfinityTs has been called, all connections created using this
// driver will decode Postgres' "-infinity" and "infinity" for "timestamp",
// "timestamp with time zone" and "date" types to the predefined minimum and
// maximum times, respectively.  When encoding time.Time values, any time which
// equals or precedes the predefined minimum time will be encoded to
// "-infinity".  Any values at or past the maximum time will similarly be
// encoded to "infinity".
//
// If EnableInfinityTs is called with negative >= positive, it will panic.
// Calling EnableInfinityTs after a connection has been established results in
// undefined behavior.  If EnableInfinityTs is called more than once, it will
// panic.
func EnableInfinityTs(negative time.Time, positive time.Time) {
	if infinityTsEnabled {
		panic(infinityTsEnabledAlready)
	}
	if !negative.Before(positive) {
		panic(infinityTsNegativeMustBeSmaller)
	}
	infinityTsEnabled = true
	infinityTsNegative = negative
	infinityTsPositive = positive
}

/*
 * Testing might want to toggle infinityTsEnabled
 */
func disableInfinityTs() {
	infinityTsEnabled = false
}

// This is a time function specific to the Postgres default DateStyle
// setting ("ISO, MDY"), the only one we currently support. This
// accounts for the discrepancies between the parsing available with
// time.Parse and the Postgres date formatting quirks.
func parseTs(currentLocation *time.Location, str string) interface{} {
	switch str {
	case "-infinity":
		if infinityTsEnabled {
			return infinityTsNegative
		}
		return []byte(str)
	case "infinity":
		if infinityTsEnabled {
			return infinityTsPositive
		}
		return []byte(str)
	}
	t, err := ParseTimestamp(currentLocation, str)
	if err != nil {
		panic(err)
	}
	return t
}

// ParseTimestamp parses Postgres' text format. It returns a time.Time in
// currentLocation iff that time's offset agrees with the offset sent from the
// Postgres server. Otherwise, ParseTimestamp returns a time.Time with the
// fixed offset offset provided by the Postgres server.
func ParseTimestamp(currentLocation *time.Location, str string) (time.Time, error) {
	p := timestampParser{}

	monSep := strings.IndexRune(str, '-')
	// this is Gregorian year, not ISO Year
	// In Gregorian system, the year 1 BC is followed by AD 1
	year := p.mustAtoi(str, 0, monSep)
	daySep := monSep + 3
	month := p.mustAtoi(str, monSep+1, daySep)
	p.expect(str, '-', daySep)
	timeSep := daySep + 3
	day := p.mustAtoi(str, daySep+1, timeSep)

	minLen := monSep + len("01-01") + 1

	isBC := strings.HasSuffix(str, " BC")
	if isBC {
		minLen += 3
	}

	var hour, minute, second int
	if len(str) > minLen {
		p.expect(str, ' ', timeSep)
		minSep := timeSep + 3
		p.expect(str, ':', minSep)
		hour = p.mustAtoi(str, timeSep+1, minSep)
		secSep := minSep + 3
		p.expect(str, ':', secSep)
		minute = p.mustAtoi(str, minSep+1, secSep)
		secEnd := secSep + 3
		second = p.mustAtoi(str, secSep+1, secEnd)
	}
	remainderIdx := monSep + len("01-01 00:00:00") + 1
	// Three optional (but ordered) sections follow: the
	// fractional seconds, the time zone offset, and the BC
	// designation. We set them up here and adjust the other
	// offsets if the preceding sections exist.

	nanoSec := 0
	tzOff := 0

	if remainderIdx < len(str) && str[remainderIdx] == '.' {
		fracStart := remainderIdx + 1
		fracOff := strings.IndexAny(str[fracStart:], "-+ ")
		if fracOff < 0 {
			fracOff = len(str) - fracStart
		}
		fracSec := p.mustAtoi(str, fracStart, fracStart+fracOff)
		nanoSec = fracSec * (1000000000 / int(math.Pow(10, float64(fracOff))))

		remainderIdx += fracOff + 1
	}
	if tzStart := remainderIdx; tzStart < len(str) && (str[tzStart] == '-' || str[tzStart] == '+') {
		// time zone separator is always '-' or '+' (UTC is +00)
		var tzSign int
		switch c := str[tzStart]; c {
		case '-':
			tzSign = -1
		case '+':
			tzSign = +1
		default:
			return time.Time{}, fmt.Errorf("expected '-' or '+' at position %v; got %v", tzStart, c)
		}
		tzHours := p.mustAtoi(str, tzStart+1, tzStart+3)
		remainderIdx += 3
		var tzMin, tzSec int
		if remainderIdx < len(str) && str[remainderIdx] == ':' {
			tzMin = p.mustAtoi(str, remainderIdx+1, remainderIdx+3)
			remainderIdx += 3
		}
		if remainderIdx < len(str) && str[remainderIdx] == ':' {
			tzSec = p.mustAtoi(str, remainderIdx+1, remainderIdx+3)
			remainderIdx += 3
		}
		tzOff = tzSign * ((tzHours * 60 * 60) + (tzMin * 60) + tzSec)
	}
	var isoYear int

	if isBC {
		isoYear = 1 - year
		remainderIdx += 3
	} else {
		isoYear = year
	}
	if remainderIdx < len(str) {
		return time.Time{}, fmt.Errorf("expected end of input, got %v", str[remainderIdx:])
	}
	t := time.Date(isoYear, time.Month(month), day,
		hour, minute, second, nanoSec,
		globalLocationCache.getLocation(tzOff))

	if currentLocation != nil {
		// Set the location of the returned Time based on the session's
		// TimeZone value, but only if the local time zone database agrees with
		// the remote database on the offset.
		lt := t.In(currentLocation)
		_, newOff := lt.Zone()
		if newOff == tzOff {
			t = lt
		}
	}

	return t, p.err
}

// formatTs formats t into a format postgres understands.
func formatTs(t time.Time) []byte {
	if infinityTsEnabled {
		// t <= -infinity : ! (t > -infinity)
		if !t.After(infinityTsNegative) {
			return []byte("-infinity")
		}
		// t >= infinity : ! (!t < infinity)
		if !t.Before(infinityTsPositive) {
			return []byte("infinity")
		}
	}
	return FormatTimestamp(t)
}

// FormatTimestamp formats t into Postgres' text format for timestamps.
func FormatTimestamp(t time.Time) []byte {
	// Need to send dates before 0001 A.D. with " BC" suffix, instead of the
	// minus sign preferred by Go.
	// Beware, "0000" in ISO is "1 BC", "-0001" is "2 BC" and so on
	bc := false
	if t.Year() <= 0 {
		// flip year sign, and add 1, e.g: "0" will be "1", and "-10" will be "11"
		t = t.AddDate((-t.Year())*2+1, 0, 0)
		bc = true
	}
	b := []byte(t.Format("2006-01-02 15:04:05.999999999Z07:00"))

	_, offset := t.Zone()
	offset = offset % 60
	if offset != 0 {
		// RFC3339Nano already printed the minus sign
		if offset < 0 {
			offset = -offset
		}

		b = append(b, ':')
		if offset < 10 {
			b = append(b, '0')
		}
		b = strconv.AppendInt(b, int64(offset), 10)
	}

	if bc {
		b = append(b, " BC"...)
	}
	return b
}

// Parse a bytea value received from the server.  Both "hex" and the legacy
// "escape" format are supported.
func parseBytea(s []byte) (result []byte, err error) {
	if len(s) >= 2 && bytes.Equal(s[:2], []byte("\\x")) {
		// bytea_output = hex
		s = s[2:] // trim off leading "\\x"
		result = make([]byte, hex.DecodedLen(len(s)))
		_, err := hex.Decode(result, s)
		if err != nil {
			return nil, err
		}
	} else {
		// bytea_output = escape
		for len(s) > 0 {
			if s[0] == '\\' {
				// escaped '\\'
				if len(s) >= 2 && s[1] == '\\' {
					result = append(result, '\\')
					s = s[2:]
					continue
				}

				// '\\' followed by an octal number
				if len(s) < 4 {
					return nil, fmt.Errorf("invalid bytea sequence %v", s)
				}
				r, err := strconv.ParseInt(string(s[1:4]), 8, 9)
				if err != nil {
					return nil, fmt.Errorf("could not parse bytea value: %s", err.Error())
				}
				result = append(result, byte(r))
				s = s[4:]
			} else {
				// We hit an unescaped, raw byte.  Try to read in as many as
				// possible in one go.
				i := bytes.IndexByte(s, '\\')
				if i == -1 {
					result = append(result, s...)
					break
				}
				result = append(result, s[:i]...)
				s = s[i:]
			}
		}
	}

	return result, nil
}

func encodeBytea(serverVersion int, v []byte) (result []byte) {
	if serverVersion >= 90000 {
		// Use the hex format if we know that the server supports it
		result = make([]byte, 2+hex.EncodedLen(len(v)))
		result[0] = '\\'
		result[1] = 'x'
		hex.Encode(result[2:], v)
	} else {
		// .. or resort to "escape"
		for _, b := range v {
			if b == '\\' {
				result = append(result, '\\', '\\')
			} else if b < 0x20 || b > 0x7e {
				result = append(result, []byte(fmt.Sprintf("\\%03o", b))...)
			} else {
				result = append(result, b)
			}
		}
	}

	return result
}

// NullTime represents a time.Time that may be null. NullTime implements the
// sql.Scanner interface so it can be used as a scan destination, similar to
// sql.NullString.
type NullTime struct {
	Time  time.Time
	Valid bool // Valid is true if Time is not NULL
}

// Scan implements the Scanner interface.
func (nt *NullTime) Scan(value interface{}) error {
	nt.Time, nt.Valid = value.(time.Time)
	return nil
}

// Value implements the driver Valuer interface.
func (nt NullTime) Value() (driver.Value, error) {
	if !nt.Valid {
		return nil, nil
	}
	return nt.Time, nil
}
