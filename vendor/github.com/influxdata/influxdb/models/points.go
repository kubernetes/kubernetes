package models

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/influxdata/influxdb/pkg/escape"
)

var (
	measurementEscapeCodes = map[byte][]byte{
		',': []byte(`\,`),
		' ': []byte(`\ `),
	}

	tagEscapeCodes = map[byte][]byte{
		',': []byte(`\,`),
		' ': []byte(`\ `),
		'=': []byte(`\=`),
	}

	ErrPointMustHaveAField  = errors.New("point without fields is unsupported")
	ErrInvalidNumber        = errors.New("invalid number")
	ErrMaxKeyLengthExceeded = errors.New("max key length exceeded")
)

const (
	MaxKeyLength = 65535
)

// Point defines the values that will be written to the database
type Point interface {
	Name() string
	SetName(string)

	Tags() Tags
	AddTag(key, value string)
	SetTags(tags Tags)

	Fields() Fields

	Time() time.Time
	SetTime(t time.Time)
	UnixNano() int64

	HashID() uint64
	Key() []byte

	Data() []byte
	SetData(buf []byte)

	// String returns a string representation of the point, if there is a
	// timestamp associated with the point then it will be specified with the default
	// precision of nanoseconds
	String() string

	// Bytes returns a []byte representation of the point similar to string.
	MarshalBinary() ([]byte, error)

	// PrecisionString returns a string representation of the point, if there
	// is a timestamp associated with the point then it will be specified in the
	// given unit
	PrecisionString(precision string) string

	// RoundedString returns a string representation of the point, if there
	// is a timestamp associated with the point, then it will be rounded to the
	// given duration
	RoundedString(d time.Duration) string
}

// Points represents a sortable list of points by timestamp.
type Points []Point

func (a Points) Len() int           { return len(a) }
func (a Points) Less(i, j int) bool { return a[i].Time().Before(a[j].Time()) }
func (a Points) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// point is the default implementation of Point.
type point struct {
	time time.Time

	// text encoding of measurement and tags
	// key must always be stored sorted by tags, if the original line was not sorted,
	// we need to resort it
	key []byte

	// text encoding of field data
	fields []byte

	// text encoding of timestamp
	ts []byte

	// binary encoded field data
	data []byte

	// cached version of parsed fields from data
	cachedFields map[string]interface{}

	// cached version of parsed name from key
	cachedName string
}

const (
	// the number of characters for the largest possible int64 (9223372036854775807)
	maxInt64Digits = 19

	// the number of characters for the smallest possible int64 (-9223372036854775808)
	minInt64Digits = 20

	// the number of characters required for the largest float64 before a range check
	// would occur during parsing
	maxFloat64Digits = 25

	// the number of characters required for smallest float64 before a range check occur
	// would occur during parsing
	minFloat64Digits = 27
)

// ParsePoints returns a slice of Points from a text representation of a point
// with each point separated by newlines.  If any points fail to parse, a non-nil error
// will be returned in addition to the points that parsed successfully.
func ParsePoints(buf []byte) ([]Point, error) {
	return ParsePointsWithPrecision(buf, time.Now().UTC(), "n")
}

// ParsePointsString is identical to ParsePoints but accepts a string
// buffer.
func ParsePointsString(buf string) ([]Point, error) {
	return ParsePoints([]byte(buf))
}

// ParseKey returns the measurement name and tags from a point.
func ParseKey(buf string) (string, Tags, error) {
	_, keyBuf, err := scanKey([]byte(buf), 0)
	tags := parseTags([]byte(buf))
	return string(keyBuf), tags, err
}

// ParsePointsWithPrecision is similar to ParsePoints, but allows the
// caller to provide a precision for time.
func ParsePointsWithPrecision(buf []byte, defaultTime time.Time, precision string) ([]Point, error) {
	points := []Point{}
	var (
		pos    int
		block  []byte
		failed []string
	)
	for {
		pos, block = scanLine(buf, pos)
		pos++

		if len(block) == 0 {
			break
		}

		// lines which start with '#' are comments
		start := skipWhitespace(block, 0)

		// If line is all whitespace, just skip it
		if start >= len(block) {
			continue
		}

		if block[start] == '#' {
			continue
		}

		// strip the newline if one is present
		if block[len(block)-1] == '\n' {
			block = block[:len(block)-1]
		}

		pt, err := parsePoint(block[start:len(block)], defaultTime, precision)
		if err != nil {
			failed = append(failed, fmt.Sprintf("unable to parse '%s': %v", string(block[start:len(block)]), err))
		} else {
			points = append(points, pt)
		}

		if pos >= len(buf) {
			break
		}

	}
	if len(failed) > 0 {
		return points, fmt.Errorf("%s", strings.Join(failed, "\n"))
	}
	return points, nil

}

func parsePoint(buf []byte, defaultTime time.Time, precision string) (Point, error) {
	// scan the first block which is measurement[,tag1=value1,tag2=value=2...]
	pos, key, err := scanKey(buf, 0)
	if err != nil {
		return nil, err
	}

	// measurement name is required
	if len(key) == 0 {
		return nil, fmt.Errorf("missing measurement")
	}

	if len(key) > MaxKeyLength {
		return nil, fmt.Errorf("max key length exceeded: %v > %v", len(key), MaxKeyLength)
	}

	// scan the second block is which is field1=value1[,field2=value2,...]
	pos, fields, err := scanFields(buf, pos)
	if err != nil {
		return nil, err
	}

	// at least one field is required
	if len(fields) == 0 {
		return nil, fmt.Errorf("missing fields")
	}

	// scan the last block which is an optional integer timestamp
	pos, ts, err := scanTime(buf, pos)

	if err != nil {
		return nil, err
	}

	pt := &point{
		key:    key,
		fields: fields,
		ts:     ts,
	}

	if len(ts) == 0 {
		pt.time = defaultTime
		pt.SetPrecision(precision)
	} else {
		ts, err := strconv.ParseInt(string(ts), 10, 64)
		if err != nil {
			return nil, err
		}
		pt.time, err = SafeCalcTime(ts, precision)
		if err != nil {
			return nil, err
		}
	}
	return pt, nil
}

// GetPrecisionMultiplier will return a multiplier for the precision specified
func GetPrecisionMultiplier(precision string) int64 {
	d := time.Nanosecond
	switch precision {
	case "u":
		d = time.Microsecond
	case "ms":
		d = time.Millisecond
	case "s":
		d = time.Second
	case "m":
		d = time.Minute
	case "h":
		d = time.Hour
	}
	return int64(d)
}

// scanKey scans buf starting at i for the measurement and tag portion of the point.
// It returns the ending position and the byte slice of key within buf.  If there
// are tags, they will be sorted if they are not already.
func scanKey(buf []byte, i int) (int, []byte, error) {
	start := skipWhitespace(buf, i)

	i = start

	// Determines whether the tags are sort, assume they are
	sorted := true

	// indices holds the indexes within buf of the start of each tag.  For example,
	// a buf of 'cpu,host=a,region=b,zone=c' would have indices slice of [4,11,20]
	// which indicates that the first tag starts at buf[4], seconds at buf[11], and
	// last at buf[20]
	indices := make([]int, 100)

	// tracks how many commas we've seen so we know how many values are indices.
	// Since indices is an arbitrarily large slice,
	// we need to know how many values in the buffer are in use.
	commas := 0

	// First scan the Point's measurement.
	state, i, err := scanMeasurement(buf, i)
	if err != nil {
		return i, buf[start:i], err
	}

	// Optionally scan tags if needed.
	if state == tagKeyState {
		i, commas, indices, err = scanTags(buf, i, indices)
		if err != nil {
			return i, buf[start:i], err
		}
	}

	// Now we know where the key region is within buf, and the locations of tags, we
	// need to determine if duplicate tags exist and if the tags are sorted.  This iterates
	// 1/2 of the list comparing each end with each other, walking towards the center from
	// both sides.
	for j := 0; j < commas/2; j++ {
		// get the left and right tags
		_, left := scanTo(buf[indices[j]:indices[j+1]-1], 0, '=')
		_, right := scanTo(buf[indices[commas-j-1]:indices[commas-j]-1], 0, '=')

		// If the tags are equal, then there are duplicate tags, and we should abort
		if bytes.Equal(left, right) {
			return i, buf[start:i], fmt.Errorf("duplicate tags")
		}

		// If left is greater than right, the tags are not sorted.  We must continue
		// since their could be duplicate tags still.
		if bytes.Compare(left, right) > 0 {
			sorted = false
		}
	}

	// If the tags are not sorted, then sort them.  This sort is inline and
	// uses the tag indices we created earlier.  The actual buffer is not sorted, the
	// indices are using the buffer for value comparison.  After the indices are sorted,
	// the buffer is reconstructed from the sorted indices.
	if !sorted && commas > 0 {
		// Get the measurement name for later
		measurement := buf[start : indices[0]-1]

		// Sort the indices
		indices := indices[:commas]
		insertionSort(0, commas, buf, indices)

		// Create a new key using the measurement and sorted indices
		b := make([]byte, len(buf[start:i]))
		pos := copy(b, measurement)
		for _, i := range indices {
			b[pos] = ','
			pos++
			_, v := scanToSpaceOr(buf, i, ',')
			pos += copy(b[pos:], v)
		}

		return i, b, nil
	}

	return i, buf[start:i], nil
}

// The following constants allow us to specify which state to move to
// next, when scanning sections of a Point.
const (
	tagKeyState = iota
	tagValueState
	fieldsState
)

// scanMeasurement examines the measurement part of a Point, returning
// the next state to move to, and the current location in the buffer.
func scanMeasurement(buf []byte, i int) (int, int, error) {
	// Check first byte of measurement, anything except a comma is fine.
	// It can't be a space, since whitespace is stripped prior to this
	// function call.
	if buf[i] == ',' {
		return -1, i, fmt.Errorf("missing measurement")
	}

	for {
		i++
		if i >= len(buf) {
			// cpu
			return -1, i, fmt.Errorf("missing fields")
		}

		if buf[i-1] == '\\' {
			// Skip character (it's escaped).
			continue
		}

		// Unescaped comma; move onto scanning the tags.
		if buf[i] == ',' {
			return tagKeyState, i + 1, nil
		}

		// Unescaped space; move onto scanning the fields.
		if buf[i] == ' ' {
			// cpu value=1.0
			return fieldsState, i, nil
		}
	}
}

// scanTags examines all the tags in a Point, keeping track of and
// returning the updated indices slice, number of commas and location
// in buf where to start examining the Point fields.
func scanTags(buf []byte, i int, indices []int) (int, int, []int, error) {
	var (
		err    error
		commas int
		state  = tagKeyState
	)

	for {
		switch state {
		case tagKeyState:
			// Grow our indices slice if we have too many tags.
			if commas >= len(indices) {
				newIndics := make([]int, cap(indices)*2)
				copy(newIndics, indices)
				indices = newIndics
			}
			indices[commas] = i
			commas++

			i, err = scanTagsKey(buf, i)
			state = tagValueState // tag value always follows a tag key
		case tagValueState:
			state, i, err = scanTagsValue(buf, i)
		case fieldsState:
			indices[commas] = i + 1
			return i, commas, indices, nil
		}

		if err != nil {
			return i, commas, indices, err
		}
	}
}

// scanTagsKey scans each character in a tag key.
func scanTagsKey(buf []byte, i int) (int, error) {
	// First character of the key.
	if i >= len(buf) || buf[i] == ' ' || buf[i] == ',' || buf[i] == '=' {
		// cpu,{'', ' ', ',', '='}
		return i, fmt.Errorf("missing tag key")
	}

	// Examine each character in the tag key until we hit an unescaped
	// equals (the tag value), or we hit an error (i.e., unescaped
	// space or comma).
	for {
		i++

		// Either we reached the end of the buffer or we hit an
		// unescaped comma or space.
		if i >= len(buf) ||
			((buf[i] == ' ' || buf[i] == ',') && buf[i-1] != '\\') {
			// cpu,tag{'', ' ', ','}
			return i, fmt.Errorf("missing tag value")
		}

		if buf[i] == '=' && buf[i-1] != '\\' {
			// cpu,tag=
			return i + 1, nil
		}
	}
}

// scanTagsValue scans each character in a tag value.
func scanTagsValue(buf []byte, i int) (int, int, error) {
	// Tag value cannot be empty.
	if i >= len(buf) || buf[i] == ',' || buf[i] == ' ' {
		// cpu,tag={',', ' '}
		return -1, i, fmt.Errorf("missing tag value")
	}

	// Examine each character in the tag value until we hit an unescaped
	// comma (move onto next tag key), an unescaped space (move onto
	// fields), or we error out.
	for {
		i++
		if i >= len(buf) {
			// cpu,tag=value
			return -1, i, fmt.Errorf("missing fields")
		}

		// An unescaped equals sign is an invalid tag value.
		if buf[i] == '=' && buf[i-1] != '\\' {
			// cpu,tag={'=', 'fo=o'}
			return -1, i, fmt.Errorf("invalid tag format")
		}

		if buf[i] == ',' && buf[i-1] != '\\' {
			// cpu,tag=foo,
			return tagKeyState, i + 1, nil
		}

		// cpu,tag=foo value=1.0
		// cpu, tag=foo\= value=1.0
		if buf[i] == ' ' && buf[i-1] != '\\' {
			return fieldsState, i, nil
		}
	}
}

func insertionSort(l, r int, buf []byte, indices []int) {
	for i := l + 1; i < r; i++ {
		for j := i; j > l && less(buf, indices, j, j-1); j-- {
			indices[j], indices[j-1] = indices[j-1], indices[j]
		}
	}
}

func less(buf []byte, indices []int, i, j int) bool {
	// This grabs the tag names for i & j, it ignores the values
	_, a := scanTo(buf, indices[i], '=')
	_, b := scanTo(buf, indices[j], '=')
	return bytes.Compare(a, b) < 0
}

func isFieldEscapeChar(b byte) bool {
	for c := range escape.Codes {
		if c == b {
			return true
		}
	}
	return false
}

// scanFields scans buf, starting at i for the fields section of a point.  It returns
// the ending position and the byte slice of the fields within buf
func scanFields(buf []byte, i int) (int, []byte, error) {
	start := skipWhitespace(buf, i)
	i = start
	quoted := false

	// tracks how many '=' we've seen
	equals := 0

	// tracks how many commas we've seen
	commas := 0

	for {
		// reached the end of buf?
		if i >= len(buf) {
			break
		}

		// escaped characters?
		if buf[i] == '\\' && i+1 < len(buf) {
			i += 2
			continue
		}

		// If the value is quoted, scan until we get to the end quote
		// Only quote values in the field value since quotes are not significant
		// in the field key
		if buf[i] == '"' && equals > commas {
			quoted = !quoted
			i++
			continue
		}

		// If we see an =, ensure that there is at least on char before and after it
		if buf[i] == '=' && !quoted {
			equals++

			// check for "... =123" but allow "a\ =123"
			if buf[i-1] == ' ' && buf[i-2] != '\\' {
				return i, buf[start:i], fmt.Errorf("missing field key")
			}

			// check for "...a=123,=456" but allow "a=123,a\,=456"
			if buf[i-1] == ',' && buf[i-2] != '\\' {
				return i, buf[start:i], fmt.Errorf("missing field key")
			}

			// check for "... value="
			if i+1 >= len(buf) {
				return i, buf[start:i], fmt.Errorf("missing field value")
			}

			// check for "... value=,value2=..."
			if buf[i+1] == ',' || buf[i+1] == ' ' {
				return i, buf[start:i], fmt.Errorf("missing field value")
			}

			if isNumeric(buf[i+1]) || buf[i+1] == '-' || buf[i+1] == 'N' || buf[i+1] == 'n' {
				var err error
				i, err = scanNumber(buf, i+1)
				if err != nil {
					return i, buf[start:i], err
				}
				continue
			}
			// If next byte is not a double-quote, the value must be a boolean
			if buf[i+1] != '"' {
				var err error
				i, _, err = scanBoolean(buf, i+1)
				if err != nil {
					return i, buf[start:i], err
				}
				continue
			}
		}

		if buf[i] == ',' && !quoted {
			commas++
		}

		// reached end of block?
		if buf[i] == ' ' && !quoted {
			break
		}
		i++
	}

	if quoted {
		return i, buf[start:i], fmt.Errorf("unbalanced quotes")
	}

	// check that all field sections had key and values (e.g. prevent "a=1,b"
	if equals == 0 || commas != equals-1 {
		return i, buf[start:i], fmt.Errorf("invalid field format")
	}

	return i, buf[start:i], nil
}

// scanTime scans buf, starting at i for the time section of a point.  It returns
// the ending position and the byte slice of the fields within buf and error if the
// timestamp is not in the correct numeric format
func scanTime(buf []byte, i int) (int, []byte, error) {
	start := skipWhitespace(buf, i)
	i = start
	for {
		// reached the end of buf?
		if i >= len(buf) {
			break
		}

		// Timestamps should be integers, make sure they are so we don't need to actually
		// parse the timestamp until needed
		if buf[i] < '0' || buf[i] > '9' {
			// Handle negative timestamps
			if i == start && buf[i] == '-' {
				i++
				continue
			}
			return i, buf[start:i], fmt.Errorf("bad timestamp")
		}

		// reached end of block?
		if buf[i] == '\n' {
			break
		}
		i++
	}
	return i, buf[start:i], nil
}

func isNumeric(b byte) bool {
	return (b >= '0' && b <= '9') || b == '.'
}

// scanNumber returns the end position within buf, start at i after
// scanning over buf for an integer, or float.  It returns an
// error if a invalid number is scanned.
func scanNumber(buf []byte, i int) (int, error) {
	start := i
	var isInt bool

	// Is negative number?
	if i < len(buf) && buf[i] == '-' {
		i++
		// There must be more characters now, as just '-' is illegal.
		if i == len(buf) {
			return i, ErrInvalidNumber
		}
	}

	// how many decimal points we've see
	decimal := false

	// indicates the number is float in scientific notation
	scientific := false

	for {
		if i >= len(buf) {
			break
		}

		if buf[i] == ',' || buf[i] == ' ' {
			break
		}

		if buf[i] == 'i' && i > start && !isInt {
			isInt = true
			i++
			continue
		}

		if buf[i] == '.' {
			// Can't have more than 1 decimal (e.g. 1.1.1 should fail)
			if decimal {
				return i, ErrInvalidNumber
			}
			decimal = true
		}

		// `e` is valid for floats but not as the first char
		if i > start && (buf[i] == 'e' || buf[i] == 'E') {
			scientific = true
			i++
			continue
		}

		// + and - are only valid at this point if they follow an e (scientific notation)
		if (buf[i] == '+' || buf[i] == '-') && (buf[i-1] == 'e' || buf[i-1] == 'E') {
			i++
			continue
		}

		// NaN is an unsupported value
		if i+2 < len(buf) && (buf[i] == 'N' || buf[i] == 'n') {
			return i, ErrInvalidNumber
		}

		if !isNumeric(buf[i]) {
			return i, ErrInvalidNumber
		}
		i++
	}

	if isInt && (decimal || scientific) {
		return i, ErrInvalidNumber
	}

	numericDigits := i - start
	if isInt {
		numericDigits--
	}
	if decimal {
		numericDigits--
	}
	if buf[start] == '-' {
		numericDigits--
	}

	if numericDigits == 0 {
		return i, ErrInvalidNumber
	}

	// It's more common that numbers will be within min/max range for their type but we need to prevent
	// out or range numbers from being parsed successfully.  This uses some simple heuristics to decide
	// if we should parse the number to the actual type.  It does not do it all the time because it incurs
	// extra allocations and we end up converting the type again when writing points to disk.
	if isInt {
		// Make sure the last char is an 'i' for integers (e.g. 9i10 is not valid)
		if buf[i-1] != 'i' {
			return i, ErrInvalidNumber
		}
		// Parse the int to check bounds the number of digits could be larger than the max range
		// We subtract 1 from the index to remove the `i` from our tests
		if len(buf[start:i-1]) >= maxInt64Digits || len(buf[start:i-1]) >= minInt64Digits {
			if _, err := strconv.ParseInt(string(buf[start:i-1]), 10, 64); err != nil {
				return i, fmt.Errorf("unable to parse integer %s: %s", buf[start:i-1], err)
			}
		}
	} else {
		// Parse the float to check bounds if it's scientific or the number of digits could be larger than the max range
		if scientific || len(buf[start:i]) >= maxFloat64Digits || len(buf[start:i]) >= minFloat64Digits {
			if _, err := strconv.ParseFloat(string(buf[start:i]), 10); err != nil {
				return i, fmt.Errorf("invalid float")
			}
		}
	}

	return i, nil
}

// scanBoolean returns the end position within buf, start at i after
// scanning over buf for boolean. Valid values for a boolean are
// t, T, true, TRUE, f, F, false, FALSE.  It returns an error if a invalid boolean
// is scanned.
func scanBoolean(buf []byte, i int) (int, []byte, error) {
	start := i

	if i < len(buf) && (buf[i] != 't' && buf[i] != 'f' && buf[i] != 'T' && buf[i] != 'F') {
		return i, buf[start:i], fmt.Errorf("invalid boolean")
	}

	i++
	for {
		if i >= len(buf) {
			break
		}

		if buf[i] == ',' || buf[i] == ' ' {
			break
		}
		i++
	}

	// Single char bool (t, T, f, F) is ok
	if i-start == 1 {
		return i, buf[start:i], nil
	}

	// length must be 4 for true or TRUE
	if (buf[start] == 't' || buf[start] == 'T') && i-start != 4 {
		return i, buf[start:i], fmt.Errorf("invalid boolean")
	}

	// length must be 5 for false or FALSE
	if (buf[start] == 'f' || buf[start] == 'F') && i-start != 5 {
		return i, buf[start:i], fmt.Errorf("invalid boolean")
	}

	// Otherwise
	valid := false
	switch buf[start] {
	case 't':
		valid = bytes.Equal(buf[start:i], []byte("true"))
	case 'f':
		valid = bytes.Equal(buf[start:i], []byte("false"))
	case 'T':
		valid = bytes.Equal(buf[start:i], []byte("TRUE")) || bytes.Equal(buf[start:i], []byte("True"))
	case 'F':
		valid = bytes.Equal(buf[start:i], []byte("FALSE")) || bytes.Equal(buf[start:i], []byte("False"))
	}

	if !valid {
		return i, buf[start:i], fmt.Errorf("invalid boolean")
	}

	return i, buf[start:i], nil

}

// skipWhitespace returns the end position within buf, starting at i after
// scanning over spaces in tags
func skipWhitespace(buf []byte, i int) int {
	for i < len(buf) {
		if buf[i] != ' ' && buf[i] != '\t' && buf[i] != 0 {
			break
		}
		i++
	}
	return i
}

// scanLine returns the end position in buf and the next line found within
// buf.
func scanLine(buf []byte, i int) (int, []byte) {
	start := i
	quoted := false
	fields := false

	// tracks how many '=' and commas we've seen
	// this duplicates some of the functionality in scanFields
	equals := 0
	commas := 0
	for {
		// reached the end of buf?
		if i >= len(buf) {
			break
		}

		// skip past escaped characters
		if buf[i] == '\\' {
			i += 2
			continue
		}

		if buf[i] == ' ' {
			fields = true
		}

		// If we see a double quote, makes sure it is not escaped
		if fields {
			if !quoted && buf[i] == '=' {
				i++
				equals++
				continue
			} else if !quoted && buf[i] == ',' {
				i++
				commas++
				continue
			} else if buf[i] == '"' && equals > commas {
				i++
				quoted = !quoted
				continue
			}
		}

		if buf[i] == '\n' && !quoted {
			break
		}

		i++
	}

	return i, buf[start:i]
}

// scanTo returns the end position in buf and the next consecutive block
// of bytes, starting from i and ending with stop byte, where stop byte
// has not been escaped.
//
// If there are leading spaces, they are skipped.
func scanTo(buf []byte, i int, stop byte) (int, []byte) {
	start := i
	for {
		// reached the end of buf?
		if i >= len(buf) {
			break
		}

		// Reached unescaped stop value?
		if buf[i] == stop && (i == 0 || buf[i-1] != '\\') {
			break
		}
		i++
	}

	return i, buf[start:i]
}

// scanTo returns the end position in buf and the next consecutive block
// of bytes, starting from i and ending with stop byte.  If there are leading
// spaces, they are skipped.
func scanToSpaceOr(buf []byte, i int, stop byte) (int, []byte) {
	start := i
	if buf[i] == stop || buf[i] == ' ' {
		return i, buf[start:i]
	}

	for {
		i++
		if buf[i-1] == '\\' {
			continue
		}

		// reached the end of buf?
		if i >= len(buf) {
			return i, buf[start:i]
		}

		// reached end of block?
		if buf[i] == stop || buf[i] == ' ' {
			return i, buf[start:i]
		}
	}
}

func scanTagValue(buf []byte, i int) (int, []byte) {
	start := i
	for {
		if i >= len(buf) {
			break
		}

		if buf[i] == ',' && buf[i-1] != '\\' {
			break
		}
		i++
	}
	return i, buf[start:i]
}

func scanFieldValue(buf []byte, i int) (int, []byte) {
	start := i
	quoted := false
	for {
		if i >= len(buf) {
			break
		}

		// Only escape char for a field value is a double-quote
		if buf[i] == '\\' && i+1 < len(buf) && buf[i+1] == '"' {
			i += 2
			continue
		}

		// Quoted value? (e.g. string)
		if buf[i] == '"' {
			i++
			quoted = !quoted
			continue
		}

		if buf[i] == ',' && !quoted {
			break
		}
		i++
	}
	return i, buf[start:i]
}

func escapeMeasurement(in []byte) []byte {
	for b, esc := range measurementEscapeCodes {
		in = bytes.Replace(in, []byte{b}, esc, -1)
	}
	return in
}

func unescapeMeasurement(in []byte) []byte {
	for b, esc := range measurementEscapeCodes {
		in = bytes.Replace(in, esc, []byte{b}, -1)
	}
	return in
}

func escapeTag(in []byte) []byte {
	for b, esc := range tagEscapeCodes {
		if bytes.IndexByte(in, b) != -1 {
			in = bytes.Replace(in, []byte{b}, esc, -1)
		}
	}
	return in
}

func unescapeTag(in []byte) []byte {
	for b, esc := range tagEscapeCodes {
		if bytes.IndexByte(in, b) != -1 {
			in = bytes.Replace(in, esc, []byte{b}, -1)
		}
	}
	return in
}

// escapeStringField returns a copy of in with any double quotes or
// backslashes with escaped values
func escapeStringField(in string) string {
	var out []byte
	i := 0
	for {
		if i >= len(in) {
			break
		}
		// escape double-quotes
		if in[i] == '\\' {
			out = append(out, '\\')
			out = append(out, '\\')
			i++
			continue
		}
		// escape double-quotes
		if in[i] == '"' {
			out = append(out, '\\')
			out = append(out, '"')
			i++
			continue
		}
		out = append(out, in[i])
		i++

	}
	return string(out)
}

// unescapeStringField returns a copy of in with any escaped double-quotes
// or backslashes unescaped
func unescapeStringField(in string) string {
	var out []byte
	i := 0
	for {
		if i >= len(in) {
			break
		}
		// unescape backslashes
		if in[i] == '\\' && i+1 < len(in) && in[i+1] == '\\' {
			out = append(out, '\\')
			i += 2
			continue
		}
		// unescape double-quotes
		if in[i] == '\\' && i+1 < len(in) && in[i+1] == '"' {
			out = append(out, '"')
			i += 2
			continue
		}
		out = append(out, in[i])
		i++

	}
	return string(out)
}

// NewPoint returns a new point with the given measurement name, tags, fields and timestamp.  If
// an unsupported field value (NaN) or out of range time is passed, this function returns an error.
func NewPoint(name string, tags Tags, fields Fields, time time.Time) (Point, error) {
	if len(fields) == 0 {
		return nil, ErrPointMustHaveAField
	}
	if !time.IsZero() {
		if err := CheckTime(time); err != nil {
			return nil, err
		}
	}

	for key, value := range fields {
		if fv, ok := value.(float64); ok {
			// Ensure the caller validates and handles invalid field values
			if math.IsNaN(fv) {
				return nil, fmt.Errorf("NaN is an unsupported value for field %s", key)
			}
		}
		if len(key) == 0 {
			return nil, fmt.Errorf("all fields must have non-empty names")
		}
	}

	key := MakeKey([]byte(name), tags)
	if len(key) > MaxKeyLength {
		return nil, fmt.Errorf("max key length exceeded: %v > %v", len(key), MaxKeyLength)
	}

	return &point{
		key:    key,
		time:   time,
		fields: fields.MarshalBinary(),
	}, nil
}

// NewPointFromBytes returns a new Point from a marshalled Point.
func NewPointFromBytes(b []byte) (Point, error) {
	p := &point{}
	if err := p.UnmarshalBinary(b); err != nil {
		return nil, err
	}
	if len(p.Fields()) == 0 {
		return nil, ErrPointMustHaveAField
	}
	return p, nil
}

// MustNewPoint returns a new point with the given measurement name, tags, fields and timestamp.  If
// an unsupported field value (NaN) is passed, this function panics.
func MustNewPoint(name string, tags Tags, fields Fields, time time.Time) Point {
	pt, err := NewPoint(name, tags, fields, time)
	if err != nil {
		panic(err.Error())
	}
	return pt
}

func (p *point) Data() []byte {
	return p.data
}

func (p *point) SetData(b []byte) {
	p.data = b
}

func (p *point) Key() []byte {
	return p.key
}

func (p *point) name() []byte {
	_, name := scanTo(p.key, 0, ',')
	return name
}

// Name return the measurement name for the point
func (p *point) Name() string {
	if p.cachedName != "" {
		return p.cachedName
	}
	p.cachedName = string(escape.Unescape(p.name()))
	return p.cachedName
}

// SetName updates the measurement name for the point
func (p *point) SetName(name string) {
	p.cachedName = ""
	p.key = MakeKey([]byte(name), p.Tags())
}

// Time return the timestamp for the point
func (p *point) Time() time.Time {
	return p.time
}

// SetTime updates the timestamp for the point
func (p *point) SetTime(t time.Time) {
	p.time = t
}

// Tags returns the tag set for the point
func (p *point) Tags() Tags {
	return parseTags(p.key)
}

func parseTags(buf []byte) Tags {
	tags := map[string]string{}

	if len(buf) != 0 {
		pos, name := scanTo(buf, 0, ',')

		// it's an empyt key, so there are no tags
		if len(name) == 0 {
			return tags
		}

		i := pos + 1
		var key, value []byte
		for {
			if i >= len(buf) {
				break
			}
			i, key = scanTo(buf, i, '=')
			i, value = scanTagValue(buf, i+1)

			if len(value) == 0 {
				continue
			}

			tags[string(unescapeTag(key))] = string(unescapeTag(value))

			i++
		}
	}
	return tags
}

// MakeKey creates a key for a set of tags.
func MakeKey(name []byte, tags Tags) []byte {
	// unescape the name and then re-escape it to avoid double escaping.
	// The key should always be stored in escaped form.
	return append(escapeMeasurement(unescapeMeasurement(name)), tags.HashKey()...)
}

// SetTags replaces the tags for the point
func (p *point) SetTags(tags Tags) {
	p.key = MakeKey([]byte(p.Name()), tags)
}

// AddTag adds or replaces a tag value for a point
func (p *point) AddTag(key, value string) {
	tags := p.Tags()
	tags[key] = value
	p.key = MakeKey([]byte(p.Name()), tags)
}

// Fields returns the fields for the point
func (p *point) Fields() Fields {
	if p.cachedFields != nil {
		return p.cachedFields
	}
	p.cachedFields = p.unmarshalBinary()
	return p.cachedFields
}

// SetPrecision will round a time to the specified precision
func (p *point) SetPrecision(precision string) {
	switch precision {
	case "n":
	case "u":
		p.SetTime(p.Time().Truncate(time.Microsecond))
	case "ms":
		p.SetTime(p.Time().Truncate(time.Millisecond))
	case "s":
		p.SetTime(p.Time().Truncate(time.Second))
	case "m":
		p.SetTime(p.Time().Truncate(time.Minute))
	case "h":
		p.SetTime(p.Time().Truncate(time.Hour))
	}
}

func (p *point) String() string {
	if p.Time().IsZero() {
		return string(p.Key()) + " " + string(p.fields)
	}
	return string(p.Key()) + " " + string(p.fields) + " " + strconv.FormatInt(p.UnixNano(), 10)
}

func (p *point) MarshalBinary() ([]byte, error) {
	tb, err := p.time.MarshalBinary()
	if err != nil {
		return nil, err
	}

	b := make([]byte, 8+len(p.key)+len(p.fields)+len(tb))
	i := 0

	binary.BigEndian.PutUint32(b[i:], uint32(len(p.key)))
	i += 4

	i += copy(b[i:], p.key)

	binary.BigEndian.PutUint32(b[i:i+4], uint32(len(p.fields)))
	i += 4

	i += copy(b[i:], p.fields)

	copy(b[i:], tb)
	return b, nil
}

func (p *point) UnmarshalBinary(b []byte) error {
	var i int
	keyLen := int(binary.BigEndian.Uint32(b[:4]))
	i += int(4)

	p.key = b[i : i+keyLen]
	i += keyLen

	fieldLen := int(binary.BigEndian.Uint32(b[i : i+4]))
	i += int(4)

	p.fields = b[i : i+fieldLen]
	i += fieldLen

	p.time = time.Now()
	p.time.UnmarshalBinary(b[i:])
	return nil
}

func (p *point) PrecisionString(precision string) string {
	if p.Time().IsZero() {
		return fmt.Sprintf("%s %s", p.Key(), string(p.fields))
	}
	return fmt.Sprintf("%s %s %d", p.Key(), string(p.fields),
		p.UnixNano()/GetPrecisionMultiplier(precision))
}

func (p *point) RoundedString(d time.Duration) string {
	if p.Time().IsZero() {
		return fmt.Sprintf("%s %s", p.Key(), string(p.fields))
	}
	return fmt.Sprintf("%s %s %d", p.Key(), string(p.fields),
		p.time.Round(d).UnixNano())
}

func (p *point) unmarshalBinary() Fields {
	return newFieldsFromBinary(p.fields)
}

func (p *point) HashID() uint64 {
	h := fnv.New64a()
	h.Write(p.key)
	sum := h.Sum64()
	return sum
}

func (p *point) UnixNano() int64 {
	return p.Time().UnixNano()
}

// Tags represents a mapping between a Point's tag names and their
// values.
type Tags map[string]string

// HashKey hashes all of a tag's keys.
func (t Tags) HashKey() []byte {
	// Empty maps marshal to empty bytes.
	if len(t) == 0 {
		return nil
	}

	escaped := Tags{}
	for k, v := range t {
		ek := escapeTag([]byte(k))
		ev := escapeTag([]byte(v))

		if len(ev) > 0 {
			escaped[string(ek)] = string(ev)
		}
	}

	// Extract keys and determine final size.
	sz := len(escaped) + (len(escaped) * 2) // separators
	keys := make([]string, len(escaped)+1)
	i := 0
	for k, v := range escaped {
		keys[i] = k
		i++
		sz += len(k) + len(v)
	}
	keys = keys[:i]
	sort.Strings(keys)
	// Generate marshaled bytes.
	b := make([]byte, sz)
	buf := b
	idx := 0
	for _, k := range keys {
		buf[idx] = ','
		idx++
		copy(buf[idx:idx+len(k)], k)
		idx += len(k)
		buf[idx] = '='
		idx++
		v := escaped[k]
		copy(buf[idx:idx+len(v)], v)
		idx += len(v)
	}
	return b[:idx]
}

// Fields represents a mapping between a Point's field names and their
// values.
type Fields map[string]interface{}

func parseNumber(val []byte) (interface{}, error) {
	if val[len(val)-1] == 'i' {
		val = val[:len(val)-1]
		return strconv.ParseInt(string(val), 10, 64)
	}
	for i := 0; i < len(val); i++ {
		// If there is a decimal or an N (NaN), I (Inf), parse as float
		if val[i] == '.' || val[i] == 'N' || val[i] == 'n' || val[i] == 'I' || val[i] == 'i' || val[i] == 'e' {
			return strconv.ParseFloat(string(val), 64)
		}
		if val[i] < '0' && val[i] > '9' {
			return string(val), nil
		}
	}
	return strconv.ParseFloat(string(val), 64)
}

func newFieldsFromBinary(buf []byte) Fields {
	fields := Fields{}
	var (
		i              int
		name, valueBuf []byte
		value          interface{}
		err            error
	)
	for {
		if i >= len(buf) {
			break
		}

		i, name = scanTo(buf, i, '=')
		name = escape.Unescape(name)

		i, valueBuf = scanFieldValue(buf, i+1)
		if len(name) > 0 {
			if len(valueBuf) == 0 {
				fields[string(name)] = nil
				continue
			}

			// If the first char is a double-quote, then unmarshal as string
			if valueBuf[0] == '"' {
				value = unescapeStringField(string(valueBuf[1 : len(valueBuf)-1]))
				// Check for numeric characters and special NaN or Inf
			} else if (valueBuf[0] >= '0' && valueBuf[0] <= '9') || valueBuf[0] == '-' || valueBuf[0] == '.' ||
				valueBuf[0] == 'N' || valueBuf[0] == 'n' || // NaN
				valueBuf[0] == 'I' || valueBuf[0] == 'i' { // Inf

				value, err = parseNumber(valueBuf)
				if err != nil {
					panic(fmt.Sprintf("unable to parse number value '%v': %v", string(valueBuf), err))
				}

				// Otherwise parse it as bool
			} else {
				value, err = strconv.ParseBool(string(valueBuf))
				if err != nil {
					panic(fmt.Sprintf("unable to parse bool value '%v': %v\n", string(valueBuf), err))
				}
			}
			fields[string(name)] = value
		}
		i++
	}
	return fields
}

// MarshalBinary encodes all the fields to their proper type and returns the binary
// represenation
// NOTE: uint64 is specifically not supported due to potential overflow when we decode
// again later to an int64
func (p Fields) MarshalBinary() []byte {
	b := []byte{}
	keys := make([]string, len(p))
	i := 0
	for k := range p {
		keys[i] = k
		i++
	}
	sort.Strings(keys)

	for _, k := range keys {
		v := p[k]
		b = append(b, []byte(escape.String(k))...)
		b = append(b, '=')
		switch t := v.(type) {
		case int:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case int8:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case int16:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case int32:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case int64:
			b = append(b, []byte(strconv.FormatInt(t, 10))...)
			b = append(b, 'i')
		case uint:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case uint8:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case uint16:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case uint32:
			b = append(b, []byte(strconv.FormatInt(int64(t), 10))...)
			b = append(b, 'i')
		case float32:
			val := []byte(strconv.FormatFloat(float64(t), 'f', -1, 32))
			b = append(b, val...)
		case float64:
			val := []byte(strconv.FormatFloat(t, 'f', -1, 64))
			b = append(b, val...)
		case bool:
			b = append(b, []byte(strconv.FormatBool(t))...)
		case []byte:
			b = append(b, t...)
		case string:
			b = append(b, '"')
			b = append(b, []byte(escapeStringField(t))...)
			b = append(b, '"')
		case nil:
			// skip
		default:
			// Can't determine the type, so convert to string
			b = append(b, '"')
			b = append(b, []byte(escapeStringField(fmt.Sprintf("%v", v)))...)
			b = append(b, '"')

		}
		b = append(b, ',')
	}
	if len(b) > 0 {
		return b[0 : len(b)-1]
	}
	return b
}

type indexedSlice struct {
	indices []int
	b       []byte
}

func (s *indexedSlice) Less(i, j int) bool {
	_, a := scanTo(s.b, s.indices[i], '=')
	_, b := scanTo(s.b, s.indices[j], '=')
	return bytes.Compare(a, b) < 0
}

func (s *indexedSlice) Swap(i, j int) {
	s.indices[i], s.indices[j] = s.indices[j], s.indices[i]
}

func (s *indexedSlice) Len() int {
	return len(s.indices)
}
