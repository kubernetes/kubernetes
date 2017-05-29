package models

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
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
	ErrInvalidPoint         = errors.New("point is invalid")
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

	// Split will attempt to return multiple points with the same timestamp whose
	// string representations are no longer than size. Points with a single field or
	// a point without a timestamp may exceed the requested size.
	Split(size int) []Point

	// Round will round the timestamp of the point to the given duration
	Round(d time.Duration)

	// StringSize returns the length of the string that would be returned by String()
	StringSize() int

	// AppendString appends the result of String() to the provided buffer and returns
	// the result, potentially reducing string allocations
	AppendString(buf []byte) []byte

	// FieldIterator retuns a FieldIterator that can be used to traverse the
	// fields of a point without constructing the in-memory map
	FieldIterator() FieldIterator
}

type FieldType int

const (
	Integer FieldType = iota
	Float
	Boolean
	String
	Empty
)

type FieldIterator interface {
	Next() bool
	FieldKey() []byte
	Type() FieldType
	StringValue() string
	IntegerValue() int64
	BooleanValue() bool
	FloatValue() float64

	Delete()
	Reset()
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

	// cached version of parsed tags
	cachedTags Tags

	it fieldIterator
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
func ParseKey(buf []byte) (string, Tags, error) {
	// Ignore the error because scanMeasurement returns "missing fields" which we ignore
	// when just parsing a key
	state, i, _ := scanMeasurement(buf, 0)

	var tags Tags
	if state == tagKeyState {
		tags = parseTags(buf)
		// scanMeasurement returns the location of the comma if there are tags, strip that off
		return string(buf[:i-1]), tags, nil
	}
	return string(buf[:i]), tags, nil
}

// ParsePointsWithPrecision is similar to ParsePoints, but allows the
// caller to provide a precision for time.
func ParsePointsWithPrecision(buf []byte, defaultTime time.Time, precision string) ([]Point, error) {
	points := make([]Point, 0, bytes.Count(buf, []byte{'\n'})+1)
	var (
		pos    int
		block  []byte
		failed []string
	)
	for pos < len(buf) {
		pos, block = scanLine(buf, pos)
		pos++

		if len(block) == 0 {
			continue
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

		pt, err := parsePoint(block[start:], defaultTime, precision)
		if err != nil {
			failed = append(failed, fmt.Sprintf("unable to parse '%s': %v", string(block[start:len(block)]), err))
		} else {
			points = append(points, pt)
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
		ts, err := parseIntBytes(ts, 10, 64)
		if err != nil {
			return nil, err
		}
		pt.time, err = SafeCalcTime(ts, precision)
		if err != nil {
			return nil, err
		}

		// Determine if there are illegal non-whitespace characters after the
		// timestamp block.
		for pos < len(buf) {
			if buf[pos] != ' ' {
				return nil, ErrInvalidPoint
			}
			pos++
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

	// Now we know where the key region is within buf, and the location of tags, we
	// need to determine if duplicate tags exist and if the tags are sorted. This iterates
	// over the list comparing each tag in the sequence with each other.
	for j := 0; j < commas-1; j++ {
		// get the left and right tags
		_, left := scanTo(buf[indices[j]:indices[j+1]-1], 0, '=')
		_, right := scanTo(buf[indices[j+1]:indices[j+2]-1], 0, '=')

		// If left is greater than right, the tags are not sorted. We do not have to
		// continue because the short path no longer works.
		// If the tags are equal, then there are duplicate tags, and we should abort.
		// If the tags are not sorted, this pass may not find duplicate tags and we
		// need to do a more exhaustive search later.
		if cmp := bytes.Compare(left, right); cmp > 0 {
			sorted = false
			break
		} else if cmp == 0 {
			return i, buf[start:i], fmt.Errorf("duplicate tags")
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

		// Check again for duplicate tags now that the tags are sorted.
		for j := 0; j < commas-1; j++ {
			// get the left and right tags
			_, left := scanTo(buf[indices[j]:], 0, '=')
			_, right := scanTo(buf[indices[j+1]:], 0, '=')

			// If the tags are equal, then there are duplicate tags, and we should abort.
			// If the tags are not sorted, this pass may not find duplicate tags and we
			// need to do a more exhaustive search later.
			if bytes.Equal(left, right) {
				return i, b, fmt.Errorf("duplicate tags")
			}
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
	if i >= len(buf) || buf[i] == ',' {
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

// scanTime scans buf, starting at i for the time section of a point. It
// returns the ending position and the byte slice of the timestamp within buf
// and and error if the timestamp is not in the correct numeric format.
func scanTime(buf []byte, i int) (int, []byte, error) {
	start := skipWhitespace(buf, i)
	i = start

	for {
		// reached the end of buf?
		if i >= len(buf) {
			break
		}

		// Reached end of block or trailing whitespace?
		if buf[i] == '\n' || buf[i] == ' ' {
			break
		}

		// Handle negative timestamps
		if i == start && buf[i] == '-' {
			i++
			continue
		}

		// Timestamps should be integers, make sure they are so we don't need
		// to actually  parse the timestamp until needed.
		if buf[i] < '0' || buf[i] > '9' {
			return i, buf[start:i], fmt.Errorf("bad timestamp")
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
			if _, err := parseIntBytes(buf[start:i-1], 10, 64); err != nil {
				return i, fmt.Errorf("unable to parse integer %s: %s", buf[start:i-1], err)
			}
		}
	} else {
		// Parse the float to check bounds if it's scientific or the number of digits could be larger than the max range
		if scientific || len(buf[start:i]) >= maxFloat64Digits || len(buf[start:i]) >= minFloat64Digits {
			if _, err := parseFloatBytes(buf[start:i], 10); err != nil {
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
	for i < len(buf) {
		// Only escape char for a field value is a double-quote and backslash
		if buf[i] == '\\' && i+1 < len(buf) && (buf[i+1] == '"' || buf[i+1] == '\\') {
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
	if bytes.IndexByte(in, '\\') == -1 {
		return in
	}

	for b, esc := range tagEscapeCodes {
		if bytes.IndexByte(in, b) != -1 {
			in = bytes.Replace(in, esc, []byte{b}, -1)
		}
	}
	return in
}

// EscapeStringField returns a copy of in with any double quotes or
// backslashes with escaped values
func EscapeStringField(in string) string {
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
	if strings.IndexByte(in, '\\') == -1 {
		return in
	}

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
func NewPoint(name string, tags Tags, fields Fields, t time.Time) (Point, error) {
	key, err := pointKey(name, tags, fields, t)
	if err != nil {
		return nil, err
	}

	return &point{
		key:    key,
		time:   t,
		fields: fields.MarshalBinary(),
	}, nil
}

// pointKey checks some basic requirements for valid points, and returns the
// key, along with an possible error
func pointKey(measurement string, tags Tags, fields Fields, t time.Time) ([]byte, error) {
	if len(fields) == 0 {
		return nil, ErrPointMustHaveAField
	}

	if !t.IsZero() {
		if err := CheckTime(t); err != nil {
			return nil, err
		}
	}

	for key, value := range fields {
		switch value := value.(type) {
		case float64:
			// Ensure the caller validates and handles invalid field values
			if math.IsNaN(value) {
				return nil, fmt.Errorf("NaN is an unsupported value for field %s", key)
			}
		case float32:
			// Ensure the caller validates and handles invalid field values
			if math.IsNaN(float64(value)) {
				return nil, fmt.Errorf("NaN is an unsupported value for field %s", key)
			}
		}
		if len(key) == 0 {
			return nil, fmt.Errorf("all fields must have non-empty names")
		}
	}

	key := MakeKey([]byte(measurement), tags)
	if len(key) > MaxKeyLength {
		return nil, fmt.Errorf("max key length exceeded: %v > %v", len(key), MaxKeyLength)
	}

	return key, nil
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

// Round implements Point.Round
func (p *point) Round(d time.Duration) {
	p.time = p.time.Round(d)
}

// Tags returns the tag set for the point
func (p *point) Tags() Tags {
	if p.cachedTags != nil {
		return p.cachedTags
	}
	p.cachedTags = parseTags(p.key)
	return p.cachedTags
}

func parseTags(buf []byte) Tags {
	if len(buf) == 0 {
		return nil
	}

	pos, name := scanTo(buf, 0, ',')

	// it's an empty key, so there are no tags
	if len(name) == 0 {
		return nil
	}

	tags := make(Tags, 0, bytes.Count(buf, []byte(",")))
	hasEscape := bytes.IndexByte(buf, '\\') != -1

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

		if hasEscape {
			tags = append(tags, Tag{Key: unescapeTag(key), Value: unescapeTag(value)})
		} else {
			tags = append(tags, Tag{Key: key, Value: value})
		}

		i++
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
	p.cachedTags = tags
}

// AddTag adds or replaces a tag value for a point
func (p *point) AddTag(key, value string) {
	tags := p.Tags()
	tags = append(tags, Tag{Key: []byte(key), Value: []byte(value)})
	sort.Sort(tags)
	p.cachedTags = tags
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

// AppendString implements Point.AppendString
func (p *point) AppendString(buf []byte) []byte {
	buf = append(buf, p.key...)
	buf = append(buf, ' ')
	buf = append(buf, p.fields...)

	if !p.time.IsZero() {
		buf = append(buf, ' ')
		buf = strconv.AppendInt(buf, p.UnixNano(), 10)
	}

	return buf
}

func (p *point) StringSize() int {
	size := len(p.key) + len(p.fields) + 1

	if !p.time.IsZero() {
		digits := 1 // even "0" has one digit
		t := p.UnixNano()
		if t < 0 {
			// account for negative sign, then negate
			digits++
			t = -t
		}
		for t > 9 { // already accounted for one digit
			digits++
			t /= 10
		}
		size += digits + 1 // digits and a space
	}

	return size
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
	iter := p.FieldIterator()
	fields := make(Fields, 8)
	for iter.Next() {
		if len(iter.FieldKey()) == 0 {
			continue
		}
		switch iter.Type() {
		case Float:
			fields[string(iter.FieldKey())] = iter.FloatValue()
		case Integer:
			fields[string(iter.FieldKey())] = iter.IntegerValue()
		case String:
			fields[string(iter.FieldKey())] = iter.StringValue()
		case Boolean:
			fields[string(iter.FieldKey())] = iter.BooleanValue()
		}
	}
	return fields
}

func (p *point) HashID() uint64 {
	h := NewInlineFNV64a()
	h.Write(p.key)
	sum := h.Sum64()
	return sum
}

func (p *point) UnixNano() int64 {
	return p.Time().UnixNano()
}

func (p *point) Split(size int) []Point {
	if p.time.IsZero() || len(p.String()) <= size {
		return []Point{p}
	}

	// key string, timestamp string, spaces
	size -= len(p.key) + len(strconv.FormatInt(p.time.UnixNano(), 10)) + 2

	var points []Point
	var start, cur int

	for cur < len(p.fields) {
		end, _ := scanTo(p.fields, cur, '=')
		end, _ = scanFieldValue(p.fields, end+1)

		if cur > start && end-start > size {
			points = append(points, &point{
				key:    p.key,
				time:   p.time,
				fields: p.fields[start : cur-1],
			})
			start = cur
		}

		cur = end + 1
	}

	points = append(points, &point{
		key:    p.key,
		time:   p.time,
		fields: p.fields[start:],
	})

	return points
}

// Tag represents a single key/value tag pair.
type Tag struct {
	Key   []byte
	Value []byte
}

// Tags represents a sorted list of tags.
type Tags []Tag

// NewTags returns a new Tags from a map.
func NewTags(m map[string]string) Tags {
	if len(m) == 0 {
		return nil
	}
	a := make(Tags, 0, len(m))
	for k, v := range m {
		a = append(a, Tag{Key: []byte(k), Value: []byte(v)})
	}
	sort.Sort(a)
	return a
}

func (a Tags) Len() int           { return len(a) }
func (a Tags) Less(i, j int) bool { return bytes.Compare(a[i].Key, a[j].Key) == -1 }
func (a Tags) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// Get returns the value for a key.
func (a Tags) Get(key []byte) []byte {
	// OPTIMIZE: Use sort.Search if tagset is large.

	for _, t := range a {
		if bytes.Equal(t.Key, key) {
			return t.Value
		}
	}
	return nil
}

// GetString returns the string value for a string key.
func (a Tags) GetString(key string) string {
	return string(a.Get([]byte(key)))
}

// Set sets the value for a key.
func (a *Tags) Set(key, value []byte) {
	for _, t := range *a {
		if bytes.Equal(t.Key, key) {
			t.Value = value
			return
		}
	}
	*a = append(*a, Tag{Key: key, Value: value})
	sort.Sort(*a)
}

// SetString sets the string value for a string key.
func (a *Tags) SetString(key, value string) {
	a.Set([]byte(key), []byte(value))
}

// Delete removes a tag by key.
func (a *Tags) Delete(key []byte) {
	for i, t := range *a {
		if bytes.Equal(t.Key, key) {
			copy((*a)[i:], (*a)[i+1:])
			(*a)[len(*a)-1] = Tag{}
			*a = (*a)[:len(*a)-1]
			return
		}
	}
}

// Map returns a map representation of the tags.
func (a Tags) Map() map[string]string {
	m := make(map[string]string, len(a))
	for _, t := range a {
		m[string(t.Key)] = string(t.Value)
	}
	return m
}

// Merge merges the tags combining the two. If both define a tag with the
// same key, the merged value overwrites the old value.
// A new map is returned.
func (a Tags) Merge(other map[string]string) Tags {
	merged := make(map[string]string, len(a)+len(other))
	for _, t := range a {
		merged[string(t.Key)] = string(t.Value)
	}
	for k, v := range other {
		merged[k] = v
	}
	return NewTags(merged)
}

// HashKey hashes all of a tag's keys.
func (a Tags) HashKey() []byte {
	// Empty maps marshal to empty bytes.
	if len(a) == 0 {
		return nil
	}

	escaped := make(Tags, 0, len(a))
	for _, t := range a {
		ek := escapeTag(t.Key)
		ev := escapeTag(t.Value)

		if len(ev) > 0 {
			escaped = append(escaped, Tag{Key: ek, Value: ev})
		}
	}

	// Extract keys and determine final size.
	sz := len(escaped) + (len(escaped) * 2) // separators
	keys := make([][]byte, len(escaped)+1)
	for i, t := range escaped {
		keys[i] = t.Key
		sz += len(t.Key) + len(t.Value)
	}
	keys = keys[:len(escaped)]
	sort.Sort(byteSlices(keys))

	// Generate marshaled bytes.
	b := make([]byte, sz)
	buf := b
	idx := 0
	for i, k := range keys {
		buf[idx] = ','
		idx++
		copy(buf[idx:idx+len(k)], k)
		idx += len(k)
		buf[idx] = '='
		idx++
		v := escaped[i].Value
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
		return parseIntBytes(val, 10, 64)
	}
	for i := 0; i < len(val); i++ {
		// If there is a decimal or an N (NaN), I (Inf), parse as float
		if val[i] == '.' || val[i] == 'N' || val[i] == 'n' || val[i] == 'I' || val[i] == 'i' || val[i] == 'e' {
			return parseFloatBytes(val, 64)
		}
		if val[i] < '0' && val[i] > '9' {
			return string(val), nil
		}
	}
	return parseFloatBytes(val, 64)
}

func (p *point) FieldIterator() FieldIterator {
	p.Reset()
	return p
}

type fieldIterator struct {
	start, end  int
	key, keybuf []byte
	valueBuf    []byte
	fieldType   FieldType
}

func (p *point) Next() bool {
	p.it.start = p.it.end
	if p.it.start >= len(p.fields) {
		return false
	}

	p.it.end, p.it.key = scanTo(p.fields, p.it.start, '=')
	if escape.IsEscaped(p.it.key) {
		p.it.keybuf = escape.AppendUnescaped(p.it.keybuf[:0], p.it.key)
		p.it.key = p.it.keybuf
	}

	p.it.end, p.it.valueBuf = scanFieldValue(p.fields, p.it.end+1)
	p.it.end++

	if len(p.it.valueBuf) == 0 {
		p.it.fieldType = Empty
		return true
	}

	c := p.it.valueBuf[0]

	if c == '"' {
		p.it.fieldType = String
		return true
	}

	if strings.IndexByte(`0123456789-.nNiI`, c) >= 0 {
		if p.it.valueBuf[len(p.it.valueBuf)-1] == 'i' {
			p.it.fieldType = Integer
			p.it.valueBuf = p.it.valueBuf[:len(p.it.valueBuf)-1]
		} else {
			p.it.fieldType = Float
		}
		return true
	}

	// to keep the same behavior that currently exists, default to boolean
	p.it.fieldType = Boolean
	return true
}

func (p *point) FieldKey() []byte {
	return p.it.key
}

func (p *point) Type() FieldType {
	return p.it.fieldType
}

func (p *point) StringValue() string {
	return unescapeStringField(string(p.it.valueBuf[1 : len(p.it.valueBuf)-1]))
}

func (p *point) IntegerValue() int64 {
	n, err := parseIntBytes(p.it.valueBuf, 10, 64)
	if err != nil {
		panic(fmt.Sprintf("unable to parse integer value %q: %v", p.it.valueBuf, err))
	}
	return n
}

func (p *point) BooleanValue() bool {
	b, err := parseBoolBytes(p.it.valueBuf)
	if err != nil {
		panic(fmt.Sprintf("unable to parse bool value %q: %v", p.it.valueBuf, err))
	}
	return b
}

func (p *point) FloatValue() float64 {
	f, err := parseFloatBytes(p.it.valueBuf, 64)
	if err != nil {
		// panic because that's what the non-iterator code does
		panic(fmt.Sprintf("unable to parse floating point value %q: %v", p.it.valueBuf, err))
	}
	return f
}

func (p *point) Delete() {
	switch {
	case p.it.end == p.it.start:
	case p.it.end >= len(p.fields):
		p.fields = p.fields[:p.it.start]
	case p.it.start == 0:
		p.fields = p.fields[p.it.end:]
	default:
		p.fields = append(p.fields[:p.it.start], p.fields[p.it.end:]...)
	}

	p.it.end = p.it.start
	p.it.key = nil
	p.it.valueBuf = nil
	p.it.fieldType = Empty
}

func (p *point) Reset() {
	p.it.fieldType = Empty
	p.it.key = nil
	p.it.valueBuf = nil
	p.it.start = 0
	p.it.end = 0
}

// MarshalBinary encodes all the fields to their proper type and returns the binary
// represenation
// NOTE: uint64 is specifically not supported due to potential overflow when we decode
// again later to an int64
// NOTE2: uint is accepted, and may be 64 bits, and is for some reason accepted...
func (p Fields) MarshalBinary() []byte {
	var b []byte
	keys := make([]string, 0, len(p))

	for k := range p {
		keys = append(keys, k)
	}

	// Not really necessary, can probably be removed.
	sort.Strings(keys)

	for i, k := range keys {
		if i > 0 {
			b = append(b, ',')
		}
		b = appendField(b, k, p[k])
	}

	return b
}

func appendField(b []byte, k string, v interface{}) []byte {
	b = append(b, []byte(escape.String(k))...)
	b = append(b, '=')

	// check popular types first
	switch v := v.(type) {
	case float64:
		b = strconv.AppendFloat(b, v, 'f', -1, 64)
	case int64:
		b = strconv.AppendInt(b, v, 10)
		b = append(b, 'i')
	case string:
		b = append(b, '"')
		b = append(b, []byte(EscapeStringField(v))...)
		b = append(b, '"')
	case bool:
		b = strconv.AppendBool(b, v)
	case int32:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case int16:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case int8:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case int:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case uint32:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case uint16:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case uint8:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	// TODO: 'uint' should be considered just as "dangerous" as a uint64,
	// perhaps the value should be checked and capped at MaxInt64? We could
	// then include uint64 as an accepted value
	case uint:
		b = strconv.AppendInt(b, int64(v), 10)
		b = append(b, 'i')
	case float32:
		b = strconv.AppendFloat(b, float64(v), 'f', -1, 32)
	case []byte:
		b = append(b, v...)
	case nil:
		// skip
	default:
		// Can't determine the type, so convert to string
		b = append(b, '"')
		b = append(b, []byte(EscapeStringField(fmt.Sprintf("%v", v)))...)
		b = append(b, '"')

	}

	return b
}

type byteSlices [][]byte

func (a byteSlices) Len() int           { return len(a) }
func (a byteSlices) Less(i, j int) bool { return bytes.Compare(a[i], a[j]) == -1 }
func (a byteSlices) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
