package models_test

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/influxdata/influxdb/models"
)

var (
	tags   = models.NewTags(map[string]string{"foo": "bar", "apple": "orange", "host": "serverA", "region": "uswest"})
	fields = models.Fields{
		"int64":         int64(math.MaxInt64),
		"uint32":        uint32(math.MaxUint32),
		"string":        "String field that has a decent length, probably some log message or something",
		"boolean":       false,
		"float64-tiny":  float64(math.SmallestNonzeroFloat64),
		"float64-large": float64(math.MaxFloat64),
	}
	maxFloat64 = strconv.FormatFloat(math.MaxFloat64, 'f', 1, 64)
	minFloat64 = strconv.FormatFloat(-math.MaxFloat64, 'f', 1, 64)
)

func TestMarshal(t *testing.T) {
	got := tags.HashKey()
	if exp := ",apple=orange,foo=bar,host=serverA,region=uswest"; string(got) != exp {
		t.Log("got: ", string(got))
		t.Log("exp: ", exp)
		t.Error("invalid match")
	}
}

func BenchmarkMarshal(b *testing.B) {
	for i := 0; i < b.N; i++ {
		tags.HashKey()
	}
}

func TestPoint_StringSize(t *testing.T) {
	testPoint_cube(t, func(p models.Point) {
		l := p.StringSize()
		s := p.String()

		if l != len(s) {
			t.Errorf("Incorrect length for %q. got %v, exp %v", s, l, len(s))
		}
	})

}

func TestPoint_AppendString(t *testing.T) {
	testPoint_cube(t, func(p models.Point) {
		got := p.AppendString(nil)
		exp := []byte(p.String())

		if !reflect.DeepEqual(exp, got) {
			t.Errorf("AppendString() didn't match String(): got %v, exp %v", got, exp)
		}
	})
}

func testPoint_cube(t *testing.T, f func(p models.Point)) {
	// heard of a table-driven test? let's make a cube-driven test...
	tagList := []models.Tags{nil, {models.Tag{Key: []byte("foo"), Value: []byte("bar")}}, tags}
	fieldList := []models.Fields{{"a": 42.0}, {"a": 42, "b": "things"}, fields}
	timeList := []time.Time{time.Time{}, time.Unix(0, 0), time.Unix(-34526, 0), time.Unix(231845, 0), time.Now()}

	for _, tagSet := range tagList {
		for _, fieldSet := range fieldList {
			for _, pointTime := range timeList {
				p, err := models.NewPoint("test", tagSet, fieldSet, pointTime)
				if err != nil {
					t.Errorf("unexpected error creating point: %v", err)
					continue
				}

				f(p)
			}
		}
	}
}

var p models.Point

func BenchmarkNewPoint(b *testing.B) {
	ts := time.Now()
	for i := 0; i < b.N; i++ {
		p, _ = models.NewPoint("measurement", tags, fields, ts)
	}
}

func BenchmarkParsePointNoTags5000(b *testing.B) {
	var batch [5000]string
	for i := 0; i < len(batch); i++ {
		batch[i] = `cpu value=1i 1000000000`
	}
	lines := strings.Join(batch[:], "\n")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		models.ParsePoints([]byte(lines))
		b.SetBytes(int64(len(lines)))
	}
}

func BenchmarkParsePointNoTags(b *testing.B) {
	line := `cpu value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointWithPrecisionN(b *testing.B) {
	line := `cpu value=1i 1000000000`
	defaultTime := time.Now().UTC()
	for i := 0; i < b.N; i++ {
		models.ParsePointsWithPrecision([]byte(line), defaultTime, "n")
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointWithPrecisionU(b *testing.B) {
	line := `cpu value=1i 1000000000`
	defaultTime := time.Now().UTC()
	for i := 0; i < b.N; i++ {
		models.ParsePointsWithPrecision([]byte(line), defaultTime, "u")
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted2(b *testing.B) {
	line := `cpu,host=serverA,region=us-west value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted5(b *testing.B) {
	line := `cpu,env=prod,host=serverA,region=us-west,target=servers,zone=1c value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted10(b *testing.B) {
	line := `cpu,env=prod,host=serverA,region=us-west,tag1=value1,tag2=value2,tag3=value3,tag4=value4,tag5=value5,target=servers,zone=1c value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsUnSorted2(b *testing.B) {
	line := `cpu,region=us-west,host=serverA value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func BenchmarkParsePointsTagsUnSorted5(b *testing.B) {
	line := `cpu,region=us-west,host=serverA,env=prod,target=servers,zone=1c value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func BenchmarkParsePointsTagsUnSorted10(b *testing.B) {
	line := `cpu,region=us-west,host=serverA,env=prod,target=servers,zone=1c,tag1=value1,tag2=value2,tag3=value3,tag4=value4,tag5=value5 value=1i 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := models.ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func BenchmarkParseKey(b *testing.B) {
	line := `cpu,region=us-west,host=serverA,env=prod,target=servers,zone=1c,tag1=value1,tag2=value2,tag3=value3,tag4=value4,tag5=value5`
	for i := 0; i < b.N; i++ {
		models.ParseKey([]byte(line))
	}
}

// TestPoint wraps a models.Point but also makes available the raw
// arguments to the Point.
//
// This is useful for ensuring that comparisons between results of
// operations on Points match the expected input data to the Point,
// since models.Point does not expose the raw input data (e.g., tags)
// via its API.
type TestPoint struct {
	RawFields models.Fields
	RawTags   models.Tags
	RawTime   time.Time
	models.Point
}

// NewTestPoint returns a new TestPoint.
//
// NewTestPoint panics if it is not a valid models.Point.
func NewTestPoint(name string, tags models.Tags, fields models.Fields, time time.Time) TestPoint {
	return TestPoint{
		RawTags:   tags,
		RawFields: fields,
		RawTime:   time,
		Point:     models.MustNewPoint(name, tags, fields, time),
	}
}

func test(t *testing.T, line string, point TestPoint) {
	pts, err := models.ParsePointsWithPrecision([]byte(line), time.Unix(0, 0), "n")
	if err != nil {
		t.Fatalf(`ParsePoints("%s") mismatch. got %v, exp nil`, line, err)
	}

	if exp := 1; len(pts) != exp {
		t.Fatalf(`ParsePoints("%s") len mismatch. got %d, exp %d`, line, len(pts), exp)
	}

	if exp := point.Key(); !bytes.Equal(pts[0].Key(), exp) {
		t.Errorf("ParsePoints(\"%s\") key mismatch.\ngot %v\nexp %v", line, string(pts[0].Key()), string(exp))
	}

	if exp := len(point.Tags()); len(pts[0].Tags()) != exp {
		t.Errorf(`ParsePoints("%s") tags mismatch. got %v, exp %v`, line, pts[0].Tags(), exp)
	}

	for _, tag := range pts[0].Tags() {
		if !bytes.Equal(tag.Value, point.RawTags.Get(tag.Key)) {
			t.Errorf(`ParsePoints("%s") tags mismatch. got %s, exp %s`, line, tag.Value, point.RawTags.Get(tag.Key))
		}
	}

	for name, value := range point.RawFields {
		val := pts[0].Fields()[name]
		expfval, ok := val.(float64)

		if ok && math.IsNaN(expfval) {
			gotfval, ok := value.(float64)
			if ok && !math.IsNaN(gotfval) {
				t.Errorf(`ParsePoints("%s") field '%s' mismatch. exp NaN`, line, name)
			}
		} else if !reflect.DeepEqual(pts[0].Fields()[name], value) {
			t.Errorf(`ParsePoints("%s") field '%s' mismatch. got %[3]v (%[3]T), exp %[4]v (%[4]T)`, line, name, pts[0].Fields()[name], value)
		}
	}

	if !pts[0].Time().Equal(point.Time()) {
		t.Errorf(`ParsePoints("%s") time mismatch. got %v, exp %v`, line, pts[0].Time(), point.Time())
	}

	if !strings.HasPrefix(pts[0].String(), line) {
		t.Errorf("ParsePoints string mismatch.\ngot: %v\nexp: %v", pts[0].String(), line)
	}
}

func TestParsePointNoValue(t *testing.T) {
	pts, err := models.ParsePointsString("")
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, "", err)
	}

	if exp := 0; len(pts) != exp {
		t.Errorf(`ParsePoints("%s") len mismatch. got %v, exp %v`, "", len(pts), exp)
	}
}

func TestParsePointWhitespaceValue(t *testing.T) {
	pts, err := models.ParsePointsString(" ")
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, "", err)
	}

	if exp := 0; len(pts) != exp {
		t.Errorf(`ParsePoints("%s") len mismatch. got %v, exp %v`, "", len(pts), exp)
	}
}

func TestParsePointNoFields(t *testing.T) {
	expectedSuffix := "missing fields"
	examples := []string{
		"cpu_load_short,host=server01,region=us-west",
		"cpu",
		"cpu,host==",
		"=",
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got nil, exp error`, i, example)
		} else if !strings.HasSuffix(err.Error(), expectedSuffix) {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got %q, exp suffix %q`, i, example, err, expectedSuffix)
		}
	}
}

func TestParsePointNoTimestamp(t *testing.T) {
	test(t, "cpu value=1", NewTestPoint("cpu", nil, models.Fields{"value": 1.0}, time.Unix(0, 0)))
}

func TestParsePointMissingQuote(t *testing.T) {
	expectedSuffix := "unbalanced quotes"
	examples := []string{
		`cpu,host=serverA value="test`,
		`cpu,host=serverA value="test""`,
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got nil, exp error`, i, example)
		} else if !strings.HasSuffix(err.Error(), expectedSuffix) {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got %q, exp suffix %q`, i, example, err, expectedSuffix)
		}
	}
}

func TestParsePointMissingTagKey(t *testing.T) {
	expectedSuffix := "missing tag key"
	examples := []string{
		`cpu, value=1`,
		`cpu,`,
		`cpu,,,`,
		`cpu,host=serverA,=us-east value=1i`,
		`cpu,host=serverAa\,,=us-east value=1i`,
		`cpu,host=serverA\,,=us-east value=1i`,
		`cpu, =serverA value=1i`,
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got nil, exp error`, i, example)
		} else if !strings.HasSuffix(err.Error(), expectedSuffix) {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got %q, exp suffix %q`, i, example, err, expectedSuffix)
		}
	}

	_, err := models.ParsePointsString(`cpu,host=serverA,\ =us-east value=1i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,\ =us-east value=1i`, err)
	}
}

func TestParsePointMissingTagValue(t *testing.T) {
	expectedSuffix := "missing tag value"
	examples := []string{
		`cpu,host`,
		`cpu,host,`,
		`cpu,host=`,
		`cpu,host value=1i`,
		`cpu,host=serverA,region value=1i`,
		`cpu,host=serverA,region= value=1i`,
		`cpu,host=serverA,region=,zone=us-west value=1i`,
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got nil, exp error`, i, example)
		} else if !strings.HasSuffix(err.Error(), expectedSuffix) {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got %q, exp suffix %q`, i, example, err, expectedSuffix)
		}
	}
}

func TestParsePointInvalidTagFormat(t *testing.T) {
	expectedSuffix := "invalid tag format"
	examples := []string{
		`cpu,host=f=o,`,
		`cpu,host=f\==o,`,
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got nil, exp error`, i, example)
		} else if !strings.HasSuffix(err.Error(), expectedSuffix) {
			t.Errorf(`[Example %d] ParsePoints("%s") mismatch. got %q, exp suffix %q`, i, example, err, expectedSuffix)
		}
	}
}

func TestParsePointMissingFieldName(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west =`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west =`)
	}

	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west =123i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west =123i`)
	}

	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west a\ =123i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west a\ =123i`)
	}
	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=123i,=456i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=123i,=456i`)
	}
}

func TestParsePointMissingFieldValue(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=`)
	}

	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value= 1000000000i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value= 1000000000i`)
	}

	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=,value2=1i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=,value2=1i`)
	}

	_, err = models.ParsePointsString(`cpu,host=server01,region=us-west 1434055562000000000i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=server01,region=us-west 1434055562000000000i`)
	}

	_, err = models.ParsePointsString(`cpu,host=server01,region=us-west value=1i,b`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=server01,region=us-west value=1i,b`)
	}
}

func TestParsePointBadNumber(t *testing.T) {
	for _, tt := range []string{
		"cpu v=- ",
		"cpu v=-i ",
		"cpu v=-. ",
		"cpu v=. ",
		"cpu v=1.0i ",
		"cpu v=1ii ",
		"cpu v=1a ",
		"cpu v=-e-e-e ",
		"cpu v=42+3 ",
		"cpu v= ",
	} {
		_, err := models.ParsePointsString(tt)
		if err == nil {
			t.Errorf("Point %q should be invalid", tt)
		}
	}
}

func TestParsePointMaxInt64(t *testing.T) {
	// out of range
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=9223372036854775808i`)
	exp := `unable to parse 'cpu,host=serverA,region=us-west value=9223372036854775808i': unable to parse integer 9223372036854775808: strconv.ParseInt: parsing "9223372036854775808": value out of range`
	if err == nil || (err != nil && err.Error() != exp) {
		t.Fatalf("Error mismatch:\nexp: %s\ngot: %v", exp, err)
	}

	// max int
	p, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=9223372036854775807i`)
	if err != nil {
		t.Fatalf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=9223372036854775807i`, err)
	}
	if exp, got := int64(9223372036854775807), p[0].Fields()["value"].(int64); exp != got {
		t.Fatalf("ParsePoints Value mismatch. \nexp: %v\ngot: %v", exp, got)
	}

	// leading zeros
	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=0009223372036854775807i`)
	if err != nil {
		t.Fatalf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=0009223372036854775807i`, err)
	}
}

func TestParsePointMinInt64(t *testing.T) {
	// out of range
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=-9223372036854775809i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=-9223372036854775809i`)
	}

	// min int
	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=-9223372036854775808i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-9223372036854775808i`, err)
	}

	// leading zeros
	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=-0009223372036854775808i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-0009223372036854775808i`, err)
	}
}

func TestParsePointMaxFloat64(t *testing.T) {
	// out of range
	_, err := models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "1"+string(maxFloat64)))
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=...`)
	}

	// max float
	_, err = models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, string(maxFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=9223372036854775807`, err)
	}

	// leading zeros
	_, err = models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "0000"+string(maxFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=0009223372036854775807`, err)
	}
}

func TestParsePointMinFloat64(t *testing.T) {
	// out of range
	_, err := models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "-1"+string(minFloat64)[1:]))
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=...`)
	}

	// min float
	_, err = models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, string(minFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=...`, err)
	}

	// leading zeros
	_, err = models.ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "-0000000"+string(minFloat64)[1:]))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=...`, err)
	}
}

func TestParsePointNumberNonNumeric(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=.1a`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=.1a`)
	}
}

func TestParsePointNegativeWrongPlace(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=0.-1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=0.-1`)
	}
}

func TestParsePointOnlyNegativeSign(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=-`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=-`)
	}
}

func TestParsePointFloatMultipleDecimals(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1.1.1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=1.1.1`)
	}
}

func TestParsePointInteger(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1i`, err)
	}
}

func TestParsePointNegativeInteger(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=-1i`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1i`, err)
	}
}

func TestParsePointNegativeFloat(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=-1.0`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0`, err)
	}
}

func TestParsePointFloatNoLeadingDigit(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=.1`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0`, err)
	}
}

func TestParsePointFloatScientific(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1.0e4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e4`, err)
	}

	pts, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1e4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e4`, err)
	}

	if pts[0].Fields()["value"] != 1e4 {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1e4`, err)
	}
}

func TestParsePointFloatScientificUpper(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1.0E4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0E4`, err)
	}

	pts, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1E4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0E4`, err)
	}

	if pts[0].Fields()["value"] != 1e4 {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1E4`, err)
	}
}

func TestParsePointFloatScientificDecimal(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=1.0e-4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e-4`, err)
	}
}

func TestParsePointFloatNegativeScientific(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=-1.0e-4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0e-4`, err)
	}
}

func TestParsePointBooleanInvalid(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=a`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=a`)
	}
}

func TestParsePointScientificIntInvalid(t *testing.T) {
	_, err := models.ParsePointsString(`cpu,host=serverA,region=us-west value=9ie10`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=9ie10`)
	}

	_, err = models.ParsePointsString(`cpu,host=serverA,region=us-west value=9e10i`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=9e10i`)
	}
}

func TestParsePointWhitespace(t *testing.T) {
	examples := []string{
		`cpu    value=1.0 1257894000000000000`,
		`cpu value=1.0     1257894000000000000`,
		`cpu      value=1.0     1257894000000000000`,
		`cpu value=1.0 1257894000000000000   `,
		`cpu value=1.0 1257894000000000000
`,
		`cpu   value=1.0 1257894000000000000
`,
	}

	expPoint := NewTestPoint("cpu", models.Tags{}, models.Fields{"value": 1.0}, time.Unix(0, 1257894000000000000))
	for i, example := range examples {
		pts, err := models.ParsePoints([]byte(example))
		if err != nil {
			t.Fatalf(`[Example %d] ParsePoints("%s") error. got %v, exp nil`, i, example, err)
		}

		if got, exp := len(pts), 1; got != exp {
			t.Fatalf("[Example %d] got %d points, expected %d", i, got, exp)
		}

		if got, exp := pts[0].Name(), expPoint.Name(); got != exp {
			t.Fatalf("[Example %d] got %v measurement, expected %v", i, got, exp)
		}

		if got, exp := len(pts[0].Fields()), len(expPoint.Fields()); got != exp {
			t.Fatalf("[Example %d] got %d fields, expected %d", i, got, exp)
		}

		if got, exp := pts[0].Fields()["value"], expPoint.Fields()["value"]; got != exp {
			t.Fatalf(`[Example %d] got %v for field "value", expected %v`, i, got, exp)
		}

		if got, exp := pts[0].Time().UnixNano(), expPoint.Time().UnixNano(); got != exp {
			t.Fatalf(`[Example %d] got %d time, expected %d`, i, got, exp)
		}
	}
}

func TestParsePointUnescape(t *testing.T) {
	// commas in measurement name
	test(t, `foo\,bar value=1i`,
		NewTestPoint(
			"foo,bar", // comma in the name
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": int64(1),
			},
			time.Unix(0, 0)))

	// comma in measurement name with tags
	test(t, `cpu\,main,regions=east value=1.0`,
		NewTestPoint(
			"cpu,main", // comma in the name
			models.NewTags(map[string]string{
				"regions": "east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in measurement name
	test(t, `cpu\ load,region=east value=1.0`,
		NewTestPoint(
			"cpu load", // space in the name
			models.NewTags(map[string]string{
				"region": "east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// equals in measurement name
	test(t, `cpu\=load,region=east value=1.0`,
		NewTestPoint(
			`cpu\=load`, // backslash is literal
			models.NewTags(map[string]string{
				"region": "east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// equals in measurement name
	test(t, `cpu=load,region=east value=1.0`,
		NewTestPoint(
			`cpu=load`, // literal equals is fine in measurement name
			models.NewTags(map[string]string{
				"region": "east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in tag names
	test(t, `cpu,region\,zone=east value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"region,zone": "east", // comma in the tag key
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in tag name
	test(t, `cpu,region\ zone=east value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"region zone": "east", // space in the tag name
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// backslash with escaped equals in tag name
	test(t, `cpu,reg\\=ion=east value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				`reg\=ion`: "east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// space is tag name
	test(t, `cpu,\ =east value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				" ": "east", // tag name is single space
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in tag values
	test(t, `cpu,regions=east\,west value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east,west", // comma in the tag value
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// backslash literal followed by escaped space
	test(t, `cpu,regions=\\ east value=1.0`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"regions": `\ east`,
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// backslash literal followed by escaped space
	test(t, `cpu,regions=eas\\ t value=1.0`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"regions": `eas\ t`,
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// backslash literal followed by trailing space
	test(t, `cpu,regions=east\\  value=1.0`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"regions": `east\ `,
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in tag values
	test(t, `cpu,regions=east\ west value=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east west", // comma in the tag value
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in field keys
	test(t, `cpu,regions=east value\,ms=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east",
			}),
			models.Fields{
				"value,ms": 1.0, // comma in the field keys
			},
			time.Unix(0, 0)))

	// spaces in field keys
	test(t, `cpu,regions=east value\ ms=1.0`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east",
			}),
			models.Fields{
				"value ms": 1.0, // comma in the field keys
			},
			time.Unix(0, 0)))

	// tag with no value
	test(t, `cpu,regions=east value="1"`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east",
				"foobar":  "",
			}),
			models.Fields{
				"value": "1",
			},
			time.Unix(0, 0)))

	// commas in field values
	test(t, `cpu,regions=east value="1,0"`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"regions": "east",
			}),
			models.Fields{
				"value": "1,0", // comma in the field value
			},
			time.Unix(0, 0)))

	// random character escaped
	test(t, `cpu,regions=eas\t value=1.0`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"regions": "eas\\t",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// backslash literal followed by escaped characters
	test(t, `cpu,regions=\\,\,\=east value=1.0`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"regions": `\,,=east`,
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// field keys using escape char.
	test(t, `cpu \a=1i`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"\\a": int64(1), // Left as parsed since it's not a known escape sequence.
			},
			time.Unix(0, 0)))

	// measurement, tag and tag value with equals
	test(t, `cpu=load,equals\=foo=tag\=value value=1i`,
		NewTestPoint(
			"cpu=load", // Not escaped
			models.NewTags(map[string]string{
				"equals=foo": "tag=value", // Tag and value unescaped
			}),
			models.Fields{
				"value": int64(1),
			},
			time.Unix(0, 0)))

}

func TestParsePointWithTags(t *testing.T) {
	test(t,
		"cpu,host=serverA,region=us-east value=1.0 1000000000",
		NewTestPoint("cpu",
			models.NewTags(map[string]string{"host": "serverA", "region": "us-east"}),
			models.Fields{"value": 1.0}, time.Unix(1, 0)))
}

func TestParsePointWithDuplicateTags(t *testing.T) {
	for i, tt := range []struct {
		line string
		err  string
	}{
		{
			line: `cpu,host=serverA,host=serverB value=1i 1000000000`,
			err:  `unable to parse 'cpu,host=serverA,host=serverB value=1i 1000000000': duplicate tags`,
		},
		{
			line: `cpu,b=2,b=1,c=3 value=1i 1000000000`,
			err:  `unable to parse 'cpu,b=2,b=1,c=3 value=1i 1000000000': duplicate tags`,
		},
		{
			line: `cpu,b=2,c=3,b=1 value=1i 1000000000`,
			err:  `unable to parse 'cpu,b=2,c=3,b=1 value=1i 1000000000': duplicate tags`,
		},
	} {
		_, err := models.ParsePointsString(tt.line)
		if err == nil || tt.err != err.Error() {
			t.Errorf("%d. ParsePoint() expected error '%s'. got '%s'", i, tt.err, err)
		}
	}
}

func TestParsePointWithStringField(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo",str2="bar" 1000000000`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   "foo",
				"str2":  "bar",
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu,host=serverA,region=us-east str="foo \" bar" 1000000000`,
		NewTestPoint("cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"str": `foo " bar`,
			},
			time.Unix(1, 0)),
	)

}

func TestParsePointWithStringWithSpaces(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo bar" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   "foo bar", // spaces in string value
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithStringWithNewline(t *testing.T) {
	test(t, "cpu,host=serverA,region=us-east value=1.0,str=\"foo\nbar\" 1000000000",
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   "foo\nbar", // newline in string value
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithStringWithCommas(t *testing.T) {
	// escaped comma
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo\,bar" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   `foo\,bar`, // commas in string value
			},
			time.Unix(1, 0)),
	)

	// non-escaped comma
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo,bar" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   "foo,bar", // commas in string value
			},
			time.Unix(1, 0)),
	)

	// string w/ trailing escape chars
	test(t, `cpu,host=serverA,region=us-east str="foo\\",str2="bar" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"str":  "foo\\", // trailing escape char
				"str2": "bar",
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointQuotedMeasurement(t *testing.T) {
	// non-escaped comma
	test(t, `"cpu",host=serverA,region=us-east value=1.0 1000000000`,
		NewTestPoint(
			`"cpu"`,
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointQuotedTags(t *testing.T) {
	test(t, `cpu,"host"="serverA",region=us-east value=1.0 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				`"host"`: `"serverA"`,
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointsUnbalancedQuotedTags(t *testing.T) {
	pts, err := models.ParsePointsString("baz,mytag=\"a x=1 1441103862125\nbaz,mytag=a z=1 1441103862126")
	if err != nil {
		t.Fatalf("ParsePoints failed: %v", err)
	}

	if exp := 2; len(pts) != exp {
		t.Fatalf("ParsePoints count mismatch. got %v, exp %v", len(pts), exp)
	}

	// Expected " in the tag value
	exp := models.MustNewPoint("baz", models.NewTags(map[string]string{"mytag": `"a`}),
		models.Fields{"x": float64(1)}, time.Unix(0, 1441103862125))

	if pts[0].String() != exp.String() {
		t.Errorf("Point mismatch:\ngot: %v\nexp: %v", pts[0].String(), exp.String())
	}

	// Expected two points to ensure we did not overscan the line
	exp = models.MustNewPoint("baz", models.NewTags(map[string]string{"mytag": `a`}),
		models.Fields{"z": float64(1)}, time.Unix(0, 1441103862126))

	if pts[1].String() != exp.String() {
		t.Errorf("Point mismatch:\ngot: %v\nexp: %v", pts[1].String(), exp.String())
	}

}

func TestParsePointEscapedStringsAndCommas(t *testing.T) {
	// non-escaped comma and quotes
	test(t, `cpu,host=serverA,region=us-east value="{Hello\"{,}\" World}" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": `{Hello"{,}" World}`,
			},
			time.Unix(1, 0)),
	)

	// escaped comma and quotes
	test(t, `cpu,host=serverA,region=us-east value="{Hello\"{\,}\" World}" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": `{Hello"{\,}" World}`,
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithStringWithEquals(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east str="foo=bar",value=1.0 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": 1.0,
				"str":   "foo=bar", // spaces in string value
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithStringWithBackslash(t *testing.T) {
	test(t, `cpu value="test\\\"" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": `test\"`,
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu value="test\\" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": `test\`,
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu value="test\\\"" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": `test\"`,
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu value="test\"" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": `test"`,
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithBoolField(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east true=true,t=t,T=T,TRUE=TRUE,True=True,false=false,f=f,F=F,FALSE=FALSE,False=False 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"t":     true,
				"T":     true,
				"true":  true,
				"True":  true,
				"TRUE":  true,
				"f":     false,
				"F":     false,
				"false": false,
				"False": false,
				"FALSE": false,
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointUnicodeString(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east value="wè" 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{
				"host":   "serverA",
				"region": "us-east",
			}),
			models.Fields{
				"value": "wè",
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointNegativeTimestamp(t *testing.T) {
	test(t, `cpu value=1 -1`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, -1)),
	)
}

func TestParsePointMaxTimestamp(t *testing.T) {
	test(t, fmt.Sprintf(`cpu value=1 %d`, models.MaxNanoTime),
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, models.MaxNanoTime)),
	)
}

func TestParsePointMinTimestamp(t *testing.T) {
	test(t, `cpu value=1 -9223372036854775806`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(0, models.MinNanoTime)),
	)
}

func TestParsePointInvalidTimestamp(t *testing.T) {
	examples := []string{
		"cpu value=1 9223372036854775808",
		"cpu value=1 -92233720368547758078",
		"cpu value=1 -",
		"cpu value=1 -/",
		"cpu value=1 -1?",
		"cpu value=1 1-",
		"cpu value=1 9223372036854775807 12",
	}

	for i, example := range examples {
		_, err := models.ParsePointsString(example)
		if err == nil {
			t.Fatalf("[Example %d] ParsePoints failed: %v", i, err)
		}
	}
}

func TestNewPointFloatWithoutDecimal(t *testing.T) {
	test(t, `cpu value=1 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(1, 0)),
	)
}
func TestNewPointNegativeFloat(t *testing.T) {
	test(t, `cpu value=-0.64 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": -0.64,
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointFloatNoDecimal(t *testing.T) {
	test(t, `cpu value=1. 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": 1.0,
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointFloatScientific(t *testing.T) {
	test(t, `cpu value=6.632243e+06 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": float64(6632243),
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointLargeInteger(t *testing.T) {
	test(t, `cpu value=6632243i 1000000000`,
		NewTestPoint(
			"cpu",
			models.NewTags(map[string]string{}),
			models.Fields{
				"value": int64(6632243), // if incorrectly encoded as a float, it would show up as 6.632243e+06
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointNaN(t *testing.T) {
	_, err := models.ParsePointsString("cpu value=NaN 1000000000")
	if err == nil {
		t.Fatalf("ParsePoints expected error, got nil")
	}

	_, err = models.ParsePointsString("cpu value=nAn 1000000000")
	if err == nil {
		t.Fatalf("ParsePoints expected error, got nil")
	}

	_, err = models.ParsePointsString("cpu value=NaN")
	if err == nil {
		t.Fatalf("ParsePoints expected error, got nil")
	}
}

func TestNewPointLargeNumberOfTags(t *testing.T) {
	tags := ""
	for i := 0; i < 255; i++ {
		tags += fmt.Sprintf(",tag%d=value%d", i, i)
	}

	pt, err := models.ParsePointsString(fmt.Sprintf("cpu%s value=1", tags))
	if err != nil {
		t.Fatalf("ParsePoints() with max tags failed: %v", err)
	}

	if len(pt[0].Tags()) != 255 {
		t.Fatalf("expected %d tags, got %d", 255, len(pt[0].Tags()))
	}
}

func TestParsePointIntsFloats(t *testing.T) {
	pts, err := models.ParsePoints([]byte(`cpu,host=serverA,region=us-east int=10i,float=11.0,float2=12.1 1000000000`))
	if err != nil {
		t.Fatalf(`ParsePoints() failed. got %s`, err)
	}

	if exp := 1; len(pts) != exp {
		t.Errorf("ParsePoint() len mismatch: got %v, exp %v", len(pts), exp)
	}
	pt := pts[0]

	if _, ok := pt.Fields()["int"].(int64); !ok {
		t.Errorf("ParsePoint() int field mismatch: got %T, exp %T", pt.Fields()["int"], int64(10))
	}

	if _, ok := pt.Fields()["float"].(float64); !ok {
		t.Errorf("ParsePoint() float field mismatch: got %T, exp %T", pt.Fields()["float64"], float64(11.0))
	}

	if _, ok := pt.Fields()["float2"].(float64); !ok {
		t.Errorf("ParsePoint() float field mismatch: got %T, exp %T", pt.Fields()["float64"], float64(12.1))
	}
}

func TestParsePointKeyUnsorted(t *testing.T) {
	pts, err := models.ParsePoints([]byte("cpu,last=1,first=2 value=1i"))
	if err != nil {
		t.Fatalf(`ParsePoints() failed. got %s`, err)
	}

	if exp := 1; len(pts) != exp {
		t.Errorf("ParsePoint() len mismatch: got %v, exp %v", len(pts), exp)
	}
	pt := pts[0]

	if exp := "cpu,first=2,last=1"; string(pt.Key()) != exp {
		t.Errorf("ParsePoint key not sorted. got %v, exp %v", string(pt.Key()), exp)
	}
}

func TestParsePointToString(t *testing.T) {
	line := `cpu,host=serverA,region=us-east bool=false,float=11,float2=12.123,int=10i,str="string val" 1000000000`
	pts, err := models.ParsePoints([]byte(line))
	if err != nil {
		t.Fatalf(`ParsePoints() failed. got %s`, err)
	}
	if exp := 1; len(pts) != exp {
		t.Errorf("ParsePoint() len mismatch: got %v, exp %v", len(pts), exp)
	}
	pt := pts[0]

	got := pt.String()
	if line != got {
		t.Errorf("ParsePoint() to string mismatch:\n got %v\n exp %v", got, line)
	}

	pt = models.MustNewPoint("cpu", models.NewTags(map[string]string{"host": "serverA", "region": "us-east"}),
		models.Fields{"int": 10, "float": float64(11.0), "float2": float64(12.123), "bool": false, "str": "string val"},
		time.Unix(1, 0))

	got = pt.String()
	if line != got {
		t.Errorf("NewPoint() to string mismatch:\n got %v\n exp %v", got, line)
	}
}

func TestParsePointsWithPrecision(t *testing.T) {
	tests := []struct {
		name      string
		line      string
		precision string
		exp       string
	}{
		{
			name:      "nanosecond by default",
			line:      `cpu,host=serverA,region=us-east value=1.0 946730096789012345`,
			precision: "",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
		},
		{
			name:      "nanosecond",
			line:      `cpu,host=serverA,region=us-east value=1.0 946730096789012345`,
			precision: "n",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
		},
		{
			name:      "microsecond",
			line:      `cpu,host=serverA,region=us-east value=1.0 946730096789012`,
			precision: "u",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012000",
		},
		{
			name:      "millisecond",
			line:      `cpu,host=serverA,region=us-east value=1.0 946730096789`,
			precision: "ms",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789000000",
		},
		{
			name:      "second",
			line:      `cpu,host=serverA,region=us-east value=1.0 946730096`,
			precision: "s",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096000000000",
		},
		{
			name:      "minute",
			line:      `cpu,host=serverA,region=us-east value=1.0 15778834`,
			precision: "m",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730040000000000",
		},
		{
			name:      "hour",
			line:      `cpu,host=serverA,region=us-east value=1.0 262980`,
			precision: "h",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946728000000000000",
		},
	}
	for _, test := range tests {
		pts, err := models.ParsePointsWithPrecision([]byte(test.line), time.Now().UTC(), test.precision)
		if err != nil {
			t.Fatalf(`%s: ParsePoints() failed. got %s`, test.name, err)
		}
		if exp := 1; len(pts) != exp {
			t.Errorf("%s: ParsePoint() len mismatch: got %v, exp %v", test.name, len(pts), exp)
		}
		pt := pts[0]

		got := pt.String()
		if got != test.exp {
			t.Errorf("%s: ParsePoint() to string mismatch:\n got %v\n exp %v", test.name, got, test.exp)
		}
	}
}

func TestParsePointsWithPrecisionNoTime(t *testing.T) {
	line := `cpu,host=serverA,region=us-east value=1.0`
	tm, _ := time.Parse(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z")
	tests := []struct {
		name      string
		precision string
		exp       string
	}{
		{
			name:      "no precision",
			precision: "",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
		},
		{
			name:      "nanosecond precision",
			precision: "n",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
		},
		{
			name:      "microsecond precision",
			precision: "u",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012000",
		},
		{
			name:      "millisecond precision",
			precision: "ms",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789000000",
		},
		{
			name:      "second precision",
			precision: "s",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096000000000",
		},
		{
			name:      "minute precision",
			precision: "m",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730040000000000",
		},
		{
			name:      "hour precision",
			precision: "h",
			exp:       "cpu,host=serverA,region=us-east value=1.0 946728000000000000",
		},
	}

	for _, test := range tests {
		pts, err := models.ParsePointsWithPrecision([]byte(line), tm, test.precision)
		if err != nil {
			t.Fatalf(`%s: ParsePoints() failed. got %s`, test.name, err)
		}
		if exp := 1; len(pts) != exp {
			t.Errorf("%s: ParsePoint() len mismatch: got %v, exp %v", test.name, len(pts), exp)
		}
		pt := pts[0]

		got := pt.String()
		if got != test.exp {
			t.Errorf("%s: ParsePoint() to string mismatch:\n got %v\n exp %v", test.name, got, test.exp)
		}
	}
}

func TestParsePointsWithPrecisionComments(t *testing.T) {
	tests := []struct {
		name      string
		batch     string
		exp       string
		lenPoints int
	}{
		{
			name:      "comment only",
			batch:     `# comment only`,
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
			lenPoints: 0,
		},
		{
			name: "point with comment above",
			batch: `# a point is below
cpu,host=serverA,region=us-east value=1.0 946730096789012345`,
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
			lenPoints: 1,
		},
		{
			name: "point with comment below",
			batch: `cpu,host=serverA,region=us-east value=1.0 946730096789012345
# end of points`,
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
			lenPoints: 1,
		},
		{
			name: "indented comment",
			batch: `	# a point is below
cpu,host=serverA,region=us-east value=1.0 946730096789012345`,
			exp:       "cpu,host=serverA,region=us-east value=1.0 946730096789012345",
			lenPoints: 1,
		},
	}
	for _, test := range tests {
		pts, err := models.ParsePointsWithPrecision([]byte(test.batch), time.Now().UTC(), "")
		if err != nil {
			t.Fatalf(`%s: ParsePoints() failed. got %s`, test.name, err)
		}
		pointsLength := len(pts)
		if exp := test.lenPoints; pointsLength != exp {
			t.Errorf("%s: ParsePoint() len mismatch: got %v, exp %v", test.name, pointsLength, exp)
		}

		if pointsLength > 0 {
			pt := pts[0]

			got := pt.String()
			if got != test.exp {
				t.Errorf("%s: ParsePoint() to string mismatch:\n got %v\n exp %v", test.name, got, test.exp)
			}
		}
	}
}

func TestNewPointEscaped(t *testing.T) {
	// commas
	pt := models.MustNewPoint("cpu,main", models.NewTags(map[string]string{"tag,bar": "value"}), models.Fields{"name,bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu\,main,tag\,bar=value name\,bar=1 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// spaces
	pt = models.MustNewPoint("cpu main", models.NewTags(map[string]string{"tag bar": "value"}), models.Fields{"name bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu\ main,tag\ bar=value name\ bar=1 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// equals
	pt = models.MustNewPoint("cpu=main", models.NewTags(map[string]string{"tag=bar": "value=foo"}), models.Fields{"name=bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu=main,tag\=bar=value\=foo name\=bar=1 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}
}

func TestNewPointWithoutField(t *testing.T) {
	_, err := models.NewPoint("cpu", models.NewTags(map[string]string{"tag": "bar"}), models.Fields{}, time.Unix(0, 0))
	if err == nil {
		t.Fatalf(`NewPoint() expected error. got nil`)
	}
}

func TestNewPointUnhandledType(t *testing.T) {
	// nil value
	pt := models.MustNewPoint("cpu", nil, models.Fields{"value": nil}, time.Unix(0, 0))
	if exp := `cpu value= 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// unsupported type gets stored as string
	now := time.Unix(0, 0).UTC()
	pt = models.MustNewPoint("cpu", nil, models.Fields{"value": now}, time.Unix(0, 0))
	if exp := `cpu value="1970-01-01 00:00:00 +0000 UTC" 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	if exp := "1970-01-01 00:00:00 +0000 UTC"; pt.Fields()["value"] != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}
}

func TestMakeKeyEscaped(t *testing.T) {
	if exp, got := `cpu\ load`, models.MakeKey([]byte(`cpu\ load`), models.NewTags(map[string]string{})); string(got) != exp {
		t.Errorf("MakeKey() mismatch.\ngot %v\nexp %v", got, exp)
	}

	if exp, got := `cpu\ load`, models.MakeKey([]byte(`cpu load`), models.NewTags(map[string]string{})); string(got) != exp {
		t.Errorf("MakeKey() mismatch.\ngot %v\nexp %v", got, exp)
	}

	if exp, got := `cpu\,load`, models.MakeKey([]byte(`cpu\,load`), models.NewTags(map[string]string{})); string(got) != exp {
		t.Errorf("MakeKey() mismatch.\ngot %v\nexp %v", got, exp)
	}

	if exp, got := `cpu\,load`, models.MakeKey([]byte(`cpu,load`), models.NewTags(map[string]string{})); string(got) != exp {
		t.Errorf("MakeKey() mismatch.\ngot %v\nexp %v", got, exp)
	}

}

func TestPrecisionString(t *testing.T) {
	tags := map[string]interface{}{"value": float64(1)}
	tm, _ := time.Parse(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z")
	tests := []struct {
		name      string
		precision string
		exp       string
	}{
		{
			name:      "no precision",
			precision: "",
			exp:       "cpu value=1 946730096789012345",
		},
		{
			name:      "nanosecond precision",
			precision: "ns",
			exp:       "cpu value=1 946730096789012345",
		},
		{
			name:      "microsecond precision",
			precision: "u",
			exp:       "cpu value=1 946730096789012",
		},
		{
			name:      "millisecond precision",
			precision: "ms",
			exp:       "cpu value=1 946730096789",
		},
		{
			name:      "second precision",
			precision: "s",
			exp:       "cpu value=1 946730096",
		},
		{
			name:      "minute precision",
			precision: "m",
			exp:       "cpu value=1 15778834",
		},
		{
			name:      "hour precision",
			precision: "h",
			exp:       "cpu value=1 262980",
		},
	}

	for _, test := range tests {
		pt := models.MustNewPoint("cpu", nil, tags, tm)
		act := pt.PrecisionString(test.precision)

		if act != test.exp {
			t.Errorf("%s: PrecisionString() mismatch:\n actual:	%v\n exp:		%v",
				test.name, act, test.exp)
		}
	}
}

func TestRoundedString(t *testing.T) {
	tags := map[string]interface{}{"value": float64(1)}
	tm, _ := time.Parse(time.RFC3339Nano, "2000-01-01T12:34:56.789012345Z")
	tests := []struct {
		name      string
		precision time.Duration
		exp       string
	}{
		{
			name:      "no precision",
			precision: time.Duration(0),
			exp:       "cpu value=1 946730096789012345",
		},
		{
			name:      "nanosecond precision",
			precision: time.Nanosecond,
			exp:       "cpu value=1 946730096789012345",
		},
		{
			name:      "microsecond precision",
			precision: time.Microsecond,
			exp:       "cpu value=1 946730096789012000",
		},
		{
			name:      "millisecond precision",
			precision: time.Millisecond,
			exp:       "cpu value=1 946730096789000000",
		},
		{
			name:      "second precision",
			precision: time.Second,
			exp:       "cpu value=1 946730097000000000",
		},
		{
			name:      "minute precision",
			precision: time.Minute,
			exp:       "cpu value=1 946730100000000000",
		},
		{
			name:      "hour precision",
			precision: time.Hour,
			exp:       "cpu value=1 946731600000000000",
		},
	}

	for _, test := range tests {
		pt := models.MustNewPoint("cpu", nil, tags, tm)
		act := pt.RoundedString(test.precision)

		if act != test.exp {
			t.Errorf("%s: RoundedString() mismatch:\n actual:	%v\n exp:		%v",
				test.name, act, test.exp)
		}
	}
}

func TestParsePointsStringWithExtraBuffer(t *testing.T) {
	b := make([]byte, 70*5000)
	buf := bytes.NewBuffer(b)
	key := "cpu,host=A,region=uswest"
	buf.WriteString(fmt.Sprintf("%s value=%.3f 1\n", key, rand.Float64()))

	points, err := models.ParsePointsString(buf.String())
	if err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	pointKey := string(points[0].Key())

	if len(key) != len(pointKey) {
		t.Fatalf("expected length of both keys are same but got %d and %d", len(key), len(pointKey))
	}

	if key != pointKey {
		t.Fatalf("expected both keys are same but got %s and %s", key, pointKey)
	}
}

func TestParsePointsQuotesInFieldKey(t *testing.T) {
	buf := `cpu "a=1
cpu value=2 1`
	points, err := models.ParsePointsString(buf)
	if err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	pointFields := points[0].Fields()
	value, ok := pointFields["\"a"]
	if !ok {
		t.Fatalf("expected to parse field '\"a'")
	}

	if value != float64(1) {
		t.Fatalf("expected field value to be 1, got %v", value)
	}

	// The following input should not parse
	buf = `cpu "\, '= "\ v=1.0`
	_, err = models.ParsePointsString(buf)
	if err == nil {
		t.Fatalf("expected parsing failure but got no error")
	}
}

func TestParsePointsQuotesInTags(t *testing.T) {
	buf := `t159,label=hey\ "ya a=1i,value=0i
t159,label=another a=2i,value=1i 1`
	points, err := models.ParsePointsString(buf)
	if err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	if len(points) != 2 {
		t.Fatalf("expected 2 points, got %d", len(points))
	}
}

func TestParsePointsBlankLine(t *testing.T) {
	buf := `cpu value=1i 1000000000

cpu value=2i 2000000000`
	points, err := models.ParsePointsString(buf)
	if err != nil {
		t.Fatalf("failed to write points: %s", err.Error())
	}

	if len(points) != 2 {
		t.Fatalf("expected 2 points, got %d", len(points))
	}
}

func TestNewPointsWithBytesWithCorruptData(t *testing.T) {
	corrupted := []byte{0, 0, 0, 3, 102, 111, 111, 0, 0, 0, 4, 61, 34, 65, 34, 1, 0, 0, 0, 14, 206, 86, 119, 24, 32, 72, 233, 168, 2, 148}
	p, err := models.NewPointFromBytes(corrupted)
	if p != nil || err == nil {
		t.Fatalf("NewPointFromBytes: got: (%v, %v), expected: (nil, error)", p, err)
	}
}

func TestNewPointsRejectsEmptyFieldNames(t *testing.T) {
	if _, err := models.NewPoint("foo", nil, models.Fields{"": 1}, time.Now()); err == nil {
		t.Fatalf("new point with empty field name. got: nil, expected: error")
	}
}

func TestNewPointsRejectsMaxKey(t *testing.T) {
	var key string
	for i := 0; i < 65536; i++ {
		key += "a"
	}

	if _, err := models.NewPoint(key, nil, models.Fields{"value": 1}, time.Now()); err == nil {
		t.Fatalf("new point with max key. got: nil, expected: error")
	}

	if _, err := models.ParsePointsString(fmt.Sprintf("%v value=1", key)); err == nil {
		t.Fatalf("parse point with max key. got: nil, expected: error")
	}
}

func TestParseKeyEmpty(t *testing.T) {
	if _, _, err := models.ParseKey(nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPoint_FieldIterator_Simple(t *testing.T) {

	p, err := models.ParsePoints([]byte(`m v=42i,f=42 36`))
	if err != nil {
		t.Fatal(err)
	}

	if len(p) != 1 {
		t.Fatalf("wrong number of points, got %d, exp %d", len(p), 1)
	}

	fi := p[0].FieldIterator()

	if !fi.Next() {
		t.Fatal("field iterator terminated before first field")
	}

	if fi.Type() != models.Integer {
		t.Fatalf("'42i' should be an Integer, got %v", fi.Type())
	}

	if fi.IntegerValue() != 42 {
		t.Fatalf("'42i' should be 42, got %d", fi.IntegerValue())
	}

	if !fi.Next() {
		t.Fatalf("field iterator terminated before second field")
	}

	if fi.Type() != models.Float {
		t.Fatalf("'42' should be a Float, got %v", fi.Type())
	}

	if fi.FloatValue() != 42.0 {
		t.Fatalf("'42' should be %f, got %f", 42.0, fi.FloatValue())
	}

	if fi.Next() {
		t.Fatal("field iterator didn't terminate")
	}
}

func toFields(fi models.FieldIterator) models.Fields {
	m := make(models.Fields)
	for fi.Next() {
		var v interface{}
		switch fi.Type() {
		case models.Float:
			v = fi.FloatValue()
		case models.Integer:
			v = fi.IntegerValue()
		case models.String:
			v = fi.StringValue()
		case models.Boolean:
			v = fi.BooleanValue()
		case models.Empty:
			v = nil
		default:
			panic("unknown type")
		}
		m[string(fi.FieldKey())] = v
	}
	return m
}

func TestPoint_FieldIterator_FieldMap(t *testing.T) {

	points, err := models.ParsePointsString(`
m v=42
m v=42i
m v="string"
m v=true
m v="string\"with\"escapes"
m v=42i,f=42,g=42.314
m a=2i,b=3i,c=true,d="stuff",e=-0.23,f=123.456
`)

	if err != nil {
		t.Fatal("failed to parse test points:", err)
	}

	for _, p := range points {
		exp := p.Fields()
		got := toFields(p.FieldIterator())

		if !reflect.DeepEqual(got, exp) {
			t.Errorf("FieldIterator failed for %#q: got %#v, exp %#v", p.String(), got, exp)
		}
	}
}

func TestPoint_FieldIterator_Delete_Begin(t *testing.T) {
	points, err := models.ParsePointsString(`m a=1,b=2,c=3`)
	if err != nil || len(points) != 1 {
		t.Fatal("failed parsing point")
	}

	fi := points[0].FieldIterator()
	fi.Next() // a
	fi.Delete()

	fi.Reset()

	got := toFields(fi)
	exp := models.Fields{"b": float64(2), "c": float64(3)}

	if !reflect.DeepEqual(got, exp) {
		t.Fatalf("Delete failed, got %#v, exp %#v", got, exp)
	}
}

func TestPoint_FieldIterator_Delete_Middle(t *testing.T) {
	points, err := models.ParsePointsString(`m a=1,b=2,c=3`)
	if err != nil || len(points) != 1 {
		t.Fatal("failed parsing point")
	}

	fi := points[0].FieldIterator()
	fi.Next() // a
	fi.Next() // b
	fi.Delete()

	fi.Reset()

	got := toFields(fi)
	exp := models.Fields{"a": float64(1), "c": float64(3)}

	if !reflect.DeepEqual(got, exp) {
		t.Fatalf("Delete failed, got %#v, exp %#v", got, exp)
	}
}

func TestPoint_FieldIterator_Delete_End(t *testing.T) {
	points, err := models.ParsePointsString(`m a=1,b=2,c=3`)
	if err != nil || len(points) != 1 {
		t.Fatal("failed parsing point")
	}

	fi := points[0].FieldIterator()
	fi.Next() // a
	fi.Next() // b
	fi.Next() // c
	fi.Delete()

	fi.Reset()

	got := toFields(fi)
	exp := models.Fields{"a": float64(1), "b": float64(2)}

	if !reflect.DeepEqual(got, exp) {
		t.Fatalf("Delete failed, got %#v, exp %#v", got, exp)
	}
}

func TestPoint_FieldIterator_Delete_Nothing(t *testing.T) {
	points, err := models.ParsePointsString(`m a=1,b=2,c=3`)
	if err != nil || len(points) != 1 {
		t.Fatal("failed parsing point")
	}

	fi := points[0].FieldIterator()
	fi.Delete()

	fi.Reset()

	got := toFields(fi)
	exp := models.Fields{"a": float64(1), "b": float64(2), "c": float64(3)}

	if !reflect.DeepEqual(got, exp) {
		t.Fatalf("Delete failed, got %#v, exp %#v", got, exp)
	}
}

func TestPoint_FieldIterator_Delete_Twice(t *testing.T) {
	points, err := models.ParsePointsString(`m a=1,b=2,c=3`)
	if err != nil || len(points) != 1 {
		t.Fatal("failed parsing point")
	}

	fi := points[0].FieldIterator()
	fi.Next() // a
	fi.Next() // b
	fi.Delete()
	fi.Delete() // no-op

	fi.Reset()

	got := toFields(fi)
	exp := models.Fields{"a": float64(1), "c": float64(3)}

	if !reflect.DeepEqual(got, exp) {
		t.Fatalf("Delete failed, got %#v, exp %#v", got, exp)
	}
}
