package tsdb

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

var (
	tags       = Tags{"foo": "bar", "apple": "orange", "host": "serverA", "region": "uswest"}
	maxFloat64 = strconv.FormatFloat(math.MaxFloat64, 'f', 1, 64)
	minFloat64 = strconv.FormatFloat(-math.MaxFloat64, 'f', 1, 64)
)

func TestMarshal(t *testing.T) {
	got := tags.hashKey()
	if exp := ",apple=orange,foo=bar,host=serverA,region=uswest"; string(got) != exp {
		t.Log("got: ", string(got))
		t.Log("exp: ", exp)
		t.Error("invalid match")
	}
}

func BenchmarkMarshal(b *testing.B) {
	for i := 0; i < b.N; i++ {
		tags.hashKey()
	}
}

func BenchmarkParsePointNoTags(b *testing.B) {
	line := `cpu value=1 1000000000`
	for i := 0; i < b.N; i++ {
		ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted2(b *testing.B) {
	line := `cpu,host=serverA,region=us-west value=1 1000000000`
	for i := 0; i < b.N; i++ {
		ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted5(b *testing.B) {
	line := `cpu,env=prod,host=serverA,region=us-west,target=servers,zone=1c value=1 1000000000`
	for i := 0; i < b.N; i++ {
		ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsSorted10(b *testing.B) {
	line := `cpu,env=prod,host=serverA,region=us-west,tag1=value1,tag2=value2,tag3=value3,tag4=value4,tag5=value5,target=servers,zone=1c value=1 1000000000`
	for i := 0; i < b.N; i++ {
		ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
	}
}

func BenchmarkParsePointsTagsUnSorted2(b *testing.B) {
	line := `cpu,region=us-west,host=serverA value=1 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func BenchmarkParsePointsTagsUnSorted5(b *testing.B) {
	line := `cpu,region=us-west,host=serverA,env=prod,target=servers,zone=1c value=1 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func BenchmarkParsePointsTagsUnSorted10(b *testing.B) {
	line := `cpu,region=us-west,host=serverA,env=prod,target=servers,zone=1c,tag1=value1,tag2=value2,tag3=value3,tag4=value4,tag5=value5 value=1 1000000000`
	for i := 0; i < b.N; i++ {
		pt, _ := ParsePoints([]byte(line))
		b.SetBytes(int64(len(line)))
		pt[0].Key()
	}
}

func test(t *testing.T, line string, point Point) {
	pts, err := ParsePointsWithPrecision([]byte(line), time.Unix(0, 0), "n")
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

	for tag, value := range point.Tags() {
		if pts[0].Tags()[tag] != value {
			t.Errorf(`ParsePoints("%s") tags mismatch. got %v, exp %v`, line, pts[0].Tags()[tag], value)
		}
	}

	for name, value := range point.Fields() {
		val := pts[0].Fields()[name]
		expfval, ok := val.(float64)

		if ok && math.IsNaN(expfval) {
			gotfval, ok := value.(float64)
			if ok && !math.IsNaN(gotfval) {
				t.Errorf(`ParsePoints("%s") field '%s' mismatch. exp NaN`, line, name)
			}
		} else if !reflect.DeepEqual(pts[0].Fields()[name], value) {
			t.Errorf(`ParsePoints("%s") field '%s' mismatch. got %v, exp %v`, line, name, pts[0].Fields()[name], value)
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
	pts, err := ParsePointsString("")
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, "", err)
	}

	if exp := 0; len(pts) != exp {
		t.Errorf(`ParsePoints("%s") len mismatch. got %v, exp %vr`, "", len(pts), exp)
	}
}

func TestParsePointNoFields(t *testing.T) {
	_, err := ParsePointsString("cpu")
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, "cpu")
	}

	_, err = ParsePointsString("cpu,")
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, "cpu,")
	}

	_, err = ParsePointsString("cpu, value=1")
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, "cpu, value=1")
	}

	_, err = ParsePointsString("cpu,,, value=1")
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, "cpu,,, value=1")
	}

}

func TestParsePointNoTimestamp(t *testing.T) {
	test(t, "cpu value=1", NewPoint("cpu", nil, nil, time.Unix(0, 0)))
}

func TestParsePointMissingQuote(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA value="test`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, "cpu")
	}
}

func TestParsePointMissingTagName(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,=us-east value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,=us-east value=1`)
	}

	_, err = ParsePointsString(`cpu,host=serverAa\,,=us-east value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverAa\,,=us-east value=1`)
	}

	_, err = ParsePointsString(`cpu,host=serverA\,,=us-east value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA\,,=us-east value=1`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,\ =us-east value=1`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,\ =us-east value=1`, err)
	}
}

func TestParsePointMissingTagValue(t *testing.T) {
	_, err := ParsePointsString(`cpu,host value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host value=1`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,region value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region value=1`)
	}
	_, err = ParsePointsString(`cpu,host=serverA,region= value=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region= value=1`)
	}
}

func TestParsePointMissingFieldName(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west =`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west =`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,region=us-west =123`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west =123`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,region=us-west a\ =123`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west a\ =123`)
	}
	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=123,=456`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=123,=456`)
	}

}

func TestParsePointMissingFieldValue(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value= 1000000000`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value= 1000000000`)
	}

	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=,value2=1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=,value2=1`)
	}

	_, err = ParsePointsString(`cpu,host=server01,region=us-west 1434055562000000000`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=server01,region=us-west 1434055562000000000`)
	}

	_, err = ParsePointsString(`cpu,host=server01,region=us-west value=1,b`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=server01,region=us-west value=1,b`)
	}
}

func TestParsePointBadNumber(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1a`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=1a`)
	}
}

func TestParsePointMaxInt64(t *testing.T) {
	// out of range
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=9223372036854775808`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=9223372036854775808`)
	}

	// max int
	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=9223372036854775807`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=9223372036854775807`, err)
	}

	// leading zeros
	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=0009223372036854775807`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=0009223372036854775807`, err)
	}
}

func TestParsePointMinInt64(t *testing.T) {
	// out of range
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=-9223372036854775809`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=-9223372036854775809`)
	}

	// min int
	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=-9223372036854775808`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-9223372036854775808`, err)
	}

	// leading zeros
	_, err = ParsePointsString(`cpu,host=serverA,region=us-west value=-0009223372036854775808`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-0009223372036854775808`, err)
	}
}

func TestParsePointMaxFloat64(t *testing.T) {
	// out of range
	_, err := ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "1"+string(maxFloat64)))
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=...`)
	}

	// max float
	_, err = ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, string(maxFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=9223372036854775807`, err)
	}

	// leading zeros
	_, err = ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "0000"+string(maxFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=0009223372036854775807`, err)
	}
}

func TestParsePointMinFloat64(t *testing.T) {
	// out of range
	_, err := ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "-1"+string(minFloat64)[1:]))
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=...`)
	}

	// min float
	_, err = ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, string(minFloat64)))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=...`, err)
	}

	// leading zeros
	_, err = ParsePointsString(fmt.Sprintf(`cpu,host=serverA,region=us-west value=%s`, "-0000000"+string(minFloat64)[1:]))
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=...`, err)
	}
}

func TestParsePointNumberNonNumeric(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=.1a`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=.1a`)
	}
}

func TestParsePointNegativeWrongPlace(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=0.-1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=0.-1`)
	}
}

func TestParsePointFloatMultipleDecimals(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1.1.1`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=1.1.1`)
	}
	println(err.Error())
}

func TestParsePointInteger(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1`, err)
	}
}

func TestParsePointNegativeInteger(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=-1`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1`, err)
	}
}

func TestParsePointNegativeFloat(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=-1.0`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0`, err)
	}
}

func TestParsePointFloatNoLeadingDigit(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=.1`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0`, err)
	}
}

func TestParsePointFloatScientific(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1.0e4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e4`, err)
	}

	pts, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1e4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e4`, err)
	}

	if pts[0].Fields()["value"] != 1e4 {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1e4`, err)
	}

}

func TestParsePointFloatScientificDecimal(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=1.0e-4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=1.0e-4`, err)
	}
}

func TestParsePointFloatNegativeScientific(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=-1.0e-4`)
	if err != nil {
		t.Errorf(`ParsePoints("%s") mismatch. got %v, exp nil`, `cpu,host=serverA,region=us-west value=-1.0e-4`, err)
	}
}

func TestParsePointBooleanInvalid(t *testing.T) {
	_, err := ParsePointsString(`cpu,host=serverA,region=us-west value=a`)
	if err == nil {
		t.Errorf(`ParsePoints("%s") mismatch. got nil, exp error`, `cpu,host=serverA,region=us-west value=a`)
	}
}

func TestParsePointUnescape(t *testing.T) {
	test(t, `foo\,bar value=1`,
		NewPoint(
			"foo,bar", // comma in the name
			Tags{},
			Fields{
				"value": 1,
			},
			time.Unix(0, 0)))

	// commas in measuremnt name
	test(t, `cpu\,main,regions=east\,west value=1.0`,
		NewPoint(
			"cpu,main", // comma in the name
			Tags{
				"regions": "east,west",
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in measurement name
	test(t, `cpu\ load,region=east value=1.0`,
		NewPoint(
			"cpu load", // space in the name
			Tags{
				"region": "east",
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in tag names
	test(t, `cpu,region\,zone=east value=1.0`,
		NewPoint("cpu",
			Tags{
				"region,zone": "east", // comma in the tag name
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in tag names
	test(t, `cpu,region\ zone=east value=1.0`,
		NewPoint("cpu",
			Tags{
				"region zone": "east", // comma in the tag name
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in tag values
	test(t, `cpu,regions=east\,west value=1.0`,
		NewPoint("cpu",
			Tags{
				"regions": "east,west", // comma in the tag value
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// spaces in tag values
	test(t, `cpu,regions=east\ west value=1.0`,
		NewPoint("cpu",
			Tags{
				"regions": "east west", // comma in the tag value
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))

	// commas in field names
	test(t, `cpu,regions=east value\,ms=1.0`,
		NewPoint("cpu",
			Tags{
				"regions": "east",
			},
			Fields{
				"value,ms": 1.0, // comma in the field name
			},
			time.Unix(0, 0)))

	// spaces in field names
	test(t, `cpu,regions=east value\ ms=1.0`,
		NewPoint("cpu",
			Tags{
				"regions": "east",
			},
			Fields{
				"value ms": 1.0, // comma in the field name
			},
			time.Unix(0, 0)))

	// commas in field values
	test(t, `cpu,regions=east value="1,0"`,
		NewPoint("cpu",
			Tags{
				"regions": "east",
			},
			Fields{
				"value": "1,0", // comma in the field value
			},
			time.Unix(0, 0)))

	// random character escaped
	test(t, `cpu,regions=eas\t value=1.0`,
		NewPoint(
			"cpu",
			Tags{
				"regions": "eas\\t",
			},
			Fields{
				"value": 1.0,
			},
			time.Unix(0, 0)))
}

func TestParsePointWithTags(t *testing.T) {
	test(t,
		"cpu,host=serverA,region=us-east value=1.0 1000000000",
		NewPoint("cpu",
			Tags{"host": "serverA", "region": "us-east"},
			Fields{"value": 1.0}, time.Unix(1, 0)))
}

func TestParsPointWithDuplicateTags(t *testing.T) {
	_, err := ParsePoints([]byte(`cpu,host=serverA,host=serverB value=1 1000000000`))
	if err == nil {
		t.Fatalf(`ParsePoint() expected error. got nil`)
	}
}

func TestParsePointWithStringField(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo",str2="bar" 1000000000`,
		NewPoint("cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": 1.0,
				"str":   "foo",
				"str2":  "bar",
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu,host=serverA,region=us-east str="foo \" bar" 1000000000`,
		NewPoint("cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"str": `foo " bar`,
			},
			time.Unix(1, 0)),
	)

}

func TestParsePointWithStringWithSpaces(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo bar" 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": 1.0,
				"str":   "foo bar", // spaces in string value
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithStringWithCommas(t *testing.T) {
	// escaped comma
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo\,bar" 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": 1.0,
				"str":   `foo\,bar`, // commas in string value
			},
			time.Unix(1, 0)),
	)

	// non-escaped comma
	test(t, `cpu,host=serverA,region=us-east value=1.0,str="foo,bar" 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": 1.0,
				"str":   "foo,bar", // commas in string value
			},
			time.Unix(1, 0)),
	)

}

func TestParsePointEscapedStringsAndCommas(t *testing.T) {
	// non-escaped comma and quotes
	test(t, `cpu,host=serverA,region=us-east value="{Hello\"{,}\" World}" 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": `{Hello"{,}" World}`,
			},
			time.Unix(1, 0)),
	)

	// escaped comma and quotes
	test(t, `cpu,host=serverA,region=us-east value="{Hello\"{\,}\" World}" 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": `{Hello"{\,}" World}`,
			},
			time.Unix(1, 0)),
	)

}

func TestParsePointWithStringWithEquals(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east str="foo=bar",value=1.0 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": 1.0,
				"str":   "foo=bar", // spaces in string value
			},
			time.Unix(1, 0)),
	)
}

func TestParsePointWithBoolField(t *testing.T) {
	test(t, `cpu,host=serverA,region=us-east true=true,t=t,T=T,TRUE=TRUE,True=True,false=false,f=f,F=F,FALSE=FALSE,False=False 1000000000`,
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
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
		NewPoint(
			"cpu",
			Tags{
				"host":   "serverA",
				"region": "us-east",
			},
			Fields{
				"value": "wè",
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointNegativeFloat(t *testing.T) {
	test(t, `cpu value=-0.64 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": -0.64,
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointFloatNoDecimal(t *testing.T) {
	test(t, `cpu value=1. 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": 1.0,
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointFloatScientific(t *testing.T) {
	test(t, `cpu value=6.632243e+06 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": float64(6632243),
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointLargeInteger(t *testing.T) {
	test(t, `cpu value=6632243 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": 6632243, // if incorrectly encoded as a float, it would show up as 6.632243e+06
			},
			time.Unix(1, 0)),
	)
}

func TestNewPointNaN(t *testing.T) {
	test(t, `cpu value=NaN 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": math.NaN(),
			},
			time.Unix(1, 0)),
	)

	test(t, `cpu value=nAn 1000000000`,
		NewPoint(
			"cpu",
			Tags{},
			Fields{
				"value": math.NaN(),
			},
			time.Unix(1, 0)),
	)

}

func TestParsePointIntsFloats(t *testing.T) {
	pts, err := ParsePoints([]byte(`cpu,host=serverA,region=us-east int=10,float=11.0,float2=12.1 1000000000`))
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
	pts, err := ParsePoints([]byte("cpu,last=1,first=2 value=1"))
	if err != nil {
		t.Fatalf(`ParsePoints() failed. got %s`, err)
	}

	if exp := 1; len(pts) != exp {
		t.Errorf("ParsePoint() len mismatch: got %v, exp %v", len(pts), exp)
	}
	pt := pts[0]

	if exp := "cpu,first=2,last=1"; string(pt.Key()) != exp {
		t.Errorf("ParsePoint key not sorted. got %v, exp %v", pt.Key(), exp)
	}
}

func TestParsePointToString(t *testing.T) {
	line := `cpu,host=serverA,region=us-east bool=false,float=11.0,float2=12.123,int=10,str="string val" 1000000000`
	pts, err := ParsePoints([]byte(line))
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

	pt = NewPoint("cpu", Tags{"host": "serverA", "region": "us-east"},
		Fields{"int": 10, "float": float64(11.0), "float2": float64(12.123), "bool": false, "str": "string val"},
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
		pts, err := ParsePointsWithPrecision([]byte(test.line), time.Now().UTC(), test.precision)
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
		pts, err := ParsePointsWithPrecision([]byte(line), tm, test.precision)
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
		pts, err := ParsePointsWithPrecision([]byte(test.batch), time.Now().UTC(), "")
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
	pt := NewPoint("cpu,main", Tags{"tag,bar": "value"}, Fields{"name,bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu\,main,tag\,bar=value name\,bar=1.0 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// spaces
	pt = NewPoint("cpu main", Tags{"tag bar": "value"}, Fields{"name bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu\ main,tag\ bar=value name\ bar=1.0 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// equals
	pt = NewPoint("cpu=main", Tags{"tag=bar": "value=foo"}, Fields{"name=bar": 1.0}, time.Unix(0, 0))
	if exp := `cpu\=main,tag\=bar=value\=foo name\=bar=1.0 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}
}

func TestNewPointUnhandledType(t *testing.T) {
	// nil value
	pt := NewPoint("cpu", nil, Fields{"value": nil}, time.Unix(0, 0))
	if exp := `cpu value= 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	// unsupported type gets stored as string
	now := time.Unix(0, 0).UTC()
	pt = NewPoint("cpu", nil, Fields{"value": now}, time.Unix(0, 0))
	if exp := `cpu value="1970-01-01 00:00:00 +0000 UTC" 0`; pt.String() != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}

	if exp := "1970-01-01 00:00:00 +0000 UTC"; pt.Fields()["value"] != exp {
		t.Errorf("NewPoint().String() mismatch.\ngot %v\nexp %v", pt.String(), exp)
	}
}
