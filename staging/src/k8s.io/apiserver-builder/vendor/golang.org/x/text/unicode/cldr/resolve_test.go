package cldr

import (
	"fmt"
	"log"
	"reflect"
	"testing"
)

func failOnError(err error) {
	if err != nil {
		log.Panic(err)
	}
}

func data() *CLDR {
	d := Decoder{}
	data, err := d.Decode(testLoader{})
	failOnError(err)
	return data
}

type h struct {
	A string `xml:"ha,attr"`
	E string `xml:"he"`
	D string `xml:",chardata"`
	X string
}

type fieldTest struct {
	Common
	To  string `xml:"to,attr"`
	Key string `xml:"key,attr"`
	E   string `xml:"e"`
	D   string `xml:",chardata"`
	X   string
	h
}

var testStruct = fieldTest{
	Common: Common{
		name: "mapping", // exclude "type" as distinguising attribute
		Type: "foo",
		Alt:  "foo",
	},
	To:  "nyc",
	Key: "k",
	E:   "E",
	D:   "D",
	h: h{
		A: "A",
		E: "E",
		D: "D",
	},
}

func TestIter(t *testing.T) {
	tests := map[string]string{
		"Type":  "foo",
		"Alt":   "foo",
		"To":    "nyc",
		"A":     "A",
		"Alias": "<nil>",
	}
	k := 0
	for i := iter(reflect.ValueOf(testStruct)); !i.done(); i.next() {
		v := i.value()
		if v.Kind() == reflect.Ptr && v.Elem().Kind() == reflect.String {
			v = v.Elem()
		}
		name := i.field().Name
		if w, ok := tests[name]; ok {
			s := fmt.Sprint(v.Interface())
			if w != s {
				t.Errorf("value: found %q; want %q", w, s)
			}
			delete(tests, name)
		}
		k++
	}
	if len(tests) != 0 {
		t.Errorf("missing fields: %v", tests)
	}
}

func TestFindField(t *testing.T) {
	tests := []struct {
		name, val string
		exist     bool
	}{
		{"type", "foo", true},
		{"alt", "foo", true},
		{"to", "nyc", true},
		{"he", "E", true},
		{"q", "", false},
	}
	vf := reflect.ValueOf(testStruct)
	for i, tt := range tests {
		v, err := findField(vf, tt.name)
		if (err == nil) != tt.exist {
			t.Errorf("%d: field %q present is %v; want %v", i, tt.name, err == nil, tt.exist)
		} else if tt.exist {
			if v.Kind() == reflect.Ptr {
				if v.IsNil() {
					continue
				}
				v = v.Elem()
			}
			if v.String() != tt.val {
				t.Errorf("%d: found value %q; want %q", i, v.String(), tt.val)
			}
		}
	}
}

var keyTests = []struct {
	exclude []string
	key     string
}{
	{[]string{}, "alt=foo;key=k;to=nyc"},
	{[]string{"type"}, "alt=foo;key=k;to=nyc"},
	{[]string{"choice"}, "alt=foo;key=k;to=nyc"},
	{[]string{"alt"}, "key=k;to=nyc"},
	{[]string{"a"}, "alt=foo;key=k;to=nyc"},
	{[]string{"to"}, "alt=foo;key=k"},
	{[]string{"alt", "to"}, "key=k"},
	{[]string{"alt", "to", "key"}, ""},
}

func TestAttrKey(t *testing.T) {
	v := reflect.ValueOf(&testStruct)
	for i, tt := range keyTests {
		key := attrKey(v, tt.exclude...)
		if key != tt.key {
			t.Errorf("%d: found %q, want %q", i, key, tt.key)
		}
	}
}

func TestKey(t *testing.T) {
	for i, tt := range keyTests {
		key := Key(&testStruct, tt.exclude...)
		if key != tt.key {
			t.Errorf("%d: found %q, want %q", i, key, tt.key)
		}
	}
}

func testEnclosing(t *testing.T, x *LDML, name string) {
	eq := func(a, b Elem, i int) {
		for ; i > 0; i-- {
			b = b.enclosing()
		}
		if a != b {
			t.Errorf("%s: found path %q, want %q", name, getPath(a), getPath(b))
		}
	}
	eq(x, x, 0)
	eq(x, x.Identity, 1)
	eq(x, x.Dates.Calendars, 2)
	eq(x, x.Dates.Calendars.Calendar[0], 3)
	eq(x, x.Dates.Calendars.Calendar[1], 3)
	//eq(x, x.Dates.Calendars.Calendar[0].Months, 4)
	eq(x, x.Dates.Calendars.Calendar[1].Months, 4)
}

func TestEnclosing(t *testing.T) {
	testEnclosing(t, data().RawLDML("de"), "enclosing-raw")
	de, _ := data().LDML("de")
	testEnclosing(t, de, "enclosing")
}

func TestDeepCopy(t *testing.T) {
	eq := func(have, want string) {
		if have != want {
			t.Errorf("found %q; want %q", have, want)
		}
	}
	x, _ := data().LDML("de")
	vc := deepCopy(reflect.ValueOf(x))
	c := vc.Interface().(*LDML)
	linkEnclosing(nil, c)
	if x == c {
		t.Errorf("did not copy")
	}

	eq(c.name, "ldml")
	eq(c.Dates.name, "dates")
	testEnclosing(t, c, "deepCopy")
}

type getTest struct {
	loc     string
	path    string
	field   string // used in combination with length
	data    string
	altData string // used for buddhist calendar if value != ""
	typ     string
	length  int
	missing bool
}

const (
	budMon = "dates/calendars/calendar[@type='buddhist']/months/"
	chnMon = "dates/calendars/calendar[@type='chinese']/months/"
	greMon = "dates/calendars/calendar[@type='gregorian']/months/"
)

func monthVal(path, context, width string, month int) string {
	const format = "%s/monthContext[@type='%s']/monthWidth[@type='%s']/month[@type='%d']"
	return fmt.Sprintf(format, path, context, width, month)
}

var rootGetTests = []getTest{
	{loc: "root", path: "identity/language", typ: "root"},
	{loc: "root", path: "characters/moreInformation", data: "?"},
	{loc: "root", path: "characters", field: "exemplarCharacters", length: 3},
	{loc: "root", path: greMon, field: "monthContext", length: 2},
	{loc: "root", path: greMon + "monthContext[@type='format']/monthWidth[@type='narrow']", field: "month", length: 4},
	{loc: "root", path: greMon + "monthContext[@type='stand-alone']/monthWidth[@type='wide']", field: "month", length: 4},
	// unescaping character data
	{loc: "root", path: "characters/exemplarCharacters[@type='punctuation']", data: `[\- ‐ – — … ' ‘ ‚ " “ „ \& #]`},
	// default resolution
	{loc: "root", path: "dates/calendars/calendar", typ: "gregorian"},
	// alias resolution
	{loc: "root", path: budMon, field: "monthContext", length: 2},
	// crossing but non-circular alias resolution
	{loc: "root", path: budMon + "monthContext[@type='format']/monthWidth[@type='narrow']", field: "month", length: 4},
	{loc: "root", path: budMon + "monthContext[@type='stand-alone']/monthWidth[@type='wide']", field: "month", length: 4},
	{loc: "root", path: monthVal(greMon, "format", "wide", 1), data: "11"},
	{loc: "root", path: monthVal(greMon, "format", "narrow", 2), data: "2"},
	{loc: "root", path: monthVal(greMon, "stand-alone", "wide", 3), data: "33"},
	{loc: "root", path: monthVal(greMon, "stand-alone", "narrow", 4), data: "4"},
	{loc: "root", path: monthVal(budMon, "format", "wide", 1), data: "11"},
	{loc: "root", path: monthVal(budMon, "format", "narrow", 2), data: "2"},
	{loc: "root", path: monthVal(budMon, "stand-alone", "wide", 3), data: "33"},
	{loc: "root", path: monthVal(budMon, "stand-alone", "narrow", 4), data: "4"},
}

// 19
var deGetTests = []getTest{
	{loc: "de", path: "identity/language", typ: "de"},
	{loc: "de", path: "posix", length: 2},
	{loc: "de", path: "characters", field: "exemplarCharacters", length: 4},
	{loc: "de", path: "characters/exemplarCharacters[@type='auxiliary']", data: `[á à ă]`},
	// identity is a blocking element, so de should not inherit generation from root.
	{loc: "de", path: "identity/generation", missing: true},
	// default resolution
	{loc: "root", path: "dates/calendars/calendar", typ: "gregorian"},

	// absolute path alias resolution
	{loc: "gsw", path: "posix", field: "messages", length: 1},
	{loc: "gsw", path: "posix/messages/yesstr", data: "yes:y"},
}

// 27(greMon) - 52(budMon) - 77(chnMon)
func calGetTests(s string) []getTest {
	tests := []getTest{
		{loc: "de", path: s, length: 2},
		{loc: "de", path: s + "monthContext[@type='format']/monthWidth[@type='wide']", field: "month", length: 5},
		{loc: "de", path: monthVal(s, "format", "wide", 1), data: "11"},
		{loc: "de", path: monthVal(s, "format", "wide", 2), data: "22"},
		{loc: "de", path: monthVal(s, "format", "wide", 3), data: "Maerz", altData: "bbb"},
		{loc: "de", path: monthVal(s, "format", "wide", 4), data: "April"},
		{loc: "de", path: monthVal(s, "format", "wide", 5), data: "Mai"},

		{loc: "de", path: s + "monthContext[@type='format']/monthWidth[@type='narrow']", field: "month", length: 5},
		{loc: "de", path: monthVal(s, "format", "narrow", 1), data: "1"},
		{loc: "de", path: monthVal(s, "format", "narrow", 2), data: "2"},
		{loc: "de", path: monthVal(s, "format", "narrow", 3), data: "M", altData: "BBB"},
		{loc: "de", path: monthVal(s, "format", "narrow", 4), data: "A"},
		{loc: "de", path: monthVal(s, "format", "narrow", 5), data: "m"},

		{loc: "de", path: s + "monthContext[@type='stand-alone']/monthWidth[@type='wide']", field: "month", length: 5},
		{loc: "de", path: monthVal(s, "stand-alone", "wide", 1), data: "11"},
		{loc: "de", path: monthVal(s, "stand-alone", "wide", 2), data: "22"},
		{loc: "de", path: monthVal(s, "stand-alone", "wide", 3), data: "Maerz", altData: "bbb"},
		{loc: "de", path: monthVal(s, "stand-alone", "wide", 4), data: "april"},
		{loc: "de", path: monthVal(s, "stand-alone", "wide", 5), data: "mai"},

		{loc: "de", path: s + "monthContext[@type='stand-alone']/monthWidth[@type='narrow']", field: "month", length: 5},
		{loc: "de", path: monthVal(s, "stand-alone", "narrow", 1), data: "1"},
		{loc: "de", path: monthVal(s, "stand-alone", "narrow", 2), data: "2"},
		{loc: "de", path: monthVal(s, "stand-alone", "narrow", 3), data: "m"},
		{loc: "de", path: monthVal(s, "stand-alone", "narrow", 4), data: "4"},
		{loc: "de", path: monthVal(s, "stand-alone", "narrow", 5), data: "m"},
	}
	if s == budMon {
		for i, t := range tests {
			if t.altData != "" {
				tests[i].data = t.altData
			}
		}
	}
	return tests
}

var getTests = append(rootGetTests,
	append(deGetTests,
		append(calGetTests(greMon),
			append(calGetTests(budMon),
				calGetTests(chnMon)...)...)...)...)

func TestPath(t *testing.T) {
	d := data()
	for i, tt := range getTests {
		x, _ := d.LDML(tt.loc)
		e, err := walkXPath(x, tt.path)
		if err != nil {
			if !tt.missing {
				t.Errorf("%d:error: %v %v", i, err, tt.missing)
			}
			continue
		}
		if tt.missing {
			t.Errorf("%d: missing is %v; want %v", i, e == nil, tt.missing)
			continue
		}
		if tt.data != "" && e.GetCommon().Data() != tt.data {
			t.Errorf("%d: data is %v; want %v", i, e.GetCommon().Data(), tt.data)
			continue
		}
		if tt.typ != "" && e.GetCommon().Type != tt.typ {
			t.Errorf("%d: type is %v; want %v", i, e.GetCommon().Type, tt.typ)
			continue
		}
		if tt.field != "" {
			slice, _ := findField(reflect.ValueOf(e), tt.field)
			if slice.Len() != tt.length {
				t.Errorf("%d: length is %v; want %v", i, slice.Len(), tt.length)
				continue
			}
		}
	}
}

func TestGet(t *testing.T) {
	d := data()
	for i, tt := range getTests {
		x, _ := d.LDML(tt.loc)
		e, err := Get(x, tt.path)
		if err != nil {
			if !tt.missing {
				t.Errorf("%d:error: %v %v", i, err, tt.missing)
			}
			continue
		}
		if tt.missing {
			t.Errorf("%d: missing is %v; want %v", i, e == nil, tt.missing)
			continue
		}
		if tt.data != "" && e.GetCommon().Data() != tt.data {
			t.Errorf("%d: data is %v; want %v", i, e.GetCommon().Data(), tt.data)
			continue
		}
		if tt.typ != "" && e.GetCommon().Type != tt.typ {
			t.Errorf("%d: type is %v; want %v", i, e.GetCommon().Type, tt.typ)
			continue
		}
		if tt.field != "" {
			slice, _ := findField(reflect.ValueOf(e), tt.field)
			if slice.Len() != tt.length {
				t.Errorf("%d: length is %v; want %v", i, slice.Len(), tt.length)
				continue
			}
		}
	}
}
