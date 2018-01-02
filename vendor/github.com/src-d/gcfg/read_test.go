package gcfg

import (
	"fmt"
	"math/big"
	"os"
	"reflect"
	"testing"
	"bytes"
	"strconv"
	"github.com/pkg/errors"
)

const (
	// 64 spaces
	sp64 = "                                                                "
	// 512 spaces
	sp512 = sp64 + sp64 + sp64 + sp64 + sp64 + sp64 + sp64 + sp64
	// 4096 spaces
	sp4096 = sp512 + sp512 + sp512 + sp512 + sp512 + sp512 + sp512 + sp512
)

type cBasic struct {
	Section           cBasicS1
	Hyphen_In_Section cBasicS2
	unexported        cBasicS1
	Exported          cBasicS3
	TagName           cBasicS1 `gcfg:"tag-name"`
}
type cBasicS1 struct {
	Name  string
	Int   int
	PName *string
}
type cBasicS2 struct {
	Hyphen_In_Name string
}
type cBasicS3 struct {
	unexported string
}

type nonMulti []string

type unmarshalable string

func (u *unmarshalable) UnmarshalText(text []byte) error {
	s := string(text)
	if s == "error" {
		return fmt.Errorf("%s", s)
	}
	*u = unmarshalable(s)
	return nil
}

var _ textUnmarshaler = new(unmarshalable)

type cUni struct {
	X甲       cUniS1
	XSection cUniS2
}
type cUniS1 struct {
	X乙 string
}
type cUniS2 struct {
	XName string
}

type cMulti struct {
	M1 cMultiS1
	M2 cMultiS2
	M3 cMultiS3
}
type cMultiS1 struct{ Multi []string }
type cMultiS2 struct{ NonMulti nonMulti }
type cMultiS3 struct{ PMulti *[]string }

type cSubs struct{ Sub map[string]*cSubsS1 }
type cSubsS1 struct{ Name string }

type cBool struct{ Section cBoolS1 }
type cBoolS1 struct{ Bool bool }

type cTxUnm struct{ Section cTxUnmS1 }
type cTxUnmS1 struct{ Name unmarshalable }

type cNum struct {
	N1 cNumS1
	N2 cNumS2
	N3 cNumS3
}
type cNumS1 struct {
	Int    int
	IntDHO int `gcfg:",int=dho"`
	Big    *big.Int
}
type cNumS2 struct {
	MultiInt []int
	MultiBig []*big.Int
}
type cNumS3 struct{ FileMode os.FileMode }
type readtest struct {
	gcfg string
	exp  interface{}
	ok   bool
}

func newString(s string) *string           { return &s }
func newStringSlice(s ...string) *[]string { return &s }

var readtests = []struct {
	group string
	tests []readtest
}{{"scanning", []readtest{
	{"[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	// hyphen in name
	{"[hyphen-in-section]\nhyphen-in-name=value", &cBasic{Hyphen_In_Section: cBasicS2{Hyphen_In_Name: "value"}}, true},
	// quoted string value
	{"[section]\nname=\"\"", &cBasic{Section: cBasicS1{Name: ""}}, true},
	{"[section]\nname=\" \"", &cBasic{Section: cBasicS1{Name: " "}}, true},
	{"[section]\nname=\"value\"", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname=\" value \"", &cBasic{Section: cBasicS1{Name: " value "}}, true},
	{"\n[section]\nname=\"va ; lue\"", &cBasic{Section: cBasicS1{Name: "va ; lue"}}, true},
	{"[section]\nname=\"val\" \"ue\"", &cBasic{Section: cBasicS1{Name: "val ue"}}, true},
	{"[section]\nname=\"value", &cBasic{}, false},
	// escape sequences
	{"[section]\nname=\"va\\\\lue\"", &cBasic{Section: cBasicS1{Name: "va\\lue"}}, true},
	{"[section]\nname=\"va\\\"lue\"", &cBasic{Section: cBasicS1{Name: "va\"lue"}}, true},
	{"[section]\nname=\"va\\nlue\"", &cBasic{Section: cBasicS1{Name: "va\nlue"}}, true},
	{"[section]\nname=\"va\\tlue\"", &cBasic{Section: cBasicS1{Name: "va\tlue"}}, true},
	{"\n[section]\nname=\\", &cBasic{}, false},
	{"\n[section]\nname=\\a", &cBasic{}, false},
	{"\n[section]\nname=\"val\\a\"", &cBasic{}, false},
	{"\n[section]\nname=val\\", &cBasic{}, false},
	{"\n[sub \"A\\\n\"]\nname=value", &cSubs{}, false},
	{"\n[sub \"A\\\t\"]\nname=value", &cSubs{}, false},
	// broken line
	{"[section]\nname=value \\\n value", &cBasic{Section: cBasicS1{Name: "value  value"}}, true},
	{"[section]\nname=\"value \\\n value\"", &cBasic{}, false},
}}, {"scanning:whitespace", []readtest{
	{" \n[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{" [section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\t[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[ section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section ]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\n name=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname =value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname= value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname=value ", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\r\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\r\nname=value\r\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{";cmnt\r\n[section]\r\nname=value\r\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	// long lines
	{sp4096 + "[section]\nname=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[" + sp4096 + "section]\nname=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section" + sp4096 + "]\nname=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]" + sp4096 + "\nname=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\n" + sp4096 + "name=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname" + sp4096 + "=value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname=" + sp4096 + "value\n", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname=value\n" + sp4096, &cBasic{Section: cBasicS1{Name: "value"}}, true},
}}, {"scanning:comments", []readtest{
	{"; cmnt\n[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"# cmnt\n[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{" ; cmnt\n[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\t; cmnt\n[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]; cmnt\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section] ; cmnt\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]\nname=value; cmnt", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]\nname=value ; cmnt", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]\nname=\"value\" ; cmnt", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]\nname=value ; \"cmnt", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"\n[section]\nname=\"va ; lue\" ; cmnt", &cBasic{Section: cBasicS1{Name: "va ; lue"}}, true},
	{"\n[section]\nname=; cmnt", &cBasic{Section: cBasicS1{Name: ""}}, true},
}}, {"scanning:subsections", []readtest{
	{"\n[sub \"A\"]\nname=value", &cSubs{map[string]*cSubsS1{"A": &cSubsS1{"value"}}}, true},
	{"\n[sub \"b\"]\nname=value", &cSubs{map[string]*cSubsS1{"b": &cSubsS1{"value"}}}, true},
	{"\n[sub \"A\\\\\"]\nname=value", &cSubs{map[string]*cSubsS1{"A\\": &cSubsS1{"value"}}}, true},
	{"\n[sub \"A\\\"\"]\nname=value", &cSubs{map[string]*cSubsS1{"A\"": &cSubsS1{"value"}}}, true},
}}, {"syntax", []readtest{
	// invalid line
	{"\n[section]\n=", &cBasic{}, false},
	// no section
	{"name=value", &cBasic{}, false},
	// empty section
	{"\n[]\nname=value", &cBasic{}, false},
	// empty subsection
	{"\n[sub \"\"]\nname=value", &cSubs{}, false},
}}, {"setting", []readtest{
	{"[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	// pointer
	{"[section]", &cBasic{Section: cBasicS1{PName: nil}}, true},
	{"[section]\npname=value", &cBasic{Section: cBasicS1{PName: newString("value")}}, true},
	{"[m3]", &cMulti{M3: cMultiS3{PMulti: nil}}, true},
	{"[m3]\npmulti", &cMulti{M3: cMultiS3{PMulti: newStringSlice()}}, true},
	{"[m3]\npmulti=value", &cMulti{M3: cMultiS3{PMulti: newStringSlice("value")}}, true},
	{"[m3]\npmulti=value1\npmulti=value2", &cMulti{M3: cMultiS3{PMulti: newStringSlice("value1", "value2")}}, true},
	// section name not matched
	{"\n[nonexistent]\nname=value", &cBasic{}, false},
	// subsection name not matched
	{"\n[section \"nonexistent\"]\nname=value", &cBasic{}, false},
	// variable name not matched
	{"\n[section]\nnonexistent=value", &cBasic{}, false},
	// hyphen in name
	{"[hyphen-in-section]\nhyphen-in-name=value", &cBasic{Hyphen_In_Section: cBasicS2{Hyphen_In_Name: "value"}}, true},
	// ignore unexported fields
	{"[unexported]\nname=value", &cBasic{}, false},
	{"[exported]\nunexported=value", &cBasic{}, false},
	// 'X' prefix for non-upper/lower-case letters
	{"[甲]\n乙=丙", &cUni{X甲: cUniS1{X乙: "丙"}}, true},
	//{"[section]\nxname=value", &cBasic{XSection: cBasicS4{XName: "value"}}, false},
	//{"[xsection]\nname=value", &cBasic{XSection: cBasicS4{XName: "value"}}, false},
	// name specified as struct tag
	{"[tag-name]\nname=value", &cBasic{TagName: cBasicS1{Name: "value"}}, true},
	// empty subsections
	{"\n[sub \"A\"]\n[sub \"B\"]", &cSubs{map[string]*cSubsS1{"A": &cSubsS1{}, "B": &cSubsS1{}}}, true},
}}, {"multivalue", []readtest{
	// unnamed slice type: treat as multi-value
	{"\n[m1]", &cMulti{M1: cMultiS1{}}, true},
	{"\n[m1]\nmulti=value", &cMulti{M1: cMultiS1{[]string{"value"}}}, true},
	{"\n[m1]\nmulti=value1\nmulti=value2", &cMulti{M1: cMultiS1{[]string{"value1", "value2"}}}, true},
	// "blank" empties multi-valued slice -- here same result as above
	{"\n[m1]\nmulti\nmulti=value1\nmulti=value2", &cMulti{M1: cMultiS1{[]string{"value1", "value2"}}}, true},
	// named slice type: do not treat as multi-value
	{"\n[m2]", &cMulti{}, true},
	{"\n[m2]\nmulti=value", &cMulti{}, false},
	{"\n[m2]\nmulti=value1\nmulti=value2", &cMulti{}, false},
}}, {"type:string", []readtest{
	{"[section]\nname=value", &cBasic{Section: cBasicS1{Name: "value"}}, true},
	{"[section]\nname=", &cBasic{Section: cBasicS1{Name: ""}}, true},
}}, {"type:bool", []readtest{
	// explicit values
	{"[section]\nbool=true", &cBool{cBoolS1{true}}, true},
	{"[section]\nbool=yes", &cBool{cBoolS1{true}}, true},
	{"[section]\nbool=on", &cBool{cBoolS1{true}}, true},
	{"[section]\nbool=1", &cBool{cBoolS1{true}}, true},
	{"[section]\nbool=tRuE", &cBool{cBoolS1{true}}, true},
	{"[section]\nbool=false", &cBool{cBoolS1{false}}, true},
	{"[section]\nbool=no", &cBool{cBoolS1{false}}, true},
	{"[section]\nbool=off", &cBool{cBoolS1{false}}, true},
	{"[section]\nbool=0", &cBool{cBoolS1{false}}, true},
	{"[section]\nbool=NO", &cBool{cBoolS1{false}}, true},
	// "blank" value handled as true
	{"[section]\nbool", &cBool{cBoolS1{true}}, true},
	// bool parse errors
	{"[section]\nbool=maybe", &cBool{}, false},
	{"[section]\nbool=t", &cBool{}, false},
	{"[section]\nbool=truer", &cBool{}, false},
	{"[section]\nbool=2", &cBool{}, false},
	{"[section]\nbool=-1", &cBool{}, false},
}}, {"type:numeric", []readtest{
	{"[section]\nint=0", &cBasic{Section: cBasicS1{Int: 0}}, true},
	{"[section]\nint=1", &cBasic{Section: cBasicS1{Int: 1}}, true},
	{"[section]\nint=-1", &cBasic{Section: cBasicS1{Int: -1}}, true},
	{"[section]\nint=0.2", &cBasic{}, false},
	{"[section]\nint=1e3", &cBasic{}, false},
	// primitive [u]int(|8|16|32|64) and big.Int is parsed as dec or hex (not octal)
	{"[n1]\nint=010", &cNum{N1: cNumS1{Int: 10}}, true},
	{"[n1]\nint=0x10", &cNum{N1: cNumS1{Int: 0x10}}, true},
	{"[n1]\nbig=1", &cNum{N1: cNumS1{Big: big.NewInt(1)}}, true},
	{"[n1]\nbig=0x10", &cNum{N1: cNumS1{Big: big.NewInt(0x10)}}, true},
	{"[n1]\nbig=010", &cNum{N1: cNumS1{Big: big.NewInt(10)}}, true},
	{"[n2]\nmultiint=010", &cNum{N2: cNumS2{MultiInt: []int{10}}}, true},
	{"[n2]\nmultibig=010", &cNum{N2: cNumS2{MultiBig: []*big.Int{big.NewInt(10)}}}, true},
	// set parse mode for int types via struct tag
	{"[n1]\nintdho=010", &cNum{N1: cNumS1{IntDHO: 010}}, true},
	// octal allowed for named type
	{"[n3]\nfilemode=0777", &cNum{N3: cNumS3{FileMode: 0777}}, true},
}}, {"type:textUnmarshaler", []readtest{
	{"[section]\nname=value", &cTxUnm{Section: cTxUnmS1{Name: "value"}}, true},
	{"[section]\nname=error", &cTxUnm{}, false},
}},
}

func TestReadStringInto(t *testing.T) {
	for _, tg := range readtests {
		for i, tt := range tg.tests {
			id := fmt.Sprintf("%s:%d", tg.group, i)
			testRead(t, id, tt)
		}
	}
}

func TestReadStringIntoMultiBlankPreset(t *testing.T) {
	tt := readtest{"\n[m1]\nmulti\nmulti=value1\nmulti=value2", &cMulti{M1: cMultiS1{[]string{"value1", "value2"}}}, true}
	cfg := &cMulti{M1: cMultiS1{[]string{"preset1", "preset2"}}}
	testReadInto(t, "multi:blank", tt, cfg)
}

func testRead(t *testing.T, id string, tt readtest) {
	// get the type of the expected result
	restyp := reflect.TypeOf(tt.exp).Elem()
	// create a new instance to hold the actual result
	res := reflect.New(restyp).Interface()
	testReadInto(t, id, tt, res)
}

func testReadInto(t *testing.T, id string, tt readtest, res interface{}) {
	err := ReadStringInto(res, tt.gcfg)
	if tt.ok {
		if err != nil {
			t.Errorf("%s fail: got error %v, wanted ok", id, err)
			return
		} else if !reflect.DeepEqual(res, tt.exp) {
			t.Errorf("%s fail: got value %#v, wanted value %#v", id, res, tt.exp)
			return
		}
		if !testing.Short() {
			t.Logf("%s pass: got value %#v", id, res)
		}
	} else { // !tt.ok
		if err == nil {
			t.Errorf("%s fail: got value %#v, wanted error", id, res)
			return
		}
		if !testing.Short() {
			t.Logf("%s pass: got error %v", id, err)
		}
	}
}

func TestReadFileInto(t *testing.T) {
	res := &struct{ Section struct{ Name string } }{}
	err := ReadFileInto(res, "testdata/gcfg_test.gcfg")
	if err != nil {
		t.Error(err)
	}
	if "value" != res.Section.Name {
		t.Errorf("got %q, wanted %q", res.Section.Name, "value")
	}
}

func TestReadFileIntoUnicode(t *testing.T) {
	res := &struct{ X甲 struct{ X乙 string } }{}
	err := ReadFileInto(res, "testdata/gcfg_unicode_test.gcfg")
	if err != nil {
		t.Error(err)
	}
	if "丙" != res.X甲.X乙 {
		t.Errorf("got %q, wanted %q", res.X甲.X乙, "丙")
	}
}

func TestReadStringIntoSubsectDefaults(t *testing.T) {
	type subsect struct {
		Color       string
		Orientation string
	}
	res := &struct {
		Default_Profile subsect
		Profile         map[string]*subsect
	}{Default_Profile: subsect{Color: "green"}}
	cfg := `
	[profile "one"]
	orientation = left`
	err := ReadStringInto(res, cfg)
	if err != nil {
		t.Error(err)
	}
	if res.Profile["one"].Color != "green" {
		t.Errorf("got %q; want %q", res.Profile["one"].Color, "green")
	}
}

func TestReadStringIntoExtraData(t *testing.T) {
	res := &struct {
		Section struct {
			Name string
		}
	}{}
	cfg := `
	[section]
	name = value
	name2 = value2`
	err := FatalOnly(ReadStringInto(res, cfg))
	if err != nil {
		t.Error(err)
	}
	if res.Section.Name != "value" {
		t.Errorf("res.Section.Name=%q; want %q", res.Section.Name, "value")
	}
}

func TestReadWithCallback(t *testing.T) {
	results := [][]string{}
	cb := func(s string, ss string, k string, v string, bv bool) error {
		results = append(results, []string{s, ss, k, v, strconv.FormatBool(bv)})
		return nil
	}
	text := `
	[sect1]
	key1=value1
	[sect1 "subsect1"]
	key2=value2
	key3=value3
	key4
	key5=
	[sect1 "subsect2"]
	[sect2]
	`
	expected := [][]string{
		[]string{"sect1", "", "", "", "true"},
		[]string{"sect1", "", "key1", "value1", "false"},
		[]string{"sect1", "subsect1", "", "", "true"},
		[]string{"sect1", "subsect1", "key2", "value2", "false"},
		[]string{"sect1", "subsect1", "key3", "value3", "false"},
		[]string{"sect1", "subsect1", "key4", "", "true"},
		[]string{"sect1", "subsect1", "key5", "", "false"},
		[]string{"sect1", "subsect2", "", "", "true"},
		[]string{"sect2", "", "", "", "true"},
	}
	err := ReadWithCallback(bytes.NewReader([]byte(text)), cb)
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(results, expected) {
		t.Errorf("expected %+v, got %+v", expected, results)
	}

	i := 0
	expectedErr := errors.New("FATAL ERROR")
	results = [][]string{}
	cbWithError := func(s string, ss string, k string, v string, bv bool) error {
		results = append(results, []string{s, ss, k, v, strconv.FormatBool(bv)})
		i += 1
		if i == 3 {
			return expectedErr
		}
		return nil
	}
	err = ReadWithCallback(bytes.NewReader([]byte(text)), cbWithError)
	if err != expectedErr {
		t.Errorf("expected error: %+v", err)
	}
	if !reflect.DeepEqual(results, expected[:3]) {
		t.Errorf("expected %+v, got %+v", expected, results[:3])
	}
}

func TestReadWithCallback_WithError(t *testing.T) {
	results := [][]string{}
	cb := func(s string, ss string, k string, v string, bv bool) error {
		results = append(results, []string{s, ss, k, v, strconv.FormatBool(bv)})
		return nil
	}
	text := `
	[sect1]
	key1=value1
	[sect1 "subsect1"]
	key2=value2
	key3=value3
	key4
	key5=
	[sect1 "subsect2"]
	[sect2]
	`
	expected := [][]string{
		[]string{"sect1", "", "", "", "true"},
		[]string{"sect1", "", "key1", "value1", "false"},
		[]string{"sect1", "subsect1", "", "", "true"},
		[]string{"sect1", "subsect1", "key2", "value2", "false"},
		[]string{"sect1", "subsect1", "key3", "value3", "false"},
		[]string{"sect1", "subsect1", "key4", "", "true"},
		[]string{"sect1", "subsect1", "key5", "", "false"},
		[]string{"sect1", "subsect2", "", "", "true"},
		[]string{"sect2", "", "", "", "true"},
	}
	err := ReadWithCallback(bytes.NewReader([]byte(text)), cb)
	if err != nil {
		t.Error(err)
	}
	if !reflect.DeepEqual(results, expected) {
		t.Errorf("expected %+v, got %+v", expected, results)
	}
}
