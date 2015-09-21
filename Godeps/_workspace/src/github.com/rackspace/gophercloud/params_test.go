package gophercloud

import (
	"net/url"
	"reflect"
	"testing"
	"time"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestMaybeString(t *testing.T) {
	testString := ""
	var expected *string
	actual := MaybeString(testString)
	th.CheckDeepEquals(t, expected, actual)

	testString = "carol"
	expected = &testString
	actual = MaybeString(testString)
	th.CheckDeepEquals(t, expected, actual)
}

func TestMaybeInt(t *testing.T) {
	testInt := 0
	var expected *int
	actual := MaybeInt(testInt)
	th.CheckDeepEquals(t, expected, actual)

	testInt = 4
	expected = &testInt
	actual = MaybeInt(testInt)
	th.CheckDeepEquals(t, expected, actual)
}

func TestBuildQueryString(t *testing.T) {
	type testVar string
	opts := struct {
		J  int       `q:"j"`
		R  string    `q:"r,required"`
		C  bool      `q:"c"`
		S  []string  `q:"s"`
		TS []testVar `q:"ts"`
		TI []int     `q:"ti"`
	}{
		J:  2,
		R:  "red",
		C:  true,
		S:  []string{"one", "two", "three"},
		TS: []testVar{"a", "b"},
		TI: []int{1, 2},
	}
	expected := &url.URL{RawQuery: "c=true&j=2&r=red&s=one&s=two&s=three&ti=1&ti=2&ts=a&ts=b"}
	actual, err := BuildQueryString(&opts)
	if err != nil {
		t.Errorf("Error building query string: %v", err)
	}
	th.CheckDeepEquals(t, expected, actual)

	opts = struct {
		J  int       `q:"j"`
		R  string    `q:"r,required"`
		C  bool      `q:"c"`
		S  []string  `q:"s"`
		TS []testVar `q:"ts"`
		TI []int     `q:"ti"`
	}{
		J: 2,
		C: true,
	}
	_, err = BuildQueryString(&opts)
	if err == nil {
		t.Errorf("Expected error: 'Required field not set'")
	}
	th.CheckDeepEquals(t, expected, actual)

	_, err = BuildQueryString(map[string]interface{}{"Number": 4})
	if err == nil {
		t.Errorf("Expected error: 'Options type is not a struct'")
	}
}

func TestBuildHeaders(t *testing.T) {
	testStruct := struct {
		Accept string `h:"Accept"`
		Num    int    `h:"Number,required"`
		Style  bool   `h:"Style"`
	}{
		Accept: "application/json",
		Num:    4,
		Style:  true,
	}
	expected := map[string]string{"Accept": "application/json", "Number": "4", "Style": "true"}
	actual, err := BuildHeaders(&testStruct)
	th.CheckNoErr(t, err)
	th.CheckDeepEquals(t, expected, actual)

	testStruct.Num = 0
	_, err = BuildHeaders(&testStruct)
	if err == nil {
		t.Errorf("Expected error: 'Required header not set'")
	}

	_, err = BuildHeaders(map[string]interface{}{"Number": 4})
	if err == nil {
		t.Errorf("Expected error: 'Options type is not a struct'")
	}
}

func TestIsZero(t *testing.T) {
	var testMap map[string]interface{}
	testMapValue := reflect.ValueOf(testMap)
	expected := true
	actual := isZero(testMapValue)
	th.CheckEquals(t, expected, actual)
	testMap = map[string]interface{}{"empty": false}
	testMapValue = reflect.ValueOf(testMap)
	expected = false
	actual = isZero(testMapValue)
	th.CheckEquals(t, expected, actual)

	var testArray [2]string
	testArrayValue := reflect.ValueOf(testArray)
	expected = true
	actual = isZero(testArrayValue)
	th.CheckEquals(t, expected, actual)
	testArray = [2]string{"one", "two"}
	testArrayValue = reflect.ValueOf(testArray)
	expected = false
	actual = isZero(testArrayValue)
	th.CheckEquals(t, expected, actual)

	var testStruct struct {
		A string
		B time.Time
	}
	testStructValue := reflect.ValueOf(testStruct)
	expected = true
	actual = isZero(testStructValue)
	th.CheckEquals(t, expected, actual)
	testStruct = struct {
		A string
		B time.Time
	}{
		B: time.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC),
	}
	testStructValue = reflect.ValueOf(testStruct)
	expected = false
	actual = isZero(testStructValue)
	th.CheckEquals(t, expected, actual)
}

func TestQueriesAreEscaped(t *testing.T) {
	type foo struct {
		Name  string `q:"something"`
		Shape string `q:"else"`
	}

	expected := &url.URL{RawQuery: "else=Triangl+e&something=blah%2B%3F%21%21foo"}

	actual, err := BuildQueryString(foo{Name: "blah+?!!foo", Shape: "Triangl e"})
	th.AssertNoErr(t, err)

	th.AssertDeepEquals(t, expected, actual)
}
