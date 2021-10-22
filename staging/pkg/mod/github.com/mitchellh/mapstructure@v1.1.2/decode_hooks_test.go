package mapstructure

import (
	"errors"
	"net"
	"reflect"
	"testing"
	"time"
)

func TestComposeDecodeHookFunc(t *testing.T) {
	f1 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return data.(string) + "foo", nil
	}

	f2 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return data.(string) + "bar", nil
	}

	f := ComposeDecodeHookFunc(f1, f2)

	result, err := DecodeHookExec(
		f, reflect.TypeOf(""), reflect.TypeOf([]byte("")), "")
	if err != nil {
		t.Fatalf("bad: %s", err)
	}
	if result.(string) != "foobar" {
		t.Fatalf("bad: %#v", result)
	}
}

func TestComposeDecodeHookFunc_err(t *testing.T) {
	f1 := func(reflect.Kind, reflect.Kind, interface{}) (interface{}, error) {
		return nil, errors.New("foo")
	}

	f2 := func(reflect.Kind, reflect.Kind, interface{}) (interface{}, error) {
		panic("NOPE")
	}

	f := ComposeDecodeHookFunc(f1, f2)

	_, err := DecodeHookExec(
		f, reflect.TypeOf(""), reflect.TypeOf([]byte("")), 42)
	if err.Error() != "foo" {
		t.Fatalf("bad: %s", err)
	}
}

func TestComposeDecodeHookFunc_kinds(t *testing.T) {
	var f2From reflect.Kind

	f1 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		return int(42), nil
	}

	f2 := func(
		f reflect.Kind,
		t reflect.Kind,
		data interface{}) (interface{}, error) {
		f2From = f
		return data, nil
	}

	f := ComposeDecodeHookFunc(f1, f2)

	_, err := DecodeHookExec(
		f, reflect.TypeOf(""), reflect.TypeOf([]byte("")), "")
	if err != nil {
		t.Fatalf("bad: %s", err)
	}
	if f2From != reflect.Int {
		t.Fatalf("bad: %#v", f2From)
	}
}

func TestStringToSliceHookFunc(t *testing.T) {
	f := StringToSliceHookFunc(",")

	strType := reflect.TypeOf("")
	sliceType := reflect.TypeOf([]byte(""))
	cases := []struct {
		f, t   reflect.Type
		data   interface{}
		result interface{}
		err    bool
	}{
		{sliceType, sliceType, 42, 42, false},
		{strType, strType, 42, 42, false},
		{
			strType,
			sliceType,
			"foo,bar,baz",
			[]string{"foo", "bar", "baz"},
			false,
		},
		{
			strType,
			sliceType,
			"",
			[]string{},
			false,
		},
	}

	for i, tc := range cases {
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestStringToTimeDurationHookFunc(t *testing.T) {
	f := StringToTimeDurationHookFunc()

	strType := reflect.TypeOf("")
	timeType := reflect.TypeOf(time.Duration(5))
	cases := []struct {
		f, t   reflect.Type
		data   interface{}
		result interface{}
		err    bool
	}{
		{strType, timeType, "5s", 5 * time.Second, false},
		{strType, timeType, "5", time.Duration(0), true},
		{strType, strType, "5", "5", false},
	}

	for i, tc := range cases {
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestStringToTimeHookFunc(t *testing.T) {
	strType := reflect.TypeOf("")
	timeType := reflect.TypeOf(time.Time{})
	cases := []struct {
		f, t   reflect.Type
		layout string
		data   interface{}
		result interface{}
		err    bool
	}{
		{strType, timeType, time.RFC3339, "2006-01-02T15:04:05Z",
			time.Date(2006, 1, 2, 15, 4, 5, 0, time.UTC), false},
		{strType, timeType, time.RFC3339, "5", time.Time{}, true},
		{strType, strType, time.RFC3339, "5", "5", false},
	}

	for i, tc := range cases {
		f := StringToTimeHookFunc(tc.layout)
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestStringToIPHookFunc(t *testing.T) {
	strType := reflect.TypeOf("")
	ipType := reflect.TypeOf(net.IP{})
	cases := []struct {
		f, t   reflect.Type
		data   interface{}
		result interface{}
		err    bool
	}{
		{strType, ipType, "1.2.3.4",
			net.IPv4(0x01, 0x02, 0x03, 0x04), false},
		{strType, ipType, "5", net.IP{}, true},
		{strType, strType, "5", "5", false},
	}

	for i, tc := range cases {
		f := StringToIPHookFunc()
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestStringToIPNetHookFunc(t *testing.T) {
	strType := reflect.TypeOf("")
	ipNetType := reflect.TypeOf(net.IPNet{})
	var nilNet *net.IPNet = nil

	cases := []struct {
		f, t   reflect.Type
		data   interface{}
		result interface{}
		err    bool
	}{
		{strType, ipNetType, "1.2.3.4/24",
			&net.IPNet{
				IP:   net.IP{0x01, 0x02, 0x03, 0x00},
				Mask: net.IPv4Mask(0xff, 0xff, 0xff, 0x00),
			}, false},
		{strType, ipNetType, "5", nilNet, true},
		{strType, strType, "5", "5", false},
	}

	for i, tc := range cases {
		f := StringToIPNetHookFunc()
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}

func TestWeaklyTypedHook(t *testing.T) {
	var f DecodeHookFunc = WeaklyTypedHook

	boolType := reflect.TypeOf(true)
	strType := reflect.TypeOf("")
	sliceType := reflect.TypeOf([]byte(""))
	cases := []struct {
		f, t   reflect.Type
		data   interface{}
		result interface{}
		err    bool
	}{
		// TO STRING
		{
			boolType,
			strType,
			false,
			"0",
			false,
		},

		{
			boolType,
			strType,
			true,
			"1",
			false,
		},

		{
			reflect.TypeOf(float32(1)),
			strType,
			float32(7),
			"7",
			false,
		},

		{
			reflect.TypeOf(int(1)),
			strType,
			int(7),
			"7",
			false,
		},

		{
			sliceType,
			strType,
			[]uint8("foo"),
			"foo",
			false,
		},

		{
			reflect.TypeOf(uint(1)),
			strType,
			uint(7),
			"7",
			false,
		},
	}

	for i, tc := range cases {
		actual, err := DecodeHookExec(f, tc.f, tc.t, tc.data)
		if tc.err != (err != nil) {
			t.Fatalf("case %d: expected err %#v", i, tc.err)
		}
		if !reflect.DeepEqual(actual, tc.result) {
			t.Fatalf(
				"case %d: expected %#v, got %#v",
				i, tc.result, actual)
		}
	}
}
