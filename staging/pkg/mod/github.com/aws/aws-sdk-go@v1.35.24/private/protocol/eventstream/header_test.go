package eventstream

import (
	"reflect"
	"testing"
	"time"
)

func TestHeaders_Set(t *testing.T) {
	expect := Headers{
		{Name: "ABC", Value: StringValue("123")},
		{Name: "EFG", Value: TimestampValue(time.Time{})},
	}

	var actual Headers
	actual.Set("ABC", Int32Value(123))
	actual.Set("ABC", StringValue("123")) // replace case
	actual.Set("EFG", TimestampValue(time.Time{}))

	if e, a := expect, actual; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v headers, got %v", e, a)
	}
}

func TestHeaders_Get(t *testing.T) {
	headers := Headers{
		{Name: "ABC", Value: StringValue("123")},
		{Name: "EFG", Value: TimestampValue(time.Time{})},
	}

	cases := []struct {
		Name  string
		Value Value
	}{
		{Name: "ABC", Value: StringValue("123")},
		{Name: "EFG", Value: TimestampValue(time.Time{})},
		{Name: "NotFound"},
	}

	for i, c := range cases {
		actual := headers.Get(c.Name)
		if e, a := c.Value, actual; !reflect.DeepEqual(e, a) {
			t.Errorf("%d, expect %v value, got %v", i, e, a)
		}
	}
}

func TestHeaders_Del(t *testing.T) {
	headers := Headers{
		{Name: "ABC", Value: StringValue("123")},
		{Name: "EFG", Value: TimestampValue(time.Time{})},
		{Name: "HIJ", Value: StringValue("123")},
		{Name: "KML", Value: TimestampValue(time.Time{})},
	}
	expectAfterDel := Headers{
		{Name: "EFG", Value: TimestampValue(time.Time{})},
	}

	headers.Del("HIJ")
	headers.Del("ABC")
	headers.Del("KML")

	if e, a := expectAfterDel, headers; !reflect.DeepEqual(e, a) {
		t.Errorf("expect %v headers, got %v", e, a)
	}
}
