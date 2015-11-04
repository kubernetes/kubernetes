package flagutil

import (
	"reflect"
	"testing"
)

func TestIPv4FlagSetInvalidArgument(t *testing.T) {
	tests := []string{
		"",
		"foo",
		"::",
		"127.0.0.1:4328",
	}

	for i, tt := range tests {
		var f IPv4Flag
		if err := f.Set(tt); err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
	}
}

func TestIPv4FlagSetValidArgument(t *testing.T) {
	tests := []string{
		"127.0.0.1",
		"0.0.0.0",
	}

	for i, tt := range tests {
		var f IPv4Flag
		if err := f.Set(tt); err != nil {
			t.Errorf("case %d: err=%v", i, err)
		}
	}
}

func TestStringSliceFlag(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{input: "", want: []string{""}},
		{input: "foo", want: []string{"foo"}},
		{input: "foo,bar", want: []string{"foo", "bar"}},
	}

	for i, tt := range tests {
		var f StringSliceFlag
		if err := f.Set(tt.input); err != nil {
			t.Errorf("case %d: err=%v", i, err)
		}
		if !reflect.DeepEqual(tt.want, []string(f)) {
			t.Errorf("case %d: want=%v got=%v", i, tt.want, f)
		}
	}
}
