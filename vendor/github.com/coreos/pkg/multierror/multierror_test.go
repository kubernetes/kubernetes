package multierror

import (
	"errors"
	"reflect"
	"testing"
)

func TestAsError(t *testing.T) {
	tests := []struct {
		multierr Error
		want     error
	}{
		{
			multierr: Error([]error{errors.New("foo"), errors.New("bar")}),
			want:     Error([]error{errors.New("foo"), errors.New("bar")}),
		},
		{
			multierr: Error([]error{}),
			want:     nil,
		},
		{
			multierr: Error(nil),
			want:     nil,
		},
	}

	for i, tt := range tests {
		got := tt.multierr.AsError()
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("case %d: incorrect error value: want=%+v got=%+v", i, tt.want, got)
		}
	}

}

func TestErrorAppend(t *testing.T) {
	var multierr Error
	multierr = append(multierr, errors.New("foo"))
	multierr = append(multierr, errors.New("bar"))
	multierr = append(multierr, errors.New("baz"))
	want := Error([]error{errors.New("foo"), errors.New("bar"), errors.New("baz")})
	got := multierr.AsError()
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("incorrect error value: want=%+v got=%+v", want, got)
	}
}

func TestErrorString(t *testing.T) {
	var multierr Error
	multierr = append(multierr, errors.New("foo"))
	multierr = append(multierr, errors.New("bar"))
	multierr = append(multierr, errors.New("baz"))
	got := multierr.Error()
	want := "[0] foo [1] bar [2] baz"
	if want != got {
		t.Fatalf("incorrect output: want=%q got=%q", want, got)
	}
}
