package benchmark

import (
	"fmt"
	"testing"
)

func TestIsValid(t *testing.T) {
	testCases := []struct {
		desc string
		c    *createAny
		want error
	}{
		{
			desc: "TemplatePath must be set",
			c:    &createAny{},
			want: fmt.Errorf("TemplatePath must be set"),
		},
		{
			desc: "valid createAny",
			c: &createAny{
				TemplatePath: "hoge",
			},
			want: nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			got := tc.c.isValid(false)
			if (got == nil) != (tc.want == nil) {
				t.Fatalf("Got %q, want %q", got, tc.want)
			}
		})
	}
}
