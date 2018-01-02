// +build linux

package volume

import (
	"strings"
	"testing"

	mounttypes "github.com/docker/docker/api/types/mount"
)

func TestConvertTmpfsOptions(t *testing.T) {
	type testCase struct {
		opt                  mounttypes.TmpfsOptions
		readOnly             bool
		expectedSubstrings   []string
		unexpectedSubstrings []string
	}
	cases := []testCase{
		{
			opt:                  mounttypes.TmpfsOptions{SizeBytes: 1024 * 1024, Mode: 0700},
			readOnly:             false,
			expectedSubstrings:   []string{"size=1m", "mode=700"},
			unexpectedSubstrings: []string{"ro"},
		},
		{
			opt:                  mounttypes.TmpfsOptions{},
			readOnly:             true,
			expectedSubstrings:   []string{"ro"},
			unexpectedSubstrings: []string{},
		},
	}
	for _, c := range cases {
		data, err := ConvertTmpfsOptions(&c.opt, c.readOnly)
		if err != nil {
			t.Fatalf("could not convert %+v (readOnly: %v) to string: %v",
				c.opt, c.readOnly, err)
		}
		t.Logf("data=%q", data)
		for _, s := range c.expectedSubstrings {
			if !strings.Contains(data, s) {
				t.Fatalf("expected substring: %s, got %v (case=%+v)", s, data, c)
			}
		}
		for _, s := range c.unexpectedSubstrings {
			if strings.Contains(data, s) {
				t.Fatalf("unexpected substring: %s, got %v (case=%+v)", s, data, c)
			}
		}
	}
}
