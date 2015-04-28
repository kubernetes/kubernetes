package types

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestLabels(t *testing.T) {
	tests := []struct {
		in        string
		errPrefix string
	}{
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "amd64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "aarch64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "armv7l"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "armv7b"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "freebsd"}, {"name": "arch", "value": "amd64"}]`,
			"",
		},
		{
			`[{"name": "os", "value": "OS/360"}, {"name": "arch", "value": "S/360"}]`,
			`bad os "OS/360"`,
		},
		{
			`[{"name": "os", "value": "freebsd"}, {"name": "arch", "value": "armv7b"}]`,
			`bad arch "armv7b" for freebsd`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "arch", "value": "arm"}]`,
			`bad arch "arm" for linux`,
		},
		{
			`[{"name": "name"}]`,
			`invalid label name: "name"`,
		},
		{
			`[{"name": "os", "value": "linux"}, {"name": "os", "value": "freebsd"}]`,
			`duplicate labels of name "os"`,
		},
		{
			`[{"name": "arch", "value": "amd64"}, {"name": "os", "value": "freebsd"}, {"name": "arch", "value": "x86_64"}]`,
			`duplicate labels of name "arch"`,
		},
		{
			`[]`,
			"",
		},
	}
	for i, tt := range tests {
		var l Labels
		if err := json.Unmarshal([]byte(tt.in), &l); err != nil {
			if tt.errPrefix == "" {
				t.Errorf("#%d: got err=%v, expected no error", i, err)
			} else if !strings.HasPrefix(err.Error(), tt.errPrefix) {
				t.Errorf("#%d: got err=%v, expected prefix %#v", i, err, tt.errPrefix)
			}
		} else {
			t.Log(l)
			if tt.errPrefix != "" {
				t.Errorf("#%d: got no err, expected prefix %#v", i, tt.errPrefix)
			}
		}
	}
}
