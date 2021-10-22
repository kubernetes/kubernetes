// +build go1.7

package strings

import (
	"strings"
	"testing"
)

func TestHasPrefixFold(t *testing.T) {
	type args struct {
		s      string
		prefix string
	}
	tests := map[string]struct {
		args args
		want bool
	}{
		"empty strings and prefix": {
			args: args{
				s:      "",
				prefix: "",
			},
			want: true,
		},
		"strings starts with prefix": {
			args: args{
				s:      "some string",
				prefix: "some",
			},
			want: true,
		},
		"prefix longer then string": {
			args: args{
				s:      "some",
				prefix: "some string",
			},
		},
		"equal length string and prefix": {
			args: args{
				s:      "short string",
				prefix: "short string",
			},
			want: true,
		},
		"different cases": {
			args: args{
				s:      "ShOrT StRING",
				prefix: "short",
			},
			want: true,
		},
		"empty prefix not empty string": {
			args: args{
				s:      "ShOrT StRING",
				prefix: "",
			},
			want: true,
		},
		"mixed-case prefixes": {
			args: args{
				s:      "SoMe String",
				prefix: "sOme",
			},
			want: true,
		},
	}
	for name, tt := range tests {
		t.Run(name, func(t *testing.T) {
			if got := HasPrefixFold(tt.args.s, tt.args.prefix); got != tt.want {
				t.Errorf("HasPrefixFold() = %v, want %v", got, tt.want)
			}
		})
	}
}

func BenchmarkHasPrefixFold(b *testing.B) {
	HasPrefixFold("SoME string", "sOmE")
}

func BenchmarkHasPrefix(b *testing.B) {
	strings.HasPrefix(strings.ToLower("SoME string"), strings.ToLower("sOmE"))
}
