package githttp

import (
	"encoding/hex"
	"errors"
	"reflect"
	"testing"
)

func TestParsePktLen(t *testing.T) {
	tests := []struct {
		in string

		wantLen int
		wantErr error
	}{
		// Valid pkt-len.
		{"00a5", 165, nil},
		{"01a5", 421, nil},
		{"0032", 50, nil},
		{"000b", 11, nil},
		{"000B", 11, nil},

		// Valud flush-pkt.
		{"0000", 0, nil},

		{"0001", 0, errors.New("invalid pkt-len: 1")},
		{"0003", 0, errors.New("invalid pkt-len: 3")},
		{"abyz", 0, hex.InvalidByteError('y')},
		{"-<%^", 0, hex.InvalidByteError('-')},

		// Maximum length.
		{"fff4", 65524, nil},
		{"fff5", 0, errors.New("invalid pkt-len: 65525")},
		{"ffff", 0, errors.New("invalid pkt-len: 65535")},
	}

	for _, tt := range tests {
		gotLen, gotErr := parsePktLen([]byte(tt.in))
		if gotLen != tt.wantLen || !reflect.DeepEqual(gotErr, tt.wantErr) {
			t.Errorf("test %q:\n got: %#v, %#v\nwant: %#v, %#v\n", tt.in, gotLen, gotErr, tt.wantLen, tt.wantErr)
		}
	}
}
