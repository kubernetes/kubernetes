// +build linux

package ipvs

import (
	"errors"
	"reflect"
	"syscall"
	"testing"
)

func Test_getIPFamily(t *testing.T) {
	testcases := []struct {
		name           string
		address        []byte
		expectedFamily uint16
		expectedErr    error
	}{
		{
			name:           "16 byte IPv4 10.0.0.1",
			address:        []byte{10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expectedFamily: syscall.AF_INET,
			expectedErr:    nil,
		},
		{
			name:           "16 byte IPv6 2001:db8:3c4d:15::1a00",
			address:        []byte{32, 1, 13, 184, 60, 77, 0, 21, 0, 0, 0, 0, 0, 0, 26, 0},
			expectedFamily: syscall.AF_INET6,
			expectedErr:    nil,
		},
		{
			name:           "zero address",
			address:        []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expectedFamily: 0,
			expectedErr:    errors.New("could not parse IP family from address data"),
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			family, err := getIPFamily(testcase.address)
			if !reflect.DeepEqual(err, testcase.expectedErr) {
				t.Logf("got err: %v", err)
				t.Logf("expected err: %v", testcase.expectedErr)
				t.Errorf("unexpected error")
			}

			if family != testcase.expectedFamily {
				t.Logf("got IP family: %v", family)
				t.Logf("expected IP family: %v", testcase.expectedFamily)
				t.Errorf("unexpected IP family")
			}
		})
	}
}
