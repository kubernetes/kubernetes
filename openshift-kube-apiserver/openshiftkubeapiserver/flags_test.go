package openshiftkubeapiserver

import (
	"testing"

	"github.com/openshift/api/config/v1"
)

func TestSNICertKeys(t *testing.T) {
	testCases := []struct {
		names    []string
		expected string
	}{
		{names: []string{"foo"}, expected: "secret.crt,secret.key:foo"},
		{names: []string{"foo", "bar"}, expected: "secret.crt,secret.key:foo,bar"},
		{expected: "secret.crt,secret.key"},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			result := sniCertKeys([]v1.NamedCertificate{{Names: tc.names, CertInfo: v1.CertInfo{CertFile: "secret.crt", KeyFile: "secret.key"}}})
			if len(result) != 1 || result[0] != tc.expected {
				t.Errorf("expected: %v, actual: %v", []string{tc.expected}, result)
			}
		})
	}
}
