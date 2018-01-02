package registry

import "testing"

func TestLookupV1Endpoints(t *testing.T) {
	s := NewService(ServiceOptions{})

	cases := []struct {
		hostname    string
		expectedLen int
	}{
		{"example.com", 1},
		{DefaultNamespace, 0},
		{DefaultV2Registry.Host, 0},
		{IndexHostname, 0},
	}

	for _, c := range cases {
		if ret, err := s.lookupV1Endpoints(c.hostname); err != nil || len(ret) != c.expectedLen {
			t.Errorf("lookupV1Endpoints(`"+c.hostname+"`) returned %+v and %+v", ret, err)
		}
	}
}
