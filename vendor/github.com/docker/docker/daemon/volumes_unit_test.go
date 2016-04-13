package daemon

import "testing"

func TestParseVolumeFrom(t *testing.T) {
	cases := []struct {
		spec    string
		expId   string
		expMode string
		fail    bool
	}{
		{"", "", "", true},
		{"foobar", "foobar", "rw", false},
		{"foobar:rw", "foobar", "rw", false},
		{"foobar:ro", "foobar", "ro", false},
		{"foobar:baz", "", "", true},
	}

	for _, c := range cases {
		id, mode, err := parseVolumesFrom(c.spec)
		if c.fail {
			if err == nil {
				t.Fatalf("Expected error, was nil, for spec %s\n", c.spec)
			}
			continue
		}

		if id != c.expId {
			t.Fatalf("Expected id %s, was %s, for spec %s\n", c.expId, id, c.spec)
		}
		if mode != c.expMode {
			t.Fatalf("Expected mode %s, was %s for spec %s\n", c.expMode, mode, c.spec)
		}
	}
}
