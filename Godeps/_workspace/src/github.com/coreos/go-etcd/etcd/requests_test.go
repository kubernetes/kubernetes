package etcd

import "testing"

func TestKeyToPath(t *testing.T) {
	tests := []struct {
		key   string
		wpath string
	}{
		{"", "keys/"},
		{"foo", "keys/foo"},
		{"foo/bar", "keys/foo/bar"},
		{"%z", "keys/%25z"},
		{"/", "keys/"},
	}
	for i, tt := range tests {
		path := keyToPath(tt.key)
		if path != tt.wpath {
			t.Errorf("#%d: path = %s, want %s", i, path, tt.wpath)
		}
	}
}
