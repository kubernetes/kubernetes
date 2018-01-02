package netlabel

import (
	"testing"

	_ "github.com/docker/libnetwork/testutils"
)

var input = []struct {
	label string
	key   string
	value string
}{
	{"com.directory.person.name=joe", "com.directory.person.name", "joe"},
	{"com.directory.person.age=24", "com.directory.person.age", "24"},
	{"com.directory.person.address=1234 First st.", "com.directory.person.address", "1234 First st."},
	{"com.directory.person.friends=", "com.directory.person.friends", ""},
	{"com.directory.person.nickname=o=u=8", "com.directory.person.nickname", "o=u=8"},
	{"", "", ""},
	{"com.directory.person.student", "com.directory.person.student", ""},
}

func TestKeyValue(t *testing.T) {
	for _, i := range input {
		k, v := KeyValue(i.label)
		if k != i.key || v != i.value {
			t.Fatalf("unexpected: %s, %s", k, v)
		}
	}
}
