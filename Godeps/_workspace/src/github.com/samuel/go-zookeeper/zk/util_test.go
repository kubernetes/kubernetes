package zk

import "testing"

func TestFormatServers(t *testing.T) {
	servers := []string{"127.0.0.1:2181", "127.0.0.42", "127.0.42.1:8811"}
	r := []string{"127.0.0.1:2181", "127.0.0.42:2181", "127.0.42.1:8811"}

	var s []string
	s = FormatServers(servers)

	for i := range s {
		if s[i] != r[i] {
			t.Errorf("%v should equal %v", s[i], r[i])
		}
	}
}
