package auth

import "testing"

func TestH(t *testing.T) {
	const hello = "Hello, world!"
	const hello_md5 = "6cd3556deb0da54bca060b4c39479839"
	h := H(hello)
	if h != hello_md5 {
		t.Fatal("Incorrect digest for test string:", h, "instead of", hello_md5)
	}
}
