package auth

import (
	"testing"
)

func TestRepoName(t *testing.T) {
	if x := repoName("/yapp.ss.git/HEAD"); x != "yapp.ss.git" {
		t.Errorf("Should have been 'yapp.js.git' is '%s'", x)
	}

	if x := repoName("aarono/gogo-proxy/HEAD"); x != "aarono/gogo-proxy" {
		t.Errorf("Should have been 'aarono/gogo-proxy' is '%s'", x)
	}
}
