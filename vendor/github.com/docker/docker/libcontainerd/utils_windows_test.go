package libcontainerd

import (
	"testing"
)

func TestEnvironmentParsing(t *testing.T) {
	env := []string{"foo=bar", "car=hat", "a=b=c"}
	result := setupEnvironmentVariables(env)
	if len(result) != 3 || result["foo"] != "bar" || result["car"] != "hat" || result["a"] != "b=c" {
		t.Fatalf("Expected map[foo:bar car:hat a:b=c], got %v", result)
	}
}
