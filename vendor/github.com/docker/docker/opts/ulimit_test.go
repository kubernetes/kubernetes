package opts

import (
	"testing"

	"github.com/docker/go-units"
)

func TestUlimitOpt(t *testing.T) {
	ulimitMap := map[string]*units.Ulimit{
		"nofile": {"nofile", 1024, 512},
	}

	ulimitOpt := NewUlimitOpt(&ulimitMap)

	expected := "[nofile=512:1024]"
	if ulimitOpt.String() != expected {
		t.Fatalf("Expected %v, got %v", expected, ulimitOpt)
	}

	// Valid ulimit append to opts
	if err := ulimitOpt.Set("core=1024:1024"); err != nil {
		t.Fatal(err)
	}

	// Invalid ulimit type returns an error and do not append to opts
	if err := ulimitOpt.Set("notavalidtype=1024:1024"); err == nil {
		t.Fatalf("Expected error on invalid ulimit type")
	}
	expected = "[nofile=512:1024 core=1024:1024]"
	expected2 := "[core=1024:1024 nofile=512:1024]"
	result := ulimitOpt.String()
	if result != expected && result != expected2 {
		t.Fatalf("Expected %v or %v, got %v", expected, expected2, ulimitOpt)
	}

	// And test GetList
	ulimits := ulimitOpt.GetList()
	if len(ulimits) != 2 {
		t.Fatalf("Expected a ulimit list of 2, got %v", ulimits)
	}
}
