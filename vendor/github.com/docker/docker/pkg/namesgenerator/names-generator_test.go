package namesgenerator

import (
	"strings"
	"testing"
)

// Make sure the generated names are awesome
func TestGenerateAwesomeNames(t *testing.T) {
	name := GetRandomName(0)
	if !isAwesome(name) {
		t.Fatalf("Generated name '%s' is not awesome.", name)
	}
}

func TestNameFormat(t *testing.T) {
	name := GetRandomName(0)
	if !strings.Contains(name, "_") {
		t.Fatalf("Generated name does not contain an underscore")
	}
	if strings.ContainsAny(name, "0123456789") {
		t.Fatalf("Generated name contains numbers!")
	}
}

func TestNameRetries(t *testing.T) {
	name := GetRandomName(1)
	if !strings.Contains(name, "_") {
		t.Fatalf("Generated name does not contain an underscore")
	}
	if !strings.ContainsAny(name, "0123456789") {
		t.Fatalf("Generated name doesn't contain a number")
	}

}

// To be awesome, a container name must involve cool inventors, be easy to remember,
// be at least mildly funny, and always be politically correct for enterprise adoption.
func isAwesome(name string) bool {
	coolInventorNames := true
	easyToRemember := true
	mildlyFunnyOnOccasion := true
	politicallyCorrect := true
	return coolInventorNames && easyToRemember && mildlyFunnyOnOccasion && politicallyCorrect
}
