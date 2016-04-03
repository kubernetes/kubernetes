package daemon

import "testing"

func TestGetFullName(t *testing.T) {
	name, err := GetFullContainerName("testing")
	if err != nil {
		t.Fatal(err)
	}
	if name != "/testing" {
		t.Fatalf("Expected /testing got %s", name)
	}
	if _, err := GetFullContainerName(""); err == nil {
		t.Fatal("Error should not be nil")
	}
}

func TestValidContainerNames(t *testing.T) {
	invalidNames := []string{"-rm", "&sdfsfd", "safd%sd"}
	validNames := []string{"word-word", "word_word", "1weoid"}

	for _, name := range invalidNames {
		if validContainerNamePattern.MatchString(name) {
			t.Fatalf("%q is not a valid container name and was returned as valid.", name)
		}
	}

	for _, name := range validNames {
		if !validContainerNamePattern.MatchString(name) {
			t.Fatalf("%q is a valid container name and was returned as invalid.", name)
		}
	}
}
