package i18n

import (
	"testing"
)

func TestTranslation(t *testing.T) {
	err := loadTranslationsInternal("en-US")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	result := T("test_string")
	if result != "foo" {
		t.Errorf("expected: %s, saw: %s", "foo", result)
	}
}
