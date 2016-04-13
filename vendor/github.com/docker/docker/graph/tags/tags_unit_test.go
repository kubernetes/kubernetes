package tags

import (
	"testing"
)

func TestValidTagName(t *testing.T) {
	validTags := []string{"9", "foo", "foo-test", "bar.baz.boo"}
	for _, tag := range validTags {
		if err := ValidateTagName(tag); err != nil {
			t.Errorf("'%s' should've been a valid tag", tag)
		}
	}
}

func TestInvalidTagName(t *testing.T) {
	validTags := []string{"-9", ".foo", "-test", ".", "-"}
	for _, tag := range validTags {
		if err := ValidateTagName(tag); err == nil {
			t.Errorf("'%s' shouldn't have been a valid tag", tag)
		}
	}
}
