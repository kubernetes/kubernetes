package opts

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQuotedStringSetWithQuotes(t *testing.T) {
	value := ""
	qs := NewQuotedString(&value)
	assert.NoError(t, qs.Set(`"something"`))
	assert.Equal(t, "something", qs.String())
	assert.Equal(t, "something", value)
}

func TestQuotedStringSetWithMismatchedQuotes(t *testing.T) {
	value := ""
	qs := NewQuotedString(&value)
	assert.NoError(t, qs.Set(`"something'`))
	assert.Equal(t, `"something'`, qs.String())
}

func TestQuotedStringSetWithNoQuotes(t *testing.T) {
	value := ""
	qs := NewQuotedString(&value)
	assert.NoError(t, qs.Set("something"))
	assert.Equal(t, "something", qs.String())
}
