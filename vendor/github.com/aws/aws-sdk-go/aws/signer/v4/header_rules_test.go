package v4

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRuleCheckWhitelist(t *testing.T) {
	w := whitelist{
		mapRule{
			"Cache-Control": struct{}{},
		},
	}

	assert.True(t, w.IsValid("Cache-Control"))
	assert.False(t, w.IsValid("Cache-"))
}

func TestRuleCheckBlacklist(t *testing.T) {
	b := blacklist{
		mapRule{
			"Cache-Control": struct{}{},
		},
	}

	assert.False(t, b.IsValid("Cache-Control"))
	assert.True(t, b.IsValid("Cache-"))
}

func TestRuleCheckPattern(t *testing.T) {
	p := patterns{"X-Amz-Meta-"}

	assert.True(t, p.IsValid("X-Amz-Meta-"))
	assert.True(t, p.IsValid("X-Amz-Meta-Star"))
	assert.False(t, p.IsValid("Cache-"))
}

func TestRuleComplexWhitelist(t *testing.T) {
	w := rules{
		whitelist{
			mapRule{
				"Cache-Control": struct{}{},
			},
		},
		patterns{"X-Amz-Meta-"},
	}

	r := rules{
		inclusiveRules{patterns{"X-Amz-"}, blacklist{w}},
	}

	assert.True(t, r.IsValid("X-Amz-Blah"))
	assert.False(t, r.IsValid("X-Amz-Meta-"))
	assert.False(t, r.IsValid("X-Amz-Meta-Star"))
	assert.False(t, r.IsValid("Cache-Control"))
}
