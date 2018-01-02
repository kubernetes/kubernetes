package docker

import (
	"testing"

	"github.com/containerd/containerd/reference"
	"github.com/stretchr/testify/assert"
)

func TestRepositoryScope(t *testing.T) {
	testCases := []struct {
		refspec  reference.Spec
		push     bool
		expected string
	}{
		{
			refspec: reference.Spec{
				Locator: "host/foo/bar",
				Object:  "ignored",
			},
			push:     false,
			expected: "repository:foo/bar:pull",
		},
		{
			refspec: reference.Spec{
				Locator: "host:4242/foo/bar",
				Object:  "ignored",
			},
			push:     true,
			expected: "repository:foo/bar:pull,push",
		},
	}
	for _, x := range testCases {
		actual, err := repositoryScope(x.refspec, x.push)
		assert.NoError(t, err)
		assert.Equal(t, x.expected, actual)
	}
}
