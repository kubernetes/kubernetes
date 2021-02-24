package deprecatedapirequest

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRemovedRelease(t *testing.T) {
	c := &controller{}
	rr := c.removedRelease("flowschemas.v1alpha1.flowcontrol.apiserver.k8s.io")
	assert.Equal(t, "1.21", rr)
}
