package rope

import (
	"bytes"
	"testing"

	"github.com/bruth/assert"
)

var (
	small = leaf("x")      // small+small is below the leaf concat threshold.
	big   = leaf("bigger") // A string above the leaf concat threshold.
)

func lowerCoalesceThreshold() func() {
	// Temporarily lower auto-coalescing threshold.
	thr := concatThreshold
	concatThreshold = 3
	return func() { concatThreshold = thr }
}

func expectConcat(t *testing.T, n node) *concat {
	cc, ok := n.(*concat)
	if !ok {
		t.Fatalf("concatenation is not *concat: %#v", n)
	}
	return cc
}

func TestConcSimple(t *testing.T) {
	assert.Equal(t, small, conc(emptyNode, small, -1, -1))
	assert.Equal(t, small, conc(small, emptyNode, -1, -1))
	assert.Equal(t, small+small, conc(small, small, -1, -1))
}

func TestConcNoCoalesce(t *testing.T) {
	defer lowerCoalesceThreshold()()

	result := conc(big, big, -1, -1)
	cc := expectConcat(t, result)

	assert.Equal(t, big+big, flatten(cc))
	assert.Equal(t, big, cc.Left)
	assert.Equal(t, big, cc.Right)
	assert.Equal(t, big.length(), cc.Split)
	assert.Equal(t, rLenT(big.length()), cc.RLen)
	assert.Equal(t, depthT(1), cc.depth())
}

func TestConcCoalesceLeft(t *testing.T) {
	defer lowerCoalesceThreshold()()

	base := conc(small, big, -1, -1)
	_ = expectConcat(t, base)

	n := conc(small, base, -1, -1)
	cc := expectConcat(t, n)

	assert.Equal(t, small+small+big, flatten(cc))
	assert.Equal(t, small+small, cc.Left)
	assert.Equal(t, big, cc.Right)
	assert.Equal(t, 2*small.length(), cc.Split)
	assert.Equal(t, rLenT(big.length()), cc.RLen)
	assert.Equal(t, depthT(1), cc.depth())
}

func TestConcCoalesceRight(t *testing.T) {
	defer lowerCoalesceThreshold()()

	base := conc(big, small, -1, -1)
	_ = expectConcat(t, base)

	n := conc(base, small, -1, -1)
	cc := expectConcat(t, n)

	assert.Equal(t, big+small+small, flatten(cc))
	assert.Equal(t, big, cc.Left)
	assert.Equal(t, big.length(), cc.Split)
	assert.Equal(t, small+small, cc.Right)
	assert.Equal(t, rLenT(2*small.length()), cc.RLen)
	assert.Equal(t, depthT(1), cc.depth())
}

func flatten(n node) leaf {
	buf := &bytes.Buffer{}
	_, _ = n.WriteTo(buf)
	return leaf(buf.String())
}
