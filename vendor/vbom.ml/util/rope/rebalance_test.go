package rope

import (
	"testing"

	"github.com/bruth/assert"
)

func disableCoalesce() func() {
	// Temporarily disable auto-coalescing.
	thr := concatThreshold
	concatThreshold = 0
	return func() { concatThreshold = thr }
}

var rebalanceTestRopes []Rope

func init() {
	defer disableCoalesce()()

	rebalanceTestRopes = []Rope{
		New("a").Append(New("bc").Append(New("d").AppendString("ef"))), //.AppendString("g"),
		New("a").Append(New("bc").Append(New("d").AppendString("ef"))).AppendString("g"),
		New("a").AppendString("bcd").AppendString("efghijkl").AppendString("mno"),
		New("abc").AppendString("def").AppendString("def").AppendString("def").AppendString("def").AppendString("def").AppendString("ghiklmnopqrstuvwxyzklmnopqrstuvwxyz").AppendString("j").AppendString("j").AppendString("j").AppendString("j").AppendString("j").AppendString("klmnopqrstuvwxyz"),
	}
	//~ for i, r := range largeRopes {
	//~ for j := 0; j < 8; j++ {
	//~ r = r.Append(r)
	//~ }
	//~ largeRopes[i] = r
	//~ }
	//~ n := New("a")

	var r Rope
	for i := 0; i < 100; i++ {
		r = r.AppendString(string(' ' + i))
	}
	rebalanceTestRopes = append(rebalanceTestRopes, r)

	rebalanceTestRopes = append(rebalanceTestRopes, emptyRope, Rope{})
}

func TestRebalance(t *testing.T) {
	defer disableCoalesce()()

	for _, orig := range rebalanceTestRopes {
		origStr := orig.String()
		rebalanced := orig.Rebalance()
		rebalancedStr := rebalanced.String()

		//~ pretty.Println(orig, "(", orig.isBalanced(), ") ==> (", rebalanced.isBalanced(), ")", rebalanced)

		assert.Equal(t, origStr, rebalancedStr)
		if rebalanced.node != nil && orig.node != nil {
			assert.True(t, rebalanced.node.depth() <= orig.node.depth())
		}
		assert.True(t, rebalanced.isBalanced())
	}
}
