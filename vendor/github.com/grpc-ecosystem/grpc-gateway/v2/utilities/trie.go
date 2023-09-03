package utilities

import (
	"sort"
)

// DoubleArray is a Double Array implementation of trie on sequences of strings.
type DoubleArray struct {
	// Encoding keeps an encoding from string to int
	Encoding map[string]int
	// Base is the base array of Double Array
	Base []int
	// Check is the check array of Double Array
	Check []int
}

// NewDoubleArray builds a DoubleArray from a set of sequences of strings.
func NewDoubleArray(seqs [][]string) *DoubleArray {
	da := &DoubleArray{Encoding: make(map[string]int)}
	if len(seqs) == 0 {
		return da
	}

	encoded := registerTokens(da, seqs)
	sort.Sort(byLex(encoded))

	root := node{row: -1, col: -1, left: 0, right: len(encoded)}
	addSeqs(da, encoded, 0, root)

	for i := len(da.Base); i > 0; i-- {
		if da.Check[i-1] != 0 {
			da.Base = da.Base[:i]
			da.Check = da.Check[:i]
			break
		}
	}
	return da
}

func registerTokens(da *DoubleArray, seqs [][]string) [][]int {
	var result [][]int
	for _, seq := range seqs {
		var encoded []int
		for _, token := range seq {
			if _, ok := da.Encoding[token]; !ok {
				da.Encoding[token] = len(da.Encoding)
			}
			encoded = append(encoded, da.Encoding[token])
		}
		result = append(result, encoded)
	}
	for i := range result {
		result[i] = append(result[i], len(da.Encoding))
	}
	return result
}

type node struct {
	row, col    int
	left, right int
}

func (n node) value(seqs [][]int) int {
	return seqs[n.row][n.col]
}

func (n node) children(seqs [][]int) []*node {
	var result []*node
	lastVal := int(-1)
	last := new(node)
	for i := n.left; i < n.right; i++ {
		if lastVal == seqs[i][n.col+1] {
			continue
		}
		last.right = i
		last = &node{
			row:  i,
			col:  n.col + 1,
			left: i,
		}
		result = append(result, last)
	}
	last.right = n.right
	return result
}

func addSeqs(da *DoubleArray, seqs [][]int, pos int, n node) {
	ensureSize(da, pos)

	children := n.children(seqs)
	var i int
	for i = 1; ; i++ {
		ok := func() bool {
			for _, child := range children {
				code := child.value(seqs)
				j := i + code
				ensureSize(da, j)
				if da.Check[j] != 0 {
					return false
				}
			}
			return true
		}()
		if ok {
			break
		}
	}
	da.Base[pos] = i
	for _, child := range children {
		code := child.value(seqs)
		j := i + code
		da.Check[j] = pos + 1
	}
	terminator := len(da.Encoding)
	for _, child := range children {
		code := child.value(seqs)
		if code == terminator {
			continue
		}
		j := i + code
		addSeqs(da, seqs, j, *child)
	}
}

func ensureSize(da *DoubleArray, i int) {
	for i >= len(da.Base) {
		da.Base = append(da.Base, make([]int, len(da.Base)+1)...)
		da.Check = append(da.Check, make([]int, len(da.Check)+1)...)
	}
}

type byLex [][]int

func (l byLex) Len() int      { return len(l) }
func (l byLex) Swap(i, j int) { l[i], l[j] = l[j], l[i] }
func (l byLex) Less(i, j int) bool {
	si := l[i]
	sj := l[j]
	var k int
	for k = 0; k < len(si) && k < len(sj); k++ {
		if si[k] < sj[k] {
			return true
		}
		if si[k] > sj[k] {
			return false
		}
	}
	return k < len(sj)
}

// HasCommonPrefix determines if any sequence in the DoubleArray is a prefix of the given sequence.
func (da *DoubleArray) HasCommonPrefix(seq []string) bool {
	if len(da.Base) == 0 {
		return false
	}

	var i int
	for _, t := range seq {
		code, ok := da.Encoding[t]
		if !ok {
			break
		}
		j := da.Base[i] + code
		if len(da.Check) <= j || da.Check[j] != i+1 {
			break
		}
		i = j
	}
	j := da.Base[i] + len(da.Encoding)
	if len(da.Check) <= j || da.Check[j] != i+1 {
		return false
	}
	return true
}
