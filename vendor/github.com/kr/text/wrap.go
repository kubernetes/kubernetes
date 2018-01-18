package text

import (
	"bytes"
	"math"
)

var (
	nl = []byte{'\n'}
	sp = []byte{' '}
)

const defaultPenalty = 1e5

// Wrap wraps s into a paragraph of lines of length lim, with minimal
// raggedness.
func Wrap(s string, lim int) string {
	return string(WrapBytes([]byte(s), lim))
}

// WrapBytes wraps b into a paragraph of lines of length lim, with minimal
// raggedness.
func WrapBytes(b []byte, lim int) []byte {
	words := bytes.Split(bytes.Replace(bytes.TrimSpace(b), nl, sp, -1), sp)
	var lines [][]byte
	for _, line := range WrapWords(words, 1, lim, defaultPenalty) {
		lines = append(lines, bytes.Join(line, sp))
	}
	return bytes.Join(lines, nl)
}

// WrapWords is the low-level line-breaking algorithm, useful if you need more
// control over the details of the text wrapping process. For most uses, either
// Wrap or WrapBytes will be sufficient and more convenient. 
//
// WrapWords splits a list of words into lines with minimal "raggedness",
// treating each byte as one unit, accounting for spc units between adjacent
// words on each line, and attempting to limit lines to lim units. Raggedness
// is the total error over all lines, where error is the square of the
// difference of the length of the line and lim. Too-long lines (which only
// happen when a single word is longer than lim units) have pen penalty units
// added to the error.
func WrapWords(words [][]byte, spc, lim, pen int) [][][]byte {
	n := len(words)

	length := make([][]int, n)
	for i := 0; i < n; i++ {
		length[i] = make([]int, n)
		length[i][i] = len(words[i])
		for j := i + 1; j < n; j++ {
			length[i][j] = length[i][j-1] + spc + len(words[j])
		}
	}

	nbrk := make([]int, n)
	cost := make([]int, n)
	for i := range cost {
		cost[i] = math.MaxInt32
	}
	for i := n - 1; i >= 0; i-- {
		if length[i][n-1] <= lim {
			cost[i] = 0
			nbrk[i] = n
		} else {
			for j := i + 1; j < n; j++ {
				d := lim - length[i][j-1]
				c := d*d + cost[j]
				if length[i][j-1] > lim {
					c += pen // too-long lines get a worse penalty
				}
				if c < cost[i] {
					cost[i] = c
					nbrk[i] = j
				}
			}
		}
	}

	var lines [][][]byte
	i := 0
	for i < n {
		lines = append(lines, words[i:nbrk[i]])
		i = nbrk[i]
	}
	return lines
}
