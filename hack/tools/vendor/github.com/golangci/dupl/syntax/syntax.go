package syntax

import (
	"crypto/sha1"

	"github.com/golangci/dupl/suffixtree"
)

type Node struct {
	Type     int
	Filename string
	Pos, End int
	Children []*Node
	Owns     int
}

func NewNode() *Node {
	return &Node{}
}

func (n *Node) AddChildren(children ...*Node) {
	n.Children = append(n.Children, children...)
}

func (n *Node) Val() int {
	return n.Type
}

type Match struct {
	Hash  string
	Frags [][]*Node
}

func Serialize(n *Node) []*Node {
	stream := make([]*Node, 0, 10)
	serial(n, &stream)
	return stream
}

func serial(n *Node, stream *[]*Node) int {
	*stream = append(*stream, n)
	var count int
	for _, child := range n.Children {
		count += serial(child, stream)
	}
	n.Owns = count
	return count + 1
}

// FindSyntaxUnits finds all complete syntax units in the match group and returns them
// with the corresponding hash.
func FindSyntaxUnits(data []*Node, m suffixtree.Match, threshold int) Match {
	if len(m.Ps) == 0 {
		return Match{}
	}
	firstSeq := data[m.Ps[0] : m.Ps[0]+m.Len]
	indexes := getUnitsIndexes(firstSeq, threshold)

	// TODO: is this really working?
	indexCnt := len(indexes)
	if indexCnt > 0 {
		lasti := indexes[indexCnt-1]
		firstn := firstSeq[lasti]
		for i := 1; i < len(m.Ps); i++ {
			n := data[int(m.Ps[i])+lasti]
			if firstn.Owns != n.Owns {
				indexes = indexes[:indexCnt-1]
				break
			}
		}
	}
	if len(indexes) == 0 || isCyclic(indexes, firstSeq) || spansMultipleFiles(indexes, firstSeq) {
		return Match{}
	}

	match := Match{Frags: make([][]*Node, len(m.Ps))}
	for i, pos := range m.Ps {
		match.Frags[i] = make([]*Node, len(indexes))
		for j, index := range indexes {
			match.Frags[i][j] = data[int(pos)+index]
		}
	}

	lastIndex := indexes[len(indexes)-1]
	match.Hash = hashSeq(firstSeq[indexes[0] : lastIndex+firstSeq[lastIndex].Owns])
	return match
}

func getUnitsIndexes(nodeSeq []*Node, threshold int) []int {
	var indexes []int
	var split bool
	for i := 0; i < len(nodeSeq); {
		n := nodeSeq[i]
		switch {
		case n.Owns >= len(nodeSeq)-i:
			// not complete syntax unit
			i++
			split = true
			continue
		case n.Owns+1 < threshold:
			split = true
		default:
			if split {
				indexes = indexes[:0]
				split = false
			}
			indexes = append(indexes, i)
		}
		i += n.Owns + 1
	}
	return indexes
}

// isCyclic finds out whether there is a repetive pattern in the found clone. If positive,
// it return false to point out that the clone would be redundant.
func isCyclic(indexes []int, nodes []*Node) bool {
	cnt := len(indexes)
	if cnt <= 1 {
		return false
	}

	alts := make(map[int]bool)
	for i := 1; i <= cnt/2; i++ {
		if cnt%i == 0 {
			alts[i] = true
		}
	}

	for i := 0; i < indexes[cnt/2]; i++ {
		nstart := nodes[i+indexes[0]]
	AltLoop:
		for alt := range alts {
			for j := alt; j < cnt; j += alt {
				index := i + indexes[j]
				if index < len(nodes) {
					nalt := nodes[index]
					if nstart.Owns == nalt.Owns && nstart.Type == nalt.Type {
						continue
					}
				} else if i >= indexes[alt] {
					return true
				}
				delete(alts, alt)
				continue AltLoop
			}
		}
		if len(alts) == 0 {
			return false
		}
	}
	return true
}

func spansMultipleFiles(indexes []int, nodes []*Node) bool {
	if len(indexes) < 2 {
		return false
	}
	f := nodes[indexes[0]].Filename
	for i := 1; i < len(indexes); i++ {
		if nodes[indexes[i]].Filename != f {
			return true
		}
	}
	return false
}

func hashSeq(nodes []*Node) string {
	h := sha1.New()
	bytes := make([]byte, len(nodes))
	for i, node := range nodes {
		bytes[i] = byte(node.Type)
	}
	h.Write(bytes)
	return string(h.Sum(nil))
}
