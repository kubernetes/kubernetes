package gc

import (
	"context"
	"reflect"
	"testing"
)

func TestTricolorBasic(t *testing.T) {
	roots := []string{"A", "C"}
	all := []string{"A", "B", "C", "D", "E", "F", "G", "H"}
	refs := map[string][]string{
		"A": {"B"},
		"B": {"A"},
		"C": {"D", "F", "B"},
		"E": {"F", "G"},
		"F": {"H"},
	}

	expected := toNodes([]string{"A", "B", "C", "D", "F", "H"})

	reachable, err := Tricolor(toNodes(roots), lookup(refs))
	if err != nil {
		t.Fatal(err)
	}

	var sweeped []Node
	for _, a := range toNodes(all) {
		if _, ok := reachable[a]; ok {
			sweeped = append(sweeped, a)
		}
	}

	if !reflect.DeepEqual(sweeped, expected) {
		t.Fatalf("incorrect unreachable set: %v != %v", sweeped, expected)
	}
}

func TestConcurrentBasic(t *testing.T) {
	roots := []string{"A", "C"}
	all := []string{"A", "B", "C", "D", "E", "F", "G", "H", "I"}
	refs := map[string][]string{
		"A": {"B"},
		"B": {"A"},
		"C": {"D", "F", "B"},
		"E": {"F", "G"},
		"F": {"H"},
		"G": {"I"},
	}

	expected := toNodes([]string{"A", "B", "C", "D", "F", "H"})

	ctx := context.Background()
	rootC := make(chan Node)
	go func() {
		writeNodes(ctx, rootC, toNodes(roots))
		close(rootC)
	}()

	reachable, err := ConcurrentMark(ctx, rootC, lookupc(refs))
	if err != nil {
		t.Fatal(err)
	}

	var sweeped []Node
	for _, a := range toNodes(all) {
		if _, ok := reachable[a]; ok {
			sweeped = append(sweeped, a)
		}
	}

	if !reflect.DeepEqual(sweeped, expected) {
		t.Fatalf("incorrect unreachable set: %v != %v", sweeped, expected)
	}
}

func writeNodes(ctx context.Context, nc chan<- Node, nodes []Node) {
	for _, n := range nodes {
		select {
		case nc <- n:
		case <-ctx.Done():
			return
		}
	}
}

func lookup(refs map[string][]string) func(id Node) ([]Node, error) {
	return func(ref Node) ([]Node, error) {
		return toNodes(refs[ref.Key]), nil
	}
}

func lookupc(refs map[string][]string) func(context.Context, Node, func(Node)) error {
	return func(ctx context.Context, ref Node, fn func(Node)) error {
		for _, n := range toNodes(refs[ref.Key]) {
			fn(n)
		}
		return nil
	}
}

func toNodes(s []string) []Node {
	n := make([]Node, len(s))
	for i := range s {
		n[i] = Node{
			Key: s[i],
		}
	}
	return n
}

func newScanner(refs []string) *stringScanner {
	return &stringScanner{
		i: -1,
		s: refs,
	}
}

type stringScanner struct {
	i int
	s []string
}

func (ss *stringScanner) Next() bool {
	ss.i++
	return ss.i < len(ss.s)
}

func (ss *stringScanner) Node() Node {
	return Node{
		Key: ss.s[ss.i],
	}
}

func (ss *stringScanner) Cleanup() error {
	ss.s[ss.i] = ""
	return nil
}

func (ss *stringScanner) Err() error {
	return nil
}

func (ss *stringScanner) All() []Node {
	remaining := make([]Node, 0, len(ss.s))
	for _, s := range ss.s {
		if s != "" {
			remaining = append(remaining, Node{
				Key: s,
			})
		}
	}
	return remaining
}
