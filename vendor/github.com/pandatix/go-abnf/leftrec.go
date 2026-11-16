package goabnf

import (
	"slices"
	"strings"
)

// Identifies left-recursive rules so the recognizer can handle them by seed-growing
// instead of refusing them. A rule is left-recursive when it can re-derive itself
// without consuming input -- i.e. it appears on a cycle of the "left-corner"
// relation: A left-corners B if B can begin a derivation of A at the same input
// position. Crucially this skips over prefixes that can vanish (zero-min repetitions,
// options, and nullable elements), so it also catches *hidden* left recursion such as
// `a = b a / "x"` where `b` is nullable -- which a purely positional check misses.

// nullableRules computes, by least fixpoint over user and core rules, which rules can
// derive the empty string.
func (g *Grammar) nullableRules() map[string]bool {
	rules := map[string]*Rule{}
	for n := range g.Rulemap {
		if r := GetRule(n, g.Rulemap); r != nil {
			rules[strings.ToLower(r.Name)] = r
		}
	}
	for n := range coreRules {
		if r := GetRule(n, g.Rulemap); r != nil {
			rules[strings.ToLower(r.Name)] = r
		}
	}
	null := map[string]bool{}
	for changed := true; changed; {
		changed = false
		for name, r := range rules {
			if !null[name] && altNullable(g, null, r.Alternation) {
				null[name] = true
				changed = true
			}
		}
	}
	return null
}

func altNullable(g *Grammar, null map[string]bool, alt Alternation) bool {
	for _, c := range alt.Concatenations {
		nullable := true
		for _, rep := range c.Repetitions {
			if !repNullable(g, null, rep) {
				nullable = false
				break
			}
		}
		if nullable {
			return true
		}
	}
	return false
}

func repNullable(g *Grammar, null map[string]bool, rep Repetition) bool {
	if rep.Min == 0 {
		return true
	}
	return elemNullable(g, null, rep.Element)
}

func elemNullable(g *Grammar, null map[string]bool, elem ElemItf) bool {
	switch v := elem.(type) {
	case ElemOption:
		return true
	case ElemGroup:
		return altNullable(g, null, v.Alternation)
	case ElemRulename:
		return null[strings.ToLower(v.Name)]
	case ElemCharVal:
		return len(v.Values) == 0
	}
	// num-val always consumes; prose-val is treated as unmatchable.
	return false
}

// leftRecursiveSCCs returns, for every left-recursive rule (keyed by lowercase
// name), the set of rule names in its left-corner strongly-connected component
// (mutually left-recursive rules grown together). Non-left-recursive rules are
// absent from the map.
func (g *Grammar) leftRecursiveSCCs() map[string][]string {
	null := g.nullableRules()

	// Build the left-corner dependency graph and reuse Tarjan's SCC finder.
	lc := Depgraph{}
	add := func(r *Rule) {
		name := strings.ToLower(r.Name)
		if _, ok := lc[name]; ok {
			return
		}
		lc[name] = &node{Rulename: r.Name, Dependencies: leftCornerDeps(g, null, r.Alternation)}
	}
	for n := range g.Rulemap {
		if r := GetRule(n, g.Rulemap); r != nil {
			add(r)
		}
	}
	for n := range coreRules {
		if r := GetRule(n, g.Rulemap); r != nil {
			add(r)
		}
	}

	finder := &cycle{index: 0, stack: []*node{}, dg: lc}
	finder.find()

	out := map[string][]string{}
	for _, scc := range finder.sccs {
		members := make([]string, len(scc))
		for i, nd := range scc {
			members[i] = strings.ToLower(nd.Rulename)
		}
		leftRecursive := len(scc) > 1
		if !leftRecursive { // size 1: left-recursive only with a self left-corner
			leftRecursive = slices.Contains(scc[0].Dependencies, members[0])
		}
		if leftRecursive {
			for _, m := range members {
				out[m] = members
			}
		}
	}
	return out
}

// leftCornerDeps collects the rules reachable as a left corner of alt: walk each
// concatenation left to right, collecting rulename references, and stop at the
// first repetition that cannot vanish.
func leftCornerDeps(g *Grammar, null map[string]bool, alt Alternation) []string {
	deps := []string{}
	var addAlt func(Alternation)
	addElem := func(e ElemItf) {
		switch v := e.(type) {
		case ElemRulename:
			deps = appendDeps(deps, strings.ToLower(v.Name))
		case ElemGroup:
			addAlt(v.Alternation)
		case ElemOption:
			addAlt(v.Alternation)
		}
	}
	addAlt = func(a Alternation) {
		for _, c := range a.Concatenations {
			for _, rep := range c.Repetitions {
				addElem(rep.Element)
				if rep.Min == 0 || elemNullable(g, null, rep.Element) {
					continue // this position can be empty; the next is also a left corner
				}
				break
			}
		}
	}
	addAlt(alt)
	return deps
}
