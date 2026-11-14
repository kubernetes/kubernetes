package goabnf

import (
	"fmt"
	"slices"
	"strings"
)

type node struct {
	Rulename       string
	index, lowlink int
	onStack        bool
	Dependencies   []string
}

// Depgraph contains for each rule the rules it depends on.
// Notice it does not differentiate between a mandatory dependency
// or an avoidable one.
type Depgraph map[string]*node

// DependencyGraph creates a dependency graph for a whole grammar.
// It includes core rules only if necessary.
func (g *Grammar) DependencyGraph() Depgraph {
	graph := Depgraph{}
	for _, corerule := range coreRules {
		graph[strings.ToLower(corerule.Name)] = &node{
			Rulename:     corerule.Name,
			Dependencies: getDependencies(corerule.Alternation),
		}
	}
	for _, rule := range g.Rulemap {
		graph[strings.ToLower(rule.Name)] = &node{
			Rulename:     rule.Name,
			Dependencies: getDependencies(rule.Alternation),
		}
	}
	return graph
}

func getDependencies(alt Alternation) []string {
	deps := []string{}
	for _, conc := range alt.Concatenations {
		for _, rep := range conc.Repetitions {
			switch v := rep.Element.(type) {
			case ElemGroup:
				deps = appendDeps(deps, getDependencies(v.Alternation)...)
			case ElemOption:
				deps = appendDeps(deps, getDependencies(v.Alternation)...)
			case ElemRulename:
				deps = appendDeps(deps, strings.ToLower(v.Name))
			}
		}
	}
	return deps
}

func appendDeps(deps []string, ndeps ...string) []string {
	for _, ndep := range ndeps {
		if !slices.Contains(deps, ndep) {
			deps = append(deps, ndep)
		}
	}
	return deps
}

// Mermaid returns a flowchart of the dependency graph.
func (dg Depgraph) Mermaid() string {
	var out strings.Builder
	out.WriteString("flowchart TD\n")
	for _, node := range dg {
		for _, dep := range node.Dependencies {
			fmt.Fprintf(&out, "\t%s --> %s\n", node.Rulename, dep)
		}
		out.WriteString("\n")
	}
	return out.String()
}

// IsDag find Strongly Connected Components using Tarjan's algorithm
// and returns whether it contains cycle or not.
// Could be improved by Nuutila's or Pearce's algorithms, or replaced
// by Kosaraju's algorithm.
func (g *Grammar) IsDAG() bool {
	scc := &cycle{
		index: 0,
		stack: []*node{},
		dg:    g.DependencyGraph(),
	}
	scc.find()

	for _, sccs := range scc.sccs {
		if len(sccs) > 1 {
			return false
		}
	}
	return true
}

// RuleContainsCycle returns whether the rule contains a cycle or not.
// It travels through the whole rule dependency graph, such that
// it checks if the rule is cyclic AND if one of its dependency is too.
func (g *Grammar) RuleContainsCycle(rulename string) (bool, error) {
	// Check the rule exists
	rule := GetRule(rulename, g.Rulemap)
	if rule == nil {
		return false, &ErrRuleNotFound{
			Rulename: rulename,
		}
	}

	// Get all SCCs
	scc := &cycle{
		index: 0,
		stack: []*node{},
		dg:    g.DependencyGraph(),
	}
	scc.find()

	return ruleContainsCycle(scc.sccs, rulename), nil
}

func ruleContainsCycle(sccs [][]*node, rulename string) bool {
	// Index nodes by lowercase name and flag those sitting in a non-trivial SCC
	// (size > 1 == a cycle). Precomputing this removes the O(V) SCC scan that
	// the previous implementation did on every recursive step.
	nodeOf := make(map[string]*node, len(sccs))
	inCycle := make(map[string]bool, len(sccs))

	for _, s := range sccs {
		nontrivial := len(s) > 1
		for _, nd := range s {
			key := strings.ToLower(nd.Rulename)
			nodeOf[key] = nd
			inCycle[key] = nontrivial
		}
	}
	return ruleCycleVisit(strings.ToLower(rulename), nodeOf, inCycle, map[string]bool{})
}

// ruleCycleVisit reports whether the rule (or any transitive dependency) is
// cyclic. The visited set keeps each rule explored at most once: without it a
// diamond-shaped dependency DAG is walked along every root-to-leaf path, which
// is exponential. Marking on entry is safe because a cyclic rule returns true
// here before it can ever be marked-and-skipped, so a memoized rule is always
// genuinely acyclic.
func ruleCycleVisit(key string, nodeOf map[string]*node, inCycle, visited map[string]bool) bool {
	if visited[key] {
		return false
	}
	visited[key] = true
	nd := nodeOf[key]
	if nd == nil {
		// Unknown rule: treat as acyclic (matches the previous behaviour).
		return false
	}

	// Self-loop (a size-1 SCC with a self-edge is not flagged by inCycle).
	for _, dep := range nd.Dependencies {
		if strings.ToLower(dep) == key {
			return true
		}
	}
	if inCycle[key] {
		// If it depends on itself or is part of an SCC, then is cylic
		return true
	}

	for _, dep := range nd.Dependencies {
		dk := strings.ToLower(dep)
		if dk == key {
			continue
		}
		if ruleCycleVisit(dk, nodeOf, inCycle, visited) {
			return true
		}
	}
	return false
}

type cycle struct {
	index int
	stack []*node
	sccs  [][]*node

	dg Depgraph
}

func (c *cycle) find() {
	for _, v := range c.dg {
		if v.index == 0 {
			c.strongconnect(v)
		}
	}
}

func (c *cycle) strongconnect(v *node) {
	// Set the depth index for v to the smallest unused index
	v.index = c.index
	v.lowlink = c.index
	c.index++
	c.stack = append(c.stack, v)
	v.onStack = true

	// Consider successors of v
	for _, dep := range v.Dependencies {
		w, ok := c.dg[dep]
		if !ok {
			// core rules, as we know they won't have a cycle thus
			// no SCC, we don't need to recurse.
			continue
		}
		if w.index == 0 {
			// Successor w has not yet been visited; recurse on it
			c.strongconnect(w)
			v.lowlink = min(v.lowlink, w.lowlink)
		} else {
			if w.onStack {
				// Successor w is in stack S and hence in the current SCC
				// If w is not on stack, then (v, w) is an edge pointing
				// to an SCC already found and must be ignored.
				v.lowlink = min(v.lowlink, w.index)
			}
		}
	}
	// If v is a root node, pop the stack and generate an SCC
	if v.lowlink == v.index {
		scc := []*node{}
		w := (*node)(nil)
		for w == nil || !strings.EqualFold(v.Rulename, w.Rulename) {
			w = c.stack[len(c.stack)-1]
			c.stack = c.stack[:len(c.stack)-1]
			w.onStack = false
			scc = append(scc, w)
		}
		c.sccs = append(c.sccs, scc)
	}
}
