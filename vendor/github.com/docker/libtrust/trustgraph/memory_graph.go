package trustgraph

import (
	"strings"

	"github.com/docker/libtrust"
)

type grantNode struct {
	grants   []*Grant
	children map[string]*grantNode
}

type memoryGraph struct {
	roots map[string]*grantNode
}

func newGrantNode() *grantNode {
	return &grantNode{
		grants:   []*Grant{},
		children: map[string]*grantNode{},
	}
}

// NewMemoryGraph returns a new in memory trust graph created from
// a static list of grants.  This graph is immutable after creation
// and any alterations should create a new instance.
func NewMemoryGraph(grants []*Grant) TrustGraph {
	roots := map[string]*grantNode{}
	for _, grant := range grants {
		parts := strings.Split(grant.Grantee, "/")
		nodes := roots
		var node *grantNode
		var nodeOk bool
		for _, part := range parts {
			node, nodeOk = nodes[part]
			if !nodeOk {
				node = newGrantNode()
				nodes[part] = node
			}
			if part != "" {
				node.grants = append(node.grants, grant)
			}
			nodes = node.children
		}
	}
	return &memoryGraph{roots}
}

func (g *memoryGraph) getGrants(name string) []*Grant {
	nameParts := strings.Split(name, "/")
	nodes := g.roots
	var node *grantNode
	var nodeOk bool
	for _, part := range nameParts {
		node, nodeOk = nodes[part]
		if !nodeOk {
			return nil
		}
		nodes = node.children
	}
	return node.grants
}

func isSubName(name, sub string) bool {
	if strings.HasPrefix(name, sub) {
		if len(name) == len(sub) || name[len(sub)] == '/' {
			return true
		}
	}
	return false
}

type walkFunc func(*Grant, []*Grant) bool

func foundWalkFunc(*Grant, []*Grant) bool {
	return true
}

func (g *memoryGraph) walkGrants(start, target string, permission uint16, f walkFunc, chain []*Grant, visited map[*Grant]bool, collect bool) bool {
	if visited == nil {
		visited = map[*Grant]bool{}
	}
	grants := g.getGrants(start)
	subGrants := make([]*Grant, 0, len(grants))
	for _, grant := range grants {
		if visited[grant] {
			continue
		}
		visited[grant] = true
		if grant.Permission&permission == permission {
			if isSubName(target, grant.Subject) {
				if f(grant, chain) {
					return true
				}
			} else {
				subGrants = append(subGrants, grant)
			}
		}
	}
	for _, grant := range subGrants {
		var chainCopy []*Grant
		if collect {
			chainCopy = make([]*Grant, len(chain)+1)
			copy(chainCopy, chain)
			chainCopy[len(chainCopy)-1] = grant
		} else {
			chainCopy = nil
		}

		if g.walkGrants(grant.Subject, target, permission, f, chainCopy, visited, collect) {
			return true
		}
	}
	return false
}

func (g *memoryGraph) Verify(key libtrust.PublicKey, node string, permission uint16) (bool, error) {
	return g.walkGrants(key.KeyID(), node, permission, foundWalkFunc, nil, nil, false), nil
}

func (g *memoryGraph) GetGrants(key libtrust.PublicKey, node string, permission uint16) ([][]*Grant, error) {
	grants := [][]*Grant{}
	collect := func(grant *Grant, chain []*Grant) bool {
		grantChain := make([]*Grant, len(chain)+1)
		copy(grantChain, chain)
		grantChain[len(grantChain)-1] = grant
		grants = append(grants, grantChain)
		return false
	}
	g.walkGrants(key.KeyID(), node, permission, collect, nil, nil, true)
	return grants, nil
}
