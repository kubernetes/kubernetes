/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package garbagecollector

import (
	"sort"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var (
	alphaNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("alpha"),
				},
			},
			owners: []metav1.OwnerReference{
				{UID: types.UID("bravo")},
				{UID: types.UID("charlie")},
			},
		}
	}
	bravoNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("bravo"),
				},
			},
			dependents: map[*node]struct{}{
				alphaNode(): {},
			},
		}
	}
	charlieNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("charlie"),
				},
			},
			dependents: map[*node]struct{}{
				alphaNode(): {},
			},
		}
	}
	deltaNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("delta"),
				},
			},
			owners: []metav1.OwnerReference{
				{UID: types.UID("foxtrot")},
			},
		}
	}
	echoNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("echo"),
				},
			},
		}
	}
	foxtrotNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("foxtrot"),
				},
			},
			owners: []metav1.OwnerReference{
				{UID: types.UID("golf")},
			},
			dependents: map[*node]struct{}{
				deltaNode(): {},
			},
		}
	}
	golfNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					UID: types.UID("golf"),
				},
			},
			dependents: map[*node]struct{}{
				foxtrotNode(): {},
			},
		}
	}
)

func TestToGonumGraph(t *testing.T) {
	tests := []struct {
		name      string
		uidToNode map[types.UID]*node
		expect    graph.Directed
	}{
		{
			name: "simple",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
			},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})
				return graphBuilder
			}(),
		},
		{
			name: "missing", // synthetic vertex created
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("charlie"): charlieNode(),
			},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})
				return graphBuilder
			}(),
		},
		{
			name: "drop-no-ref",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("echo"):    echoNode(),
			},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})
				return graphBuilder
			}(),
		},
		{
			name: "two-chains",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("delta"):   deltaNode(),
				types.UID("foxtrot"): foxtrotNode(),
				types.UID("golf"):    golfNode(),
			},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})

				deltaVertex := NewGonumVertex(deltaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(deltaVertex)
				foxtrotVertex := NewGonumVertex(foxtrotNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(foxtrotVertex)
				golfVertex := NewGonumVertex(golfNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(golfVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: deltaVertex,
					T: foxtrotVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: foxtrotVertex,
					T: golfVertex,
				})

				return graphBuilder
			}(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := toGonumGraph(test.uidToNode)

			compareGraphs(test.expect, actual, t)
		})
	}

}

func TestToGonumGraphObj(t *testing.T) {
	tests := []struct {
		name      string
		uidToNode map[types.UID]*node
		uids      []types.UID
		expect    graph.Directed
	}{
		{
			name: "simple",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
			},
			uids: []types.UID{types.UID("bravo")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})
				return graphBuilder
			}(),
		},
		{
			name: "missing", // synthetic vertex created
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("charlie"): charlieNode(),
			},
			uids: []types.UID{types.UID("bravo")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				return graphBuilder
			}(),
		},
		{
			name: "drop-no-ref",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("echo"):    echoNode(),
			},
			uids: []types.UID{types.UID("echo")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				return graphBuilder
			}(),
		},
		{
			name: "two-chains-from-owner",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("delta"):   deltaNode(),
				types.UID("foxtrot"): foxtrotNode(),
				types.UID("golf"):    golfNode(),
			},
			uids: []types.UID{types.UID("golf")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				deltaVertex := NewGonumVertex(deltaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(deltaVertex)
				foxtrotVertex := NewGonumVertex(foxtrotNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(foxtrotVertex)
				golfVertex := NewGonumVertex(golfNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(golfVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: deltaVertex,
					T: foxtrotVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: foxtrotVertex,
					T: golfVertex,
				})

				return graphBuilder
			}(),
		},
		{
			name: "two-chains-from-child",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("delta"):   deltaNode(),
				types.UID("foxtrot"): foxtrotNode(),
				types.UID("golf"):    golfNode(),
			},
			uids: []types.UID{types.UID("delta")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				deltaVertex := NewGonumVertex(deltaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(deltaVertex)
				foxtrotVertex := NewGonumVertex(foxtrotNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(foxtrotVertex)
				golfVertex := NewGonumVertex(golfNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(golfVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: deltaVertex,
					T: foxtrotVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: foxtrotVertex,
					T: golfVertex,
				})

				return graphBuilder
			}(),
		},
		{
			name: "two-chains-choose-both",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("delta"):   deltaNode(),
				types.UID("foxtrot"): foxtrotNode(),
				types.UID("golf"):    golfNode(),
			},
			uids: []types.UID{types.UID("delta"), types.UID("charlie")},
			expect: func() graph.Directed {
				graphBuilder := simple.NewDirectedGraph()
				alphaVertex := NewGonumVertex(alphaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(alphaVertex)
				bravoVertex := NewGonumVertex(bravoNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(bravoVertex)
				charlieVertex := NewGonumVertex(charlieNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(charlieVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: bravoVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: alphaVertex,
					T: charlieVertex,
				})

				deltaVertex := NewGonumVertex(deltaNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(deltaVertex)
				foxtrotVertex := NewGonumVertex(foxtrotNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(foxtrotVertex)
				golfVertex := NewGonumVertex(golfNode(), graphBuilder.NewNode().ID())
				graphBuilder.AddNode(golfVertex)
				graphBuilder.SetEdge(simple.Edge{
					F: deltaVertex,
					T: foxtrotVertex,
				})
				graphBuilder.SetEdge(simple.Edge{
					F: foxtrotVertex,
					T: golfVertex,
				})

				return graphBuilder
			}(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := toGonumGraphForObj(test.uidToNode, test.uids...)

			compareGraphs(test.expect, actual, t)
		})
	}
}

func compareGraphs(expected, actual graph.Directed, t *testing.T) {
	// sort the edges by from ID, then to ID
	// (the slices we get back are from map iteration, where order is not guaranteed)
	expectedNodes := expected.Nodes()
	actualNodes := actual.Nodes()
	sort.Sort(gonumByUID(expectedNodes))
	sort.Sort(gonumByUID(actualNodes))

	if len(expectedNodes) != len(actualNodes) {
		t.Fatal(spew.Sdump(actual))
	}

	for i := range expectedNodes {
		currExpected := *expectedNodes[i].(*gonumVertex)
		currActual := *actualNodes[i].(*gonumVertex)
		if currExpected.uid != currActual.uid {
			t.Errorf("expected %v, got %v", spew.Sdump(currExpected), spew.Sdump(currActual))
		}

		expectedFrom := append([]graph.Node{}, expected.From(expectedNodes[i].ID())...)
		actualFrom := append([]graph.Node{}, actual.From(actualNodes[i].ID())...)
		sort.Sort(gonumByUID(expectedFrom))
		sort.Sort(gonumByUID(actualFrom))
		if len(expectedFrom) != len(actualFrom) {
			t.Errorf("%q: expected %v, got %v", currExpected.uid, spew.Sdump(expectedFrom), spew.Sdump(actualFrom))
		}
		for i := range expectedFrom {
			currExpectedFrom := *expectedFrom[i].(*gonumVertex)
			currActualFrom := *actualFrom[i].(*gonumVertex)
			if currExpectedFrom.uid != currActualFrom.uid {
				t.Errorf("expected %v, got %v", spew.Sdump(currExpectedFrom), spew.Sdump(currActualFrom))
			}
		}
	}
}

type gonumByUID []graph.Node

func (s gonumByUID) Len() int      { return len(s) }
func (s gonumByUID) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s gonumByUID) Less(i, j int) bool {
	lhs := s[i].(*gonumVertex)
	lhsUID := string(lhs.uid)
	rhs := s[j].(*gonumVertex)
	rhsUID := string(rhs.uid)

	return lhsUID < rhsUID
}
