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
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/dump"
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

func TestToDOTGraph(t *testing.T) {
	tests := []struct {
		name        string
		uidToNode   map[types.UID]*node
		expectNodes []*dotVertex
		expectEdges []dotEdge
	}{
		{
			name: "simple",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
			},
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
			},
		},
		{
			name: "missing", // synthetic vertex created
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("charlie"): charlieNode(),
			},
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
			},
		},
		{
			name: "drop-no-ref",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("echo"):    echoNode(),
			},
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
			},
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
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
				NewDOTVertex(deltaNode()),
				NewDOTVertex(foxtrotNode()),
				NewDOTVertex(golfNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
				{F: types.UID("delta"), T: types.UID("foxtrot")},
				{F: types.UID("foxtrot"), T: types.UID("golf")},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualNodes, actualEdges := toDOTNodesAndEdges(test.uidToNode)
			compareGraphs(test.expectNodes, actualNodes, test.expectEdges, actualEdges, t)
		})
	}
}

func TestToDOTGraphObj(t *testing.T) {
	tests := []struct {
		name        string
		uidToNode   map[types.UID]*node
		uids        []types.UID
		expectNodes []*dotVertex
		expectEdges []dotEdge
	}{
		{
			name: "simple",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
			},
			uids: []types.UID{types.UID("bravo")},
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
			},
		},
		{
			name: "missing", // synthetic vertex created
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("charlie"): charlieNode(),
			},
			uids:        []types.UID{types.UID("bravo")},
			expectNodes: []*dotVertex{},
			expectEdges: []dotEdge{},
		},
		{
			name: "drop-no-ref",
			uidToNode: map[types.UID]*node{
				types.UID("alpha"):   alphaNode(),
				types.UID("bravo"):   bravoNode(),
				types.UID("charlie"): charlieNode(),
				types.UID("echo"):    echoNode(),
			},
			uids:        []types.UID{types.UID("echo")},
			expectNodes: []*dotVertex{},
			expectEdges: []dotEdge{},
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
			expectNodes: []*dotVertex{
				NewDOTVertex(deltaNode()),
				NewDOTVertex(foxtrotNode()),
				NewDOTVertex(golfNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("delta"), T: types.UID("foxtrot")},
				{F: types.UID("foxtrot"), T: types.UID("golf")},
			},
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
			expectNodes: []*dotVertex{
				NewDOTVertex(deltaNode()),
				NewDOTVertex(foxtrotNode()),
				NewDOTVertex(golfNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("delta"), T: types.UID("foxtrot")},
				{F: types.UID("foxtrot"), T: types.UID("golf")},
			},
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
			expectNodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
				NewDOTVertex(deltaNode()),
				NewDOTVertex(foxtrotNode()),
				NewDOTVertex(golfNode()),
			},
			expectEdges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
				{F: types.UID("delta"), T: types.UID("foxtrot")},
				{F: types.UID("foxtrot"), T: types.UID("golf")},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualNodes, actualEdges := toDOTNodesAndEdgesForObj(test.uidToNode, test.uids...)
			compareGraphs(test.expectNodes, actualNodes, test.expectEdges, actualEdges, t)
		})
	}
}

func compareGraphs(expectedNodes, actualNodes []*dotVertex, expectedEdges, actualEdges []dotEdge, t *testing.T) {
	if len(expectedNodes) != len(actualNodes) {
		t.Fatal(dump.Pretty(actualNodes))
	}
	for i := range expectedNodes {
		currExpected := expectedNodes[i]
		currActual := actualNodes[i]
		if currExpected.uid != currActual.uid {
			t.Errorf("expected %v, got %v", dump.Pretty(currExpected), dump.Pretty(currActual))
		}
	}
	if len(expectedEdges) != len(actualEdges) {
		t.Fatal(dump.Pretty(actualEdges))
	}
	for i := range expectedEdges {
		currExpected := expectedEdges[i]
		currActual := actualEdges[i]
		if currExpected != currActual {
			t.Errorf("expected %v, got %v", dump.Pretty(currExpected), dump.Pretty(currActual))
		}
	}
}

func TestMarshalDOT(t *testing.T) {
	ref1 := objectReference{
		OwnerReference: metav1.OwnerReference{
			UID:        types.UID("ref1-[]\"\\Iñtërnâtiônàlizætiøn,🐹"),
			Name:       "ref1name-Iñtërnâtiônàlizætiøn,🐹",
			Kind:       "ref1kind-Iñtërnâtiônàlizætiøn,🐹",
			APIVersion: "ref1group/version",
		},
		Namespace: "ref1ns",
	}
	ref2 := objectReference{
		OwnerReference: metav1.OwnerReference{
			UID:        types.UID("ref2-"),
			Name:       "ref2name-",
			Kind:       "ref2kind-",
			APIVersion: "ref2group/version",
		},
		Namespace: "ref2ns",
	}
	testcases := []struct {
		file  string
		nodes []*dotVertex
		edges []dotEdge
	}{
		{
			file: "empty.dot",
		},
		{
			file: "simple.dot",
			nodes: []*dotVertex{
				NewDOTVertex(alphaNode()),
				NewDOTVertex(bravoNode()),
				NewDOTVertex(charlieNode()),
				NewDOTVertex(deltaNode()),
				NewDOTVertex(foxtrotNode()),
				NewDOTVertex(golfNode()),
			},
			edges: []dotEdge{
				{F: types.UID("alpha"), T: types.UID("bravo")},
				{F: types.UID("alpha"), T: types.UID("charlie")},
				{F: types.UID("delta"), T: types.UID("foxtrot")},
				{F: types.UID("foxtrot"), T: types.UID("golf")},
			},
		},
		{
			file: "escaping.dot",
			nodes: []*dotVertex{
				NewDOTVertex(makeNode(ref1, withOwners(ref2))),
				NewDOTVertex(makeNode(ref2)),
			},
			edges: []dotEdge{
				{F: types.UID(ref1.UID), T: types.UID(ref2.UID)},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.file, func(t *testing.T) {
			goldenData, err := os.ReadFile(filepath.Join("testdata", tc.file))
			if err != nil {
				t.Fatal(err)
			}
			b := bytes.NewBuffer(nil)
			if err := marshalDOT(b, tc.nodes, tc.edges); err != nil {
				t.Fatal(err)
			}

			if e, a := string(goldenData), string(b.Bytes()); cmp.Diff(e, a) != "" {
				t.Logf("got\n%s", string(a))
				t.Fatalf("unexpected diff:\n%s", cmp.Diff(e, a))
			}
		})
	}
}
