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
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

type dotVertex struct {
	uid                types.UID
	gvk                schema.GroupVersionKind
	namespace          string
	name               string
	missingFromGraph   bool
	beingDeleted       bool
	deletingDependents bool
	virtual            bool
}

func (v *dotVertex) MarshalDOT(w io.Writer) error {
	attrs := v.Attributes()
	if _, err := fmt.Fprintf(w, "  %q [\n", v.uid); err != nil {
		return err
	}
	for _, a := range attrs {
		if _, err := fmt.Fprintf(w, "    %s=%q\n", a.Key, a.Value); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "  ];\n"); err != nil {
		return err
	}
	return nil
}

func (v *dotVertex) String() string {
	kind := v.gvk.Kind + "." + v.gvk.Version
	if len(v.gvk.Group) > 0 {
		kind = kind + "." + v.gvk.Group
	}
	missing := ""
	if v.missingFromGraph {
		missing = "(missing)"
	}
	deleting := ""
	if v.beingDeleted {
		deleting = "(deleting)"
	}
	deletingDependents := ""
	if v.deletingDependents {
		deleting = "(deletingDependents)"
	}
	virtual := ""
	if v.virtual {
		virtual = "(virtual)"
	}
	return fmt.Sprintf(`%s/%s[%s]-%v%s%s%s%s`, kind, v.name, v.namespace, v.uid, missing, deleting, deletingDependents, virtual)
}

type attribute struct {
	Key   string
	Value string
}

func (v *dotVertex) Attributes() []attribute {
	kubectlString := v.gvk.Kind + "." + v.gvk.Version
	if len(v.gvk.Group) > 0 {
		kubectlString = kubectlString + "." + v.gvk.Group
	}
	kubectlString = kubectlString + "/" + v.name

	label := fmt.Sprintf(`uid=%v
namespace=%v
%v
`,
		v.uid,
		v.namespace,
		kubectlString,
	)

	conditionStrings := []string{}
	if v.beingDeleted {
		conditionStrings = append(conditionStrings, "beingDeleted")
	}
	if v.deletingDependents {
		conditionStrings = append(conditionStrings, "deletingDependents")
	}
	if v.virtual {
		conditionStrings = append(conditionStrings, "virtual")
	}
	if v.missingFromGraph {
		conditionStrings = append(conditionStrings, "missingFromGraph")
	}
	conditionString := strings.Join(conditionStrings, ",")
	if len(conditionString) > 0 {
		label = label + conditionString + "\n"
	}

	return []attribute{
		{Key: "label", Value: label},
		// these place metadata in the correct location, but don't conform to any normal attribute for rendering
		{Key: "group", Value: v.gvk.Group},
		{Key: "version", Value: v.gvk.Version},
		{Key: "kind", Value: v.gvk.Kind},
		{Key: "namespace", Value: v.namespace},
		{Key: "name", Value: v.name},
		{Key: "uid", Value: string(v.uid)},
		{Key: "missing", Value: fmt.Sprintf(`%v`, v.missingFromGraph)},
		{Key: "beingDeleted", Value: fmt.Sprintf(`%v`, v.beingDeleted)},
		{Key: "deletingDependents", Value: fmt.Sprintf(`%v`, v.deletingDependents)},
		{Key: "virtual", Value: fmt.Sprintf(`%v`, v.virtual)},
	}
}

// NewDOTVertex creates a new dotVertex.
func NewDOTVertex(node *node) *dotVertex {
	gv, err := schema.ParseGroupVersion(node.identity.APIVersion)
	if err != nil {
		// this indicates a bad data serialization that should be prevented during storage of the API
		utilruntime.HandleError(err)
	}
	return &dotVertex{
		uid:                node.identity.UID,
		gvk:                gv.WithKind(node.identity.Kind),
		namespace:          node.identity.Namespace,
		name:               node.identity.Name,
		beingDeleted:       node.beingDeleted,
		deletingDependents: node.deletingDependents,
		virtual:            node.virtual,
	}
}

// NewMissingdotVertex creates a new dotVertex.
func NewMissingdotVertex(ownerRef metav1.OwnerReference) *dotVertex {
	gv, err := schema.ParseGroupVersion(ownerRef.APIVersion)
	if err != nil {
		// this indicates a bad data serialization that should be prevented during storage of the API
		utilruntime.HandleError(err)
	}
	return &dotVertex{
		uid:              ownerRef.UID,
		gvk:              gv.WithKind(ownerRef.Kind),
		name:             ownerRef.Name,
		missingFromGraph: true,
	}
}

func (m *concurrentUIDToNode) ToDOTNodesAndEdges() ([]*dotVertex, []dotEdge) {
	m.uidToNodeLock.Lock()
	defer m.uidToNodeLock.Unlock()

	return toDOTNodesAndEdges(m.uidToNode)
}

type dotEdge struct {
	F types.UID
	T types.UID
}

func (e dotEdge) MarshalDOT(w io.Writer) error {
	_, err := fmt.Fprintf(w, "  %q -> %q;\n", e.F, e.T)
	return err
}

func toDOTNodesAndEdges(uidToNode map[types.UID]*node) ([]*dotVertex, []dotEdge) {
	nodes := []*dotVertex{}
	edges := []dotEdge{}

	uidToVertex := map[types.UID]*dotVertex{}

	// add the vertices first, then edges.  That avoids having to deal with missing refs.
	for _, node := range uidToNode {
		// skip adding objects that don't have owner references and aren't referred to.
		if len(node.dependents) == 0 && len(node.owners) == 0 {
			continue
		}
		vertex := NewDOTVertex(node)
		uidToVertex[node.identity.UID] = vertex
		nodes = append(nodes, vertex)
	}
	for _, node := range uidToNode {
		currVertex := uidToVertex[node.identity.UID]
		for _, ownerRef := range node.owners {
			currOwnerVertex, ok := uidToVertex[ownerRef.UID]
			if !ok {
				currOwnerVertex = NewMissingdotVertex(ownerRef)
				uidToVertex[node.identity.UID] = currOwnerVertex
				nodes = append(nodes, currOwnerVertex)
			}
			edges = append(edges, dotEdge{F: currVertex.uid, T: currOwnerVertex.uid})
		}
	}

	sort.SliceStable(nodes, func(i, j int) bool { return nodes[i].uid < nodes[j].uid })
	sort.SliceStable(edges, func(i, j int) bool {
		if edges[i].F != edges[j].F {
			return edges[i].F < edges[j].F
		}
		return edges[i].T < edges[j].T
	})

	return nodes, edges
}

func (m *concurrentUIDToNode) ToDOTNodesAndEdgesForObj(uids ...types.UID) ([]*dotVertex, []dotEdge) {
	m.uidToNodeLock.Lock()
	defer m.uidToNodeLock.Unlock()

	return toDOTNodesAndEdgesForObj(m.uidToNode, uids...)
}

func toDOTNodesAndEdgesForObj(uidToNode map[types.UID]*node, uids ...types.UID) ([]*dotVertex, []dotEdge) {
	uidsToCheck := append([]types.UID{}, uids...)
	interestingNodes := map[types.UID]*node{}

	// build the set of nodes to inspect first, then use the normal construction on the subset
	for i := 0; i < len(uidsToCheck); i++ {
		uid := uidsToCheck[i]
		// if we've already been observed, there was a bug, but skip it so we don't loop forever
		if _, ok := interestingNodes[uid]; ok {
			continue
		}
		node, ok := uidToNode[uid]
		// if there is no node for the UID, skip over it.  We may add it to the list multiple times
		// but we won't loop forever and hopefully the condition doesn't happen very often
		if !ok {
			continue
		}

		interestingNodes[node.identity.UID] = node

		for _, ownerRef := range node.owners {
			// if we've already inspected this UID, don't add it to be inspected again
			if _, ok := interestingNodes[ownerRef.UID]; ok {
				continue
			}
			uidsToCheck = append(uidsToCheck, ownerRef.UID)
		}
		for dependent := range node.dependents {
			// if we've already inspected this UID, don't add it to be inspected again
			if _, ok := interestingNodes[dependent.identity.UID]; ok {
				continue
			}
			uidsToCheck = append(uidsToCheck, dependent.identity.UID)
		}
	}

	return toDOTNodesAndEdges(interestingNodes)
}

// NewDebugHandler creates a new debugHTTPHandler.
func NewDebugHandler(controller *GarbageCollector) http.Handler {
	return &debugHTTPHandler{controller: controller}
}

type debugHTTPHandler struct {
	controller *GarbageCollector
}

func marshalDOT(w io.Writer, nodes []*dotVertex, edges []dotEdge) error {
	if _, err := w.Write([]byte("strict digraph full {\n")); err != nil {
		return err
	}
	if len(nodes) > 0 {
		if _, err := w.Write([]byte("  // Node definitions.\n")); err != nil {
			return err
		}
		for _, node := range nodes {
			if err := node.MarshalDOT(w); err != nil {
				return err
			}
		}
	}
	if len(edges) > 0 {
		if _, err := w.Write([]byte("  // Edge definitions.\n")); err != nil {
			return err
		}
		for _, edge := range edges {
			if err := edge.MarshalDOT(w); err != nil {
				return err
			}
		}
	}
	if _, err := w.Write([]byte("}\n")); err != nil {
		return err
	}
	return nil
}

func (h *debugHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.URL.Path != "/graph" {
		http.Error(w, "", http.StatusNotFound)
		return
	}

	var nodes []*dotVertex
	var edges []dotEdge
	if uidStrings := req.URL.Query()["uid"]; len(uidStrings) > 0 {
		uids := []types.UID{}
		for _, uidString := range uidStrings {
			uids = append(uids, types.UID(uidString))
		}
		nodes, edges = h.controller.dependencyGraphBuilder.uidToNode.ToDOTNodesAndEdgesForObj(uids...)

	} else {
		nodes, edges = h.controller.dependencyGraphBuilder.uidToNode.ToDOTNodesAndEdges()
	}

	b := bytes.NewBuffer(nil)
	if err := marshalDOT(b, nodes, edges); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/vnd.graphviz")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Write(b.Bytes())
	w.WriteHeader(http.StatusOK)
}

func (gc *GarbageCollector) DebuggingHandler() http.Handler {
	return NewDebugHandler(gc)
}
