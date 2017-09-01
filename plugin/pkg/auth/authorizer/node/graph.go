/*
Copyright 2017 The Kubernetes Authors.

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

package node

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	pvutil "k8s.io/kubernetes/pkg/api/persistentvolume"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/third_party/forked/gonum/graph"
	"k8s.io/kubernetes/third_party/forked/gonum/graph/simple"
)

// namedVertex implements graph.Node and remembers the type, namespace, and name of its related API object
type namedVertex struct {
	name       string
	namespace  string
	id         int
	vertexType vertexType
}

func newNamedVertex(vertexType vertexType, namespace, name string, id int) *namedVertex {
	return &namedVertex{
		vertexType: vertexType,
		name:       name,
		namespace:  namespace,
		id:         id,
	}
}
func (n *namedVertex) ID() int {
	return n.id
}
func (n *namedVertex) String() string {
	if len(n.namespace) == 0 {
		return vertexTypes[n.vertexType] + ":" + n.name
	}
	return vertexTypes[n.vertexType] + ":" + n.namespace + "/" + n.name
}

// destinationEdge is a graph edge that includes a denormalized reference to the final destination vertex.
// This should only be used when there is a single leaf vertex reachable from T.
type destinationEdge struct {
	F           graph.Node
	T           graph.Node
	Destination graph.Node
}

func newDestinationEdge(from, to, destination graph.Node) graph.Edge {
	return &destinationEdge{F: from, T: to, Destination: destination}
}
func (e *destinationEdge) From() graph.Node   { return e.F }
func (e *destinationEdge) To() graph.Node     { return e.T }
func (e *destinationEdge) Weight() float64    { return 0 }
func (e *destinationEdge) DestinationID() int { return e.Destination.ID() }

// Graph holds graph vertices and a way to look up a vertex for a particular API type/namespace/name.
// All edges point toward the vertices representing Kubernetes nodes:
//
// node <- pod
// pod  <- secret,configmap,pvc
// pvc  <- pv
// pv   <- secret
type Graph struct {
	lock  sync.RWMutex
	graph *simple.DirectedAcyclicGraph
	// vertices is a map of type -> namespace -> name -> vertex
	vertices map[vertexType]namespaceVertexMapping
}

// namespaceVertexMapping is a map of namespace -> name -> vertex
type namespaceVertexMapping map[string]nameVertexMapping

// nameVertexMapping is a map of name -> vertex
type nameVertexMapping map[string]*namedVertex

func NewGraph() *Graph {
	return &Graph{
		vertices: map[vertexType]namespaceVertexMapping{},
		graph:    simple.NewDirectedAcyclicGraph(0, 0),
	}
}

// vertexType indicates the type of the API object the vertex represents.
// represented as a byte to minimize space used in the vertices.
type vertexType byte

const (
	configMapVertexType vertexType = iota
	nodeVertexType
	podVertexType
	pvcVertexType
	pvVertexType
	secretVertexType
)

var vertexTypes = map[vertexType]string{
	configMapVertexType: "configmap",
	nodeVertexType:      "node",
	podVertexType:       "pod",
	pvcVertexType:       "pvc",
	pvVertexType:        "pv",
	secretVertexType:    "secret",
}

// must be called under a write lock
func (g *Graph) getOrCreateVertex_locked(vertexType vertexType, namespace, name string) *namedVertex {
	if vertex, exists := g.getVertex_rlocked(vertexType, namespace, name); exists {
		return vertex
	}
	return g.createVertex_locked(vertexType, namespace, name)
}

// must be called under a read lock
func (g *Graph) getVertex_rlocked(vertexType vertexType, namespace, name string) (*namedVertex, bool) {
	vertex, exists := g.vertices[vertexType][namespace][name]
	return vertex, exists
}

// must be called under a write lock
func (g *Graph) createVertex_locked(vertexType vertexType, namespace, name string) *namedVertex {
	typedVertices, exists := g.vertices[vertexType]
	if !exists {
		typedVertices = namespaceVertexMapping{}
		g.vertices[vertexType] = typedVertices
	}

	namespacedVertices, exists := typedVertices[namespace]
	if !exists {
		namespacedVertices = map[string]*namedVertex{}
		typedVertices[namespace] = namespacedVertices
	}

	vertex := newNamedVertex(vertexType, namespace, name, g.graph.NewNodeID())
	namespacedVertices[name] = vertex
	g.graph.AddNode(vertex)

	return vertex
}

// must be called under write lock
func (g *Graph) deleteVertex_locked(vertexType vertexType, namespace, name string) {
	vertex, exists := g.getVertex_rlocked(vertexType, namespace, name)
	if !exists {
		return
	}

	// find existing neighbors with a single edge (meaning we are their only neighbor)
	neighborsToRemove := []graph.Node{}
	g.graph.VisitFrom(vertex, func(neighbor graph.Node) bool {
		// this downstream neighbor has only one edge (which must be from us), so remove them as well
		if g.graph.Degree(neighbor) == 1 {
			neighborsToRemove = append(neighborsToRemove, neighbor)
		}
		return true
	})
	g.graph.VisitTo(vertex, func(neighbor graph.Node) bool {
		// this upstream neighbor has only one edge (which must be to us), so remove them as well
		if g.graph.Degree(neighbor) == 1 {
			neighborsToRemove = append(neighborsToRemove, neighbor)
		}
		return true
	})

	// remove the vertex
	g.graph.RemoveNode(vertex)
	delete(g.vertices[vertexType][namespace], name)
	if len(g.vertices[vertexType][namespace]) == 0 {
		delete(g.vertices[vertexType], namespace)
	}

	// remove neighbors that are now edgeless
	for _, neighbor := range neighborsToRemove {
		g.graph.RemoveNode(neighbor)
		n := neighbor.(*namedVertex)
		delete(g.vertices[n.vertexType][n.namespace], n.name)
		if len(g.vertices[n.vertexType][n.namespace]) == 0 {
			delete(g.vertices[n.vertexType], n.namespace)
		}
	}
}

// AddPod should only be called once spec.NodeName is populated.
// It sets up edges for the following relationships (which are immutable for a pod once bound to a node):
//
//   pod -> node
//
//   secret    -> pod
//   configmap -> pod
//   pvc       -> pod
func (g *Graph) AddPod(pod *api.Pod) {
	g.lock.Lock()
	defer g.lock.Unlock()

	g.deleteVertex_locked(podVertexType, pod.Namespace, pod.Name)
	podVertex := g.getOrCreateVertex_locked(podVertexType, pod.Namespace, pod.Name)
	nodeVertex := g.getOrCreateVertex_locked(nodeVertexType, "", pod.Spec.NodeName)
	g.graph.SetEdge(newDestinationEdge(podVertex, nodeVertex, nodeVertex))

	podutil.VisitPodSecretNames(pod, func(secret string) bool {
		g.graph.SetEdge(newDestinationEdge(g.getOrCreateVertex_locked(secretVertexType, pod.Namespace, secret), podVertex, nodeVertex))
		return true
	})

	podutil.VisitPodConfigmapNames(pod, func(configmap string) bool {
		g.graph.SetEdge(newDestinationEdge(g.getOrCreateVertex_locked(configMapVertexType, pod.Namespace, configmap), podVertex, nodeVertex))
		return true
	})

	for _, v := range pod.Spec.Volumes {
		if v.PersistentVolumeClaim != nil {
			g.graph.SetEdge(newDestinationEdge(g.getOrCreateVertex_locked(pvcVertexType, pod.Namespace, v.PersistentVolumeClaim.ClaimName), podVertex, nodeVertex))
		}
	}
}
func (g *Graph) DeletePod(name, namespace string) {
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertex_locked(podVertexType, namespace, name)
}

// AddPV sets up edges for the following relationships:
//
//   secret -> pv
//
//   pv -> pvc
func (g *Graph) AddPV(pv *api.PersistentVolume) {
	g.lock.Lock()
	defer g.lock.Unlock()

	// clear existing edges
	g.deleteVertex_locked(pvVertexType, "", pv.Name)

	// if we have a pvc, establish new edges
	if pv.Spec.ClaimRef != nil {
		pvVertex := g.getOrCreateVertex_locked(pvVertexType, "", pv.Name)

		// since we don't know the other end of the pvc -> pod -> node chain (or it may not even exist yet), we can't decorate these edges with kubernetes node info
		g.graph.SetEdge(simple.Edge{F: pvVertex, T: g.getOrCreateVertex_locked(pvcVertexType, pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name)})
		pvutil.VisitPVSecretNames(pv, func(namespace, secret string) bool {
			// This grants access to the named secret in the same namespace as the bound PVC
			g.graph.SetEdge(simple.Edge{F: g.getOrCreateVertex_locked(secretVertexType, namespace, secret), T: pvVertex})
			return true
		})
	}
}
func (g *Graph) DeletePV(name string) {
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertex_locked(pvVertexType, "", name)
}
