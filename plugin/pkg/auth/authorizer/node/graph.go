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

	pvutil "k8s.io/kubernetes/pkg/api/persistentvolume"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
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
	vaVertexType
	serviceAccountVertexType
)

var vertexTypes = map[vertexType]string{
	configMapVertexType:      "configmap",
	nodeVertexType:           "node",
	podVertexType:            "pod",
	pvcVertexType:            "pvc",
	pvVertexType:             "pv",
	secretVertexType:         "secret",
	vaVertexType:             "volumeattachment",
	serviceAccountVertexType: "serviceAccount",
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

// must be called under write lock
// deletes edges from a given vertex type to a specific vertex
// will delete each orphaned "from" vertex, but will never delete the "to" vertex
func (g *Graph) deleteEdges_locked(fromType, toType vertexType, toNamespace, toName string) {
	// get the "to" side
	toVert, exists := g.getVertex_rlocked(toType, toNamespace, toName)
	if !exists {
		return
	}

	// get potential "from" verts that match fromType
	namespaces, exists := g.vertices[fromType]
	if !exists {
		return
	}

	// delete all edges between vertices of fromType and toVert
	removeVerts := []*namedVertex{}
	for _, vertexMapping := range namespaces {
		for _, fromVert := range vertexMapping {
			if g.graph.HasEdgeBetween(fromVert, toVert) {
				// remove the edge (no-op if edge doesn't exist)
				g.graph.RemoveEdge(newDestinationEdge(fromVert, toVert, nil))
				// remember to clean up the fromVert if we orphaned it
				if g.graph.Degree(fromVert) == 0 {
					removeVerts = append(removeVerts, fromVert)
				}
			}
		}
	}

	// clean up orphaned verts
	for _, v := range removeVerts {
		g.graph.RemoveNode(v)
		delete(g.vertices[v.vertexType][v.namespace], v.name)
		if len(g.vertices[v.vertexType][v.namespace]) == 0 {
			delete(g.vertices[v.vertexType], v.namespace)
		}
		if len(g.vertices[v.vertexType]) == 0 {
			delete(g.vertices, v.vertexType)
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
//   svcacct   -> pod
func (g *Graph) AddPod(pod *api.Pod) {
	g.lock.Lock()
	defer g.lock.Unlock()

	g.deleteVertex_locked(podVertexType, pod.Namespace, pod.Name)
	podVertex := g.getOrCreateVertex_locked(podVertexType, pod.Namespace, pod.Name)
	nodeVertex := g.getOrCreateVertex_locked(nodeVertexType, "", pod.Spec.NodeName)
	g.graph.SetEdge(newDestinationEdge(podVertex, nodeVertex, nodeVertex))

	// TODO(mikedanese): If the pod doesn't mount the service account secrets,
	// should the node still get access to the service account?
	//
	// ref https://github.com/kubernetes/kubernetes/issues/58790
	if len(pod.Spec.ServiceAccountName) > 0 {
		g.graph.SetEdge(newDestinationEdge(g.getOrCreateVertex_locked(serviceAccountVertexType, pod.Namespace, pod.Spec.ServiceAccountName), podVertex, nodeVertex))
	}

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
		pvutil.VisitPVSecretNames(pv, func(namespace, secret string, kubeletVisible bool) bool {
			// This grants access to the named secret in the same namespace as the bound PVC
			if kubeletVisible {
				g.graph.SetEdge(simple.Edge{F: g.getOrCreateVertex_locked(secretVertexType, namespace, secret), T: pvVertex})
			}
			return true
		})
	}
}
func (g *Graph) DeletePV(name string) {
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertex_locked(pvVertexType, "", name)
}

// AddVolumeAttachment sets up edges for the following relationships:
//
//   volume attachment -> node
func (g *Graph) AddVolumeAttachment(attachmentName, nodeName string) {
	g.lock.Lock()
	defer g.lock.Unlock()

	// clear existing edges
	g.deleteVertex_locked(vaVertexType, "", attachmentName)

	// if we have a node, establish new edges
	if len(nodeName) > 0 {
		vaVertex := g.getOrCreateVertex_locked(vaVertexType, "", attachmentName)
		nodeVertex := g.getOrCreateVertex_locked(nodeVertexType, "", nodeName)
		g.graph.SetEdge(newDestinationEdge(vaVertex, nodeVertex, nodeVertex))
	}
}
func (g *Graph) DeleteVolumeAttachment(name string) {
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertex_locked(vaVertexType, "", name)
}

// SetNodeConfigMap sets up edges for the Node.Spec.ConfigSource.ConfigMap relationship:
//
// configmap -> node
func (g *Graph) SetNodeConfigMap(nodeName, configMapName, configMapNamespace string) {
	g.lock.Lock()
	defer g.lock.Unlock()

	// TODO(mtaufen): ensure len(nodeName) > 0 in all cases (would sure be nice to have a dependently-typed language here...)

	// clear edges configmaps -> node where the destination is the current node *only*
	// at present, a node can only have one *direct* configmap reference at a time
	g.deleteEdges_locked(configMapVertexType, nodeVertexType, "", nodeName)

	// establish new edges if we have a real ConfigMap to reference
	if len(configMapName) > 0 && len(configMapNamespace) > 0 {
		configmapVertex := g.getOrCreateVertex_locked(configMapVertexType, configMapNamespace, configMapName)
		nodeVertex := g.getOrCreateVertex_locked(nodeVertexType, "", nodeName)
		g.graph.SetEdge(newDestinationEdge(configmapVertex, nodeVertex, nodeVertex))
	}

}
