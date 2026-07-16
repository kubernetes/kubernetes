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
	"fmt"
	"sync"
	"time"

	certsv1beta1 "k8s.io/api/certificates/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/storage/ephemeral"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	pvutil "k8s.io/kubernetes/pkg/api/v1/persistentvolume"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
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
	lock sync.RWMutex
	// Calling graph.SetEdge is restricted to the addEdgeLocked method of Graph.
	graph *simple.DirectedAcyclicGraph
	// vertices is a map of type -> namespace -> name -> vertex
	vertices map[vertexType]namespaceVertexMapping

	// destinationEdgeIndex is a map of vertex -> set of destination IDs
	destinationEdgeIndex map[int]*intSet
	// destinationEdgeThreshold is the minimum number of distinct destination IDs at which to maintain an index
	destinationEdgeThreshold int
}

// namespaceVertexMapping is a map of namespace -> name -> vertex
type namespaceVertexMapping map[string]nameVertexMapping

// nameVertexMapping is a map of name -> vertex
type nameVertexMapping map[string]*namedVertex

func NewGraph() *Graph {
	return &Graph{
		vertices: map[vertexType]namespaceVertexMapping{},
		graph:    simple.NewDirectedAcyclicGraph(0, 0),

		destinationEdgeIndex: map[int]*intSet{},
		// experimentally determined to be the point at which iteration adds an order of magnitude to the authz check.
		// since maintaining indexes costs time/memory while processing graph changes, we don't want to make this too low.
		destinationEdgeThreshold: 200,
	}
}

// vertexType indicates the type of the API object the vertex represents.
// represented as a byte to minimize space used in the vertices.
type vertexType byte

const (
	configMapVertexType vertexType = iota
	sliceVertexType
	nodeVertexType
	podVertexType
	pvcVertexType
	pvVertexType
	resourceClaimVertexType
	secretVertexType
	vaVertexType
	serviceAccountVertexType
	pcrVertexType
)

var vertexTypes = map[vertexType]string{
	configMapVertexType:      "configmap",
	sliceVertexType:          "resourceslice",
	nodeVertexType:           "node",
	podVertexType:            "pod",
	pvcVertexType:            "pvc",
	pvVertexType:             "pv",
	resourceClaimVertexType:  "resourceclaim",
	secretVertexType:         "secret",
	vaVertexType:             "volumeattachment",
	serviceAccountVertexType: "serviceAccount",
	pcrVertexType:            "podcertificaterequest",
}

// vertexTypeWithAuthoritativeIndex indicates which types of vertices can hold
// a destination edge index that is authoritative i.e. if the index exists,
// then it always stores all of the Nodes that are reachable from that vertex
// in the graph.
var vertexTypeWithAuthoritativeIndex = map[vertexType]bool{
	configMapVertexType:      true,
	sliceVertexType:          true,
	podVertexType:            true,
	pvcVertexType:            true,
	resourceClaimVertexType:  true,
	vaVertexType:             true,
	serviceAccountVertexType: true,
	pcrVertexType:            true,
}

// must be called under a write lock
func (g *Graph) getOrCreateVertexLocked(vertexType vertexType, namespace, name string) *namedVertex {
	if vertex, exists := g.getVertexRLocked(vertexType, namespace, name); exists {
		return vertex
	}
	return g.createVertexLocked(vertexType, namespace, name)
}

// must be called under a read lock
func (g *Graph) getVertexRLocked(vertexType vertexType, namespace, name string) (*namedVertex, bool) {
	vertex, exists := g.vertices[vertexType][namespace][name]
	return vertex, exists
}

// must be called under a write lock
func (g *Graph) createVertexLocked(vertexType vertexType, namespace, name string) *namedVertex {
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
func (g *Graph) deleteVertexLocked(vertexType vertexType, namespace, name string) {
	vertex, exists := g.getVertexRLocked(vertexType, namespace, name)
	if !exists {
		return
	}

	// find existing neighbors with a single edge (meaning we are their only neighbor)
	neighborsToRemove := []graph.Node{}
	edgesToRemoveFromIndexes := []graph.Edge{}
	g.graph.VisitFrom(vertex, func(neighbor graph.Node) bool {
		// this downstream neighbor has only one edge (which must be from us), so remove them as well
		if g.graph.Degree(neighbor) == 1 {
			neighborsToRemove = append(neighborsToRemove, neighbor)
		}
		return true
	})
	g.graph.VisitTo(vertex, func(neighbor graph.Node) bool {
		if g.graph.Degree(neighbor) == 1 {
			// this upstream neighbor has only one edge (which must be to us), so remove them as well
			neighborsToRemove = append(neighborsToRemove, neighbor)
		} else {
			// decrement the destination edge index on this neighbor if the edge between us was a destination edge
			edgesToRemoveFromIndexes = append(edgesToRemoveFromIndexes, g.graph.EdgeBetween(vertex, neighbor))
		}
		return true
	})

	// remove the vertex
	g.removeVertexLocked(vertex)

	// remove neighbors that are now edgeless
	for _, neighbor := range neighborsToRemove {
		g.removeVertexLocked(neighbor.(*namedVertex))
	}

	// remove edges from destination indexes for neighbors that dropped outbound edges
	for _, edge := range edgesToRemoveFromIndexes {
		g.removeEdgeFromDestinationIndexLocked(edge)
	}
}

// must be called under write lock
// deletes edges from a given vertex type to a specific vertex
// will delete each orphaned "from" vertex, but will never delete the "to" vertex
func (g *Graph) deleteEdgesLocked(fromType, toType vertexType, toNamespace, toName string) {
	// get the "to" side
	toVert, exists := g.getVertexRLocked(toType, toNamespace, toName)
	if !exists {
		return
	}

	// delete all edges between vertices of fromType and toVert
	neighborsToRemove := []*namedVertex{}
	edgesToRemove := []graph.Edge{}
	g.graph.VisitTo(toVert, func(from graph.Node) bool {
		fromVert := from.(*namedVertex)
		if fromVert.vertexType != fromType {
			return true
		}
		// this neighbor has only one edge (which must be to us), so remove them as well
		if g.graph.Degree(fromVert) == 1 {
			neighborsToRemove = append(neighborsToRemove, fromVert)
		} else {
			edgesToRemove = append(edgesToRemove, g.graph.EdgeBetween(from, toVert))
		}
		return true
	})

	// clean up orphaned verts
	for _, v := range neighborsToRemove {
		g.removeVertexLocked(v)
	}

	// remove edges and decrement destination indexes for neighbors that dropped outbound edges
	for _, edge := range edgesToRemove {
		g.graph.RemoveEdge(edge)
		g.removeEdgeFromDestinationIndexLocked(edge)
	}
}

// A fastpath for recomputeDestinationIndexLocked for "removing edge" case.
func (g *Graph) removeEdgeFromDestinationIndexLocked(e graph.Edge) {
	n := e.From()
	// don't maintain indices for nodes with few edges
	edgeCount := g.graph.Degree(n)
	if edgeCount < g.destinationEdgeThreshold {
		delete(g.destinationEdgeIndex, n.ID())
		return
	}

	// decrement the nodeID->destinationID refcount in the index, if the index exists
	index := g.destinationEdgeIndex[n.ID()]
	if index == nil {
		return
	}
	if destinationEdge, ok := e.(*destinationEdge); ok {
		index.decrement(destinationEdge.DestinationID())
	}
}

// A fastpath for recomputeDestinationIndexLocked for "adding edge case".
func (g *Graph) addEdgeToDestinationIndexLocked(e graph.Edge) {
	n := e.From()
	index := g.destinationEdgeIndex[n.ID()]
	if index == nil {
		// There is no index, use the full index computation method
		g.recomputeDestinationIndexLocked(n)
		return
	}
	// fast-add the new edge to an existing index
	if destinationEdge, ok := e.(*destinationEdge); ok {
		index.increment(destinationEdge.DestinationID())
	}
}

// must be called under write lock
// removeVertexLocked removes the specified vertex from the graph and from the maintained indices.
// It does nothing to indexes of neighbor vertices.
func (g *Graph) removeVertexLocked(v *namedVertex) {
	g.graph.RemoveNode(v)
	delete(g.destinationEdgeIndex, v.ID())
	delete(g.vertices[v.vertexType][v.namespace], v.name)
	if len(g.vertices[v.vertexType][v.namespace]) == 0 {
		delete(g.vertices[v.vertexType], v.namespace)
	}
}

// must be called under write lock
// recomputeDestinationIndexLocked recomputes the index of destination ids for the specified vertex
func (g *Graph) recomputeDestinationIndexLocked(n graph.Node) {
	// don't maintain indices for nodes with few edges
	edgeCount := g.graph.Degree(n)
	if edgeCount < g.destinationEdgeThreshold {
		delete(g.destinationEdgeIndex, n.ID())
		return
	}

	// get or create the index
	index := g.destinationEdgeIndex[n.ID()]
	if index == nil {
		index = newIntSet()
	} else {
		index.reset()
	}

	// populate the index
	g.graph.VisitFrom(n, func(dest graph.Node) bool {
		if destinationEdge, ok := g.graph.EdgeBetween(n, dest).(*destinationEdge); ok {
			index.increment(destinationEdge.DestinationID())
		}
		return true
	})
	g.destinationEdgeIndex[n.ID()] = index
}

// AddPod should only be called once spec.NodeName is populated.
// It sets up edges for the following relationships (which are immutable for a pod once bound to a node):
//
//	pod       -> node
//	secret    -> pod
//	configmap -> pod
//	pvc       -> pod
//	svcacct   -> pod
func (g *Graph) AddPod(pod *corev1.Pod) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("AddPod").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()

	g.deleteVertexLocked(podVertexType, pod.Namespace, pod.Name)
	podVertex := g.getOrCreateVertexLocked(podVertexType, pod.Namespace, pod.Name)
	nodeVertex := g.getOrCreateVertexLocked(nodeVertexType, "", pod.Spec.NodeName)
	// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
	g.addEdgeLocked(podVertex, nodeVertex, nodeVertex)

	// Short-circuit adding edges to other resources for mirror pods.
	// A node must never be able to create a pod that grants them permissions on other API objects.
	// The NodeRestriction admission plugin prevents creation of such pods, but short-circuiting here gives us defense in depth.
	if _, isMirrorPod := pod.Annotations[corev1.MirrorPodAnnotationKey]; isMirrorPod {
		return
	}

	// The pod unconditionally gets access to the pod's service account.  In the
	// future, this access could be restricted based on whether or not the pod
	// actually mounts a service account token, or has a podcertificate volume.
	if len(pod.Spec.ServiceAccountName) > 0 {
		serviceAccountVertex := g.getOrCreateVertexLocked(serviceAccountVertexType, pod.Namespace, pod.Spec.ServiceAccountName)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(serviceAccountVertex, podVertex, nodeVertex)
	}

	podutil.VisitPodSecretNames(pod, func(secret string) bool {
		secretVertex := g.getOrCreateVertexLocked(secretVertexType, pod.Namespace, secret)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(secretVertex, podVertex, nodeVertex)
		return true
	})

	podutil.VisitPodConfigmapNames(pod, func(configmap string) bool {
		configmapVertex := g.getOrCreateVertexLocked(configMapVertexType, pod.Namespace, configmap)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(configmapVertex, podVertex, nodeVertex)
		return true
	})

	for _, v := range pod.Spec.Volumes {
		claimName := ""
		if v.PersistentVolumeClaim != nil {
			claimName = v.PersistentVolumeClaim.ClaimName
		} else if v.Ephemeral != nil {
			claimName = ephemeral.VolumeClaimName(pod, &v)
		}
		if claimName != "" {
			pvcVertex := g.getOrCreateVertexLocked(pvcVertexType, pod.Namespace, claimName)
			// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
			g.addEdgeLocked(pvcVertex, podVertex, nodeVertex)
		}
	}

	for _, podResourceClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &podResourceClaim)
		// Do we have a valid claim name? If yes, add an edge that grants
		// kubelet access to that claim. An error indicates that a claim
		// still needs to be created, nil that intentionally no claim
		// was created and never will be because it isn't needed.
		if err == nil && claimName != nil {
			claimVertex := g.getOrCreateVertexLocked(resourceClaimVertexType, pod.Namespace, *claimName)
			// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
			g.addEdgeLocked(claimVertex, podVertex, nodeVertex)
		}
	}

	if pod.Status.ExtendedResourceClaimStatus != nil && len(pod.Status.ExtendedResourceClaimStatus.ResourceClaimName) > 0 {
		claimVertex := g.getOrCreateVertexLocked(resourceClaimVertexType, pod.Namespace, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(claimVertex, podVertex, nodeVertex)
	}
}

// Must be called under a write lock.
// All edge adds must be handled by that method rather than by calling
// g.graph.SetEdge directly.
// Note: if "from" belongs to vertexTypeWithAuthoritativeIndex, then
// "destination" must be non-nil.
func (g *Graph) addEdgeLocked(from, to, destination *namedVertex) {
	if destination != nil {
		e := newDestinationEdge(from, to, destination)
		g.graph.SetEdge(e)
		g.addEdgeToDestinationIndexLocked(e)
		return
	}

	// We must not create edges without a Node label from a vertex that is
	// supposed to hold authoritative destination edge index only.
	// Entering this branch would mean there's a bug in the Node authorizer.
	if vertexTypeWithAuthoritativeIndex[from.vertexType] {
		panic(fmt.Sprintf("vertex of type %q must have destination edges only", vertexTypes[from.vertexType]))
	}
	g.graph.SetEdge(simple.Edge{F: from, T: to})
}

func (g *Graph) DeletePod(name, namespace string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("DeletePod").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertexLocked(podVertexType, namespace, name)
}

// AddPodCertificateRequest adds a PodCertificateRequest to the graph.
//
// PCRs technically have two valid edges:
//
// * PCR -> Pod (-> Node)
//
// * PCR -> Node
//
// We only add the direct PCR -> Node edge, since that is enough to perform the
// authorization, and it's a shorter graph traversal.  The noderestriction
// admission plugin ensures that all PCRs created have a valid node,
// serviceaccount, and pod combination that actually exists in the cluster.
func (g *Graph) AddPodCertificateRequest(pcr *certsv1beta1.PodCertificateRequest) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("AddPodCertificateRequest").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()

	g.deleteVertexLocked(pcrVertexType, pcr.ObjectMeta.Namespace, pcr.ObjectMeta.Name)
	pcrVertex := g.getOrCreateVertexLocked(pcrVertexType, pcr.ObjectMeta.Namespace, pcr.ObjectMeta.Name)
	nodeVertex := g.getOrCreateVertexLocked(nodeVertexType, "", string(pcr.Spec.NodeName))
	g.addEdgeLocked(pcrVertex, nodeVertex, nodeVertex)
}

// DeletePodCertificateRequest removes it from the graph.
func (g *Graph) DeletePodCertificateRequest(pcr *certsv1beta1.PodCertificateRequest) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("DeletePodCertificateRequest").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertexLocked(pcrVertexType, pcr.ObjectMeta.Namespace, pcr.ObjectMeta.Name)
}

// AddPV sets up edges for the following relationships:
//
//	secret -> pv
//
//	pv -> pvc
func (g *Graph) AddPV(pv *corev1.PersistentVolume) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("AddPV").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()

	// clear existing edges
	g.deleteVertexLocked(pvVertexType, "", pv.Name)

	// if we have a pvc, establish new edges
	if pv.Spec.ClaimRef != nil {
		pvVertex := g.getOrCreateVertexLocked(pvVertexType, "", pv.Name)

		// since we don't know the other end of the pvc -> pod -> node chain (or it may not even exist yet), we can't decorate these edges with kubernetes node info
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(pvVertex, g.getOrCreateVertexLocked(pvcVertexType, pv.Spec.ClaimRef.Namespace, pv.Spec.ClaimRef.Name), nil)
		pvutil.VisitPVSecretNames(pv, func(namespace, secret string, kubeletVisible bool) bool {
			// This grants access to the named secret in the same namespace as the bound PVC
			if kubeletVisible {
				// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
				g.addEdgeLocked(g.getOrCreateVertexLocked(secretVertexType, namespace, secret), pvVertex, nil)
			}
			return true
		})
	}
}
func (g *Graph) DeletePV(name string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("DeletePV").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertexLocked(pvVertexType, "", name)
}

// AddVolumeAttachment sets up edges for the following relationships:
//
//	volume attachment -> node
func (g *Graph) AddVolumeAttachment(attachmentName, nodeName string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("AddVolumeAttachment").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()

	// clear existing edges
	g.deleteVertexLocked(vaVertexType, "", attachmentName)

	// if we have a node, establish new edges
	if len(nodeName) > 0 {
		vaVertex := g.getOrCreateVertexLocked(vaVertexType, "", attachmentName)
		nodeVertex := g.getOrCreateVertexLocked(nodeVertexType, "", nodeName)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(vaVertex, nodeVertex, nodeVertex)
	}
}
func (g *Graph) DeleteVolumeAttachment(name string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("DeleteVolumeAttachment").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertexLocked(vaVertexType, "", name)
}

// AddResourceSlice sets up edges for the following relationships:
//
//	node resource slice -> node
func (g *Graph) AddResourceSlice(sliceName, nodeName string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("AddResourceSlice").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()

	// clear existing edges
	g.deleteVertexLocked(sliceVertexType, "", sliceName)

	// if we have a node, establish new edges
	if len(nodeName) > 0 {
		sliceVertex := g.getOrCreateVertexLocked(sliceVertexType, "", sliceName)
		nodeVertex := g.getOrCreateVertexLocked(nodeVertexType, "", nodeName)
		// Edge adds must be handled by addEdgeLocked instead of direct g.graph.SetEdge calls.
		g.addEdgeLocked(sliceVertex, nodeVertex, nodeVertex)
	}
}
func (g *Graph) DeleteResourceSlice(sliceName string) {
	start := time.Now()
	defer func() {
		graphActionsDuration.WithLabelValues("DeleteResourceSlice").Observe(time.Since(start).Seconds())
	}()
	g.lock.Lock()
	defer g.lock.Unlock()
	g.deleteVertexLocked(sliceVertexType, "", sliceName)
}
