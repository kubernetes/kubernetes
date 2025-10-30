package store

import (
	"fmt"
	"sync"
	"time"

	"github.com/Ayobami-00/k8s-lite-go/pkg/api"
)

// InMemoryStore is an in-memory implementation of the Store interface.
// It is primarily for testing and simplicity, not for production use.
type InMemoryStore struct {
	mu    sync.RWMutex
	pods  map[string]*api.Pod  // Key: "namespace/name"
	nodes map[string]*api.Node // Key: "name"
}

// NewInMemoryStore creates a new InMemoryStore.
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		pods:  make(map[string]*api.Pod),
		nodes: make(map[string]*api.Node),
	}
}

func podKey(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}

// CreatePod adds a new pod to the store.
func (s *InMemoryStore) CreatePod(pod *api.Pod) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := podKey(pod.Namespace, pod.Name)
	if _, exists := s.pods[key]; exists {
		return fmt.Errorf("pod %s in namespace %s already exists", pod.Name, pod.Namespace)
	}
	s.pods[key] = pod
	return nil
}

// GetPod retrieves a pod from the store.
func (s *InMemoryStore) GetPod(namespace, name string) (*api.Pod, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	key := podKey(namespace, name)
	pod, exists := s.pods[key]
	if !exists {
		return nil, fmt.Errorf("pod %s in namespace %s not found", name, namespace)
	}
	return pod, nil
}

// UpdatePod updates an existing pod in the store.
// It prevents updates to NodeName or Phase if the pod is already marked for deletion,
// but allows Kubelet to update phase to Succeeded/Failed.
func (s *InMemoryStore) UpdatePod(pod *api.Pod) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := podKey(pod.Namespace, pod.Name)
	existingPod, exists := s.pods[key]
	if !exists {
		return fmt.Errorf("pod %s in namespace %s not found for update", pod.Name, pod.Namespace)
	}

	if existingPod.DeletionTimestamp != nil {
		// Pod is already marked for deletion in the store.

		// Ensure the incoming update acknowledges the existing DeletionTimestamp.
		// This prevents a stale update from before deletion was initiated from overwriting it.
		if pod.DeletionTimestamp == nil || !pod.DeletionTimestamp.Equal(*existingPod.DeletionTimestamp) {
			return fmt.Errorf("cannot update pod %s in namespace %s: incoming update does not have matching DeletionTimestamp for an already terminating pod", pod.Name, pod.Namespace)
		}

		// Allow updates to phase to Succeeded or Failed, or if phase is still Terminating (e.g. Kubelet updating other statuses).
		// Also, ensure NodeName does not change during termination.
		if pod.Phase == api.PodSucceeded || pod.Phase == api.PodFailed || pod.Phase == api.PodTerminating || pod.Phase == api.PodDeleted { // ADD PodDeleted HERE
			if pod.NodeName != existingPod.NodeName {
				return fmt.Errorf("cannot change NodeName of pod %s in namespace %s as it is terminating", pod.Name, pod.Namespace)
			}
			s.pods[key] = pod
			return nil
		}

		// If it's terminating and the update tries to set it to something other than Succeeded, Failed, or Terminating
		return fmt.Errorf("cannot update pod %s in namespace %s to phase %s as it is terminating; only Succeeded, Failed, or Terminating are allowed", pod.Name, pod.Namespace, pod.Phase)
	}

	// If the existing pod is NOT terminating, but the update tries to set a DeletionTimestamp,
	// guide to use DeletePod.
	if pod.DeletionTimestamp != nil && existingPod.DeletionTimestamp == nil {
		return fmt.Errorf("to mark pod %s in namespace %s for deletion, use DeletePod method", pod.Name, pod.Namespace)
	}

	// Standard update for non-terminating pods
	s.pods[key] = pod
	return nil
}

// DeletePod marks a pod for deletion by setting its DeletionTimestamp and Phase.
// It does not immediately remove the pod from the store.
func (s *InMemoryStore) DeletePod(namespace, name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	key := podKey(namespace, name)
	pod, exists := s.pods[key]
	if !exists {
		return fmt.Errorf("pod %s in namespace %s not found for deletion", name, namespace)
	}

	if pod.DeletionTimestamp != nil {
		// Already marked for deletion, could return a specific error or just succeed
		return fmt.Errorf("pod %s in namespace %s is already being deleted", name, namespace)
	}

	now := time.Now()
	pod.DeletionTimestamp = &now
	pod.Phase = api.PodTerminating // Set phase to Terminating
	s.pods[key] = pod              // Update the pod in the store with new phase and timestamp

	return nil
}

// ListPods retrieves all pods in a given namespace.
// If namespace is empty, it could be interpreted as list all pods across all namespaces (not implemented here for simplicity yet).
func (s *InMemoryStore) ListPods(namespace string) ([]*api.Pod, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []*api.Pod
	for _, pod := range s.pods {
		if pod.Namespace == namespace {
			result = append(result, pod)
		}
	}
	return result, nil
}

// CreateNode adds a new node to the store.
func (s *InMemoryStore) CreateNode(node *api.Node) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.nodes[node.Name]; exists {
		return fmt.Errorf("node %s already exists", node.Name)
	}
	s.nodes[node.Name] = node
	return nil
}

// GetNode retrieves a node from the store.
func (s *InMemoryStore) GetNode(name string) (*api.Node, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	node, exists := s.nodes[name]
	if !exists {
		return nil, fmt.Errorf("node %s not found", name)
	}
	return node, nil
}

// UpdateNode updates an existing node in the store.
func (s *InMemoryStore) UpdateNode(node *api.Node) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.nodes[node.Name]; !exists {
		return fmt.Errorf("node %s not found for update", node.Name)
	}
	s.nodes[node.Name] = node
	return nil
}

// DeleteNode removes a node from the store.
func (s *InMemoryStore) DeleteNode(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.nodes[name]; !exists {
		return fmt.Errorf("node %s not found for deletion", name)
	}
	delete(s.nodes, name)
	return nil
}

// ListNodes retrieves all nodes.
func (s *InMemoryStore) ListNodes() ([]*api.Node, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var result []*api.Node
	for _, node := range s.nodes {
		result = append(result, node)
	}
	return result, nil
}
