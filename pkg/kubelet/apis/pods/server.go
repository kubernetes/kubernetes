/*
Copyright The Kubernetes Authors.

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

package pods

import (
	"context"
	"sort"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

type PodWatchEvent struct {
	Type watch.EventType
	Pod  *v1.Pod
}

type broadcaster struct {
	lock    sync.Mutex
	clients map[chan PodWatchEvent]struct{}
}

func NewBroadcaster() *broadcaster {
	return &broadcaster{
		clients: make(map[chan PodWatchEvent]struct{}),
	}
}

func (b *broadcaster) Register(client chan PodWatchEvent) {
	b.lock.Lock()
	defer b.lock.Unlock()
	b.clients[client] = struct{}{}
	logger := klog.FromContext(context.Background())
	logger.Info("Registered new watch client", "totalClients", len(b.clients))
}

func (b *broadcaster) Unregister(client chan PodWatchEvent) {
	b.lock.Lock()
	defer b.lock.Unlock()
	delete(b.clients, client)
	logger := klog.FromContext(context.Background())
	logger.Info("Unregistered watch client", "totalClients", len(b.clients))
}

func (b *broadcaster) Broadcast(event PodWatchEvent) {
	b.lock.Lock()
	defer b.lock.Unlock()
	logger := klog.FromContext(context.Background())
	for client := range b.clients {
		select {
		case client <- event:
		default:
			logger.Info("Watch client channel is full, dropping event.")
			metrics.PodWatchEventsDroppedTotal.Inc()
		}
	}
}

// PodsServer is the gRPC server that provides pod information.
type PodsServer struct {
	podsv1alpha1.UnimplementedPodsServer
	lock        sync.RWMutex
	pods        map[types.UID]*v1.Pod
	broadcaster *broadcaster
}

// NewPodsServer creates a new PodServer for production use.
func NewPodsServer(broadcaster *broadcaster) *PodsServer {
	return &PodsServer{
		pods:        make(map[types.UID]*v1.Pod),
		broadcaster: broadcaster,
	}
}

// NewPodsServerForTest creates a new PodServer with an injectable ticker for testing.
func NewPodsServerForTest(broadcaster *broadcaster) *PodsServer {
	return &PodsServer{
		pods:        make(map[types.UID]*v1.Pod),
		broadcaster: broadcaster,
	}
}

// OnPodAdded is called when a pod is added.
func (s *PodsServer) OnPodAdded(pod *v1.Pod) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.pods[pod.UID] = pod.DeepCopy()
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Added, Pod: pod})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod added to storage", "podUID", pod.UID)
}

// OnPodUpdated is called when a pod is updated.
func (s *PodsServer) OnPodUpdated(pod *v1.Pod) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.pods[pod.UID] = pod.DeepCopy()
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Modified, Pod: pod})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod updated in storage", "podUID", pod.UID)
}

// OnPodRemoved is called when a pod is removed.
func (s *PodsServer) OnPodRemoved(pod *v1.Pod) {
	s.lock.Lock()
	defer s.lock.Unlock()
	delete(s.pods, pod.UID)
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Deleted, Pod: pod.DeepCopy()})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod removed from storage", "podUID", pod.UID)
}

// OnPodStatusUpdated is called when a pod's status is updated.
func (s *PodsServer) OnPodStatusUpdated(pod *v1.Pod, status v1.PodStatus) {
	s.lock.Lock()
	defer s.lock.Unlock()
	if storedPod, ok := s.pods[pod.UID]; ok {
		storedPod.Status = status
		s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Modified, Pod: storedPod})
		logger := klog.FromContext(context.Background())
		logger.Info("Pod status updated in storage", "podUID", pod.UID)
	}
}

// Get returns a pod by UID.
func (s *PodsServer) Get(uid types.UID) (*v1.Pod, bool) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	pod, ok := s.pods[uid]
	if !ok {
		return nil, false
	}
	return pod.DeepCopy(), true
}

// List returns all pods.
func (s *PodsServer) List() []*v1.Pod {
	s.lock.RLock()
	defer s.lock.RUnlock()
	pods := make([]*v1.Pod, 0, len(s.pods))
	for _, pod := range s.pods {
		pods = append(pods, pod.DeepCopy())
	}
	sort.Slice(pods, func(i, j int) bool {
		return pods[i].UID < pods[j].UID
	})
	return pods
}

// ListPods returns a list of pods.
func (s *PodsServer) ListPods(ctx context.Context, req *podsv1alpha1.ListPodsRequest) (*podsv1alpha1.ListPodsResponse, error) {
	logger := klog.FromContext(ctx)
	logger.Info("ListPods called")

	// TODO: Implement filtering based on req.Filter, pagination with req.PageToken and req.PageSize
	podsToReturn := s.List()

	protoPods := make([][]byte, len(podsToReturn))
	for i, p := range podsToReturn {
		podBytes, err := p.Marshal()
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to marshal pod: %v", err)
		}
		protoPods[i] = podBytes
	}

	return &podsv1alpha1.ListPodsResponse{Pods: protoPods}, nil
}

// GetPod returns a single pod by UID.
func (s *PodsServer) GetPod(ctx context.Context, req *podsv1alpha1.GetPodRequest) (*podsv1alpha1.GetPodResponse, error) {
	logger := klog.FromContext(ctx)
	logger.Info("GetPod called", "podUID", req.PodUID)

	podUID := types.UID(req.PodUID)
	pod, ok := s.Get(podUID)
	if !ok {
		return nil, status.Errorf(codes.NotFound, "pod with UID %s not found", req.PodUID)
	}

	podBytes, err := pod.Marshal()
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to marshal pod: %v", err)
	}

	return &podsv1alpha1.GetPodResponse{Pod: podBytes}, nil
}

// WatchPods streams pod events.
func (s *PodsServer) WatchPods(req *podsv1alpha1.WatchPodsRequest, stream podsv1alpha1.Pods_WatchPodsServer) error {
	clientAddr := "unknown"
	if p, ok := peer.FromContext(stream.Context()); ok {
		clientAddr = p.Addr.String()
	}
	logger := klog.FromContext(stream.Context())
	logger.Info("WatchPods called", "client", clientAddr)

	clientChannel := make(chan PodWatchEvent, 100)
	s.broadcaster.Register(clientChannel)
	defer func() {
		s.broadcaster.Unregister(clientChannel)
		logger.Info("Watch client disconnected", "client", clientAddr)
	}()

	// Send initial ADDED events
	initialPods := s.List()
	for _, p := range initialPods {
		podBytes, err := p.Marshal()
		if err != nil {
			logger.Error(err, "Error marshalling initial watch event pod")
			metrics.PodWatchEventsDroppedTotal.Inc()
			continue
		}
		if err := stream.Send(&podsv1alpha1.WatchPodsEvent{
			Type: podsv1alpha1.EventType_ADDED,
			Pod:  podBytes,
		}); err != nil {
			logger.Error(err, "Error sending initial watch event")
			return err
		}
	}

	for {
		select {
		case <-stream.Context().Done():
			logger.Info("Watch context cancelled", "client", clientAddr)
			return stream.Context().Err()
		case event := <-clientChannel:
			podBytes, err := event.Pod.Marshal()
			if err != nil {
				logger.Error(err, "Error marshalling watch event pod")
				metrics.PodWatchEventsDroppedTotal.Inc()
				continue
			}
			if err := stream.Send(&podsv1alpha1.WatchPodsEvent{
				Type: convertWatchEventType(event.Type),
				Pod:  podBytes,
			}); err != nil {
				logger.Error(err, "Error sending watch event to client", "client", clientAddr)
				return err
			}
		}
	}
}

func convertWatchEventType(watchType watch.EventType) podsv1alpha1.EventType {
	switch watchType {
	case watch.Added:
		return podsv1alpha1.EventType_ADDED
	case watch.Modified:
		return podsv1alpha1.EventType_MODIFIED
	case watch.Deleted:
		return podsv1alpha1.EventType_DELETED
	default:
		return podsv1alpha1.EventType_ADDED
	}
}
