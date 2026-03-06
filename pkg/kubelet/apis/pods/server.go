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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/pod"
)

type PodWatchEvent struct {
	Type watch.EventType
	UID  types.UID
	Pod  *v1.Pod
}

type broadcaster struct {
	lock     sync.RWMutex
	clients  map[chan PodWatchEvent]struct{}
	incoming chan PodWatchEvent
}

func NewBroadcaster() *broadcaster {
	b := &broadcaster{
		clients:  make(map[chan PodWatchEvent]struct{}),
		incoming: make(chan PodWatchEvent, 1000),
	}
	go b.run()
	return b
}

func (b *broadcaster) run() {
	logger := klog.FromContext(context.Background())
	for event := range b.incoming {
		b.lock.RLock()
		// We collect clients to drop to avoid modifying the map during iteration
		// which would require a Write lock and block other RLockers.
		var clientsToDrop []chan PodWatchEvent

		for client := range b.clients {
			select {
			case client <- event:
			default:
				logger.Info("Watch client channel is full, dropping client.")
				metrics.PodWatchEventsDroppedTotal.Inc()
				clientsToDrop = append(clientsToDrop, client)
			}
		}
		b.lock.RUnlock()

		if len(clientsToDrop) > 0 {
			b.lock.Lock()
			for _, client := range clientsToDrop {
				if _, ok := b.clients[client]; ok {
					delete(b.clients, client)
					close(client)
				}
			}
			b.lock.Unlock()
		}
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
	if _, ok := b.clients[client]; ok {
		delete(b.clients, client)
		close(client)
	}
	logger := klog.FromContext(context.Background())
	logger.Info("Unregistered watch client", "totalClients", len(b.clients))
}

func (b *broadcaster) Broadcast(event PodWatchEvent) {
	select {
	case b.incoming <- event:
	default:
		// This should realistically never happen because run() purges slow clients.
		logger := klog.FromContext(context.Background())
		logger.Info("Broadcaster internal buffer full, dropping event.")
		metrics.PodWatchEventsDroppedTotal.Inc()
	}
}

// PodsServer is the gRPC server that provides pod information.
type PodsServer struct {
	podsv1alpha1.UnimplementedPodsServer
	podManager  pod.Manager
	broadcaster *broadcaster
}

// NewPodsServer creates a new PodServer for production use.
func NewPodsServer(broadcaster *broadcaster, podManager pod.Manager) *PodsServer {
	return &PodsServer{
		podManager:  podManager,
		broadcaster: broadcaster,
	}
}

// NewPodsServerForTest creates a new PodServer with an injectable ticker for testing.
func NewPodsServerForTest(broadcaster *broadcaster, podManager pod.Manager) *PodsServer {
	return &PodsServer{
		podManager:  podManager,
		broadcaster: broadcaster,
	}
}

// OnPodUpdated is called when a pod's spec and status are coherent.
func (s *PodsServer) OnPodUpdated(pod *v1.Pod, status v1.PodStatus, eventType watch.EventType) {
	podCopy := *pod
	podCopy.Status = status
	s.broadcaster.Broadcast(PodWatchEvent{Type: eventType, UID: pod.UID, Pod: &podCopy})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod update broadcasted", "podUID", pod.UID, "type", eventType)
}

// OnPodRemoved is called when a pod is removed.
func (s *PodsServer) OnPodRemoved(pod *v1.Pod) {
	minimalPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			UID:       pod.UID,
		},
	}
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Deleted, UID: pod.UID, Pod: minimalPod})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod removed broadcasted", "podUID", pod.UID)
}

// ListPods returns a list of pods.
func (s *PodsServer) ListPods(ctx context.Context, req *podsv1alpha1.ListPodsRequest) (*podsv1alpha1.ListPodsResponse, error) {
	logger := klog.FromContext(ctx)
	logger.Info("ListPods called")

	// TODO: Implement filtering based on req.Filter, pagination with req.PageToken and req.PageSize
	podsToReturn := s.podManager.GetPods()
	sort.Slice(podsToReturn, func(i, j int) bool {
		return podsToReturn[i].UID < podsToReturn[j].UID
	})

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
	pod, ok := s.podManager.GetPodByUID(podUID)
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
	initialPods := s.podManager.GetPods()
	sort.Slice(initialPods, func(i, j int) bool {
		return initialPods[i].UID < initialPods[j].UID
	})

	for _, p := range initialPods {
		podBytes, err := p.Marshal()
		if err != nil {
			logger.Error(err, "Error marshalling initial watch event pod")
			metrics.PodWatchEventsDroppedTotal.Inc()
			return status.Errorf(codes.Internal, "error marshalling initial watch event pod: %v", err)
		}
		if err := stream.Send(&podsv1alpha1.WatchPodsEvent{
			Type: podsv1alpha1.EventType_ADDED,
			Pod:  podBytes,
		}); err != nil {
			logger.Error(err, "Error sending initial watch event")
			return err
		}
	}

	// Send INITIAL_SYNC_COMPLETE event to indicate that the initial list is complete
	if err := stream.Send(&podsv1alpha1.WatchPodsEvent{
		Type: podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE,
	}); err != nil {
		logger.Error(err, "Error sending initial sync watch event")
		return err
	}

	for {
		select {
		case <-stream.Context().Done():
			logger.Info("Watch context cancelled", "client", clientAddr)
			return stream.Context().Err()
		case event, ok := <-clientChannel:
			if !ok {
				logger.Info("Watch client channel closed", "client", clientAddr)
				return status.Errorf(codes.ResourceExhausted, "watch client channel closed")
			}
			var podToMarshal *v1.Pod
			if event.Pod != nil {
				podToMarshal = event.Pod
			} else if event.Type != watch.Bookmark {
				// Fallback to looking up the pod if it is not provided in the event.
				// This should not happen with the current implementation but is kept for safety.
				p, ok := s.podManager.GetPodByUID(event.UID)
				if ok {
					podToMarshal = p
				} else {
					logger.Info("Pod not found in manager during watch event processing", "uid", event.UID, "type", event.Type)
					continue
				}
			}

			var podBytes []byte
			if podToMarshal != nil {
				var err error
				podBytes, err = podToMarshal.Marshal()
				if err != nil {
					logger.Error(err, "Error marshalling watch event pod")
					metrics.PodWatchEventsDroppedTotal.Inc()
					return status.Errorf(codes.Internal, "error marshalling watch event pod: %v", err)
				}
			}

			eventType, err := convertWatchEventType(event.Type)
			if err != nil {
				logger.Error(err, "Unknown watch event type")
				continue
			}

			if err := stream.Send(&podsv1alpha1.WatchPodsEvent{
				Type: eventType,
				Pod:  podBytes,
			}); err != nil {
				logger.Error(err, "Error sending watch event to client", "client", clientAddr)
				return err
			}
		}
	}
}

func convertWatchEventType(watchType watch.EventType) (podsv1alpha1.EventType, error) {
	switch watchType {
	case watch.Added:
		return podsv1alpha1.EventType_ADDED, nil
	case watch.Modified:
		return podsv1alpha1.EventType_MODIFIED, nil
	case watch.Deleted:
		return podsv1alpha1.EventType_DELETED, nil
	case watch.Bookmark:
		return podsv1alpha1.EventType_INITIAL_SYNC_COMPLETE, nil
	default:
		return podsv1alpha1.EventType_UNSPECIFIED, status.Errorf(codes.Internal, "unknown watch event type: %v", watchType)
	}
}
