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
	"k8s.io/kubernetes/pkg/kubelet/pod"
)

type PodWatchEvent struct {
	Type watch.EventType
	UID  types.UID
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

// OnPodAdded is called when a pod is added.
func (s *PodsServer) OnPodAdded(pod *v1.Pod) {
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Added, UID: pod.UID})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod added to storage", "podUID", pod.UID)
}

// OnPodUpdated is called when a pod is updated.
func (s *PodsServer) OnPodUpdated(pod *v1.Pod) {
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Modified, UID: pod.UID})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod updated in storage", "podUID", pod.UID)
}

// OnPodRemoved is called when a pod is removed.
func (s *PodsServer) OnPodRemoved(pod *v1.Pod) {
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Deleted, UID: pod.UID, Pod: pod})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod removed from storage", "podUID", pod.UID)
}

// OnPodStatusUpdated is called when a pod's status is updated.
func (s *PodsServer) OnPodStatusUpdated(pod *v1.Pod, status v1.PodStatus) {
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Modified, UID: pod.UID})
	logger := klog.FromContext(context.Background())
	logger.Info("Pod status updated in storage", "podUID", pod.UID)
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
			var podToMarshal *v1.Pod
			if event.Type == watch.Deleted {
				podToMarshal = event.Pod
			} else {
				p, ok := s.podManager.GetPodByUID(event.UID)
				if ok {
					podToMarshal = p
				} else {
					logger.Info("Pod not found in manager during watch event processing", "uid", event.UID, "type", event.Type)
					continue
				}
			}

			if podToMarshal == nil {
				continue
			}

			podBytes, err := podToMarshal.Marshal()
			if err != nil {
				logger.Error(err, "Error marshalling watch event pod")
				metrics.PodWatchEventsDroppedTotal.Inc()
				continue
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
	default:
		return podsv1alpha1.EventType_ADDED, status.Errorf(codes.Internal, "unknown watch event type: %v", watchType)
	}
}