/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"sort"
	"strings"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/fieldmaskpb"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"
	podsv1alpha1 "k8s.io/kubelet/pkg/apis/pods/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

const (
	// FieldMaskMetadataKey is the key for the fieldmask in the gRPC metadata.
	FieldMaskMetadataKey = "x-kubernetes-fieldmask"
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
	s.broadcaster.Broadcast(PodWatchEvent{Type: watch.Deleted, Pod: pod})
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

// applyFieldMask uses Go's standard reflection to copy fields from src to dest based on the mask.
// It supports traversing into slices to select specific fields from the elements.
func ApplyFieldMask(mask *fieldmaskpb.FieldMask, src, dest interface{}) error {
	srcVal := reflect.ValueOf(src)
	destVal := reflect.ValueOf(dest)

	if srcVal.Kind() != reflect.Ptr || destVal.Kind() != reflect.Ptr {
		return fmt.Errorf("src and dest must be pointers")
	}

	for _, path := range mask.GetPaths() {
		parts := strings.Split(path, ".")
		if err := applySinglePath(srcVal, destVal, parts); err != nil {
			return fmt.Errorf("failed to apply field mask path %q: %w", path, err)
		}
	}
	return nil
}

// applySinglePath recursively traverses the src and dest objects according to the path parts,
// creating nested objects in dest as needed and handling slice traversal.
// `dest` must always be a pointer type.
func applySinglePath(src, dest reflect.Value, path []string) error {
	logger := klog.FromContext(context.Background())
	// Invariant: dest is always a pointer.
	for i, part := range path {
		// Source can be a pointer or a struct. If pointer, dereference it.
		if src.Kind() == reflect.Ptr {
			if src.IsNil() {
				return nil // Source path is nil, nothing to copy.
			}
			src = src.Elem()
		}

		// Ensure dest is a non-nil pointer to a struct.
		if dest.Kind() != reflect.Ptr {
			return fmt.Errorf("internal error: dest is not a pointer at path part %q", part)
		}
		if dest.IsNil() {
			dest.Set(reflect.New(dest.Type().Elem()))
		}
		destStruct := dest.Elem()

		// We can only traverse structs.
		if src.Kind() != reflect.Struct {
			return fmt.Errorf("source is not a struct at path part %q", part)
		}
		if destStruct.Kind() != reflect.Struct {
			return fmt.Errorf("destination is not a struct at path part %q", part)
		}

		// Find the field in the source struct.
		// First, try to match by JSON tag. If that fails, fall back to Go field name.
		var srcField reflect.Value
		var destField reflect.Value
		var fieldName string

		found := false
		srcType := src.Type()
		for j := 0; j < src.NumField(); j++ {
			field := srcType.Field(j)
			tag := field.Tag.Get("json")
			if strings.Split(tag, ",")[0] == part {
				srcField = src.Field(j)
				fieldName = field.Name
				destField = destStruct.FieldByName(fieldName)
				found = true
				break
			}
		}

		if !found {
			// Fallback to capitalized field name.
			fieldName = strings.ToUpper(string(part[0])) + part[1:]
			srcField = src.FieldByName(fieldName)
			destField = destStruct.FieldByName(fieldName)
		}

		if !srcField.IsValid() {
			return fmt.Errorf("field %q not found in source struct", part)
		}
		if !destField.IsValid() {
			return fmt.Errorf("field %q not found in destination struct", fieldName)
		}
		if !destField.CanSet() {
			return fmt.Errorf("cannot set destination field %q", fieldName)
		}

		// If this is the last part of the path, we set the value.
		if i == len(path)-1 {
			destField.Set(srcField)
			return nil
		}

		if srcField.Kind() == reflect.Slice {
			remainingPath := path[i+1:]

			// Ensure destination slice is initialized and has the correct size.
			if destField.IsNil() || destField.Len() != srcField.Len() {
				destField.Set(reflect.MakeSlice(srcField.Type(), srcField.Len(), srcField.Len()))
			}

			for j := 0; j < srcField.Len(); j++ {
				srcElem := srcField.Index(j)
				destElem := destField.Index(j)

				// We need pointers for recursion.
				srcElemPtr := srcElem
				if srcElem.Kind() != reflect.Ptr {
					srcElemPtr = srcElem.Addr()
				}

				destElemPtr := destElem
				if destElem.Kind() != reflect.Ptr {
					destElemPtr = destElem.Addr()
				} else if destElem.IsNil() {
					newElem := reflect.New(destElem.Type().Elem())
					destElem.Set(newElem)
					destElemPtr = newElem
				}

				// Recurse.
				if err := applySinglePath(srcElemPtr, destElemPtr, remainingPath); err != nil {
					logger.V(5).Info("Error applying sub-path to slice element", "subpath", remainingPath, "err", err)
				}
			}
			return nil
		}

		// For the next iteration, update src and dest.
		src = srcField

		if destField.Kind() == reflect.Ptr {
			dest = destField
		} else if destField.CanAddr() {
			dest = destField.Addr()
		} else {
			return fmt.Errorf("cannot take address of destination field %q", fieldName)
		}
	}

	return nil
}

// applyFieldMaskToPod is a wrapper to call applyFieldMask for a k8s Pod
func applyFieldMaskToPod(pod *v1.Pod, fieldmask *fieldmaskpb.FieldMask) (*v1.Pod, error) {
	if fieldmask == nil || len(fieldmask.Paths) == 0 {
		return pod, nil
	}
	maskedPod := &v1.Pod{}
	err := ApplyFieldMask(fieldmask, pod, maskedPod)
	if err != nil {
		return nil, fmt.Errorf("failed to apply field mask: %w", err)
	}
	return maskedPod, nil
}

// ListPods returns a list of pods.
func (s *PodsServer) ListPods(ctx context.Context, req *podsv1alpha1.ListPodsRequest) (*podsv1alpha1.ListPodsResponse, error) {
	logger := klog.FromContext(ctx)
	logger.Info("ListPods called", "filter", req.Filter)

	// TODO: Implement filtering based on req.Filter, pagination with req.PageToken and req.PageSize
	podsToReturn := s.List()

	fieldMask, err := getFieldMaskFromContext(ctx)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid field mask: %v", err)
	}

	protoPods := make([][]byte, len(podsToReturn))
	for i, p := range podsToReturn {
		maskedPod, err := applyFieldMaskToPod(p, fieldMask)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "failed to apply field mask: %v", err)
		}
		podBytes, err := maskedPod.Marshal()
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

	fieldMask, err := getFieldMaskFromContext(ctx)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid field mask: %v", err)
	}

	maskedPod, err := applyFieldMaskToPod(pod, fieldMask)
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "failed to apply field mask: %v", err)
	}
	podBytes, err := maskedPod.Marshal()
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

	fieldMask, err := getFieldMaskFromContext(stream.Context())
	if err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid field mask: %v", err)
	}

	// Validate the field mask before starting the watch.
	if _, err := applyFieldMaskToPod(&v1.Pod{}, fieldMask); err != nil {
		return status.Errorf(codes.InvalidArgument, "invalid field mask: %v", err)
	}

	// Send initial ADDED events
	initialPods := s.List()
	for _, p := range initialPods {
		maskedPod, err := applyFieldMaskToPod(p, fieldMask)
		if err != nil {
			// Should not happen, the field mask was already validated.
			logger.Error(err, "Error applying field mask to initial watch event pod")
			metrics.PodWatchEventsDroppedTotal.Inc()
			continue
		}
		podBytes, err := maskedPod.Marshal()
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
			maskedPod, err := applyFieldMaskToPod(event.Pod, fieldMask)
			if err != nil {
				logger.Error(err, "Error applying field mask to watch event pod")
				metrics.PodWatchEventsDroppedTotal.Inc()
				continue
			}
			podBytes, err := maskedPod.Marshal()
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

// getFieldMaskFromContext extracts the field mask from the gRPC metadata.
func getFieldMaskFromContext(ctx context.Context) (*fieldmaskpb.FieldMask, error) {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return nil, nil
	}
	fieldMasks := md.Get(FieldMaskMetadataKey)
	if len(fieldMasks) == 0 {
		return nil, nil
	}
	if len(fieldMasks) > 1 {
		return nil, fmt.Errorf("multiple field masks not supported")
	}
	if fieldMasks[0] == "" {
		return nil, nil
	}
	mask := &fieldmaskpb.FieldMask{Paths: strings.Split(fieldMasks[0], ",")}
	return mask, nil
}
