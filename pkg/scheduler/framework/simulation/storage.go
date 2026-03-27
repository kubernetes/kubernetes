package simulation

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

type ClusterState interface {
	AddPod(podInfo *framework.PodInfo) error
	RemovePod(podInfo *framework.PodInfo) error
	AddNode(node *v1.Node)
	RemoveNode(node *v1.Node)

	Fork()
	Commit() error
	Revert() error

	Snapshot() fwk.SharedLister
}

var _ ClusterState = &deltaStorage{}
var _ ClusterState = &undoLogStorage{}

type deltaStorage struct {
	storage *DeltaSnapshotStore
}

func (s *deltaStorage) AddPod(podInfo *framework.PodInfo) error {
	return s.storage.StorePodInfo(podInfo, podInfo.Pod.Spec.NodeName)
}

func (s *deltaStorage) RemovePod(podInfo *framework.PodInfo) error {
	return s.storage.RemovePodInfo(podInfo.Pod.Namespace, podInfo.Pod.Name, podInfo.Pod.Spec.NodeName)
}

func (s *deltaStorage) AddNode(node *v1.Node) {
	//TODO: implement
}

func (s *deltaStorage) RemoveNode(node *v1.Node) {
	//TODO: implement
}

func (s *deltaStorage) Fork() {
	s.storage.Fork()
}

func (s *deltaStorage) Commit() error {
	return s.storage.Commit()
}

func (s *deltaStorage) Revert() error {
	s.storage.Revert()
	return nil
}

func (s *deltaStorage) Snapshot() fwk.SharedLister {
	return s.storage
}

type undoLogStorage struct {
	storage    *cache.Snapshot
	restoreFns []func() error
}

func (s *undoLogStorage) AddPod(podInfo *framework.PodInfo) error {
	if err := s.assumePod(podInfo); err != nil {
		return err
	}
	s.restoreFns = append(s.restoreFns, func() error {
		return s.forgetPod(podInfo)
	})

	return nil
}

func (s *undoLogStorage) assumePod(podInfo *framework.PodInfo) error {
	return s.storage.AssumePod(podInfo)
}

func (s *undoLogStorage) RemovePod(podInfo *framework.PodInfo) error {
	logger := klog.FromContext(context.TODO())

	err := s.forgetPod(podInfo)
	if err != nil {
		nodeName := podInfo.Pod.Spec.NodeName
		nodeInfo, getErr := s.storage.NodeInfos().Get(nodeName)
		if getErr != nil {
			return fmt.Errorf("failed to get node %q: %v", nodeName, getErr)
		}
		if removeErr := nodeInfo.RemovePod(logger, podInfo.Pod); removeErr != nil {
			return fmt.Errorf("failed to remove pod from NodeInfo: %v", removeErr)
		}
		s.restoreFns = append(s.restoreFns, func() error {
			nodeInfo.AddPodInfo(podInfo)
			return nil
		})
	} else {
		s.restoreFns = append(s.restoreFns, func() error {
			return s.assumePod(podInfo)
		})
	}

	return nil
}

func (s *undoLogStorage) forgetPod(podInfo *framework.PodInfo) error {
	logger := klog.FromContext(context.TODO())
	return s.storage.ForgetPod(logger, podInfo.Pod)
}

func (s *undoLogStorage) AddNode(node *v1.Node) {
	//TODO: implement
}

func (s *undoLogStorage) RemoveNode(node *v1.Node) {
	//TODO: implement
}

func (s *undoLogStorage) Fork() {
	s.restoreFns = nil
}

func (s *undoLogStorage) Commit() error {
	return nil
}

func (s *undoLogStorage) Revert() error {
	for i := range s.restoreFns {
		err := s.restoreFns[len(s.restoreFns)-1-i]()
		if err != nil {
			return err
		}
	}
	s.restoreFns = nil
	return nil
}

func (s *undoLogStorage) Snapshot() fwk.SharedLister {
	return s.storage
}
