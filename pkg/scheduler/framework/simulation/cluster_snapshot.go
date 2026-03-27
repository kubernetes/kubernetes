package simulation

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	fwk "k8s.io/kube-scheduler/framework"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

type ClusterData struct {
	pods            []*v1.Pod
	nodes           []*v1.Node
	informerFactory informers.SharedInformerFactory
}

type SchedulablePod struct {
	pod            *v1.Pod
	candidateNodes []string
}

type ScheduledPod struct {
	pod          *v1.Pod
	selectedNode string
}

type Options struct {
	StopOnFailure bool
}

type SchedulingSimulator interface {
	ClusterSnapshot
}

type schedulingSimulator struct {
	*clusterSnapshot
}

type StorageType string

const (
	Delta   StorageType = "delta"
	UndoLog StorageType = "undo-log"
)

func NewSchedulingSimulator(ctx context.Context, clusterData *ClusterData, schedConfig *schedulerconfig.KubeSchedulerConfiguration, storageType StorageType) (SchedulingSimulator, error) {
	var clusterState ClusterState
	if storageType == Delta {
		clusterState = &deltaStorage{
			storage: NewDeltaSnapshotStore(1, clusterData.pods, clusterData.nodes),
		}
	} else {
		clusterState = &undoLogStorage{
			storage: cache.NewSnapshot(clusterData.pods, clusterData.nodes),
		}
	}

	frameworks := make(map[string]schedulerframework.Framework)
	for _, profile := range schedConfig.Profiles {
		framework, err := runtime.NewFramework(
			ctx,
			plugins.NewInTreeRegistry(),
			&schedConfig.Profiles[0],
			runtime.WithSnapshotSharedLister(clusterState.Snapshot()),
			runtime.WithInformerFactory(clusterData.informerFactory),
		)
		if err != nil {
			return nil, err
		}
		frameworks[profile.SchedulerName] = framework
	}

	clusterSnapshot := &clusterSnapshot{
		clusterState: clusterState,
		frameworks:   frameworks,
	}
	return &schedulingSimulator{clusterSnapshot: clusterSnapshot}, nil
}

type ClusterSnapshot interface {
	SchedulePods(pods []SchedulablePod, opts Options) ([]ScheduledPod, error)
	UnschedulePods(pods []ScheduledPod) error
	ScaleUp(node *v1.Node)
	ScaleDown(node *v1.Node)

	Fork()
	Commit() error
	Revert() error
}

var _ ClusterSnapshot = &clusterSnapshot{}

type clusterSnapshot struct {
	clusterState ClusterState
	frameworks   map[string]schedulerframework.Framework
}

func (c *clusterSnapshot) SchedulePods(pods []SchedulablePod, opts Options) ([]ScheduledPod, error) {
	result := make([]ScheduledPod, 0)
	if len(pods) == 0 {
		return result, nil
	}

	framework, err := c.getFramework(pods[0].pod)
	if err != nil {
		return nil, err
	}

	for _, pod := range pods {
		cycleState := schedulerframework.NewCycleState()
		ctx := context.TODO()
		framework.RunPreFilterPlugins(ctx, cycleState, pod.pod)
		success := false
		for _, nodeName := range pod.candidateNodes {
			node, err := framework.SnapshotSharedLister().NodeInfos().Get(nodeName)
			if err != nil {
				return nil, err
			}
			status := framework.RunFilterPlugins(ctx, cycleState, pod.pod, node)
			if status.IsSuccess() {
				result = append(result, ScheduledPod{
					pod:          pod.pod,
					selectedNode: nodeName,
				})
				err := c.assumeAndReserve(ctx, pod.pod, nodeName, cycleState)
				if err != nil {
					unreserveErr := c.unreserveAndForget(ctx, pod.pod, nodeName, cycleState)
					if unreserveErr != nil {
						return nil, unreserveErr
					}
					return nil, err
				}
				success = true
				break
			}
		}
		if !success {
			if opts.StopOnFailure {
				return result, nil
			}
		}
	}
	return result, nil
}

func (c *clusterSnapshot) assumeAndReserve(ctx context.Context, pod *v1.Pod, nodeName string, cycleState fwk.CycleState) error {
	framework, err := c.getFramework(pod)
	if err != nil {
		return err
	}

	podInfo, err := schedulerframework.NewPodInfo(pod.DeepCopy())
	if err != nil {
		return err
	}
	podInfo.Pod.Spec.NodeName = nodeName
	c.clusterState.AddPod(podInfo)
	status := framework.RunReservePluginsReserve(ctx, cycleState, pod, nodeName)
	if !status.IsSuccess() {
		return status.AsError()
	}
	return nil
}

func (c *clusterSnapshot) unreserveAndForget(ctx context.Context, pod *v1.Pod, nodeName string, cycleState fwk.CycleState) error {
	framework, err := c.getFramework(pod)
	if err != nil {
		return err
	}

	framework.RunReservePluginsUnreserve(ctx, cycleState, pod, nodeName)
	// TODO: should I remove node from podInfo?
	podInfo, err := schedulerframework.NewPodInfo(pod.DeepCopy())
	if err != nil {
		return err
	}
	podInfo.Pod.Spec.NodeName = nodeName
	err = c.clusterState.RemovePod(podInfo)
	podInfo.Pod.Spec.NodeName = "" // clear after use
	return err
}

func (c *clusterSnapshot) UnschedulePods(pods []ScheduledPod) error {
	for _, sp := range pods {
		podInfo, err := schedulerframework.NewPodInfo(sp.pod.DeepCopy())
		if err != nil {
			return err
		}
		podInfo.Pod.Spec.NodeName = sp.selectedNode
		deleteErr := c.clusterState.RemovePod(podInfo)
		if deleteErr != nil {
			return deleteErr
		}
	}
	return nil
}

// TODO: revisit when will implement autoscaling
func (c *clusterSnapshot) ScaleUp(node *v1.Node) {
	c.clusterState.AddNode(node)
}

func (c *clusterSnapshot) ScaleDown(node *v1.Node) {
	c.clusterState.RemoveNode(node)
}

func (c *clusterSnapshot) Fork() {
	c.clusterState.Fork()
}

func (c *clusterSnapshot) Commit() error {
	return c.clusterState.Commit()
}

func (c *clusterSnapshot) Revert() error {
	return c.clusterState.Revert()
}

func (c *clusterSnapshot) getFramework(pod *v1.Pod) (schedulerframework.Framework, error) {
	schedulerName := pod.Spec.SchedulerName
	if schedulerName == "" {
		schedulerName = v1.DefaultSchedulerName
	}

	framework, ok := c.frameworks[schedulerName]
	if !ok {
		return nil, fmt.Errorf("no framework found for scheduler: %q", schedulerName)
	}

	return framework, nil
}
