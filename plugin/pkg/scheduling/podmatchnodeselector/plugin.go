package podmatchnodeselector

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// PodMatchNodeSelector is the a scheduling framework plugin.
type PodMatchNodeSelector struct {
	nodeInfoSnapshot *nodeinfo.Snapshot

	wantPorts []*v1.ContainerPort
}

// New returns a plugin.
func New() framework.PluginFactory {
	return func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
		return &PodMatchNodeSelector{
			nodeInfoSnapshot: fh.NodeInfoSnapshot(),
		}, nil
	}
}

// Name is the plugin name.
const Name = "pod-match-node-selector-plugin"

// Name .
func (p *PodMatchNodeSelector) Name() string {
	return Name
}

// Filter checks if a pod node selector matches the node label.
func (p *PodMatchNodeSelector) Filter(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	nodeInfo, ok := p.nodeInfoSnapshot.NodeInfoMap[nodeName]
	if !ok {
		return framework.NewStatus(framework.Error, "node not found")
	}

	fit, reasons, err := predicates.PodMatchNodeSelector(pod, nil, nodeInfo)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if !fit {
		return framework.NewPredicateStatus(reasons)
	}

	return nil
}
