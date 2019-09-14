package podfithostports

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// PodFitsHostPorts is the a scheduling framework plugin.
type PodFitsHostPorts struct {
	nodeInfoSnapshot *nodeinfo.Snapshot

	wantPorts []*v1.ContainerPort
}

// New returns a plugin.
func New() framework.PluginFactory {
	return func(_ *runtime.Unknown, fh framework.FrameworkHandle) (framework.Plugin, error) {
		return &PodFitsHostPorts{
			nodeInfoSnapshot: fh.NodeInfoSnapshot(),
		}, nil
	}
}

// Name is the plugin name.
const Name = "pod-fits-host-ports-plugin"

// Name .
func (p *PodFitsHostPorts) Name() string {
	return Name
}

// Filter checks if a pod spec node name matches the current node.
func (p *PodFitsHostPorts) Filter(pc *framework.PluginContext, pod *v1.Pod, nodeName string) *framework.Status {
	nodeInfo, ok := p.nodeInfoSnapshot.NodeInfoMap[nodeName]
	if !ok {
		return framework.NewStatus(framework.Error, "node not found")
	}

	metadata, _ := pc.Read(framework.ContextKeyPredicateMetadata)

	fit, reasons, err := predicates.PodFitsHostPorts(pod, metadata.(predicates.PredicateMetadata), nodeInfo)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	if !fit {
		return framework.NewPredicateStatus(reasons)
	}

	return nil
}
