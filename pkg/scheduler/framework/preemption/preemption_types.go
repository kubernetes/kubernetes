package preemption

import (
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
)

// Preemptor abstracts the entity that initiates preemption.
// It acts as a unified interface for both single Pods and collective Workloads PodGroups,
// allowing the preemption logic to treat them polymorphically.
type Preemptor interface {
	// Priority returns the scheduling priority of the preemptor.
	// This value is used to identify potential victims (which must have lower priority).
	Priority() int32

	// IsPodGroup returns true if the preemptor represents a collective PodGroup
	// that must be scheduled atomically, or false if it represents a single Pod.
	IsPodGroup() bool

	// Members returns the list of Pods that belong to this preemptor.
	// For a single Pod preemptor, this returns a slice containing only that Pod.
	// For a PodGroup, this returns all Pods in the group.
	Members() []*v1.Pod

	// IsEligibleToPreemptOthers checks if the preemptor is allowed to preempt other Pods.
	// This typically validates the PreemptionPolicy (e.g., returning false if the policy is "Never").
	IsEligibleToPreemptOthers() bool

	// SupportExtenders indicates whether this preemptor type supports the execution
	// of scheduler extenders during the preemption cycle.
	SupportExtenders() bool

	// GetNamespace returns the namespace of the preemptor.
	GetNamespace() string

	// GetName returns the name of the preemptor.
	// For a single Pod, this is the Pod's name. For a PodGroup, this is the PodGroup's name.
	// returns "unknown" if Members list is empty
	GetName() string

	// GetRepresentativePod returns a single Pod to act as a proxy for the entire preemptor.
	// For a single Pod, this returns the Pod itself.
	// For a PodGroup, this returns a designated member (e.g., the first Pod) to be used
	// in contexts that require a single v1.Pod object (such as event recording, logging,
	// or legacy utility functions).
	GetRepresentativePod() *v1.Pod
}

type preemptor struct {
	Preemptor
	priority         int32
	pods             []*v1.Pod
	preemptionPolicy *v1.PreemptionPolicy
	isWorkload       bool
}

func NewPodPreemptor(p *v1.Pod) Preemptor {
	return &preemptor{
		priority:         corev1helpers.PodPriority(p),
		pods:             []*v1.Pod{p},
		preemptionPolicy: p.Spec.PreemptionPolicy,
		isWorkload:       false,
	}
}

func (p *preemptor) Priority() int32 {
	return p.priority
}

func (p *preemptor) IsPodGroup() bool {
	return p.isWorkload
}

func (p *preemptor) Members() []*v1.Pod {
	return p.pods
}

func (p *preemptor) IsEligibleToPreemptOthers() bool {
	return p.preemptionPolicy == nil || *p.preemptionPolicy != v1.PreemptNever
}

func (p *preemptor) SupportExtenders() bool {
	return !p.isWorkload
}

func (p *preemptor) GetNamespace() string {
	if len(p.pods) > 0 {
		return p.pods[0].Namespace
	}
	return ""
}

func (p *preemptor) GetName() string {
	if len(p.pods) == 0 {
		return "unknown"
	}

	firstPod := p.GetRepresentativePod()

	if p.isWorkload {
		ref := firstPod.Spec.WorkloadRef

		// Start with the Workload Name (e.g., "my-job")
		name := ref.Name

		// Append PodGroup if distinct (e.g., "my-job/group-1")
		if ref.PodGroup != "" {
			name = name + "/" + ref.PodGroup
		}

		// Append ReplicaKey if present (e.g., "my-job/group-1/idx-0")
		// This is crucial for distinguishing between retries of the same job.
		if ref.PodGroupReplicaKey != "" {
			name = name + "/" + ref.PodGroupReplicaKey
		}

		return name
	}

	return firstPod.Name
}

func (p *preemptor) GetRepresentativePod() *v1.Pod {
	if len(p.pods) == 0 {
		return nil
	}

	return p.pods[0]
}

// Domain represents the boundary or scope within which the preemption logic is evaluated.
// It abstracts the scheduling domain, which can range from a single Node (for standard Pod preemption)
// to a group of Nodes or the entire Cluster (for Workload preemption).
type Domain interface {
	// Nodes returns a list of NodeInfo objects that belong to this domain.
	// The preemption logic uses this to check feasibility and resource availability
	// within the specific scope.
	Nodes() []fwk.NodeInfo

	// GetAllPossibleVictims returns all potential victims (PreemptionUnits) running within this domain. (single Pod or a PodGroup)
	GetAllPossibleVictims(nodeInfoLister fwk.NodeInfoLister) []PreemptionUnit

	// GetName returns a unique identifier or name for the domain.
	// This is primarily used for logging and debugging purposes.
	GetName() string

	// Snapshot creates a deep copy of the Domain.
	// This allows the preemption logic to perform dry-run simulations (adding/removing victims)
	// without mutating the actual scheduler state until a final decision is made.
	Snapshot() Domain
}

type domain struct {
	Domain
	nodes         []fwk.NodeInfo
	name          string
	podGroupIndex map[podGroupKey][]fwk.PodInfo
}

func (d *domain) Nodes() []fwk.NodeInfo {
	return d.nodes
}

func (d *domain) GetAllPossibleVictims(nodeInfoLister fwk.NodeInfoLister) []PreemptionUnit {
	processedPodGroups := make(map[podGroupKey]bool)
	var allVictims []PreemptionUnit

	for _, node := range d.nodes {
		for _, pi := range node.GetPods() {
			pod := pi.GetPod()

			workloadAwarePreemptionEnabeld := utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption)

			if ref := pod.Spec.WorkloadRef; workloadAwarePreemptionEnabeld && ref != nil {
				key := newPodGroupKey(pod.Namespace, pod.Spec.WorkloadRef)

				// Deduplication Check
				if processedPodGroups[key] {
					continue
				}

				// Get all pods for this gang (Global Lookup via Index)
				gangPods := d.podGroupIndex[key]

				if len(gangPods) > 0 {
					// Collect NodeInfo for ALL pods in the gang
					var gangNodes []fwk.NodeInfo
					for _, gp := range gangPods {
						nodeName := gp.GetPod().Spec.NodeName
						// Use the global lister to find the node, even if it's not in domain
						if n, err := nodeInfoLister.Get(nodeName); err == nil {
							gangNodes = append(gangNodes, n.Snapshot())
						}
					}

					unit := d.newPreemptionUnit(gangPods, corev1helpers.PodPriority(pod), gangNodes)
					allVictims = append(allVictims, unit)
				}

				// Mark as processed so we don't do this again for the next pod in this gang
				processedPodGroups[key] = true

			} else {
				// We just pass the current node (slice of 1)
				unit := d.newPreemptionUnit(
					[]fwk.PodInfo{pi},
					corev1helpers.PodPriority(pod),
					[]fwk.NodeInfo{node}, // Single node slice
				)
				allVictims = append(allVictims, unit)
			}
		}
	}

	return allVictims
}

func (d *domain) Snapshot() Domain {
	var snapshotNodes []fwk.NodeInfo

	for _, node := range d.nodes {
		snapshotNodes = append(snapshotNodes, node.Snapshot())
	}

	return &domain{
		nodes:         snapshotNodes,
		name:          d.name,
		podGroupIndex: d.podGroupIndex,
	}
}

func (d *domain) GetName() string {
	return d.name
}

// PreemptionUnit represents an atomic entity that can be preempted (a victim).
// It abstracts individual Pods and PodGroup, ensuring that
// atomic entities are treated as a single unit during eviction.
type PreemptionUnit interface {
	// Priority returns the priority of the preemption unit.
	// For a single Pod, this is the Pod's priority.
	// For a PodGroup, this is typically the priority of the PodGroup (or its members).
	Priority() int32

	// IsPodGroup returns true if the unit represents a collective pod group
	// that must be preempted atomically, or false if it is a single Pod.
	IsPodGroup() bool

	// AffectedNodes returns a map of Node names to NodeInfo for all nodes
	// where members of this preemption unit are currently running.
	// This allows the preemption logic to identify the blast radius of evicting this unit.
	AffectedNodes() map[string]fwk.NodeInfo

	// Pods returns the list of all Pods that belong to this preemption unit.
	// Evicting this unit implies evicting all Pods in this list.
	Pods() []fwk.PodInfo
}

type preemptionUnit struct {
	PreemptionUnit
	pods          []fwk.PodInfo
	priority      int32
	affectedNodes map[string]fwk.NodeInfo //TODO: should I store that here?
	isPodGroup    bool
}

func (d *domain) newPreemptionUnit(pods []fwk.PodInfo, priority int32, nodeInfos []fwk.NodeInfo) PreemptionUnit {
	nodes := make(map[string]fwk.NodeInfo)
	for _, node := range nodeInfos {
		nodes[node.Node().Name] = node
	}

	return &preemptionUnit{
		affectedNodes: nodes,
		priority:      priority,
		isPodGroup:    len(pods) > 1,
		pods:          pods,
	}
}

func (pu *preemptionUnit) Pods() []fwk.PodInfo {
	return pu.pods
}

func (pu *preemptionUnit) Priority() int32 {
	return pu.priority
}

func (pu *preemptionUnit) IsPodGroup() bool {
	return pu.isPodGroup
}

func (pu *preemptionUnit) AffectedNodes() map[string]fwk.NodeInfo {
	return pu.affectedNodes
}
