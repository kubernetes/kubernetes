/*
Copyright 2021 The Kubernetes Authors.

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

package preemption

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	apipod "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Candidate represents a nominated node on which the preemptor can be scheduled,
// along with the list of victims that should be evicted for the preemptor to fit the node.
type Candidate interface {
	// Victims wraps a list of to-be-preempted Pods and the number of PDB violation.
	Victims() *extenderv1.Victims
	// Name returns the target node name where the preemptor gets nominated to run.
	Name() string
}

type candidate struct {
	victims *extenderv1.Victims
	name    string
}

// Victims returns s.victims.
func (s *candidate) Victims() *extenderv1.Victims {
	return s.victims
}

// Name returns s.name.
func (s *candidate) Name() string {
	return s.name
}

type candidateList struct {
	idx   int32
	items []Candidate
}

func newCandidateList(size int32) *candidateList {
	return &candidateList{idx: -1, items: make([]Candidate, size)}
}

// add adds a new candidate to the internal array atomically.
func (cl *candidateList) add(c *candidate) {
	if idx := atomic.AddInt32(&cl.idx, 1); idx < int32(len(cl.items)) {
		cl.items[idx] = c
	}
}

// size returns the number of candidate stored. Note that some add() operations
// might still be executing when this is called, so care must be taken to
// ensure that all add() operations complete before accessing the elements of
// the list.
func (cl *candidateList) size() int32 {
	n := atomic.LoadInt32(&cl.idx) + 1
	if n >= int32(len(cl.items)) {
		n = int32(len(cl.items))
	}
	return n
}

// get returns the internal candidate array. This function is NOT atomic and
// assumes that all add() operations have been completed.
func (cl *candidateList) get() []Candidate {
	return cl.items[:cl.size()]
}

type Preemptor interface {
	Priority() int32
	IsWorkload() bool
	Members() []*v1.Pod
	IsElgigibleToPreemptOthers() bool
	SupportExtenders() bool
	GetNamespace() string
	GetName() string
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

func NewWorkloadPreemptor(priority int32, members []*v1.Pod, policy *v1.PreemptionPolicy) Preemptor {
	return &preemptor{
		priority:         priority,
		pods:             members,
		isWorkload:       true,
		preemptionPolicy: nil,
	}
}

func (p *preemptor) Priority() int32 {
	return p.priority
}

func (p *preemptor) IsWorkload() bool {
	return p.isWorkload
}

func (p *preemptor) Members() []*v1.Pod {
	return p.pods
}

func (p *preemptor) IsElgigibleToPreemptOthers() bool {
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

	firstPod := p.pods[0]

	if p.isWorkload {
		return firstPod.Spec.WorkloadRef.PodGroup
	}

	return firstPod.Name
}

type Domain interface {
	Nodes() []fwk.NodeInfo
	GetAllPossibleVictims() []PreemptionUnit
	GetName() string
	Snapshot() Domain
}

type domain struct {
	Domain
	nodeInfoLister fwk.NodeInfoLister
	nodes          []fwk.NodeInfo
	name           string
	podGroupIndex  map[string][]fwk.PodInfo
}

func (d *domain) Nodes() []fwk.NodeInfo {
	return d.nodes
}

func (d *domain) GetAllPossibleVictims() []PreemptionUnit {
	processedPodGroups := make(map[string]bool)
	var allVictims []PreemptionUnit

	for _, node := range d.nodes {
		for _, pi := range node.GetPods() {
			pod := pi.GetPod()

			if ref := pod.Spec.WorkloadRef; ref != nil {
				// Deduplication Check
				if processedPodGroups[ref.PodGroup] {
					continue
				}

				// Get all pods for this gang (Global Lookup via Index)
				gangPods := d.podGroupIndex[ref.PodGroup]

				if len(gangPods) > 0 {
					// Collect NodeInfo for ALL pods in the gang
					var gangNodes []fwk.NodeInfo
					for _, gp := range gangPods {
						nodeName := gp.GetPod().Spec.NodeName
						// Use the global lister to find the node, even if it's not in domain
						if n, err := d.nodeInfoLister.Get(nodeName); err == nil {
							gangNodes = append(gangNodes, n)
						}
					}

					unit := d.newPreemptionUnit(gangPods, corev1helpers.PodPriority(pod), gangNodes)
					allVictims = append(allVictims, unit)
				}

				// Mark as processed so we don't do this again for the next pod in this gang
				processedPodGroups[ref.PodGroup] = true

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
		nodes:          snapshotNodes,
		name:           d.name,
		nodeInfoLister: d.nodeInfoLister,
		podGroupIndex:  d.podGroupIndex,
	}
}

type PreemptionUnit interface {
	Priority() int32
	IsWorkload() bool
	AffectedNodes() map[string]fwk.NodeInfo
	Pods() []fwk.PodInfo
}

type preemptionUnit struct {
	PreemptionUnit
	pods          []fwk.PodInfo
	priority      int32
	affectedNodes map[string]fwk.NodeInfo //TODO: should I store that here?
	isWorkload    bool
}

func (d *domain) newPreemptionUnit(pods []fwk.PodInfo, priority int32, nodeInfos []fwk.NodeInfo) PreemptionUnit {
	nodes := make(map[string]fwk.NodeInfo)
	for _, node := range nodeInfos {
		nodes[node.Node().Name] = node
	}

	return &preemptionUnit{
		affectedNodes: nodes,
		priority:      priority,
		isWorkload:    len(pods) > 1,
		pods:          pods,
	}
}

func (pu *preemptionUnit) Pods() []fwk.PodInfo {
	return pu.pods
}

func (pu *preemptionUnit) Priority() int32 {
	return pu.priority
}

func (pu *preemptionUnit) IsWorkload() bool {
	return pu.isWorkload
}

func (pu *preemptionUnit) AffectedNodes() map[string]fwk.NodeInfo {
	return pu.affectedNodes
}

// Interface is expected to be implemented by different preemption plugins as all those member
// methods might have different behavior compared with the default preemption.
type Interface interface {
	// GetOffsetAndNumCandidates chooses a random offset and calculates the number of candidates that should be
	// shortlisted for dry running preemption.
	GetOffsetAndNumCandidates(nodes int32) (int32, int32)
	// CandidatesToVictimsMap builds a map from the target node to a list of to-be-preempted Pods and the number of PDB violation.
	CandidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims
	// PodEligibleToPreemptOthers returns one bool and one string. The bool indicates whether this pod should be considered for
	// preempting other pods or not. The string includes the reason if this pod isn't eligible.
	PodEligibleToPreemptOthers(ctx context.Context, pod *v1.Pod, nominatedNodeStatus *fwk.Status) (bool, string)
	// SelectVictimsOnDomain finds minimum set of pods on the given domain that should be preempted in order to make enough room
	// for "pods" from preemptor to be scheduled.
	// Note that both `state` and `nodeInfo` are deep copied.
	SelectVictimsOnDomain(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domain Domain, pdbs []*policy.PodDisruptionBudget) ([]*v1.Pod, int, *fwk.Status)
	// OrderedScoreFuncs returns a list of ordered score functions to select preferable node where victims will be preempted.
	// The ordered score functions will be processed one by one if we find more than one node with the highest score.
	// Default score functions will be processed if nil returned here for backwards-compatibility.
	OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64
}

type Evaluator struct {
	PluginName string
	Handler    fwk.Handle
	PodLister  corelisters.PodLister
	PdbLister  policylisters.PodDisruptionBudgetLister

	enableAsyncPreemption bool
	mu                    sync.RWMutex
	// preempting is a set that records the pods that are currently triggering preemption asynchronously,
	// which is used to prevent the pods from entering the scheduling cycle meanwhile.
	preempting sets.Set[types.UID]
	// lastVictimsPendingPreemption is a map that records the victim pods that are currently being preempted for a given preemptor pod,
	// with a condition that the preemptor is waiting for one last victim to be preempted. This is used together with `preempting`
	// to prevent the pods from entering the scheduling cycle while waiting for preemption to complete.
	lastVictimsPendingPreemption map[types.UID]pendingVictim

	// PreemptPod is a function that actually makes API calls to preempt a specific Pod.
	// This is exposed to be replaced during tests.
	PreemptPod func(ctx context.Context, c Candidate, preemptor Preemptor, victim *v1.Pod, pluginName string) error

	Interface
}

type pendingVictim struct {
	namespace string
	name      string
}

func NewEvaluator(pluginName string, fh fwk.Handle, i Interface, enableAsyncPreemption bool) *Evaluator {
	podLister := fh.SharedInformerFactory().Core().V1().Pods().Lister()
	pdbLister := fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister()

	ev := &Evaluator{
		PluginName:                   pluginName,
		Handler:                      fh,
		PodLister:                    podLister,
		PdbLister:                    pdbLister,
		Interface:                    i,
		enableAsyncPreemption:        enableAsyncPreemption,
		preempting:                   sets.New[types.UID](),
		lastVictimsPendingPreemption: make(map[types.UID]pendingVictim),
	}

	// PreemptPod actually makes API calls to preempt a specific Pod.
	//
	// We implement it here directly, rather than creating a separate method like ev.preemptPod(...)
	// to prevent the misuse of the PreemptPod function.
	ev.PreemptPod = func(ctx context.Context, c Candidate, preemptor Preemptor, victim *v1.Pod, pluginName string) error {
		logger := klog.FromContext(ctx)

		representative := preemptor.Members()[0]
		skipAPICall := false
		// If the victim is a WaitingPod, try to preempt it without a delete call (victim will go back to backoff queue).
		// Otherwise we should delete the victim.
		if waitingPod := ev.Handler.GetWaitingPod(victim.UID); waitingPod != nil {
			if waitingPod.Preempt(pluginName, "preempted") {
				logger.V(2).Info("Preemptor pod preempted a waiting pod", "preemptor", klog.KObj(preemptor), "waitingPod", klog.KObj(victim), "domain", c.Name())
				skipAPICall = true
			}
		}
		if !skipAPICall {
			message := fmt.Sprintf("%s: preempting to accommodate a higher priority pod", representative.Spec.SchedulerName)
			if preemptor.IsWorkload() {
				message = fmt.Sprintf("%s: preempting to accommodate a higher priority workload", representative.Spec.SchedulerName)
			}
			condition := &v1.PodCondition{
				Type:               v1.DisruptionTarget,
				ObservedGeneration: apipod.CalculatePodConditionObservedGeneration(&victim.Status, victim.Generation, v1.DisruptionTarget),
				Status:             v1.ConditionTrue,
				Reason:             v1.PodReasonPreemptionByScheduler,
				Message:            message,
			}
			newStatus := victim.Status.DeepCopy()
			updated := apipod.UpdatePodCondition(newStatus, condition)
			if updated {
				if err := util.PatchPodStatus(ctx, ev.Handler.ClientSet(), victim.Name, victim.Namespace, &victim.Status, newStatus); err != nil {
					if !apierrors.IsNotFound(err) {
						logger.Error(err, "Could not add DisruptionTarget condition due to preemption", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim))
						return err
					}
					logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
					return nil
				}
			}
			if err := util.DeletePod(ctx, ev.Handler.ClientSet(), victim); err != nil {
				if !apierrors.IsNotFound(err) {
					logger.Error(err, "Tried to preempted pod", "pod", klog.KObj(victim), "preemptor", klog.KObj(preemptor))
					return err
				}
				logger.V(2).Info("Victim Pod is already deleted", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
				return nil
			}
			logger.V(2).Info("Preemptor Pod preempted victim Pod", "preemptor", klog.KObj(preemptor), "victim", klog.KObj(victim), "node", c.Name())
		}

		ev.Handler.EventRecorder().Eventf(victim, representative, v1.EventTypeNormal, "Preempted", "Preempting", "Preempted by pod %v on node %v", representative.UID, c.Name())

		return nil
	}

	return ev
}

// IsPodRunningPreemption returns true if the pod is currently triggering preemption asynchronously.
func (ev *Evaluator) IsPodRunningPreemption(podUID types.UID) bool {
	ev.mu.RLock()
	defer ev.mu.RUnlock()

	if !ev.preempting.Has(podUID) {
		return false
	}

	victim, ok := ev.lastVictimsPendingPreemption[podUID]
	if !ok {
		// Since pod is in `preempting` but last victim is not registered yet, the async preemption is pending.
		return true
	}
	// Pod is waiting for preemption of one last victim. We can check if the victim has already been deleted.
	victimPod, err := ev.PodLister.Pods(victim.namespace).Get(victim.name)
	if err != nil {
		// Victim already deleted, preemption is done.
		return false
	}
	if victimPod.DeletionTimestamp != nil {
		// Victim deletion has started, preemption is done.
		return false
	}
	// Preemption of the last pod is still ongoing.
	return true
}

// Preempt returns a PostFilterResult carrying suggested nominatedNodeName, along with a Status.
// The semantics of returned <PostFilterResult, Status> varies on different scenarios:
//
//   - <nil, Error>. This denotes it's a transient/rare error that may be self-healed in future cycles.
//
//   - <nil, Unschedulable>. This status is mostly as expected like the preemptor is waiting for the
//     victims to be fully terminated.
//
//   - In both cases above, a nil PostFilterResult is returned to keep the pod's nominatedNodeName unchanged.
//
//   - <non-nil PostFilterResult, Unschedulable>. It indicates the pod cannot be scheduled even with preemption.
//     In this case, a non-nil PostFilterResult is returned and result.NominatingMode instructs how to deal with
//     the nominatedNodeName.
//
//   - <non-nil PostFilterResult, Success>. It's the regular happy path
//     and the non-empty nominatedNodeName will be applied to the preemptor pod.
func (ev *Evaluator) Preempt(ctx context.Context, state fwk.CycleState, preemptor Preemptor, m fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	if !preemptor.IsElgigibleToPreemptOthers() {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Preemptor couldn't preempt other victims because of preemptionPolicy: never")
	}

	for _, pod := range preemptor.Members() {
		// 0) Fetch the latest version of <pod>.
		// It's safe to directly fetch pod here. Because the informer cache has already been
		// initialized when creating the Scheduler obj.
		// However, tests may need to manually initialize the shared pod informer.
		podNamespace, podName := pod.Namespace, pod.Name
		pod, err := ev.PodLister.Pods(pod.Namespace).Get(pod.Name)
		if err != nil {
			logger.Error(err, "Could not get the updated preemptor pod object", "pod", klog.KRef(podNamespace, podName))
			return nil, fwk.AsStatus(err)
		}

		// 1) Ensure that pod related to the preemptor is eligible to preempt other pods.
		nominatedNodeStatus := m.Get(pod.Status.NominatedNodeName)
		if ok, msg := ev.PodEligibleToPreemptOthers(ctx, pod, nominatedNodeStatus); !ok {
			logger.V(5).Info("Pod is not eligible for preemption", "pod", klog.KObj(pod), "reason", msg)
			return nil, fwk.NewStatus(fwk.Unschedulable, msg)
		}
	}

	// 2) Find all preemption candidates.
	allNodes, err := ev.Handler.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	candidates, nodeToStatusMap, err := ev.findCandidates(ctx, state, allNodes, preemptor, m)
	if err != nil && len(candidates) == 0 {
		return nil, fwk.AsStatus(err)
	}

	// Return a FitError only when there are no candidates that fit the pod.
	if len(candidates) == 0 {
		logger.V(2).Info("No preemption candidate is found; preemption is not helpful for scheduling", "pod", klog.KObj(preemptor))
		fitError := &framework.FitError{
			Pod:         preemptor.Members()[0], //TODO: not sure, assuming we have single pod as preemptor for now
			NumAllNodes: len(allNodes),
			Diagnosis: framework.Diagnosis{
				NodeToStatus: nodeToStatusMap,
				// Leave UnschedulablePlugins or PendingPlugins as nil as it won't be used on moving Pods.
			},
		}
		fitError.Diagnosis.NodeToStatus.SetAbsentNodesStatus(fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Preemption is not helpful for scheduling"))
		// Specify nominatedNodeName to clear the pod's nominatedNodeName status, if applicable.
		return framework.NewPostFilterResultWithNominatedNode(""), fwk.NewStatus(fwk.Unschedulable, fitError.Error())
	}

	// 3) Interact with registered Extenders to filter out some candidates if needed.
	candidates, status := ev.callExtenders(logger, preemptor, candidates)
	if !status.IsSuccess() {
		return nil, status
	}

	// 4) Find the best candidate.
	bestCandidate := ev.SelectCandidate(ctx, candidates)
	if bestCandidate == nil || len(bestCandidate.Name()) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no candidate node for preemption")
	}

	logger.V(2).Info("the best candidate for the preemption is determined", "domain", bestCandidate.Name(), "preemptor", klog.KObj(preemptor))

	// 5) Perform preparation work before nominating the selected candidate.
	if ev.enableAsyncPreemption {
		ev.prepareCandidateAsync(bestCandidate, preemptor, ev.PluginName)
	} else {
		if status := ev.prepareCandidate(ctx, bestCandidate, preemptor, ev.PluginName); !status.IsSuccess() {
			return nil, status
		}
	}

	return framework.NewPostFilterResultWithNominatedNode(bestCandidate.Name()), fwk.NewStatus(fwk.Success)
}

// FindCandidates calculates a slice of preemption candidates.
// Each candidate is executable to make the given <pod> schedulable.
func (ev *Evaluator) findCandidates(ctx context.Context, state fwk.CycleState, allNodes []fwk.NodeInfo, preemptor Preemptor, m fwk.NodeToStatusReader) ([]Candidate, *framework.NodeToStatus, error) {
	pod := preemptor.Members()[0]

	if len(allNodes) == 0 {
		return nil, nil, errors.New("no nodes available")
	}
	logger := klog.FromContext(ctx)

	var potentialNodes []fwk.NodeInfo
	if !preemptor.IsWorkload() {
		nodes, err := m.NodesForStatusCode(ev.Handler.SnapshotSharedLister().NodeInfos(), fwk.Unschedulable)
		if err != nil {
			return nil, nil, err
		}
		potentialNodes = nodes
	} else {
		potentialNodes = allNodes
	}

	if len(potentialNodes) == 0 {
		logger.V(3).Info("Preemption will not help schedule pod on any node", "pod", klog.KObj(pod))
		return nil, framework.NewDefaultNodeToStatus(), nil
	}

	pdbs, err := getPodDisruptionBudgets(ev.PdbLister)
	if err != nil {
		return nil, nil, err
	}

	domains := ev.NewDomains(preemptor, potentialNodes)

	offset, candidatesNum := ev.GetOffsetAndNumCandidates(int32(len(domains)))
	return ev.DryRunPreemption(ctx, state, preemptor, domains, pdbs, offset, candidatesNum)
}

// callExtenders calls given <extenders> to select the list of feasible candidates.
// We will only check <candidates> with extenders that support preemption.
// Extenders which do not support preemption may later prevent preemptor from being scheduled on the nominated
// node. In that case, scheduler will find a different host for the preemptor in subsequent scheduling cycles.
func (ev *Evaluator) callExtenders(logger klog.Logger, preemptor Preemptor, candidates []Candidate) ([]Candidate, *fwk.Status) {
	if !preemptor.SupportExtenders() {
		return candidates, nil
	}

	extenders := ev.Handler.Extenders()
	nodeLister := ev.Handler.SnapshotSharedLister().NodeInfos()
	if len(extenders) == 0 {
		return candidates, nil
	}

	// Migrate candidate slice to victimsMap to adapt to the Extender interface.
	// It's only applicable for candidate slice that have unique nominated node name.
	victimsMap := ev.CandidatesToVictimsMap(candidates)
	if len(victimsMap) == 0 {
		return candidates, nil
	}
	for _, extender := range extenders {
		pod := preemptor.Members()[0]
		if !extender.SupportsPreemption() || !extender.IsInterested(pod) {
			continue
		}
		nodeNameToVictims, err := extender.ProcessPreemption(pod, victimsMap, nodeLister)
		if err != nil {
			if extender.IsIgnorable() {
				logger.V(2).Info("Skipped extender as it returned error and has ignorable flag set",
					"extender", extender.Name(), "err", err)
				continue
			}
			return nil, fwk.AsStatus(err)
		}
		// Check if the returned victims are valid.
		for nodeName, victims := range nodeNameToVictims {
			if victims == nil || len(victims.Pods) == 0 {
				if extender.IsIgnorable() {
					delete(nodeNameToVictims, nodeName)
					logger.V(2).Info("Ignored node for which the extender didn't report victims", "node", klog.KRef("", nodeName), "extender", extender.Name())
					continue
				}
				return nil, fwk.AsStatus(fmt.Errorf("expected at least one victim pod on node %q", nodeName))
			}
		}
		// Replace victimsMap with new result after preemption. So the
		// rest of extenders can continue use it as parameter.
		victimsMap = nodeNameToVictims

		// If node list becomes empty, no preemption can happen regardless of other extenders.
		if len(victimsMap) == 0 {
			break
		}
	}

	var newCandidates []Candidate
	for nodeName := range victimsMap {
		newCandidates = append(newCandidates, &candidate{
			victims: victimsMap[nodeName],
			name:    nodeName,
		})
	}
	return newCandidates, nil
}

// SelectCandidate chooses the best-fit candidate from given <candidates> and return it.
// NOTE: This method is exported for easier testing in default preemption.
func (ev *Evaluator) SelectCandidate(ctx context.Context, candidates []Candidate) Candidate {
	logger := klog.FromContext(ctx)

	if len(candidates) == 0 {
		return nil
	}
	if len(candidates) == 1 {
		return candidates[0]
	}

	victimsMap := ev.CandidatesToVictimsMap(candidates)
	scoreFuncs := ev.OrderedScoreFuncs(ctx, victimsMap)
	candidateNode := pickOneNodeForPreemption(logger, victimsMap, scoreFuncs)

	// Same as candidatesToVictimsMap, this logic is not applicable for out-of-tree
	// preemption plugins that exercise different candidates on the same nominated node.
	if victims := victimsMap[candidateNode]; victims != nil {
		return &candidate{
			victims: victims,
			name:    candidateNode,
		}
	}

	// We shouldn't reach here.
	utilruntime.HandleErrorWithContext(ctx, nil, "Unexpected case no candidate was selected", "candidates", candidates)
	// To not break the whole flow, return the first candidate.
	return candidates[0]
}

// prepareCandidate does some preparation work before nominating the selected candidate:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
func (ev *Evaluator) prepareCandidate(ctx context.Context, c Candidate, preemptor Preemptor, pluginName string) *fwk.Status {
	metrics.PreemptionVictims.Observe(float64(len(c.Victims().Pods)))

	fh := ev.Handler
	cs := ev.Handler.ClientSet()

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(ctx)
	errCh := parallelize.NewResultChannel[error]()
	fh.Parallelizer().Until(ctx, len(c.Victims().Pods), func(index int) {
		victim := c.Victims().Pods[index]
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(preemptor), "node", c.Name(), "victim", klog.KObj(victim))
			return
		}
		if err := ev.PreemptPod(ctx, c, preemptor, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}, ev.PluginName)
	if err := errCh.Receive(); err != nil {
		return fwk.AsStatus(err)
	}

	// Lower priority pods nominated to run on this node, may no longer fit on
	// this node. So, we should remove their nomination. Removing their
	// nomination updates these pods and moves them to the active queue. It
	// lets scheduler find another place for them sooner than after waiting for preemption completion.
	nominatedPods := getLowerPriorityNominatedPods(logger, fh, preemptor.Members()[0], c.Name())
	if err := clearNominatedNodeName(ctx, cs, ev.Handler.APICacher(), nominatedPods...); err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Cannot clear 'NominatedNodeName' field")
		// We do not return as this error is not critical.
	}

	return nil
}

// clearNominatedNodeName internally submit a patch request to API server
// to set each pods[*].Status.NominatedNodeName> to "".
func clearNominatedNodeName(ctx context.Context, cs clientset.Interface, apiCacher fwk.APICacher, pods ...*v1.Pod) utilerrors.Aggregate {
	var errs []error
	for _, p := range pods {
		if apiCacher != nil {
			// When API cacher is available, use it to clear the NominatedNodeName.
			_, err := apiCacher.PatchPodStatus(p, nil, &fwk.NominatingInfo{NominatedNodeName: "", NominatingMode: fwk.ModeOverride})
			if err != nil {
				errs = append(errs, err)
			}
		} else {
			if len(p.Status.NominatedNodeName) == 0 {
				continue
			}
			podStatusCopy := p.Status.DeepCopy()
			podStatusCopy.NominatedNodeName = ""
			if err := util.PatchPodStatus(ctx, cs, p.Name, p.Namespace, &p.Status, podStatusCopy); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

// prepareCandidateAsync triggers a goroutine for some preparation work:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
// The Pod won't be retried until the goroutine triggered here completes.
//
// See http://kep.k8s.io/4832 for how the async preemption works.
func (ev *Evaluator) prepareCandidateAsync(c Candidate, preemptor Preemptor, pluginName string) {
	representative := preemptor.Members()[0]

	// Intentionally create a new context, not using a ctx from the scheduling cycle, to create ctx,
	// because this process could continue even after this scheduling cycle finishes.
	ctx, cancel := context.WithCancel(context.Background())
	logger := klog.FromContext(ctx)

	victimPods := make([]*v1.Pod, 0, len(c.Victims().Pods))
	for _, victim := range c.Victims().Pods {
		if victim.DeletionTimestamp != nil {
			// Graceful pod deletion has already started. Sending another API call is unnecessary.
			logger.V(2).Info("Victim Pod is already being deleted, skipping the API call for it", "preemptor", klog.KObj(preemptor), "node", c.Name(), "victim", klog.KObj(victim))
			continue
		}
		victimPods = append(victimPods, victim)
	}
	if len(victimPods) == 0 {
		cancel()
		return
	}

	metrics.PreemptionVictims.Observe(float64(len(c.Victims().Pods)))

	errCh := parallelize.NewResultChannel[error]()
	preemptPod := func(index int) {
		victim := victimPods[index]
		if err := ev.PreemptPod(ctx, c, preemptor, victim, pluginName); err != nil {
			errCh.SendWithCancel(err, cancel)
		}
	}

	ev.mu.Lock()
	ev.preempting.Insert(representative.UID)
	ev.mu.Unlock()

	go func() {
		logger := klog.FromContext(ctx)
		startTime := time.Now()
		result := metrics.GoroutineResultSuccess
		defer metrics.PreemptionGoroutinesDuration.WithLabelValues(result).Observe(metrics.SinceInSeconds(startTime))
		defer metrics.PreemptionGoroutinesExecutionTotal.WithLabelValues(result).Inc()
		defer func() {
			if result == metrics.GoroutineResultError {
				podsToActivate := make(map[string]*v1.Pod)
				for _, p := range preemptor.Members() {
					podsToActivate[p.Name] = p
				}
				// When API call isn't successful, the Pod may get stuck in the unschedulable pod pool in the worst case.
				// So, we should move the Pod to the activeQ.
				ev.Handler.Activate(logger, podsToActivate)
			}
		}()
		defer cancel()
		logger.V(2).Info("Start the preemption asynchronously", "preemptor", klog.KObj(preemptor), "node", c.Name(), "numVictims", len(c.Victims().Pods), "numVictimsToDelete", len(victimPods))

		// Lower priority pods nominated to run on this node, may no longer fit on
		// this node. So, we should remove their nomination. Removing their
		// nomination updates these pods and moves them to the active queue. It
		// lets scheduler find another place for them sooner than after waiting for preemption completion.
		nominatedPods := getLowerPriorityNominatedPods(logger, ev.Handler, representative, c.Name())
		if err := clearNominatedNodeName(ctx, ev.Handler.ClientSet(), ev.Handler.APICacher(), nominatedPods...); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Cannot clear 'NominatedNodeName' field from lower priority pods on the same target node", "node", c.Name())
			result = metrics.GoroutineResultError
			// We do not return as this error is not critical.
		}

		preemptLastVictim := true
		if len(victimPods) > 1 {
			// In order to prevent requesting preemption of the same pod multiple times for the same preemptor,
			// preemptor is marked as "waiting for preemption of a victim" (by adding it to preempting map).
			// We can evict all victims in parallel, but the last one.
			// While deleting the last victim, we want the PreEnqueue check to be able to verify if the eviction
			// is in fact ongoing, or if it has already completed, but the function has not returned yet.
			// In the latter case, PreEnqueue (in `IsPodRunningPreemption`) reads the state of the last victim in
			// `lastVictimsPendingPreemption` and does not block the pod.
			// This helps us avoid the situation where pod removal might be notified to the scheduling queue before
			// the preemptor completes the deletion API calls and is removed from the `preempting` map - that way
			// the preemptor could end up stuck in the unschedulable pool, with all pod removal events being ignored.
			ev.Handler.Parallelizer().Until(ctx, len(victimPods)-1, preemptPod, ev.PluginName)
			if err := errCh.Receive(); err != nil {
				utilruntime.HandleErrorWithContext(ctx, err, "Error occurred during async preemption")
				result = metrics.GoroutineResultError
				preemptLastVictim = false
			}
		}

		// If any of the previous victims failed to be preempted, then we can skip
		// the preemption attempt for the last victim Pod to expedite the preemptor's
		// re-entry to the scheduling cycle.
		if preemptLastVictim {
			lastVictim := victimPods[len(victimPods)-1]
			ev.mu.Lock()
			ev.lastVictimsPendingPreemption[representative.UID] = pendingVictim{namespace: lastVictim.Namespace, name: lastVictim.Name}
			ev.mu.Unlock()

			if err := ev.PreemptPod(ctx, c, preemptor, lastVictim, pluginName); err != nil {
				utilruntime.HandleErrorWithContext(ctx, err, "Error occurred during async preemption of the last victim")
				result = metrics.GoroutineResultError
			}
		}
		ev.mu.Lock()
		ev.preempting.Delete(representative.UID)
		delete(ev.lastVictimsPendingPreemption, representative.UID)
		ev.mu.Unlock()

		logger.V(2).Info("Async Preemption finished completely", "preemptor", klog.KObj(preemptor), "node", c.Name(), "result", result)
	}()
}

func getPodDisruptionBudgets(pdbLister policylisters.PodDisruptionBudgetLister) ([]*policy.PodDisruptionBudget, error) {
	if pdbLister != nil {
		return pdbLister.List(labels.Everything())
	}
	return nil, nil
}

// pickOneNodeForPreemption chooses one node among the given nodes.
// It assumes pods in each map entry are ordered by decreasing priority.
// If the scoreFuns is not empty, It picks a node based on score scoreFuns returns.
// If the scoreFuns is empty,
// It picks a node based on the following criteria:
// 1. A node with minimum number of PDB violations.
// 2. A node with minimum highest priority victim is picked.
// 3. Ties are broken by sum of priorities of all victims.
// 4. If there are still ties, node with the minimum number of victims is picked.
// 5. If there are still ties, node with the latest start time of all highest priority victims is picked.
// 6. If there are still ties, the first such node is picked (sort of randomly).
// The 'minNodes1' and 'minNodes2' are being reused here to save the memory
// allocation and garbage collection time.
func pickOneNodeForPreemption(logger klog.Logger, nodesToVictims map[string]*extenderv1.Victims, scoreFuncs []func(node string) int64) string {
	if len(nodesToVictims) == 0 {
		return ""
	}

	allCandidates := make([]string, 0, len(nodesToVictims))
	for node := range nodesToVictims {
		allCandidates = append(allCandidates, node)
	}

	if len(scoreFuncs) == 0 {
		minNumPDBViolatingScoreFunc := func(node string) int64 {
			// The smaller the NumPDBViolations, the higher the score.
			return -nodesToVictims[node].NumPDBViolations
		}
		minHighestPriorityScoreFunc := func(node string) int64 {
			// highestPodPriority is the highest priority among the victims on this node.
			highestPodPriority := corev1helpers.PodPriority(nodesToVictims[node].Pods[0])
			// The smaller the highestPodPriority, the higher the score.
			return -int64(highestPodPriority)
		}
		minSumPrioritiesScoreFunc := func(node string) int64 {
			var sumPriorities int64
			for _, pod := range nodesToVictims[node].Pods {
				// We add MaxInt32+1 to all priorities to make all of them >= 0. This is
				// needed so that a node with a few pods with negative priority is not
				// picked over a node with a smaller number of pods with the same negative
				// priority (and similar scenarios).
				sumPriorities += int64(corev1helpers.PodPriority(pod)) + int64(math.MaxInt32+1)
			}
			// The smaller the sumPriorities, the higher the score.
			return -sumPriorities
		}
		minNumPodsScoreFunc := func(node string) int64 {
			// The smaller the length of pods, the higher the score.
			return -int64(len(nodesToVictims[node].Pods))
		}
		latestStartTimeScoreFunc := func(node string) int64 {
			// Get the earliest start time of all pods on the current node.
			earliestStartTimeOnNode := util.GetEarliestPodStartTime(nodesToVictims[node])
			if earliestStartTimeOnNode == nil {
				utilruntime.HandleErrorWithLogger(logger, nil, "Unexpected nil earliestStartTime for node", "node", node)
				return int64(math.MinInt64)
			}
			// The bigger the earliestStartTimeOnNode, the higher the score.
			return earliestStartTimeOnNode.UnixNano()
		}

		// Each scoreFunc scores the nodes according to specific rules and keeps the name of the node
		// with the highest score. If and only if the scoreFunc has more than one node with the highest
		// score, we will execute the other scoreFunc in order of precedence.
		scoreFuncs = []func(string) int64{
			// A node with a minimum number of PDB is preferable.
			minNumPDBViolatingScoreFunc,
			// A node with a minimum highest priority victim is preferable.
			minHighestPriorityScoreFunc,
			// A node with the smallest sum of priorities is preferable.
			minSumPrioritiesScoreFunc,
			// A node with the minimum number of pods is preferable.
			minNumPodsScoreFunc,
			// A node with the latest start time of all highest priority victims is preferable.
			latestStartTimeScoreFunc,
			// If there are still ties, then the first Node in the list is selected.
		}
	}

	for _, f := range scoreFuncs {
		selectedNodes := []string{}
		maxScore := int64(math.MinInt64)
		for _, node := range allCandidates {
			score := f(node)
			if score > maxScore {
				maxScore = score
				selectedNodes = []string{}
			}
			if score == maxScore {
				selectedNodes = append(selectedNodes, node)
			}
		}
		if len(selectedNodes) == 1 {
			return selectedNodes[0]
		}
		allCandidates = selectedNodes
	}

	return allCandidates[0]
}

// getLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given node.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PreFilter state per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func getLowerPriorityNominatedPods(logger klog.Logger, pn fwk.PodNominator, pod *v1.Pod, nodeName string) []*v1.Pod {
	podInfos := pn.NominatedPodsForNode(nodeName)

	if len(podInfos) == 0 {
		return nil
	}

	var lowerPriorityPods []*v1.Pod
	podPriority := corev1helpers.PodPriority(pod)
	for _, pi := range podInfos {
		if corev1helpers.PodPriority(pi.GetPod()) < podPriority {
			lowerPriorityPods = append(lowerPriorityPods, pi.GetPod())
		}
	}
	return lowerPriorityPods
}

// DryRunPreemption simulates Preemption logic on <potentialNodes> in parallel,
// returns preemption candidates and a map indicating filtered nodes statuses.
// The number of candidates depends on the constraints defined in the plugin's args. In the returned list of
// candidates, ones that do not violate PDB are preferred over ones that do.
// NOTE: This method is exported for easier testing in default preemption.
func (ev *Evaluator) DryRunPreemption(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domains []Domain,
	pdbs []*policy.PodDisruptionBudget, offset int32, candidatesNum int32) ([]Candidate, *framework.NodeToStatus, error) {

	fh := ev.Handler
	nonViolatingCandidates := newCandidateList(candidatesNum)
	violatingCandidates := newCandidateList(candidatesNum)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	nodeStatuses := framework.NewDefaultNodeToStatus()

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Dry run the preemption", "domainsNumber", len(domains), "pdbsNumber", len(pdbs), "offset", offset, "candidatesNumber", candidatesNum)

	var statusesLock sync.Mutex
	var errs []error
	checkDomain := func(i int) {
		domain := domains[(int(offset)+i)%len(domains)].Snapshot()
		logger.V(5).Info("Check the potential domain for preemption", "domain", domain)

		stateCopy := state.Clone()
		pods, numPDBViolations, status := ev.SelectVictimsOnDomain(ctx, stateCopy, preemptor, domain, pdbs)

		if status.IsSuccess() && len(pods) != 0 {
			victims := extenderv1.Victims{
				Pods:             pods,
				NumPDBViolations: int64(numPDBViolations),
			}
			c := &candidate{
				victims: &victims,
				name:    domain.GetName(),
			}
			if numPDBViolations == 0 {
				nonViolatingCandidates.add(c)
			} else {
				violatingCandidates.add(c)
			}
			nvcSize, vcSize := nonViolatingCandidates.size(), violatingCandidates.size()
			if nvcSize > 0 && nvcSize+vcSize >= candidatesNum {
				cancel()
			}
			return
		}
		if status.IsSuccess() && len(pods) == 0 {
			status = fwk.AsStatus(fmt.Errorf("expected at least one victim pod on domain %q", domain.GetName()))
		}
		statusesLock.Lock()
		if status.Code() == fwk.Error {
			errs = append(errs, status.AsError())
		}
		for _, node := range domain.Nodes() {
			nodeStatuses.Set(node.Node().Name, status)
		}
		statusesLock.Unlock()
	}
	fh.Parallelizer().Until(ctx, len(domains), checkDomain, ev.PluginName)
	return append(nonViolatingCandidates.get(), violatingCandidates.get()...), nodeStatuses, utilerrors.NewAggregate(errs)
}

func (ev *Evaluator) NewDomains(preemptor Preemptor, potentialNodes []fwk.NodeInfo) []Domain {
	// Get the global lister once
	nodeInfoLister := ev.Handler.SnapshotSharedLister().NodeInfos()

	if preemptor.IsWorkload() {
		if len(potentialNodes) == 0 {
			return []Domain{}
		}

		podGroupIndex := ev.buildPodGroupIndex(nodeInfoLister)

		representative := preemptor.Members()[0].Spec.WorkloadRef.PodGroup
		domainName := fmt.Sprintf("Cluster-Scope-%s", representative)

		d := &domain{
			nodes:          potentialNodes,
			name:           domainName,
			nodeInfoLister: nodeInfoLister,
			podGroupIndex:  podGroupIndex, // Pass the cache!
		}
		return []Domain{d}
	}

	// Single Pod Case: Standard Node-by-Node domains
	domains := make([]Domain, 0, len(potentialNodes))

	podGroupIndex := ev.buildPodGroupIndex(nodeInfoLister)

	for _, node := range potentialNodes {
		domains = append(domains, &domain{
			nodes:          []fwk.NodeInfo{node},
			name:           node.Node().Name,
			nodeInfoLister: nodeInfoLister,
			podGroupIndex:  podGroupIndex,
		})
	}

	return domains
}

// Helper to build the cache
func (ev *Evaluator) buildPodGroupIndex(lister fwk.NodeInfoLister) map[string][]fwk.PodInfo {
	index := make(map[string][]fwk.PodInfo)
	allNodes, err := lister.List()
	if err != nil {
		return index
	}

	for _, node := range allNodes {
		for _, pi := range node.GetPods() {
			if ref := pi.GetPod().Spec.WorkloadRef; ref != nil {
				index[ref.PodGroup] = append(index[ref.PodGroup], pi)
			}
		}
	}
	return index
}

func (d *domain) GetName() string {
	return d.name
}
