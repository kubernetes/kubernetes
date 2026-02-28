package interpodaffinity

import (
	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

func (pl *InterPodAffinity) processIncomingPodAffinityNoCache(
	state *preScoreState,
	existingPod fwk.PodInfo,
	existingPodNodeInfo fwk.NodeInfo,
	incomingPod *v1.Pod,
	topoScore scoreMap,
) {
	existingPodNode := existingPodNodeInfo.Node()
	if len(existingPodNode.Labels) == 0 {
		return
	}

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	// Note that the incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
	topoScore.processTerms(state.podInfo.GetPreferredAffinityTerms(), existingPod.GetPod(), nil, existingPodNode, 1)

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	// Note that the incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
	topoScore.processTerms(state.podInfo.GetPreferredAntiAffinityTerms(), existingPod.GetPod(), nil, existingPodNode, -1)
}

func (pl *InterPodAffinity) processExistingPodAffinityNoCache(
	state *preScoreState,
	existingPod fwk.PodInfo,
	existingPodNodeInfo fwk.NodeInfo,
	incomingPod *v1.Pod,
	topoScore scoreMap,
) {
	existingPodNode := existingPodNodeInfo.Node()
	if len(existingPodNode.Labels) == 0 {
		return
	}

	// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the constant <args.hardPodAffinityWeight>
	if pl.args.HardPodAffinityWeight > 0 && len(existingPodNode.Labels) != 0 {
		for _, t := range existingPod.GetRequiredAffinityTerms() {
			topoScore.processTerm(&t, pl.args.HardPodAffinityWeight, incomingPod, state.namespaceLabels, existingPodNode, 1)
		}
	}

	// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTerms(existingPod.GetPreferredAffinityTerms(), incomingPod, state.namespaceLabels, existingPodNode, 1)

	// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
	// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTerms(existingPod.GetPreferredAntiAffinityTerms(), incomingPod, state.namespaceLabels, existingPodNode, -1)
}

func (pl *InterPodAffinity) processIncomingPodAffinity(
	state *preScoreState,
	existingPod fwk.PodInfo,
	existingPodNodeInfo fwk.NodeInfo,
	incomingPod *v1.Pod,
	topoScore scoreMap,
	topoScoreByPod scoreMap,
) {
	existingPodNode := existingPodNodeInfo.Node()
	if len(existingPodNode.Labels) == 0 {
		return
	}

	// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPods>`s node by the term`s weight.
	// Note that the incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
	topoScore.processTermsWithCache(state.podInfo.GetPreferredAffinityTerms(), existingPod.GetPod(), nil, existingPodNode, 1, topoScoreByPod)

	// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
	// decrement <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>`s node by the term`s weight.
	// Note that the incoming pod's terms have the namespaceSelector merged into the namespaces, and so
	// here we don't lookup the existing pod's namespace labels, hence passing nil for nsLabels.
	topoScore.processTermsWithCache(state.podInfo.GetPreferredAntiAffinityTerms(), existingPod.GetPod(), nil, existingPodNode, -1, topoScoreByPod)
}

func (pl *InterPodAffinity) processExistingPodAffinity(
	state *preScoreState,
	existingPod fwk.PodInfo,
	existingPodNodeInfo fwk.NodeInfo,
	incomingPod *v1.Pod,
	topoScore scoreMap,
	topoScoreByPod scoreMap,
) {
	existingPodNode := existingPodNodeInfo.Node()
	if len(existingPodNode.Labels) == 0 {
		return
	}

	// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the constant <args.hardPodAffinityWeight>
	if pl.args.HardPodAffinityWeight > 0 && len(existingPodNode.Labels) != 0 {
		for _, t := range existingPod.GetRequiredAffinityTerms() {
			topoScore.processTermWithCache(&t, pl.args.HardPodAffinityWeight, incomingPod, state.namespaceLabels, existingPodNode, 1, topoScoreByPod)
		}
	}

	// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
	// increment <p.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTermsWithCache(existingPod.GetPreferredAffinityTerms(), incomingPod, state.namespaceLabels, existingPodNode, 1, topoScoreByPod)

	// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
	// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
	// value as that of <existingPod>'s node by the term's weight.
	topoScore.processTermsWithCache(existingPod.GetPreferredAntiAffinityTerms(), incomingPod, state.namespaceLabels, existingPodNode, -1, topoScoreByPod)
}
