package benchmark

// High Level Configuration for all predicates and priorities.
type schedulerPerfConfig struct {
	NodeAffinity *nodeAffinity
}

// nodeAffinity priority configuration details.
type nodeAffinity struct {
	numGroups       int    // number of Node-Pod sets with Pods NodeAffinity matching given Nodes.
	nodeAffinityKey string // Node Selection Key.
}
