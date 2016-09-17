package agent

import (
	"fmt"

	"github.com/hashicorp/consul/consul/structs"
)

// translateAddress is used to provide the final, translated address for a node,
// depending on how the agent and the other node are configured. The dc
// parameter is the dc the datacenter this node is from.
func translateAddress(config *Config, dc string, addr string, taggedAddresses map[string]string) string {
	if config.TranslateWanAddrs && (config.Datacenter != dc) {
		wanAddr := taggedAddresses["wan"]
		if wanAddr != "" {
			addr = wanAddr
		}
	}
	return addr
}

// translateAddresses translates addresses in the given structure into the
// final, translated address, depending on how the agent and the other node are
// configured. The dc parameter is the datacenter this structure is from.
func translateAddresses(config *Config, dc string, subj interface{}) {
	// CAUTION - SUBTLE! An agent running on a server can, in some cases,
	// return pointers directly into the immutable state store for
	// performance (it's via the in-memory RPC mechanism). It's never safe
	// to modify those values, so we short circuit here so that we never
	// update any structures that are from our own datacenter. This works
	// for address translation because we *never* need to translate local
	// addresses, but this is super subtle, so we've piped all the in-place
	// address translation into this function which makes sure this check is
	// done. This also happens to skip looking at any of the incoming
	// structure for the common case of not needing to translate, so it will
	// skip a lot of work if no translation needs to be done.
	if !config.TranslateWanAddrs || (config.Datacenter == dc) {
		return
	}

	// Translate addresses in-place, subject to the condition checked above
	// which ensures this is safe to do since we are operating on a local
	// copy of the data.
	switch v := subj.(type) {
	case structs.CheckServiceNodes:
		for _, entry := range v {
			entry.Node.Address = translateAddress(config, dc,
				entry.Node.Address, entry.Node.TaggedAddresses)
		}
	case *structs.Node:
		v.Address = translateAddress(config, dc,
			v.Address, v.TaggedAddresses)
	case structs.Nodes:
		for _, node := range v {
			node.Address = translateAddress(config, dc,
				node.Address, node.TaggedAddresses)
		}
	case structs.ServiceNodes:
		for _, entry := range v {
			entry.Address = translateAddress(config, dc,
				entry.Address, entry.TaggedAddresses)
		}
	default:
		panic(fmt.Errorf("Unhandled type passed to address translator: %#v", subj))

	}
}
