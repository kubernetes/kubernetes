package consul

import (
	"fmt"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/serf/serf"
)

// Internal endpoint is used to query the miscellaneous info that
// does not necessarily fit into the other systems. It is also
// used to hold undocumented APIs that users should not rely on.
type Internal struct {
	srv *Server
}

// NodeInfo is used to retrieve information about a specific node.
func (m *Internal) NodeInfo(args *structs.NodeSpecificRequest,
	reply *structs.IndexedNodeDump) error {
	if done, err := m.srv.forward("Internal.NodeInfo", args, args, reply); done {
		return err
	}

	// Get the node info
	state := m.srv.fsm.State()
	return m.srv.blockingRPC(
		&args.QueryOptions,
		&reply.QueryMeta,
		state.GetQueryWatch("NodeInfo"),
		func() error {
			index, dump, err := state.NodeInfo(args.Node)
			if err != nil {
				return err
			}

			reply.Index, reply.Dump = index, dump
			return m.srv.filterACL(args.Token, reply)
		})
}

// NodeDump is used to generate information about all of the nodes.
func (m *Internal) NodeDump(args *structs.DCSpecificRequest,
	reply *structs.IndexedNodeDump) error {
	if done, err := m.srv.forward("Internal.NodeDump", args, args, reply); done {
		return err
	}

	// Get all the node info
	state := m.srv.fsm.State()
	return m.srv.blockingRPC(
		&args.QueryOptions,
		&reply.QueryMeta,
		state.GetQueryWatch("NodeDump"),
		func() error {
			index, dump, err := state.NodeDump()
			if err != nil {
				return err
			}

			reply.Index, reply.Dump = index, dump
			return m.srv.filterACL(args.Token, reply)
		})
}

// EventFire is a bit of an odd endpoint, but it allows for a cross-DC RPC
// call to fire an event. The primary use case is to enable user events being
// triggered in a remote DC.
func (m *Internal) EventFire(args *structs.EventFireRequest,
	reply *structs.EventFireResponse) error {
	if done, err := m.srv.forward("Internal.EventFire", args, args, reply); done {
		return err
	}

	// Check ACLs
	acl, err := m.srv.resolveToken(args.Token)
	if err != nil {
		return err
	}

	if acl != nil && !acl.EventWrite(args.Name) {
		m.srv.logger.Printf("[WARN] consul: user event %q blocked by ACLs", args.Name)
		return permissionDeniedErr
	}

	// Set the query meta data
	m.srv.setQueryMeta(&reply.QueryMeta)

	// Add the consul prefix to the event name
	eventName := userEventName(args.Name)

	// Fire the event
	return m.srv.serfLAN.UserEvent(eventName, args.Payload, false)
}

// KeyringOperation will query the WAN and LAN gossip keyrings of all nodes.
func (m *Internal) KeyringOperation(
	args *structs.KeyringRequest,
	reply *structs.KeyringResponses) error {

	// Check ACLs
	acl, err := m.srv.resolveToken(args.Token)
	if err != nil {
		return err
	}
	if acl != nil {
		switch args.Operation {
		case structs.KeyringList:
			if !acl.KeyringRead() {
				return fmt.Errorf("Reading keyring denied by ACLs")
			}
		case structs.KeyringInstall:
			fallthrough
		case structs.KeyringUse:
			fallthrough
		case structs.KeyringRemove:
			if !acl.KeyringWrite() {
				return fmt.Errorf("Modifying keyring denied due to ACLs")
			}
		default:
			panic("Invalid keyring operation")
		}
	}

	// Only perform WAN keyring querying and RPC forwarding once
	if !args.Forwarded {
		args.Forwarded = true
		m.executeKeyringOp(args, reply, true)
		return m.srv.globalRPC("Internal.KeyringOperation", args, reply)
	}

	// Query the LAN keyring of this node's DC
	m.executeKeyringOp(args, reply, false)
	return nil
}

// executeKeyringOp executes the appropriate keyring-related function based on
// the type of keyring operation in the request. It takes the KeyManager as an
// argument, so it can handle any operation for either LAN or WAN pools.
func (m *Internal) executeKeyringOp(
	args *structs.KeyringRequest,
	reply *structs.KeyringResponses,
	wan bool) {

	var serfResp *serf.KeyResponse
	var err error
	var mgr *serf.KeyManager

	if wan {
		mgr = m.srv.KeyManagerWAN()
	} else {
		mgr = m.srv.KeyManagerLAN()
	}

	switch args.Operation {
	case structs.KeyringList:
		serfResp, err = mgr.ListKeys()
	case structs.KeyringInstall:
		serfResp, err = mgr.InstallKey(args.Key)
	case structs.KeyringUse:
		serfResp, err = mgr.UseKey(args.Key)
	case structs.KeyringRemove:
		serfResp, err = mgr.RemoveKey(args.Key)
	}

	errStr := ""
	if err != nil {
		errStr = err.Error()
	}

	reply.Responses = append(reply.Responses, &structs.KeyringResponse{
		WAN:        wan,
		Datacenter: m.srv.config.Datacenter,
		Messages:   serfResp.Messages,
		Keys:       serfResp.Keys,
		NumNodes:   serfResp.NumNodes,
		Error:      errStr,
	})
}
