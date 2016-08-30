package consul

import (
	"fmt"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/structs"
)

// KVS endpoint is used to manipulate the Key-Value store
type KVS struct {
	srv *Server
}

// Apply is used to apply a KVS request to the data store. This should
// only be used for operations that modify the data
func (k *KVS) Apply(args *structs.KVSRequest, reply *bool) error {
	if done, err := k.srv.forward("KVS.Apply", args, args, reply); done {
		return err
	}
	defer metrics.MeasureSince([]string{"consul", "kvs", "apply"}, time.Now())

	// Verify the args
	if args.DirEnt.Key == "" && args.Op != structs.KVSDeleteTree {
		return fmt.Errorf("Must provide key")
	}

	// Apply the ACL policy if any
	acl, err := k.srv.resolveToken(args.Token)
	if err != nil {
		return err
	} else if acl != nil {
		switch args.Op {
		case structs.KVSDeleteTree:
			if !acl.KeyWritePrefix(args.DirEnt.Key) {
				return permissionDeniedErr
			}
		default:
			if !acl.KeyWrite(args.DirEnt.Key) {
				return permissionDeniedErr
			}
		}
	}

	// If this is a lock, we must check for a lock-delay. Since lock-delay
	// is based on wall-time, each peer expire the lock-delay at a slightly
	// different time. This means the enforcement of lock-delay cannot be done
	// after the raft log is committed as it would lead to inconsistent FSMs.
	// Instead, the lock-delay must be enforced before commit. This means that
	// only the wall-time of the leader node is used, preventing any inconsistencies.
	if args.Op == structs.KVSLock {
		state := k.srv.fsm.State()
		expires := state.KVSLockDelay(args.DirEnt.Key)
		if expires.After(time.Now()) {
			k.srv.logger.Printf("[WARN] consul.kvs: Rejecting lock of %s due to lock-delay until %v",
				args.DirEnt.Key, expires)
			*reply = false
			return nil
		}
	}

	// Apply the update
	resp, err := k.srv.raftApply(structs.KVSRequestType, args)
	if err != nil {
		k.srv.logger.Printf("[ERR] consul.kvs: Apply failed: %v", err)
		return err
	}
	if respErr, ok := resp.(error); ok {
		return respErr
	}

	// Check if the return type is a bool
	if respBool, ok := resp.(bool); ok {
		*reply = respBool
	}
	return nil
}

// Get is used to lookup a single key
func (k *KVS) Get(args *structs.KeyRequest, reply *structs.IndexedDirEntries) error {
	if done, err := k.srv.forward("KVS.Get", args, args, reply); done {
		return err
	}

	acl, err := k.srv.resolveToken(args.Token)
	if err != nil {
		return err
	}

	// Get the local state
	state := k.srv.fsm.State()
	return k.srv.blockingRPC(
		&args.QueryOptions,
		&reply.QueryMeta,
		state.GetKVSWatch(args.Key),
		func() error {
			index, ent, err := state.KVSGet(args.Key)
			if err != nil {
				return err
			}
			if acl != nil && !acl.KeyRead(args.Key) {
				ent = nil
			}
			if ent == nil {
				// Must provide non-zero index to prevent blocking
				// Index 1 is impossible anyways (due to Raft internals)
				if index == 0 {
					reply.Index = 1
				} else {
					reply.Index = index
				}
				reply.Entries = nil
			} else {
				reply.Index = ent.ModifyIndex
				reply.Entries = structs.DirEntries{ent}
			}
			return nil
		})
}

// List is used to list all keys with a given prefix
func (k *KVS) List(args *structs.KeyRequest, reply *structs.IndexedDirEntries) error {
	if done, err := k.srv.forward("KVS.List", args, args, reply); done {
		return err
	}

	acl, err := k.srv.resolveToken(args.Token)
	if err != nil {
		return err
	}

	// Get the local state
	state := k.srv.fsm.State()
	return k.srv.blockingRPC(
		&args.QueryOptions,
		&reply.QueryMeta,
		state.GetKVSWatch(args.Key),
		func() error {
			index, ent, err := state.KVSList(args.Key)
			if err != nil {
				return err
			}
			if acl != nil {
				ent = FilterDirEnt(acl, ent)
			}

			if len(ent) == 0 {
				// Must provide non-zero index to prevent blocking
				// Index 1 is impossible anyways (due to Raft internals)
				if index == 0 {
					reply.Index = 1
				} else {
					reply.Index = index
				}
				reply.Entries = nil
			} else {
				reply.Index = index
				reply.Entries = ent
			}
			return nil
		})
}

// ListKeys is used to list all keys with a given prefix to a separator
func (k *KVS) ListKeys(args *structs.KeyListRequest, reply *structs.IndexedKeyList) error {
	if done, err := k.srv.forward("KVS.ListKeys", args, args, reply); done {
		return err
	}

	acl, err := k.srv.resolveToken(args.Token)
	if err != nil {
		return err
	}

	// Get the local state
	state := k.srv.fsm.State()
	return k.srv.blockingRPC(
		&args.QueryOptions,
		&reply.QueryMeta,
		state.GetKVSWatch(args.Prefix),
		func() error {
			index, keys, err := state.KVSListKeys(args.Prefix, args.Seperator)
			if err != nil {
				return err
			}

			// Must provide non-zero index to prevent blocking
			// Index 1 is impossible anyways (due to Raft internals)
			if index == 0 {
				reply.Index = 1
			} else {
				reply.Index = index
			}

			if acl != nil {
				keys = FilterKeys(acl, keys)
			}
			reply.Keys = keys
			return nil
		})
}
