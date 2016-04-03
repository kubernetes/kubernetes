/*
Copyright 2012 Google Inc.

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

// peers.go defines how processes find and communicate with their peers.

package groupcache

import (
	pb "github.com/golang/groupcache/groupcachepb"
)

// Context is an opaque value passed through calls to the
// ProtoGetter. It may be nil if your ProtoGetter implementation does
// not require a context.
type Context interface{}

// ProtoGetter is the interface that must be implemented by a peer.
type ProtoGetter interface {
	Get(context Context, in *pb.GetRequest, out *pb.GetResponse) error
}

// PeerPicker is the interface that must be implemented to locate
// the peer that owns a specific key.
type PeerPicker interface {
	// PickPeer returns the peer that owns the specific key
	// and true to indicate that a remote peer was nominated.
	// It returns nil, false if the key owner is the current peer.
	PickPeer(key string) (peer ProtoGetter, ok bool)
}

// NoPeers is an implementation of PeerPicker that never finds a peer.
type NoPeers struct{}

func (NoPeers) PickPeer(key string) (peer ProtoGetter, ok bool) { return }

var (
	portPicker func() PeerPicker
)

// RegisterPeerPicker registers the peer initialization function.
// It is called once, when the first group is created.
func RegisterPeerPicker(fn func() PeerPicker) {
	if portPicker != nil {
		panic("RegisterPeerPicker called more than once")
	}
	portPicker = fn
}

func getPeers() PeerPicker {
	if portPicker == nil {
		return NoPeers{}
	}
	pk := portPicker()
	if pk == nil {
		pk = NoPeers{}
	}
	return pk
}
