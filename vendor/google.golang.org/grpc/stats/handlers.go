/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package stats

import (
	"net"
	"sync/atomic"

	"golang.org/x/net/context"
	"google.golang.org/grpc/grpclog"
)

// ConnTagInfo defines the relevant information needed by connection context tagger.
type ConnTagInfo struct {
	// RemoteAddr is the remote address of the corresponding connection.
	RemoteAddr net.Addr
	// LocalAddr is the local address of the corresponding connection.
	LocalAddr net.Addr
	// TODO add QOS related fields.
}

// RPCTagInfo defines the relevant information needed by RPC context tagger.
type RPCTagInfo struct {
	// FullMethodName is the RPC method in the format of /package.service/method.
	FullMethodName string
}

var (
	on          = new(int32)
	rpcHandler  func(context.Context, RPCStats)
	connHandler func(context.Context, ConnStats)
	connTagger  func(context.Context, *ConnTagInfo) context.Context
	rpcTagger   func(context.Context, *RPCTagInfo) context.Context
)

// HandleRPC processes the RPC stats using the rpc handler registered by the user.
func HandleRPC(ctx context.Context, s RPCStats) {
	if rpcHandler == nil {
		return
	}
	rpcHandler(ctx, s)
}

// RegisterRPCHandler registers the user handler function for RPC stats processing.
// It should be called only once. The later call will overwrite the former value if it is called multiple times.
// This handler function will be called to process the rpc stats.
func RegisterRPCHandler(f func(context.Context, RPCStats)) {
	rpcHandler = f
}

// HandleConn processes the stats using the call back function registered by user.
func HandleConn(ctx context.Context, s ConnStats) {
	if connHandler == nil {
		return
	}
	connHandler(ctx, s)
}

// RegisterConnHandler registers the user handler function for conn stats.
// It should be called only once. The later call will overwrite the former value if it is called multiple times.
// This handler function will be called to process the conn stats.
func RegisterConnHandler(f func(context.Context, ConnStats)) {
	connHandler = f
}

// TagConn calls user registered connection context tagger.
func TagConn(ctx context.Context, info *ConnTagInfo) context.Context {
	if connTagger == nil {
		return ctx
	}
	return connTagger(ctx, info)
}

// RegisterConnTagger registers the user connection context tagger function.
// The connection context tagger can attach some information to the given context.
// The returned context will be used for stats handling.
// For conn stats handling, the context used in connHandler for this
// connection will be derived from the context returned.
// For RPC stats handling,
//  - On server side, the context used in rpcHandler for all RPCs on this
// connection will be derived from the context returned.
//  - On client side, the context is not derived from the context returned.
func RegisterConnTagger(t func(context.Context, *ConnTagInfo) context.Context) {
	connTagger = t
}

// TagRPC calls the user registered RPC context tagger.
func TagRPC(ctx context.Context, info *RPCTagInfo) context.Context {
	if rpcTagger == nil {
		return ctx
	}
	return rpcTagger(ctx, info)
}

// RegisterRPCTagger registers the user RPC context tagger function.
// The RPC context tagger can attach some information to the given context.
// The context used in stats rpcHandler for this RPC will be derived from the
// context returned.
func RegisterRPCTagger(t func(context.Context, *RPCTagInfo) context.Context) {
	rpcTagger = t
}

// Start starts the stats collection and processing if there is a registered stats handle.
func Start() {
	if rpcHandler == nil && connHandler == nil {
		grpclog.Println("rpcHandler and connHandler are both nil when starting stats. Stats is not started")
		return
	}
	atomic.StoreInt32(on, 1)
}

// Stop stops the stats collection and processing.
// Stop does not unregister the handlers.
func Stop() {
	atomic.StoreInt32(on, 0)
}

// On indicates whether the stats collection and processing is on.
func On() bool {
	return atomic.CompareAndSwapInt32(on, 1, 1)
}
