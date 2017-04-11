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

package main

import (
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/benchmark"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
)

var (
	// File path related to google.golang.org/grpc.
	certFile = "benchmark/server/testdata/server1.pem"
	keyFile  = "benchmark/server/testdata/server1.key"
)

type benchmarkServer struct {
	port          int
	cores         int
	closeFunc     func()
	mu            sync.RWMutex
	lastResetTime time.Time
}

func printServerConfig(config *testpb.ServerConfig) {
	// Some config options are ignored:
	// - server type:
	//     will always start sync server
	// - async server threads
	// - core list
	grpclog.Printf(" * server type: %v (ignored, always starts sync server)", config.ServerType)
	grpclog.Printf(" * async server threads: %v (ignored)", config.AsyncServerThreads)
	// TODO: use cores specified by CoreList when setting list of cores is supported in go.
	grpclog.Printf(" * core list: %v (ignored)", config.CoreList)

	grpclog.Printf(" - security params: %v", config.SecurityParams)
	grpclog.Printf(" - core limit: %v", config.CoreLimit)
	grpclog.Printf(" - port: %v", config.Port)
	grpclog.Printf(" - payload config: %v", config.PayloadConfig)
}

func startBenchmarkServer(config *testpb.ServerConfig, serverPort int) (*benchmarkServer, error) {
	printServerConfig(config)

	// Use all cpu cores available on machine by default.
	// TODO: Revisit this for the optimal default setup.
	numOfCores := runtime.NumCPU()
	if config.CoreLimit > 0 {
		numOfCores = int(config.CoreLimit)
	}
	runtime.GOMAXPROCS(numOfCores)

	var opts []grpc.ServerOption

	// Sanity check for server type.
	switch config.ServerType {
	case testpb.ServerType_SYNC_SERVER:
	case testpb.ServerType_ASYNC_SERVER:
	case testpb.ServerType_ASYNC_GENERIC_SERVER:
	default:
		return nil, grpc.Errorf(codes.InvalidArgument, "unknow server type: %v", config.ServerType)
	}

	// Set security options.
	if config.SecurityParams != nil {
		creds, err := credentials.NewServerTLSFromFile(abs(certFile), abs(keyFile))
		if err != nil {
			grpclog.Fatalf("failed to generate credentials %v", err)
		}
		opts = append(opts, grpc.Creds(creds))
	}

	// Priority: config.Port > serverPort > default (0).
	port := int(config.Port)
	if port == 0 {
		port = serverPort
	}

	// Create different benchmark server according to config.
	var (
		addr      string
		closeFunc func()
		err       error
	)
	if config.PayloadConfig != nil {
		switch payload := config.PayloadConfig.Payload.(type) {
		case *testpb.PayloadConfig_BytebufParams:
			opts = append(opts, grpc.CustomCodec(byteBufCodec{}))
			addr, closeFunc = benchmark.StartServer(benchmark.ServerInfo{
				Addr:     ":" + strconv.Itoa(port),
				Type:     "bytebuf",
				Metadata: payload.BytebufParams.RespSize,
			}, opts...)
		case *testpb.PayloadConfig_SimpleParams:
			addr, closeFunc = benchmark.StartServer(benchmark.ServerInfo{
				Addr: ":" + strconv.Itoa(port),
				Type: "protobuf",
			}, opts...)
		case *testpb.PayloadConfig_ComplexParams:
			return nil, grpc.Errorf(codes.Unimplemented, "unsupported payload config: %v", config.PayloadConfig)
		default:
			return nil, grpc.Errorf(codes.InvalidArgument, "unknow payload config: %v", config.PayloadConfig)
		}
	} else {
		// Start protobuf server if payload config is nil.
		addr, closeFunc = benchmark.StartServer(benchmark.ServerInfo{
			Addr: ":" + strconv.Itoa(port),
			Type: "protobuf",
		}, opts...)
	}

	grpclog.Printf("benchmark server listening at %v", addr)
	addrSplitted := strings.Split(addr, ":")
	p, err := strconv.Atoi(addrSplitted[len(addrSplitted)-1])
	if err != nil {
		grpclog.Fatalf("failed to get port number from server address: %v", err)
	}

	return &benchmarkServer{port: p, cores: numOfCores, closeFunc: closeFunc, lastResetTime: time.Now()}, nil
}

// getStats returns the stats for benchmark server.
// It resets lastResetTime if argument reset is true.
func (bs *benchmarkServer) getStats(reset bool) *testpb.ServerStats {
	// TODO wall time, sys time, user time.
	bs.mu.RLock()
	defer bs.mu.RUnlock()
	timeElapsed := time.Since(bs.lastResetTime).Seconds()
	if reset {
		bs.lastResetTime = time.Now()
	}
	return &testpb.ServerStats{TimeElapsed: timeElapsed, TimeUser: 0, TimeSystem: 0}
}
