/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package service defines methods to register a gRPC client/service for a
// profiling service that is exposed in the same server. This service can be
// queried by a client to remotely manage the gRPC profiling behaviour of an
// application.
//
// This package and all its methods are EXPERIMENTAL.
package service

//go:generate protoc --go_out=plugins=grpc,paths=source_relative:../proto -I../proto ../proto/service.proto

import (
	"context"
	"errors"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/profiling"
	ppb "google.golang.org/grpc/profiling/proto"
)

// ProfilingConfig defines configuration options for the Init method.
type ProfilingConfig struct {
	// Setting this to true will enable profiling.
	Enabled bool

	// Profiling uses a circular buffer (ring buffer) to store statistics for
	// only the last few RPCs so that profiling stats do not grow unbounded. This
	// parameter defines the upper limit on the number of RPCs for which
	// statistics should be stored at any given time. An average RPC requires
	// approximately 2-3 KiB of memory for profiling-related statistics, so
	// choose an appropriate number based on the amount of memory you can afford.
	StreamStatsSize uint32

	// To expose the profiling service and its methods, a *grpc.Server must be
	// provided.
	Server *grpc.Server
}

var errorNilServer = errors.New("profiling: no grpc.Server provided")

// Init takes a *ProfilingConfig to initialize profiling (turned on/off
// depending on the value set in pc.Enabled) and register the profiling service
// in the server provided in pc.Server.
func Init(pc *ProfilingConfig) error {
	if pc.Server == nil {
		return errorNilServer
	}

	if err := profiling.InitStats(pc.StreamStatsSize); err != nil {
		return err
	}

	ppb.RegisterProfilingServer(pc.Server, getProfilingServerInstance())

	// Do this last after everything has been initialized and allocated.
	profiling.Enable(pc.Enabled)

	return nil
}

type profilingServer struct {
	drainMutex sync.Mutex
}

var profilingServerInstance *profilingServer
var profilingServerOnce sync.Once

// getProfilingServerInstance creates and returns a singleton instance of
// profilingServer. Only one instance of profilingServer is created to use a
// shared mutex across all profilingServer instances.
func getProfilingServerInstance() *profilingServer {
	profilingServerOnce.Do(func() {
		profilingServerInstance = &profilingServer{}
	})

	return profilingServerInstance
}

func (s *profilingServer) Enable(ctx context.Context, req *ppb.EnableRequest) (*ppb.EnableResponse, error) {
	if req.Enabled {
		grpclog.Infof("profilingServer: Enable: enabling profiling")
	} else {
		grpclog.Infof("profilingServer: Enable: disabling profiling")
	}
	profiling.Enable(req.Enabled)

	return &ppb.EnableResponse{}, nil
}

func timerToProtoTimer(timer *profiling.Timer) *ppb.Timer {
	return &ppb.Timer{
		Tags:      timer.Tags,
		BeginSec:  timer.Begin.Unix(),
		BeginNsec: int32(timer.Begin.Nanosecond()),
		EndSec:    timer.End.Unix(),
		EndNsec:   int32(timer.End.Nanosecond()),
		GoId:      timer.GoID,
	}
}

func statToProtoStat(stat *profiling.Stat) *ppb.Stat {
	protoStat := &ppb.Stat{
		Tags:     stat.Tags,
		Timers:   make([]*ppb.Timer, 0, len(stat.Timers)),
		Metadata: stat.Metadata,
	}
	for _, t := range stat.Timers {
		protoStat.Timers = append(protoStat.Timers, timerToProtoTimer(t))
	}
	return protoStat
}

func (s *profilingServer) GetStreamStats(ctx context.Context, req *ppb.GetStreamStatsRequest) (*ppb.GetStreamStatsResponse, error) {
	// Since the drain operation is destructive, only one client request should
	// be served at a time.
	grpclog.Infof("profilingServer: GetStreamStats: processing request")
	s.drainMutex.Lock()
	results := profiling.StreamStats.Drain()
	s.drainMutex.Unlock()

	grpclog.Infof("profilingServer: GetStreamStats: returning %v records", len(results))
	streamStats := make([]*ppb.Stat, 0)
	for _, stat := range results {
		streamStats = append(streamStats, statToProtoStat(stat.(*profiling.Stat)))
	}
	return &ppb.GetStreamStatsResponse{StreamStats: streamStats}, nil
}
