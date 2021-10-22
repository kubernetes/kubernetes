/*
 *
 * Copyright 2017 gRPC authors.
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

/*
Package main provides benchmark with setting flags.

An example to run some benchmarks with profiling enabled:

go run benchmark/benchmain/main.go -benchtime=10s -workloads=all \
  -compression=gzip -maxConcurrentCalls=1 -trace=off \
  -reqSizeBytes=1,1048576 -respSizeBytes=1,1048576 -networkMode=Local \
  -cpuProfile=cpuProf -memProfile=memProf -memProfileRate=10000 -resultFile=result

As a suggestion, when creating a branch, you can run this benchmark and save the result
file "-resultFile=basePerf", and later when you at the middle of the work or finish the
work, you can get the benchmark result and compare it with the base anytime.

Assume there are two result files names as "basePerf" and "curPerf" created by adding
-resultFile=basePerf and -resultFile=curPerf.
	To format the curPerf, run:
  	go run benchmark/benchresult/main.go curPerf
	To observe how the performance changes based on a base result, run:
  	go run benchmark/benchresult/main.go basePerf curPerf
*/
package main

import (
	"context"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"os"
	"reflect"
	"runtime"
	"runtime/pprof"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	bm "google.golang.org/grpc/benchmark"
	"google.golang.org/grpc/benchmark/flags"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/benchmark/latency"
	"google.golang.org/grpc/benchmark/stats"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/test/bufconn"
)

var (
	workloads = flags.StringWithAllowedValues("workloads", workloadsAll,
		fmt.Sprintf("Workloads to execute - One of: %v", strings.Join(allWorkloads, ", ")), allWorkloads)
	traceMode = flags.StringWithAllowedValues("trace", toggleModeOff,
		fmt.Sprintf("Trace mode - One of: %v", strings.Join(allToggleModes, ", ")), allToggleModes)
	preloaderMode = flags.StringWithAllowedValues("preloader", toggleModeOff,
		fmt.Sprintf("Preloader mode - One of: %v", strings.Join(allToggleModes, ", ")), allToggleModes)
	channelzOn = flags.StringWithAllowedValues("channelz", toggleModeOff,
		fmt.Sprintf("Channelz mode - One of: %v", strings.Join(allToggleModes, ", ")), allToggleModes)
	compressorMode = flags.StringWithAllowedValues("compression", compModeOff,
		fmt.Sprintf("Compression mode - One of: %v", strings.Join(allCompModes, ", ")), allCompModes)
	networkMode = flags.StringWithAllowedValues("networkMode", networkModeNone,
		"Network mode includes LAN, WAN, Local and Longhaul", allNetworkModes)
	readLatency           = flags.DurationSlice("latency", defaultReadLatency, "Simulated one-way network latency - may be a comma-separated list")
	readKbps              = flags.IntSlice("kbps", defaultReadKbps, "Simulated network throughput (in kbps) - may be a comma-separated list")
	readMTU               = flags.IntSlice("mtu", defaultReadMTU, "Simulated network MTU (Maximum Transmission Unit) - may be a comma-separated list")
	maxConcurrentCalls    = flags.IntSlice("maxConcurrentCalls", defaultMaxConcurrentCalls, "Number of concurrent RPCs during benchmarks")
	readReqSizeBytes      = flags.IntSlice("reqSizeBytes", nil, "Request size in bytes - may be a comma-separated list")
	readRespSizeBytes     = flags.IntSlice("respSizeBytes", nil, "Response size in bytes - may be a comma-separated list")
	reqPayloadCurveFiles  = flags.StringSlice("reqPayloadCurveFiles", nil, "comma-separated list of CSV files describing the shape a random distribution of request payload sizes")
	respPayloadCurveFiles = flags.StringSlice("respPayloadCurveFiles", nil, "comma-separated list of CSV files describing the shape a random distribution of response payload sizes")
	benchTime             = flag.Duration("benchtime", time.Second, "Configures the amount of time to run each benchmark")
	memProfile            = flag.String("memProfile", "", "Enables memory profiling output to the filename provided.")
	memProfileRate        = flag.Int("memProfileRate", 512*1024, "Configures the memory profiling rate. \n"+
		"memProfile should be set before setting profile rate. To include every allocated block in the profile, "+
		"set MemProfileRate to 1. To turn off profiling entirely, set MemProfileRate to 0. 512 * 1024 by default.")
	cpuProfile          = flag.String("cpuProfile", "", "Enables CPU profiling output to the filename provided")
	benchmarkResultFile = flag.String("resultFile", "", "Save the benchmark result into a binary file")
	useBufconn          = flag.Bool("bufconn", false, "Use in-memory connection instead of system network I/O")
	enableKeepalive     = flag.Bool("enable_keepalive", false, "Enable client keepalive. \n"+
		"Keepalive.Time is set to 10s, Keepalive.Timeout is set to 1s, Keepalive.PermitWithoutStream is set to true.")
)

const (
	workloadsUnary         = "unary"
	workloadsStreaming     = "streaming"
	workloadsUnconstrained = "unconstrained"
	workloadsAll           = "all"
	// Compression modes.
	compModeOff  = "off"
	compModeGzip = "gzip"
	compModeNop  = "nop"
	compModeAll  = "all"
	// Toggle modes.
	toggleModeOff  = "off"
	toggleModeOn   = "on"
	toggleModeBoth = "both"
	// Network modes.
	networkModeNone  = "none"
	networkModeLocal = "Local"
	networkModeLAN   = "LAN"
	networkModeWAN   = "WAN"
	networkLongHaul  = "Longhaul"

	numStatsBuckets = 10
	warmupCallCount = 10
	warmuptime      = time.Second
)

var (
	allWorkloads              = []string{workloadsUnary, workloadsStreaming, workloadsUnconstrained, workloadsAll}
	allCompModes              = []string{compModeOff, compModeGzip, compModeNop, compModeAll}
	allToggleModes            = []string{toggleModeOff, toggleModeOn, toggleModeBoth}
	allNetworkModes           = []string{networkModeNone, networkModeLocal, networkModeLAN, networkModeWAN, networkLongHaul}
	defaultReadLatency        = []time.Duration{0, 40 * time.Millisecond} // if non-positive, no delay.
	defaultReadKbps           = []int{0, 10240}                           // if non-positive, infinite
	defaultReadMTU            = []int{0}                                  // if non-positive, infinite
	defaultMaxConcurrentCalls = []int{1, 8, 64, 512}
	defaultReqSizeBytes       = []int{1, 1024, 1024 * 1024}
	defaultRespSizeBytes      = []int{1, 1024, 1024 * 1024}
	networks                  = map[string]latency.Network{
		networkModeLocal: latency.Local,
		networkModeLAN:   latency.LAN,
		networkModeWAN:   latency.WAN,
		networkLongHaul:  latency.Longhaul,
	}
	keepaliveTime    = 10 * time.Second
	keepaliveTimeout = 1 * time.Second
	// This is 0.8*keepaliveTime to prevent connection issues because of server
	// keepalive enforcement.
	keepaliveMinTime = 8 * time.Second
)

// runModes indicates the workloads to run. This is initialized with a call to
// `runModesFromWorkloads`, passing the workloads flag set by the user.
type runModes struct {
	unary, streaming, unconstrained bool
}

// runModesFromWorkloads determines the runModes based on the value of
// workloads flag set by the user.
func runModesFromWorkloads(workload string) runModes {
	r := runModes{}
	switch workload {
	case workloadsUnary:
		r.unary = true
	case workloadsStreaming:
		r.streaming = true
	case workloadsUnconstrained:
		r.unconstrained = true
	case workloadsAll:
		r.unary = true
		r.streaming = true
		r.unconstrained = true
	default:
		log.Fatalf("Unknown workloads setting: %v (want one of: %v)",
			workloads, strings.Join(allWorkloads, ", "))
	}
	return r
}

type startFunc func(mode string, bf stats.Features)
type stopFunc func(count uint64)
type ucStopFunc func(req uint64, resp uint64)
type rpcCallFunc func(pos int)
type rpcSendFunc func(pos int)
type rpcRecvFunc func(pos int)
type rpcCleanupFunc func()

func unaryBenchmark(start startFunc, stop stopFunc, bf stats.Features, s *stats.Stats) {
	caller, cleanup := makeFuncUnary(bf)
	defer cleanup()
	runBenchmark(caller, start, stop, bf, s, workloadsUnary)
}

func streamBenchmark(start startFunc, stop stopFunc, bf stats.Features, s *stats.Stats) {
	caller, cleanup := makeFuncStream(bf)
	defer cleanup()
	runBenchmark(caller, start, stop, bf, s, workloadsStreaming)
}

func unconstrainedStreamBenchmark(start startFunc, stop ucStopFunc, bf stats.Features, s *stats.Stats) {
	var sender rpcSendFunc
	var recver rpcRecvFunc
	var cleanup rpcCleanupFunc
	if bf.EnablePreloader {
		sender, recver, cleanup = makeFuncUnconstrainedStreamPreloaded(bf)
	} else {
		sender, recver, cleanup = makeFuncUnconstrainedStream(bf)
	}
	defer cleanup()

	var req, resp uint64
	go func() {
		// Resets the counters once warmed up
		<-time.NewTimer(warmuptime).C
		atomic.StoreUint64(&req, 0)
		atomic.StoreUint64(&resp, 0)
		start(workloadsUnconstrained, bf)
	}()

	bmEnd := time.Now().Add(bf.BenchTime + warmuptime)
	var wg sync.WaitGroup
	wg.Add(2 * bf.MaxConcurrentCalls)
	for i := 0; i < bf.MaxConcurrentCalls; i++ {
		go func(pos int) {
			defer wg.Done()
			for {
				t := time.Now()
				if t.After(bmEnd) {
					return
				}
				sender(pos)
				atomic.AddUint64(&req, 1)
			}
		}(i)
		go func(pos int) {
			defer wg.Done()
			for {
				t := time.Now()
				if t.After(bmEnd) {
					return
				}
				recver(pos)
				atomic.AddUint64(&resp, 1)
			}
		}(i)
	}
	wg.Wait()
	stop(req, resp)
}

// makeClient returns a gRPC client for the grpc.testing.BenchmarkService
// service. The client is configured using the different options in the passed
// 'bf'. Also returns a cleanup function to close the client and release
// resources.
func makeClient(bf stats.Features) (testpb.BenchmarkServiceClient, func()) {
	nw := &latency.Network{Kbps: bf.Kbps, Latency: bf.Latency, MTU: bf.MTU}
	opts := []grpc.DialOption{}
	sopts := []grpc.ServerOption{}
	if bf.ModeCompressor == compModeNop {
		sopts = append(sopts,
			grpc.RPCCompressor(nopCompressor{}),
			grpc.RPCDecompressor(nopDecompressor{}),
		)
		opts = append(opts,
			grpc.WithCompressor(nopCompressor{}),
			grpc.WithDecompressor(nopDecompressor{}),
		)
	}
	if bf.ModeCompressor == compModeGzip {
		sopts = append(sopts,
			grpc.RPCCompressor(grpc.NewGZIPCompressor()),
			grpc.RPCDecompressor(grpc.NewGZIPDecompressor()),
		)
		opts = append(opts,
			grpc.WithCompressor(grpc.NewGZIPCompressor()),
			grpc.WithDecompressor(grpc.NewGZIPDecompressor()),
		)
	}
	if bf.EnableKeepalive {
		sopts = append(sopts,
			grpc.KeepaliveParams(keepalive.ServerParameters{
				Time:    keepaliveTime,
				Timeout: keepaliveTimeout,
			}),
			grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
				MinTime:             keepaliveMinTime,
				PermitWithoutStream: true,
			}),
		)
		opts = append(opts,
			grpc.WithKeepaliveParams(keepalive.ClientParameters{
				Time:                keepaliveTime,
				Timeout:             keepaliveTimeout,
				PermitWithoutStream: true,
			}),
		)
	}
	sopts = append(sopts, grpc.MaxConcurrentStreams(uint32(bf.MaxConcurrentCalls+1)))
	opts = append(opts, grpc.WithInsecure())

	var lis net.Listener
	if bf.UseBufConn {
		bcLis := bufconn.Listen(256 * 1024)
		lis = bcLis
		opts = append(opts, grpc.WithContextDialer(func(ctx context.Context, address string) (net.Conn, error) {
			return nw.ContextDialer(func(context.Context, string, string) (net.Conn, error) {
				return bcLis.Dial()
			})(ctx, "", "")
		}))
	} else {
		var err error
		lis, err = net.Listen("tcp", "localhost:0")
		if err != nil {
			grpclog.Fatalf("Failed to listen: %v", err)
		}
		opts = append(opts, grpc.WithContextDialer(func(ctx context.Context, address string) (net.Conn, error) {
			return nw.ContextDialer((&net.Dialer{}).DialContext)(ctx, "tcp", lis.Addr().String())
		}))
	}
	lis = nw.Listener(lis)
	stopper := bm.StartServer(bm.ServerInfo{Type: "protobuf", Listener: lis}, sopts...)
	conn := bm.NewClientConn("" /* target not used */, opts...)
	return testpb.NewBenchmarkServiceClient(conn), func() {
		conn.Close()
		stopper()
	}
}

func makeFuncUnary(bf stats.Features) (rpcCallFunc, rpcCleanupFunc) {
	tc, cleanup := makeClient(bf)
	return func(int) {
		reqSizeBytes := bf.ReqSizeBytes
		respSizeBytes := bf.RespSizeBytes
		if bf.ReqPayloadCurve != nil {
			reqSizeBytes = bf.ReqPayloadCurve.ChooseRandom()
		}
		if bf.RespPayloadCurve != nil {
			respSizeBytes = bf.RespPayloadCurve.ChooseRandom()
		}
		unaryCaller(tc, reqSizeBytes, respSizeBytes)
	}, cleanup
}

func makeFuncStream(bf stats.Features) (rpcCallFunc, rpcCleanupFunc) {
	tc, cleanup := makeClient(bf)

	streams := make([]testpb.BenchmarkService_StreamingCallClient, bf.MaxConcurrentCalls)
	for i := 0; i < bf.MaxConcurrentCalls; i++ {
		stream, err := tc.StreamingCall(context.Background())
		if err != nil {
			grpclog.Fatalf("%v.StreamingCall(_) = _, %v", tc, err)
		}
		streams[i] = stream
	}

	return func(pos int) {
		reqSizeBytes := bf.ReqSizeBytes
		respSizeBytes := bf.RespSizeBytes
		if bf.ReqPayloadCurve != nil {
			reqSizeBytes = bf.ReqPayloadCurve.ChooseRandom()
		}
		if bf.RespPayloadCurve != nil {
			respSizeBytes = bf.RespPayloadCurve.ChooseRandom()
		}
		streamCaller(streams[pos], reqSizeBytes, respSizeBytes)
	}, cleanup
}

func makeFuncUnconstrainedStreamPreloaded(bf stats.Features) (rpcSendFunc, rpcRecvFunc, rpcCleanupFunc) {
	streams, req, cleanup := setupUnconstrainedStream(bf)

	preparedMsg := make([]*grpc.PreparedMsg, len(streams))
	for i, stream := range streams {
		preparedMsg[i] = &grpc.PreparedMsg{}
		err := preparedMsg[i].Encode(stream, req)
		if err != nil {
			grpclog.Fatalf("%v.Encode(%v, %v) = %v", preparedMsg[i], req, stream, err)
		}
	}

	return func(pos int) {
			streams[pos].SendMsg(preparedMsg[pos])
		}, func(pos int) {
			streams[pos].Recv()
		}, cleanup
}

func makeFuncUnconstrainedStream(bf stats.Features) (rpcSendFunc, rpcRecvFunc, rpcCleanupFunc) {
	streams, req, cleanup := setupUnconstrainedStream(bf)

	return func(pos int) {
			streams[pos].Send(req)
		}, func(pos int) {
			streams[pos].Recv()
		}, cleanup
}

func setupUnconstrainedStream(bf stats.Features) ([]testpb.BenchmarkService_StreamingCallClient, *testpb.SimpleRequest, rpcCleanupFunc) {
	tc, cleanup := makeClient(bf)

	streams := make([]testpb.BenchmarkService_StreamingCallClient, bf.MaxConcurrentCalls)
	for i := 0; i < bf.MaxConcurrentCalls; i++ {
		stream, err := tc.UnconstrainedStreamingCall(context.Background())
		if err != nil {
			grpclog.Fatalf("%v.UnconstrainedStreamingCall(_) = _, %v", tc, err)
		}
		streams[i] = stream
	}

	pl := bm.NewPayload(testpb.PayloadType_COMPRESSABLE, bf.ReqSizeBytes)
	req := &testpb.SimpleRequest{
		ResponseType: pl.Type,
		ResponseSize: int32(bf.RespSizeBytes),
		Payload:      pl,
	}

	return streams, req, cleanup
}

// Makes a UnaryCall gRPC request using the given BenchmarkServiceClient and
// request and response sizes.
func unaryCaller(client testpb.BenchmarkServiceClient, reqSize, respSize int) {
	if err := bm.DoUnaryCall(client, reqSize, respSize); err != nil {
		grpclog.Fatalf("DoUnaryCall failed: %v", err)
	}
}

func streamCaller(stream testpb.BenchmarkService_StreamingCallClient, reqSize, respSize int) {
	if err := bm.DoStreamingRoundTrip(stream, reqSize, respSize); err != nil {
		grpclog.Fatalf("DoStreamingRoundTrip failed: %v", err)
	}
}

func runBenchmark(caller rpcCallFunc, start startFunc, stop stopFunc, bf stats.Features, s *stats.Stats, mode string) {
	// Warm up connection.
	for i := 0; i < warmupCallCount; i++ {
		caller(0)
	}

	// Run benchmark.
	start(mode, bf)
	var wg sync.WaitGroup
	wg.Add(bf.MaxConcurrentCalls)
	bmEnd := time.Now().Add(bf.BenchTime)
	var count uint64
	for i := 0; i < bf.MaxConcurrentCalls; i++ {
		go func(pos int) {
			defer wg.Done()
			for {
				t := time.Now()
				if t.After(bmEnd) {
					return
				}
				start := time.Now()
				caller(pos)
				elapse := time.Since(start)
				atomic.AddUint64(&count, 1)
				s.AddDuration(elapse)
			}
		}(i)
	}
	wg.Wait()
	stop(count)
}

// benchOpts represents all configurable options available while running this
// benchmark. This is built from the values passed as flags.
type benchOpts struct {
	rModes              runModes
	benchTime           time.Duration
	memProfileRate      int
	memProfile          string
	cpuProfile          string
	networkMode         string
	benchmarkResultFile string
	useBufconn          bool
	enableKeepalive     bool
	features            *featureOpts
}

// featureOpts represents options which can have multiple values. The user
// usually provides a comma-separated list of options for each of these
// features through command line flags. We generate all possible combinations
// for the provided values and run the benchmarks for each combination.
type featureOpts struct {
	enableTrace        []bool
	readLatencies      []time.Duration
	readKbps           []int
	readMTU            []int
	maxConcurrentCalls []int
	reqSizeBytes       []int
	respSizeBytes      []int
	reqPayloadCurves   []*stats.PayloadCurve
	respPayloadCurves  []*stats.PayloadCurve
	compModes          []string
	enableChannelz     []bool
	enablePreloader    []bool
}

// makeFeaturesNum returns a slice of ints of size 'maxFeatureIndex' where each
// element of the slice (indexed by 'featuresIndex' enum) contains the number
// of features to be exercised by the benchmark code.
// For example: Index 0 of the returned slice contains the number of values for
// enableTrace feature, while index 1 contains the number of value of
// readLatencies feature and so on.
func makeFeaturesNum(b *benchOpts) []int {
	featuresNum := make([]int, stats.MaxFeatureIndex)
	for i := 0; i < len(featuresNum); i++ {
		switch stats.FeatureIndex(i) {
		case stats.EnableTraceIndex:
			featuresNum[i] = len(b.features.enableTrace)
		case stats.ReadLatenciesIndex:
			featuresNum[i] = len(b.features.readLatencies)
		case stats.ReadKbpsIndex:
			featuresNum[i] = len(b.features.readKbps)
		case stats.ReadMTUIndex:
			featuresNum[i] = len(b.features.readMTU)
		case stats.MaxConcurrentCallsIndex:
			featuresNum[i] = len(b.features.maxConcurrentCalls)
		case stats.ReqSizeBytesIndex:
			featuresNum[i] = len(b.features.reqSizeBytes)
		case stats.RespSizeBytesIndex:
			featuresNum[i] = len(b.features.respSizeBytes)
		case stats.ReqPayloadCurveIndex:
			featuresNum[i] = len(b.features.reqPayloadCurves)
		case stats.RespPayloadCurveIndex:
			featuresNum[i] = len(b.features.respPayloadCurves)
		case stats.CompModesIndex:
			featuresNum[i] = len(b.features.compModes)
		case stats.EnableChannelzIndex:
			featuresNum[i] = len(b.features.enableChannelz)
		case stats.EnablePreloaderIndex:
			featuresNum[i] = len(b.features.enablePreloader)
		default:
			log.Fatalf("Unknown feature index %v in generateFeatures. maxFeatureIndex is %v", i, stats.MaxFeatureIndex)
		}
	}
	return featuresNum
}

// sharedFeatures returns a bool slice which acts as a bitmask. Each item in
// the slice represents a feature, indexed by 'featureIndex' enum.  The bit is
// set to 1 if the corresponding feature does not have multiple value, so is
// shared amongst all benchmarks.
func sharedFeatures(featuresNum []int) []bool {
	result := make([]bool, len(featuresNum))
	for i, num := range featuresNum {
		if num <= 1 {
			result[i] = true
		}
	}
	return result
}

// generateFeatures generates all combinations of the provided feature options.
// While all the feature options are stored in the benchOpts struct, the input
// parameter 'featuresNum' is a slice indexed by 'featureIndex' enum containing
// the number of values for each feature.
// For example, let's say the user sets -workloads=all and
// -maxConcurrentCalls=1,100, this would end up with the following
// combinations:
// [workloads: unary, maxConcurrentCalls=1]
// [workloads: unary, maxConcurrentCalls=1]
// [workloads: streaming, maxConcurrentCalls=100]
// [workloads: streaming, maxConcurrentCalls=100]
// [workloads: unconstrained, maxConcurrentCalls=1]
// [workloads: unconstrained, maxConcurrentCalls=100]
func (b *benchOpts) generateFeatures(featuresNum []int) []stats.Features {
	// curPos and initialPos are two slices where each value acts as an index
	// into the appropriate feature slice maintained in benchOpts.features. This
	// loop generates all possible combinations of features by changing one value
	// at a time, and once curPos becomes equal to initialPos, we have explored
	// all options.
	var result []stats.Features
	var curPos []int
	initialPos := make([]int, stats.MaxFeatureIndex)
	for !reflect.DeepEqual(initialPos, curPos) {
		if curPos == nil {
			curPos = make([]int, stats.MaxFeatureIndex)
		}
		f := stats.Features{
			// These features stay the same for each iteration.
			NetworkMode:     b.networkMode,
			UseBufConn:      b.useBufconn,
			EnableKeepalive: b.enableKeepalive,
			BenchTime:       b.benchTime,
			// These features can potentially change for each iteration.
			EnableTrace:        b.features.enableTrace[curPos[stats.EnableTraceIndex]],
			Latency:            b.features.readLatencies[curPos[stats.ReadLatenciesIndex]],
			Kbps:               b.features.readKbps[curPos[stats.ReadKbpsIndex]],
			MTU:                b.features.readMTU[curPos[stats.ReadMTUIndex]],
			MaxConcurrentCalls: b.features.maxConcurrentCalls[curPos[stats.MaxConcurrentCallsIndex]],
			ModeCompressor:     b.features.compModes[curPos[stats.CompModesIndex]],
			EnableChannelz:     b.features.enableChannelz[curPos[stats.EnableChannelzIndex]],
			EnablePreloader:    b.features.enablePreloader[curPos[stats.EnablePreloaderIndex]],
		}
		if len(b.features.reqPayloadCurves) == 0 {
			f.ReqSizeBytes = b.features.reqSizeBytes[curPos[stats.ReqSizeBytesIndex]]
		} else {
			f.ReqPayloadCurve = b.features.reqPayloadCurves[curPos[stats.ReqPayloadCurveIndex]]
		}
		if len(b.features.respPayloadCurves) == 0 {
			f.RespSizeBytes = b.features.respSizeBytes[curPos[stats.RespSizeBytesIndex]]
		} else {
			f.RespPayloadCurve = b.features.respPayloadCurves[curPos[stats.RespPayloadCurveIndex]]
		}
		result = append(result, f)
		addOne(curPos, featuresNum)
	}
	return result
}

// addOne mutates the input slice 'features' by changing one feature, thus
// arriving at the next combination of feature values. 'featuresMaxPosition'
// provides the numbers of allowed values for each feature, indexed by
// 'featureIndex' enum.
func addOne(features []int, featuresMaxPosition []int) {
	for i := len(features) - 1; i >= 0; i-- {
		if featuresMaxPosition[i] == 0 {
			continue
		}
		features[i] = (features[i] + 1)
		if features[i]/featuresMaxPosition[i] == 0 {
			break
		}
		features[i] = features[i] % featuresMaxPosition[i]
	}
}

// processFlags reads the command line flags and builds benchOpts. Specifying
// invalid values for certain flags will cause flag.Parse() to fail, and the
// program to terminate.
// This *SHOULD* be the only place where the flags are accessed. All other
// parts of the benchmark code should rely on the returned benchOpts.
func processFlags() *benchOpts {
	flag.Parse()
	if flag.NArg() != 0 {
		log.Fatal("Error: unparsed arguments: ", flag.Args())
	}

	opts := &benchOpts{
		rModes:              runModesFromWorkloads(*workloads),
		benchTime:           *benchTime,
		memProfileRate:      *memProfileRate,
		memProfile:          *memProfile,
		cpuProfile:          *cpuProfile,
		networkMode:         *networkMode,
		benchmarkResultFile: *benchmarkResultFile,
		useBufconn:          *useBufconn,
		enableKeepalive:     *enableKeepalive,
		features: &featureOpts{
			enableTrace:        setToggleMode(*traceMode),
			readLatencies:      append([]time.Duration(nil), *readLatency...),
			readKbps:           append([]int(nil), *readKbps...),
			readMTU:            append([]int(nil), *readMTU...),
			maxConcurrentCalls: append([]int(nil), *maxConcurrentCalls...),
			reqSizeBytes:       append([]int(nil), *readReqSizeBytes...),
			respSizeBytes:      append([]int(nil), *readRespSizeBytes...),
			compModes:          setCompressorMode(*compressorMode),
			enableChannelz:     setToggleMode(*channelzOn),
			enablePreloader:    setToggleMode(*preloaderMode),
		},
	}

	if len(*reqPayloadCurveFiles) == 0 {
		if len(opts.features.reqSizeBytes) == 0 {
			opts.features.reqSizeBytes = defaultReqSizeBytes
		}
	} else {
		if len(opts.features.reqSizeBytes) != 0 {
			log.Fatalf("you may not specify -reqPayloadCurveFiles and -reqSizeBytes at the same time")
		}
		for _, file := range *reqPayloadCurveFiles {
			pc, err := stats.NewPayloadCurve(file)
			if err != nil {
				log.Fatalf("cannot load payload curve file %s: %v", file, err)
			}
			opts.features.reqPayloadCurves = append(opts.features.reqPayloadCurves, pc)
		}
		opts.features.reqSizeBytes = nil
	}
	if len(*respPayloadCurveFiles) == 0 {
		if len(opts.features.respSizeBytes) == 0 {
			opts.features.respSizeBytes = defaultRespSizeBytes
		}
	} else {
		if len(opts.features.respSizeBytes) != 0 {
			log.Fatalf("you may not specify -respPayloadCurveFiles and -respSizeBytes at the same time")
		}
		for _, file := range *respPayloadCurveFiles {
			pc, err := stats.NewPayloadCurve(file)
			if err != nil {
				log.Fatalf("cannot load payload curve file %s: %v", file, err)
			}
			opts.features.respPayloadCurves = append(opts.features.respPayloadCurves, pc)
		}
		opts.features.respSizeBytes = nil
	}

	// Re-write latency, kpbs and mtu if network mode is set.
	if network, ok := networks[opts.networkMode]; ok {
		opts.features.readLatencies = []time.Duration{network.Latency}
		opts.features.readKbps = []int{network.Kbps}
		opts.features.readMTU = []int{network.MTU}
	}
	return opts
}

func setToggleMode(val string) []bool {
	switch val {
	case toggleModeOn:
		return []bool{true}
	case toggleModeOff:
		return []bool{false}
	case toggleModeBoth:
		return []bool{false, true}
	default:
		// This should never happen because a wrong value passed to this flag would
		// be caught during flag.Parse().
		return []bool{}
	}
}

func setCompressorMode(val string) []string {
	switch val {
	case compModeNop, compModeGzip, compModeOff:
		return []string{val}
	case compModeAll:
		return []string{compModeNop, compModeGzip, compModeOff}
	default:
		// This should never happen because a wrong value passed to this flag would
		// be caught during flag.Parse().
		return []string{}
	}
}

func main() {
	opts := processFlags()
	before(opts)

	s := stats.NewStats(numStatsBuckets)
	featuresNum := makeFeaturesNum(opts)
	sf := sharedFeatures(featuresNum)

	var (
		start  = func(mode string, bf stats.Features) { s.StartRun(mode, bf, sf) }
		stop   = func(count uint64) { s.EndRun(count) }
		ucStop = func(req uint64, resp uint64) { s.EndUnconstrainedRun(req, resp) }
	)

	for _, bf := range opts.generateFeatures(featuresNum) {
		grpc.EnableTracing = bf.EnableTrace
		if bf.EnableChannelz {
			channelz.TurnOn()
		}
		if opts.rModes.unary {
			unaryBenchmark(start, stop, bf, s)
		}
		if opts.rModes.streaming {
			streamBenchmark(start, stop, bf, s)
		}
		if opts.rModes.unconstrained {
			unconstrainedStreamBenchmark(start, ucStop, bf, s)
		}
	}
	after(opts, s.GetResults())
}

func before(opts *benchOpts) {
	if opts.memProfile != "" {
		runtime.MemProfileRate = opts.memProfileRate
	}
	if opts.cpuProfile != "" {
		f, err := os.Create(opts.cpuProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			return
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't start cpu profile: %s\n", err)
			f.Close()
			return
		}
	}
}

func after(opts *benchOpts, data []stats.BenchResults) {
	if opts.cpuProfile != "" {
		pprof.StopCPUProfile() // flushes profile to disk
	}
	if opts.memProfile != "" {
		f, err := os.Create(opts.memProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testing: %s\n", err)
			os.Exit(2)
		}
		runtime.GC() // materialize all statistics
		if err = pprof.WriteHeapProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "testing: can't write heap profile %s: %s\n", opts.memProfile, err)
			os.Exit(2)
		}
		f.Close()
	}
	if opts.benchmarkResultFile != "" {
		f, err := os.Create(opts.benchmarkResultFile)
		if err != nil {
			log.Fatalf("testing: can't write benchmark result %s: %s\n", opts.benchmarkResultFile, err)
		}
		dataEncoder := gob.NewEncoder(f)
		dataEncoder.Encode(data)
		f.Close()
	}
}

// nopCompressor is a compressor that just copies data.
type nopCompressor struct{}

func (nopCompressor) Do(w io.Writer, p []byte) error {
	n, err := w.Write(p)
	if err != nil {
		return err
	}
	if n != len(p) {
		return fmt.Errorf("nopCompressor.Write: wrote %v bytes; want %v", n, len(p))
	}
	return nil
}

func (nopCompressor) Type() string { return compModeNop }

// nopDecompressor is a decompressor that just copies data.
type nopDecompressor struct{}

func (nopDecompressor) Do(r io.Reader) ([]byte, error) { return ioutil.ReadAll(r) }
func (nopDecompressor) Type() string                   { return compModeNop }
