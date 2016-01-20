package main

import (
	"flag"
	"math"
	"net"
	"net/http"
	_ "net/http/pprof"
	"sync"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/benchmark"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/benchmark/stats"
	"google.golang.org/grpc/grpclog"
)

var (
	server            = flag.String("server", "", "The server address")
	maxConcurrentRPCs = flag.Int("max_concurrent_rpcs", 1, "The max number of concurrent RPCs")
	duration          = flag.Int("duration", math.MaxInt32, "The duration in seconds to run the benchmark client")
	trace             = flag.Bool("trace", true, "Whether tracing is on")
	rpcType           = flag.Int("rpc_type", 0,
		`Configure different client rpc type. Valid options are:
		   0 : unary call;
		   1 : streaming call.`)
)

func unaryCaller(client testpb.TestServiceClient) {
	benchmark.DoUnaryCall(client, 1, 1)
}

func streamCaller(client testpb.TestServiceClient, stream testpb.TestService_StreamingCallClient) {
	benchmark.DoStreamingRoundTrip(client, stream, 1, 1)
}

func buildConnection() (s *stats.Stats, conn *grpc.ClientConn, tc testpb.TestServiceClient) {
	s = stats.NewStats(256)
	conn = benchmark.NewClientConn(*server)
	tc = testpb.NewTestServiceClient(conn)
	return s, conn, tc
}

func closeLoopUnary() {
	s, conn, tc := buildConnection()

	for i := 0; i < 100; i++ {
		unaryCaller(tc)
	}
	ch := make(chan int, *maxConcurrentRPCs*4)
	var (
		mu sync.Mutex
		wg sync.WaitGroup
	)
	wg.Add(*maxConcurrentRPCs)

	for i := 0; i < *maxConcurrentRPCs; i++ {
		go func() {
			for _ = range ch {
				start := time.Now()
				unaryCaller(tc)
				elapse := time.Since(start)
				mu.Lock()
				s.Add(elapse)
				mu.Unlock()
			}
			wg.Done()
		}()
	}
	// Stop the client when time is up.
	done := make(chan struct{})
	go func() {
		<-time.After(time.Duration(*duration) * time.Second)
		close(done)
	}()
	ok := true
	for ok {
		select {
		case ch <- 0:
		case <-done:
			ok = false
		}
	}
	close(ch)
	wg.Wait()
	conn.Close()
	grpclog.Println(s.String())

}

func closeLoopStream() {
	s, conn, tc := buildConnection()
	stream, err := tc.StreamingCall(context.Background())
	if err != nil {
		grpclog.Fatalf("%v.StreamingCall(_) = _, %v", tc, err)
	}
	for i := 0; i < 100; i++ {
		streamCaller(tc, stream)
	}
	ch := make(chan int, *maxConcurrentRPCs*4)
	var (
		mu sync.Mutex
		wg sync.WaitGroup
	)
	wg.Add(*maxConcurrentRPCs)
	// Distribute RPCs over maxConcurrentCalls workers.
	for i := 0; i < *maxConcurrentRPCs; i++ {
		go func() {
			for _ = range ch {
				start := time.Now()
				streamCaller(tc, stream)
				elapse := time.Since(start)
				mu.Lock()
				s.Add(elapse)
				mu.Unlock()
			}
			wg.Done()
		}()
	}
	// Stop the client when time is up.
	done := make(chan struct{})
	go func() {
		<-time.After(time.Duration(*duration) * time.Second)
		close(done)
	}()
	ok := true
	for ok {
		select {
		case ch <- 0:
		case <-done:
			ok = false
		}
	}
	close(ch)
	wg.Wait()
	conn.Close()
	grpclog.Println(s.String())
}

func main() {
	flag.Parse()
	grpc.EnableTracing = *trace
	go func() {
		lis, err := net.Listen("tcp", ":0")
		if err != nil {
			grpclog.Fatalf("Failed to listen: %v", err)
		}
		grpclog.Println("Client profiling address: ", lis.Addr().String())
		if err := http.Serve(lis, nil); err != nil {
			grpclog.Fatalf("Failed to serve: %v", err)
		}
	}()
	switch *rpcType {
	case 0:
		closeLoopUnary()
	case 1:
		closeLoopStream()
	}
}
