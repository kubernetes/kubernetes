package main

import (
	"flag"
	"math"
	"net"
	"net/http"
	_ "net/http/pprof"
	"time"

	"google.golang.org/grpc/benchmark"
	"google.golang.org/grpc/grpclog"
)

var (
	duration = flag.Int("duration", math.MaxInt32, "The duration in seconds to run the benchmark server")
)

func main() {
	flag.Parse()
	go func() {
		lis, err := net.Listen("tcp", ":0")
		if err != nil {
			grpclog.Fatalf("Failed to listen: %v", err)
		}
		grpclog.Println("Server profiling address: ", lis.Addr().String())
		if err := http.Serve(lis, nil); err != nil {
			grpclog.Fatalf("Failed to serve: %v", err)
		}
	}()
	addr, stopper := benchmark.StartServer(":0") // listen on all interfaces
	grpclog.Println("Server Address: ", addr)
	<-time.After(time.Duration(*duration) * time.Second)
	stopper()
}
