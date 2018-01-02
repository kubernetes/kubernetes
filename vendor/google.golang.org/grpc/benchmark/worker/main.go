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
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"strconv"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
)

var (
	driverPort    = flag.Int("driver_port", 10000, "port for communication with driver")
	serverPort    = flag.Int("server_port", 0, "port for benchmark server if not specified by server config message")
	pprofPort     = flag.Int("pprof_port", -1, "Port for pprof debug server to listen on. Pprof server doesn't start if unset")
	blockProfRate = flag.Int("block_prof_rate", 0, "fraction of goroutine blocking events to report in blocking profile")
)

type byteBufCodec struct {
}

func (byteBufCodec) Marshal(v interface{}) ([]byte, error) {
	b, ok := v.(*[]byte)
	if !ok {
		return nil, fmt.Errorf("failed to marshal: %v is not type of *[]byte", v)
	}
	return *b, nil
}

func (byteBufCodec) Unmarshal(data []byte, v interface{}) error {
	b, ok := v.(*[]byte)
	if !ok {
		return fmt.Errorf("failed to marshal: %v is not type of *[]byte", v)
	}
	*b = data
	return nil
}

func (byteBufCodec) String() string {
	return "bytebuffer"
}

// workerServer implements WorkerService rpc handlers.
// It can create benchmarkServer or benchmarkClient on demand.
type workerServer struct {
	stop       chan<- bool
	serverPort int
}

func (s *workerServer) RunServer(stream testpb.WorkerService_RunServerServer) error {
	var bs *benchmarkServer
	defer func() {
		// Close benchmark server when stream ends.
		grpclog.Printf("closing benchmark server")
		if bs != nil {
			bs.closeFunc()
		}
	}()
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		var out *testpb.ServerStatus
		switch argtype := in.Argtype.(type) {
		case *testpb.ServerArgs_Setup:
			grpclog.Printf("server setup received:")
			if bs != nil {
				grpclog.Printf("server setup received when server already exists, closing the existing server")
				bs.closeFunc()
			}
			bs, err = startBenchmarkServer(argtype.Setup, s.serverPort)
			if err != nil {
				return err
			}
			out = &testpb.ServerStatus{
				Stats: bs.getStats(false),
				Port:  int32(bs.port),
				Cores: int32(bs.cores),
			}

		case *testpb.ServerArgs_Mark:
			grpclog.Printf("server mark received:")
			grpclog.Printf(" - %v", argtype)
			if bs == nil {
				return grpc.Errorf(codes.InvalidArgument, "server does not exist when mark received")
			}
			out = &testpb.ServerStatus{
				Stats: bs.getStats(argtype.Mark.Reset_),
				Port:  int32(bs.port),
				Cores: int32(bs.cores),
			}
		}

		if err := stream.Send(out); err != nil {
			return err
		}
	}
}

func (s *workerServer) RunClient(stream testpb.WorkerService_RunClientServer) error {
	var bc *benchmarkClient
	defer func() {
		// Shut down benchmark client when stream ends.
		grpclog.Printf("shuting down benchmark client")
		if bc != nil {
			bc.shutdown()
		}
	}()
	for {
		in, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		var out *testpb.ClientStatus
		switch t := in.Argtype.(type) {
		case *testpb.ClientArgs_Setup:
			grpclog.Printf("client setup received:")
			if bc != nil {
				grpclog.Printf("client setup received when client already exists, shuting down the existing client")
				bc.shutdown()
			}
			bc, err = startBenchmarkClient(t.Setup)
			if err != nil {
				return err
			}
			out = &testpb.ClientStatus{
				Stats: bc.getStats(false),
			}

		case *testpb.ClientArgs_Mark:
			grpclog.Printf("client mark received:")
			grpclog.Printf(" - %v", t)
			if bc == nil {
				return grpc.Errorf(codes.InvalidArgument, "client does not exist when mark received")
			}
			out = &testpb.ClientStatus{
				Stats: bc.getStats(t.Mark.Reset_),
			}
		}

		if err := stream.Send(out); err != nil {
			return err
		}
	}
}

func (s *workerServer) CoreCount(ctx context.Context, in *testpb.CoreRequest) (*testpb.CoreResponse, error) {
	grpclog.Printf("core count: %v", runtime.NumCPU())
	return &testpb.CoreResponse{Cores: int32(runtime.NumCPU())}, nil
}

func (s *workerServer) QuitWorker(ctx context.Context, in *testpb.Void) (*testpb.Void, error) {
	grpclog.Printf("quiting worker")
	s.stop <- true
	return &testpb.Void{}, nil
}

func main() {
	grpc.EnableTracing = false

	flag.Parse()
	lis, err := net.Listen("tcp", ":"+strconv.Itoa(*driverPort))
	if err != nil {
		grpclog.Fatalf("failed to listen: %v", err)
	}
	grpclog.Printf("worker listening at port %v", *driverPort)

	s := grpc.NewServer()
	stop := make(chan bool)
	testpb.RegisterWorkerServiceServer(s, &workerServer{
		stop:       stop,
		serverPort: *serverPort,
	})

	go func() {
		<-stop
		// Wait for 1 second before stopping the server to make sure the return value of QuitWorker is sent to client.
		// TODO revise this once server graceful stop is supported in gRPC.
		time.Sleep(time.Second)
		s.Stop()
	}()

	runtime.SetBlockProfileRate(*blockProfRate)

	if *pprofPort >= 0 {
		go func() {
			grpclog.Println("Starting pprof server on port " + strconv.Itoa(*pprofPort))
			grpclog.Println(http.ListenAndServe("localhost:"+strconv.Itoa(*pprofPort), nil))
		}()
	}

	s.Serve(lis)
}
