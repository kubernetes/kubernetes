/*
 *
 * Copyright 2016 gRPC authors.
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

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"strconv"
	"time"

	"google.golang.org/grpc"
	testpb "google.golang.org/grpc/benchmark/grpc_testing"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/status"
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
		grpclog.Infof("closing benchmark server")
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
			grpclog.Infof("server setup received:")
			if bs != nil {
				grpclog.Infof("server setup received when server already exists, closing the existing server")
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
			grpclog.Infof("server mark received:")
			grpclog.Infof(" - %v", argtype)
			if bs == nil {
				return status.Error(codes.InvalidArgument, "server does not exist when mark received")
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
		grpclog.Infof("shuting down benchmark client")
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
			grpclog.Infof("client setup received:")
			if bc != nil {
				grpclog.Infof("client setup received when client already exists, shuting down the existing client")
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
			grpclog.Infof("client mark received:")
			grpclog.Infof(" - %v", t)
			if bc == nil {
				return status.Error(codes.InvalidArgument, "client does not exist when mark received")
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
	grpclog.Infof("core count: %v", runtime.NumCPU())
	return &testpb.CoreResponse{Cores: int32(runtime.NumCPU())}, nil
}

func (s *workerServer) QuitWorker(ctx context.Context, in *testpb.Void) (*testpb.Void, error) {
	grpclog.Infof("quitting worker")
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
	grpclog.Infof("worker listening at port %v", *driverPort)

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
			grpclog.Infoln("Starting pprof server on port " + strconv.Itoa(*pprofPort))
			grpclog.Infoln(http.ListenAndServe("localhost:"+strconv.Itoa(*pprofPort), nil))
		}()
	}

	s.Serve(lis)
}
