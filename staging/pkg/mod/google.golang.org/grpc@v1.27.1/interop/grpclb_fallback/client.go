// +build !appengine
// +build go1.11

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

package main

import (
	"context"
	"flag"
	"log"
	"net"
	"os"
	"os/exec"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
	"google.golang.org/grpc"
	_ "google.golang.org/grpc/balancer/grpclb"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/alts"
	"google.golang.org/grpc/credentials/google"
	testpb "google.golang.org/grpc/interop/grpc_testing"
)

var (
	customCredentialsType         = flag.String("custom_credentials_type", "", "Client creds to use")
	serverURI                     = flag.String("server_uri", "dns:///staging-grpc-directpath-fallback-test.googleapis.com:443", "The server host name")
	unrouteLBAndBackendAddrsCmd   = flag.String("unroute_lb_and_backend_addrs_cmd", "", "Command to make LB and backend address unroutable")
	blackholeLBAndBackendAddrsCmd = flag.String("blackhole_lb_and_backend_addrs_cmd", "", "Command to make LB and backend addresses blackholed")
	testCase                      = flag.String("test_case", "",
		`Configure different test cases. Valid options are:
        fast_fallback_before_startup : LB/backend connections fail fast before RPC's have been made;
        fast_fallback_after_startup : LB/backend connections fail fast after RPC's have been made;
        slow_fallback_before_startup : LB/backend connections black hole before RPC's have been made;
        slow_fallback_after_startup : LB/backend connections black hole after RPC's have been made;`)
	infoLog  = log.New(os.Stderr, "INFO: ", log.Ldate|log.Ltime|log.Lshortfile)
	errorLog = log.New(os.Stderr, "ERROR: ", log.Ldate|log.Ltime|log.Lshortfile)
)

func doRPCAndGetPath(client testpb.TestServiceClient, timeout time.Duration) testpb.GrpclbRouteType {
	infoLog.Printf("doRPCAndGetPath timeout:%v\n", timeout)
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	req := &testpb.SimpleRequest{
		FillGrpclbRouteType: true,
	}
	reply, err := client.UnaryCall(ctx, req)
	if err != nil {
		infoLog.Printf("doRPCAndGetPath error:%v\n", err)
		return testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_UNKNOWN
	}
	g := reply.GetGrpclbRouteType()
	infoLog.Printf("doRPCAndGetPath got grpclb route type: %v\n", g)
	if g != testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_FALLBACK && g != testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_BACKEND {
		errorLog.Fatalf("Expected grpclb route type to be either backend or fallback; got: %d", g)
	}
	return g
}

func dialTCPUserTimeout(ctx context.Context, addr string) (net.Conn, error) {
	control := func(network, address string, c syscall.RawConn) error {
		var syscallErr error
		controlErr := c.Control(func(fd uintptr) {
			syscallErr = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, unix.TCP_USER_TIMEOUT, 20000)
		})
		if syscallErr != nil {
			errorLog.Fatalf("syscall error setting sockopt TCP_USER_TIMEOUT: %v", syscallErr)
		}
		if controlErr != nil {
			errorLog.Fatalf("control error setting sockopt TCP_USER_TIMEOUT: %v", syscallErr)
		}
		return nil
	}
	d := &net.Dialer{
		Control: control,
	}
	return d.DialContext(ctx, "tcp", addr)
}

func createTestConn() *grpc.ClientConn {
	opts := []grpc.DialOption{
		grpc.WithContextDialer(dialTCPUserTimeout),
		grpc.WithBlock(),
	}
	switch *customCredentialsType {
	case "tls":
		creds := credentials.NewClientTLSFromCert(nil, "")
		opts = append(opts, grpc.WithTransportCredentials(creds))
	case "alts":
		creds := alts.NewClientCreds(alts.DefaultClientOptions())
		opts = append(opts, grpc.WithTransportCredentials(creds))
	case "google_default_credentials":
		opts = append(opts, grpc.WithCredentialsBundle(google.NewDefaultCredentials()))
	case "compute_engine_channel_creds":
		opts = append(opts, grpc.WithCredentialsBundle(google.NewComputeEngineCredentials()))
	default:
		errorLog.Fatalf("Invalid --custom_credentials_type:%v", *customCredentialsType)
	}
	conn, err := grpc.Dial(*serverURI, opts...)
	if err != nil {
		errorLog.Fatalf("Fail to dial: %v", err)
	}
	return conn
}

func runCmd(command string) {
	infoLog.Printf("Running cmd:|%v|\n", command)
	if err := exec.Command("bash", "-c", command).Run(); err != nil {
		errorLog.Fatalf("error running cmd:|%v| : %v", command, err)
	}
}

func waitForFallbackAndDoRPCs(client testpb.TestServiceClient, fallbackDeadline time.Time) {
	fallbackRetryCount := 0
	fellBack := false
	for time.Now().Before(fallbackDeadline) {
		g := doRPCAndGetPath(client, 1*time.Second)
		if g == testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_FALLBACK {
			infoLog.Println("Made one successul RPC to a fallback. Now expect the same for the rest.")
			fellBack = true
			break
		} else if g == testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_BACKEND {
			errorLog.Fatalf("Got RPC type backend. This suggests an error in test implementation")
		} else {
			infoLog.Println("Retryable RPC failure on iteration:", fallbackRetryCount)
		}
		fallbackRetryCount++
	}
	if !fellBack {
		infoLog.Fatalf("Didn't fall back before deadline: %v\n", fallbackDeadline)
	}
	for i := 0; i < 30; i++ {
		if g := doRPCAndGetPath(client, 20*time.Second); g != testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_FALLBACK {
			errorLog.Fatalf("Expected RPC to take grpclb route type FALLBACK. Got: %v", g)
		}
		time.Sleep(time.Second)
	}
}

func doFastFallbackBeforeStartup() {
	runCmd(*unrouteLBAndBackendAddrsCmd)
	fallbackDeadline := time.Now().Add(5 * time.Second)
	conn := createTestConn()
	defer conn.Close()
	client := testpb.NewTestServiceClient(conn)
	waitForFallbackAndDoRPCs(client, fallbackDeadline)
}

func doSlowFallbackBeforeStartup() {
	runCmd(*blackholeLBAndBackendAddrsCmd)
	fallbackDeadline := time.Now().Add(20 * time.Second)
	conn := createTestConn()
	defer conn.Close()
	client := testpb.NewTestServiceClient(conn)
	waitForFallbackAndDoRPCs(client, fallbackDeadline)
}

func doFastFallbackAfterStartup() {
	conn := createTestConn()
	defer conn.Close()
	client := testpb.NewTestServiceClient(conn)
	if g := doRPCAndGetPath(client, 20*time.Second); g != testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_BACKEND {
		errorLog.Fatalf("Expected RPC to take grpclb route type BACKEND. Got: %v", g)
	}
	runCmd(*unrouteLBAndBackendAddrsCmd)
	fallbackDeadline := time.Now().Add(40 * time.Second)
	waitForFallbackAndDoRPCs(client, fallbackDeadline)
}

func doSlowFallbackAfterStartup() {
	conn := createTestConn()
	defer conn.Close()
	client := testpb.NewTestServiceClient(conn)
	if g := doRPCAndGetPath(client, 20*time.Second); g != testpb.GrpclbRouteType_GRPCLB_ROUTE_TYPE_BACKEND {
		errorLog.Fatalf("Expected RPC to take grpclb route type BACKEND. Got: %v", g)
	}
	runCmd(*blackholeLBAndBackendAddrsCmd)
	fallbackDeadline := time.Now().Add(40 * time.Second)
	waitForFallbackAndDoRPCs(client, fallbackDeadline)
}

func main() {
	flag.Parse()
	if len(*unrouteLBAndBackendAddrsCmd) == 0 {
		errorLog.Fatalf("--unroute_lb_and_backend_addrs_cmd unset")
	}
	if len(*blackholeLBAndBackendAddrsCmd) == 0 {
		errorLog.Fatalf("--blackhole_lb_and_backend_addrs_cmd unset")
	}
	switch *testCase {
	case "fast_fallback_before_startup":
		doFastFallbackBeforeStartup()
		log.Printf("FastFallbackBeforeStartup done!\n")
	case "fast_fallback_after_startup":
		doFastFallbackAfterStartup()
		log.Printf("FastFallbackAfterStartup done!\n")
	case "slow_fallback_before_startup":
		doSlowFallbackBeforeStartup()
		log.Printf("SlowFallbackBeforeStartup done!\n")
	case "slow_fallback_after_startup":
		doSlowFallbackAfterStartup()
		log.Printf("SlowFallbackAfterStartup done!\n")
	default:
		errorLog.Fatalf("Unsupported test case: %v", *testCase)
	}
}
