/*
Copyright 2022 The Kubernetes Authors.

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

// Package grpchealthchecking offers a tiny grpc health checking endpoint.
package grpchealthchecking

import (
	"context"
	"fmt"
	"log"
	"net"
	"time"

	"net/http"

	"github.com/spf13/cobra"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

// CmdGrpcHealthChecking is used by agnhost Cobra.
var CmdGrpcHealthChecking = &cobra.Command{
	Use:   "grpc-health-checking",
	Short: "Starts a simple grpc health checking endpoint",
	Long:  "Starts a simple grpc health checking endpoint with --port to serve on and --service to check status for. The endpoint returns SERVING for the first --delay-unhealthy-sec, and NOT_SERVING after this. NOT_FOUND will be returned for the requests for non-configured service name. Probe can be forced to be set NOT_SERVING by calling /make-not-serving http endpoint.",
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	port              int
	httpPort          int
	delayUnhealthySec int
	service           string
	forceUnhealthy    *bool
	certFile          string
	privKeyFile       string
)

func init() {
	CmdGrpcHealthChecking.Flags().IntVar(&port, "port", 5000, "Port number.")
	CmdGrpcHealthChecking.Flags().IntVar(&httpPort, "http-port", 8080, "Port number for the /make-serving and /make-not-serving.")
	CmdGrpcHealthChecking.Flags().IntVar(&delayUnhealthySec, "delay-unhealthy-sec", -1, "Number of seconds to delay before start reporting NOT_SERVING, negative value indicates never.")
	CmdGrpcHealthChecking.Flags().StringVar(&service, "service", "", "Service name to register the health check for.")
	CmdGrpcHealthChecking.Flags().StringVar(&certFile, "tls-cert-file", "",
		"File containing an x509 certificate for gRPC TLS. (CA cert, if any, concatenated after server cert).")
	CmdGrpcHealthChecking.Flags().StringVar(&privKeyFile, "tls-private-key-file", "",
		"File containing an x509 private key matching --tls-cert-file.")
	forceUnhealthy = nil
}

type HealthChecker struct {
	started time.Time
}

func (s *HealthChecker) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	log.Printf("Serving the Check request for health check, started at %v", s.started)

	if req.Service != service {
		return nil, status.Errorf(codes.NotFound, "unknown service")
	}

	duration := time.Since(s.started)
	if ((forceUnhealthy != nil) && *forceUnhealthy) || ((delayUnhealthySec >= 0) && (duration.Seconds() >= float64(delayUnhealthySec))) {
		return &grpc_health_v1.HealthCheckResponse{
			Status: grpc_health_v1.HealthCheckResponse_NOT_SERVING,
		}, nil
	}

	return &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_SERVING,
	}, nil
}

func (s *HealthChecker) Watch(req *grpc_health_v1.HealthCheckRequest, server grpc_health_v1.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "unimplemented")
}

func NewHealthChecker(started time.Time) *HealthChecker {
	return &HealthChecker{
		started: started,
	}
}

func main(cmd *cobra.Command, args []string) {
	started := time.Now()

	// Validate flags
	//
	// if certFile or privKeyFile are not both set, exit with error
	if (certFile == "" && privKeyFile != "") || (certFile != "" && privKeyFile == "") {
		log.Fatalf("Both --tls-cert-file and --tls-private-key-file must be set")
	}

	http.HandleFunc("/make-not-serving", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Mark as unhealthy")
		forceUnhealthy = new(bool)
		*forceUnhealthy = true
		w.WriteHeader(200)
		data := (time.Since(started)).String()
		w.Write([]byte(data))
	})

	http.HandleFunc("/make-serving", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("Mark as healthy")
		forceUnhealthy = new(bool)
		*forceUnhealthy = false
		w.WriteHeader(200)
		data := (time.Since(started)).String()
		w.Write([]byte(data))
	})

	go func() {
		httpServerAdr := fmt.Sprintf(":%d", httpPort)
		log.Printf("Http server starting to listen on %s", httpServerAdr)
		log.Fatal(http.ListenAndServe(httpServerAdr, nil))
	}()

	serverAdr := fmt.Sprintf(":%d", port)
	listenAddr, err := net.Listen("tcp", serverAdr)

	if err != nil {
		log.Fatalf("Error while starting the listening service %v", err)
	}

	var grpcServer *grpc.Server

	if certFile != "" && privKeyFile != "" {
		creds, err := credentials.NewServerTLSFromFile(certFile, privKeyFile)
		if err != nil {
			log.Fatalf("Failed to generate credentials %v", err)
		}
		grpcServer = grpc.NewServer(grpc.Creds(creds))
	} else {
		grpcServer = grpc.NewServer()
	}

	healthService := NewHealthChecker(started)
	grpc_health_v1.RegisterHealthServer(grpcServer, healthService)

	log.Printf("gRPC server starting to listen on %s", serverAdr)
	if err = grpcServer.Serve(listenAddr); err != nil {
		log.Fatalf("Error while starting the gRPC server on the %s listen address %v", listenAddr, err)
	}

	select {}
}
