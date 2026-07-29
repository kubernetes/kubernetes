/*
Copyright 2023 The Kubernetes Authors.

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

package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"
	"time"

	"k8s.io/kms/pkg/service"
	"k8s.io/kms/pkg/util"
	"k8s.io/kms/plugins/mock/pkcs11"
)

var (
	listenAddr     = flag.String("listen-addr", "unix:///tmp/kms.socket", "gRPC listen address")
	timeout        = flag.Duration("timeout", 5*time.Second, "gRPC timeout")
	configFilePath = flag.String("config-file-path", "/etc/softhsm-config.json", "SoftHSM config file path")
)

func main() {
	flag.Parse()

	addr, err := util.ParseEndpoint(*listenAddr)
	if err != nil {
		panic("failed to parse endpoint: " + err.Error())
	}

	remoteKMSService, err := pkcs11.NewPKCS11RemoteService(*configFilePath, "kms-test")
	if err != nil {
		panic("failed to create remote service: " + err.Error())
	}

	ctx := withShutdownSignal(context.Background())
	grpcService := service.NewGRPCService(
		addr,
		*timeout,
		remoteKMSService,
	)

	go func() {
		if err := grpcService.ListenAndServe(); err != nil {
			panic("failed to serve: " + err.Error())
		}
	}()

	<-ctx.Done()
	grpcService.Shutdown()
}

// withShutdownSignal returns a copy of the parent context that will close if
// the process receives termination signals.
func withShutdownSignal(ctx context.Context) context.Context {
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGTERM, syscall.SIGINT, os.Interrupt)

	nctx, cancel := context.WithCancel(ctx)

	go func() {
		<-signalChan
		cancel()
	}()
	return nctx
}
