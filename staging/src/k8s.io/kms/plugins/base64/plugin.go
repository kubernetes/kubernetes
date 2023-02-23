package main

import (
	"context"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/kms/pkg/hierarchy"
	"k8s.io/kms/pkg/service"
)

var (
	listenAddr = flag.String("listen-addr", "unix:///tmp/kms.socket", "gRPC listen address")
	timeout    = flag.Duration("timeout", 5*time.Second, "gRPC timeout")
)

type remoteService struct {
	mu    sync.Mutex
	keyID string
}

func main() {
	flag.Parse()

	addr, err := parseEndpoint(*listenAddr)
	if err != nil {
		klog.ErrorS(err, "failed to parse endpoint")
		os.Exit(1)
	}

	ctx := withShutdownSignal(context.Background())
	localKEKService := hierarchy.NewLocalKEKService(
		ctx,
		&remoteService{
			keyID: "base64-key-id",
		},
	)
	grpcService := service.NewGRPCService(addr, *timeout, localKEKService)

	klog.InfoS("starting server", "listen-addr", *listenAddr)
	if err := grpcService.ListenAndServe(); err != nil {
		klog.ErrorS(err, "failed to serve")
		os.Exit(1)
	}
}

func (s *remoteService) Encrypt(ctx context.Context, uid string, plaintext []byte) (*service.EncryptResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return &service.EncryptResponse{
		KeyID:      s.keyID,
		Ciphertext: []byte(base64.StdEncoding.EncodeToString(plaintext)),
		Annotations: map[string][]byte{
			"version.encryption.remote.io": []byte("1"),
		},
	}, nil
}

func (s *remoteService) Decrypt(ctx context.Context, uid string, req *service.DecryptRequest) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(req.Annotations) != 1 {
		return nil, errors.New("invalid annotations")
	}

	if v, ok := req.Annotations["version.encryption.remote.io"]; !ok || string(v) != "1" {
		return nil, errors.New("invalid version in annotations")
	}

	return base64.StdEncoding.DecodeString(string(req.Ciphertext))
}

func (s *remoteService) Status(ctx context.Context) (*service.StatusResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return &service.StatusResponse{
		Version: "v2alpha1",
		Healthz: "ok",
		KeyID:   s.keyID,
	}, nil
}

func parseEndpoint(endpoint string) (string, error) {
	if strings.HasPrefix(strings.ToLower(endpoint), "unix://") {
		s := strings.SplitN(endpoint, "://", 2)
		if s[1] != "" {
			return s[1], nil
		}
	}
	return "", fmt.Errorf("invalid endpoint: %v", endpoint)
}

// withShutdownSignal returns a copy of the parent context that will close if
// the process receives termination signals.
func withShutdownSignal(ctx context.Context) context.Context {
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGTERM, syscall.SIGINT, os.Interrupt)

	nctx, cancel := context.WithCancel(ctx)

	go func() {
		<-signalChan
		klog.Info("received shutdown signal")
		cancel()
	}()
	return nctx
}
