/*
Copyright 2024 The Kubernetes Authors.

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

package v1alpha1

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/timestamppb"

	"k8s.io/externaljwt/apis/v1alpha1"
	"k8s.io/klog/v2"
)

type MockSigner struct {
	socketPath string
	server     *grpc.Server
	listener   net.Listener

	SigningKey                *rsa.PrivateKey
	SigningKeyID              string
	SigningAlg                string
	TokenType                 string
	MaxTokenExpirationSeconds int64

	supportedKeys        map[string]KeyT
	supportedKeysLock    sync.RWMutex
	supportedKeysFetched *sync.Cond

	FetchError    error
	MetadataError error
	errorLock     sync.RWMutex
}

type KeyT struct {
	Key                      []byte
	ExcludeFromOidcDiscovery bool
}

// NewMockSigner starts and returns a new MockSigner
// It servers on the provided socket.
func NewMockSigner(t *testing.T, socketPath string) *MockSigner {
	server := grpc.NewServer()

	m := &MockSigner{
		socketPath:                socketPath,
		server:                    server,
		MaxTokenExpirationSeconds: 10 * 60, // 10m
	}
	m.supportedKeysFetched = sync.NewCond(&m.supportedKeysLock)

	if err := m.Reset(); err != nil {
		t.Fatalf("failed to load keys for mock signer: %v", err)
	}

	v1alpha1.RegisterExternalJWTSignerServer(server, m)
	if err := m.start(t); err != nil {
		t.Fatalf("failed to start Mock Signer with error: %v", err)
	}

	t.Cleanup(m.CleanUp)
	if err := m.waitForMockServerToStart(); err != nil {
		t.Fatalf("failed to start Mock Signer with error %v", err)
	}

	return m
}

func (m *MockSigner) GetSupportedKeys() map[string]KeyT {
	m.supportedKeysLock.RLock()
	defer m.supportedKeysLock.RUnlock()
	return m.supportedKeys
}
func (m *MockSigner) SetSupportedKeys(keys map[string]KeyT) {
	m.supportedKeysLock.Lock()
	defer m.supportedKeysLock.Unlock()
	m.supportedKeys = keys
}
func (m *MockSigner) WaitForSupportedKeysFetch() {
	m.supportedKeysLock.Lock()
	defer m.supportedKeysLock.Unlock()
	m.supportedKeysFetched.Wait()
}

func (m *MockSigner) Sign(ctx context.Context, req *v1alpha1.SignJWTRequest) (*v1alpha1.SignJWTResponse, error) {

	header := &struct {
		Algorithm string `json:"alg,omitempty"`
		KeyID     string `json:"kid,omitempty"`
		Type      string `json:"typ,omitempty"`
	}{
		Type:      m.TokenType,
		Algorithm: m.SigningAlg,
		KeyID:     m.SigningKeyID,
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return nil, fmt.Errorf("failed to create header for JWT response")
	}

	base64Header := base64.RawURLEncoding.EncodeToString(headerJSON)

	toBeSignedHash := hashBytes([]byte(base64Header + "." + req.Claims))

	signature, err := rsa.SignPKCS1v15(rand.Reader, m.SigningKey, crypto.SHA256, toBeSignedHash)
	if err != nil {
		return nil, fmt.Errorf("unable to sign payload: %w", err)
	}

	return &v1alpha1.SignJWTResponse{
		Header:    base64Header,
		Signature: base64.RawURLEncoding.EncodeToString(signature),
	}, nil
}

func (m *MockSigner) FetchKeys(ctx context.Context, req *v1alpha1.FetchKeysRequest) (*v1alpha1.FetchKeysResponse, error) {
	m.errorLock.RLocker().Lock()
	defer m.errorLock.RLocker().Unlock()
	if m.FetchError != nil {
		return nil, m.FetchError
	}

	keys := []*v1alpha1.Key{}

	m.supportedKeysLock.RLock()
	for id, k := range m.supportedKeys {
		keys = append(keys, &v1alpha1.Key{
			KeyId:                    id,
			Key:                      k.Key,
			ExcludeFromOidcDiscovery: k.ExcludeFromOidcDiscovery,
		})
	}
	m.supportedKeysFetched.Broadcast()
	m.supportedKeysLock.RUnlock()

	now := time.Now()
	return &v1alpha1.FetchKeysResponse{
		RefreshHintSeconds: 5,
		DataTimestamp:      &timestamppb.Timestamp{Seconds: now.Unix(), Nanos: int32(now.Nanosecond())},
		Keys:               keys,
	}, nil
}

func (m *MockSigner) Metadata(ctx context.Context, req *v1alpha1.MetadataRequest) (*v1alpha1.MetadataResponse, error) {
	m.errorLock.RLocker().Lock()
	defer m.errorLock.RLocker().Unlock()
	if m.MetadataError != nil {
		return nil, m.MetadataError
	}
	return &v1alpha1.MetadataResponse{
		MaxTokenExpirationSeconds: m.MaxTokenExpirationSeconds,
	}, nil
}

// Reset genrate and adds signing/supported keys to MockSigner instance.
func (m *MockSigner) Reset() error {

	priv1, pub1, err := generateKeyPair()
	if err != nil {
		return err
	}

	_, pub2, err := generateKeyPair()
	if err != nil {
		return err
	}

	_, pub3, err := generateKeyPair()
	if err != nil {
		return err
	}

	m.SigningKey = priv1
	m.SigningKeyID = "kid-1"
	m.SigningAlg = "RS256"
	m.TokenType = "JWT"
	m.SetSupportedKeys(map[string]KeyT{
		"kid-1": {Key: pub1},
		"kid-2": {Key: pub2},
		"kid-3": {Key: pub3},
	})
	m.errorLock.Lock()
	defer m.errorLock.Unlock()
	m.FetchError = nil
	m.MetadataError = nil
	m.MaxTokenExpirationSeconds = 10 * 60 // 10m

	return nil
}

// start makes the gRpc MockServer listen on unix socket.
func (m *MockSigner) start(t *testing.T) error {
	var err error

	m.listener, err = net.Listen("unix", m.socketPath)
	if err != nil {
		return fmt.Errorf("failed to listen on the unix socket, error: %w", err)
	}

	klog.Infof("Starting Mock Signer at socketPath %s", m.socketPath)
	go func() {
		if err := m.server.Serve(m.listener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			t.Error(err)
		}
	}()
	klog.Infof("Mock Signer listening at socketPath %s", m.socketPath)

	return nil
}

// waitForMockServerToStart waits until Mock signer is ready to server.
// waits for a max of 30s before failing.
func (m *MockSigner) waitForMockServerToStart() error {
	var gRPCErr error

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()
	doneCh := ctx.Done()

	for range 30 {
		select {
		case <-doneCh:
			return fmt.Errorf("failed to start Mock signer: %w", ctx.Err())
		default:
		}
		if _, gRPCErr = m.FetchKeys(context.Background(), &v1alpha1.FetchKeysRequest{}); gRPCErr == nil {
			break
		}
		time.Sleep(time.Second)
	}

	if gRPCErr != nil {
		return fmt.Errorf("failed to start Mock signer, gRPC error: %w", gRPCErr)
	}

	return nil
}

// CleanUp stops gRPC server and the underlying listener.
func (m *MockSigner) CleanUp() {
	m.server.GracefulStop()
	_ = m.listener.Close()
	_ = os.Remove(m.socketPath)
}

func generateKeyPair() (*rsa.PrivateKey, []byte, error) {

	// Generate a new private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		klog.Errorf("Error generating private key: %v", err)
		return nil, nil, err
	}

	publicKeyBytes, err := x509.MarshalPKIXPublicKey(&privateKey.PublicKey)
	if err != nil {
		klog.Errorf("Error marshaling public key: %v", err)
		return nil, nil, err
	}

	return privateKey, publicKeyBytes, nil
}

func hashBytes(bytes []byte) []byte {
	hasher := crypto.SHA256.New()
	hasher.Write(bytes)
	return hasher.Sum(nil)
}
