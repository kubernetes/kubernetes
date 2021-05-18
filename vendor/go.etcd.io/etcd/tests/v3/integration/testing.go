// Copyright 2021 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"os"
	"path/filepath"
	"testing"

	grpc_logsettable "github.com/grpc-ecosystem/go-grpc-middleware/logging/settable"
	"go.etcd.io/etcd/client/pkg/v3/testutil"
	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"
	"go.etcd.io/etcd/server/v3/verify"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zapgrpc"
	"go.uber.org/zap/zaptest"
)

var grpc_logger grpc_logsettable.SettableLoggerV2

func init() {
	grpc_logger = grpc_logsettable.ReplaceGrpcLoggerV2()
}

func BeforeTest(t testutil.TB) {
	testutil.BeforeTest(t)

	grpc_logger.Set(zapgrpc.NewLogger(zaptest.NewLogger(t).Named("grpc")))

	// Integration tests should verify written state as much as possible.
	os.Setenv(verify.ENV_VERIFY, verify.ENV_VERIFY_ALL_VALUE)

	previousWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	os.Chdir(t.TempDir())
	t.Cleanup(func() {
		grpc_logger.Reset()
		os.Chdir(previousWD)
	})

}

func MustAbsPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		panic(err)
	}
	return abs
}

func NewEmbedConfig(t testing.TB, name string) *embed.Config {
	cfg := embed.NewConfig()
	cfg.Name = name
	lg := zaptest.NewLogger(t, zaptest.Level(zapcore.InfoLevel)).Named(cfg.Name)
	cfg.ZapLoggerBuilder = embed.NewZapLoggerBuilder(lg)
	cfg.Dir = t.TempDir()
	return cfg
}

func NewClient(t testing.TB, cfg clientv3.Config) (*clientv3.Client, error) {
	if cfg.Logger != nil {
		cfg.Logger = zaptest.NewLogger(t)
	}
	return clientv3.New(cfg)
}
