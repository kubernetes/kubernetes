// Copyright 2017 The etcd Authors
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

package grpcproxy

import (
	"encoding/json"
	"os"

	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"go.etcd.io/etcd/client/v3/naming/endpoints"

	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// allow maximum 1 retry per second
const registerRetryRate = 1

// Register registers itself as a grpc-proxy server by writing prefixed-key
// with session of specified TTL (in seconds). The returned channel is closed
// when the client's context is canceled.
func Register(lg *zap.Logger, c *clientv3.Client, prefix string, addr string, ttl int) <-chan struct{} {
	rm := rate.NewLimiter(rate.Limit(registerRetryRate), registerRetryRate)

	donec := make(chan struct{})
	go func() {
		defer close(donec)

		for rm.Wait(c.Ctx()) == nil {
			ss, err := registerSession(lg, c, prefix, addr, ttl)
			if err != nil {
				lg.Warn("failed to create a session", zap.Error(err))
				continue
			}
			select {
			case <-c.Ctx().Done():
				ss.Close()
				return

			case <-ss.Done():
				lg.Warn("session expired; possible network partition or server restart")
				lg.Warn("creating a new session to rejoin")
				continue
			}
		}
	}()

	return donec
}

func registerSession(lg *zap.Logger, c *clientv3.Client, prefix string, addr string, ttl int) (*concurrency.Session, error) {
	ss, err := concurrency.NewSession(c, concurrency.WithTTL(ttl))
	if err != nil {
		return nil, err
	}

	em, err := endpoints.NewManager(c, prefix)
	if err != nil {
		return nil, err
	}
	endpoint := endpoints.Endpoint{Addr: addr, Metadata: getMeta()}
	if err = em.AddEndpoint(c.Ctx(), prefix+"/"+addr, endpoint, clientv3.WithLease(ss.Lease())); err != nil {
		return nil, err
	}

	lg.Info(
		"registered session with lease",
		zap.String("addr", addr),
		zap.Int("lease-ttl", ttl),
	)
	return ss, nil
}

// meta represents metadata of proxy register.
type meta struct {
	Name string `json:"name"`
}

func getMeta() string {
	hostname, _ := os.Hostname()
	bts, _ := json.Marshal(meta{Name: hostname})
	return string(bts)
}

func decodeMeta(s string) (meta, error) {
	m := meta{}
	err := json.Unmarshal([]byte(s), &m)
	return m, err
}
