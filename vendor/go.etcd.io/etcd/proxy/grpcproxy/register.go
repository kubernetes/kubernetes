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

	"go.etcd.io/etcd/clientv3"
	"go.etcd.io/etcd/clientv3/concurrency"
	"go.etcd.io/etcd/clientv3/naming"

	"golang.org/x/time/rate"
	gnaming "google.golang.org/grpc/naming"
)

// allow maximum 1 retry per second
const registerRetryRate = 1

// Register registers itself as a grpc-proxy server by writing prefixed-key
// with session of specified TTL (in seconds). The returned channel is closed
// when the client's context is canceled.
func Register(c *clientv3.Client, prefix string, addr string, ttl int) <-chan struct{} {
	rm := rate.NewLimiter(rate.Limit(registerRetryRate), registerRetryRate)

	donec := make(chan struct{})
	go func() {
		defer close(donec)

		for rm.Wait(c.Ctx()) == nil {
			ss, err := registerSession(c, prefix, addr, ttl)
			if err != nil {
				plog.Warningf("failed to create a session %v", err)
				continue
			}
			select {
			case <-c.Ctx().Done():
				ss.Close()
				return

			case <-ss.Done():
				plog.Warning("session expired; possible network partition or server restart")
				plog.Warning("creating a new session to rejoin")
				continue
			}
		}
	}()

	return donec
}

func registerSession(c *clientv3.Client, prefix string, addr string, ttl int) (*concurrency.Session, error) {
	ss, err := concurrency.NewSession(c, concurrency.WithTTL(ttl))
	if err != nil {
		return nil, err
	}

	gr := &naming.GRPCResolver{Client: c}
	if err = gr.Update(c.Ctx(), prefix, gnaming.Update{Op: gnaming.Add, Addr: addr, Metadata: getMeta()}, clientv3.WithLease(ss.Lease())); err != nil {
		return nil, err
	}

	plog.Infof("registered %q with %d-second lease", addr, ttl)
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
