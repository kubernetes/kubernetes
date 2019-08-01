/*
Copyright 2019 The Kubernetes Authors.

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

package keepalive

import (
	"context"
	"time"
)

type Pinger interface {
	Ping() error
}

func KeepAliveWithPinger(ctx context.Context, pinger Pinger, pingInterval time.Duration) error {
	if ctx == nil || pinger == nil || pingInterval == 0 {
		return nil
	}
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				pinger.Ping()
				time.Sleep(pingInterval)
			}
		}
	}()
	return nil
}
