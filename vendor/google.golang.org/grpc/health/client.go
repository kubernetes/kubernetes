/*
 *
 * Copyright 2018 gRPC authors.
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

package health

import (
	"context"
	"fmt"
	"io"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/internal"
	"google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/status"
)

const maxDelay = 120 * time.Second

var backoffStrategy = backoff.Exponential{MaxDelay: maxDelay}
var backoffFunc = func(ctx context.Context, retries int) bool {
	d := backoffStrategy.Backoff(retries)
	timer := time.NewTimer(d)
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		timer.Stop()
		return false
	}
}

func init() {
	internal.HealthCheckFunc = clientHealthCheck
}

const healthCheckMethod = "/grpc.health.v1.Health/Watch"

// This function implements the protocol defined at:
// https://github.com/grpc/grpc/blob/master/doc/health-checking.md
func clientHealthCheck(ctx context.Context, newStream func(string) (interface{}, error), setConnectivityState func(connectivity.State), service string) error {
	tryCnt := 0

retryConnection:
	for {
		// Backs off if the connection has failed in some way without receiving a message in the previous retry.
		if tryCnt > 0 && !backoffFunc(ctx, tryCnt-1) {
			return nil
		}
		tryCnt++

		if ctx.Err() != nil {
			return nil
		}
		setConnectivityState(connectivity.Connecting)
		rawS, err := newStream(healthCheckMethod)
		if err != nil {
			continue retryConnection
		}

		s, ok := rawS.(grpc.ClientStream)
		// Ideally, this should never happen. But if it happens, the server is marked as healthy for LBing purposes.
		if !ok {
			setConnectivityState(connectivity.Ready)
			return fmt.Errorf("newStream returned %v (type %T); want grpc.ClientStream", rawS, rawS)
		}

		if err = s.SendMsg(&healthpb.HealthCheckRequest{Service: service}); err != nil && err != io.EOF {
			// Stream should have been closed, so we can safely continue to create a new stream.
			continue retryConnection
		}
		s.CloseSend()

		resp := new(healthpb.HealthCheckResponse)
		for {
			err = s.RecvMsg(resp)

			// Reports healthy for the LBing purposes if health check is not implemented in the server.
			if status.Code(err) == codes.Unimplemented {
				setConnectivityState(connectivity.Ready)
				return err
			}

			// Reports unhealthy if server's Watch method gives an error other than UNIMPLEMENTED.
			if err != nil {
				setConnectivityState(connectivity.TransientFailure)
				continue retryConnection
			}

			// As a message has been received, removes the need for backoff for the next retry by reseting the try count.
			tryCnt = 0
			if resp.Status == healthpb.HealthCheckResponse_SERVING {
				setConnectivityState(connectivity.Ready)
			} else {
				setConnectivityState(connectivity.TransientFailure)
			}
		}
	}
}
