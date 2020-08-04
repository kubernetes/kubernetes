// +build go1.6,!go1.7

/*
 *
 * Copyright 2016 gRPC authors.
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

package grpc

import (
	"fmt"
	"io"
	"net"
	"net/http"

	"golang.org/x/net/context"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/transport"
)

// dialContext connects to the address on the named network.
func dialContext(ctx context.Context, network, address string) (net.Conn, error) {
	return (&net.Dialer{Cancel: ctx.Done()}).Dial(network, address)
}

func sendHTTPRequest(ctx context.Context, req *http.Request, conn net.Conn) error {
	req.Cancel = ctx.Done()
	if err := req.Write(conn); err != nil {
		return fmt.Errorf("failed to write the HTTP request: %v", err)
	}
	return nil
}

// toRPCErr converts an error into an error from the status package.
func toRPCErr(err error) error {
	if err == nil || err == io.EOF {
		return err
	}
	if _, ok := status.FromError(err); ok {
		return err
	}
	switch e := err.(type) {
	case transport.StreamError:
		return status.Error(e.Code, e.Desc)
	case transport.ConnectionError:
		return status.Error(codes.Unavailable, e.Desc)
	default:
		switch err {
		case context.DeadlineExceeded:
			return status.Error(codes.DeadlineExceeded, err.Error())
		case context.Canceled:
			return status.Error(codes.Canceled, err.Error())
		}
	}
	return status.Error(codes.Unknown, err.Error())
}
