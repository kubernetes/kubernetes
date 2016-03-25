// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"bytes"
	"io/ioutil"
	"net/http"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
)

type ProtoClient struct {
	client    *http.Client
	endpoint  string
	userAgent string
}

func (c *ProtoClient) Call(ctx context.Context, method string, req, resp proto.Message) error {
	payload, err := proto.Marshal(req)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequest("POST", c.endpoint+method, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/x-protobuf")
	if ua := c.userAgent; ua != "" {
		httpReq.Header.Set("User-Agent", ua)
	}

	errc := make(chan error, 1)
	cancel := makeReqCancel(httpReq)

	go func() {
		r, err := c.client.Do(httpReq)
		if err != nil {
			errc <- err
			return
		}
		defer r.Body.Close()

		body, err := ioutil.ReadAll(r.Body)
		if r.StatusCode != http.StatusOK {
			err = &ErrHTTP{
				StatusCode: r.StatusCode,
				Body:       body,
				err:        err,
			}
		}
		if err != nil {
			errc <- err
			return
		}
		errc <- proto.Unmarshal(body, resp)
	}()

	select {
	case <-ctx.Done():
		cancel(c.client.Transport) // Cancel the HTTP request.
		return ctx.Err()
	case err := <-errc:
		return err
	}
}
