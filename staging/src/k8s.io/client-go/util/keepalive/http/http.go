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

package http

import (
	"fmt"
	"net/http"
)

type HttpPinger struct {
	httpClient *http.Client
	scheme     string
	hostPort   string
}

func NewHttpPinger(httpClient *http.Client, scheme, hostPort string) HttpPinger {
	return HttpPinger{
		httpClient: httpClient,
		scheme:     scheme,
		hostPort:   hostPort,
	}
}

func (p HttpPinger) Ping() error {
	_, err := p.httpClient.Get(fmt.Sprintf("%s://%s/healthz", p.scheme, p.hostPort))
	return err
}
