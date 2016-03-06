/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package gcp

import (
	"net/http"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"k8s.io/kubernetes/pkg/client/restclient"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("gcp", newGCPAuthProvider); err != nil {
		glog.Fatalf("Failed to register gcp auth plugin: %v", err)
	}
}

type gcpAuthProvider struct {
	tokenSource oauth2.TokenSource
}

func newGCPAuthProvider() (restclient.AuthProvider, error) {
	ts, err := google.DefaultTokenSource(context.TODO(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, err
	}
	return &gcpAuthProvider{ts}, nil
}

func (g *gcpAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &oauth2.Transport{
		Source: g.tokenSource,
		Base:   rt,
	}
}
