/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package gce_cloud

import (
	"encoding/json"
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/util"

	"code.google.com/p/google-api-go-client/googleapi"
	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	// Max QPS to allow through to the token URL.
	tokenURLQPS = .05 // back off to once every 20 seconds when failing
	// Maximum burst of requests to token URL before limiting.
	tokenURLBurst = 3
)

var (
	getTokenCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "get_token_count",
			Help: "Counter of total Token() requests to the alternate token source",
		},
	)
	getTokenFailCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "get_token_fail_count",
			Help: "Counter of failed Token() requests to the alternate token source",
		},
	)
)

func init() {
	prometheus.MustRegister(getTokenCounter)
	prometheus.MustRegister(getTokenFailCounter)
}

type altTokenSource struct {
	oauthClient *http.Client
	tokenURL    string
	throttle    util.RateLimiter
}

func (a *altTokenSource) Token() (*oauth2.Token, error) {
	a.throttle.Accept()
	getTokenCounter.Inc()
	t, err := a.token()
	if err != nil {
		getTokenFailCounter.Inc()
	}
	return t, err
}

func (a *altTokenSource) token() (*oauth2.Token, error) {
	req, err := http.NewRequest("GET", a.tokenURL, nil)
	if err != nil {
		return nil, err
	}
	res, err := a.oauthClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	var tok struct {
		AccessToken       string `json:"accessToken"`
		ExpiryTimeSeconds int64  `json:"expiryTimeSeconds,string"`
	}
	if err := json.NewDecoder(res.Body).Decode(&tok); err != nil {
		return nil, err
	}
	return &oauth2.Token{
		AccessToken: tok.AccessToken,
		Expiry:      time.Unix(tok.ExpiryTimeSeconds, 0),
	}, nil
}

func newAltTokenSource(tokenURL string) oauth2.TokenSource {
	client := oauth2.NewClient(oauth2.NoContext, google.ComputeTokenSource(""))
	a := &altTokenSource{
		oauthClient: client,
		tokenURL:    tokenURL,
		throttle:    util.NewTokenBucketRateLimiter(tokenURLQPS, tokenURLBurst),
	}
	return oauth2.ReuseTokenSource(nil, a)
}
