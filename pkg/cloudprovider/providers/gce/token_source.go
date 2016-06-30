/*
Copyright 2015 The Kubernetes Authors.

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

package gce

import (
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/util/flowcontrol"

	"github.com/prometheus/client_golang/prometheus"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/googleapi"
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

type AltTokenSource struct {
	oauthClient *http.Client
	tokenURL    string
	tokenBody   string
	throttle    flowcontrol.RateLimiter
}

func (a *AltTokenSource) Token() (*oauth2.Token, error) {
	a.throttle.Accept()
	getTokenCounter.Inc()
	t, err := a.token()
	if err != nil {
		getTokenFailCounter.Inc()
	}
	return t, err
}

func (a *AltTokenSource) token() (*oauth2.Token, error) {
	req, err := http.NewRequest("POST", a.tokenURL, strings.NewReader(a.tokenBody))
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
		AccessToken string    `json:"accessToken"`
		ExpireTime  time.Time `json:"expireTime"`
	}
	if err := json.NewDecoder(res.Body).Decode(&tok); err != nil {
		return nil, err
	}
	return &oauth2.Token{
		AccessToken: tok.AccessToken,
		Expiry:      tok.ExpireTime,
	}, nil
}

func NewAltTokenSource(tokenURL, tokenBody string) oauth2.TokenSource {
	client := oauth2.NewClient(oauth2.NoContext, google.ComputeTokenSource(""))
	a := &AltTokenSource{
		oauthClient: client,
		tokenURL:    tokenURL,
		tokenBody:   tokenBody,
		throttle:    flowcontrol.NewTokenBucketRateLimiter(tokenURLQPS, tokenURLBurst),
	}
	return oauth2.ReuseTokenSource(nil, a)
}
