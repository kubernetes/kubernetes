// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package transport

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	authchallenge "github.com/docker/distribution/registry/client/auth/challenge"
	"github.com/google/go-containerregistry/pkg/logs"
	"github.com/google/go-containerregistry/pkg/name"
)

// 300ms is the default fallback period for go's DNS dialer but we could make this configurable.
var fallbackDelay = 300 * time.Millisecond

type Challenge struct {
	Scheme string

	// Following the challenge there are often key/value pairs
	// e.g. Bearer service="gcr.io",realm="https://auth.gcr.io/v36/tokenz"
	Parameters map[string]string

	// Whether we had to use http to complete the Ping.
	Insecure bool
}

// Ping does a GET /v2/ against the registry and returns the response.
func Ping(ctx context.Context, reg name.Registry, t http.RoundTripper) (*Challenge, error) {
	// This first attempts to use "https" for every request, falling back to http
	// if the registry matches our localhost heuristic or if it is intentionally
	// set to insecure via name.NewInsecureRegistry.
	schemes := []string{"https"}
	if reg.Scheme() == "http" {
		schemes = append(schemes, "http")
	}
	if len(schemes) == 1 {
		return pingSingle(ctx, reg, t, schemes[0])
	}
	return pingParallel(ctx, reg, t, schemes)
}

func pingSingle(ctx context.Context, reg name.Registry, t http.RoundTripper, scheme string) (*Challenge, error) {
	client := http.Client{Transport: t}
	url := fmt.Sprintf("%s://%s/v2/", scheme, reg.RegistryStr())
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req.WithContext(ctx))
	if err != nil {
		return nil, err
	}
	defer func() {
		// By draining the body, make sure to reuse the connection made by
		// the ping for the following access to the registry
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}()

	insecure := scheme == "http"

	switch resp.StatusCode {
	case http.StatusOK:
		// If we get a 200, then no authentication is needed.
		return &Challenge{
			Insecure: insecure,
		}, nil
	case http.StatusUnauthorized:
		if challenges := authchallenge.ResponseChallenges(resp); len(challenges) != 0 {
			// If we hit more than one, let's try to find one that we know how to handle.
			wac := pickFromMultipleChallenges(challenges)
			return &Challenge{
				Scheme:     wac.Scheme,
				Parameters: wac.Parameters,
				Insecure:   insecure,
			}, nil
		}
		// Otherwise, just return the challenge without parameters.
		return &Challenge{
			Scheme:   resp.Header.Get("WWW-Authenticate"),
			Insecure: insecure,
		}, nil
	default:
		return nil, CheckError(resp, http.StatusOK, http.StatusUnauthorized)
	}
}

// Based on the golang happy eyeballs dialParallel impl in net/dial.go.
func pingParallel(ctx context.Context, reg name.Registry, t http.RoundTripper, schemes []string) (*Challenge, error) {
	returned := make(chan struct{})
	defer close(returned)

	type pingResult struct {
		*Challenge
		error
		primary bool
		done    bool
	}

	results := make(chan pingResult)

	startRacer := func(ctx context.Context, scheme string) {
		pr, err := pingSingle(ctx, reg, t, scheme)
		select {
		case results <- pingResult{Challenge: pr, error: err, primary: scheme == "https", done: true}:
		case <-returned:
			if pr != nil {
				logs.Debug.Printf("%s lost race", scheme)
			}
		}
	}

	var primary, fallback pingResult

	primaryCtx, primaryCancel := context.WithCancel(ctx)
	defer primaryCancel()
	go startRacer(primaryCtx, schemes[0])

	fallbackTimer := time.NewTimer(fallbackDelay)
	defer fallbackTimer.Stop()

	for {
		select {
		case <-fallbackTimer.C:
			fallbackCtx, fallbackCancel := context.WithCancel(ctx)
			defer fallbackCancel()
			go startRacer(fallbackCtx, schemes[1])

		case res := <-results:
			if res.error == nil {
				return res.Challenge, nil
			}
			if res.primary {
				primary = res
			} else {
				fallback = res
			}
			if primary.done && fallback.done {
				return nil, multierrs{primary.error, fallback.error}
			}
			if res.primary && fallbackTimer.Stop() {
				// Primary failed and we haven't started the fallback,
				// reset time to start fallback immediately.
				fallbackTimer.Reset(0)
			}
		}
	}
}

func pickFromMultipleChallenges(challenges []authchallenge.Challenge) authchallenge.Challenge {
	// It might happen there are multiple www-authenticate headers, e.g. `Negotiate` and `Basic`.
	// Picking simply the first one could result eventually in `unrecognized challenge` error,
	// that's why we're looping through the challenges in search for one that can be handled.
	allowedSchemes := []string{"basic", "bearer"}

	for _, wac := range challenges {
		currentScheme := strings.ToLower(wac.Scheme)
		for _, allowed := range allowedSchemes {
			if allowed == currentScheme {
				return wac
			}
		}
	}

	return challenges[0]
}

type multierrs []error

func (m multierrs) Error() string {
	var b strings.Builder
	hasWritten := false
	for _, err := range m {
		if hasWritten {
			b.WriteString("; ")
		}
		hasWritten = true
		b.WriteString(err.Error())
	}
	return b.String()
}

func (m multierrs) As(target any) bool {
	for _, err := range m {
		if errors.As(err, target) {
			return true
		}
	}
	return false
}

func (m multierrs) Is(target error) bool {
	for _, err := range m {
		if errors.Is(err, target) {
			return true
		}
	}
	return false
}
