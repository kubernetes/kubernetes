// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package httpcache

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/bartventer/httpcache/internal"
)

type Option interface {
	apply(*transport)
}

type optionFunc func(*transport)

func (f optionFunc) apply(r *transport) {
	f(r)
}

// WithUpstream sets the underlying [http.RoundTripper] used for upstream/origin
// requests. Default: [http.DefaultTransport].
//
// Note: Headers added by the upstream roundtripper (e.g., authentication
// headers) do not affect cache key calculation or Vary header matching
// (RFC 9111 §4.1). The cache operates on the original client request, not the
// mutated request seen by the upstream roundtripper.
func WithUpstream(upstream http.RoundTripper) Option {
	return optionFunc(func(r *transport) {
		r.upstream = upstream
	})
}

// WithSWRTimeout sets the timeout for Stale-While-Revalidate requests;
// default: [DefaultSWRTimeout].
func WithSWRTimeout(timeout time.Duration) Option {
	return optionFunc(func(r *transport) {
		r.swrTimeout = timeout
	})
}

// WithLogger sets the logger for debug output; default:
// [slog.New]([slog.DiscardHandler]).
func WithLogger(logger *slog.Logger) Option {
	return optionFunc(func(r *transport) {
		if logger != nil {
			r.logger = internal.NewLogger(logger.Handler())
		}
	})
}
