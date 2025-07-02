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

package internal

import (
	"log/slog"
	"net/http"
)

const (
	CacheStatusHeader = "X-Httpcache-Status"
	FromCacheHeader   = "X-From-Cache" // Deprecated: use [CacheStatusHeader] instead
)

type CacheStatus struct {
	Value string
	// Value for compatibility with github.com/gregjones/httpcache:
	// 	"1" means served from cache (less specific "HIT")
	// 	"" means not served from cache (less specific "MISS")
	//
	// Deprecated: only used for compatibility with unmaintained (still widely
	// used) github.com/gregjones/httpcache; use Value instead.
	Legacy string
}

func (s CacheStatus) ApplyTo(header http.Header) {
	header.Set(CacheStatusHeader, s.Value)
	if s.Legacy != "" {
		header.Set(FromCacheHeader, s.Legacy)
	}
}

var _ slog.LogValuer = (*CacheStatus)(nil)

func (s CacheStatus) LogValue() slog.Value {
	return slog.GroupValue(
		slog.String("value", s.Value),
		slog.Bool("from_cache", s.Legacy == FromCache),
	)
}

const (
	FromCache    = "1"
	NotFromCache = ""
)

var (
	CacheStatusHit         = CacheStatus{"HIT", FromCache}         // served from cache
	CacheStatusMiss        = CacheStatus{"MISS", NotFromCache}     // served from origin
	CacheStatusStale       = CacheStatus{"STALE", FromCache}       // served from cache but stale
	CacheStatusRevalidated = CacheStatus{"REVALIDATED", FromCache} // revalidated with origin server
	CacheStatusBypass      = CacheStatus{"BYPASS", NotFromCache}   // cache bypassed
)
