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
	"context"
	"log/slog"
	"net/http"
	"runtime"
	"slices"
	"time"
)

type contextKey struct{}

func (c contextKey) String() string { return "httpcache context key" }

var TraceIDKey contextKey

func TraceIDFromContext(ctx context.Context) (string, bool) {
	traceID, ok := ctx.Value(TraceIDKey).(string)
	return traceID, ok
}

/*
	+-------------------------------+
	| Log Field Types   			|
	+-------------------------------+
*/

type (
	LogEntry struct {
		CacheStatus  string
		URLKey       string
		MiscProvider MiscProvider
		Error        error
	}

	Misc struct {
		CCReq     CCRequestDirectives
		CCResp    CCResponseDirectives
		Stored    *Response
		Freshness *Freshness
		Refs      ResponseRefs
		RefIndex  int
	}
)

func (m Misc) LogValue() slog.Value {
	attrs := make([]slog.Attr, 0, 5)
	if m.CCReq != nil {
		attrs = append(attrs, slog.Any("cc_request", m.CCReq))
	}
	if m.CCResp != nil {
		attrs = append(attrs, slog.Any("cc_response", m.CCResp))
	}
	if m.Stored != nil {
		attrs = append(attrs, slog.Any("stored", m.Stored))
	}
	if m.Freshness != nil {
		attrs = append(attrs, slog.Any("freshness", m.Freshness))
	}
	if m.RefIndex >= 0 && m.RefIndex < len(m.Refs) {
		attrs = append(attrs, slog.Any("ref", m.Refs[m.RefIndex]))
	}
	return slog.GroupValue(attrs...)
}

/*
	+-------------------------------+
	| Provider Interfaces			|
	+-------------------------------+
*/

type (
	LogProvider interface {
		MakeLog() (CacheStatus, *http.Request, LogEntry)
	}
	MiscProvider interface {
		MakeMisc() Misc
	}
)

type (
	LogFunc      func() (CacheStatus, *http.Request, LogEntry)
	MiscFunc     func() Misc
	LogValueFunc func() slog.Value
)

func (f LogFunc) MakeLog() (event CacheStatus, req *http.Request, cl LogEntry) {
	return f()
}

func (f MiscFunc) MakeMisc() Misc           { return f() }
func (f LogValueFunc) LogValue() slog.Value { return f() }

/*
	+-------------------------------+
	| Logger Implementation			|
	+-------------------------------+
*/

type Logger struct{ handler slog.Handler }

func NewLogger(h slog.Handler) *Logger { return &Logger{handler: h} }

func (l *Logger) Handler() slog.Handler { return l.handler }

func (l *Logger) Enabled(ctx context.Context, level slog.Level) bool {
	return l.handler.Enabled(ctx, level)
}

func (l *Logger) With(attrs ...slog.Attr) *Logger {
	if len(attrs) == 0 {
		return l
	}
	return &Logger{handler: l.handler.WithAttrs(attrs)}
}

func (l *Logger) WithGroup(name string) *Logger {
	if name == "" {
		return l
	}
	return &Logger{handler: l.handler.WithGroup(name)}
}

func (l *Logger) LogCacheHit(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Hit; served from cache.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusHit, req, LogEntry{
				CacheStatus:  CacheStatusHit.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheMiss(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Miss; served from origin.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusMiss, req, LogEntry{
				CacheStatus:  CacheStatusMiss.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheStale(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Stale; served from cache.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusStale, req, LogEntry{
				CacheStatus:  CacheStatusStale.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheStaleIfError(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Stale; served from cache; stale-if-error policy applied.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusStale, req, LogEntry{
				CacheStatus:  CacheStatusStale.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheStaleRevalidate(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Stale; served from cache; revalidating.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusStale, req, LogEntry{
				CacheStatus:  CacheStatusStale.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheRevalidated(req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		"Revalidated; served cached response.",
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusRevalidated, req, LogEntry{
				CacheStatus:  CacheStatusRevalidated.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheBypass(msg string, req *http.Request, urlKey string, mp MiscProvider) {
	l.logCache(
		req.Context(),
		slog.LevelDebug,
		msg,
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatusBypass, req, LogEntry{
				CacheStatus:  CacheStatusBypass.Value,
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        nil,
			}
		}),
	)
}

func (l *Logger) LogCacheError(
	msg string,
	err error,
	req *http.Request,
	urlKey string,
	mp MiscProvider,
) {
	l.logCache(
		req.Context(),
		slog.LevelWarn,
		msg,
		LogFunc(func() (CacheStatus, *http.Request, LogEntry) {
			return CacheStatus{Value: "error"}, req, LogEntry{
				CacheStatus:  "error",
				URLKey:       urlKey,
				MiscProvider: mp,
				Error:        err,
			}
		}),
	)
}

func (l *Logger) logCache(ctx context.Context, level slog.Level, msg string, lp LogProvider) {
	if !l.handler.Enabled(ctx, level) {
		return
	}
	event, req, cl := lp.MakeLog()
	attrs := make([]slog.Attr, 0, 7)
	attrs = append(attrs,
		slog.String("service", "httpcache"),
		slog.String("event", event.Value),
	)
	if cl.Error != nil {
		attrs = append(attrs, slog.Any("error", cl.Error))
	}
	if tracedID, ok := TraceIDFromContext(ctx); ok && tracedID != "" {
		attrs = append(attrs, slog.String("trace_id", tracedID))
	}
	attrs = append(attrs,
		groupAttrs("request",
			slog.String("method", req.Method),
			slog.String("url", req.URL.String()),
			slog.String("host", req.Host),
		),
		groupAttrs("cache",
			slog.String("status", cl.CacheStatus),
			slog.String("url_key", cl.URLKey),
		),
	)
	if mp := cl.MiscProvider; mp != nil {
		attrs = append(attrs, slog.Any("misc", mp.MakeMisc()))
	}
	if cap(attrs) > len(attrs) {
		attrs = slices.Clip(attrs)
	}
	var pcs [1]uintptr
	runtime.Callers(3, pcs[:])
	r := slog.NewRecord(time.Now(), level, msg, pcs[0])
	r.AddAttrs(attrs...)
	_ = l.handler.Handle(ctx, r)
}

/*
	+-------------------------------+
	| Helpers  						|
	+-------------------------------+
*/

func groupAttrs(key string, args ...slog.Attr) slog.Attr {
	return slog.Attr{Key: key, Value: slog.GroupValue(args...)}
}
