/*
Copyright 2026 The Kubernetes Authors.

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

package etcd3

import (
	"bytes"
	"context"
	"flag"
	"os"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

func TestTracingComparison(t *testing.T) {
	var buf bytes.Buffer
	flags := &flag.FlagSet{}
	klog.InitFlags(flags)
	if err := flags.Set("v", "3"); err != nil {
		t.Fatal(err)
	}
	klog.LogToStderr(false)
	klog.SetOutput(&buf)
	defer func() {
		klog.LogToStderr(true)
		klog.SetOutput(nil)
	}()

	ctx := context.Background()
	ctx = audit.WithAuditContext(ctx)
	audit.WithAuditID(ctx, "test-audit-id-123")

	ctx, span := tracing.Start(ctx, "a")

	var wg sync.WaitGroup

	// 1. Run utiltrace in a sub-goroutine and capture log output
	wg.Add(1)
	go func() {
		defer wg.Done()
		runUtilTrace(ctx, "/pods/test", "core", "pods")
	}()
	wg.Wait()
	span.End(time.Minute)

	utLog := buf.String()
	buf.Reset()

	// 2. Run locklessTraceLog in a sub-goroutine and capture log output
	wg.Add(1)
	go func() {
		defer wg.Done()
		runLocklessTrace(ctx, "/pods/test", "core", "pods")
	}()
	wg.Wait()

	ltLog := buf.String()

	t.Logf("UtilTrace output:\n%s", utLog)
	t.Logf("LocklessTraceLog output:\n%s", ltLog)

	// 3. Normalize both logs and run a line-by-line diff comparison using cmp.Diff
	utLines := normalizeLog(utLog)
	ltLines := normalizeLog(ltLog)

	if diff := cmp.Diff(utLines, ltLines); diff != "" {
		t.Errorf("Normalized log output diff mismatch (-utiltrace +locklessTraceLog):\n%s", diff)
	}
}

func normalizeLog(log string) string {
	if idx := strings.Index(log, "] "); idx != -1 {
		log = log[idx+2:]
	}
	// 1. Replace date/timestamps in headers: (22-Jul-2026 15:12:12.126) -> (TIMESTAMP)
	reDate := regexp.MustCompile(`\(\d{2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2}\.\d{3}\)`)
	log = reDate.ReplaceAllString(log, "(TIMESTAMP)")

	// 2. Replace step timestamps: (15:12:12.131) -> (TIMESTAMP)
	reTimestamp := regexp.MustCompile(`\(\d{2}:\d{2}:\d{2}\.\d{3}\)`)
	log = reTimestamp.ReplaceAllString(log, "(TIMESTAMP)")

	// 3. Replace durations in END line: [20.410901ms] -> [DURATION]
	reDuration := regexp.MustCompile(`\[\d+(\.\d+)?[a-zµ]+\]`)
	log = reDuration.ReplaceAllString(log, "[DURATION]")

	return log
}

func runUtilTrace(ctx context.Context, key, group, resource string) {
	ctx, span := tracing.Start(ctx, "Get etcd3",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.String("key", key),
		attribute.String("group", group),
		attribute.String("resource", resource),
	)
	defer span.End(time.Millisecond)

	time.Sleep(5 * time.Millisecond)
	span.AddEvent("Get call succeeded")
	time.Sleep(10 * time.Millisecond)
	span.AddEvent("TransformFromStorage succeeded")
	time.Sleep(5 * time.Millisecond)
	span.AddEvent("Decode succeeded", attribute.Int("len", 100))
}

func runLocklessTrace(ctx context.Context, key, group, resource string) {
	span := newLocklessTraceLog(ctx, "Get etcd3",
		attribute.String("audit-id", audit.GetAuditIDTruncated(ctx)),
		attribute.String("key", key),
		attribute.String("group", group),
		attribute.String("resource", resource),
	)
	defer span.End(time.Millisecond)

	time.Sleep(5 * time.Millisecond)
	span.AddEvent("Get call succeeded")
	time.Sleep(10 * time.Millisecond)
	span.AddEvent("TransformFromStorage succeeded")
	time.Sleep(5 * time.Millisecond)
	span.AddEvent("Decode succeeded", attribute.Int("len", 100))
}

func BenchmarkUtilTraceParallel(b *testing.B) {
	tempFile, err := os.CreateTemp(".", "trace_bench_*.log")
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}()

	klog.LogToStderr(false)
	klog.SetOutput(tempFile)
	defer func() {
		klog.LogToStderr(true)
		klog.SetOutput(nil)
	}()

	flags := &flag.FlagSet{}
	klog.InitFlags(flags)
	if err := flags.Set("v", "3"); err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	ctx = audit.WithAuditContext(ctx)
	audit.WithAuditID(ctx, "benchmark-audit-id")

	b.SetParallelism(1000)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ctx, span := tracing.Start(ctx, "a")
			runUtilTrace(ctx, "/pods/test", "core", "pods")
			span.End(time.Minute)
		}
	})
}

func BenchmarkLocklessTraceParallel(b *testing.B) {
	tempFile, err := os.CreateTemp(".", "trace_bench_*.log")
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		tempFile.Close()
		os.Remove(tempFile.Name())
	}()

	klog.LogToStderr(false)
	klog.SetOutput(tempFile)
	defer func() {
		klog.LogToStderr(true)
		klog.SetOutput(nil)
	}()

	flags := &flag.FlagSet{}
	klog.InitFlags(flags)
	if err := flags.Set("v", "3"); err != nil {
		b.Fatal(err)
	}

	ctx := context.Background()
	ctx = audit.WithAuditContext(ctx)
	audit.WithAuditID(ctx, "benchmark-audit-id")

	b.SetParallelism(1000)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			ctx, span := tracing.Start(ctx, "a")
			runLocklessTrace(ctx, "/pods/test", "core", "pods")
			span.End(time.Minute)
		}
	})
}

