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

// Command analyze-kas-traces downloads kube-apiserver logs from a Prow job and
// summarizes WatchList request traces and their correlated storage/APF phases.
//
// Usage:
//
//	go run ./hack/analyze-kas-traces [flags] <prow-job-downloads-url>
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	defaultOutputDir = "kas-logs"
	maxHTMLBytes     = 16 << 20
)

var (
	hrefPattern  = regexp.MustCompile(`(?i)href\s*=\s*"([^"]+)"`)
	tracePattern = regexp.MustCompile(`"(TRACE-[A-Z-]+)"`)
	httpPattern  = regexp.MustCompile(`"HTTP"`)
)

type trace struct {
	kind     string
	file     string
	line     int
	duration time.Duration
	fields   map[string]string
}

type apfRecord struct {
	file   string
	line   int
	fields map[string]string
}

func main() {
	outputDir := flag.String("out", defaultOutputDir, "directory for downloaded kube-apiserver logs")
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage: %s [flags] <prow-job-downloads-url>\n\n", os.Args[0])
		fmt.Fprintln(flag.CommandLine.Output(), "Downloads kube-apiserver logs, summarizes WatchList traces, and ranks the five slowest WatchList requests.")
		fmt.Fprintln(flag.CommandLine.Output(), "A URL ending in /downloads is resolved through the Prow job's artifacts index.")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 1 {
		flag.Usage()
		os.Exit(2)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	if err := run(ctx, flag.Arg(0), *outputDir, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "analyze-kas-traces:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, inputURL, outputDir string, output io.Writer) error {
	client := &http.Client{Timeout: 2 * time.Minute}
	jobURL, err := normalizeJobURL(inputURL)
	if err != nil {
		return err
	}

	jobIndex, err := getPage(ctx, client, jobURL)
	if err != nil {
		return fmt.Errorf("read Prow job index %s: %w", jobURL, err)
	}
	artifactsURL := findLink(jobURL, jobIndex, func(link *url.URL) bool {
		return strings.HasSuffix(strings.TrimRight(link.Path, "/"), "/artifacts")
	})
	if artifactsURL == nil {
		return fmt.Errorf("Prow job index %s has no artifacts directory", jobURL)
	}
	artifactsIndex, err := getPage(ctx, client, artifactsURL)
	if err != nil {
		return fmt.Errorf("read artifacts index %s: %w", artifactsURL, err)
	}
	linkURL := findLink(artifactsURL, artifactsIndex, func(link *url.URL) bool {
		return pathBase(link.Path) == "master-and-node-logs.link.txt"
	})
	if linkURL == nil {
		return fmt.Errorf("artifacts index %s has no master-and-node-logs.link.txt", artifactsURL)
	}
	linkData, err := get(ctx, client, linkURL.String(), 1<<20)
	if err != nil {
		return fmt.Errorf("read log directory pointer %s: %w", linkURL, err)
	}
	bucket, prefix, err := parseGCSPath(string(linkData))
	if err != nil {
		return fmt.Errorf("parse log directory pointer: %w", err)
	}

	logURLs := make(map[string]*url.URL)
	if err := findKASLogs(ctx, client, bucket, prefix, make(map[string]bool), logURLs, 0); err != nil {
		return err
	}
	if len(logURLs) == 0 {
		return fmt.Errorf("no kube-apiserver .log files found under gs://%s/%s", bucket, prefix)
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("create output directory %q: %w", outputDir, err)
	}

	localFiles := make([]string, 0, len(logURLs))
	for _, logURL := range sortedURLValues(logURLs) {
		name := logFileName(logURL, bucket, prefix)
		localPath := filepath.Join(outputDir, name)
		if err := download(ctx, client, logURL.String(), localPath); err != nil {
			return fmt.Errorf("download %s: %w", logURL, err)
		}
		localFiles = append(localFiles, localPath)
	}
	if _, err := fmt.Fprintf(output, "Downloaded %d kube-apiserver log(s) to %s\n", len(localFiles), outputDir); err != nil {
		return err
	}

	traces, counts, apfRecords, queueWaits, userAgents, err := scrape(localFiles)
	if err != nil {
		return err
	}
	printReport(output, traces, counts, apfRecords, queueWaits, userAgents)
	return nil
}

func normalizeJobURL(input string) (*url.URL, error) {
	u, err := url.Parse(input)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return nil, fmt.Errorf("expected an http(s) Prow job URL, got %q", input)
	}
	path := strings.TrimRight(u.Path, "/")
	if strings.HasSuffix(path, "/downloads") {
		path = strings.TrimSuffix(path, "/downloads")
	} else if strings.HasSuffix(path, "/artifacts") {
		path = strings.TrimSuffix(path, "/artifacts")
	}
	u.Path = strings.TrimRight(path, "/") + "/"
	u.RawPath = ""
	u.RawQuery = ""
	u.Fragment = ""
	return u, nil
}

func getPage(ctx context.Context, client *http.Client, u *url.URL) ([]byte, error) {
	return get(ctx, client, u.String(), maxHTMLBytes)
}

func get(ctx context.Context, client *http.Client, rawURL string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("response exceeded %d bytes", limit)
	}
	return body, nil
}

func findLink(base *url.URL, page []byte, keep func(*url.URL) bool) *url.URL {
	for _, match := range hrefPattern.FindAllSubmatch(page, -1) {
		raw := html.UnescapeString(string(match[1]))
		ref, err := url.Parse(raw)
		if err != nil {
			continue
		}
		link := base.ResolveReference(ref)
		if keep(link) {
			return link
		}
	}
	return nil
}

func parseGCSPath(value string) (bucket, prefix string, err error) {
	value = strings.TrimSpace(value)
	value = strings.TrimPrefix(value, "gs://")
	parts := strings.SplitN(strings.Trim(value, "/"), "/", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", fmt.Errorf("expected gs://<bucket>/<prefix>, got %q", value)
	}
	return parts[0], strings.Trim(parts[1], "/"), nil
}

func findKASLogs(ctx context.Context, client *http.Client, bucket, prefix string, visited map[string]bool, logs map[string]*url.URL, depth int) error {
	if depth > 8 {
		return fmt.Errorf("log directory nesting exceeds 8 levels below gs://%s/%s", bucket, prefix)
	}
	pageURL := &url.URL{Scheme: "https", Host: "gcsweb.k8s.io", Path: "/gcs/" + bucket + "/" + prefix + "/"}
	key := pageURL.String()
	if visited[key] {
		return nil
	}
	visited[key] = true
	page, err := getPage(ctx, client, pageURL)
	if err != nil {
		return fmt.Errorf("list GCS log directory gs://%s/%s: %w", bucket, prefix, err)
	}

	for _, match := range hrefPattern.FindAllSubmatch(page, -1) {
		ref, err := url.Parse(html.UnescapeString(string(match[1])))
		if err != nil {
			continue
		}
		link := pageURL.ResolveReference(ref)
		if link.Host == "gcsweb.k8s.io" && strings.HasPrefix(link.Path, "/gcs/"+bucket+"/"+prefix+"/") && strings.HasSuffix(link.Path, "/") {
			rel := strings.TrimPrefix(strings.TrimSuffix(link.Path, "/"), "/gcs/"+bucket+"/")
			if rel != prefix {
				if err := findKASLogs(ctx, client, bucket, rel, visited, logs, depth+1); err != nil {
					return err
				}
			}
			continue
		}
		if isKASLog(link) && belongsToGCSObject(link, bucket, prefix) {
			logs[link.String()] = link
		}
	}
	return nil
}

func isKASLog(u *url.URL) bool {
	base := strings.ToLower(pathBase(u.Path))
	return strings.Contains(base, "kube-apiserver") && strings.HasSuffix(base, ".log")
}

func belongsToGCSObject(u *url.URL, bucket, prefix string) bool {
	if u.Host != "storage.googleapis.com" {
		return false
	}
	objectPrefix := "/" + bucket + "/" + prefix + "/"
	return strings.HasPrefix(u.Path, objectPrefix)
}

func pathBase(path string) string {
	parts := strings.Split(strings.TrimRight(path, "/"), "/")
	if len(parts) == 0 {
		return ""
	}
	return parts[len(parts)-1]
}

func sortedURLValues(values map[string]*url.URL) []*url.URL {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]*url.URL, 0, len(keys))
	for _, key := range keys {
		result = append(result, values[key])
	}
	return result
}

func logFileName(u *url.URL, bucket, prefix string) string {
	relative := strings.TrimPrefix(u.Path, "/"+bucket+"/"+prefix+"/")
	name := strings.ReplaceAll(relative, "/", "_")
	name = filepath.Base(filepath.Clean(name))
	if name == "." || name == string(filepath.Separator) || name == "" {
		return "kube-apiserver.log"
	}
	return name
}

func download(ctx context.Context, client *http.Client, rawURL, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP %s", resp.Status)
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(file, resp.Body)
	closeErr := file.Close()
	return errors.Join(copyErr, closeErr)
}

func scrape(files []string) ([]trace, map[string]int, map[string]apfRecord, map[string]string, map[string]string, error) {
	var traces []trace
	counts := make(map[string]int)
	apfRecords := make(map[string]apfRecord)
	queueWaits := make(map[string]string)
	userAgents := make(map[string]string)
	for _, file := range files {
		input, err := os.Open(file)
		if err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("open downloaded log %q: %w", file, err)
		}
		scanner := bufio.NewScanner(input)
		scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
		lineNumber := 0
		for scanner.Scan() {
			lineNumber++
			line := scanner.Text()
			if match := httpPattern.FindStringIndex(line); match != nil {
				fields := parseFields(line[match[1]:])
				auditID := fields["audit-ID"]
				if auditID == "" {
					auditID = fields["auditID"]
				}
				if auditID != "" && fields["userAgent"] != "" {
					userAgents[auditID] = fields["userAgent"]
				}
				if auditID != "" && fields["fl_priorityandfairness"] != "" {
					apfRecords[auditID] = apfRecord{file: filepath.Base(file), line: lineNumber, fields: fields}
				}
			}
			if strings.Contains(strings.ToLower(filepath.Base(file)), "audit") && strings.HasPrefix(strings.TrimSpace(line), "{") {
				var event struct {
					AuditID     string            `json:"auditID"`
					Annotations map[string]string `json:"annotations"`
				}
				if json.Unmarshal([]byte(line), &event) == nil && event.AuditID != "" {
					if wait := event.Annotations["apiserver.latency.k8s.io/apf-queue-wait"]; wait != "" {
						queueWaits[event.AuditID] = wait
					}
				}
			}
			match := tracePattern.FindStringSubmatchIndex(line)
			if match == nil {
				continue
			}
			kind := line[match[2]:match[3]]
			if kind == "TRACE-LIST" {
				continue
			}
			fields := parseFields(line[match[1]:])
			entry := trace{kind: kind, file: filepath.Base(file), line: lineNumber, fields: fields}
			entry.duration = traceDuration(entry)
			traces = append(traces, entry)
			counts[kind]++
		}
		scanErr := scanner.Err()
		closeErr := input.Close()
		if err := errors.Join(scanErr, closeErr); err != nil {
			return nil, nil, nil, nil, nil, fmt.Errorf("read downloaded log %q: %w", file, err)
		}
	}
	return traces, counts, apfRecords, queueWaits, userAgents, nil
}

func parseFields(rest string) map[string]string {
	fields := make(map[string]string)
	for _, token := range splitFields(rest) {
		key, value, ok := strings.Cut(token, "=")
		if !ok {
			continue
		}
		value = strings.TrimSpace(value)
		if strings.HasPrefix(value, `"`) {
			if unquoted, err := strconv.Unquote(value); err == nil {
				value = unquoted
			} else {
				value = strings.Trim(value, `"`)
			}
		}
		fields[key] = value
	}
	return fields
}

func splitFields(value string) []string {
	var fields []string
	start := -1
	quoted, escaped := false, false
	for i, r := range value {
		if start < 0 {
			if r == ' ' || r == '\t' {
				continue
			}
			start = i
		}
		if escaped {
			escaped = false
			continue
		}
		if quoted && r == '\\' {
			escaped = true
			continue
		}
		if r == '"' {
			quoted = !quoted
			continue
		}
		if !quoted && (r == ' ' || r == '\t') {
			fields = append(fields, value[start:i])
			start = -1
		}
	}
	if start >= 0 {
		fields = append(fields, value[start:])
	}
	return fields
}

func traceDuration(entry trace) time.Duration {
	key := ""
	switch entry.kind {
	case "TRACE-LIST", "TRACE-WATCHLIST", "TRACE-CACHER", "TRACE-REFLECTOR", "TRACE-STREAMWATCHER":
		key = "total"
	case "TRACE-WATCHLIST-RV":
		key = "duration"
	}
	if key == "" {
		return 0
	}
	duration, err := time.ParseDuration(entry.fields[key])
	if err != nil {
		return 0
	}
	return duration
}

func printReport(output io.Writer, traces []trace, counts map[string]int, apfRecords map[string]apfRecord, queueWaits map[string]string, userAgents map[string]string) {
	watchListAuditIDs := make(map[string]struct{})
	for _, entry := range traces {
		if entry.kind == "TRACE-WATCHLIST" && entry.fields["auditID"] != "" {
			watchListAuditIDs[entry.fields["auditID"]] = struct{}{}
		}
	}
	watchListTraces := make([]trace, 0, len(traces))
	watchListCounts := make(map[string]int)
	for _, entry := range traces {
		_, relatedToWatchList := watchListAuditIDs[entry.fields["auditID"]]
		if entry.kind != "TRACE-WATCHLIST" && !relatedToWatchList {
			continue
		}
		watchListTraces = append(watchListTraces, entry)
		watchListCounts[entry.kind]++
	}
	traces = watchListTraces
	counts = watchListCounts

	fmt.Fprintf(output, "\nTrace entries: %d\n", len(traces))
	markers := make([]string, 0, len(counts))
	for marker := range counts {
		markers = append(markers, marker)
	}
	sort.Strings(markers)
	for _, marker := range markers {
		fmt.Fprintf(output, "  %-28s %d\n", marker, counts[marker])
	}

	byAuditID := make(map[string][]trace)
	var requests []trace
	for _, entry := range traces {
		if id := entry.fields["auditID"]; id != "" {
			byAuditID[id] = append(byAuditID[id], entry)
		}
		if entry.kind == "TRACE-WATCHLIST" && entry.duration > 0 {
			requests = append(requests, entry)
		}
	}
	printRequestStats(output, requests)
	printResourceVersionStats(output, traces)
	printWatchListPhaseStats(output, traces)
	printPhaseLegend(output)
	printSlowest(output, requests, byAuditID, apfRecords, queueWaits, userAgents)
}

func printRequestStats(output io.Writer, requests []trace) {
	fmt.Fprintf(output, "\nWatchList request traces (only requests above the apiserver trace threshold)\n")
	if len(requests) == 0 {
		fmt.Fprintln(output, "  No TRACE-WATCHLIST request summaries found.")
		return
	}
	sort.Slice(requests, func(i, j int) bool { return requests[i].duration < requests[j].duration })
	durations := make([]time.Duration, 0, len(requests))
	for _, entry := range requests {
		durations = append(durations, entry.duration)
	}
	fmt.Fprintf(output, "  %-16s n=%d avg=%s p50=%s p95=%s max=%s\n", "TRACE-WATCHLIST", len(durations), average(durations), percentile(durations, 0.50), percentile(durations, 0.95), durations[len(durations)-1])
}

func printResourceVersionStats(output io.Writer, traces []trace) {
	var etcdDurations []time.Duration
	requestRVCount := 0
	for _, entry := range traces {
		if entry.kind != "TRACE-WATCHLIST-RV" {
			continue
		}
		switch entry.fields["source"] {
		case "request":
			requestRVCount++
		case "etcd":
			if duration, err := time.ParseDuration(entry.fields["duration"]); err == nil {
				etcdDurations = append(etcdDurations, duration)
			}
		default:
			// Older traces were emitted only for GetCurrentResourceVersion calls.
			if entry.duration > 0 {
				etcdDurations = append(etcdDurations, entry.duration)
			}
		}
	}
	fmt.Fprintln(output, "\nWatchList resource-version resolution")
	if len(etcdDurations) == 0 {
		fmt.Fprintln(output, "  No etcd GetCurrentResourceVersion calls recorded.")
	} else {
		sort.Slice(etcdDurations, func(i, j int) bool { return etcdDurations[i] < etcdDurations[j] })
		fmt.Fprintf(output, "  etcd GetCurrentResourceVersion: n=%d avg=%s p50=%s p95=%s max=%s\n", len(etcdDurations), average(etcdDurations), percentile(etcdDurations, 0.50), percentile(etcdDurations, 0.95), etcdDurations[len(etcdDurations)-1])
	}
	if requestRVCount > 0 {
		fmt.Fprintf(output, "  resourceVersion supplied by request: n=%d\n", requestRVCount)
	}
}

func printWatchListPhaseStats(output io.Writer, traces []trace) {
	fields := []string{"setup", "sending", "encode", "flush", "other"}
	var watchLists []trace
	for _, entry := range traces {
		if entry.kind == "TRACE-WATCHLIST" {
			watchLists = append(watchLists, entry)
		}
	}
	if len(watchLists) == 0 {
		return
	}
	fmt.Fprintf(output, "\nWatchList phase averages across %d request trace(s)\n", len(watchLists))
	averages := make(map[string][]time.Duration, len(fields))
	for _, field := range fields {
		for _, entry := range watchLists {
			if value, err := time.ParseDuration(entry.fields[field]); err == nil {
				averages[field] = append(averages[field], value)
			}
		}
	}
	fmt.Fprintf(output, "  setup=%s sending=%s (encode=%s, flush=%s, other=%s)\n", average(averages["setup"]), average(averages["sending"]), average(averages["encode"]), average(averages["flush"]), average(averages["other"]))
}

func printPhaseLegend(output io.Writer) {
	fmt.Fprintln(output, "\nPhase definitions")
	fmt.Fprintln(output, "  setup: request arrival until the first initial watch event starts; this includes APF and watch initialization, so it is not queue time.")
	fmt.Fprintln(output, "  TRACE-WATCHLIST total is setup+sending. Sending runs from the first event through the end of the initial stream; encode and flush are parts of sending, not extra time to add.")
	fmt.Fprintln(output, "  objectEncode: serialize the Kubernetes object; watchEventEncode: serialize the WatchEvent envelope containing it.")
	fmt.Fprintln(output, "  framedWrite: add the stream frame and write downstream; gzip: compress bytes; responseWrite: write compressed bytes to the HTTP response.")
	fmt.Fprintln(output, "  Timers overlap: framedWrite is inside encoder total, and gzip/responseWrite measure work lower in that write path. Do not add them to encoder total.")
	fmt.Fprintln(output, "  apf_queue_wait is measured directly by APF. fl_priorityandfairness is total APF filter latency; apf_execution_time covers its execution phase. Do not subtract these for queue time.")
	fmt.Fprintln(output, "  watch_storage_init is the synchronous storage Watch call; watch_first_event_wait is the time from its return until the handler receives the first event.")
	fmt.Fprintln(output, "  TRACE-WATCHLIST-CACHE-SETUP breaks cacher Watch into resource-version lookup, cache-freshness wait, event-interval selection, and watcher registration.")
	fmt.Fprintln(output, "  watchlist_rv_source and watchlist_rv_lookup are request-scoped fields; TRACE-WATCHLIST-RV is emitted only when GetCurrentResourceVersion is called.")
}

func printSlowest(output io.Writer, requests []trace, byAuditID map[string][]trace, apfRecords map[string]apfRecord, queueWaits map[string]string, userAgents map[string]string) {
	fmt.Fprintln(output, "\nTop 5 slowest traced requests")
	if len(requests) == 0 {
		fmt.Fprintln(output, "  No request summaries to rank.")
		return
	}
	sort.Slice(requests, func(i, j int) bool {
		if requests[i].duration == requests[j].duration {
			return requests[i].fields["path"] < requests[j].fields["path"]
		}
		return requests[i].duration > requests[j].duration
	})
	limit := min(5, len(requests))
	for i, entry := range requests[:limit] {
		fmt.Fprintf(output, "  %d. %s %s %s", i+1, entry.duration, entry.kind, entry.fields["path"])
		if events := entry.fields["events"]; events != "" {
			fmt.Fprintf(output, " events=%s", events)
		}
		if auditID := entry.fields["auditID"]; auditID != "" {
			fmt.Fprintf(output, " auditID=%s", auditID)
		}
		fmt.Fprintf(output, " (%s:%d)\n", entry.file, entry.line)
		printRequestDetails(output, entry)
		if auditID := entry.fields["auditID"]; auditID != "" {
			if userAgent := userAgents[auditID]; userAgent != "" {
				fmt.Fprintf(output, "     user agent: %s\n", userAgent)
			} else {
				fmt.Fprintln(output, "     user agent: no correlated HTTP log")
			}
			printAPFDetails(output, auditID, apfRecords, queueWaits)
			foundResourceVersionTrace := false
			for _, related := range byAuditID[auditID] {
				if related.kind == "TRACE-WATCHLIST-RV" {
					foundResourceVersionTrace = true
				}
				if detail := relatedTraceSummary(related); detail != "" {
					fmt.Fprintf(output, "     %s\n", detail)
				}
			}
			if entry.kind == "TRACE-WATCHLIST" && !foundResourceVersionTrace && apfRecords[auditID].fields["watchlist_rv_source"] == "" {
				fmt.Fprintln(output, "     resource-version resolution: no TRACE-WATCHLIST-RV with this audit ID")
			}
		}
	}
}

func printAPFDetails(output io.Writer, auditID string, records map[string]apfRecord, queueWaits map[string]string) {
	record, ok := records[auditID]
	if !ok {
		fmt.Fprintln(output, "     APF: no correlated HTTP timing record")
		return
	}
	fields := record.fields
	fmt.Fprintf(output, "     APF: priority=%s flowSchema=%s filter=%s execution=%s init=%s", fields["apf_pl"], fields["apf_fs"], fieldDuration(trace{fields: fields}, "fl_priorityandfairness"), fieldDuration(trace{fields: fields}, "apf_execution_time"), fieldDuration(trace{fields: fields}, "apf_init_latency"))
	if wait := fields["apf_queue_wait"]; wait != "" {
		fmt.Fprintf(output, " queue-wait=%s (HTTP log)", wait)
	} else if wait, ok := queueWaits[auditID]; ok {
		fmt.Fprintf(output, " queue-wait=%s (audit annotation)", wait)
	} else {
		fmt.Fprint(output, " queue-wait=not recorded (logs lack per-request APF queue timing)")
	}
	fmt.Fprintf(output, " (%s:%d)\n", record.file, record.line)
	if fields["watch_storage_init"] != "" || fields["watch_first_event_wait"] != "" {
		fmt.Fprintf(output, "     watch setup: storage-init=%s first-event-wait=%s\n", fieldDuration(trace{fields: fields}, "watch_storage_init"), fieldDuration(trace{fields: fields}, "watch_first_event_wait"))
	}
	if fields["watchlist_rv_source"] != "" {
		fmt.Fprintf(output, "     watch resource-version: source=%s requested=%s lookup=%s\n", fields["watchlist_rv_source"], fields["watchlist_requested_rv"], fieldDuration(trace{fields: fields}, "watchlist_rv_lookup"))
	}
}

func printRequestDetails(output io.Writer, entry trace) {
	if entry.kind == "TRACE-LIST" {
		fmt.Fprintf(output, "     list=%s transform=%s mediaType=%s\n", fieldDuration(entry, "list"), fieldDuration(entry, "transform"), entry.fields["mediaType"])
		return
	}
	if value := entry.fields["uncompressedMB"]; value != "" {
		fmt.Fprintf(output, "     payload=%s MB uncompressed / %s MB compressed, encoding=%s, mediaType=%s\n", value, entry.fields["compressedMB"], entry.fields["contentEncoding"], entry.fields["mediaType"])
	}
	fmt.Fprintf(output, "     request phases: setup=%s sending=%s (encode=%s, flush=%s, other=%s)\n", fieldDuration(entry, "setup"), fieldDuration(entry, "sending"), fieldDuration(entry, "encode"), fieldDuration(entry, "flush"), fieldDuration(entry, "other"))
}

func relatedTraceSummary(entry trace) string {
	switch entry.kind {
	case "TRACE-WATCHLIST-RV":
		if entry.fields["source"] == "request" {
			return fmt.Sprintf("resource-version resolution=request-supplied rv=%s", entry.fields["requestedRV"])
		}
		if entry.fields["source"] == "etcd" {
			return fmt.Sprintf("resource-version resolution=etcd GetCurrentResourceVersion=%s rv=%s", fieldDuration(entry, "duration"), entry.fields["resourceVersion"])
		}
		return "resource-version etcd lookup=" + fieldDuration(entry, "duration")
	case "TRACE-CACHER":
		return fmt.Sprintf("watch-cache total=%s first-event=%s (next=%s convert=%s send=%s other=%s)", fieldDuration(entry, "total"), fieldDuration(entry, "firstEvent"), fieldDuration(entry, "next"), fieldDuration(entry, "convert"), fieldDuration(entry, "send"), fieldDuration(entry, "other"))
	case "TRACE-WATCHLIST-CACHE-SETUP":
		return fmt.Sprintf("watch-cache setup total=%s rv-lookup=%s source=%s requested-rv=%s required-rv=%s cache-fresh=%s cache-interval=%s register=%s other=%s", fieldDuration(entry, "total"), fieldDuration(entry, "rvLookup"), entry.fields["rvSource"], entry.fields["requestedRV"], entry.fields["requiredRV"], fieldDuration(entry, "cacheFresh"), fieldDuration(entry, "cacheInterval"), fieldDuration(entry, "register"), fieldDuration(entry, "other"))
	case "TRACE-WATCHLIST-ENCODER":
		return fmt.Sprintf("encoder total=%s object=%s watch-event=%s framed-write=%s other=%s", fieldDuration(entry, "total"), fieldDuration(entry, "objectEncode"), fieldDuration(entry, "watchEventEncode"), fieldDuration(entry, "framedWrite"), fieldDuration(entry, "other"))
	case "TRACE-WATCHLIST-GZIP":
		return fmt.Sprintf("gzip=%s response-write=%s response-flush=%s", fieldDuration(entry, "gzip"), fieldDuration(entry, "responseWrite"), fieldDuration(entry, "responseFlush"))
	default:
		return ""
	}
}

func fieldDuration(entry trace, key string) string {
	if entry.fields[key] == "" {
		return "n/a"
	}
	if duration, err := time.ParseDuration(entry.fields[key]); err == nil {
		return duration.String()
	}
	return entry.fields[key]
}

func average(values []time.Duration) time.Duration {
	if len(values) == 0 {
		return 0
	}
	var total time.Duration
	for _, value := range values {
		total += value
	}
	return total / time.Duration(len(values))
}

func percentile(sortedValues []time.Duration, percentile float64) time.Duration {
	if len(sortedValues) == 0 {
		return 0
	}
	index := int(math.Ceil(float64(len(sortedValues))*percentile)) - 1
	if index < 0 {
		index = 0
	}
	return sortedValues[index]
}
