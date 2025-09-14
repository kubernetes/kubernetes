/*
Copyright 2024 The Kubernetes Authors.

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

package statusz

import (
	"fmt"
	"html"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/component-base/compatibility"
	"k8s.io/component-base/zpages/httputil"
	"k8s.io/klog/v2"
)

var (
	delimiters            = []string{":", ": ", "=", " "}
	nonDebuggingEndpoints = map[string]bool{
		"/apis":        true,
		"/api":         true,
		"/openid":      true,
		"/openapi":     true,
		"/.well-known": true,
	}
)

const DefaultStatuszPath = "/statusz"

const headerFmt = `
%s statusz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.
`

type mux interface {
	Handle(path string, handler http.Handler)
}

type ListedPathsOption []string

func NewRegistry(effectiveVersion compatibility.EffectiveVersion, opts ...func(*registry)) statuszRegistry {
	r := &registry{effectiveVersion: effectiveVersion}
	for _, opt := range opts {
		opt(r)
	}

	return r
}

func Install(m mux, componentName string, reg statuszRegistry) {
	m.Handle(DefaultStatuszPath, handleStatusz(componentName, reg))
}

func handleStatusz(componentName string, reg statuszRegistry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !httputil.AcceptableMediaType(r) {
			http.Error(w, httputil.ErrUnsupportedMediaType.Error(), http.StatusNotAcceptable)
			return
		}

		fmt.Fprintf(w, headerFmt, componentName)
		data, err := populateStatuszData(reg, componentName)
		if err != nil {
			klog.Errorf("error while populating statusz data: %v", err)
			http.Error(w, "error while populating statusz data", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprint(w, data)
	}
}

func populateStatuszData(reg statuszRegistry, componentName string) (string, error) {
	randomIndex := rand.Intn(len(delimiters))
	delim := html.EscapeString(delimiters[randomIndex])
	startTime := html.EscapeString(reg.processStartTime().Format(time.UnixDate))
	uptime := html.EscapeString(uptime(reg.processStartTime()))
	goVersion := html.EscapeString(reg.goVersion())
	binaryVersion := html.EscapeString(reg.binaryVersion().String())

	var emulationVersion string
	if reg.emulationVersion() != nil {
		emulationVersion = fmt.Sprintf(`Emulation version%s %s`, delim, html.EscapeString(reg.emulationVersion().String()))
	}
	paths := aggregatePaths(reg.paths())
	if paths != "" {
		paths = fmt.Sprintf(`Paths%s %s`, delim, html.EscapeString(paths))
	}

	status := fmt.Sprintf(`
Started%[1]s %[2]s
Up%[1]s %[3]s
Go version%[1]s %[4]s
Binary version%[1]s %[5]s
%[6]s
%[7]s
`, delim, startTime, uptime, goVersion, binaryVersion, emulationVersion, paths)

	return status, nil
}

func uptime(t time.Time) string {
	upSince := int64(time.Since(t).Seconds())
	return fmt.Sprintf("%d hr %02d min %02d sec",
		upSince/3600, (upSince/60)%60, upSince%60)
}

func aggregatePaths(listedPaths []string) string {
	paths := make(map[string]bool)
	for _, listedPath := range listedPaths {
		folder := "/" + strings.Split(listedPath, "/")[1]
		if !paths[folder] && !nonDebuggingEndpoints[folder] {
			paths[folder] = true
		}
	}

	var sortedPaths []string
	for p := range paths {
		sortedPaths = append(sortedPaths, p)
	}
	sort.Strings(sortedPaths)

	var path string
	for _, p := range sortedPaths {
		path += " " + p
	}

	return path
}
