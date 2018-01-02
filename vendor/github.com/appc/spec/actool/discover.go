// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/json"
	"fmt"
	"runtime"
	"strings"

	"github.com/appc/spec/discovery"
)

var (
	outputJson  bool
	cmdDiscover = &Command{
		Name:        "discover",
		Description: "Discover the download URLs for an app",
		Summary:     "Discover the download URLs for one or more app container images",
		Usage:       "[--json] [--port] [--insecure] APP...",
		Run:         runDiscover,
	}
	flagPort uint
)

func init() {
	cmdDiscover.Flags.BoolVar(&transportFlags.Insecure, "insecure", false,
		"Don't check TLS certificates and allow insecure non-TLS downloads over http")
	cmdDiscover.Flags.BoolVar(&outputJson, "json", false,
		"Output result as JSON")
	cmdDiscover.Flags.UintVar(&flagPort, "port", 0,
		"Port to connect to when performing discovery")
}

func runDiscover(args []string) (exit int) {
	if len(args) < 1 {
		stderr("discover: at least one name required")
	}

	for _, name := range args {
		app, err := discovery.NewAppFromString(name)
		if app.Labels["os"] == "" {
			app.Labels["os"] = runtime.GOOS
		}
		if app.Labels["arch"] == "" {
			app.Labels["arch"] = runtime.GOARCH
		}
		if err != nil {
			stderr("%s: %s", name, err)
			return 1
		}
		insecure := discovery.InsecureNone
		if transportFlags.Insecure {
			insecure = discovery.InsecureTLS | discovery.InsecureHTTP
		}
		eps, attempts, err := discovery.DiscoverACIEndpoints(*app, nil, insecure, flagPort)
		if err != nil {
			stderr("error fetching endpoints for %s: %s", name, err)
			return 1
		}
		for _, a := range attempts {
			fmt.Printf("discover endpoints walk: prefix: %s error: %v\n", a.Prefix, a.Error)
		}
		publicKeys, attempts, err := discovery.DiscoverPublicKeys(*app, nil, insecure, flagPort)
		if err != nil {
			stderr("error fetching public keys for %s: %s", name, err)
			return 1
		}
		for _, a := range attempts {
			fmt.Printf("discover public keys walk: prefix: %s error: %v\n", a.Prefix, a.Error)
		}

		type discoveryData struct {
			ACIEndpoints []discovery.ACIEndpoint
			PublicKeys   []string
		}

		if outputJson {
			dd := discoveryData{ACIEndpoints: eps, PublicKeys: publicKeys}
			jsonBytes, err := json.MarshalIndent(dd, "", "    ")
			if err != nil {
				stderr("error generating JSON: %s", err)
				return 1
			}
			fmt.Println(string(jsonBytes))
		} else {
			for _, aciEndpoint := range eps {
				fmt.Printf("ACI: %s, ASC: %s\n", aciEndpoint.ACI, aciEndpoint.ASC)
			}
			if len(publicKeys) > 0 {
				fmt.Println("PublicKeys: " + strings.Join(publicKeys, ","))
			}
		}
	}

	return
}
