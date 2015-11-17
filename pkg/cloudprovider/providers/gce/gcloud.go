/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bufio"
	"bytes"
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/exec"
)

// GCloud knows how to use the gcloud command line tool to discover things about the GCE environment
type GCloud struct {
	e exec.Interface
}

func (g *GCloud) getProjectAndZone() (string, string, error) {
	data, err := g.e.Command("gcloud", "config", "list").CombinedOutput()
	if err != nil {
		return "", "", err
	}

	// Ugh, this isn't the best parser, but gcfg is too picky and requires all fields to be
	// present which is brittle.
	scanner := bufio.NewScanner(bytes.NewBuffer(data))
	project := ""
	zone := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		// skip sections
		if strings.HasPrefix(line, "[") {
			continue
		}
		// skip things w/o '='
		if strings.Index(line, "=") == -1 {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			return "", "", fmt.Errorf("unexpected format: %s", line)
		}
		property := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		switch property {
		case "project":
			project = value
		case "zone":
			zone = value
		}
	}
	return project, zone, nil
}
