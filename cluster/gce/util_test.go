/*
Copyright The Kubernetes Authors.

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
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

func TestImageSelection(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("GCE cluster scripts require a Unix environment")
	}
	if _, err := exec.LookPath("bash"); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name            string
		command         string
		availableImages bool
		masterOverride  string
		nodeOverride    string
		wantError       bool
		wantCalls       []string
		wantOutput      string
	}{
		{
			name:    "empty selections do not prevent sourcing",
			command: `test -z "${MASTER_IMAGE}" && test -z "${NODE_IMAGE}"`,
		},
		{
			name:      "master creation rejects an empty image",
			command:   "NODE_IMAGE=fixture-node; create-master-instance-internal fixture-master",
			wantError: true,
		},
		{
			name:      "linux template rejects an empty image before deletion",
			command:   `MASTER_IMAGE=fixture-master; create-node-template fixture-template "--no-scopes" "" "" linux n2-standard-2`,
			wantError: true,
		},
		{
			name:      "heapster creation rejects an empty image",
			command:   "create-heapster-node",
			wantError: true,
		},
		{
			name:      "conditional kube-up rejects an empty master image",
			command:   `NODE_IMAGE=fixture-node; if kube-up; then exit 0; else exit 1; fi`,
			wantError: true,
		},
		{
			name:      "kube-up rejects an empty node image",
			command:   `MASTER_IMAGE=fixture-master; kube-up`,
			wantError: true,
		},
		{
			name:      "replication rejects an empty master image",
			command:   `KUBE_REPLICATE_EXISTING_MASTER=true; if kube-up; then exit 0; else exit 1; fi`,
			wantError: true,
		},
		{
			name:            "master creation uses the automatic lookup result",
			command:         "create-master-instance-internal fixture-instance",
			availableImages: true,
			wantCalls:       []string{"compute instances create", "--image fixture-master"},
		},
		{
			name:            "linux template uses the automatic lookup result",
			command:         `create-node-template fixture-template "--no-scopes" "" "" linux n2-standard-2`,
			availableImages: true,
			wantCalls:       []string{"compute instance-templates describe", "compute instance-templates delete", "compute instance-templates create", "--image fixture-node"},
		},
		{
			name:            "master override skips only the master lookup",
			command:         `test "${MASTER_IMAGE}" = explicit-master && test "${NODE_IMAGE}" = fixture-node`,
			availableImages: true,
			masterOverride:  "explicit-master",
		},
		{
			name:            "node override skips only the node lookup",
			command:         `test "${MASTER_IMAGE}" = fixture-master && test "${NODE_IMAGE}" = explicit-node`,
			availableImages: true,
			nodeOverride:    "explicit-node",
		},
		{
			name:            "both overrides skip both lookups",
			command:         `test "${MASTER_IMAGE}" = explicit-master && test "${NODE_IMAGE}" = explicit-node`,
			availableImages: true,
			masterOverride:  "explicit-master",
			nodeOverride:    "explicit-node",
		},
		{
			name: "upgrade rejects an empty image before deleting the master",
			command: `
# Load the function without running the upgrade CLI.
eval "$(sed -n '/^function upgrade-master() {$/,/^}$/p' ./upgrade.sh)"
get-master-replicas-count() { echo 1; }
get-kubeconfig-basicauth() { :; }
get-kubeconfig-bearertoken() { :; }
detect-master() { :; }
parse-master-env() { :; }
upgrade-master-env() { :; }
create-master-instance() { create-master-instance-internal fixture-master; }
wait-for-master() { :; }
MASTER_NAME=fixture-master
SERVER_BINARY_TAR_URL=fixture-tar
upgrade-master`,
			wantError: true,
		},
		{
			name: "replication permits an empty node image",
			command: `
KUBE_REPLICATE_EXISTING_MASTER=true
kube::util::ensure-temp-dir() { echo "Reached preparation"; exit 0; }
kube-up`,
			masterOverride: "fixture-master",
			wantOutput:     "Reached preparation",
		},
		{
			name: "existing master permits both images to be empty",
			command: `
KUBE_USE_EXISTING_MASTER=true
kube::util::ensure-temp-dir() { echo "Reached preparation"; exit 0; }
kube-up`,
			wantOutput: "Reached preparation",
		},
		{
			name:      "windows template does not require a linux image",
			command:   `create-node-template fixture-template "--no-scopes" "" "" windows n2-standard-2`,
			wantCalls: []string{"compute instance-templates create", "--image windows-server-"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			trace := filepath.Join(t.TempDir(), "calls")
			t.Setenv("IMAGE_TEST_TRACE", trace)
			t.Setenv("IMAGE_TEST_AVAILABLE", strconv.FormatBool(test.availableImages))
			t.Setenv("KUBERNETES_PROVIDER", "gce")
			t.Setenv("KUBE_CONFIG_FILE", "config-default.sh")
			t.Setenv("KUBE_MASTER_OS_DISTRIBUTION", "gci")
			t.Setenv("KUBE_NODE_OS_DISTRIBUTION", "gci")
			t.Setenv("WINDOWS_NODE_OS_DISTRIBUTION", "win2019")
			t.Setenv("KUBE_ENABLE_CLUSTER_AUTOSCALER", "false")
			t.Setenv("KUBE_GCE_MASTER_IMAGE", test.masterOverride)
			t.Setenv("KUBE_GCE_NODE_IMAGE", test.nodeOverride)
			t.Setenv("KUBE_GCE_MASTER_PROJECT", "fixture-master-project")
			t.Setenv("KUBE_GCE_NODE_PROJECT", "fixture-node-project")
			t.Setenv("KUBE_GCE_MASTER_IMAGE_FAMILY", "fixture-master-family")
			t.Setenv("KUBE_GCE_NODE_IMAGE_FAMILY", "fixture-node-family")
			t.Setenv("KUBE_USE_EXISTING_MASTER", "false")
			t.Setenv("KUBE_REPLICATE_EXISTING_MASTER", "false")
			t.Setenv("KUBE_TEMP", t.TempDir())
			output, err := exec.CommandContext(t.Context(), "bash", "-c", `
set -o errexit
set -o nounset
set -o pipefail
gcloud() {
  if [[ "$1 $2 $3" == "compute images list" ]]; then
    local role
    case "${4:-}" in
      --project=fixture-master-project) role=master ;;
      --project=fixture-node-project) role=node ;;
      *) echo "Unexpected image lookup project: $*" >&2; return 1 ;;
    esac
    local expected=(compute images list "--project=fixture-${role}-project" --no-standard-images
      "--filter=family:fixture-${role}-family architecture:X86_64 status=(READY)"
      '--sort-by=~creationTimestamp' --limit=1 --format 'value(name)')
    if [[ "$#" -ne "${#expected[@]}" ]]; then
      echo "Unexpected image lookup arguments: $*" >&2
      return 1
    fi
    local arg
    for arg in "${expected[@]}"; do
      if [[ "$1" != "${arg}" ]]; then
        printf 'Image lookup argument: got %q, want %q\n' "$1" "${arg}" >&2
        return 1
      fi
      shift
    done
    printf 'IMAGE_LOOKUP:%s\n' "${role}" >&2
    if [[ "${IMAGE_TEST_AVAILABLE}" == true ]]; then
      printf 'fixture-%s\n' "${role}"
    fi
    return
  fi
  printf '%s\n' "$*" >> "${IMAGE_TEST_TRACE}"
}
source ./util.sh
PROJECT=fixture-project
NETWORK_PROJECT=fixture-project
detect-project() { :; }
detect-subnetworks() { :; }
make-gcloud-network-argument() { echo --network=fixture-network; }
get-node-instance-metadata-from-file() { echo fixture-metadata; }
kube::util::ensure-temp-dir() { echo "Unexpected kube-up preparation" >&2; return 1; }
eval "$1"
`, "image-test", test.command).CombinedOutput()
			if (err != nil) != test.wantError {
				t.Fatalf("command error = %v, wantError = %v\n%s", err, test.wantError, output)
			}
			if test.wantError && !strings.Contains(string(output), "No READY X86_64") {
				t.Fatalf("expected an empty image error, got:\n%s", output)
			}
			if test.wantOutput != "" && !strings.Contains(string(output), test.wantOutput) {
				t.Errorf("output does not contain %q:\n%s", test.wantOutput, output)
			}
			for _, lookup := range []struct{ role, override string }{
				{"master", test.masterOverride},
				{"node", test.nodeOverride},
			} {
				want := 1
				if lookup.override != "" {
					want = 0
				}
				if got := strings.Count(string(output), "IMAGE_LOOKUP:"+lookup.role+"\n"); got != want {
					t.Errorf("%s lookup count = %d, want %d\n%s", lookup.role, got, want, output)
				}
			}
			calls, err := os.ReadFile(trace)
			if err != nil && !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
			if len(test.wantCalls) == 0 && len(calls) > 0 {
				t.Fatalf("unexpected cloud calls:\n%s", calls)
			}
			for _, want := range test.wantCalls {
				if !strings.Contains(string(calls), want) {
					t.Errorf("cloud calls do not contain %q:\n%s", want, calls)
				}
			}
		})
	}
}
