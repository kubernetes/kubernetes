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

package tools

import (
	"os/exec"
	"strings"
	"testing"
)

func TestImagePullRetries(t *testing.T) {
	script := `
KUBE_IMAGE_PULL_RETRIES=3
KUBE_IMAGE_PULL_RETRY_INITIAL_DELAY=1
source ./common.sh

pull_attempts=0
inspect_attempts=0
failures_before_success=0
image_present=false
sleep_calls=()

fake_docker() {
  case "$1" in
    image)
      if [[ "$2" != inspect ]]; then
        return 2
      fi
      inspect_attempts=$((inspect_attempts + 1))
      [[ "${image_present}" == true ]]
      ;;
    pull)
      pull_attempts=$((pull_attempts + 1))
      if (( pull_attempts <= failures_before_success )); then
        return 1
      fi
      image_present=true
      ;;
    *)
      return 2
      ;;
  esac
}

sleep() {
  sleep_calls+=("$1")
}

assert_equal() {
  if [[ "$1" != "$2" ]]; then
    echo "expected '$1', got '$2'" >&2
    return 1
  fi
}

reset_fake() {
  pull_attempts=0
  inspect_attempts=0
  failures_before_success=0
  image_present=false
  sleep_calls=()
}

DOCKER=(fake_docker)

failures_before_success=2
kube::build::pull_image_with_retry example.com/image:test
assert_equal 3 "${pull_attempts}"
assert_equal "1 2" "${sleep_calls[*]}"

reset_fake
failures_before_success=3
if kube::build::pull_image_with_retry example.com/image:test; then
  echo "expected image pull to fail" >&2
  exit 1
fi
assert_equal 3 "${pull_attempts}"
assert_equal "1 2" "${sleep_calls[*]}"

reset_fake
image_present=true
kube::build::ensure_image example.com/image:test
assert_equal 1 "${inspect_attempts}"
assert_equal 0 "${pull_attempts}"

reset_fake
kube::build::ensure_image example.com/image:test
assert_equal 1 "${inspect_attempts}"
assert_equal 1 "${pull_attempts}"
`

	cmd := exec.Command("bash")
	cmd.Dir = "."
	cmd.Stdin = strings.NewReader(script)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("image pull tests failed: %v\n%s", err, output)
	}
}
