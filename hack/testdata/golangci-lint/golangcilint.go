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

// Package golangcilint contains intentional linter issues for
// testing of golangci-lint and its configuration. Use with:
//
//	go test test/integration/golangci-lint
//
// Expected hints (only reported when using `verify-golangci-lint.sh -n`
// and golangci-hints.yaml) and issues (always reported) are documented
// here via comments, see examples below.
//
// Some patterns are explicitly excluded in the golangci-lint configuration
// and must *not* be reported. Those are included without a marker comment:
// if they were ever reported, the "unexpected issue" check in the test
// would fail.
//
// sorted untested: restricted to a fixed list of feature gate files, none of which live here.
// kubeapilinter untested: restricted to staging/src/k8s.io/api/.*, which this package is not part of.
// revive untested: its "exported" rule is excluded everywhere except cmd/kubeadm, so it can never fire here.
package golangcilint

import (
	"context"
	"crypto/md5"

	"errors"
	"fmt"
	"io"
	"k8s.io/client-go/rest"
	"k8s.io/controller-manager/pkg/clientbuilder"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/klog/v2"
	"k8s.io/ktesting" // depguard issue: ktesting is allowed only in test files
)

func Printf() {
	fmt.Sprintf("%d", "foo") // govet issue: format %d has arg "foo" of wrong type string
}

func CopyLocks(mu sync.Mutex) { // govet issue: passes lock by value: sync.Mutex
	mu.Lock()
}

func Unreachable() int {
	return 1
	return 2 // govet issue: unreachable code
}

func Unlambda() func(string) string {
	return func(s string) string { // gocritic hint: (?s)unlambda: replace .* with `strings.ToUpper`
		return strings.ToUpper(s)
	}
}

func ValSwap(a, b int) (int, int) {
	tmp := a // gocritic hint: can re-write as `a, b = b, a`
	a = b
	b = tmp
	return a, b
}

func WrapperFunc(s string) string {
	return strings.Replace(s, "a", "b", -1) // gocritic hint: use strings.ReplaceAll method
}

func SingleCaseSwitch(x int) {
	switch x { // gocritic hint: should rewrite switch statement to if statement
	case 1:
		fmt.Println("one")
	}
}

func staticcheckTarget(context.Context) {}

func Staticcheck() {
	staticcheckTarget(nil) // staticcheck issue: do not pass a nil Context
}

func StaticcheckHint() error {
	return errors.New("Something failed.") // staticcheck hint: error strings should not end with punctuation
}

func unusedHelper() int { // unused issue: func unusedHelper is unused
	return 1
}

func Ineffassign() int {
	x := 1 // ineffassign issue: ineffectual assignment to x
	x = 2
	return x
}

func Forbidigo() [16]byte {
	return md5.Sum(nil) // forbidigo issue: md5 is oudated, insecure
}

func ForbidigoClientBuilderConfig(builder clientbuilder.ControllerClientBuilder) *rest.Config {
	return builder.ConfigOrDie("controller") // forbidigo issue: ControllerClientBuilder.Config and ConfigOrDie are deprecated
}

func Modernize(a, b int) int {
	var m int
	if a > b { // modernize hint: if/else statement can be modernized using max
		m = a
	} else {
		m = b
	}
	return m
}

func Logcheck() {
	klog.InfoS("message", "key") // logcheck issue: Additional arguments to InfoS should always be Key Value pairs
}

func Errcheck() {
	os.Remove("/no-such-file") // errcheck hint: Error return value of `os.Remove` is not checked
}

func Errorlint(err error) bool {
	return err == io.EOF // errorlint hint: comparing with == will fail on wrapped errors
}

func Usestdlibvars() (*http.Request, error) {
	return http.NewRequest("GET", "http://example.com", nil) // usestdlibvars hint: "GET" can be replaced by http.MethodGet
}

func Test(tCtx ktesting.TContext) {
	tCtx.Logf("hello %s") // govet issue: format %s reads arg #1, but call has 0 args
	tCtx.Expect(1)        // ginkgolinter issue: missing assertion method
}

func Testify(t *testing.T, items []string) {
	assert.True(t, len(items) == 0) // testifylint issue: use assert.Equal
}

// gocritic's ifElseChain check is disabled.
func IfElseChain(a, b, c int) string {
	if a > 0 {
		return "positive a"
	} else if b < 0 {
		return "negative b"
	} else if c == 0 {
		return "zero c"
	}
	return "other"
}

// Convert_v1_Pod_To_v2_Pod is an allowed naming despite the underscores.
func Convert_v1_Pod_To_v2_Pod() {}

// Only cmd/kubeadm requires full doc comments.
func ExportedNoDoc() {}
