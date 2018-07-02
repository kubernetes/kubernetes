// +build tools

// This source file is never built (because of the "tools" build requirement above).
// Instead, it exists to indicate to "go mod -sync" and "go mod -vendor" that
// the commands below are expected to be tracked as part of the k8s.io/kubernetes
// module's dependencies and included in the vendor directory.

package tools

import (
	_ "github.com/bazelbuild/bazel-gazelle/cmd/gazelle"
	_ "github.com/client9/misspell/cmd/misspell"
	_ "github.com/jteeuwen/go-bindata/go-bindata"
	_ "github.com/kubernetes/repo-infra/kazel"
	_ "github.com/onsi/ginkgo/ginkgo"
	_ "github.com/tools/godep"
	_ "k8s.io/code-generator/cmd/client-gen"
	_ "k8s.io/code-generator/cmd/conversion-gen"
	_ "k8s.io/code-generator/cmd/deepcopy-gen"
	_ "k8s.io/code-generator/cmd/defaulter-gen"
	_ "k8s.io/code-generator/cmd/go-to-protobuf"
	_ "k8s.io/code-generator/cmd/go-to-protobuf/protoc-gen-gogo"
	_ "k8s.io/code-generator/cmd/import-boss"
	_ "k8s.io/code-generator/cmd/informer-gen"
	_ "k8s.io/code-generator/cmd/lister-gen"
	_ "k8s.io/code-generator/cmd/openapi-gen"
	_ "k8s.io/code-generator/cmd/set-gen"
	_ "k8s.io/gengo/args"
)
