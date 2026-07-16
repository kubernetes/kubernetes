package aaa

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/allowed-by-both"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/allowed-by-root"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/allowed-by-sub"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/bbb"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-both"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-root"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/forbidden-by-sub"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/nested-fwd/neither/n1"
)

var X = "aaa"
