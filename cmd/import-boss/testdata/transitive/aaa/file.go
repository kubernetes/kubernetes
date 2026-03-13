package aaa

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed/a1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden/f1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/neither"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/neither/n1"
)

var X = "aaa"
