package aaa

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed/a1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden/f1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/neither"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/neither/n1"
)

var X = "aaa"
