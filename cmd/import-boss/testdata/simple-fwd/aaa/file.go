package aaa

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/allowed"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/allowed/a1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/forbidden"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/forbidden/f1"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/neither"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/simple-fwd/neither/n1"
)

var X = "aaa"
