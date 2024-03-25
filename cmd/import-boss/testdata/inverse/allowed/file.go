package allowed

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/allowed/a2"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/forbidden/f2"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/inverse/neither/n2"
)

var X = "allowed"
