package allowed

import (
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/allowed/a2"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/forbidden/f2"
	_ "k8s.io/kubernetes/cmd/import-boss/testdata/transitive/neither/n2"
)

var X = "allowed"
