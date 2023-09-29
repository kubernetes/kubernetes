package pager

import (
	"os"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/kubectl/pkg/util/term"
)

type OutputPager interface {
	Start(*genericclioptions.IOStreams) error
	Done() error
}

func GetOutputPager(streams genericclioptions.IOStreams) OutputPager {
	if !term.IsTerminal(streams.Out) {
		return &catOutputPager{}
	}
	pagerExecutable, shouldUsePager := getPagerExecutable()
	if shouldUsePager {
		return newCustomOutputPager(pagerExecutable)
	} else {
		return &catOutputPager{}
	}
}

func getPagerExecutable() (string, bool) {
	for _, envName := range pagerEnvs() {
		if value, ok := os.LookupEnv(envName); ok {
			if value == "" {
				return "", false
			} else {
				return value, true
			}
		}
	}
	return "", false
}

func pagerEnvs() []string {
	return []string{
		"KUBE_PAGER",
		"PAGER",
	}
}
