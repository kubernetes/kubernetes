package pager

import "k8s.io/cli-runtime/pkg/genericclioptions"

type catOutputPager struct{}

func (p *catOutputPager) Start(streams *genericclioptions.IOStreams) error {
	return nil
}

func (p *catOutputPager) Done() error {
	return nil
}
