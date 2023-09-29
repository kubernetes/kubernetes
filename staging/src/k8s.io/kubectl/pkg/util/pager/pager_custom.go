package pager

import (
	"errors"
	"io"
	"os/exec"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/kubectl/pkg/util/term"
)

type customOutputPager struct {
	pagerCmd            *exec.Cmd
	terminalOutput      io.Writer
	programStdoutWriter io.WriteCloser
	programStdoutReader io.Reader
}

func newCustomOutputPager(pagerCmdStr string) *customOutputPager {
	return &customOutputPager{
		pagerCmd: exec.Command(pagerCmdStr),
	}
}

func (p *customOutputPager) Start(streams *genericclioptions.IOStreams) error {
	if err := p.transformIOStreams(streams); err != nil {
		return err
	}

	p.pagerCmd.Stdout = p.terminalOutput
	p.pagerCmd.Stdin = p.programStdoutReader

	if err := p.pagerCmd.Start(); err != nil {
		return err
	}

	return nil
}

func (p *customOutputPager) Done() error {
	p.programStdoutWriter.Close()
	if err := p.pagerCmd.Wait(); err != nil {
		return err
	}
	return nil
}

func (p *customOutputPager) transformIOStreams(streams *genericclioptions.IOStreams) error {
	reader, writer := io.Pipe()
	p.programStdoutReader = reader
	p.programStdoutWriter = writer

	if term.IsTerminal(streams.Out) {
		p.terminalOutput = streams.Out
		streams.Out = writer
	} else {
		return errors.New("streams.Out must be a terminal")
	}

	if term.IsTerminal(streams.ErrOut) {
		streams.ErrOut = writer
	}

	return nil
}
