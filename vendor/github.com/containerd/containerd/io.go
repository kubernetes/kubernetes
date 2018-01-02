package containerd

import (
	"context"
	"fmt"
	"io"
	"os"
	"sync"
)

// IOConfig holds the io configurations.
type IOConfig struct {
	// Terminal is true if one has been allocated
	Terminal bool
	// Stdin path
	Stdin string
	// Stdout path
	Stdout string
	// Stderr path
	Stderr string
}

// IO holds the io information for a task or process
type IO interface {
	// Config returns the IO configuration.
	Config() IOConfig
	// Cancel aborts all current io operations
	Cancel()
	// Wait blocks until all io copy operations have completed
	Wait()
	// Close cleans up all open io resources
	Close() error
}

// cio is a basic container IO implementation.
type cio struct {
	config IOConfig

	closer *wgCloser
}

func (c *cio) Config() IOConfig {
	return c.config
}

func (c *cio) Cancel() {
	if c.closer == nil {
		return
	}
	c.closer.Cancel()
}

func (c *cio) Wait() {
	if c.closer == nil {
		return
	}
	c.closer.Wait()
}

func (c *cio) Close() error {
	if c.closer == nil {
		return nil
	}
	return c.closer.Close()
}

// IOCreation creates new IO sets for a task
type IOCreation func(id string) (IO, error)

// IOAttach allows callers to reattach to running tasks
//
// There should only be one reader for a task's IO set
// because fifo's can only be read from one reader or the output
// will be sent only to the first reads
type IOAttach func(*FIFOSet) (IO, error)

// NewIO returns an IOCreation that will provide IO sets without a terminal
func NewIO(stdin io.Reader, stdout, stderr io.Writer) IOCreation {
	return NewIOWithTerminal(stdin, stdout, stderr, false)
}

// NewIOWithTerminal creates a new io set with the provied io.Reader/Writers for use with a terminal
func NewIOWithTerminal(stdin io.Reader, stdout, stderr io.Writer, terminal bool) IOCreation {
	return func(id string) (_ IO, err error) {
		paths, err := NewFifos(id)
		if err != nil {
			return nil, err
		}
		defer func() {
			if err != nil && paths.Dir != "" {
				os.RemoveAll(paths.Dir)
			}
		}()
		cfg := IOConfig{
			Terminal: terminal,
			Stdout:   paths.Out,
			Stderr:   paths.Err,
			Stdin:    paths.In,
		}
		i := &cio{config: cfg}
		set := &ioSet{
			in:  stdin,
			out: stdout,
			err: stderr,
		}
		closer, err := copyIO(paths, set, cfg.Terminal)
		if err != nil {
			return nil, err
		}
		i.closer = closer
		return i, nil
	}
}

// WithAttach attaches the existing io for a task to the provided io.Reader/Writers
func WithAttach(stdin io.Reader, stdout, stderr io.Writer) IOAttach {
	return func(paths *FIFOSet) (IO, error) {
		if paths == nil {
			return nil, fmt.Errorf("cannot attach to existing fifos")
		}
		cfg := IOConfig{
			Terminal: paths.Terminal,
			Stdout:   paths.Out,
			Stderr:   paths.Err,
			Stdin:    paths.In,
		}
		i := &cio{config: cfg}
		set := &ioSet{
			in:  stdin,
			out: stdout,
			err: stderr,
		}
		closer, err := copyIO(paths, set, cfg.Terminal)
		if err != nil {
			return nil, err
		}
		i.closer = closer
		return i, nil
	}
}

// Stdio returns an IO set to be used for a task
// that outputs the container's IO as the current processes Stdio
func Stdio(id string) (IO, error) {
	return NewIO(os.Stdin, os.Stdout, os.Stderr)(id)
}

// StdioTerminal will setup the IO for the task to use a terminal
func StdioTerminal(id string) (IO, error) {
	return NewIOWithTerminal(os.Stdin, os.Stdout, os.Stderr, true)(id)
}

// NullIO redirects the container's IO into /dev/null
func NullIO(id string) (IO, error) {
	return &cio{}, nil
}

// FIFOSet is a set of fifos for use with tasks
type FIFOSet struct {
	// Dir is the directory holding the task fifos
	Dir string
	// In, Out, and Err fifo paths
	In, Out, Err string
	// Terminal returns true if a terminal is being used for the task
	Terminal bool
}

type ioSet struct {
	in       io.Reader
	out, err io.Writer
}

type wgCloser struct {
	wg     *sync.WaitGroup
	dir    string
	set    []io.Closer
	cancel context.CancelFunc
}

func (g *wgCloser) Wait() {
	g.wg.Wait()
}

func (g *wgCloser) Close() error {
	for _, f := range g.set {
		f.Close()
	}
	if g.dir != "" {
		return os.RemoveAll(g.dir)
	}
	return nil
}

func (g *wgCloser) Cancel() {
	g.cancel()
}
