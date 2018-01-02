package stream

import (
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"sync"

	"github.com/docker/docker/libcontainerd"
	"github.com/docker/docker/pkg/broadcaster"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/pools"
	"github.com/sirupsen/logrus"
)

// Config holds information about I/O streams managed together.
//
// config.StdinPipe returns a WriteCloser which can be used to feed data
// to the standard input of the streamConfig's active process.
// config.StdoutPipe and streamConfig.StderrPipe each return a ReadCloser
// which can be used to retrieve the standard output (and error) generated
// by the container's active process. The output (and error) are actually
// copied and delivered to all StdoutPipe and StderrPipe consumers, using
// a kind of "broadcaster".
type Config struct {
	sync.WaitGroup
	stdout    *broadcaster.Unbuffered
	stderr    *broadcaster.Unbuffered
	stdin     io.ReadCloser
	stdinPipe io.WriteCloser
}

// NewConfig creates a stream config and initializes
// the standard err and standard out to new unbuffered broadcasters.
func NewConfig() *Config {
	return &Config{
		stderr: new(broadcaster.Unbuffered),
		stdout: new(broadcaster.Unbuffered),
	}
}

// Stdout returns the standard output in the configuration.
func (c *Config) Stdout() *broadcaster.Unbuffered {
	return c.stdout
}

// Stderr returns the standard error in the configuration.
func (c *Config) Stderr() *broadcaster.Unbuffered {
	return c.stderr
}

// Stdin returns the standard input in the configuration.
func (c *Config) Stdin() io.ReadCloser {
	return c.stdin
}

// StdinPipe returns an input writer pipe as an io.WriteCloser.
func (c *Config) StdinPipe() io.WriteCloser {
	return c.stdinPipe
}

// StdoutPipe creates a new io.ReadCloser with an empty bytes pipe.
// It adds this new out pipe to the Stdout broadcaster.
// This will block stdout if unconsumed.
func (c *Config) StdoutPipe() io.ReadCloser {
	bytesPipe := ioutils.NewBytesPipe()
	c.stdout.Add(bytesPipe)
	return bytesPipe
}

// StderrPipe creates a new io.ReadCloser with an empty bytes pipe.
// It adds this new err pipe to the Stderr broadcaster.
// This will block stderr if unconsumed.
func (c *Config) StderrPipe() io.ReadCloser {
	bytesPipe := ioutils.NewBytesPipe()
	c.stderr.Add(bytesPipe)
	return bytesPipe
}

// NewInputPipes creates new pipes for both standard inputs, Stdin and StdinPipe.
func (c *Config) NewInputPipes() {
	c.stdin, c.stdinPipe = io.Pipe()
}

// NewNopInputPipe creates a new input pipe that will silently drop all messages in the input.
func (c *Config) NewNopInputPipe() {
	c.stdinPipe = ioutils.NopWriteCloser(ioutil.Discard)
}

// CloseStreams ensures that the configured streams are properly closed.
func (c *Config) CloseStreams() error {
	var errors []string

	if c.stdin != nil {
		if err := c.stdin.Close(); err != nil {
			errors = append(errors, fmt.Sprintf("error close stdin: %s", err))
		}
	}

	if err := c.stdout.Clean(); err != nil {
		errors = append(errors, fmt.Sprintf("error close stdout: %s", err))
	}

	if err := c.stderr.Clean(); err != nil {
		errors = append(errors, fmt.Sprintf("error close stderr: %s", err))
	}

	if len(errors) > 0 {
		return fmt.Errorf(strings.Join(errors, "\n"))
	}

	return nil
}

// CopyToPipe connects streamconfig with a libcontainerd.IOPipe
func (c *Config) CopyToPipe(iop libcontainerd.IOPipe) {
	copyFunc := func(w io.Writer, r io.ReadCloser) {
		c.Add(1)
		go func() {
			if _, err := pools.Copy(w, r); err != nil {
				logrus.Errorf("stream copy error: %+v", err)
			}
			r.Close()
			c.Done()
		}()
	}

	if iop.Stdout != nil {
		copyFunc(c.Stdout(), iop.Stdout)
	}
	if iop.Stderr != nil {
		copyFunc(c.Stderr(), iop.Stderr)
	}

	if stdin := c.Stdin(); stdin != nil {
		if iop.Stdin != nil {
			go func() {
				pools.Copy(iop.Stdin, stdin)
				if err := iop.Stdin.Close(); err != nil {
					logrus.Warnf("failed to close stdin: %+v", err)
				}
			}()
		}
	}
}
