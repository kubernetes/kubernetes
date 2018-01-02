package stream

import (
	"io"
	"sync"

	"golang.org/x/net/context"

	"github.com/docker/docker/pkg/pools"
	"github.com/docker/docker/pkg/promise"
	"github.com/docker/docker/pkg/term"
	"github.com/sirupsen/logrus"
)

var defaultEscapeSequence = []byte{16, 17} // ctrl-p, ctrl-q

// AttachConfig is the config struct used to attach a client to a stream's stdio
type AttachConfig struct {
	// Tells the attach copier that the stream's stdin is a TTY and to look for
	// escape sequences in stdin to detach from the stream.
	// When true the escape sequence is not passed to the underlying stream
	TTY bool
	// Specifies the detach keys the client will be using
	// Only useful when `TTY` is true
	DetachKeys []byte

	// CloseStdin signals that once done, stdin for the attached stream should be closed
	// For example, this would close the attached container's stdin.
	CloseStdin bool

	// UseStd* indicate whether the client has requested to be connected to the
	// given stream or not.  These flags are used instead of checking Std* != nil
	// at points before the client streams Std* are wired up.
	UseStdin, UseStdout, UseStderr bool

	// CStd* are the streams directly connected to the container
	CStdin           io.WriteCloser
	CStdout, CStderr io.ReadCloser

	// Provide client streams to wire up to
	Stdin          io.ReadCloser
	Stdout, Stderr io.Writer
}

// AttachStreams attaches the container's streams to the AttachConfig
func (c *Config) AttachStreams(cfg *AttachConfig) {
	if cfg.UseStdin {
		cfg.CStdin = c.StdinPipe()
	}

	if cfg.UseStdout {
		cfg.CStdout = c.StdoutPipe()
	}

	if cfg.UseStderr {
		cfg.CStderr = c.StderrPipe()
	}
}

// CopyStreams starts goroutines to copy data in and out to/from the container
func (c *Config) CopyStreams(ctx context.Context, cfg *AttachConfig) chan error {
	var (
		wg     sync.WaitGroup
		errors = make(chan error, 3)
	)

	if cfg.Stdin != nil {
		wg.Add(1)
	}

	if cfg.Stdout != nil {
		wg.Add(1)
	}

	if cfg.Stderr != nil {
		wg.Add(1)
	}

	// Connect stdin of container to the attach stdin stream.
	go func() {
		if cfg.Stdin == nil {
			return
		}
		logrus.Debug("attach: stdin: begin")

		var err error
		if cfg.TTY {
			_, err = copyEscapable(cfg.CStdin, cfg.Stdin, cfg.DetachKeys)
		} else {
			_, err = pools.Copy(cfg.CStdin, cfg.Stdin)
		}
		if err == io.ErrClosedPipe {
			err = nil
		}
		if err != nil {
			logrus.Errorf("attach: stdin: %s", err)
			errors <- err
		}
		if cfg.CloseStdin && !cfg.TTY {
			cfg.CStdin.Close()
		} else {
			// No matter what, when stdin is closed (io.Copy unblock), close stdout and stderr
			if cfg.CStdout != nil {
				cfg.CStdout.Close()
			}
			if cfg.CStderr != nil {
				cfg.CStderr.Close()
			}
		}
		logrus.Debug("attach: stdin: end")
		wg.Done()
	}()

	attachStream := func(name string, stream io.Writer, streamPipe io.ReadCloser) {
		if stream == nil {
			return
		}

		logrus.Debugf("attach: %s: begin", name)
		_, err := pools.Copy(stream, streamPipe)
		if err == io.ErrClosedPipe {
			err = nil
		}
		if err != nil {
			logrus.Errorf("attach: %s: %v", name, err)
			errors <- err
		}
		// Make sure stdin gets closed
		if cfg.Stdin != nil {
			cfg.Stdin.Close()
		}
		streamPipe.Close()
		logrus.Debugf("attach: %s: end", name)
		wg.Done()
	}

	go attachStream("stdout", cfg.Stdout, cfg.CStdout)
	go attachStream("stderr", cfg.Stderr, cfg.CStderr)

	return promise.Go(func() error {
		done := make(chan struct{})
		go func() {
			wg.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-ctx.Done():
			// close all pipes
			if cfg.CStdin != nil {
				cfg.CStdin.Close()
			}
			if cfg.CStdout != nil {
				cfg.CStdout.Close()
			}
			if cfg.CStderr != nil {
				cfg.CStderr.Close()
			}
			<-done
		}
		close(errors)
		for err := range errors {
			if err != nil {
				return err
			}
		}
		return nil
	})
}

func copyEscapable(dst io.Writer, src io.ReadCloser, keys []byte) (written int64, err error) {
	if len(keys) == 0 {
		keys = defaultEscapeSequence
	}
	pr := term.NewEscapeProxy(src, keys)
	defer src.Close()

	return pools.Copy(dst, pr)
}
