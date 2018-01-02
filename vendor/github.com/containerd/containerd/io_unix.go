// +build !windows

package containerd

import (
	"context"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/containerd/fifo"
)

// NewFifos returns a new set of fifos for the task
func NewFifos(id string) (*FIFOSet, error) {
	root := "/run/containerd/fifo"
	if err := os.MkdirAll(root, 0700); err != nil {
		return nil, err
	}
	dir, err := ioutil.TempDir(root, "")
	if err != nil {
		return nil, err
	}
	return &FIFOSet{
		Dir: dir,
		In:  filepath.Join(dir, id+"-stdin"),
		Out: filepath.Join(dir, id+"-stdout"),
		Err: filepath.Join(dir, id+"-stderr"),
	}, nil
}

func copyIO(fifos *FIFOSet, ioset *ioSet, tty bool) (_ *wgCloser, err error) {
	var (
		f           io.ReadWriteCloser
		set         []io.Closer
		ctx, cancel = context.WithCancel(context.Background())
		wg          = &sync.WaitGroup{}
	)
	defer func() {
		if err != nil {
			for _, f := range set {
				f.Close()
			}
			cancel()
		}
	}()

	if f, err = fifo.OpenFifo(ctx, fifos.In, syscall.O_WRONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		return nil, err
	}
	set = append(set, f)
	go func(w io.WriteCloser) {
		io.Copy(w, ioset.in)
		w.Close()
	}(f)

	if f, err = fifo.OpenFifo(ctx, fifos.Out, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		return nil, err
	}
	set = append(set, f)
	wg.Add(1)
	go func(r io.ReadCloser) {
		io.Copy(ioset.out, r)
		r.Close()
		wg.Done()
	}(f)

	if f, err = fifo.OpenFifo(ctx, fifos.Err, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		return nil, err
	}
	set = append(set, f)

	if !tty {
		wg.Add(1)
		go func(r io.ReadCloser) {
			io.Copy(ioset.err, r)
			r.Close()
			wg.Done()
		}(f)
	}
	return &wgCloser{
		wg:     wg,
		dir:    fifos.Dir,
		set:    set,
		cancel: cancel,
	}, nil
}

// NewDirectIO returns an IO implementation that exposes the pipes directly
func NewDirectIO(ctx context.Context, terminal bool) (*DirectIO, error) {
	set, err := NewFifos("")
	if err != nil {
		return nil, err
	}
	f := &DirectIO{
		set:      set,
		terminal: terminal,
	}
	defer func() {
		if err != nil {
			f.Delete()
		}
	}()
	if f.Stdin, err = fifo.OpenFifo(ctx, set.In, syscall.O_WRONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		return nil, err
	}
	if f.Stdout, err = fifo.OpenFifo(ctx, set.Out, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		f.Stdin.Close()
		return nil, err
	}
	if f.Stderr, err = fifo.OpenFifo(ctx, set.Err, syscall.O_RDONLY|syscall.O_CREAT|syscall.O_NONBLOCK, 0700); err != nil {
		f.Stdin.Close()
		f.Stdout.Close()
		return nil, err
	}
	return f, nil
}

// DirectIO allows task IO to be handled externally by the caller
type DirectIO struct {
	Stdin  io.WriteCloser
	Stdout io.ReadCloser
	Stderr io.ReadCloser

	set      *FIFOSet
	terminal bool
}

// IOCreate returns IO avaliable for use with task creation
func (f *DirectIO) IOCreate(id string) (IO, error) {
	return f, nil
}

// IOAttach returns IO avaliable for use with task attachment
func (f *DirectIO) IOAttach(set *FIFOSet) (IO, error) {
	return f, nil
}

// Config returns the IOConfig
func (f *DirectIO) Config() IOConfig {
	return IOConfig{
		Terminal: f.terminal,
		Stdin:    f.set.In,
		Stdout:   f.set.Out,
		Stderr:   f.set.Err,
	}
}

// Cancel stops any IO copy operations
//
// Not applicable for DirectIO
func (f *DirectIO) Cancel() {
	// nothing to cancel as all operations are handled externally
}

// Wait on any IO copy operations
//
// Not applicable for DirectIO
func (f *DirectIO) Wait() {
	// nothing to wait on as all operations are handled externally
}

// Close closes all open fds
func (f *DirectIO) Close() error {
	err := f.Stdin.Close()
	if err2 := f.Stdout.Close(); err == nil {
		err = err2
	}
	if err2 := f.Stderr.Close(); err == nil {
		err = err2
	}
	return err
}

// Delete removes the underlying directory containing fifos
func (f *DirectIO) Delete() error {
	if f.set.Dir == "" {
		return nil
	}
	return os.RemoveAll(f.set.Dir)
}
