// +build !windows

package shim

import (
	"context"
	"fmt"
	"io"
	"sync"
	"syscall"

	"github.com/containerd/console"
	"github.com/containerd/fifo"
	runc "github.com/containerd/go-runc"
)

func copyConsole(ctx context.Context, console console.Console, stdin, stdout, stderr string, wg, cwg *sync.WaitGroup) error {
	if stdin != "" {
		in, err := fifo.OpenFifo(ctx, stdin, syscall.O_RDONLY, 0)
		if err != nil {
			return err
		}
		cwg.Add(1)
		go func() {
			cwg.Done()
			io.Copy(console, in)
		}()
	}
	outw, err := fifo.OpenFifo(ctx, stdout, syscall.O_WRONLY, 0)
	if err != nil {
		return err
	}
	outr, err := fifo.OpenFifo(ctx, stdout, syscall.O_RDONLY, 0)
	if err != nil {
		return err
	}
	wg.Add(1)
	cwg.Add(1)
	go func() {
		cwg.Done()
		io.Copy(outw, console)
		console.Close()
		outr.Close()
		outw.Close()
		wg.Done()
	}()
	return nil
}

func copyPipes(ctx context.Context, rio runc.IO, stdin, stdout, stderr string, wg, cwg *sync.WaitGroup) error {
	for name, dest := range map[string]func(wc io.WriteCloser, rc io.Closer){
		stdout: func(wc io.WriteCloser, rc io.Closer) {
			wg.Add(1)
			cwg.Add(1)
			go func() {
				cwg.Done()
				io.Copy(wc, rio.Stdout())
				wg.Done()
				wc.Close()
				rc.Close()
			}()
		},
		stderr: func(wc io.WriteCloser, rc io.Closer) {
			wg.Add(1)
			cwg.Add(1)
			go func() {
				cwg.Done()
				io.Copy(wc, rio.Stderr())
				wg.Done()
				wc.Close()
				rc.Close()
			}()
		},
	} {
		fw, err := fifo.OpenFifo(ctx, name, syscall.O_WRONLY, 0)
		if err != nil {
			return fmt.Errorf("containerd-shim: opening %s failed: %s", name, err)
		}
		fr, err := fifo.OpenFifo(ctx, name, syscall.O_RDONLY, 0)
		if err != nil {
			return fmt.Errorf("containerd-shim: opening %s failed: %s", name, err)
		}
		dest(fw, fr)
	}
	if stdin == "" {
		rio.Stdin().Close()
		return nil
	}
	f, err := fifo.OpenFifo(ctx, stdin, syscall.O_RDONLY, 0)
	if err != nil {
		return fmt.Errorf("containerd-shim: opening %s failed: %s", stdin, err)
	}
	cwg.Add(1)
	go func() {
		cwg.Done()
		io.Copy(rio.Stdin(), f)
		rio.Stdin().Close()
		f.Close()
	}()
	return nil
}
