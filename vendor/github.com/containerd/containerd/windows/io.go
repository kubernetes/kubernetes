// +build windows

package windows

import (
	"context"
	"net"
	"sync"
	"time"

	"github.com/Microsoft/go-winio"
	"github.com/containerd/containerd/runtime"
	"github.com/pkg/errors"
)

type pipeSet struct {
	src    runtime.IO
	stdin  net.Conn
	stdout net.Conn
	stderr net.Conn
}

// NewIO connects to the provided pipe addresses
func newPipeSet(ctx context.Context, io runtime.IO) (*pipeSet, error) {
	var (
		err    error
		c      net.Conn
		wg     sync.WaitGroup
		set    = &pipeSet{src: io}
		ch     = make(chan error)
		opened = 0
	)

	defer func() {
		if err != nil {
			go func() {
				for i := 0; i < opened; i++ {
					// Drain the channel to avoid leaking the goroutines
					<-ch
				}
				close(ch)
				wg.Wait()
				set.Close()
			}()
		}
	}()

	for _, p := range [3]struct {
		name string
		open bool
		conn *net.Conn
	}{
		{
			name: io.Stdin,
			open: io.Stdin != "",
			conn: &set.stdin,
		},
		{
			name: io.Stdout,
			open: io.Stdout != "",
			conn: &set.stdout,
		},
		{
			name: io.Stderr,
			open: !io.Terminal && io.Stderr != "",
			conn: &set.stderr,
		},
	} {
		if p.open {
			wg.Add(1)
			opened++
			go func(name string, conn *net.Conn) {
				dialTimeout := 3 * time.Second
				c, err = winio.DialPipe(name, &dialTimeout)
				if err != nil {
					ch <- errors.Wrapf(err, "failed to connect to %s", name)
				}
				*conn = c
				ch <- nil
				wg.Done()
			}(p.name, p.conn)
		}
	}

	for i := 0; i < opened; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case e := <-ch:
			if e != nil {
				if err == nil {
					err = e
				} else {
					err = errors.Wrapf(err, e.Error())
				}
			}
		}
	}

	return set, err
}

// Close terminates all successfully dialed IO connections
func (p *pipeSet) Close() {
	for _, cn := range []net.Conn{p.stdin, p.stdout, p.stderr} {
		if cn != nil {
			cn.Close()
		}
	}
}
