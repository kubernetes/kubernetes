package shim

import (
	"io"
	"sync"
	"syscall"

	"github.com/containerd/console"
	"github.com/containerd/fifo"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

type linuxPlatform struct {
	epoller *console.Epoller
}

func (p *linuxPlatform) copyConsole(ctx context.Context, console console.Console, stdin, stdout, stderr string, wg, cwg *sync.WaitGroup) (console.Console, error) {
	if p.epoller == nil {
		return nil, errors.New("uninitialized epoller")
	}

	epollConsole, err := p.epoller.Add(console)
	if err != nil {
		return nil, err
	}

	if stdin != "" {
		in, err := fifo.OpenFifo(ctx, stdin, syscall.O_RDONLY, 0)
		if err != nil {
			return nil, err
		}
		cwg.Add(1)
		go func() {
			cwg.Done()
			io.Copy(epollConsole, in)
		}()
	}

	outw, err := fifo.OpenFifo(ctx, stdout, syscall.O_WRONLY, 0)
	if err != nil {
		return nil, err
	}
	outr, err := fifo.OpenFifo(ctx, stdout, syscall.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	wg.Add(1)
	cwg.Add(1)
	go func() {
		cwg.Done()
		io.Copy(outw, epollConsole)
		epollConsole.Close()
		outr.Close()
		outw.Close()
		wg.Done()
	}()
	return epollConsole, nil
}

func (p *linuxPlatform) shutdownConsole(ctx context.Context, cons console.Console) error {
	if p.epoller == nil {
		return errors.New("uninitialized epoller")
	}
	epollConsole, ok := cons.(*console.EpollConsole)
	if !ok {
		return errors.Errorf("expected EpollConsole, got %#v", cons)
	}
	return epollConsole.Shutdown(p.epoller.CloseConsole)
}

func (p *linuxPlatform) close() error {
	return p.epoller.Close()
}

// initialize a single epoll fd to manage our consoles. `initPlatform` should
// only be called once.
func (s *Service) initPlatform() error {
	if s.platform != nil {
		return nil
	}
	epoller, err := console.NewEpoller()
	if err != nil {
		return errors.Wrap(err, "failed to initialize epoller")
	}
	s.platform = &linuxPlatform{
		epoller: epoller,
	}
	go epoller.Wait()
	return nil
}
