package portmapper

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"time"
)

const userlandProxyCommandName = "docker-proxy"

type userlandProxy interface {
	Start() error
	Stop() error
}

// proxyCommand wraps an exec.Cmd to run the userland TCP and UDP
// proxies as separate processes.
type proxyCommand struct {
	cmd *exec.Cmd
}

func (p *proxyCommand) Start() error {
	r, w, err := os.Pipe()
	if err != nil {
		return fmt.Errorf("proxy unable to open os.Pipe %s", err)
	}
	defer r.Close()
	p.cmd.ExtraFiles = []*os.File{w}
	if err := p.cmd.Start(); err != nil {
		return err
	}
	w.Close()

	errchan := make(chan error, 1)
	go func() {
		buf := make([]byte, 2)
		r.Read(buf)

		if string(buf) != "0\n" {
			errStr, err := ioutil.ReadAll(r)
			if err != nil {
				errchan <- fmt.Errorf("Error reading exit status from userland proxy: %v", err)
				return
			}

			errchan <- fmt.Errorf("Error starting userland proxy: %s", errStr)
			return
		}
		errchan <- nil
	}()

	select {
	case err := <-errchan:
		return err
	case <-time.After(16 * time.Second):
		return fmt.Errorf("Timed out proxy starting the userland proxy")
	}
}

func (p *proxyCommand) Stop() error {
	if p.cmd.Process != nil {
		if err := p.cmd.Process.Signal(os.Interrupt); err != nil {
			return err
		}
		return p.cmd.Wait()
	}
	return nil
}

// dummyProxy just listen on some port, it is needed to prevent accidental
// port allocations on bound port, because without userland proxy we using
// iptables rules and not net.Listen
type dummyProxy struct {
	listener io.Closer
	addr     net.Addr
}

func newDummyProxy(proto string, hostIP net.IP, hostPort int) userlandProxy {
	switch proto {
	case "tcp":
		addr := &net.TCPAddr{IP: hostIP, Port: hostPort}
		return &dummyProxy{addr: addr}
	case "udp":
		addr := &net.UDPAddr{IP: hostIP, Port: hostPort}
		return &dummyProxy{addr: addr}
	}
	return nil
}

func (p *dummyProxy) Start() error {
	switch addr := p.addr.(type) {
	case *net.TCPAddr:
		l, err := net.ListenTCP("tcp", addr)
		if err != nil {
			return err
		}
		p.listener = l
	case *net.UDPAddr:
		l, err := net.ListenUDP("udp", addr)
		if err != nil {
			return err
		}
		p.listener = l
	default:
		return fmt.Errorf("Unknown addr type: %T", p.addr)
	}
	return nil
}

func (p *dummyProxy) Stop() error {
	if p.listener != nil {
		return p.listener.Close()
	}
	return nil
}
