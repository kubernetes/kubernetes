package commands

import (
	gocontext "context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/containerd/containerd"
	"github.com/sirupsen/logrus"
)

type killer interface {
	Kill(gocontext.Context, syscall.Signal, ...containerd.KillOpts) error
}

// ForwardAllSignals forwards signals
func ForwardAllSignals(ctx gocontext.Context, task killer) chan os.Signal {
	sigc := make(chan os.Signal, 128)
	signal.Notify(sigc)
	go func() {
		for s := range sigc {
			logrus.Debug("forwarding signal ", s)
			if err := task.Kill(ctx, s.(syscall.Signal)); err != nil {
				logrus.WithError(err).Errorf("forward signal %s", s)
			}
		}
	}()
	return sigc
}

// StopCatch stops and closes a channel
func StopCatch(sigc chan os.Signal) {
	signal.Stop(sigc)
	close(sigc)
}

// ParseSignal parses a given string into a syscall.Signal
// it checks that the signal exists in the platform-appropriate signalMap
func ParseSignal(rawSignal string) (syscall.Signal, error) {
	s, err := strconv.Atoi(rawSignal)
	if err == nil {
		sig := syscall.Signal(s)
		for _, msig := range signalMap {
			if sig == msig {
				return sig, nil
			}
		}
		return -1, fmt.Errorf("unknown signal %q", rawSignal)
	}
	signal, ok := signalMap[strings.TrimPrefix(strings.ToUpper(rawSignal), "SIG")]
	if !ok {
		return -1, fmt.Errorf("unknown signal %q", rawSignal)
	}
	return signal, nil
}
