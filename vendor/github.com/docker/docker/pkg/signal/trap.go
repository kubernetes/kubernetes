package signal

import (
	"os"
	gosignal "os/signal"
	"runtime"
	"sync/atomic"
	"syscall"

	"github.com/Sirupsen/logrus"
)

// Trap sets up a simplified signal "trap", appropriate for common
// behavior expected from a vanilla unix command-line tool in general
// (and the Docker engine in particular).
//
// * If SIGINT or SIGTERM are received, `cleanup` is called, then the process is terminated.
// * If SIGINT or SIGTERM are received 3 times before cleanup is complete, then cleanup is
//   skipped and the process is terminated immediately (allows force quit of stuck daemon)
// * A SIGQUIT always causes an exit without cleanup, with a goroutine dump preceding exit.
//
func Trap(cleanup func()) {
	c := make(chan os.Signal, 1)
	// we will handle INT, TERM, QUIT here
	signals := []os.Signal{os.Interrupt, syscall.SIGTERM, syscall.SIGQUIT}
	gosignal.Notify(c, signals...)
	go func() {
		interruptCount := uint32(0)
		for sig := range c {
			go func(sig os.Signal) {
				logrus.Infof("Processing signal '%v'", sig)
				switch sig {
				case os.Interrupt, syscall.SIGTERM:
					if atomic.LoadUint32(&interruptCount) < 3 {
						// Initiate the cleanup only once
						if atomic.AddUint32(&interruptCount, 1) == 1 {
							// Call the provided cleanup handler
							cleanup()
							os.Exit(0)
						} else {
							return
						}
					} else {
						// 3 SIGTERM/INT signals received; force exit without cleanup
						logrus.Infof("Forcing docker daemon shutdown without cleanup; 3 interrupts received")
					}
				case syscall.SIGQUIT:
					DumpStacks()
					logrus.Infof("Forcing docker daemon shutdown without cleanup on SIGQUIT")
				}
				//for the SIGINT/TERM, and SIGQUIT non-clean shutdown case, exit with 128 + signal #
				os.Exit(128 + int(sig.(syscall.Signal)))
			}(sig)
		}
	}()
}

func DumpStacks() {
	buf := make([]byte, 16384)
	buf = buf[:runtime.Stack(buf, true)]
	// Note that if the daemon is started with a less-verbose log-level than "info" (the default), the goroutine
	// traces won't show up in the log.
	logrus.Infof("=== BEGIN goroutine stack dump ===\n%s\n=== END goroutine stack dump ===", buf)
}
