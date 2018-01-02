// +build darwin linux solaris

package signal

import (
	"os"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestCatchAll(t *testing.T) {
	sigs := make(chan os.Signal, 1)
	CatchAll(sigs)
	defer StopCatch(sigs)

	listOfSignals := map[string]string{
		"CONT": syscall.SIGCONT.String(),
		"HUP":  syscall.SIGHUP.String(),
		"CHLD": syscall.SIGCHLD.String(),
		"ILL":  syscall.SIGILL.String(),
		"FPE":  syscall.SIGFPE.String(),
		"CLD":  syscall.SIGCLD.String(),
	}

	for sigStr := range listOfSignals {
		signal, ok := SignalMap[sigStr]
		if ok {
			go func() {
				time.Sleep(1 * time.Millisecond)
				syscall.Kill(syscall.Getpid(), signal)
			}()

			s := <-sigs
			assert.EqualValues(t, s.String(), signal.String())
		}

	}
}

func TestStopCatch(t *testing.T) {
	signal, _ := SignalMap["HUP"]
	channel := make(chan os.Signal, 1)
	CatchAll(channel)
	go func() {

		time.Sleep(1 * time.Millisecond)
		syscall.Kill(syscall.Getpid(), signal)
	}()
	signalString := <-channel
	assert.EqualValues(t, signalString.String(), signal.String())

	StopCatch(channel)
	_, ok := <-channel
	assert.EqualValues(t, ok, false)
}
