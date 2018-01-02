package signal

import (
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseSignal(t *testing.T) {
	_, checkAtoiError := ParseSignal("0")
	assert.EqualError(t, checkAtoiError, "Invalid signal: 0")

	_, error := ParseSignal("SIG")
	assert.EqualError(t, error, "Invalid signal: SIG")

	for sigStr := range SignalMap {
		responseSignal, error := ParseSignal(sigStr)
		assert.NoError(t, error)
		signal := SignalMap[sigStr]
		assert.EqualValues(t, signal, responseSignal)
	}
}

func TestValidSignalForPlatform(t *testing.T) {
	isValidSignal := ValidSignalForPlatform(syscall.Signal(0))
	assert.EqualValues(t, false, isValidSignal)

	for _, sigN := range SignalMap {
		isValidSignal = ValidSignalForPlatform(syscall.Signal(sigN))
		assert.EqualValues(t, true, isValidSignal)
	}
}
