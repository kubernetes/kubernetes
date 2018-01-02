//+build linux

package term

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

// RequiresRoot skips tests that require root, unless the test.root flag has
// been set
func RequiresRoot(t *testing.T) {
	if os.Getuid() != 0 {
		t.Skip("skipping test that requires root")
		return
	}
}

func newTtyForTest(t *testing.T) (*os.File, error) {
	RequiresRoot(t)
	return os.OpenFile("/dev/tty", os.O_RDWR, os.ModeDevice)
}

func newTempFile() (*os.File, error) {
	return ioutil.TempFile(os.TempDir(), "temp")
}

func TestGetWinsize(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	winSize, err := GetWinsize(tty.Fd())
	require.NoError(t, err)
	require.NotNil(t, winSize)
	require.NotNil(t, winSize.Height)
	require.NotNil(t, winSize.Width)
	newSize := Winsize{Width: 200, Height: 200, x: winSize.x, y: winSize.y}
	err = SetWinsize(tty.Fd(), &newSize)
	require.NoError(t, err)
	winSize, err = GetWinsize(tty.Fd())
	require.NoError(t, err)
	require.Equal(t, *winSize, newSize)
}

func TestSetWinsize(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	winSize, err := GetWinsize(tty.Fd())
	require.NoError(t, err)
	require.NotNil(t, winSize)
	newSize := Winsize{Width: 200, Height: 200, x: winSize.x, y: winSize.y}
	err = SetWinsize(tty.Fd(), &newSize)
	require.NoError(t, err)
	winSize, err = GetWinsize(tty.Fd())
	require.NoError(t, err)
	require.Equal(t, *winSize, newSize)
}

func TestGetFdInfo(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	inFd, isTerminal := GetFdInfo(tty)
	require.Equal(t, inFd, tty.Fd())
	require.Equal(t, isTerminal, true)
	tmpFile, err := newTempFile()
	defer tmpFile.Close()
	inFd, isTerminal = GetFdInfo(tmpFile)
	require.Equal(t, inFd, tmpFile.Fd())
	require.Equal(t, isTerminal, false)
}

func TestIsTerminal(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	isTerminal := IsTerminal(tty.Fd())
	require.Equal(t, isTerminal, true)
	tmpFile, err := newTempFile()
	defer tmpFile.Close()
	isTerminal = IsTerminal(tmpFile.Fd())
	require.Equal(t, isTerminal, false)
}

func TestSaveState(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	state, err := SaveState(tty.Fd())
	require.NoError(t, err)
	require.NotNil(t, state)
	tty, err = newTtyForTest(t)
	defer tty.Close()
	err = RestoreTerminal(tty.Fd(), state)
	require.NoError(t, err)
}

func TestDisableEcho(t *testing.T) {
	tty, err := newTtyForTest(t)
	defer tty.Close()
	require.NoError(t, err)
	state, err := SetRawTerminal(tty.Fd())
	defer RestoreTerminal(tty.Fd(), state)
	require.NoError(t, err)
	require.NotNil(t, state)
	err = DisableEcho(tty.Fd(), state)
	require.NoError(t, err)
}
