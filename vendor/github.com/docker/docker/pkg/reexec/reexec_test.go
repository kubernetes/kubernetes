package reexec

import (
	"os"
	"os/exec"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func init() {
	Register("reexec", func() {
		panic("Return Error")
	})
	Init()
}

func TestRegister(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			require.Equal(t, `reexec func already registered under name "reexec"`, r)
		}
	}()
	Register("reexec", func() {})
}

func TestCommand(t *testing.T) {
	cmd := Command("reexec")
	w, err := cmd.StdinPipe()
	require.NoError(t, err, "Error on pipe creation: %v", err)
	defer w.Close()

	err = cmd.Start()
	require.NoError(t, err, "Error on re-exec cmd: %v", err)
	err = cmd.Wait()
	require.EqualError(t, err, "exit status 2")
}

func TestNaiveSelf(t *testing.T) {
	if os.Getenv("TEST_CHECK") == "1" {
		os.Exit(2)
	}
	cmd := exec.Command(naiveSelf(), "-test.run=TestNaiveSelf")
	cmd.Env = append(os.Environ(), "TEST_CHECK=1")
	err := cmd.Start()
	require.NoError(t, err, "Unable to start command")
	err = cmd.Wait()
	require.EqualError(t, err, "exit status 2")

	os.Args[0] = "mkdir"
	assert.NotEqual(t, naiveSelf(), os.Args[0])
}
