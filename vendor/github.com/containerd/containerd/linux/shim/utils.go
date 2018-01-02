// +build !windows

package shim

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"time"

	"github.com/containerd/containerd/errdefs"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	runc "github.com/containerd/go-runc"
	"github.com/pkg/errors"
	"golang.org/x/sys/unix"
)

// TODO(mlaventure): move to runc package?
func getLastRuntimeError(r *runc.Runc) (string, error) {
	if r.Log == "" {
		return "", nil
	}

	f, err := os.OpenFile(r.Log, os.O_RDONLY, 0400)
	if err != nil {
		return "", err
	}

	var (
		errMsg string
		log    struct {
			Level string
			Msg   string
			Time  time.Time
		}
	)

	dec := json.NewDecoder(f)
	for err = nil; err == nil; {
		if err = dec.Decode(&log); err != nil && err != io.EOF {
			return "", err
		}
		if log.Level == "error" {
			errMsg = strings.TrimSpace(log.Msg)
		}
	}

	return errMsg, nil
}

// criuError returns only the first line of the error message from criu
// it tries to add an invalid dump log location when returning the message
func criuError(err error) string {
	parts := strings.Split(err.Error(), "\n")
	return parts[0]
}

func copyFile(to, from string) error {
	ff, err := os.Open(from)
	if err != nil {
		return err
	}
	defer ff.Close()
	tt, err := os.Create(to)
	if err != nil {
		return err
	}
	defer tt.Close()
	_, err = io.Copy(tt, ff)
	return err
}

func checkKillError(err error) error {
	if err == nil {
		return nil
	}
	if strings.Contains(err.Error(), "os: process already finished") || err == unix.ESRCH {
		return errors.Wrapf(errdefs.ErrNotFound, "process already finished")
	}
	return errors.Wrapf(err, "unknown error after kill")
}

func hasNoIO(r *shimapi.CreateTaskRequest) bool {
	return r.Stdin == "" && r.Stdout == "" && r.Stderr == ""
}
