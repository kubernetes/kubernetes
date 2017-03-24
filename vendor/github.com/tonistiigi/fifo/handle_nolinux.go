// +build !linux

package fifo

import (
	"syscall"

	"github.com/pkg/errors"
)

type handle struct {
	fn  string
	dev uint64
	ino uint64
}

func getHandle(fn string) (*handle, error) {
	var stat syscall.Stat_t
	if err := syscall.Stat(fn, &stat); err != nil {
		return nil, errors.Wrapf(err, "failed to stat %v", fn)
	}

	h := &handle{
		fn:  fn,
		dev: uint64(stat.Dev),
		ino: stat.Ino,
	}

	return h, nil
}

func (h *handle) Path() (string, error) {
	var stat syscall.Stat_t
	if err := syscall.Stat(h.fn, &stat); err != nil {
		return "", errors.Wrapf(err, "path %v could not be statted", h.fn)
	}
	if uint64(stat.Dev) != h.dev || stat.Ino != h.ino {
		return "", errors.Errorf("failed to verify handle %v/%v %v/%v for %v", stat.Dev, h.dev, stat.Ino, h.ino, h.fn)
	}
	return h.fn, nil
}

func (h *handle) Name() string {
	return h.fn
}

func (h *handle) Close() error {
	return nil
}
