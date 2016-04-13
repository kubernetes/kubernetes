// +build linux

package lxc

import (
	"bufio"
	"errors"
	"strconv"
	"strings"
)

var (
	ErrCannotParse = errors.New("cannot parse raw input")
)

type lxcInfo struct {
	Running bool
	Pid     int
}

func parseLxcInfo(raw string) (*lxcInfo, error) {
	if raw == "" {
		return nil, ErrCannotParse
	}
	var (
		err  error
		s    = bufio.NewScanner(strings.NewReader(raw))
		info = &lxcInfo{}
	)
	for s.Scan() {
		text := s.Text()

		if s.Err() != nil {
			return nil, s.Err()
		}

		parts := strings.Split(text, ":")
		if len(parts) < 2 {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(parts[0])) {
		case "state":
			info.Running = strings.TrimSpace(parts[1]) == "RUNNING"
		case "pid":
			info.Pid, err = strconv.Atoi(strings.TrimSpace(parts[1]))
			if err != nil {
				return nil, err
			}
		}
	}
	return info, nil
}
