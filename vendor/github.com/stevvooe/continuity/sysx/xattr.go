package sysx

import (
	"bytes"
	"syscall"
)

const defaultXattrBufferSize = 5

type listxattrFunc func(path string, dest []byte) (int, error)

func listxattrAll(path string, listFunc listxattrFunc) ([]string, error) {
	var p []byte // nil on first execution

	for {
		n, err := listFunc(path, p) // first call gets buffer size.
		if err != nil {
			return nil, err
		}

		if n > len(p) {
			p = make([]byte, n)
			continue
		}

		p = p[:n]

		ps := bytes.Split(bytes.TrimSuffix(p, []byte{0}), []byte{0})
		var entries []string
		for _, p := range ps {
			s := string(p)
			if s != "" {
				entries = append(entries, s)
			}
		}

		return entries, nil
	}
}

type getxattrFunc func(string, string, []byte) (int, error)

func getxattrAll(path, attr string, getFunc getxattrFunc) ([]byte, error) {
	p := make([]byte, defaultXattrBufferSize)
	for {
		n, err := getFunc(path, attr, p)
		if err != nil {
			if errno, ok := err.(syscall.Errno); ok && errno == syscall.ERANGE {
				p = make([]byte, len(p)*2) // this can't be ideal.
				continue                   // try again!
			}

			return nil, err
		}

		// realloc to correct size and repeat
		if n > len(p) {
			p = make([]byte, n)
			continue
		}

		return p[:n], nil
	}
}
