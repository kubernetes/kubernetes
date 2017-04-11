package htpasswd

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"golang.org/x/crypto/bcrypt"
)

// htpasswd holds a path to a system .htpasswd file and the machinery to parse
// it. Only bcrypt hash entries are supported.
type htpasswd struct {
	entries map[string][]byte // maps username to password byte slice.
}

// newHTPasswd parses the reader and returns an htpasswd or an error.
func newHTPasswd(rd io.Reader) (*htpasswd, error) {
	entries, err := parseHTPasswd(rd)
	if err != nil {
		return nil, err
	}

	return &htpasswd{entries: entries}, nil
}

// AuthenticateUser checks a given user:password credential against the
// receiving HTPasswd's file. If the check passes, nil is returned.
func (htpasswd *htpasswd) authenticateUser(username string, password string) error {
	credentials, ok := htpasswd.entries[username]
	if !ok {
		// timing attack paranoia
		bcrypt.CompareHashAndPassword([]byte{}, []byte(password))

		return ErrAuthenticationFailure
	}

	err := bcrypt.CompareHashAndPassword([]byte(credentials), []byte(password))
	if err != nil {
		return ErrAuthenticationFailure
	}

	return nil
}

// parseHTPasswd parses the contents of htpasswd. This will read all the
// entries in the file, whether or not they are needed. An error is returned
// if an syntax errors are encountered or if the reader fails.
func parseHTPasswd(rd io.Reader) (map[string][]byte, error) {
	entries := map[string][]byte{}
	scanner := bufio.NewScanner(rd)
	var line int
	for scanner.Scan() {
		line++ // 1-based line numbering
		t := strings.TrimSpace(scanner.Text())

		if len(t) < 1 {
			continue
		}

		// lines that *begin* with a '#' are considered comments
		if t[0] == '#' {
			continue
		}

		i := strings.Index(t, ":")
		if i < 0 || i >= len(t) {
			return nil, fmt.Errorf("htpasswd: invalid entry at line %d: %q", line, scanner.Text())
		}

		entries[t[:i]] = []byte(t[i+1:])
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return entries, nil
}
