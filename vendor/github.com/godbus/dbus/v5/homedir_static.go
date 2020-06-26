// +build static_build

package dbus

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func lookupHomeDir() string {
	myUid := os.Getuid()

	f, err := os.Open("/etc/passwd")
	if err != nil {
		return "/"
	}
	defer f.Close()

	s := bufio.NewScanner(f)

	for s.Scan() {
		if err := s.Err(); err != nil {
			break
		}

		line := strings.TrimSpace(s.Text())
		if line == "" {
			continue
		}

		parts := strings.Split(line, ":")

		if len(parts) >= 6 {
			uid, err := strconv.Atoi(parts[2])
			if err == nil && uid == myUid {
				return parts[5]
			}
		}
	}

	// Default to / if we can't get a better value
	return "/"
}
