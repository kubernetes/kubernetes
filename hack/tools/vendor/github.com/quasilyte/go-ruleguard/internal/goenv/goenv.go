package goenv

import (
	"errors"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

func Read() (map[string]string, error) {
	out, err := exec.Command("go", "env").CombinedOutput()
	if err != nil {
		return nil, err
	}
	return parseGoEnv(out, runtime.GOOS)
}

func parseGoEnv(data []byte, goos string) (map[string]string, error) {
	vars := make(map[string]string)

	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")

	if goos == "windows" {
		// Line format is: `set $name=$value`
		for _, l := range lines {
			l = strings.TrimPrefix(l, "set ")
			parts := strings.Split(l, "=")
			if len(parts) != 2 {
				continue
			}
			vars[parts[0]] = parts[1]
		}
	} else {
		// Line format is: `$name="$value"`
		for _, l := range lines {
			parts := strings.Split(strings.TrimSpace(l), "=")
			if len(parts) != 2 {
				continue
			}
			val, err := strconv.Unquote(parts[1])
			if err != nil {
				continue
			}
			vars[parts[0]] = val
		}
	}

	if len(vars) == 0 {
		return nil, errors.New("empty env set")
	}

	return vars, nil
}
