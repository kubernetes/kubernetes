package flagutil

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// SetFlagsFromEnvFile iterates the given flagset and if any flags are not
// already set it attempts to set their values from the given env file. Env
// files may have KEY=VALUE lines where the environment variable names are
// in UPPERCASE, prefixed by the given PREFIX, and dashes are replaced by
// underscores. For example, if prefix=PREFIX, some-flag is named
// PREFIX_SOME_FLAG.
// Comment lines are skipped, but more complex env file parsing is not
// performed.
func SetFlagsFromEnvFile(fs *flag.FlagSet, prefix string, path string) (err error) {
	alreadySet := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		alreadySet[f.Name] = true
	})
	envs, err := parseEnvFile(path)
	if err != nil {
		return err
	}
	fs.VisitAll(func(f *flag.Flag) {
		if !alreadySet[f.Name] {
			key := prefix + "_" + strings.ToUpper(strings.Replace(f.Name, "-", "_", -1))
			val := envs[key]
			if val != "" {
				if serr := fs.Set(f.Name, val); serr != nil {
					err = fmt.Errorf("invalid value %q for %s: %v", val, key, serr)
				}
			}
		}
	})
	return err
}

func parseEnvFile(path string) (map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	envs := make(map[string]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		token := scanner.Text()
		if !skipLine(token) {
			key, val, err := parseLine(token)
			if err == nil {
				envs[key] = val
			}
		}
	}
	return envs, nil
}

func skipLine(line string) bool {
	return len(line) == 0 || strings.HasPrefix(line, "#")
}

func parseLine(line string) (key string, val string, err error) {
	trimmed := strings.TrimSpace(line)
	pair := strings.SplitN(trimmed, "=", 2)
	if len(pair) != 2 {
		err = fmt.Errorf("invalid KEY=value line: %q", line)
		return
	}
	key = strings.TrimSpace(pair[0])
	val = strings.TrimSpace(pair[1])
	return
}
