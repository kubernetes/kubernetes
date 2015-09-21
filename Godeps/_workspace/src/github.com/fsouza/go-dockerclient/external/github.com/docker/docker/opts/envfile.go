package opts

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
)

var (
	// EnvironmentVariableRegexp A regexp to validate correct environment variables
	// Environment variables set by the user must have a name consisting solely of
	// alphabetics, numerics, and underscores - the first of which must not be numeric.
	EnvironmentVariableRegexp = regexp.MustCompile("^[[:alpha:]_][[:alpha:][:digit:]_]*$")
)

// ParseEnvFile Read in a line delimited file with environment variables enumerated
func ParseEnvFile(filename string) ([]string, error) {
	fh, err := os.Open(filename)
	if err != nil {
		return []string{}, err
	}
	defer fh.Close()

	lines := []string{}
	scanner := bufio.NewScanner(fh)
	for scanner.Scan() {
		line := scanner.Text()
		// line is not empty, and not starting with '#'
		if len(line) > 0 && !strings.HasPrefix(line, "#") {
			data := strings.SplitN(line, "=", 2)

			// trim the front of a variable, but nothing else
			variable := strings.TrimLeft(data[0], whiteSpaces)

			if !EnvironmentVariableRegexp.MatchString(variable) {
				return []string{}, ErrBadEnvVariable{fmt.Sprintf("variable '%s' is not a valid environment variable", variable)}
			}
			if len(data) > 1 {

				// pass the value through, no trimming
				lines = append(lines, fmt.Sprintf("%s=%s", variable, data[1]))
			} else {
				// if only a pass-through variable is given, clean it up.
				lines = append(lines, fmt.Sprintf("%s=%s", strings.TrimSpace(line), os.Getenv(line)))
			}
		}
	}
	return lines, scanner.Err()
}

var whiteSpaces = " \t"

// ErrBadEnvVariable typed error for bad environment variable
type ErrBadEnvVariable struct {
	msg string
}

func (e ErrBadEnvVariable) Error() string {
	return fmt.Sprintf("poorly formatted environment: %s", e.msg)
}
