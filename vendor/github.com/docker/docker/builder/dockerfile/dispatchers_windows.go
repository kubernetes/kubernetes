package dockerfile

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/docker/docker/pkg/system"
)

var pattern = regexp.MustCompile(`^[a-zA-Z]:\.$`)

// normaliseWorkdir normalises a user requested working directory in a
// platform semantically consistent way.
func normaliseWorkdir(current string, requested string) (string, error) {
	if requested == "" {
		return "", errors.New("cannot normalise nothing")
	}

	// `filepath.Clean` will replace "" with "." so skip in that case
	if current != "" {
		current = filepath.Clean(current)
	}
	if requested != "" {
		requested = filepath.Clean(requested)
	}

	// If either current or requested in Windows is:
	// C:
	// C:.
	// then an error will be thrown as the definition for the above
	// refers to `current directory on drive C:`
	// Since filepath.Clean() will automatically normalize the above
	// to `C:.`, we only need to check the last format
	if pattern.MatchString(current) {
		return "", fmt.Errorf("%s is not a directory. If you are specifying a drive letter, please add a trailing '\\'", current)
	}
	if pattern.MatchString(requested) {
		return "", fmt.Errorf("%s is not a directory. If you are specifying a drive letter, please add a trailing '\\'", requested)
	}

	// Target semantics is C:\somefolder, specifically in the format:
	// UPPERCASEDriveLetter-Colon-Backslash-FolderName. We are already
	// guaranteed that `current`, if set, is consistent. This allows us to
	// cope correctly with any of the following in a Dockerfile:
	//	WORKDIR a                       --> C:\a
	//	WORKDIR c:\\foo                 --> C:\foo
	//	WORKDIR \\foo                   --> C:\foo
	//	WORKDIR /foo                    --> C:\foo
	//	WORKDIR c:\\foo \ WORKDIR bar   --> C:\foo --> C:\foo\bar
	//	WORKDIR C:/foo \ WORKDIR bar    --> C:\foo --> C:\foo\bar
	//	WORKDIR C:/foo \ WORKDIR \\bar  --> C:\foo --> C:\bar
	//	WORKDIR /foo \ WORKDIR c:/bar   --> C:\foo --> C:\bar
	if len(current) == 0 || system.IsAbs(requested) {
		if (requested[0] == os.PathSeparator) ||
			(len(requested) > 1 && string(requested[1]) != ":") ||
			(len(requested) == 1) {
			requested = filepath.Join(`C:\`, requested)
		}
	} else {
		requested = filepath.Join(current, requested)
	}
	// Upper-case drive letter
	return (strings.ToUpper(string(requested[0])) + requested[1:]), nil
}

func errNotJSON(command, original string) error {
	// For Windows users, give a hint if it looks like it might contain
	// a path which hasn't been escaped such as ["c:\windows\system32\prog.exe", "-param"],
	// as JSON must be escaped. Unfortunate...
	//
	// Specifically looking for quote-driveletter-colon-backslash, there's no
	// double backslash and a [] pair. No, this is not perfect, but it doesn't
	// have to be. It's simply a hint to make life a little easier.
	extra := ""
	original = filepath.FromSlash(strings.ToLower(strings.Replace(strings.ToLower(original), strings.ToLower(command)+" ", "", -1)))
	if len(regexp.MustCompile(`"[a-z]:\\.*`).FindStringSubmatch(original)) > 0 &&
		!strings.Contains(original, `\\`) &&
		strings.Contains(original, "[") &&
		strings.Contains(original, "]") {
		extra = fmt.Sprintf(`. It looks like '%s' includes a file path without an escaped back-slash. JSON requires back-slashes to be escaped such as ["c:\\path\\to\\file.exe", "/parameter"]`, original)
	}
	return fmt.Errorf("%s requires the arguments to be in JSON form%s", command, extra)
}

// equalEnvKeys compare two strings and returns true if they are equal. On
// Windows this comparison is case insensitive.
func equalEnvKeys(from, to string) bool {
	return strings.ToUpper(from) == strings.ToUpper(to)
}
