package fileutils

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/fsouza/go-dockerclient/vendor/github.com/Sirupsen/logrus"
)

func Exclusion(pattern string) bool {
	return pattern[0] == '!'
}

func Empty(pattern string) bool {
	return pattern == ""
}

// Cleanpatterns takes a slice of patterns returns a new
// slice of patterns cleaned with filepath.Clean, stripped
// of any empty patterns and lets the caller know whether the
// slice contains any exception patterns (prefixed with !).
func CleanPatterns(patterns []string) ([]string, [][]string, bool, error) {
	// Loop over exclusion patterns and:
	// 1. Clean them up.
	// 2. Indicate whether we are dealing with any exception rules.
	// 3. Error if we see a single exclusion marker on it's own (!).
	cleanedPatterns := []string{}
	patternDirs := [][]string{}
	exceptions := false
	for _, pattern := range patterns {
		// Eliminate leading and trailing whitespace.
		pattern = strings.TrimSpace(pattern)
		if Empty(pattern) {
			continue
		}
		if Exclusion(pattern) {
			if len(pattern) == 1 {
				logrus.Errorf("Illegal exclusion pattern: %s", pattern)
				return nil, nil, false, errors.New("Illegal exclusion pattern: !")
			}
			exceptions = true
		}
		pattern = filepath.Clean(pattern)
		cleanedPatterns = append(cleanedPatterns, pattern)
		if Exclusion(pattern) {
			pattern = pattern[1:]
		}
		patternDirs = append(patternDirs, strings.Split(pattern, "/"))
	}

	return cleanedPatterns, patternDirs, exceptions, nil
}

// Matches returns true if file matches any of the patterns
// and isn't excluded by any of the subsequent patterns.
func Matches(file string, patterns []string) (bool, error) {
	file = filepath.Clean(file)

	if file == "." {
		// Don't let them exclude everything, kind of silly.
		return false, nil
	}

	patterns, patDirs, _, err := CleanPatterns(patterns)
	if err != nil {
		return false, err
	}

	return OptimizedMatches(file, patterns, patDirs)
}

// Matches is basically the same as fileutils.Matches() but optimized for archive.go.
// It will assume that the inputs have been preprocessed and therefore the function
// doen't need to do as much error checking and clean-up. This was done to avoid
// repeating these steps on each file being checked during the archive process.
// The more generic fileutils.Matches() can't make these assumptions.
func OptimizedMatches(file string, patterns []string, patDirs [][]string) (bool, error) {
	matched := false
	parentPath := filepath.Dir(file)
	parentPathDirs := strings.Split(parentPath, "/")

	for i, pattern := range patterns {
		negative := false

		if Exclusion(pattern) {
			negative = true
			pattern = pattern[1:]
		}

		match, err := filepath.Match(pattern, file)
		if err != nil {
			logrus.Errorf("Error matching: %s (pattern: %s)", file, pattern)
			return false, err
		}

		if !match && parentPath != "." {
			// Check to see if the pattern matches one of our parent dirs.
			if len(patDirs[i]) <= len(parentPathDirs) {
				match, _ = filepath.Match(strings.Join(patDirs[i], "/"),
					strings.Join(parentPathDirs[:len(patDirs[i])], "/"))
			}
		}

		if match {
			matched = !negative
		}
	}

	if matched {
		logrus.Debugf("Skipping excluded path: %s", file)
	}
	return matched, nil
}

func CopyFile(src, dst string) (int64, error) {
	cleanSrc := filepath.Clean(src)
	cleanDst := filepath.Clean(dst)
	if cleanSrc == cleanDst {
		return 0, nil
	}
	sf, err := os.Open(cleanSrc)
	if err != nil {
		return 0, err
	}
	defer sf.Close()
	if err := os.Remove(cleanDst); err != nil && !os.IsNotExist(err) {
		return 0, err
	}
	df, err := os.Create(cleanDst)
	if err != nil {
		return 0, err
	}
	defer df.Close()
	return io.Copy(df, sf)
}

func GetTotalUsedFds() int {
	if fds, err := ioutil.ReadDir(fmt.Sprintf("/proc/%d/fd", os.Getpid())); err != nil {
		logrus.Errorf("Error opening /proc/%d/fd: %s", os.Getpid(), err)
	} else {
		return len(fds)
	}
	return -1
}

// ReadSymlinkedDirectory returns the target directory of a symlink.
// The target of the symbolic link may not be a file.
func ReadSymlinkedDirectory(path string) (string, error) {
	var realPath string
	var err error
	if realPath, err = filepath.Abs(path); err != nil {
		return "", fmt.Errorf("unable to get absolute path for %s: %s", path, err)
	}
	if realPath, err = filepath.EvalSymlinks(realPath); err != nil {
		return "", fmt.Errorf("failed to canonicalise path for %s: %s", path, err)
	}
	realPathInfo, err := os.Stat(realPath)
	if err != nil {
		return "", fmt.Errorf("failed to stat target '%s' of '%s': %s", realPath, path, err)
	}
	if !realPathInfo.Mode().IsDir() {
		return "", fmt.Errorf("canonical path points to a file '%s'", realPath)
	}
	return realPath, nil
}
