package fileutils

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"text/scanner"

	"github.com/sirupsen/logrus"
)

// PatternMatcher allows checking paths agaist a list of patterns
type PatternMatcher struct {
	patterns   []*Pattern
	exclusions bool
}

// NewPatternMatcher creates a new matcher object for specific patterns that can
// be used later to match against patterns against paths
func NewPatternMatcher(patterns []string) (*PatternMatcher, error) {
	pm := &PatternMatcher{
		patterns: make([]*Pattern, 0, len(patterns)),
	}
	for _, p := range patterns {
		// Eliminate leading and trailing whitespace.
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		p = filepath.Clean(p)
		newp := &Pattern{}
		if p[0] == '!' {
			if len(p) == 1 {
				return nil, errors.New("illegal exclusion pattern: \"!\"")
			}
			newp.exclusion = true
			p = p[1:]
			pm.exclusions = true
		}
		// Do some syntax checking on the pattern.
		// filepath's Match() has some really weird rules that are inconsistent
		// so instead of trying to dup their logic, just call Match() for its
		// error state and if there is an error in the pattern return it.
		// If this becomes an issue we can remove this since its really only
		// needed in the error (syntax) case - which isn't really critical.
		if _, err := filepath.Match(p, "."); err != nil {
			return nil, err
		}
		newp.cleanedPattern = p
		newp.dirs = strings.Split(p, string(os.PathSeparator))
		pm.patterns = append(pm.patterns, newp)
	}
	return pm, nil
}

// Matches matches path against all the patterns. Matches is not safe to be
// called concurrently
func (pm *PatternMatcher) Matches(file string) (bool, error) {
	matched := false
	file = filepath.FromSlash(file)
	parentPath := filepath.Dir(file)
	parentPathDirs := strings.Split(parentPath, string(os.PathSeparator))

	for _, pattern := range pm.patterns {
		negative := false

		if pattern.exclusion {
			negative = true
		}

		match, err := pattern.match(file)
		if err != nil {
			return false, err
		}

		if !match && parentPath != "." {
			// Check to see if the pattern matches one of our parent dirs.
			if len(pattern.dirs) <= len(parentPathDirs) {
				match, _ = pattern.match(strings.Join(parentPathDirs[:len(pattern.dirs)], string(os.PathSeparator)))
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

// Exclusions returns true if any of the patterns define exclusions
func (pm *PatternMatcher) Exclusions() bool {
	return pm.exclusions
}

// Patterns returns array of active patterns
func (pm *PatternMatcher) Patterns() []*Pattern {
	return pm.patterns
}

// Pattern defines a single regexp used used to filter file paths.
type Pattern struct {
	cleanedPattern string
	dirs           []string
	regexp         *regexp.Regexp
	exclusion      bool
}

func (p *Pattern) String() string {
	return p.cleanedPattern
}

// Exclusion returns true if this pattern defines exclusion
func (p *Pattern) Exclusion() bool {
	return p.exclusion
}

func (p *Pattern) match(path string) (bool, error) {

	if p.regexp == nil {
		if err := p.compile(); err != nil {
			return false, filepath.ErrBadPattern
		}
	}

	b := p.regexp.MatchString(path)

	return b, nil
}

func (p *Pattern) compile() error {
	regStr := "^"
	pattern := p.cleanedPattern
	// Go through the pattern and convert it to a regexp.
	// We use a scanner so we can support utf-8 chars.
	var scan scanner.Scanner
	scan.Init(strings.NewReader(pattern))

	sl := string(os.PathSeparator)
	escSL := sl
	if sl == `\` {
		escSL += `\`
	}

	for scan.Peek() != scanner.EOF {
		ch := scan.Next()

		if ch == '*' {
			if scan.Peek() == '*' {
				// is some flavor of "**"
				scan.Next()

				// Treat **/ as ** so eat the "/"
				if string(scan.Peek()) == sl {
					scan.Next()
				}

				if scan.Peek() == scanner.EOF {
					// is "**EOF" - to align with .gitignore just accept all
					regStr += ".*"
				} else {
					// is "**"
					// Note that this allows for any # of /'s (even 0) because
					// the .* will eat everything, even /'s
					regStr += "(.*" + escSL + ")?"
				}
			} else {
				// is "*" so map it to anything but "/"
				regStr += "[^" + escSL + "]*"
			}
		} else if ch == '?' {
			// "?" is any char except "/"
			regStr += "[^" + escSL + "]"
		} else if ch == '.' || ch == '$' {
			// Escape some regexp special chars that have no meaning
			// in golang's filepath.Match
			regStr += `\` + string(ch)
		} else if ch == '\\' {
			// escape next char. Note that a trailing \ in the pattern
			// will be left alone (but need to escape it)
			if sl == `\` {
				// On windows map "\" to "\\", meaning an escaped backslash,
				// and then just continue because filepath.Match on
				// Windows doesn't allow escaping at all
				regStr += escSL
				continue
			}
			if scan.Peek() != scanner.EOF {
				regStr += `\` + string(scan.Next())
			} else {
				regStr += `\`
			}
		} else {
			regStr += string(ch)
		}
	}

	regStr += "$"

	re, err := regexp.Compile(regStr)
	if err != nil {
		return err
	}

	p.regexp = re
	return nil
}

// Matches returns true if file matches any of the patterns
// and isn't excluded by any of the subsequent patterns.
func Matches(file string, patterns []string) (bool, error) {
	pm, err := NewPatternMatcher(patterns)
	if err != nil {
		return false, err
	}
	file = filepath.Clean(file)

	if file == "." {
		// Don't let them exclude everything, kind of silly.
		return false, nil
	}

	return pm.Matches(file)
}

// CopyFile copies from src to dst until either EOF is reached
// on src or an error occurs. It verifies src exists and removes
// the dst if it exists.
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

// CreateIfNotExists creates a file or a directory only if it does not already exist.
func CreateIfNotExists(path string, isDir bool) error {
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			if isDir {
				return os.MkdirAll(path, 0755)
			}
			if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
				return err
			}
			f, err := os.OpenFile(path, os.O_CREATE, 0755)
			if err != nil {
				return err
			}
			f.Close()
		}
	}
	return nil
}
