package revgrep

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// Checker provides APIs to filter static analysis tools to specific commits,
// such as showing only issues since last commit.
type Checker struct {
	// Patch file (unified) to read to detect lines being changed, if nil revgrep
	// will attempt to detect the VCS and generate an appropriate patch. Auto
	// detection will search for uncommitted changes first, if none found, will
	// generate a patch from last committed change. File paths within patches
	// must be relative to current working directory.
	Patch io.Reader
	// NewFiles is a list of file names (with absolute paths) where the entire
	// contents of the file is new.
	NewFiles []string
	// Debug sets the debug writer for additional output.
	Debug io.Writer
	// RevisionFrom check revision starting at, leave blank for auto detection
	// ignored if patch is set.
	RevisionFrom string
	// WholeFiles indicates that the user wishes to see all issues that comes up
	// anywhere in any file that has been changed in this revision or patch.
	WholeFiles bool
	// RevisionTo checks revision finishing at, leave blank for auto detection
	// ignored if patch is set.
	RevisionTo string
	// Regexp to match path, line number, optional column number, and message.
	Regexp string
	// AbsPath is used to make an absolute path of an issue's filename to be
	// relative in order to match patch file. If not set, current working
	// directory is used.
	AbsPath string

	// Calculated changes for next calls to IsNewIssue
	changes map[string][]pos
}

// Issue contains metadata about an issue found.
type Issue struct {
	// File is the name of the file as it appeared from the patch.
	File string
	// LineNo is the line number of the file.
	LineNo int
	// ColNo is the column number or 0 if none could be parsed.
	ColNo int
	// HunkPos is position from file's first @@, for new files this will be the
	// line number.
	//
	// See also: https://developer.github.com/v3/pulls/comments/#create-a-comment
	HunkPos int
	// Issue text as it appeared from the tool.
	Issue string
	// Message is the issue without file name, line number and column number.
	Message string
}

func (c *Checker) preparePatch() error {
	// Check if patch is supplied, if not, retrieve from VCS
	if c.Patch == nil {
		var err error
		c.Patch, c.NewFiles, err = GitPatch(c.RevisionFrom, c.RevisionTo)
		if err != nil {
			return fmt.Errorf("could not read git repo: %s", err)
		}
		if c.Patch == nil {
			return errors.New("no version control repository found")
		}
	}

	return nil
}

// InputIssue represents issue found by some linter
type InputIssue interface {
	FilePath() string
	Line() int
}

type simpleInputIssue struct {
	filePath   string
	lineNumber int
}

func (i simpleInputIssue) FilePath() string {
	return i.filePath
}

func (i simpleInputIssue) Line() int {
	return i.lineNumber
}

// Prepare extracts a patch and changed lines
func (c *Checker) Prepare() error {
	returnErr := c.preparePatch()
	c.changes = c.linesChanged()
	return returnErr
}

// IsNewIssue checks whether issue found by linter is new: it was found in changed lines
func (c Checker) IsNewIssue(i InputIssue) (hunkPos int, isNew bool) {
	fchanges, ok := c.changes[i.FilePath()]
	if !ok { // file wasn't changed
		return 0, false
	}

	if c.WholeFiles {
		return i.Line(), true
	}

	var (
		fpos    pos
		changed bool
	)
	// found file, see if lines matched
	for _, pos := range fchanges {
		if pos.lineNo == i.Line() {
			fpos = pos
			changed = true
			break
		}
	}

	if changed || fchanges == nil {
		// either file changed or it's a new file
		hunkPos := fpos.lineNo
		if changed { // existing file changed
			hunkPos = fpos.hunkPos
		}

		return hunkPos, true
	}

	return 0, false
}

// Check scans reader and writes any lines to writer that have been added in
// Checker.Patch.
//
// Returns issues written to writer when no error occurs.
//
// If no VCS could be found or other VCS errors occur, all issues are written
// to writer and an error is returned.
//
// File paths in reader must be relative to current working directory or
// absolute.
func (c Checker) Check(reader io.Reader, writer io.Writer) (issues []Issue, err error) {
	returnErr := c.Prepare()
	writeAll := returnErr != nil

	// file.go:lineNo:colNo:message
	// colNo is optional, strip spaces before message
	lineRE := regexp.MustCompile(`(.*?\.go):([0-9]+):([0-9]+)?:?\s*(.*)`)
	if c.Regexp != "" {
		lineRE, err = regexp.Compile(c.Regexp)
		if err != nil {
			return nil, fmt.Errorf("could not parse regexp: %v", err)
		}
	}

	// TODO consider lazy loading this, if there's nothing in stdin, no point
	// checking for recent changes
	c.debugf("lines changed: %+v", c.changes)

	absPath := c.AbsPath
	if absPath == "" {
		absPath, err = os.Getwd()
		if err != nil {
			returnErr = fmt.Errorf("could not get current working directory: %s", err)
		}
	}

	// Scan each line in reader and only write those lines if lines changed
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		line := lineRE.FindSubmatch(scanner.Bytes())
		if line == nil {
			c.debugf("cannot parse file+line number: %s", scanner.Text())
			continue
		}

		if writeAll {
			fmt.Fprintln(writer, scanner.Text())
			continue
		}

		// Make absolute path names relative
		path := string(line[1])
		if rel, err := filepath.Rel(absPath, path); err == nil {
			c.debugf("rewrote path from %q to %q (absPath: %q)", path, rel, absPath)
			path = rel
		}

		// Parse line number
		lno, err := strconv.ParseUint(string(line[2]), 10, 64)
		if err != nil {
			c.debugf("cannot parse line number: %q", scanner.Text())
			continue
		}

		// Parse optional column number
		var cno uint64
		if len(line[3]) > 0 {
			cno, err = strconv.ParseUint(string(line[3]), 10, 64)
			if err != nil {
				c.debugf("cannot parse column number: %q", scanner.Text())
				// Ignore this error and continue
			}
		}

		// Extract message
		msg := string(line[4])

		c.debugf("path: %q, lineNo: %v, colNo: %v, msg: %q", path, lno, cno, msg)
		i := simpleInputIssue{
			filePath:   path,
			lineNumber: int(lno),
		}
		hunkPos, changed := c.IsNewIssue(i)
		if changed {
			issue := Issue{
				File:    path,
				LineNo:  int(lno),
				ColNo:   int(cno),
				HunkPos: hunkPos,
				Issue:   scanner.Text(),
				Message: msg,
			}
			issues = append(issues, issue)
			fmt.Fprintln(writer, scanner.Text())
		} else {
			c.debugf("unchanged: %s", scanner.Text())
		}
	}
	if err := scanner.Err(); err != nil {
		returnErr = fmt.Errorf("error reading standard input: %s", err)
	}
	return issues, returnErr
}

func (c Checker) debugf(format string, s ...interface{}) {
	if c.Debug != nil {
		fmt.Fprint(c.Debug, "DEBUG: ")
		fmt.Fprintf(c.Debug, format+"\n", s...)
	}
}

type pos struct {
	lineNo  int // line number
	hunkPos int // position relative to first @@ in file
}

// linesChanges returns a map of file names to line numbers being changed.
// If key is nil, the file has been recently added, else it contains a slice
// of positions that have been added.
func (c Checker) linesChanged() map[string][]pos {
	type state struct {
		file    string
		lineNo  int   // current line number within chunk
		hunkPos int   // current line count since first @@ in file
		changes []pos // position of changes
	}

	var (
		s       state
		changes = make(map[string][]pos)
	)

	for _, file := range c.NewFiles {
		changes[file] = nil
	}

	if c.Patch == nil {
		return changes
	}

	scanner := bufio.NewReader(c.Patch)
	var scanErr error
	for {
		lineB, isPrefix, err := scanner.ReadLine()
		if isPrefix {
			// If a single line overflowed the buffer, don't bother processing it as
			// it's likey part of a file and not relevant to the patch.
			continue
		}
		if err != nil {
			scanErr = err
			break
		}
		line := strings.TrimRight(string(lineB), "\n")

		c.debugf(line)
		s.lineNo++
		s.hunkPos++
		switch {
		case strings.HasPrefix(line, "+++ ") && len(line) > 4:
			if s.changes != nil {
				// record the last state
				changes[s.file] = s.changes
			}
			// 6 removes "+++ b/"
			s = state{file: line[6:], hunkPos: -1, changes: []pos{}}
		case strings.HasPrefix(line, "@@ "):
			//      @@ -1 +2,4 @@
			// chdr ^^^^^^^^^^^^^
			// ahdr       ^^^^
			// cstart      ^
			chdr := strings.Split(line, " ")
			ahdr := strings.Split(chdr[2], ",")
			// [1:] to remove leading plus
			cstart, err := strconv.ParseUint(ahdr[0][1:], 10, 64)
			if err != nil {
				panic(err)
			}
			s.lineNo = int(cstart) - 1 // -1 as cstart is the next line number
		case strings.HasPrefix(line, "-"):
			s.lineNo--
		case strings.HasPrefix(line, "+"):
			s.changes = append(s.changes, pos{lineNo: s.lineNo, hunkPos: s.hunkPos})
		}

	}
	if scanErr != nil && scanErr != io.EOF {
		fmt.Fprintln(os.Stderr, "reading standard input:", scanErr)
	}
	// record the last state
	changes[s.file] = s.changes

	return changes
}

// GitPatch returns a patch from a git repository, if no git repository was
// was found and no errors occurred, nil is returned, else an error is returned
// revisionFrom and revisionTo defines the git diff parameters, if left blank
// and there are unstaged changes or untracked files, only those will be returned
// else only check changes since HEAD~. If revisionFrom is set but revisionTo
// is not, untracked files will be included, to exclude untracked files set
// revisionTo to HEAD~. It's incorrect to specify revisionTo without a
// revisionFrom.
func GitPatch(revisionFrom, revisionTo string) (io.Reader, []string, error) {
	var patch bytes.Buffer

	// check if git repo exists
	if err := exec.Command("git", "status").Run(); err != nil {
		// don't return an error, we assume the error is not repo exists
		return nil, nil, nil
	}

	// make a patch for untracked files
	var newFiles []string
	ls, err := exec.Command("git", "ls-files", "--others", "--exclude-standard").CombinedOutput()
	if err != nil {
		return nil, nil, fmt.Errorf("error executing git ls-files: %s", err)
	}
	for _, file := range bytes.Split(ls, []byte{'\n'}) {
		if len(file) == 0 || bytes.HasSuffix(file, []byte{'/'}) {
			// ls-files was sometimes showing directories when they were ignored
			// I couldn't create a test case for this as I couldn't reproduce correctly
			// for the moment, just exclude files with trailing /
			continue
		}
		newFiles = append(newFiles, string(file))
	}

	if revisionFrom != "" {
		cmd := exec.Command("git", "diff", "--relative", revisionFrom)
		if revisionTo != "" {
			cmd.Args = append(cmd.Args, revisionTo)
		}
		cmd.Stdout = &patch
		if err := cmd.Run(); err != nil {
			return nil, nil, fmt.Errorf("error executing git diff %q %q: %s", revisionFrom, revisionTo, err)
		}

		if revisionTo == "" {
			return &patch, newFiles, nil
		}
		return &patch, nil, nil
	}

	// make a patch for unstaged changes
	// use --no-prefix to remove b/ given: +++ b/main.go
	cmd := exec.Command("git", "diff", "--relative")
	cmd.Stdout = &patch
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("error executing git diff: %s", err)
	}
	unstaged := patch.Len() > 0

	// If there's unstaged changes OR untracked changes (or both), then this is
	// a suitable patch
	if unstaged || newFiles != nil {
		return &patch, newFiles, nil
	}

	// check for changes in recent commit

	cmd = exec.Command("git", "diff", "--relative", "HEAD~")
	cmd.Stdout = &patch
	if err := cmd.Run(); err != nil {
		return nil, nil, fmt.Errorf("error executing git diff HEAD~: %s", err)
	}

	return &patch, nil, nil
}
