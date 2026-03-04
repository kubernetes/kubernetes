package watch

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"
)

var goTestRegExp = regexp.MustCompile(`_test\.go$`)

type PackageHash struct {
	CodeModifiedTime time.Time
	TestModifiedTime time.Time
	Deleted          bool

	path        string
	codeHash    string
	testHash    string
	watchRegExp *regexp.Regexp
}

func NewPackageHash(path string, watchRegExp *regexp.Regexp) *PackageHash {
	p := &PackageHash{
		path:        path,
		watchRegExp: watchRegExp,
	}

	p.codeHash, _, p.testHash, _, p.Deleted = p.computeHashes()

	return p
}

func (p *PackageHash) CheckForChanges() bool {
	codeHash, codeModifiedTime, testHash, testModifiedTime, deleted := p.computeHashes()

	if deleted {
		if !p.Deleted {
			t := time.Now()
			p.CodeModifiedTime = t
			p.TestModifiedTime = t
		}
		p.Deleted = true
		return true
	}

	modified := false
	p.Deleted = false

	if p.codeHash != codeHash {
		p.CodeModifiedTime = codeModifiedTime
		modified = true
	}
	if p.testHash != testHash {
		p.TestModifiedTime = testModifiedTime
		modified = true
	}

	p.codeHash = codeHash
	p.testHash = testHash
	return modified
}

func (p *PackageHash) computeHashes() (codeHash string, codeModifiedTime time.Time, testHash string, testModifiedTime time.Time, deleted bool) {
	entries, err := os.ReadDir(p.path)

	if err != nil {
		deleted = true
		return
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		if isHiddenFile(info) {
			continue
		}

		if goTestRegExp.MatchString(info.Name()) {
			testHash += p.hashForFileInfo(info)
			if info.ModTime().After(testModifiedTime) {
				testModifiedTime = info.ModTime()
			}
			continue
		}

		if p.watchRegExp.MatchString(info.Name()) {
			codeHash += p.hashForFileInfo(info)
			if info.ModTime().After(codeModifiedTime) {
				codeModifiedTime = info.ModTime()
			}
		}
	}

	testHash += codeHash
	if codeModifiedTime.After(testModifiedTime) {
		testModifiedTime = codeModifiedTime
	}

	return
}

func isHiddenFile(info os.FileInfo) bool {
	return strings.HasPrefix(info.Name(), ".") || strings.HasPrefix(info.Name(), "_")
}

func (p *PackageHash) hashForFileInfo(info os.FileInfo) string {
	return fmt.Sprintf("%s_%d_%d", info.Name(), info.Size(), info.ModTime().UnixNano())
}
