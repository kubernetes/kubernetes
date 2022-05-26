package filewatcher

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"gotest.tools/gotestsum/log"
)

const maxDepth = 7

type RunOptions struct {
	PkgPath string
	Debug   bool
	resume  chan struct{}
}

func Watch(dirs []string, run func(opts RunOptions) error) error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	toWatch := findAllDirs(dirs, maxDepth)
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("failed to create file watcher: %w", err)
	}
	defer watcher.Close() // nolint: errcheck // always returns nil error

	fmt.Printf("Watching %v directories. Use Ctrl-c to to stop a run or exit.\n", len(toWatch))
	for _, dir := range toWatch {
		if err = watcher.Add(dir); err != nil {
			return fmt.Errorf("failed to watch directory %v: %w", dir, err)
		}
	}

	timer := time.NewTimer(maxIdleTime)
	defer timer.Stop()

	redo := newRedoHandler()
	defer redo.ResetTerm()
	go redo.Run(ctx)

	h := &handler{last: time.Now(), fn: run}
	for {
		select {
		case <-timer.C:
			return fmt.Errorf("exceeded idle timeout while watching files")

		case opts := <-redo.Ch():
			resetTimer(timer)

			redo.ResetTerm()
			opts.PkgPath = h.lastPath
			if err := h.runTests(opts); err != nil {
				return fmt.Errorf("failed to rerun tests for %v: %v", opts.PkgPath, err)
			}
			redo.SetupTerm()
			close(opts.resume)

		case event := <-watcher.Events:
			resetTimer(timer)
			log.Debugf("handling event %v", event)

			if handleDirCreated(watcher, event) {
				continue
			}

			if err := h.handleEvent(event); err != nil {
				return fmt.Errorf("failed to run tests for %v: %v", event.Name, err)
			}

		case err := <-watcher.Errors:
			return fmt.Errorf("failed while watching files: %v", err)
		}
	}
}

const maxIdleTime = time.Hour

func resetTimer(timer *time.Timer) {
	if !timer.Stop() {
		<-timer.C
	}
	timer.Reset(maxIdleTime)
}

func findAllDirs(dirs []string, maxDepth int) []string {
	if len(dirs) == 0 {
		dirs = []string{"./..."}
	}

	var output []string // nolint: prealloc
	for _, dir := range dirs {
		const recur = "/..."
		if strings.HasSuffix(dir, recur) {
			dir = strings.TrimSuffix(dir, recur)
			output = append(output, findSubDirs(dir, maxDepth)...)
			continue
		}
		output = append(output, dir)
	}
	return output
}

func findSubDirs(rootDir string, maxDepth int) []string {
	var output []string
	// add root dir depth so that maxDepth is relative to the root dir
	maxDepth += pathDepth(rootDir)
	walker := func(path string, info os.FileInfo, err error) error {
		if err != nil {
			log.Warnf("failed to watch %v: %v", path, err)
			return nil
		}
		if !info.IsDir() {
			return nil
		}
		if pathDepth(path) > maxDepth || exclude(path) {
			log.Debugf("Ignoring %v because of max depth or exclude list", path)
			return filepath.SkipDir
		}
		if !hasGoFiles(path) {
			log.Debugf("Ignoring %v because it has no .go files", path)
			return nil
		}
		output = append(output, path)
		return nil
	}
	// nolint: errcheck // error is handled by walker func
	filepath.Walk(rootDir, walker)
	return output
}

func pathDepth(path string) int {
	return strings.Count(filepath.Clean(path), string(filepath.Separator))
}

// return true if path is vendor, testdata, or starts with a dot
func exclude(path string) bool {
	base := filepath.Base(path)
	switch {
	case strings.HasPrefix(base, ".") && len(base) > 1:
		return true
	case base == "vendor" || base == "testdata":
		return true
	}
	return false
}

func hasGoFiles(path string) bool {
	fh, err := os.Open(path)
	if err != nil {
		return false
	}
	defer fh.Close() // nolint: errcheck // fh is opened read-only

	for {
		names, err := fh.Readdirnames(20)
		switch {
		case err == io.EOF:
			return false
		case err != nil:
			log.Warnf("failed to read directory %v: %v", path, err)
			return false
		}

		for _, name := range names {
			if strings.HasSuffix(name, ".go") {
				return true
			}
		}
	}
}

func handleDirCreated(watcher *fsnotify.Watcher, event fsnotify.Event) (handled bool) {
	if event.Op&fsnotify.Create != fsnotify.Create {
		return false
	}

	fileInfo, err := os.Stat(event.Name)
	if err != nil {
		log.Warnf("failed to stat %s: %s", event.Name, err)
		return false
	}

	if !fileInfo.IsDir() {
		return false
	}

	if err := watcher.Add(event.Name); err != nil {
		log.Warnf("failed to watch new directory %v: %v", event.Name, err)
	}
	return true
}

type handler struct {
	last     time.Time
	lastPath string
	fn       func(opts RunOptions) error
}

const floodThreshold = 250 * time.Millisecond

func (h *handler) handleEvent(event fsnotify.Event) error {
	if event.Op&(fsnotify.Write|fsnotify.Create) == 0 {
		return nil
	}

	if !strings.HasSuffix(event.Name, ".go") {
		return nil
	}

	if time.Since(h.last) < floodThreshold {
		log.Debugf("skipping event received less than %v after the previous", floodThreshold)
		return nil
	}
	return h.runTests(RunOptions{PkgPath: "./" + filepath.Dir(event.Name)})
}

func (h *handler) runTests(opts RunOptions) error {
	fmt.Printf("\nRunning tests in %v\n", opts.PkgPath)

	if err := h.fn(opts); err != nil {
		return err
	}
	h.last = time.Now()
	h.lastPath = opts.PkgPath
	return nil
}
