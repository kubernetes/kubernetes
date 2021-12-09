// Copyright (c) 2015 HPE Software Inc. All rights reserved.
// Copyright (c) 2013 ActiveState Software Inc. All rights reserved.

package watch

import (
	"log"
	"os"
	"path/filepath"
	"sync"
	"syscall"

	"github.com/hpcloud/tail/util"

	"gopkg.in/fsnotify.v1"
)

type InotifyTracker struct {
	mux       sync.Mutex
	watcher   *fsnotify.Watcher
	chans     map[string]chan fsnotify.Event
	done      map[string]chan bool
	watchNums map[string]int
	watch     chan *watchInfo
	remove    chan *watchInfo
	error     chan error
}

type watchInfo struct {
	op    fsnotify.Op
	fname string
}

func (this *watchInfo) isCreate() bool {
	return this.op == fsnotify.Create
}

var (
	// globally shared InotifyTracker; ensures only one fsnotify.Watcher is used
	shared *InotifyTracker

	// these are used to ensure the shared InotifyTracker is run exactly once
	once  = sync.Once{}
	goRun = func() {
		shared = &InotifyTracker{
			mux:       sync.Mutex{},
			chans:     make(map[string]chan fsnotify.Event),
			done:      make(map[string]chan bool),
			watchNums: make(map[string]int),
			watch:     make(chan *watchInfo),
			remove:    make(chan *watchInfo),
			error:     make(chan error),
		}
		go shared.run()
	}

	logger = log.New(os.Stderr, "", log.LstdFlags)
)

// Watch signals the run goroutine to begin watching the input filename
func Watch(fname string) error {
	return watch(&watchInfo{
		fname: fname,
	})
}

// Watch create signals the run goroutine to begin watching the input filename
// if call the WatchCreate function, don't call the Cleanup, call the RemoveWatchCreate
func WatchCreate(fname string) error {
	return watch(&watchInfo{
		op:    fsnotify.Create,
		fname: fname,
	})
}

func watch(winfo *watchInfo) error {
	// start running the shared InotifyTracker if not already running
	once.Do(goRun)

	winfo.fname = filepath.Clean(winfo.fname)
	shared.watch <- winfo
	return <-shared.error
}

// RemoveWatch signals the run goroutine to remove the watch for the input filename
func RemoveWatch(fname string) {
	remove(&watchInfo{
		fname: fname,
	})
}

// RemoveWatch create signals the run goroutine to remove the watch for the input filename
func RemoveWatchCreate(fname string) {
	remove(&watchInfo{
		op:    fsnotify.Create,
		fname: fname,
	})
}

func remove(winfo *watchInfo) {
	// start running the shared InotifyTracker if not already running
	once.Do(goRun)

	winfo.fname = filepath.Clean(winfo.fname)
	shared.mux.Lock()
	done := shared.done[winfo.fname]
	if done != nil {
		delete(shared.done, winfo.fname)
		close(done)
	}

	fname := winfo.fname
	if winfo.isCreate() {
		// Watch for new files to be created in the parent directory.
		fname = filepath.Dir(fname)
	}
	shared.watchNums[fname]--
	watchNum := shared.watchNums[fname]
	if watchNum == 0 {
		delete(shared.watchNums, fname)
	}
	shared.mux.Unlock()

	// If we were the last ones to watch this file, unsubscribe from inotify.
	// This needs to happen after releasing the lock because fsnotify waits
	// synchronously for the kernel to acknowledge the removal of the watch
	// for this file, which causes us to deadlock if we still held the lock.
	if watchNum == 0 {
		shared.watcher.Remove(fname)
	}
	shared.remove <- winfo
}

// Events returns a channel to which FileEvents corresponding to the input filename
// will be sent. This channel will be closed when removeWatch is called on this
// filename.
func Events(fname string) <-chan fsnotify.Event {
	shared.mux.Lock()
	defer shared.mux.Unlock()

	return shared.chans[fname]
}

// Cleanup removes the watch for the input filename if necessary.
func Cleanup(fname string) {
	RemoveWatch(fname)
}

// watchFlags calls fsnotify.WatchFlags for the input filename and flags, creating
// a new Watcher if the previous Watcher was closed.
func (shared *InotifyTracker) addWatch(winfo *watchInfo) error {
	shared.mux.Lock()
	defer shared.mux.Unlock()

	if shared.chans[winfo.fname] == nil {
		shared.chans[winfo.fname] = make(chan fsnotify.Event)
		shared.done[winfo.fname] = make(chan bool)
	}

	fname := winfo.fname
	if winfo.isCreate() {
		// Watch for new files to be created in the parent directory.
		fname = filepath.Dir(fname)
	}

	// already in inotify watch
	if shared.watchNums[fname] > 0 {
		shared.watchNums[fname]++
		if winfo.isCreate() {
			shared.watchNums[winfo.fname]++
		}
		return nil
	}

	err := shared.watcher.Add(fname)
	if err == nil {
		shared.watchNums[fname]++
		if winfo.isCreate() {
			shared.watchNums[winfo.fname]++
		}
	}
	return err
}

// removeWatch calls fsnotify.RemoveWatch for the input filename and closes the
// corresponding events channel.
func (shared *InotifyTracker) removeWatch(winfo *watchInfo) {
	shared.mux.Lock()
	defer shared.mux.Unlock()

	ch := shared.chans[winfo.fname]
	if ch == nil {
		return
	}

	delete(shared.chans, winfo.fname)
	close(ch)

	if !winfo.isCreate() {
		return
	}

	shared.watchNums[winfo.fname]--
	if shared.watchNums[winfo.fname] == 0 {
		delete(shared.watchNums, winfo.fname)
	}
}

// sendEvent sends the input event to the appropriate Tail.
func (shared *InotifyTracker) sendEvent(event fsnotify.Event) {
	name := filepath.Clean(event.Name)

	shared.mux.Lock()
	ch := shared.chans[name]
	done := shared.done[name]
	shared.mux.Unlock()

	if ch != nil && done != nil {
		select {
		case ch <- event:
		case <-done:
		}
	}
}

// run starts the goroutine in which the shared struct reads events from its
// Watcher's Event channel and sends the events to the appropriate Tail.
func (shared *InotifyTracker) run() {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		util.Fatal("failed to create Watcher")
	}
	shared.watcher = watcher

	for {
		select {
		case winfo := <-shared.watch:
			shared.error <- shared.addWatch(winfo)

		case winfo := <-shared.remove:
			shared.removeWatch(winfo)

		case event, open := <-shared.watcher.Events:
			if !open {
				return
			}
			shared.sendEvent(event)

		case err, open := <-shared.watcher.Errors:
			if !open {
				return
			} else if err != nil {
				sysErr, ok := err.(*os.SyscallError)
				if !ok || sysErr.Err != syscall.EINTR {
					logger.Printf("Error in Watcher Error channel: %s", err)
				}
			}
		}
	}
}
