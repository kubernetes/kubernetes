package state

import (
	"sort"
	"strings"
	"testing"
)

// verifyWatch will set up a watch channel, call the given function, and then
// make sure the watch fires.
func verifyWatch(t *testing.T, watch Watch, fn func()) {
	ch := make(chan struct{}, 1)
	watch.Wait(ch)

	fn()

	select {
	case <-ch:
	default:
		t.Fatalf("watch should have been notified")
	}
}

// verifyNoWatch will set up a watch channel, call the given function, and then
// make sure the watch never fires.
func verifyNoWatch(t *testing.T, watch Watch, fn func()) {
	ch := make(chan struct{}, 1)
	watch.Wait(ch)

	fn()

	select {
	case <-ch:
		t.Fatalf("watch should not been notified")
	default:
	}
}

func TestWatch_FullTableWatch(t *testing.T) {
	w := NewFullTableWatch()

	// Test the basic trigger with a single watcher.
	verifyWatch(t, w, func() {
		w.Notify()
	})

	// Run multiple watchers and make sure they both fire.
	verifyWatch(t, w, func() {
		verifyWatch(t, w, func() {
			w.Notify()
		})
	})

	// Make sure clear works.
	ch := make(chan struct{}, 1)
	w.Wait(ch)
	w.Clear(ch)
	w.Notify()
	select {
	case <-ch:
		t.Fatalf("watch should not have been notified")
	default:
	}

	// Make sure notify is a one shot.
	w.Wait(ch)
	w.Notify()
	select {
	case <-ch:
	default:
		t.Fatalf("watch should have been notified")
	}
	w.Notify()
	select {
	case <-ch:
		t.Fatalf("watch should not have been notified")
	default:
	}
}

func TestWatch_DumbWatchManager(t *testing.T) {
	watches := map[string]*FullTableWatch{
		"alice": NewFullTableWatch(),
		"bob":   NewFullTableWatch(),
		"carol": NewFullTableWatch(),
	}

	// Notify with nothing armed and make sure nothing triggers.
	func() {
		w := NewDumbWatchManager(watches)
		verifyNoWatch(t, watches["alice"], func() {
			verifyNoWatch(t, watches["bob"], func() {
				verifyNoWatch(t, watches["carol"], func() {
					w.Notify()
				})
			})
		})
	}()

	// Trigger one watch.
	func() {
		w := NewDumbWatchManager(watches)
		verifyWatch(t, watches["alice"], func() {
			verifyNoWatch(t, watches["bob"], func() {
				verifyNoWatch(t, watches["carol"], func() {
					w.Arm("alice")
					w.Notify()
				})
			})
		})
	}()

	// Trigger two watches.
	func() {
		w := NewDumbWatchManager(watches)
		verifyWatch(t, watches["alice"], func() {
			verifyNoWatch(t, watches["bob"], func() {
				verifyWatch(t, watches["carol"], func() {
					w.Arm("alice")
					w.Arm("carol")
					w.Notify()
				})
			})
		})
	}()

	// Trigger all three watches.
	func() {
		w := NewDumbWatchManager(watches)
		verifyWatch(t, watches["alice"], func() {
			verifyWatch(t, watches["bob"], func() {
				verifyWatch(t, watches["carol"], func() {
					w.Arm("alice")
					w.Arm("bob")
					w.Arm("carol")
					w.Notify()
				})
			})
		})
	}()

	// Trigger multiple times.
	func() {
		w := NewDumbWatchManager(watches)
		verifyWatch(t, watches["alice"], func() {
			verifyNoWatch(t, watches["bob"], func() {
				verifyNoWatch(t, watches["carol"], func() {
					w.Arm("alice")
					w.Arm("alice")
					w.Notify()
				})
			})
		})
	}()

	// Make sure it panics when asked to arm an unknown table.
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("didn't get expected panic")
			}
		}()
		w := NewDumbWatchManager(watches)
		w.Arm("nope")
	}()
}

func verifyWatches(t *testing.T, w *PrefixWatchManager, expected string) {
	var found []string
	fn := func(k string, v interface{}) bool {
		if k == "" {
			k = "(full)"
		}
		found = append(found, k)
		return false
	}
	w.watches.WalkPrefix("", fn)

	sort.Strings(found)
	actual := strings.Join(found, "|")
	if expected != actual {
		t.Fatalf("bad: %s != %s", expected, actual)
	}
}

func TestWatch_PrefixWatchManager(t *testing.T) {
	w := NewPrefixWatchManager()
	verifyWatches(t, w, "")

	// This will create the watch group.
	ch1 := make(chan struct{}, 1)
	w.Wait("hello", ch1)
	verifyWatches(t, w, "hello")

	// This will add to the existing one.
	ch2 := make(chan struct{}, 1)
	w.Wait("hello", ch2)
	verifyWatches(t, w, "hello")

	// This will add to the existing as well.
	ch3 := make(chan struct{}, 1)
	w.Wait("hello", ch3)
	verifyWatches(t, w, "hello")

	// Remove one of the watches.
	w.Clear("hello", ch2)
	verifyWatches(t, w, "hello")

	// Do "clear" for one that was never added.
	ch4 := make(chan struct{}, 1)
	w.Clear("hello", ch4)
	verifyWatches(t, w, "hello")

	// Add a full table watch.
	full := make(chan struct{}, 1)
	w.Wait("", full)
	verifyWatches(t, w, "(full)|hello")

	// Add another channel for a different prefix.
	nope := make(chan struct{}, 1)
	w.Wait("nope", nope)
	verifyWatches(t, w, "(full)|hello|nope")

	// Fire off the notification and make sure channels were pinged (or not)
	// as expected.
	w.Notify("hello", false)
	verifyWatches(t, w, "(full)|nope")
	select {
	case <-ch1:
	default:
		t.Fatalf("ch1 should have been notified")
	}
	select {
	case <-ch2:
		t.Fatalf("ch2 should not have been notified")
	default:
	}
	select {
	case <-ch3:
	default:
		t.Fatalf("ch3 should have been notified")
	}
	select {
	case <-ch4:
		t.Fatalf("ch4 should not have been notified")
	default:
	}
	select {
	case <-nope:
		t.Fatalf("nope should not have been notified")
	default:
	}
	select {
	case <-full:
	default:
		t.Fatalf("full should have been notified")
	}
}

func TestWatch_PrefixWatch(t *testing.T) {
	w := NewPrefixWatchManager()

	// Hit a specific key.
	verifyWatch(t, w.NewPrefixWatch(""), func() {
		verifyWatch(t, w.NewPrefixWatch("foo/bar/baz"), func() {
			verifyNoWatch(t, w.NewPrefixWatch("foo/bar/zoo"), func() {
				verifyNoWatch(t, w.NewPrefixWatch("nope"), func() {
					w.Notify("foo/bar/baz", false)
				})
			})
		})
	})

	// Make sure cleanup is happening. All that should be left is the
	// full-table watch and the un-fired watches.
	verifyWatches(t, w, "(full)|foo/bar/zoo|nope")

	// Delete a subtree.
	verifyWatch(t, w.NewPrefixWatch(""), func() {
		verifyWatch(t, w.NewPrefixWatch("foo/bar/baz"), func() {
			verifyWatch(t, w.NewPrefixWatch("foo/bar/zoo"), func() {
				verifyNoWatch(t, w.NewPrefixWatch("nope"), func() {
					w.Notify("foo/", true)
				})
			})
		})
	})
	verifyWatches(t, w, "(full)|nope")

	// Hit an unknown key.
	verifyWatch(t, w.NewPrefixWatch(""), func() {
		verifyNoWatch(t, w.NewPrefixWatch("foo/bar/baz"), func() {
			verifyNoWatch(t, w.NewPrefixWatch("foo/bar/zoo"), func() {
				verifyNoWatch(t, w.NewPrefixWatch("nope"), func() {
					w.Notify("not/in/there", false)
				})
			})
		})
	})
	verifyWatches(t, w, "(full)|foo/bar/baz|foo/bar/zoo|nope")

	// Make sure a watch can be reused.
	watch := w.NewPrefixWatch("over/and/over")
	for i := 0; i < 10; i++ {
		verifyWatch(t, watch, func() {
			w.Notify("over/and/over", false)
		})
	}
}

type MockWatch struct {
	Waits  map[chan struct{}]int
	Clears map[chan struct{}]int
}

func NewMockWatch() *MockWatch {
	return &MockWatch{
		Waits:  make(map[chan struct{}]int),
		Clears: make(map[chan struct{}]int),
	}
}

func (m *MockWatch) Wait(notifyCh chan struct{}) {
	if _, ok := m.Waits[notifyCh]; ok {
		m.Waits[notifyCh]++
	} else {
		m.Waits[notifyCh] = 1
	}
}

func (m *MockWatch) Clear(notifyCh chan struct{}) {
	if _, ok := m.Clears[notifyCh]; ok {
		m.Clears[notifyCh]++
	} else {
		m.Clears[notifyCh] = 1
	}
}

func TestWatch_MultiWatch(t *testing.T) {
	w1, w2 := NewMockWatch(), NewMockWatch()
	w := NewMultiWatch(w1, w2)

	// Do some activity.
	c1, c2 := make(chan struct{}), make(chan struct{})
	w.Wait(c1)
	w.Clear(c1)
	w.Wait(c1)
	w.Wait(c2)
	w.Clear(c1)
	w.Clear(c2)

	// Make sure all the events were forwarded.
	if cnt, ok := w1.Waits[c1]; !ok || cnt != 2 {
		t.Fatalf("bad: %d", w1.Waits[c1])
	}
	if cnt, ok := w1.Clears[c1]; !ok || cnt != 2 {
		t.Fatalf("bad: %d", w1.Clears[c1])
	}
	if cnt, ok := w1.Waits[c2]; !ok || cnt != 1 {
		t.Fatalf("bad: %d", w1.Waits[c2])
	}
	if cnt, ok := w1.Clears[c2]; !ok || cnt != 1 {
		t.Fatalf("bad: %d", w1.Clears[c2])
	}
	if cnt, ok := w2.Waits[c1]; !ok || cnt != 2 {
		t.Fatalf("bad: %d", w2.Waits[c1])
	}
	if cnt, ok := w2.Clears[c1]; !ok || cnt != 2 {
		t.Fatalf("bad: %d", w2.Clears[c1])
	}
	if cnt, ok := w2.Waits[c2]; !ok || cnt != 1 {
		t.Fatalf("bad: %d", w2.Waits[c2])
	}
	if cnt, ok := w2.Clears[c2]; !ok || cnt != 1 {
		t.Fatalf("bad: %d", w2.Clears[c2])
	}
}
