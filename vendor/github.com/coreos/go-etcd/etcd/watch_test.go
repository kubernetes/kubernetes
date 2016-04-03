package etcd

import (
	"fmt"
	"runtime"
	"testing"
	"time"
)

func TestWatch(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("watch_foo", true)
	}()

	go setHelper("watch_foo", "bar", c)

	resp, err := c.Watch("watch_foo", 0, false, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/watch_foo" && resp.Node.Value == "bar") {
		t.Fatalf("Watch 1 failed: %#v", resp)
	}

	go setHelper("watch_foo", "bar", c)

	resp, err = c.Watch("watch_foo", resp.Node.ModifiedIndex+1, false, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/watch_foo" && resp.Node.Value == "bar") {
		t.Fatalf("Watch 2 failed: %#v", resp)
	}

	routineNum := runtime.NumGoroutine()

	ch := make(chan *Response, 10)
	stop := make(chan bool, 1)

	go setLoop("watch_foo", "bar", c)

	go receiver(ch, stop)

	_, err = c.Watch("watch_foo", 0, false, ch, stop)
	if err != ErrWatchStoppedByUser {
		t.Fatalf("Watch returned a non-user stop error")
	}

	if newRoutineNum := runtime.NumGoroutine(); newRoutineNum != routineNum {
		t.Fatalf("Routine numbers differ after watch stop: %v, %v", routineNum, newRoutineNum)
	}
}

func TestWatchAll(t *testing.T) {
	c := NewClient(nil)
	defer func() {
		c.Delete("watch_foo", true)
	}()

	go setHelper("watch_foo/foo", "bar", c)

	resp, err := c.Watch("watch_foo", 0, true, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/watch_foo/foo" && resp.Node.Value == "bar") {
		t.Fatalf("WatchAll 1 failed: %#v", resp)
	}

	go setHelper("watch_foo/foo", "bar", c)

	resp, err = c.Watch("watch_foo", resp.Node.ModifiedIndex+1, true, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !(resp.Node.Key == "/watch_foo/foo" && resp.Node.Value == "bar") {
		t.Fatalf("WatchAll 2 failed: %#v", resp)
	}

	ch := make(chan *Response, 10)
	stop := make(chan bool, 1)

	routineNum := runtime.NumGoroutine()

	go setLoop("watch_foo/foo", "bar", c)

	go receiver(ch, stop)

	_, err = c.Watch("watch_foo", 0, true, ch, stop)
	if err != ErrWatchStoppedByUser {
		t.Fatalf("Watch returned a non-user stop error")
	}

	if newRoutineNum := runtime.NumGoroutine(); newRoutineNum != routineNum {
		t.Fatalf("Routine numbers differ after watch stop: %v, %v", routineNum, newRoutineNum)
	}
}

func setHelper(key, value string, c *Client) {
	time.Sleep(time.Second)
	c.Set(key, value, 100)
}

func setLoop(key, value string, c *Client) {
	time.Sleep(time.Second)
	for i := 0; i < 10; i++ {
		newValue := fmt.Sprintf("%s_%v", value, i)
		c.Set(key, newValue, 100)
		time.Sleep(time.Second / 10)
	}
}

func receiver(c chan *Response, stop chan bool) {
	for i := 0; i < 10; i++ {
		<-c
	}
	stop <- true
}
