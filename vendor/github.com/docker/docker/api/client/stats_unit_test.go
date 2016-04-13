package client

import (
	"bytes"
	"sync"
	"testing"
)

func TestDisplay(t *testing.T) {
	c := &containerStats{
		Name:             "app",
		CPUPercentage:    30.0,
		Memory:           100 * 1024 * 1024.0,
		MemoryLimit:      2048 * 1024 * 1024.0,
		MemoryPercentage: 100.0 / 2048.0 * 100.0,
		NetworkRx:        100 * 1024 * 1024,
		NetworkTx:        800 * 1024 * 1024,
		mu:               sync.RWMutex{},
	}
	var b bytes.Buffer
	if err := c.Display(&b); err != nil {
		t.Fatalf("c.Display() gave error: %s", err)
	}
	got := b.String()
	want := "app\t30.00%\t104.9 MB/2.147 GB\t4.88%\t104.9 MB/838.9 MB\n"
	if got != want {
		t.Fatalf("c.Display() = %q, want %q", got, want)
	}
}
