package example

import (
	"testing"
)

func TestParse(t *testing.T) {
	load := func(fname string, dest *ComponentConfigControllerManager) {
		dest.Foo.F = 100
		dest.Bar = nil
	}

	cfg, err := Parse([]string{"--config", "config.json", "--bar-g", "10"}, load)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	t.Logf("cfg = %#v", cfg)

	if cfg.Foo.F != 100 {
		t.Errorf("cfg.Bar.F expected to be 100, not %v", cfg.Foo.F)
	}
	if cfg.Bar == nil {
		t.Fatalf("cfg.Bar shouldn't be nil")
	}
	if cfg.Bar.G != 10 {
		t.Errorf("cfg.Bar.G expected to be 10, not %v", cfg.Bar.G)
	}
}
