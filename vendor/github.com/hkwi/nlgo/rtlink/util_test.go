package rtlink

import (
	"github.com/hkwi/nlgo"
	"testing"
)

func TestLo(t *testing.T) {
	if hub, err := nlgo.NewRtHub(); err != nil {
		t.Error(err)
	} else if info, err := GetByName(hub, "lo"); err != nil {
		t.Error(err)
	} else {
		t.Log(info)
	}
}

func TestIdx(t *testing.T) {
	if hub, err := nlgo.NewRtHub(); err != nil {
		t.Error(err)
	} else if name, err := GetNameByIndex(hub, 1); err != nil {
		t.Error(err)
	} else {
		t.Log(name)
	}
}
