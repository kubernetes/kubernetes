package state

import (
	"testing"
)

func TestNotifyGroup(t *testing.T) {
	grp := &NotifyGroup{}

	ch1 := grp.WaitCh()
	ch2 := grp.WaitCh()

	select {
	case <-ch1:
		t.Fatalf("should block")
	default:
	}
	select {
	case <-ch2:
		t.Fatalf("should block")
	default:
	}

	grp.Notify()

	select {
	case <-ch1:
	default:
		t.Fatalf("should not block")
	}
	select {
	case <-ch2:
	default:
		t.Fatalf("should not block")
	}

	// Should be unregistered
	ch3 := grp.WaitCh()
	grp.Notify()

	select {
	case <-ch1:
		t.Fatalf("should block")
	default:
	}
	select {
	case <-ch2:
		t.Fatalf("should block")
	default:
	}
	select {
	case <-ch3:
	default:
		t.Fatalf("should not block")
	}
}

func TestNotifyGroup_Clear(t *testing.T) {
	grp := &NotifyGroup{}

	ch1 := grp.WaitCh()
	grp.Clear(ch1)

	grp.Notify()

	// Should not get anything
	select {
	case <-ch1:
		t.Fatalf("should not get message")
	default:
	}
}
