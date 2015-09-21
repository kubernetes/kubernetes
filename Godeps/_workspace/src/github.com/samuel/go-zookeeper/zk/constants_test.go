package zk

import (
	"fmt"
	"testing"
)

func TestModeString(t *testing.T) {
	if fmt.Sprintf("%v", ModeUnknown) != "unknown" {
		t.Errorf("unknown value should be 'unknown'")
	}

	if fmt.Sprintf("%v", ModeLeader) != "leader" {
		t.Errorf("leader value should be 'leader'")
	}

	if fmt.Sprintf("%v", ModeFollower) != "follower" {
		t.Errorf("follower value should be 'follower'")
	}

	if fmt.Sprintf("%v", ModeStandalone) != "standalone" {
		t.Errorf("standlone value should be 'standalone'")
	}
}
