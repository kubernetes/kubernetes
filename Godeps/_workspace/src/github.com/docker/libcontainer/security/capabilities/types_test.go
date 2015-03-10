package capabilities

import (
	"testing"
)

func TestCapabilitiesContains(t *testing.T) {
	caps := Capabilities{
		GetCapability("MKNOD"),
		GetCapability("SETPCAP"),
	}

	if caps.contains("SYS_ADMIN") {
		t.Fatal("capabilities should not contain SYS_ADMIN")
	}
	if !caps.contains("MKNOD") {
		t.Fatal("capabilities should contain MKNOD but does not")
	}
}
