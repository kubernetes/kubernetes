package libcontainer

import "testing"

func TestErrorCode(t *testing.T) {
	codes := map[ErrorCode]string{
		IdInUse:            "Id already in use",
		InvalidIdFormat:    "Invalid format",
		ContainerPaused:    "Container paused",
		ConfigInvalid:      "Invalid configuration",
		SystemError:        "System error",
		ContainerNotExists: "Container does not exist",
	}

	for code, expected := range codes {
		if actual := code.String(); actual != expected {
			t.Fatalf("expected string %q but received %q", expected, actual)
		}
	}
}
