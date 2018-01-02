// +build linux,386 linux,amd64 linux,arm64 s390x

package platform

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCharToString(t *testing.T) {
	machineInBytes := [65]int8{120, 56, 54, 95, 54, 52}
	machineInString := charsToString(machineInBytes)
	assert.NotNil(t, machineInString, "Unable to convert char into string.")
	assert.Equal(t, string("x86_64"), machineInString, "Parsed machine code not equal.")
}
