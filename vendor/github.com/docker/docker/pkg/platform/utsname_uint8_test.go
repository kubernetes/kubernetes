// +build linux,arm linux,ppc64 linux,ppc64le

package platform

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTestCharToString(t *testing.T) {
	machineInBytes := [65]uint8{120, 56, 54, 95, 54, 52}
	machineInString := charsToString(machineInBytes)
	assert.NotNil(t, machineInString, "Unable to convert char into string.")
	assert.Equal(t, string("x86_64"), machineInString, "Parsed machine code not equal.")
}
