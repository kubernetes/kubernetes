package testing

import (
	"github.com/stretchr/testify/assert"
	"k8s.io/component-helpers/node/utils/sysctl"
	"testing"
)

//
func Test_Write_Int(t *testing.T) {
	sys := sysctl.NewFs()
	path := "/Users/mac/sys/module/nf_conntrack/parameters/hashsize"
	WRInt(t, sys, path, 1)
	WRInt(t, sys, path, -1)
	WRInt(t, sys, path, 0)
	WRInt(t, sys, path, 65535)
	WRInt(t, sys, path, -65536)
	WRInt(t, sys, path, 2147483647)
	WRInt(t, sys, path, -2147483648)
}

func WRInt(t *testing.T, sys sysctl.SysfsInterface, path string, value int) {
	sys.WriteInt(path, value)
	res, _ := sys.ReadInt(path)
	assert.Equal(t, value, res)
}

func Test_Str(t *testing.T) {
	sys := sysctl.NewFs()
	path := "/Users/mac/sys/module/nf_conntrack/parameters/hashsize"
	WRStr(t, sys, path, "1")
	WRStr(t, sys, path, "-1")
	WRStr(t, sys, path, "0")
	WRStr(t, sys, path, "65535")
	WRStr(t, sys, path, "-65536")
	WRStr(t, sys, path, "2147483647")
	WRStr(t, sys, path, "-2147483648")
	WRStr(t, sys, path, "abcdefghigklmnopqrstuvwxyz")
	WRStr(t, sys, path, "1234567890")
}

func WRStr(t *testing.T, sys sysctl.SysfsInterface, path string, value string) {
	sys.WriteStr(path, value)
	res, _ := sys.ReadStr(path)
	assert.Equal(t, value, res)
}

func Test_Float(t *testing.T) {
	sys := sysctl.NewFs()
	path := "/Users/mac/sys/module/nf_conntrack/parameters/hashsize"
	WRFloat(t, sys, path, 1.1)
	WRFloat(t, sys, path, -1.0)
	WRFloat(t, sys, path, 0.123)
	WRFloat(t, sys, path, 65535.3123131)
	WRFloat(t, sys, path, -65536.312312)
	WRFloat(t, sys, path, 2147483647.42342342)
	WRFloat(t, sys, path, -2147483648.423423424242)
}

func WRFloat(t *testing.T, sys sysctl.SysfsInterface, path string, value float64) {
	sys.WriteFloat(path, value)
	res, _ := sys.ReadFloat(path)
	assert.Equal(t, value, res)
}

func Test_Bool(t *testing.T) {
	sys := sysctl.NewFs()
	path := "/Users/mac/sys/module/nf_conntrack/parameters/hashsize"
	WRBool(t, sys, path, true)
	WRBool(t, sys, path, false)
}

func WRBool(t *testing.T, sys sysctl.SysfsInterface, path string, value bool) {
	sys.WriteBool(path, value)
	res, _ := sys.ReadBool(path)
	assert.Equal(t, value, res)
}