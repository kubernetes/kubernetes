package testing

import (
	"k8s.io/component-helpers/node/utils/sysctl"
	"testing"
)

//
func Test_Write_Int(t *testing.T) {
	sys := sysctl.NewFs()
	var path string
	var err error

	path = "/var/sys/fs/int_fs_0"
	err = sys.WriteInt(path,0)
	if err != nil{
		t.Errorf(" Int value is written to the system file failed (%v)",err.Error())
	}

	path = "/var/sys/fs/int_fs_1"
	err = sys.WriteInt(path,1)
	if err != nil{
		t.Errorf(" Int value is written to the system file failed (%v)",err.Error())
	}

	path = "/var/sys/fs/int_fs_n1"
	err = sys.WriteInt(path,-1)
	if err != nil{
		t.Errorf(" Int value is written to the system file failed (%v)",err.Error())
	}

	path = "/var/sys/fs/int_fs_9999999999999999999"
	err = sys.WriteInt(path,2147483646)
	if err != nil{
		t.Errorf(" Int value is written to the system file failed (%v)",err.Error())
	}

	path = "/var/sys/fs/int_fs_n9999999999999999999"
	err = sys.WriteInt(path,-2147483647)
	if err != nil{
		t.Errorf(" Int value is written to the system file failed (%v)",err.Error())
	}

}

func Test_Str(t *testing.T) {

}

func Test_Float(t *testing.T) {

}

func Test_Bool(t *testing.T) {

}