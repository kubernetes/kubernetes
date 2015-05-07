package types

import (
	"reflect"
	"testing"
)

func TestMountPointFromString(t *testing.T) {
	tests := []struct {
		s     string
		mount MountPoint
	}{
		{
			"foobar,path=/tmp",
			MountPoint{
				Name:     "foobar",
				Path:     "/tmp",
				ReadOnly: false,
			},
		},
		{
			"foobar,path=/tmp,readOnly=false",
			MountPoint{
				Name:     "foobar",
				Path:     "/tmp",
				ReadOnly: false,
			},
		},
		{
			"foobar,path=/tmp,readOnly=true",
			MountPoint{
				Name:     "foobar",
				Path:     "/tmp",
				ReadOnly: true,
			},
		},
	}
	for i, tt := range tests {
		mount, err := MountPointFromString(tt.s)
		if err != nil {
			t.Errorf("#%d: got err=%v, want nil", i, err)
		}
		if !reflect.DeepEqual(*mount, tt.mount) {
			t.Errorf("#%d: mount=%v, want %v", i, *mount, tt.mount)
		}
	}
}

func TestMountPointFromStringBad(t *testing.T) {
	tests := []string{
		"#foobar,path=/tmp",
		"foobar,path=/tmp,readOnly=true,asdf=asdf",
		"foobar,path=/tmp,readOnly=maybe",
		"foobar,path=/tmp,readOnly=",
		"foobar,path=",
		"foobar",
		"",
		",path=/",
	}
	for i, in := range tests {
		l, err := MountPointFromString(in)
		if l != nil {
			t.Errorf("#%d: got l=%v, want nil", i, l)
		}
		if err == nil {
			t.Errorf("#%d: got err=nil, want non-nil", i)
		}
	}
}
