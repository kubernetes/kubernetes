package sftp

import (
	"bytes"
	"os"
	"reflect"
	"testing"
	"time"
)

// ensure that attrs implemenst os.FileInfo
var _ os.FileInfo = new(fileInfo)

var unmarshalAttrsTests = []struct {
	b    []byte
	want *fileInfo
	rest []byte
}{
	{marshal(nil, struct{ Flags uint32 }{}), &fileInfo{mtime: time.Unix(int64(0), 0)}, nil},
	{marshal(nil, struct {
		Flags uint32
		Size  uint64
	}{ssh_FILEXFER_ATTR_SIZE, 20}), &fileInfo{size: 20, mtime: time.Unix(int64(0), 0)}, nil},
	{marshal(nil, struct {
		Flags       uint32
		Size        uint64
		Permissions uint32
	}{ssh_FILEXFER_ATTR_SIZE | ssh_FILEXFER_ATTR_PERMISSIONS, 20, 0644}), &fileInfo{size: 20, mode: os.FileMode(0644), mtime: time.Unix(int64(0), 0)}, nil},
	{marshal(nil, struct {
		Flags                 uint32
		Size                  uint64
		UID, GID, Permissions uint32
	}{ssh_FILEXFER_ATTR_SIZE | ssh_FILEXFER_ATTR_UIDGID | ssh_FILEXFER_ATTR_UIDGID | ssh_FILEXFER_ATTR_PERMISSIONS, 20, 1000, 1000, 0644}), &fileInfo{size: 20, mode: os.FileMode(0644), mtime: time.Unix(int64(0), 0)}, nil},
}

func TestUnmarshalAttrs(t *testing.T) {
	for _, tt := range unmarshalAttrsTests {
		stat, rest := unmarshalAttrs(tt.b)
		got := fileInfoFromStat(stat, "")
		tt.want.sys = got.Sys()
		if !reflect.DeepEqual(got, tt.want) || !bytes.Equal(tt.rest, rest) {
			t.Errorf("unmarshalAttrs(%#v): want %#v, %#v, got: %#v, %#v", tt.b, tt.want, tt.rest, got, rest)
		}
	}
}
