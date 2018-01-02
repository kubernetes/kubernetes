package sftp

import (
	"errors"
	"io"
	"os"
	"reflect"
	"testing"

	"github.com/kr/fs"
)

// assert that *Client implements fs.FileSystem
var _ fs.FileSystem = new(Client)

// assert that *File implements io.ReadWriteCloser
var _ io.ReadWriteCloser = new(File)

func TestNormaliseError(t *testing.T) {
	var (
		ok         = &StatusError{Code: ssh_FX_OK}
		eof        = &StatusError{Code: ssh_FX_EOF}
		fail       = &StatusError{Code: ssh_FX_FAILURE}
		noSuchFile = &StatusError{Code: ssh_FX_NO_SUCH_FILE}
		foo        = errors.New("foo")
	)

	var tests = []struct {
		desc string
		err  error
		want error
	}{
		{
			desc: "nil error",
		},
		{
			desc: "not *StatusError",
			err:  foo,
			want: foo,
		},
		{
			desc: "*StatusError with ssh_FX_EOF",
			err:  eof,
			want: io.EOF,
		},
		{
			desc: "*StatusError with ssh_FX_NO_SUCH_FILE",
			err:  noSuchFile,
			want: os.ErrNotExist,
		},
		{
			desc: "*StatusError with ssh_FX_OK",
			err:  ok,
		},
		{
			desc: "*StatusError with ssh_FX_FAILURE",
			err:  fail,
			want: fail,
		},
	}

	for _, tt := range tests {
		got := normaliseError(tt.err)
		if got != tt.want {
			t.Errorf("normaliseError(%#v), test %q\n- want: %#v\n-  got: %#v",
				tt.err, tt.desc, tt.want, got)
		}
	}
}

var flagsTests = []struct {
	flags int
	want  uint32
}{
	{os.O_RDONLY, ssh_FXF_READ},
	{os.O_WRONLY, ssh_FXF_WRITE},
	{os.O_RDWR, ssh_FXF_READ | ssh_FXF_WRITE},
	{os.O_RDWR | os.O_CREATE | os.O_TRUNC, ssh_FXF_READ | ssh_FXF_WRITE | ssh_FXF_CREAT | ssh_FXF_TRUNC},
	{os.O_WRONLY | os.O_APPEND, ssh_FXF_WRITE | ssh_FXF_APPEND},
}

func TestFlags(t *testing.T) {
	for i, tt := range flagsTests {
		got := flags(tt.flags)
		if got != tt.want {
			t.Errorf("test %v: flags(%x): want: %x, got: %x", i, tt.flags, tt.want, got)
		}
	}
}

func TestUnmarshalStatus(t *testing.T) {
	requestID := uint32(1)

	id := marshalUint32([]byte{}, requestID)
	idCode := marshalUint32(id, ssh_FX_FAILURE)
	idCodeMsg := marshalString(idCode, "err msg")
	idCodeMsgLang := marshalString(idCodeMsg, "lang tag")

	var tests = []struct {
		desc   string
		reqID  uint32
		status []byte
		want   error
	}{
		{
			desc:   "well-formed status",
			reqID:  1,
			status: idCodeMsgLang,
			want: &StatusError{
				Code: ssh_FX_FAILURE,
				msg:  "err msg",
				lang: "lang tag",
			},
		},
		{
			desc:   "missing error message and language tag",
			reqID:  1,
			status: idCode,
			want:   errShortPacket,
		},
		{
			desc:   "missing language tag",
			reqID:  1,
			status: idCodeMsg,
			want: &StatusError{
				Code: ssh_FX_FAILURE,
				msg:  "err msg",
			},
		},
		{
			desc:   "request identifier mismatch",
			reqID:  2,
			status: idCodeMsgLang,
			want:   &unexpectedIDErr{2, requestID},
		},
	}

	for _, tt := range tests {
		got := unmarshalStatus(tt.reqID, tt.status)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("unmarshalStatus(%v, %v), test %q\n- want: %#v\n-  got: %#v",
				requestID, tt.status, tt.desc, tt.want, got)
		}
	}
}
