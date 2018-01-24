package githttp_test

import (
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/AaronO/go-git-http"
)

func TestRpcReader(t *testing.T) {
	tests := []struct {
		rpc  string
		file string

		want []githttp.Event
	}{
		{
			rpc:  "receive-pack",
			file: "receive-pack.0",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.PUSH),
					Commit:  (string)("92eef6dcb9cc198bc3ac6010c108fa482773f116"),
					Dir:     (string)(""),
					Tag:     (string)(""),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)("master"),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},

		// A tag using letters only.
		{
			rpc:  "receive-pack",
			file: "receive-pack.1",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.TAG),
					Commit:  (string)("3da295397738f395c2ca5fd5570f01a9fcea3be3"),
					Dir:     (string)(""),
					Tag:     (string)("sometextualtag"),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},

		// A tag containing the string "00".
		{
			rpc:  "receive-pack",
			file: "receive-pack.2",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.TAG),
					Commit:  (string)("3da295397738f395c2ca5fd5570f01a9fcea3be3"),
					Dir:     (string)(""),
					Tag:     (string)("1.000.1"),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},

		// Multiple tags containing string "00" in one git push operation.
		{
			rpc:  "receive-pack",
			file: "receive-pack.3",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.TAG),
					Commit:  (string)("3da295397738f395c2ca5fd5570f01a9fcea3be3"),
					Dir:     (string)(""),
					Tag:     (string)("1.000.2"),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.TAG),
					Commit:  (string)("3da295397738f395c2ca5fd5570f01a9fcea3be3"),
					Dir:     (string)(""),
					Tag:     (string)("1.000.3"),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.TAG),
					Commit:  (string)("3da295397738f395c2ca5fd5570f01a9fcea3be3"),
					Dir:     (string)(""),
					Tag:     (string)("1.000.4"),
					Last:    (string)("0000000000000000000000000000000000000000"),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},

		{
			rpc:  "upload-pack",
			file: "upload-pack.0",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.FETCH),
					Commit:  (string)("a647ec2ea40ee9ca35d32232dc28de22b1537e00"),
					Dir:     (string)(""),
					Tag:     (string)(""),
					Last:    (string)(""),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},

		{
			rpc:  "upload-pack",
			file: "upload-pack.1",

			want: []githttp.Event{
				(githttp.Event)(githttp.Event{
					Type:    (githttp.EventType)(githttp.FETCH),
					Commit:  (string)("92eef6dcb9cc198bc3ac6010c108fa482773f116"),
					Dir:     (string)(""),
					Tag:     (string)(""),
					Last:    (string)(""),
					Branch:  (string)(""),
					Error:   (error)(nil),
					Request: (*http.Request)(nil),
				}),
			},
		},
	}

	for _, tt := range tests {
		f, err := os.Open(filepath.Join("testdata", tt.file))
		if err != nil {
			t.Fatal(err)
		}

		r := fragmentedReader{f}

		rr := &githttp.RpcReader{
			Reader: r,
			Rpc:    tt.rpc,
		}

		_, err = io.Copy(ioutil.Discard, rr)
		if err != nil {
			t.Errorf("io.Copy: %v", err)
		}

		f.Close()

		if got := rr.Events; !reflect.DeepEqual(got, tt.want) {
			t.Errorf("test %q/%q:\n got: %#v\nwant: %#v\n", tt.rpc, tt.file, got, tt.want)
		}
	}
}

// fragmentedReader reads from R, with each Read call returning at most fragmentLen bytes even
// if len(p) is greater than fragmentLen.
// It purposefully adds a layer of inefficiency around R, and exists for testing purposes only.
type fragmentedReader struct {
	R io.Reader // Underlying reader.
}

func (r fragmentedReader) Read(p []byte) (n int, err error) {
	const fragmentLen = 1
	if len(p) <= fragmentLen {
		return r.R.Read(p)
	}
	return r.R.Read(p[:fragmentLen])
}
