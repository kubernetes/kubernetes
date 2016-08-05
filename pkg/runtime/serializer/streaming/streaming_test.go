/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package streaming

import (
	"bytes"
	"io"
	"io/ioutil"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/framer"
)

type fakeDecoder struct {
	got []byte
	obj runtime.Object
	err error
}

func (d *fakeDecoder) Decode(data []byte, gvk *unversioned.GroupVersionKind, into runtime.Object) (runtime.Object, *unversioned.GroupVersionKind, error) {
	d.got = data
	return d.obj, nil, d.err
}

func TestEmptyDecoder(t *testing.T) {
	buf := bytes.NewBuffer([]byte{})
	d := &fakeDecoder{}
	_, _, err := NewDecoder(ioutil.NopCloser(buf), d).Decode(nil, nil)
	if err != io.EOF {
		t.Fatal(err)
	}
}

func TestDecoder(t *testing.T) {
	frames := [][]byte{
		make([]byte, 1025),
		make([]byte, 1024*5),
		make([]byte, 1024*1024*5),
		make([]byte, 1025),
	}
	pr, pw := io.Pipe()
	fw := framer.NewLengthDelimitedFrameWriter(pw)
	go func() {
		for i := range frames {
			fw.Write(frames[i])
		}
		pw.Close()
	}()

	r := framer.NewLengthDelimitedFrameReader(pr)
	d := &fakeDecoder{}
	dec := NewDecoder(r, d)
	if _, _, err := dec.Decode(nil, nil); err != nil || !bytes.Equal(d.got, frames[0]) {
		t.Fatalf("unexpected %v %v", err, len(d.got))
	}
	if _, _, err := dec.Decode(nil, nil); err != nil || !bytes.Equal(d.got, frames[1]) {
		t.Fatalf("unexpected %v %v", err, len(d.got))
	}
	if _, _, err := dec.Decode(nil, nil); err != ErrObjectTooLarge || !bytes.Equal(d.got, frames[1]) {
		t.Fatalf("unexpected %v %v", err, len(d.got))
	}
	if _, _, err := dec.Decode(nil, nil); err != nil || !bytes.Equal(d.got, frames[3]) {
		t.Fatalf("unexpected %v %v", err, len(d.got))
	}
	if _, _, err := dec.Decode(nil, nil); err != io.EOF {
		t.Fatalf("unexpected %v %v", err, len(d.got))
	}
}
