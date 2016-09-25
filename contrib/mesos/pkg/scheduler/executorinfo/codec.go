/*
Copyright 2015 The Kubernetes Authors.

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

package executorinfo

import (
	"encoding/base64"
	"io"

	"bufio"

	"github.com/gogo/protobuf/proto"
	"github.com/mesos/mesos-go/mesosproto"
)

var base64Codec = base64.StdEncoding

// EncodeResources encodes the given resource slice to the given writer.
// The resource slice is encoded as a comma separated string of
// base64 encoded resource protobufs.
func EncodeResources(w io.Writer, rs []*mesosproto.Resource) error {
	sep := ""

	for _, r := range rs {
		_, err := io.WriteString(w, sep)
		if err != nil {
			return err
		}

		buf, err := proto.Marshal(r)
		if err != nil {
			return err
		}

		encoded := base64Codec.EncodeToString(buf)
		_, err = io.WriteString(w, encoded)
		if err != nil {
			return err
		}

		sep = ","
	}

	return nil
}

// DecodeResources decodes a resource slice from the given reader.
// The format is expected to be the same as in EncodeResources.
func DecodeResources(r io.Reader) (rs []*mesosproto.Resource, err error) {
	delimited := bufio.NewReader(r)
	rs = []*mesosproto.Resource{}

	for err != io.EOF {
		var encoded string
		encoded, err = delimited.ReadString(',')

		switch {
		case err == io.EOF:
		case err == nil:
			encoded = encoded[:len(encoded)-1]
		default: // err != nil && err != io.EOF
			return nil, err
		}

		decoded, err := base64Codec.DecodeString(encoded)
		if err != nil {
			return nil, err
		}

		r := mesosproto.Resource{}
		if err := proto.Unmarshal(decoded, &r); err != nil {
			return nil, err
		}

		rs = append(rs, &r)
	}

	return rs, nil
}
