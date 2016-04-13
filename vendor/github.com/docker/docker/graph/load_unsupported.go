// +build !linux,!windows

package graph

import (
	"fmt"
	"io"
)

func (s *TagStore) Load(inTar io.ReadCloser, outStream io.Writer) error {
	return fmt.Errorf("Load is not supported on this platform")
}
