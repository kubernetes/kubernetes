package coloredwriter

import (
	"bytes"
	"io"
	"k8s.io/kubectl/pkg/coloredwriter/yh"
)

type ColoredYamlWriter struct {
	Delegate io.Writer
}

func (c ColoredYamlWriter) Write(p []byte) (int, error) {
	n := len(p)

	h, err := yh.Highlight(bytes.NewReader(p))
	if err != nil {
		return c.Delegate.Write(p)
	}

	if _, err := c.Delegate.Write([]byte(h)); err != nil {
		return c.Delegate.Write(p)
	}

	return n, nil
}
