package coloredwriter

import (
	"io"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubectl/pkg/coloredwriter/prettyjson"
)

type ColoredJsonWriter struct {
	Delegate io.Writer
}

func (c ColoredJsonWriter) Write(p []byte) (int, error) {
	n := len(p)

	var v interface{}

	// TODO:
	// - check if is 1 line json

	if err := json.Unmarshal(p, &v); err != nil {
		return c.Delegate.Write(p)
	}

	s, err := prettyjson.Marshal(v)
	if err != nil {
		return c.Delegate.Write(p)
	}

	_, err = c.Delegate.Write(s)
	if err != nil {
		return c.Delegate.Write(p)
	}

	return n, nil
}
