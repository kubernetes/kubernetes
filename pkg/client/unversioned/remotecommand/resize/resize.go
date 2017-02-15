package resize

import (
	"encoding/json"
	"io"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/term"
)

// GetResizeFunc will return function that handles terminal resize
func GetResizeFunc(resizeQueue term.TerminalSizeQueue) func(io.Writer) {
	return func(stream io.Writer) {
		defer runtime.HandleCrash()

		encoder := json.NewEncoder(stream)
		for {
			size := resizeQueue.Next()
			if size == nil {
				return
			}
			if err := encoder.Encode(&size); err != nil {
				runtime.HandleError(err)
			}
		}
	}
}
