package printers

import (
	"io"

	"k8s.io/apimachinery/pkg/runtime"
)

func NewDiscardingPrinter() ResourcePrinterFunc {
	return ResourcePrinterFunc(func(runtime.Object, io.Writer) error {
		return nil
	})
}
