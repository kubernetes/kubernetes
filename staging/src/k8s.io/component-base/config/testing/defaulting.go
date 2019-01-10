package testing

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

func GetDefaultingTestCases(scheme *runtime.Scheme) []TestCase {
	cases := []TestCase{}
	for gvk := range scheme.AllKnownTypes() {
		fmt.Println(gvk)
		beforeDir := fmt.Sprintf("testdata/%s/before", gvk.Kind)
		afterDir := fmt.Sprintf("testdata/%s/after", gvk.Kind)
		utilruntime.Must(filepath.Walk(beforeDir, func(path string, info os.FileInfo, err error) error {
			fmt.Println(info.Name(), path)
			if err != nil {
				return err
			}
			if info.IsDir() {
				if info.Name() == "before" {
					return nil
				}
				return filepath.SkipDir
			}
			if !strings.HasSuffix(info.Name(), ".yaml") {
				return nil
			}
			cases = append(cases, TestCase{
				name:  fmt.Sprintf("default_%s", info.Name()),
				in:    filepath.Join(beforeDir, info.Name()),
				inGVK: gvk,
				out:   filepath.Join(afterDir, info.Name()),
				outGV: gvk.GroupVersion(),
			})
			return nil
		}))
	}
	return cases
}
