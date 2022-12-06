package unfocus

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2/ginkgo/command"
)

func BuildUnfocusCommand() command.Command {
	return command.Command{
		Name:     "unfocus",
		Usage:    "ginkgo unfocus",
		ShortDoc: "Recursively unfocus any focused tests under the current directory",
		DocLink:  "filtering-specs",
		Command: func(_ []string, _ []string) {
			unfocusSpecs()
		},
	}
}

func unfocusSpecs() {
	fmt.Println("Scanning for focus...")

	goFiles := make(chan string)
	go func() {
		unfocusDir(goFiles, ".")
		close(goFiles)
	}()

	const workers = 10
	wg := sync.WaitGroup{}
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		go func() {
			for path := range goFiles {
				unfocusFile(path)
			}
			wg.Done()
		}()
	}

	wg.Wait()
}

func unfocusDir(goFiles chan string, path string) {
	files, err := os.ReadDir(path)
	if err != nil {
		fmt.Println(err.Error())
		return
	}

	for _, f := range files {
		switch {
		case f.IsDir() && shouldProcessDir(f.Name()):
			unfocusDir(goFiles, filepath.Join(path, f.Name()))
		case !f.IsDir() && shouldProcessFile(f.Name()):
			goFiles <- filepath.Join(path, f.Name())
		}
	}
}

func shouldProcessDir(basename string) bool {
	return basename != "vendor" && !strings.HasPrefix(basename, ".")
}

func shouldProcessFile(basename string) bool {
	return strings.HasSuffix(basename, ".go")
}

func unfocusFile(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("error reading file '%s': %s\n", path, err.Error())
		return
	}

	ast, err := parser.ParseFile(token.NewFileSet(), path, bytes.NewReader(data), parser.ParseComments)
	if err != nil {
		fmt.Printf("error parsing file '%s': %s\n", path, err.Error())
		return
	}

	eliminations := scanForFocus(ast)
	if len(eliminations) == 0 {
		return
	}

	fmt.Printf("...updating %s\n", path)
	backup, err := writeBackup(path, data)
	if err != nil {
		fmt.Printf("error creating backup file: %s\n", err.Error())
		return
	}

	if err := updateFile(path, data, eliminations); err != nil {
		fmt.Printf("error writing file '%s': %s\n", path, err.Error())
		return
	}

	os.Remove(backup)
}

func writeBackup(path string, data []byte) (string, error) {
	t, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path))

	if err != nil {
		return "", fmt.Errorf("error creating temporary file: %w", err)
	}
	defer t.Close()

	if _, err := io.Copy(t, bytes.NewReader(data)); err != nil {
		return "", fmt.Errorf("error writing to temporary file: %w", err)
	}

	return t.Name(), nil
}

func updateFile(path string, data []byte, eliminations [][]int64) error {
	to, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("error opening file for writing '%s': %w\n", path, err)
	}
	defer to.Close()

	from := bytes.NewReader(data)
	var cursor int64
	for _, eliminationRange := range eliminations {
		positionToEliminate, lengthToEliminate := eliminationRange[0]-1, eliminationRange[1]
		if _, err := io.CopyN(to, from, positionToEliminate-cursor); err != nil {
			return fmt.Errorf("error copying data: %w", err)
		}

		cursor = positionToEliminate + lengthToEliminate

		if _, err := from.Seek(lengthToEliminate, io.SeekCurrent); err != nil {
			return fmt.Errorf("error seeking to position in buffer: %w", err)
		}
	}

	if _, err := io.Copy(to, from); err != nil {
		return fmt.Errorf("error copying end data: %w", err)
	}

	return nil
}

func scanForFocus(file *ast.File) (eliminations [][]int64) {
	ast.Inspect(file, func(n ast.Node) bool {
		if c, ok := n.(*ast.CallExpr); ok {
			if i, ok := c.Fun.(*ast.Ident); ok {
				if isFocus(i.Name) {
					eliminations = append(eliminations, []int64{int64(i.Pos()), 1})
				}
			}
		}

		if i, ok := n.(*ast.Ident); ok {
			if i.Name == "Focus" {
				eliminations = append(eliminations, []int64{int64(i.Pos()), 6})
			}
		}

		return true
	})

	return eliminations
}

func isFocus(name string) bool {
	switch name {
	case "FDescribe", "FContext", "FIt", "FDescribeTable", "FEntry", "FSpecify", "FWhen":
		return true
	default:
		return false
	}
}
