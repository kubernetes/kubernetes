package outline

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"

	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/types"
)

const (
	// indentWidth is the width used by the 'indent' output
	indentWidth = 4
	// stdinAlias is a portable alias for stdin. This convention is used in
	// other CLIs, e.g., kubectl.
	stdinAlias   = "-"
	usageCommand = "ginkgo outline <filename>"
)

type outlineConfig struct {
	Format string
}

func BuildOutlineCommand() command.Command {
	conf := outlineConfig{
		Format: "csv",
	}
	flags, err := types.NewGinkgoFlagSet(
		types.GinkgoFlags{
			{Name: "format", KeyPath: "Format",
				Usage:             "Format of outline",
				UsageArgument:     "one of 'csv', 'indent', or 'json'",
				UsageDefaultValue: conf.Format,
			},
		},
		&conf,
		types.GinkgoFlagSections{},
	)
	if err != nil {
		panic(err)
	}

	return command.Command{
		Name:          "outline",
		Usage:         "ginkgo outline <filename>",
		ShortDoc:      "Create an outline of Ginkgo symbols for a file",
		Documentation: "To read from stdin, use: `ginkgo outline -`",
		DocLink:       "creating-an-outline-of-specs",
		Flags:         flags,
		Command: func(args []string, _ []string) {
			outlineFile(args, conf.Format)
		},
	}
}

func outlineFile(args []string, format string) {
	if len(args) != 1 {
		command.AbortWithUsage("outline expects exactly one argument")
	}

	filename := args[0]
	var src *os.File
	if filename == stdinAlias {
		src = os.Stdin
	} else {
		var err error
		src, err = os.Open(filename)
		command.AbortIfError("Failed to open file:", err)
	}

	fset := token.NewFileSet()

	parsedSrc, err := parser.ParseFile(fset, filename, src, 0)
	command.AbortIfError("Failed to parse source:", err)

	o, err := FromASTFile(fset, parsedSrc)
	command.AbortIfError("Failed to create outline:", err)

	var oerr error
	switch format {
	case "csv":
		_, oerr = fmt.Print(o)
	case "indent":
		_, oerr = fmt.Print(o.StringIndent(indentWidth))
	case "json":
		b, err := json.Marshal(o)
		if err != nil {
			println(fmt.Sprintf("error marshalling to json: %s", err))
		}
		_, oerr = fmt.Println(string(b))
	default:
		command.AbortWith("Format %s not accepted", format)
	}
	command.AbortIfError("Failed to write outline:", oerr)
}
