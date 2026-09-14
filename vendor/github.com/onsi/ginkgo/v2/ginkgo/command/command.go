package command

import (
	"fmt"
	"io"
	"strings"

	"github.com/onsi/ginkgo/v2/formatter"
	"github.com/onsi/ginkgo/v2/types"
)

type Command struct {
	Name          string
	Flags         types.GinkgoFlagSet
	Usage         string
	ShortDoc      string
	Documentation string
	DocLink       string
	Command       func(args []string, additionalArgs []string)
}

func (c Command) Run(args []string, additionalArgs []string) {
	args, err := c.Flags.Parse(args)
	if err != nil {
		AbortWithUsage("%s", err.Error())
	}
	for _, arg := range args {
		if len(arg) > 1 && strings.HasPrefix(arg, "-") {
			AbortWith("%s", types.GinkgoErrors.FlagAfterPositionalParameter().Error())
		}
	}
	c.Command(args, additionalArgs)
}

func (c Command) EmitUsage(writer io.Writer) {
	fmt.Fprintln(writer, formatter.F("{{bold}}"+c.Usage+"{{/}}"))
	fmt.Fprintln(writer, formatter.F("{{gray}}%s{{/}}", strings.Repeat("-", len(c.Usage))))
	if c.ShortDoc != "" {
		fmt.Fprintln(writer, formatter.Fiw(0, formatter.COLS, c.ShortDoc))
		fmt.Fprintln(writer, "")
	}
	if c.Documentation != "" {
		fmt.Fprintln(writer, formatter.Fiw(0, formatter.COLS, c.Documentation))
		fmt.Fprintln(writer, "")
	}
	if c.DocLink != "" {
		fmt.Fprintln(writer, formatter.Fi(0, "{{bold}}Learn more at:{{/}} {{cyan}}{{underline}}http://onsi.github.io/ginkgo/#%s{{/}}", c.DocLink))
		fmt.Fprintln(writer, "")
	}
	flagUsage := c.Flags.Usage()
	if flagUsage != "" {
		fmt.Fprint(writer, formatter.F(flagUsage))
	}
}
