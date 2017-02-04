# loreley

Easy and extensible colorizer for the programs' output.

Basically, loreley turns this:

```
{bold}{fg 15}{bg 27} hello {from "" 29} there {to 16 ""},
```

Into this:

![2016-06-27-13t53t45](https://raw.githubusercontent.com/reconquest/loreley/master/demo.png)

# Usage

```go
package main

import "fmt"
import "github.com/reconquest/loreley"

func main() {
	text, err := loreley.CompileAndExecuteToString(
		`{bold}{fg 15}{bg 27} hello {from "" 29} {.where} {to 16 ""}{reset}`,
		nil,
		map[string]interface{}{"where": "there"},
	)
	if err != nil {
		fmt.Errorf(`can't compile loreley template: %s`, err)
	}

	fmt.Println(text)
}
```

# Colors in `text/tabwriter`

Unfortunately, stdlib tabwriter does not implement proper column width
calculation if you use escape sequences in your data to highlight some
output.

So, probably, You will see something like this trying tabwriter:

![tabwriter-before](https://raw.githubusercontent.com/reconquest/loreley/master/tabwriter-before.png)

Using loreley you can achieve exactly what you're expecting to see:

![tabwriter-after](https://raw.githubusercontent.com/reconquest/loreley/master/tabwriter-after.png)

```go
package main

import (
	"bytes"
	"fmt"
	"strings"
	"text/tabwriter"

	"github.com/reconquest/loreley"
)

const ()

func main() {
	buffer := &bytes.Buffer{}

	writer := tabwriter.NewWriter(buffer, 2, 4, 2, ' ', tabwriter.FilterHTML)

	writer.Write([]byte(strings.Join(
		[]string{
			"<underline>CORES<reset>",
			"<underline>DESCRIPTION<reset>\n",
		}, "\t",
	)))

	writer.Write([]byte(strings.Join(
		[]string{
			"<fg 15><bg 1><bold> 1 <reset> <fg 15><bg 243><bold> 3 <reset>",
			"test\n",
		}, "\t",
	)))

	writer.Flush()

	loreley.DelimLeft = "<"
	loreley.DelimRight = ">"

	result, err := loreley.CompileAndExecuteToString(
		buffer.String(),
		nil,
		nil,
	)
	if err != nil {
		panic(err)
	}

	fmt.Print(result)
}
```

# Reference

loreley extends Go-lang template system. So, fully syntax is supported with
exception, that `{` and `}` will be used as delimiters.

All `<color>`, accepted by loreley, should be the 256-color code.

Available template functions:

* `{bg <color>}` sets background color for the next text;
* `{fg <color>}` sets foreground color for the next text;
* `{nobg}` resets background color to the default;
* `{nofg}` resets foreground color to the default;
* `{bold}` set bold mode on for the next text;
* `{nobold}` set bold mode to off;
* `{reverse}` set reverse mode on for the next text;
* `{noreverse}` set reverse mode off;
* `{underline}` set underline mode on for the next text;
* `{nounderline}` set underline mode off;
* `{reset}` resets all styles to default;
* `{from <text> <bg>}` reuse current fg as specified `<text>`'s bg color,
  specified `<bg>` will be used as fg color and as bg color for the following
  text;
* `{to <bg> <text>}` reuse current bg as specified `<text>`'s bg color,
  specified `<bg>` will be used as fg color for the following text;
