/*
Package color is an ANSI color package to output colorized or SGR defined
output to the standard output. The API can be used in several way, pick one
that suits you.

Use simple and default helper functions with predefined foreground colors:

    color.Cyan("Prints text in cyan.")

    // a newline will be appended automatically
    color.Blue("Prints %s in blue.", "text")

    // More default foreground colors..
    color.Red("We have red")
    color.Yellow("Yellow color too!")
    color.Magenta("And many others ..")

    // Hi-intensity colors
    color.HiGreen("Bright green color.")
    color.HiBlack("Bright black means gray..")
    color.HiWhite("Shiny white color!")

However there are times where custom color mixes are required. Below are some
examples to create custom color objects and use the print functions of each
separate color object.

    // Create a new color object
    c := color.New(color.FgCyan).Add(color.Underline)
    c.Println("Prints cyan text with an underline.")

    // Or just add them to New()
    d := color.New(color.FgCyan, color.Bold)
    d.Printf("This prints bold cyan %s\n", "too!.")


    // Mix up foreground and background colors, create new mixes!
    red := color.New(color.FgRed)

    boldRed := red.Add(color.Bold)
    boldRed.Println("This will print text in bold red.")

    whiteBackground := red.Add(color.BgWhite)
    whiteBackground.Println("Red text with White background.")

    // Use your own io.Writer output
    color.New(color.FgBlue).Fprintln(myWriter, "blue color!")

    blue := color.New(color.FgBlue)
    blue.Fprint(myWriter, "This will print text in blue.")

You can create PrintXxx functions to simplify even more:

    // Create a custom print function for convenient
    red := color.New(color.FgRed).PrintfFunc()
    red("warning")
    red("error: %s", err)

    // Mix up multiple attributes
    notice := color.New(color.Bold, color.FgGreen).PrintlnFunc()
    notice("don't forget this...")

You can also FprintXxx functions to pass your own io.Writer:

    blue := color.New(FgBlue).FprintfFunc()
    blue(myWriter, "important notice: %s", stars)

    // Mix up with multiple attributes
    success := color.New(color.Bold, color.FgGreen).FprintlnFunc()
    success(myWriter, don't forget this...")


Or create SprintXxx functions to mix strings with other non-colorized strings:

    yellow := New(FgYellow).SprintFunc()
    red := New(FgRed).SprintFunc()

    fmt.Printf("this is a %s and this is %s.\n", yellow("warning"), red("error"))

    info := New(FgWhite, BgGreen).SprintFunc()
    fmt.Printf("this %s rocks!\n", info("package"))

Windows support is enabled by default. All Print functions work as intended.
However only for color.SprintXXX functions, user should use fmt.FprintXXX and
set the output to color.Output:

    fmt.Fprintf(color.Output, "Windows support: %s", color.GreenString("PASS"))

    info := New(FgWhite, BgGreen).SprintFunc()
    fmt.Fprintf(color.Output, "this %s rocks!\n", info("package"))

Using with existing code is possible. Just use the Set() method to set the
standard output to the given parameters. That way a rewrite of an existing
code is not required.

    // Use handy standard colors.
    color.Set(color.FgYellow)

    fmt.Println("Existing text will be now in Yellow")
    fmt.Printf("This one %s\n", "too")

    color.Unset() // don't forget to unset

    // You can mix up parameters
    color.Set(color.FgMagenta, color.Bold)
    defer color.Unset() // use it in your function

    fmt.Println("All text will be now bold magenta.")

There might be a case where you want to disable color output (for example to
pipe the standard output of your app to somewhere else). `Color` has support to
disable colors both globally and for single color definition. For example
suppose you have a CLI app and a `--no-color` bool flag. You can easily disable
the color output with:

    var flagNoColor = flag.Bool("no-color", false, "Disable color output")

    if *flagNoColor {
    	color.NoColor = true // disables colorized output
    }

It also has support for single color definitions (local). You can
disable/enable color output on the fly:

     c := color.New(color.FgCyan)
     c.Println("Prints cyan text")

     c.DisableColor()
     c.Println("This is printed without any color")

     c.EnableColor()
     c.Println("This prints again cyan...")
*/
package color
