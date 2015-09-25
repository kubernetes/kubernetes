# Cobra

A Commander for modern go CLI interactions

[![Build Status](https://travis-ci.org/spf13/cobra.svg)](https://travis-ci.org/spf13/cobra)

## Overview

Cobra is a commander providing a simple interface to create powerful modern CLI
interfaces similar to git & go tools. In addition to providing an interface, Cobra
simultaneously provides a controller to organize your application code.

Inspired by go, go-Commander, gh and subcommand, Cobra improves on these by
providing **fully posix compliant flags** (including short & long versions),
**nesting commands**, and the ability to **define your own help and usage** for any or
all commands.

Cobra has an exceptionally clean interface and simple design without needless
constructors or initialization methods.

Applications built with Cobra commands are designed to be as user friendly as
possible. Flags can be placed before or after the command (as long as a
confusing space isn’t provided). Both short and long flags can be used. A
command need not even be fully typed. The shortest unambiguous string will
suffice. Help is automatically generated and available for the application or
for a specific command using either the help command or the --help flag.

## Concepts

Cobra is built on a structure of commands & flags.

**Commands** represent actions and **Flags** are modifiers for those actions.

In the following example 'server' is a command and 'port' is a flag.

    hugo server --port=1313

### Commands

Command is the central point of the application. Each interaction that
the application supports will be contained in a Command. A command can
have children commands and optionally run an action.

In the example above 'server' is the command

A Command has the following structure:

    type Command struct {
        Use string // The one-line usage message.
        Short string // The short description shown in the 'help' output.
        Long string // The long message shown in the 'help <this-command>' output.
        Run func(cmd *Command, args []string) // Run runs the command.
    }

### Flags

A Flag is a way to modify the behavior of an command. Cobra supports
fully posix compliant flags as well as the go flag package. 
A Cobra command can define flags that persist through to children commands
and flags that are only available to that command.

In the example above 'port' is the flag.

Flag functionality is provided by the [pflag
library](https://github.com/ogier/pflag), a fork of the flag standard library
which maintains the same interface while adding posix compliance.

## Usage

Cobra works by creating a set of commands and then organizing them into a tree.
The tree defines the structure of the application.

Once each command is defined with it's corresponding flags, then the
tree is assigned to the commander which is finally executed.

### Installing
Using Cobra is easy. First use go get to install the latest version
of the library.

    $ go get github.com/spf13/cobra

Next include cobra in your application.

    import "github.com/spf13/cobra"

### Create the root command

The root command represents your binary itself.

Cobra doesn't require any special constructors. Simply create your commands.

    var HugoCmd = &cobra.Command{
        Use:   "hugo",
        Short: "Hugo is a very fast static site generator",
        Long: `A Fast and Flexible Static Site Generator built with
                love by spf13 and friends in Go.
                Complete documentation is available at http://hugo.spf13.com`,
        Run: func(cmd *cobra.Command, args []string) {
            // Do Stuff Here
        },
    }

### Create additional commands

Additional commands can be defined.

    var versionCmd = &cobra.Command{
        Use:   "version",
        Short: "Print the version number of Hugo",
        Long:  `All software has versions. This is Hugo's`,
        Run: func(cmd *cobra.Command, args []string) {
            fmt.Println("Hugo Static Site Generator v0.9 -- HEAD")
        },
    }

### Attach command to its parent
In this example we are attaching it to the root, but commands can be attached at any level.

	HugoCmd.AddCommand(versionCmd)

### Assign flags to a command

Since the flags are defined and used in different locations, we need to
define a variable outside with the correct scope to assign the flag to
work with.

    var Verbose bool
    var Source string

There are two different approaches to assign a flag.

#### Persistent Flags

A flag can be 'persistent' meaning that this flag will be available to the
command it's assigned to as well as every command under that command. For
global flags assign a flag as a persistent flag on the root.

	HugoCmd.PersistentFlags().BoolVarP(&Verbose, "verbose", "v", false, "verbose output")

#### Local Flags

A flag can also be assigned locally which will only apply to that specific command.

	HugoCmd.Flags().StringVarP(&Source, "source", "s", "", "Source directory to read from")

### Remove a command from its parent

Removing a command is not a common action in simple programs but it allows 3rd parties to customize an existing command tree.

In this example, we remove the existing `VersionCmd` command of an existing root command, and we replace it by our own version.

	mainlib.RootCmd.RemoveCommand(mainlib.VersionCmd)
	mainlib.RootCmd.AddCommand(versionCmd)

### Once all commands and flags are defined, Execute the commands

Execute should be run on the root for clarity, though it can be called on any command.

    HugoCmd.Execute()

## Example

In the example below we have defined three commands. Two are at the top level
and one (cmdTimes) is a child of one of the top commands. In this case the root
is not executable meaning that a subcommand is required. This is accomplished
by not providing a 'Run' for the 'rootCmd'.

We have only defined one flag for a single command.

More documentation about flags is available at https://github.com/spf13/pflag

    import(
        "github.com/spf13/cobra"
        "fmt"
        "strings"
    )

    func main() {

        var echoTimes int

        var cmdPrint = &cobra.Command{
            Use:   "print [string to print]",
            Short: "Print anything to the screen",
            Long:  `print is for printing anything back to the screen.
            For many years people have printed back to the screen.
            `,
            Run: func(cmd *cobra.Command, args []string) {
                fmt.Println("Print: " + strings.Join(args, " "))
            },
        }

        var cmdEcho = &cobra.Command{
            Use:   "echo [string to echo]",
            Short: "Echo anything to the screen",
            Long:  `echo is for echoing anything back.
            Echo works a lot like print, except it has a child command.
            `,
            Run: func(cmd *cobra.Command, args []string) {
                fmt.Println("Print: " + strings.Join(args, " "))
            },
        }

        var cmdTimes = &cobra.Command{
            Use:   "times [# times] [string to echo]",
            Short: "Echo anything to the screen more times",
            Long:  `echo things multiple times back to the user by providing
            a count and a string.`,
            Run: func(cmd *cobra.Command, args []string) {
                for i:=0; i < echoTimes; i++ {
                    fmt.Println("Echo: " + strings.Join(args, " "))
                }
            },
        }

        cmdTimes.Flags().IntVarP(&echoTimes, "times", "t", 1, "times to echo the input")

        var rootCmd = &cobra.Command{Use: "app"}
        rootCmd.AddCommand(cmdPrint, cmdEcho)
        cmdEcho.AddCommand(cmdTimes)
        rootCmd.Execute()
    }

For a more complete example of a larger application, please checkout [Hugo](http://hugo.spf13.com)

## The Help Command

Cobra automatically adds a help command to your application when you have subcommands.
This will be called when a user runs 'app help'. Additionally help will also
support all other commands as input. Say for instance you have a command called
'create' without any additional configuration cobra will work when 'app help
create' is called.  Every command will automatically have the '--help' flag added.

### Example

The following output is automatically generated by cobra. Nothing beyond the
command and flag definitions are needed.

    > hugo help

    A Fast and Flexible Static Site Generator built with
    love by spf13 and friends in Go.

    Complete documentation is available at http://hugo.spf13.com

    Usage:
      hugo [flags]
      hugo [command]

    Available Commands:
      server          :: Hugo runs it's own a webserver to render the files
      version         :: Print the version number of Hugo
      check           :: Check content in the source directory
      benchmark       :: Benchmark hugo by building a site a number of times
      help [command]  :: Help about any command

     Available Flags:
      -b, --base-url="": hostname (and path) to the root eg. http://spf13.com/
      -D, --build-drafts=false: include content marked as draft
          --config="": config file (default is path/config.yaml|json|toml)
      -d, --destination="": filesystem path to write files to
      -s, --source="": filesystem path to read files relative from
          --stepAnalysis=false: display memory and timing of different steps of the program
          --uglyurls=false: if true, use /filename.html instead of /filename/
      -v, --verbose=false: verbose output
      -w, --watch=false: watch filesystem for changes and recreate as needed

    Use "hugo help [command]" for more information about that command.



Help is just a command like any other. There is no special logic or behavior
around it. In fact you can provide your own if you want.

### Defining your own help

You can provide your own Help command or you own template for the default command to use.

The default help command is 

    func (c *Command) initHelp() {
        if c.helpCommand == nil {
            c.helpCommand = &Command{
                Use:   "help [command]",
                Short: "Help about any command",
                Long: `Help provides help for any command in the application.
        Simply type ` + c.Name() + ` help [path to command] for full details.`,
                Run: c.HelpFunc(),
            }
        }
        c.AddCommand(c.helpCommand)
    }

You can provide your own command, function or template through the following methods.

    command.SetHelpCommand(cmd *Command)

    command.SetHelpFunc(f func(*Command, []string))

    command.SetHelpTemplate(s string)

The latter two will also apply to any children commands.

## Usage

When the user provides an invalid flag or invalid command Cobra responds by
showing the user the 'usage'

### Example
You may recognize this from the help above. That's because the default help
embeds the usage as part of it's output.

    Usage:
      hugo [flags]
      hugo [command]

    Available Commands:
      server          Hugo runs it's own a webserver to render the files
      version         Print the version number of Hugo
      check           Check content in the source directory
      benchmark       Benchmark hugo by building a site a number of times
      help [command]  Help about any command

     Available Flags:
      -b, --base-url="": hostname (and path) to the root eg. http://spf13.com/
      -D, --build-drafts=false: include content marked as draft
          --config="": config file (default is path/config.yaml|json|toml)
      -d, --destination="": filesystem path to write files to
      -s, --source="": filesystem path to read files relative from
          --stepAnalysis=false: display memory and timing of different steps of the program
          --uglyurls=false: if true, use /filename.html instead of /filename/
      -v, --verbose=false: verbose output
      -w, --watch=false: watch filesystem for changes and recreate as needed

### Defining your own usage
You can provide your own usage function or template for cobra to use.

The default usage function is

		return func(c *Command) error {
			err := tmpl(c.Out(), c.UsageTemplate(), c)
			return err
		}

Like help the function and template are over ridable through public methods.

    command.SetUsageFunc(f func(*Command) error)

    command.SetUsageTemplate(s string)

## PreRun or PostRun Hooks

It is possible to run functions before or after the main `Run` function of your command. The `PersistentPreRun` and `PreRun` functions will be executed before `Run`. `PersistendPostRun` and `PostRun` will be executed after `Run`.  The `Persistent*Run` functions will be inherrited by children if they do not declare their own.  These function are run in the following order:

- `PersistentPreRun`
- `PreRun`
- `Run`
- `PostRun`
- `PersistenPostRun`

And example of two commands which use all of these features is below.  When the subcommand in executed it will run the root command's `PersistentPreRun` but not the root command's `PersistentPostRun`

```go
package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

func main() {

	var rootCmd = &cobra.Command{
		Use:   "root [sub]",
		Short: "My root command",
		PersistentPreRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside rootCmd PersistentPreRun with args: %v\n", args)
		},
		PreRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside rootCmd PreRun with args: %v\n", args)
		},
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside rootCmd Run with args: %v\n", args)
		},
		PostRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside rootCmd PostRun with args: %v\n", args)
		},
		PersistentPostRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside rootCmd PersistentPostRun with args: %v\n", args)
		},
	}

	var subCmd = &cobra.Command{
		Use:   "sub [no options!]",
		Short: "My sub command",
		PreRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside subCmd PreRun with args: %v\n", args)
		},
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside subCmd Run with args: %v\n", args)
		},
		PostRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside subCmd PostRun with args: %v\n", args)
		},
		PersistentPostRun: func(cmd *cobra.Command, args []string) {
			fmt.Printf("Inside subCmd PersistentPostRun with args: %v\n", args)
		},
	}

	rootCmd.AddCommand(subCmd)

	rootCmd.SetArgs([]string{""})
	_ = rootCmd.Execute()
	fmt.Print("\n")
	rootCmd.SetArgs([]string{"sub", "arg1", "arg2"})
	_ = rootCmd.Execute()
}
```

## Generating markdown formatted documentation for your command

Cobra can generate a markdown formatted document based on the subcommands, flags, etc. A simple example of how to do this for your command can be found in [Markdown Docs](md_docs.md)

## Generating man pages for your command

Cobra can generate a man page based on the subcommands, flags, etc. A simple example of how to do this for your command can be found in [Man Docs](man_docs.md)

## Generating bash completions for your command

Cobra can generate a bash completions file. If you add more information to your command these completions can be amazingly powerful and flexible.  Read more about [Bash Completions](bash_completions.md)

## Debugging

Cobra provides a ‘DebugFlags’ method on a command which when called will print
out everything Cobra knows about the flags for each command

### Example

    command.DebugFlags()

## Release Notes
* **0.9.0** June 17, 2014
  * flags can appears anywhere in the args (provided they are unambiguous)
  * --help prints usage screen for app or command
  * Prefix matching for commands
  * Cleaner looking help and usage output
  * Extensive test suite
* **0.8.0** Nov 5, 2013
  * Reworked interface to remove commander completely
  * Command now primary structure
  * No initialization needed
  * Usage & Help templates & functions definable at any level
  * Updated Readme
* **0.7.0** Sept 24, 2013
  * Needs more eyes
  * Test suite
  * Support for automatic error messages
  * Support for help command
  * Support for printing to any io.Writer instead of os.Stderr
  * Support for persistent flags which cascade down tree
  * Ready for integration into Hugo
* **0.1.0** Sept 3, 2013
  * Implement first draft

## ToDo
* Launch proper documentation site

## Contributing

1. Fork it
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

## Contributors

Names in no particular order:

* [spf13](https://github.com/spf13)

## License

Cobra is released under the Apache 2.0 license. See [LICENSE.txt](https://github.com/spf13/cobra/blob/master/LICENSE.txt)


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/spf13/cobra/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

