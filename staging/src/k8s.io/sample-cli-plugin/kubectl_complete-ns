#!/usr/bin/env bash

# If we are completing a flag, use Cobra's builtin completion system.
# To know if we are completing a flag we need the last argument starts with a `-` and does not contain an `=`
args=("$@")
lastArg=${args[((${#args[@]}-1))]}
if [[ "$lastArg" == -* ]]; then
   if [[ "$lastArg" != *=* ]]; then
      kubectl ns __complete "$@"
   fi
else
   # TODO Make sure we are not completing the value of a flag.
   # TODO Only complete a single argument.
   # Both are pretty hard to do in a shell script.  The better way to do this would be to let
   # Cobra do all the completions by using `cobra.ValidArgsFunction` in the program.
   # But the below, although imperfect, is a nice example for plugins that don't use Cobra.

   # We are probably completing an argument.  This plugin only accepts namespaces, let's fetch them.
   kubectl get namespaces --output go-template='{{ range .items }}{{ .metadata.name }}{{"\n"}}{{ end }}'

   # Turn off file completion.  See the ShellCompDirective documentation within
   # https://github.com/spf13/cobra/blob/main/shell_completions.md#completion-of-nouns
   echo :4
fi
