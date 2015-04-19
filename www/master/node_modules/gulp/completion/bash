#!/bin/bash

# Borrowed from grunt-cli
# http://gruntjs.com/
#
# Copyright (c) 2012 Tyler Kellen, contributors
# Licensed under the MIT license.
# https://github.com/gruntjs/grunt/blob/master/LICENSE-MIT

# Usage:
#
# To enable bash <tab> completion for gulp, add the following line (minus the
# leading #, which is the bash comment character) to your ~/.bashrc file:
#
# eval "$(gulp --completion=bash)"

# Enable bash autocompletion.
function _gulp_completions() {
  # The currently-being-completed word.
  local cur="${COMP_WORDS[COMP_CWORD]}"
  #Grab tasks
  local compls=$(gulp --tasks-simple)
  # Tell complete what stuff to show.
  COMPREPLY=($(compgen -W "$compls" -- "$cur"))
}

complete -o default -F _gulp_completions gulp
