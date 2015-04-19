#!/bin/zsh

# Borrowed from grunt-cli
# http://gruntjs.com/
#
# Copyright (c) 2012 Tyler Kellen, contributors
# Licensed under the MIT license.
# https://github.com/gruntjs/grunt/blob/master/LICENSE-MIT

# Usage:
#
# To enable zsh <tab> completion for gulp, add the following line (minus the
# leading #, which is the zsh comment character) to your ~/.zshrc file:
#
# eval "$(gulp --completion=zsh)"

# Enable zsh autocompletion.
function _gulp_completion() {
  # Grab tasks
  compls=$(gulp --tasks-simple)
  completions=(${=compls})
  compadd -- $completions
}

compdef _gulp_completion gulp
