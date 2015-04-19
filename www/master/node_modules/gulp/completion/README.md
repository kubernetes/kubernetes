# Completion for gulp
> Thanks to grunt team and Tyler Kellen

To enable tasks auto-completion in shell you should add `eval "$(gulp --completion=shell)"` in your `.shellrc` file.

## Bash

Add `eval "$(gulp --completion=bash)"` to `~/.bashrc`.

## Zsh

Add `eval "$(gulp --completion=zsh)"` to `~/.zshrc`.

## Powershell

Add `Invoke-Expression ((gulp --completion=powershell) -join [System.Environment]::NewLine)` to `$PROFILE`.

## Fish

Add `gulp --completion=fish | source` to `~/.config/fish/config.fish`.
