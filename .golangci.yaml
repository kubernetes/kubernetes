run:
  timeout: 30m
  skip-files:
    - "^zz_generated.*"

issues:
  max-same-issues: 0
  # Excluding configuration per-path, per-linter, per-text and per-source
  exclude-rules:
    # exclude ineffassing linter for generated files for conversion
    - path: conversion\.go
      linters:
        - ineffassign

linters:
  disable-all: true
  enable: # please keep this alphabetized
  # Don't use soon to deprecated[1] linters that lead to false
  # https://github.com/golangci/golangci-lint/issues/1841
  # - deadcode
  # - structcheck
  # - varcheck
    - ineffassign
    - staticcheck
    - unused

linters-settings: # please keep this alphabetized
  staticcheck:
    go: "1.17"
    checks: [
      "all",
      "-S1*",    # TODO(fix) Omit code simplifications for now.
      "-ST1*",   # Mostly stylistic, redundant w/ golint
      "-SA5011", # TODO(fix) Possible nil pointer dereference
      "-SA1019", # TODO(fix) Using a deprecated function, variable, constant or field
      "-SA2002"  # TODO(fix) Called testing.T.FailNow or SkipNow in a goroutine, which isn’t allowed
    ]
  unused:
    go: "1.17"
