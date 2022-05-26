# gomodguard
[![License](https://img.shields.io/github/license/ryancurrah/gomodguard?style=flat-square)](/LICENSE)
[![Codecov](https://img.shields.io/codecov/c/gh/ryancurrah/gomodguard?style=flat-square)](https://codecov.io/gh/ryancurrah/gomodguard)
[![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ryancurrah/gomodguard/Go?logo=Go&style=flat-square)](https://github.com/ryancurrah/gomodguard/actions?query=workflow%3AGo)
[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/ryancurrah/gomodguard?style=flat-square)](https://github.com/ryancurrah/gomodguard/releases/latest)
[![Docker](https://img.shields.io/docker/pulls/ryancurrah/gomodguard?style=flat-square)](https://hub.docker.com/r/ryancurrah/gomodguard)
[![Github Releases Stats of golangci-lint](https://img.shields.io/github/downloads/ryancurrah/gomodguard/total.svg?logo=github&style=flat-square)](https://somsubhra.com/github-release-stats/?username=ryancurrah&repository=gomodguard)

<img src="https://storage.googleapis.com/gopherizeme.appspot.com/gophers/9afcc208898c763be95f046eb2f6080146607209.png" width="30%">

Allow and block list linter for direct Go module dependencies. This is useful for organizations where they want to standardize on the modules used and be able to recommend alternative modules.

## Description

Allowed and blocked modules are defined in a `./.gomodguard.yaml` or `~/.gomodguard.yaml` file. 

Modules can be allowed by module or domain name. When allowed modules are specified any modules not in the allowed configuration are blocked.

If no allowed modules or domains are specified then all modules are allowed except for blocked ones.

The linter looks for blocked modules in `go.mod` and searches for imported packages where the imported packages module is blocked. Indirect modules are not considered.

Alternative modules can be optionally recommended in the blocked modules list.

If the linted module imports a blocked module but the linted module is in the recommended modules list the blocked module is ignored. Usually, this means the linted module wraps that blocked module for use by other modules, therefore the import of the blocked module should not be blocked.

Version constraints can be specified for modules as well which lets you block new or old versions of modules or specific versions.

Results are printed to `stdout`.

Logging statements are printed to `stderr`.

Results can be exported to different report formats. Which can be imported into CI tools. See the help section for more information.

## Configuration

```yaml
allowed:
  modules:                                                      # List of allowed modules
    - gopkg.in/yaml.v2
    - github.com/go-xmlfmt/xmlfmt
    - github.com/phayes/checkstyle
    - github.com/mitchellh/go-homedir
  domains:                                                      # List of allowed module domains
    - golang.org

blocked:
  modules:                                                      # List of blocked modules
    - github.com/uudashr/go-module:                             # Blocked module
        recommendations:                                        # Recommended modules that should be used instead (Optional)
          - golang.org/x/mod                           
        reason: "`mod` is the official go.mod parser library."  # Reason why the recommended module should be used (Optional)
  versions:                                                     # List of blocked module version constraints.
    - github.com/mitchellh/go-homedir:                          # Blocked module with version constraint.
        version: "<= 1.1.0"                                     # Version constraint, see https://github.com/Masterminds/semver#basic-comparisons.
        reason: "testing if blocked version constraint works."  # Reason why the version constraint exists.
```

## Usage

```
╰─ ./gomodguard -h
Usage: gomodguard <file> [files...]
Also supports package syntax but will use it in relative path, i.e. ./pkg/...
Flags:
  -f string
    	Report results to the specified file. A report type must also be specified
  -file string

  -h	Show this help text
  -help

  -i int
    	Exit code when issues were found (default 2)
  -issues-exit-code int 
      (default 2)
  
  -n	Don't lint test files
  -no-test

  -r string
    	Report results to one of the following formats: checkstyle. A report file destination must also be specified
  -report string
```

## Example

```
╰─ ./gomodguard -r checkstyle -f gomodguard-checkstyle.xml ./...

info: allowed modules, [gopkg.in/yaml.v2 github.com/go-xmlfmt/xmlfmt github.com/phayes/checkstyle github.com/mitchellh/go-homedir]
info: allowed module domains, [golang.org]
info: blocked modules, [github.com/uudashr/go-module]
info: found `2` blocked modules in the go.mod file, [github.com/gofrs/uuid github.com/uudashr/go-module]
blocked_example.go:6: import of package `github.com/gofrs/uuid` is blocked because the module is not in the allowed modules list.
blocked_example.go:7: import of package `github.com/uudashr/go-module` is blocked because the module is in the blocked modules list. `golang.org/x/mod` is a recommended module. `mod` is the official go.mod parser library.
```

Resulting checkstyle file

```
╰─ cat gomodguard-checkstyle.xml

<?xml version="1.0" encoding="UTF-8"?>
<checkstyle version="1.0.0">
  <file name="blocked_example.go">
    <error line="6" column="1" severity="error" message="import of package `github.com/gofrs/uuid` is blocked because the module is not in the allowed modules list." source="gomodguard">
    </error>
    <error line="7" column="1" severity="error" message="import of package `github.com/uudashr/go-module` is blocked because the module is in the blocked modules list. `golang.org/x/mod` is a recommended module. `mod` is the official go.mod parser library." source="gomodguard">
    </error>
  </file>
</checkstyle>
```

## Install

```
go get -u github.com/ryancurrah/gomodguard/cmd/gomodguard
```

## Develop

```
git clone https://github.com/ryancurrah/gomodguard.git && cd gomodguard

go build -o gomodguard cmd/gomodguard/main.go
```

## License

**MIT**
