# Profile Builder

> Note: This tool is related to the Azure-SDK-for-Go, but not truly part of it. As such, the SemVers associated with this repository do not extend to the packages associated with `profileBuilder`.

## Overview

Azure Profiles offer a means of virtualizing the API Versions of services that should be targeted by an application or SDK.
This concept was introduced for [Azure Stack](https://azure.microsoft.com/overview/azure-stack), where the environment in
which applications will be executed is less consistent than when targeting the public cloud. However, its usefulness as a
means of easily snapping to versions of a service is broadly applicapable. Using profiles, it is easy to use a single version
of models and operations throughout an application, or a means of locking to versions of services that have been tested and
are guaranteed to work together.

[Type aliases were introduced in Go 1.9](https://golang.org/doc/go1.9#language), effectively allowing for multiple symbols
to be mapped to a single type. The impact of this for our support of profiles is tremendous. It allows for seamless
interoperability between packages using different profiles, but where those profiles still target the same API Version of a
service. Without type aliases, we would have been forced to generate code in a way that required some ugly casts to be
scattered throughout the consumer's code.

## Installation

> *Note:* These installation notes assume that you have [Go 1.9](https://blog.golang.org/go1.9) or higher, and [Git](https://git-scm.com/) installed.

The simplest version of installation is very easy but not stable, just run the following command:

``` bash
go get -u github.com/Azure/azure-sdk-for-go/tools/profileBuilder
```

If that causes you trouble, run the following commands:

``` bash
# bash
go get -d github.com/Azure/azure-sdk-for-go/tools/profileBuilder
cd $GOPATH/src/github.com/Azure/azure-sdk-for-go/tools/profileBuilder
go install
```

``` PowerShell
# PowerShell
go get -d github.com/Azure/azure-sdk-for-go/tools/profileBuilder
cd $env:GOPATH\src\github.com\Azure\azure-sdk-for-go\tools\profileBuilder
go install
```

## Usage
### Basics
Once installed, running `profileBuilder` should be straight-forward. Each sub-command is a different strategy for finding the packages to include in the profile. The most flexible and broadly applicable sub-command is `list`.

For the first example, we'll use the `list` sub-command without any commands. It will read from `stdin`, looking for line delimited Go package names.

``` bash
$> profileBuilder list
github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2016-06-01/logic
github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis
```

Because we didn't specify a name for this profile, a random one will be generated. Assuming that it chooses the name `YellowIceberg84`, the files that would be produced would be in the structure:

```
$GOPATH
    /src
    |    /github.com
    |    |    /Azure
    |    |    |   /azure-sdk-for-go
    |    |    |    |    /profiles
    |    |    |    |    |    /YellowIceberg84
    |    |    |    |    |    |    /logic
    |    |    |    |    |    |    |    /mgmt
    |    |    |    |    |    |    |    |    /logic
    |    |    |    |    |    |    |    |    |    models.go
    |    |    |    |    |    |    /redis
    |    |    |    |    |    |    |    /mgmt
    |    |    |    |    |    |    |    |    /redis
    |    |    |    |    |    |    |    |    |    models.go
                            
```

Each of the files named `models.go` is composed of type definitions which will either duplicate or delegate all calls back to the original package's definition.

Clearly, typing each package name on demand, as profiles needs to be generated is error-prone and inconvenient. For that reason, using the piping operator to read the contents of a file into `stdin` is a much better idea. Using the file:


```
<myProfileDefinition.txt>
github.com/Azure/azure-sdk-for-go/services/logic/mgmt/2016-06-01/logic
github.com/Azure/azure-sdk-for-go/services/redis/mgmt/2017-02-01/redis
```

Will allow for the command:

``` bash
$> cat myProfileDefinition.txt | profileBuilder list
```

This command then yields the same results as the first example.

### Latest Command

The `latest` command reflects on the packages in the `services` directory of the Azure-SDK-for-Go, and picks the most up-to-date API Versions for inclusion in a profile. Optionally, it will include API Versions that are labeled as "preview".

### Arguments

#### Input

|              |              |
|--------------|--------------|
| Long Form    | --input      |
| Short Form   | -i           |
| Default      | \<stdin>     |
| Sub-Commands | list         |

When using the `list` sub-command, instead of reading from stdin, read from the file specified .
#### Help

|              |              |
|--------------|--------------|
| Long Form    | --help       |
| Short Form   | -h           |
| Default      | false        |
| Sub-Commands | list, latest |

The behavior of `profileBuilder` can be confured by passing in command-line arguments as flags. If you have any doubt or question about how a command operates, pass `--help` for `profileBuilder` to get a brief description of the command you're using, and all of the arguments it accepts.

#### Name

|              |                       |
|--------------|-----------------------|
| Long Form    | --name                |
| Short Form   | -n                    |
| Default      | \<randomly generated> |
| Sub-Commands | list, latest          |

You can opt-to not have `profileBuilder` use a randomly generated name for your profile by passing this argument.

#### Output Location

|              |                                                        |
|--------------|--------------------------------------------------------|
| Long Form    | --output-location                                      |
| Short Form   | -n                                                     |
| Default      | $GOPATH/src/github.com/Azure/azure-sdk-for-go/profiles |
| Sub-Commands | list, latest                                           |

The directory that profileBuilder should use to write the profile that is created.

#### Preview

|              |                 |
|--------------|-----------------|
| Long Form    | --preview       |
| Short Form   | -p              |
| Default      | false           |
| Sub-Commands | latest          |

While the `latest` command is iterating over the known Azure-SDK-for-Go packages, it needs to decide whether or not to disclude versions it deems "preview" versions. The `latest` command relies of the suffix "-preview" at the end of the API Version name to make this determination.

#### Verbose

|              |                 |
|--------------|-----------------|
| Long Form    | --verbose       |
| Short Form   | -v              |
| Default      | false           |
| Sub-Commands | list, latest    |

If you're looking for more information about the intermediate status of `profileBuilder`, this flag is for you. It may be most useful if you're not seeing the API Version you expected in your profile.

## Go Generate

The `go generate` command cat take the place of `make` in some circumstances. The big benefit of using it is that it ships with `go`, and is more portable than `make`. To use it, one simply adds a comment into a Go source file that invokes an arbitrary command. When combined with the `profileBuilder`, this can be a powerful combination.