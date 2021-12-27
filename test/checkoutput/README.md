# checkoutput tool

The checkoutput tool parses a text or Markdown that has a command and the expected output sequence, runs the command and checks the output of the command against the expected output. 

## Usage
```
<tool> <path/to/directory> -v X

If directory path is not specified then the current working directory is taken as the root.
-v specifies the klog logging level for verbosity, 1 and 2 will provide more information.
```

The tool has 2 major functions

1. Identify all directories with the main packages and corresponding Markdown files that has the Run/Verify blocks.
2. For each such directory that is identified:
2.1. Parse and capture the Run/Verify blocks
2.2. Run and verify the output of the blocks captured in the previous step

## Usage

The typical use case is a directory with the package main and a Markdown file that has the following syntax. In the below example the go will run the main package and the output of it will be checked against the execpted output. The current working directory will be set to the directory containing the Markdown file.  A [working example](https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/component-base/logs/example) is in staging.

### Metadata

The MetaData field is nominally optional, but will need to be specified to process out the timestamps and location based information. Each line in the block is either a regex or a keyword that will be applied to remove headers in the log. It needs to be specified in the following format.

`(?m)(?P<replacement-text><regex>)(.*)`

klog example:

`(?m)(?P<klog_header>^[IWEF][0-9]{4}\\s[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{6}\\s*[0-9]*\\s.*\\.go:[0-9]{2}])(.*)`

The replacement-text is printed in-lieu of the header and is not used in comparison. Keywords refer to commonly used regexes. 

Supported keywords: (see 'regex_header' in code for more info and up-to-date set of regexes)

1. ignore_klog_header
2. ignore_json_header

### Example

Run:
```console
go run .
```
Metadata:
```console
(?m)(?P<json_header>"ts":\d*.\d*\,"caller":"[0-9A-Za-z/.]*:\d*",?)(.*)
ignore_klog_header
ignore_json_header
```
Expected output:
```
I0605 22:03:07.224293 3228948 logger.go:58] Log using Infof, key: value
I0605 22:03:07.224378 3228948 logger.go:59] "Log using InfoS" key="value"
E0605 22:03:07.224393 3228948 logger.go:61] Log using Errorf, err: fail
E0605 22:03:07.224402 3228948 logger.go:62] "Log using ErrorS" err="fail"
I0605 22:03:07.224407 3228948 logger.go:64] Log message has been redacted. Log argument #0 contains: [secret-key]
```
## Run and Check output
The command is run , its output captured and compared with the expected output. If a mismatch is seen the tool flags it as a validation failure. Passes are not reported, but will be output using the verbose flag.

## Failure

Multiple Run/Verify blocks can exist and a failure is treated as an individual failure and will not affect subsequent Run/Verify blocks.

# Open Issues
## stdout and stderr
Currently we validate the combined output of stdout and stderr. We need to come up with a proper mechanism to validate outputs in stdout and stderr as this tool is intended to check logging.
