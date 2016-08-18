Hybrid configuration library
============================

Combine standard go flags with ini files.

Usage:

```bash

go get -u -a github.com/vharitonsky/iniflags
```

main.go
```go
package main

import (
	"flag"
	...
	"github.com/vharitonsky/iniflags"
	...
)

var (
	flag1 = flag.String("flag1", "default1", "Description1")
	...
	flagN = flag.Int("flagN", 123, "DescriptionN")
)

func main() {
	iniflags.Parse()  // use instead of flag.Parse()
}
```

dev.ini

```ini
    # comment1
    flag1 = "val1"  # comment2

    ...
    [section]
    flagN = 4  # comment3
```

```bash

go run main.go -config dev.ini -flagX=foobar

```

Now all unset flags obtain their value from .ini file provided in -config path.
If value is not found in the .ini, flag will retain its' default value.

Flag value priority:
  - value set via command-line
  - value from ini file
  - default value

Iniflags is compatible with real .ini config files with [sections] and #comments.
Sections and comments are skipped during config file parsing.

Iniflags can #import another ini files. For example,

base.ini
```ini
flag1 = value1
flag2 = value2
```

dev.ini
```ini
#import "base.ini"
# Now flag1="value1", flag2="value2"

flag2 = foobar
# Now flag1="value1", while flag2="foobar"
```

Both -config path and imported ini files can be addressed via http
or https links:

```bash
/path/to/app -config=https://google.com/path/to/config.ini
```

config.ini
```ini
# The following line will import configs from the given http link.
#import "http://google.com/path/to/config.ini"
```

All flags defined in the app can be dumped into stdout with ini-compatible sytax
by passing -dumpflags flag to the app. The following command creates ini-file 
with all the flags defined in the app:

```bash
/path/to/the/app -dumpflags > initial-config.ini
```


Iniflags also supports two types of online config reload:

  * Via SIGHUP signal:

```bash
kill -s SIGHUP <app_pid>
```

  * Via -configUpdateInterval flag. The following line will re-read config every 5 seconds:

```bash
/path/to/app -config=/path/to/config.ini -configUpdateInterval=5s
```


Advanced usage.

```go
package main

import (
	"flag"
	"iniflags"
	"log"
)

var listenPort = flag.Int("listenPort", 1234, "Port to listen to")

func init() {
	iniflags.OnFlagChange("listenPort", func() {
		startServerOnPort(*listenPort)
	})
}

func main() {
	// iniflags.Parse() starts the server on the -listenPort via OnFlagChange()
	// callback registered above.
	iniflags.Parse()
}
```
