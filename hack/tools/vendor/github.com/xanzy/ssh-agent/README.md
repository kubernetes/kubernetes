# ssh-agent

Create a new [agent.Agent](https://godoc.org/golang.org/x/crypto/ssh/agent#Agent) on any type of OS (so including Windows) from any [Go](https://golang.org) application.

## Limitations

When compiled for Windows, it will only support [Pageant](http://the.earth.li/~sgtatham/putty/0.66/htmldoc/Chapter9.html#pageant) as the SSH authentication agent.

## Credits

Big thanks to [Давид Мзареулян (David Mzareulyan)](https://github.com/davidmz) for creating the [go-pageant](https://github.com/davidmz/go-pageant) package!

## Issues

If you have an issue: report it on the [issue tracker](https://github.com/xanzy/ssh-agent/issues)

## Author

Sander van Harmelen (<sander@xanzy.io>)

## License

The files `pageant_windows.go` and `sshagent_windows.go` have their own license (see file headers). The rest of this package is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>
