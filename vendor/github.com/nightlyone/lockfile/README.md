lockfile
=========
Handle locking via pid files.

[![Build Status Unix][1]][2]
[![Build status Windows][3]][4]

[1]: https://secure.travis-ci.org/nightlyone/lockfile.png
[2]: https://travis-ci.org/nightlyone/lockfile
[3]: https://ci.appveyor.com/api/projects/status/7mojkmauj81uvp8u/branch/master?svg=true
[4]: https://ci.appveyor.com/project/nightlyone/lockfile/branch/master



install
-------
Install [Go 1][5], either [from source][6] or [with a prepackaged binary][7].
For Windows suport, Go 1.4 or newer is required.

Then run

	go get github.com/nightlyone/lockfile

[5]: http://golang.org
[6]: http://golang.org/doc/install/source
[7]: http://golang.org/doc/install

LICENSE
-------
BSD

documentation
-------------
[package documentation at godoc.org](http://godoc.org/github.com/nightlyone/lockfile)

install
-------------------
	go get github.com/nightlyone/lockfile


contributing
============

Contributions are welcome. Please open an issue or send me a pull request for a dedicated branch.
Make sure the git commit hooks show it works.

git commit hooks
-----------------------
enable commit hooks via

        cd .git ; rm -rf hooks; ln -s ../git-hooks hooks ; cd ..

