phantomjs
=========

An NPM wrapper for [PhantomJS](http://phantomjs.org/), headless webkit with JS API.

[![Build Status](https://travis-ci.org/Medium/phantomjs.svg?branch=master)](https://travis-ci.org/Medium/phantomjs)

Building and Installing
-----------------------

```shell
npm install phantomjs
```

Or grab the source and

```shell
node ./install.js
```

What this installer is really doing is just grabbing a particular "blessed" (by
this module) version of Phantom. As new versions of Phantom are released
and vetted, this module will be updated accordingly.

The package has been set up to fetch and run Phantom for MacOS (darwin),
Linux based platforms (as identified by nodejs), and -- as of version 0.2.0 --
Windows (thanks to [Domenic Denicola](https://github.com/domenic)).  If you
spot any platform weirdnesses, let us know or send a patch.

### Custom binaries url
To use a mirror of the phantomjs binaries use npm config property `phantomjs_cdnurl`.
Default is `https://bitbucket.org/ariya/phantomjs/downloads`.

```shell
npm install phantomjs --phantomjs_cdnurl=http://cnpmjs.org/downloads
```

Or add property into your `.npmrc` file (https://www.npmjs.org/doc/files/npmrc.html)

```
phantomjs_cdnurl=http://cnpmjs.org/downloads
```

Another option is to use PATH variable `PHANTOMJS_CDNURL`.
```shell
PHANTOMJS_CDNURL=http://cnpmjs.org/downloads npm install phantomjs
```


Running
-------

```shell
bin/phantomjs [phantom arguments]
```

And npm will install a link to the binary in `node_modules/.bin` as
it is wont to do.

Running via node
----------------

The package exports a `path` string that contains the path to the
phantomjs binary/executable.

Below is an example of using this package via node.

```javascript
var path = require('path')
var childProcess = require('child_process')
var phantomjs = require('phantomjs')
var binPath = phantomjs.path

var childArgs = [
  path.join(__dirname, 'phantomjs-script.js'),
  'some other argument (passed to phantomjs script)'
]

childProcess.execFile(binPath, childArgs, function(err, stdout, stderr) {
  // handle results
})

```

Versioning
----------

The major and minor number tracks the version of PhantomJS that will be
installed. The patch number is incremented when there is either an installer
update or a patch build of the phantom binary.

A Note on PhantomJS
-------------------

PhantomJS is not a library for NodeJS.  It's a separate environment and code
written for node is unlikely to be compatible.  In particular PhantomJS does
not expose a Common JS package loader.

This is an _NPM wrapper_ and can be used to conveniently make Phantom available
It is not a Node JS wrapper.

I have had reasonable experiences writing standalone Phantom scripts which I
then drive from within a node program by spawning phantom in a child process.

Read the PhantomJS FAQ for more details: http://phantomjs.org/faq.html

### Linux Note

An extra note on Linux usage, from the PhantomJS download page:

 > This package is built on CentOS 5.8. It should run successfully on Lucid or
 > more modern systems (including other distributions). There is no requirement
 > to install Qt, WebKit, or any other libraries. It is however expected that
 > some base libraries necessary for rendering (FreeType, Fontconfig) and the
 > basic font files are available in the system.

Troubleshooting
---------------

##### Installation fails with `spawn ENOENT`

This is NPM's way of telling you that it was not able to start a `node` process. It usually means
Node is not on your PATH, or otherwise not properly installed.

##### Installation fails with `Error: EPERM` or `operation not permitted`

This error means that NPM was not able to install phantomjs to the file system. There are three
major reasons why this could happen:

- You don't have write access to the installation directory.
- The permissions in the NPM cache got messed up, and you need to run `npm cache clean` to fix them.
- You have over-zealous anti-virus software installed, and it's blocking file system writes.

##### Installation fails with `Error: read ECONNRESET` or `Error: connect ETIMEDOUT`

This error means that something went wrong with your internet connection, and the installer
was not able to download the PhantomJS binary for your platform. Please try again.

##### I tried again, but I get `ECONNRESET` or `ETIMEDOUT` consistently.

Do you live in China, or a country with an authoritarian government? We've seen problems where
the GFW or local ISP blocks bitbucket, preventing the installer from downloading the binary.

Try visiting the [the download page](http://cdn.bitbucket.org/ariya/phantomjs/downloads) manually.
If that page is blocked, you can try using a different CDN with the `PHANTOMJS_CDNURL`
env variable described above.

##### I am behind a corporate proxy that uses self-signed SSL certificates to intercept encrypted traffic.

You can tell NPM and the PhantomJS installer to skip validation of ssl keys with NPM's 
[strict-ssl](https://www.npmjs.org/doc/misc/npm-config.html#strict-ssl) setting:

```
npm set strict-ssl false
```

WARNING: Turning off `strict-ssl` leaves you vulnerable to attackers reading
your encrypted traffic, so run this at your own risk!

##### I tried everything, but my network is b0rked. What do I do?

If you install PhantomJS manually, and put it on PATH, the installer will try to
use the manually-installed binaries.

##### I'm on Debian or Ubuntu, and the installer failed because it couldn't find `node`

Some Linux distros tried to rename `node` to `nodejs` due to a package
conflict. This is a non-portable change, and we do not try to support this. The
[official documentation](https://github.com/joyent/node/wiki/Installing-Node.js-via-package-manager#ubuntu-mint-elementary-os)
recommends that you run `apt-get install nodejs-legacy` to symlink `node` to `nodejs` 
on those platforms, or many NodeJS programs won't work properly.

Contributing
------------

Questions, comments, bug reports, and pull requests are all welcome.  Submit them at
[the project on GitHub](https://github.com/Obvious/phantomjs/).  If you haven't contributed to an
[Obvious](http://github.com/Obvious/) project before please head over to the
[Open Source Project](https://github.com/Obvious/open-source#note-to-external-contributors) and fill
out an OCLA (it should be pretty painless).

Bug reports that include steps-to-reproduce (including code) are the
best. Even better, make them in the form of pull requests.

Author
------

[Dan Pupius](https://github.com/dpup)
([personal website](http://pupius.co.uk)), supported by
[The Obvious Corporation](http://obvious.com/).

License
-------

Copyright 2012 [The Obvious Corporation](http://obvious.com/).

Licensed under the Apache License, Version 2.0.
See the top-level file `LICENSE.txt` and
(http://www.apache.org/licenses/LICENSE-2.0).
