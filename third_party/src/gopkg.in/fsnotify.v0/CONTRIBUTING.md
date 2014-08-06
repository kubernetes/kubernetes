# Contributing

* Send questions to [golang-dev@googlegroups.com](mailto:golang-dev@googlegroups.com). 

### Issues

* Request features and report bugs using the [GitHub Issue Tracker](https://github.com/go-fsnotify/fsnotify/issues).
* Please indicate the platform you are running on.

### Pull Requests

A future version of Go will have [fsnotify in the standard library](https://code.google.com/p/go/issues/detail?id=4068), therefore fsnotify carries the same [LICENSE](https://github.com/go-fsnotify/fsnotify/blob/master/LICENSE) as Go. Contributors retain their copyright, so we need you to fill out a short form before we can accept your contribution: [Google Individual Contributor License Agreement](https://developers.google.com/open-source/cla/individual).

Please indicate that you have signed the CLA in your pull request.

To hack on fsnotify:

1. Install as usual (`go get -u github.com/go-fsnotify/fsnotify`)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Ensure everything works and the tests pass (see below)
4. Commit your changes (`git commit -am 'Add some feature'`)

Contribute upstream:

1. Fork fsnotify on GitHub
2. Add your remote (`git remote add fork git@github.com:mycompany/repo.git`)
3. Push to the branch (`git push fork my-new-feature`)
4. Create a new Pull Request on GitHub

If other team members need your patch before I merge it:

1. Install as usual (`go get -u github.com/go-fsnotify/fsnotify`)
2. Add your remote (`git remote add fork git@github.com:mycompany/repo.git`)
3. Pull your revisions (`git fetch fork; git checkout -b my-new-feature fork/my-new-feature`)

Notice: For smooth sailing, always use the original import path. Installing with `go get` makes this easy.

Note: The maintainers will update the CHANGELOG on your behalf. Please don't modify it in your pull request.

### Testing

fsnotify uses build tags to compile different code on Linux, BSD, OS X, and Windows.

Before doing a pull request, please do your best to test your changes on multiple platforms, and list which platforms you were able/unable to test on.

To make cross-platform testing easier, I've created a Vagrantfile for Linux and BSD.

* Install [Vagrant](http://www.vagrantup.com/) and [VirtualBox](https://www.virtualbox.org/)
* Setup [Vagrant Gopher](https://github.com/nathany/vagrant-gopher) in your `src` folder.
* Run `vagrant up` from the project folder. You can also setup just one box with `vagrant up linux` or `vagrant up bsd` (note: the BSD box doesn't support Windows hosts at this time, and NFS may prompt for your host OS password)
* Once setup, you can run the test suite on a given OS with a single command `vagrant ssh linux -c 'cd go-fsnotify/fsnotify; go test'`.
* When you're done, you will want to halt or destroy the Vagrant boxes.

Notice: fsnotify file system events don't work on shared folders. The tests get around this limitation by using a tmp directory, but it is something to be aware of.

Right now I don't have an equivalent solution for Windows and OS X, but there are Windows VMs [freely available from Microsoft](http://www.modern.ie/en-us/virtualization-tools#downloads).
