# Tests on several Linux distributions

rkt aims to be supported on several Linux distributions.
In order to notice distro-specific issues, Continuous Integration should ideally run the tests on several Linux distributions.
This is not done yet but there is a script to help with manual testing on several Linux distributions.

rkt tests can be intrusive and require full root privileges.
Each test should be run on a fresh VM.
VMs should not be reused for next tests.

## AWS

The script `tests/aws.sh` can automatically spawn a fresh VM of the specified Linux distribution and start the rkt tests.

First, install [aws-cli](https://github.com/aws/aws-cli) and configure it with your AWS credentials.
Then, create a key pair and a security group for rkt tests:

```
$ tests/aws.sh setup
```

Then run the tests with the specified Linux distribution:

```
$ tests/aws.sh fedora-22
$ tests/aws.sh fedora-23
$ tests/aws.sh fedora-rawhide
$ tests/aws.sh ubuntu-1604
$ tests/aws.sh ubuntu-1510
$ tests/aws.sh debian
$ tests/aws.sh centos
```

By default, this tests the upstream master branch.
A specific branch can be tested with:

```
$ tests/aws.sh fedora-23 https://github.com/coreos/rkt.git branch-name
```

Additionally, a stage1 flavor can be selected:
```
$ tests/aws.sh fedora-23 https://github.com/coreos/rkt.git branch-name coreos
```

The VM instances are configured to terminate automatically on shutdown to reduce costs.
However they are not shut down automatically after the tests.
It is recommended to check in the AWS console that instances are terminated.

## Jenkins

This could be automatised with [Jenkins](https://jenkins-ci.org/).
In order to spawn a new VM for each test, the [Amazon EC2 Plugin](https://wiki.jenkins-ci.org/display/JENKINS/Amazon+EC2+Plugin) could be used.
However, it would require additional fixes: [JENKINS-8618](https://issues.jenkins-ci.org/browse/JENKINS-8618).


