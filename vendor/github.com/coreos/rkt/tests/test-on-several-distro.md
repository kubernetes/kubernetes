# Tests on several Linux distributions

rkt aims to be supported on several Linux distributions.
In order to notice distro-specific issues, Continuous Integration should ideally run the tests on several Linux distributions.

rkt tests can be intrusive and require full root privileges.
Each test should be run on a fresh VM.
VMs should not be reused for next tests.

## Jenkins

Tests run on Jenkins [Jenkins](https://jenkins-ci.org/) [for each PR](https://jenkins-rkt-public.prod.coreos.systems/job/rkt-github-ci/) and [periodically on the master branch](https://jenkins-rkt-public.prod.coreos.systems/job/rkt-master-periodic/).

### AMIs

The script `tests/aws.sh` can generate a AMI of the specified Linux distribution with all the dependencies rkt needs.

First, install [aws-cli](https://github.com/aws/aws-cli) and configure it with your AWS credentials.
Then, create a key pair and a security group for rkt tests:

```
$ tests/aws.sh setup
```

Then generate an AMI of the specified Linux distribution:

```
$ tests/aws.sh fedora-22
$ tests/aws.sh fedora-23
$ tests/aws.sh fedora-24
$ tests/aws.sh fedora-rawhide
$ tests/aws.sh ubuntu-1604
$ tests/aws.sh ubuntu-1510
$ tests/aws.sh debian
$ tests/aws.sh centos
```

The generated AMIs can then be used to configure Jenkins.

If new packages are needed they can be added to the corresponding cloudinit files in `test/cloudinit`.
