AC_PREREQ([2.63])
AC_INIT([rkt], [1.25.0], [https://github.com/coreos/rkt/issues])

AC_PROG_CC
AC_PROG_CXX

AC_DEFUN([RKT_CHECK_PROG],
         [AS_VAR_IF([$1], [],
                    [AC_CHECK_PROG($@)],[:])])

AC_DEFUN([RKT_REQ_PROG],
         [AS_VAR_IF([$1], [],
                    [AC_CHECK_PROG($@)
                     AS_VAR_IF([$1], [$3], [:], AC_MSG_ERROR([*** $2 not found]))],
                    [:])])

AC_DEFUN([RKT_REQ_ABS_PROG],
         [AS_VAR_IF([$1], [],
                    [AC_PATH_PROG($@)
                     AS_IF([test "x$]$1[" = 'x'], [AC_MSG_ERROR([*** $2 not found])])],
                    [:])])

dnl We assume having bash in lots of places, so check it in the
dnl beginning.
RKT_REQ_ABS_PROG([BASH_SHELL], [bash])
dnl for git suffix
RKT_REQ_PROG([GIT],[git],[git])

dnl this will be printed in configure summary and by rkt version
RKT_FEATURES=""
RKT_ADD_FEATURE() {
    RKT_FEATURES=`AS_ECHO(["${RKT_FEATURES} $1"]) | sed -e 's/^ \+//'`
}
dnl if version ends with +git, append a short git-hash.
AS_IF([test `expr match 'AC_PACKAGE_VERSION' '.*+git$'` -gt 0],
      dnl version has +git suffix, ignore errors (not a git repo)
      [RKT_VERSION="AC_PACKAGE_VERSION`git rev-parse --short HEAD 2>/dev/null``git diff-index --quiet HEAD 2>/dev/null || AS_ECHO(["-dirty"])`"],
      dnl version has no +git suffix
      [RKT_VERSION="AC_PACKAGE_VERSION"])

AC_CANONICAL_HOST
AC_CANONICAL_BUILD

AS_CASE([${host_cpu}],
        [aarch64],             [RKT_ACI_ARCH=aarch64; GOARCH=arm64;],
        [armv6l],              [RKT_ACI_ARCH=armv6l;  GOARCH=arm; GOARM=6;],
        [armv7l],              [RKT_ACI_ARCH=armv7l;  GOARCH=arm; GOARM=7;],
        [i?86 | x86],          [RKT_ACI_ARCH=i386;    GOARCH=386;],
        [powerpc | powerpc64], [RKT_ACI_ARCH=ppc64;   GOARCH=ppc64;],
        [powerpc64le],         [RKT_ACI_ARCH=ppc64le; GOARCH=ppc64le;],
        [x86_64],              [RKT_ACI_ARCH=amd64;   GOARCH=amd64;],
        [AC_MSG_ERROR([*** unsupported host arch specified: ${host_cpu}])])

#### FLAGS

## STAGE1 - detailed setup

AC_ARG_WITH(stage1-default-name,
            [AS_HELP_STRING([--with-stage1-default-name],
                            [default name of stage1, default: 'coreos.com/rkt/stage1-${flavor}'])],
            [RKT_STAGE1_DEFAULT_NAME="${withval}"],
            [RKT_STAGE1_DEFAULT_NAME=auto])

AC_ARG_WITH(stage1-default-version,
            [AS_HELP_STRING([--with-stage1-default-version],
                            [default version of stage1, default: same as rkt version])],
            [RKT_STAGE1_DEFAULT_VERSION="${withval}"],
            [RKT_STAGE1_DEFAULT_VERSION=auto])

## STAGE1 - flavor setup

AC_ARG_WITH([stage1-default-flavor],
            [AS_HELP_STRING([--with-stage1-default-flavor],
                            [stage1 flavor that will be loaded by default; has to be listed in --with-stage1-flavors, default: first from the stage1 flavors list])],
            [RKT_STAGE1_DEFAULT_FLAVOR="${withval}"],
            [RKT_STAGE1_DEFAULT_FLAVOR=auto])

## STAGE1 - common setup

AC_ARG_WITH([stage1-flavors],
            [AS_HELP_STRING([--with-stage1-flavors],
                            [comma-separated list of stage1 flavors; choose from 'src', 'coreos', 'host', 'kvm', 'fly'; default: 'coreos,kvm,fly'])],
            [RKT_STAGE1_FLAVORS="${withval}"],
            [RKT_STAGE1_FLAVORS=auto])

AC_ARG_WITH([stage1-flavors-version-override],
            [AS_HELP_STRING([--with-stage1-flavors-version-override],
                            [override the version of stage1 with a custom one; it also changes the default stage1 version in flavor setup])],
            [RKT_STAGE1_FLAVORS_VERSION_OVERRIDE="${withval}"],
            [RKT_STAGE1_FLAVORS_VERSION_OVERRIDE=auto])

AC_ARG_WITH(stage1-default-location,
            [AS_HELP_STRING([--with-stage1-default-location],
                            [default location of stage1, default: none])],
            [RKT_STAGE1_DEFAULT_LOCATION="${withval}"],
            [RKT_STAGE1_DEFAULT_LOCATION=auto])

AC_ARG_WITH(stage1-default-images-directory,
            [AS_HELP_STRING([--with-stage1-default-images-directory],
                            [default directory for stage1 images, default: LIBDIR/rkt/stage1-images])],
            [RKT_STAGE1_DEFAULT_IMAGES_DIR="${withval}"],
            [RKT_STAGE1_DEFAULT_IMAGES_DIR=auto])

# STAGE1 - systemd src and version for src flavor

AC_ARG_WITH([stage1-systemd-src],
            [AS_HELP_STRING([--with-stage1-systemd-src],
                            [address to git repository of systemd, used in 'src' stage1 flavor (default: 'https://github.com/systemd/systemd.git')])],
            [RKT_STAGE1_SYSTEMD_SRC="${withval}"],
            [RKT_STAGE1_SYSTEMD_SRC='auto'])

AC_ARG_WITH([stage1-systemd-version],
            [AS_HELP_STRING([--with-stage1-systemd-version],
                            [systemd version to build, used in 'src' stage1 flavor (default: 'v229', should be in format 'v<number>', like v229)])],
            [RKT_STAGE1_SYSTEMD_VER="${withval}"],
            [RKT_STAGE1_SYSTEMD_VER='auto'])

AC_ARG_WITH([stage1-systemd-revision],
            [AS_HELP_STRING([--with-stage1-systemd-revision],
                            [systemd revision to check out to build, used in 'src' stage1 flavor (default: the same value as --stage1-systemd-version)])],
            [RKT_STAGE1_SYSTEMD_REV="${withval}"],
            [RKT_STAGE1_SYSTEMD_REV='auto'])

# STAGE1 - kvm hypervisor. lkvm or/and qemu can be chosen
AC_ARG_WITH([stage1-kvm-hypervisors],
            [AS_HELP_STRING([--with-stage1-kvm-hypervisors],
                            [select hypervisors for kvm flavor; build stage1 aci for each of them. select "lkvm", "qemu" or both (default: 'qemu')])],
            [RKT_STAGE1_KVM_HV="${withval}"],
            [RKT_STAGE1_KVM_HV='qemu'])

## STAGE1 - path to coreos pxe and its systemd version for kvm and coreos flavors

AC_ARG_WITH([coreos-local-pxe-image-path],
            [AS_HELP_STRING([--with-coreos-local-pxe-image-path],
                            [path to local CoreOS PXE image, used in 'coreos' and 'kvm' stage1 flavors (default: empty) (the GPG signature of this file will not be checked!)])],
            [RKT_LOCAL_COREOS_PXE_IMAGE_PATH="${withval}"],
            [RKT_LOCAL_COREOS_PXE_IMAGE_PATH=""])

AC_ARG_WITH([coreos-local-pxe-image-systemd-version],
            [AS_HELP_STRING([--with-coreos-local-pxe-image-systemd-version],
                            [version of systemd in local CoreOS PXE image, used in 'coreos' and 'kvm' stage1 flavors (should be in format 'v<number>', like v229)])],
            [RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER="${withval}"],
            [RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER=])

RKT_STAGE1_COREOS_BOARD=${GOARCH}-usr;

## Functional tests

AC_ARG_ENABLE([functional-tests],
              [AS_HELP_STRING([--enable-functional-tests],
                              [enable functional tests on make check (linux only, uses sudo, default: 'no', use 'auto' to enable if possible, for host stage1 flavor systemd version 227 or higher on host is required)])],
              [RKT_RUN_FUNCTIONAL_TESTS="${enableval}"],
              [RKT_RUN_FUNCTIONAL_TESTS="no"])

## Incremental build

AC_ARG_ENABLE([incremental-build],
              [AS_HELP_STRING([--enable-incremental-build],
                              [enable incremental build. Note: this does not work for cross compile builds.])],
              [INCREMENTAL_BUILD="${enableval}"],
              [INCREMENTAL_BUILD="no"])
## TPM

AC_ARG_ENABLE([tpm],
              [AS_HELP_STRING([--enable-tpm],
                              [enable logging to TPM, use 'auto' to enable it if the required development files are found, default: 'yes'])],
              [RKT_ENABLE_TPM="${enableval}"],
              [RKT_ENABLE_TPM="yes"])

## Systemd-journal support on host

AC_ARG_ENABLE([sdjournal],
              [AS_HELP_STRING([--enable-sdjournal],
                              [enable systemd-journal features on host, use 'auto' to enable it if the required development files are found, default: 'yes'])],
              [RKT_ENABLE_SDJOURNAL="${enableval}"],
              [RKT_ENABLE_SDJOURNAL="yes"])

## insecure go

AC_ARG_ENABLE([insecure-go],
              [AS_HELP_STRING([--enable-insecure-go],
                              [allow building rkt with go having known security issues, default: no])],
              [RKT_INSECURE_GO="${enableval}"],
              [RKT_INSECURE_GO="no"])

#### CHECKING

## STAGE1 - initial verification and setup

dnl There are two kinds of setup - flavor and detailed. A flavor setup
dnl is when we specify a default flavor. A detailed setup is when we
dnl specify a default name and version instead.
AS_IF([test "x${RKT_STAGE1_DEFAULT_FLAVOR}" != 'xauto' -a '(' "x${RKT_STAGE1_DEFAULT_NAME}" != 'xauto' -o "x${RKT_STAGE1_DEFAULT_VERSION}" != 'xauto' ')'],
              dnl Passed both --with-stage1-default-flavor and
              dnl --with-stage1-default-{name,version} - bail out,
              dnl they are mutually exclusive.
              [AC_MSG_ERROR([*** --with-stage1-default-flavor and --with-stage1-default-{name,version} flags are mutually exclusive; use either default flavor or specific default name and version])],
      [test "x${RKT_STAGE1_DEFAULT_FLAVOR}" = 'xauto' -a "x${RKT_STAGE1_DEFAULT_NAME}" = 'xauto' -a "x${RKT_STAGE1_DEFAULT_VERSION}" = 'xauto'],
              dnl Passed neither --with-stage1-default-flavor nor
              dnl --with-stage1-default-{name,version} - default to
              dnl flavor setup.
              [RKT_STAGE1_SETUP_KIND=flavor],
      [test "x${RKT_STAGE1_DEFAULT_FLAVOR}" != 'xauto'],
              dnl Passed --with-stage1-default-flavor flag - default
              dnl to flavor setup.
              [RKT_STAGE1_SETUP_KIND=flavor],
      [test "x${RKT_STAGE1_DEFAULT_NAME}" != 'xauto' -o "x${RKT_STAGE1_DEFAULT_VERSION}" != 'xauto'],
              dnl Passed one or both from
              dnl --with-stage1-default-{name,version} - default to
              dnl detailed setup.
              [RKT_STAGE1_SETUP_KIND=detailed])

AS_CASE([${RKT_STAGE1_SETUP_KIND}],
        [detailed],
                  dnl Detailed setup - do not build any flavor unless
                  dnl explicitly asked for it and make sure that both
                  dnl name and version were passed.
                  [AS_VAR_IF([RKT_STAGE1_FLAVORS],[auto],
                             [RKT_STAGE1_FLAVORS=''])
                   RKT_STAGE1_DEFAULT_FLAVOR=''
                   AS_IF([test "x${RKT_STAGE1_DEFAULT_NAME}" = 'xauto' -o "x${RKT_STAGE1_DEFAULT_VERSION}" = 'xauto'],
                         [AC_MSG_ERROR([*** Both --with-stage1-default-name and --with-stage1-default-version must be specified])])],
        [flavor],
                  dnl The flavor setup - build coreos and kvm flavors
                  dnl and set the first flavor in the list as a
                  dnl default unless specified.
                  [AS_VAR_IF([RKT_STAGE1_FLAVORS],[auto],
                             [RKT_STAGE1_FLAVORS='coreos,kvm,fly'])
                   AS_VAR_IF([RKT_STAGE1_DEFAULT_FLAVOR],[auto],
                             [RKT_STAGE1_DEFAULT_FLAVOR=`AS_ECHO([${RKT_STAGE1_FLAVORS}]) | cut -d, -f1`])
                   RKT_STAGE1_DEFAULT_NAME=''
                   RKT_STAGE1_DEFAULT_VERSION=''],
        [AC_MSG_ERROR([*** Unknown setup kind, should not happen])])

## Stage1 flavor version override verification

AS_VAR_IF([RKT_STAGE1_FLAVORS_VERSION_OVERRIDE], [auto],
          dnl no override, no checking
          [RKT_STAGE1_FLAVORS_VERSION_OVERRIDE=''],
          dnl check if we override the version, even if we build no flavors at all
          [AS_VAR_IF([RKT_STAGE1_FLAVORS], [],
                     [AC_MSG_WARN([* Overriding stage1 version, but no stage1 flavors are built])])])

## Stage1 default image directory

dnl Normally it is autoconf which sets prefix and exec_prefix
dnl variables to some default values if they were NONE, but that
dnl happens way too late, so we need to do it ourselves right now.
AS_VAR_IF([prefix],[NONE],
          [prefix="${ac_default_prefix}"])
AS_VAR_IF([exec_prefix],[NONE],
          [exec_prefix="${prefix}"])
AS_VAR_IF([RKT_STAGE1_DEFAULT_IMAGES_DIR],[auto],
          dnl Calling eval makes sure that ${libdir} is expanded.
          dnl autoconf defines it as '${eprefix}/lib' - note the
          dnl single quotes, which make ${eprefix} to be not
          dnl expanded. There is some rationale behind all this, but
          dnl we need no features coming from unexpanded variables.
          [eval RKT_STAGE1_DEFAULT_IMAGES_DIR="${libdir}/rkt/stage1-images"])
AS_CASE([${RKT_STAGE1_DEFAULT_IMAGES_DIR}],
        [/*],
                dnl alright, the path is absolute
                [:],
        [AC_MSG_ERROR([*** Path to the stage1 images directory must be absolute])])

## Built stage1 flavors and hypervisors verification

dnl a list of all flavors
RKT_STAGE1_ALL_FLAVORS=coreos,kvm,host,src,fly

dnl a list of all kvm hypervisors
RKT_STAGE1_KVM_ALL_HV=lkvm,qemu

dnl RKT_IS_VALID_FLAVOR checks if $1 is a valid flavor, $2 is an
dnl additional string to an error message.
AC_DEFUN([RKT_IS_VALID_FLAVOR],
         [AS_CASE([$1],
                  dnl Correct flavor, nothing to do.
                  [coreos|kvm|host|src|fly],
                          [],
                  dnl Bogus flavor, bail out.
                  [AC_MSG_ERROR([*** Unknown stage1 flavor "$1" $2])])])

dnl RKT_IS_VALID_HYPERVISOR checks if $1 is a valid kvm hypervisor, $2 is an
dnl additional string to an error message.
AC_DEFUN([RKT_IS_VALID_HYPERVISOR],
         [AS_CASE([$1],
                  dnl Correct hypervisor, nothing to do.
                  [lkvm|qemu],
                          [],
                  dnl Bogus hypervisor, bail out.
                  [AC_MSG_ERROR([*** Unknown stage1 kvm hypervisor "$1" $2])])])

dnl RKT_ITERATE iterates all comma-separated values stored in
dnl $1 using an iterator variable $2 and executes body $3.
AC_DEFUN([RKT_ITERATE],
         [for $2 in `AS_ECHO(["$1"]) | tr ',' ' '`
          do
              :
              $3
          done])

dnl RKT_IF_HAS_VALUE checks if the comma-separated list of values in
dnl $1 contains any value in the comma-separated list of values in
dnl $2; if yes, executes $3, otherwise - $4.
AC_DEFUN([RKT_IF_HAS_VALUE],
         [value_found=0
          RKT_ITERATE([$2],[value],
                              [AS_IF([AS_ECHO([",$1,"]) | grep --silent ",${value},"],
                                     [value_found=1
                                      break])])
          AS_IF([test ${value_found} -eq 1],[:;$3],[:;$4])])

dnl common deps for flavors using CoreOS PXE image; not checking for
dnl gpg - it will be checked when we will actually download the image
dnl from the network
AC_DEFUN([RKT_COMMON_COREOS_PROGS],
         [RKT_REQ_PROG([CAT],[cat],[cat])
          RKT_REQ_PROG([COMM],[comm],[comm])
          RKT_REQ_PROG([CPIO],[cpio],[cpio])
          RKT_REQ_PROG([GZIP],[gzip],[gzip])
          RKT_REQ_PROG([MD5SUM],[md5sum],[md5sum])
          RKT_REQ_PROG([MKTEMP],[mktemp],[mktemp])
          RKT_REQ_PROG([SORT],[sort],[sort])
          RKT_REQ_PROG([TOUCH],[touch],[touch])
          RKT_REQ_PROG([UNSQUASHFS],[unsquashfs],[unsquashfs])
          RKT_REQ_PROG([WGET],[wget],[wget])])

dnl Validate passed flavors to build - make sure that valid flavors
dnl were passed and each flavor was specified only once. Also, do some
dnl basic program checks for each requested flavor.
RKT_SPECIFIED_FLAVORS=''
RKT_ITERATE([${RKT_STAGE1_FLAVORS}],[flavor],
                    [RKT_IS_VALID_FLAVOR([${flavor}],[in --with-stage1-flavors])
                     RKT_IF_HAS_VALUE([${RKT_SPECIFIED_FLAVORS}],[${flavor}],
                                       [AC_MSG_ERROR([*** "${flavor}" flavor specified twice in --with-stage1-flavors])])
                     RKT_SPECIFIED_FLAVORS="${RKT_SPECIFIED_FLAVORS},${flavor}"
                     AS_CASE([${flavor}],
                             [src],
                                     [AC_MSG_NOTICE([will build systemd from source, make sure that all its build requirements are fulfilled])],
                             [coreos],
                                     [RKT_COMMON_COREOS_PROGS],
                             [kvm],
                                     [AC_MSG_NOTICE([will build linux kernel and kvm hypervisors (${RKT_STAGE1_KVM_HV}) from source, make sure that all their build requirements are fulfilled])
                                      RKT_COMMON_COREOS_PROGS
                                      RKT_REQ_PROG([PATCH],[patch],[patch])
                                      RKT_REQ_PROG([TAR],[tar],[tar])
                                      RKT_REQ_PROG([XZ],[xz],[xz])
                                      AC_CHECK_HEADER([openssl/bio.h],
                                                      [],
                                                      [AC_MSG_ERROR([** No development headers for openssl found])],
                                                      [AC_INCLUDES_DEFAULT])],
                             [host],
                                     [],
                             [fly],
                                     [AC_MSG_WARN([* fly is an experimental stage1 implementation with almost no isolation and less features])],
                             [AC_MSG_ERROR([*** Unhandled flavor "${flavor}", should not happen])])])


AS_VAR_IF([RKT_SPECIFIED_FLAVORS],[],
          dnl no flavors specified, no checks
          [],
          dnl some flavor are specified, do some checks for stuff needed by any flavor
          [AC_LANG_PUSH([C])
           AC_CHECK_HEADER([acl/libacl.h],
                           [],
                           [AC_MSG_ERROR([*** No development headers for libacl found])],
                           [AC_INCLUDES_DEFAULT])
           AC_CHECK_HEADER([dlfcn.h],
                           [],
                           [AC_MSG_ERROR([*** No development headers for libdl found])],
                           [AC_INCLUDES_DEFAULT])
           # check for static libc
           SAVE_LDFLAGS="$LDFLAGS"
           LDFLAGS="-static $LDFLAGS"
           AC_CHECK_LIB([c], [printf],
                        dnl static libc is there
                        [:],
                        dnl static libc is not there
                        [AC_MSG_ERROR([*** No static libc found. Try to install glibc-static or libc6-dev.])])
           LDFLAGS="$SAVE_LDFLAGS"

           # check for setns syscall, drop it in 2020 (centos 6.7 support ends
           # then).
           AC_MSG_CHECKING([whether C library provides setns function])
           AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                                             #define _GNU_SOURCE
                                             #include <sched.h>
                                           ]], [[(void)setns(0, 0);]])],
                          [AC_MSG_RESULT([yes])],
                          [AC_MSG_RESULT([no])
                           AC_MSG_CHECKING([whether Linux kernel headers provide __NR_setns macro])
                           AC_LINK_IFELSE([AC_LANG_PROGRAM([[
                                                             #include <linux/unistd.h>
                                                           ]], [[(void)syscall(__NR_setns,0,0);]])],
                                          [AC_MSG_RESULT([yes])
                                           RKT_DEFINES_FOR_ENTER=-DNO_SETNS_AVAILABLE],
                                          [AC_MSG_RESULT([no])
                                           AC_MSG_ERROR([*** Neither setns function nor __NR_setns macro are available - either both glibc and kernel are too old or their development headers are not installed])])])

           AC_LANG_POP([C])])

dnl Validate passed default flavor, it should be one of the built
dnl stage1 flavors

AS_VAR_IF([RKT_STAGE1_SETUP_KIND],['flavor'],
          [RKT_IS_VALID_FLAVOR([${RKT_STAGE1_DEFAULT_FLAVOR}],[in --with-stage1-default-flavor])
           RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}],[${RKT_STAGE1_DEFAULT_FLAVOR}],
                             dnl valid default flavor, alright
                             [],
                             dnl not a part of built flavors, bail out
                             [AC_MSG_ERROR([*** --with-stage1-default-flavor should use one of flavors from --with-stage1-flavors (${RKT_STAGE1_FLAVORS})])])])

## Options specific for src flavor

# Process systemd src repo and version for src flavor.
RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}],[src],
                  dnl we are building src flavor
                  [AS_VAR_IF([RKT_STAGE1_SYSTEMD_SRC], [auto],
                             dnl systemd source not specified, use default
                             [RKT_STAGE1_SYSTEMD_SRC='https://github.com/systemd/systemd.git'],
                             dnl systemd source specified, use it
                             [:])
                   AS_VAR_IF([RKT_STAGE1_SYSTEMD_VER], [auto],
                             dnl systemd version not specified, use default
                             [RKT_STAGE1_SYSTEMD_VER='v999'],
                             dnl systemd version specified, use it
                             [:])
                   AS_VAR_IF([RKT_STAGE1_SYSTEMD_REV], [auto],
                             dnl systemd revision not specified, use default
                             [RKT_STAGE1_SYSTEMD_REV="master"],
                             dnl systemd revision specified, use it
                             [:])],
                  dnl we are not building src flavor
                  [AS_VAR_IF([RKT_STAGE1_SYSTEMD_SRC], [auto],
                             dnl systemd source not specified, alright, zero the value just in case
                             [RKT_STAGE1_SYSTEMD_SRC=''],
                             dnl systemd source specified, warn about it being ignored
                             [AC_MSG_WARN([* --with-stage1-systemd-src is ignored, src flavor is not built])])
                   AS_VAR_IF([RKT_STAGE1_SYSTEMD_REV], [auto],
                             dnl systemd source not specified, alright, zero the value just in case
                             [RKT_STAGE1_SYSTEMD_REV=''],
                             dnl systemd source specified, warn about it being ignored
                             [AC_MSG_WARN([* --with-stage1-systemd-revision is ignored, src flavor is not built])])
                   AS_VAR_IF([RKT_STAGE1_SYSTEMD_VER], [auto],
                             dnl systemd version not specified, alright, zero the value just in case
                             [RKT_STAGE1_SYSTEMD_VER=''],
                             dnl systemd version specified, warn about it being ignored
                             [AC_MSG_WARN([* --with-stage1-systemd-version is ignored, src flavor is not built])])])

dnl Warn if version is HEAD, just use master.
AS_VAR_IF([RKT_STAGE1_SYSTEMD_VER], [HEAD],
          [AC_MSG_WARN([* 'HEAD' is not a systemd version, setting it to 'master' instead])
           RKT_STAGE1_SYSTEMD_VER='master'])

## Options specific for kvm flavor

RKT_SPECIFIED_HYPERVISORS=''
RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}],[kvm],
          [RKT_ITERATE([${RKT_STAGE1_KVM_HV}],[hypervisor],
                     [RKT_IS_VALID_HYPERVISOR([${hypervisor}],[in --with-stage1-kvm-hypervisors])
                      RKT_IF_HAS_VALUE([${RKT_SPECIFIED_HYPERVISORS}],[${hypervisor}],
                                [AC_MSG_ERROR([*** "${hypervisor}" hypervisor specified twice in --with-stage1-kvm-hypervisors])])
                                RKT_SPECIFIED_HYPERVISORS="${RKT_SPECIFIED_HYPERVISORS},${hypervisor}"
                                AS_CASE([${hypervisor}],
                                          [lkvm],
                                                    [AC_MSG_NOTICE([including lkvm...])],
                                          [qemu],
                                                    [AC_MSG_NOTICE([including qemu...])],
                                          [AC_MSG_ERROR([*** Unhandled hypervisor "${hypervisor}". should not happen])])])])

## Options specific to coreos/kvm flavors

# Process local CoreOS PXE image path and its systemd version for
# coreos/kvm flavors.
RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}],[coreos,kvm],
                  dnl for kvm and coreos check if both pxe image path and
                  dnl systemd version are specified or neither
                  [AS_VAR_IF([RKT_LOCAL_COREOS_PXE_IMAGE_PATH], [],
                             dnl no local coreos pxe image path specified
                             [AS_VAR_IF([RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER],[],
                                        dnl no version specified, alright, keep quiet
                                        [:],
                                        dnl version specified, warn about it being ignored
                                        [AC_MSG_WARN([* --with-coreos-local-pxe-image-systemd-version is ignored, because --with-coreos-local-pxe-image-path was not specified])
                                         RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER=''])
                              dnl gpg is required for signature
                              dnl verification
                              RKT_REQ_PROG([GPG],[gpg],[gpg])],
                             dnl local coreos pxe image path specified
                             [AS_VAR_IF([RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER],[],
                                        dnl no version specified, bail out, version is required
                                        [AC_MSG_ERROR([*** No systemd version for local CoreOS PXE image specified, use --with-coreos-local-pxe-image-systemd-version])],
                                        dnl version specified, alright, keep quiet
                                        [:])])],
                  dnl for other flavors just warn if either is specified
                  [AS_VAR_IF([RKT_LOCAL_COREOS_PXE_IMAGE_PATH], [],
                             dnl empty value, alright
                             [:],
                             dnl not empty, warn about it, zero it just in case
                             [AC_MSG_WARN([* --with-coreos-local-pxe-image-path is ignored by ${RKT_STAGE1_DEFAULT_FLAVOR} stage1 flavor])
                              RKT_LOCAL_COREOS_PXE_IMAGE_PATH=''])
                   AS_VAR_IF([RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER], [],
                             dnl empty value, alright
                             [:],
                             dnl not empty, warn about it, zero it just in case
                             [AC_MSG_WARN([* --with-coreos-local-pxe-image-systemd-version is ignored by ${RKT_STAGE1_DEFAULT_FLAVOR} stage1 flavor])
                              RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER=''])])

## Setup default name, version, location and a filename in rkt dir,
## take version override into account for flavored setup

# This will be set to something like stage1-coreos.aci, so rkt will
# know what to look for in the rkt directory if default stage1 image
# location is empty.
RKT_STAGE1_DEFAULT_IMAGE_FILENAME_IN_RKT_DIRECTORY=''

dnl Validate a default stage1 image location - check whether the
dnl location is either a URL with supported scheme (docker, https,
dnl http or file) or an absolute path. If it is the latter, get a
dnl filename which will be used by rkt to load the stage1 image file
dnl from the same directory if the image in the default location is
dnl absent.
AS_CASE([${RKT_STAGE1_DEFAULT_LOCATION}],
        [auto],
                dnl no location passed at all, nix it
                [RKT_STAGE1_DEFAULT_LOCATION=''],
        [docker://|https://|http://|file://],
                dnl alright, a probably valid URL
                [:],
        [/*],
                dnl alright, an absolute path, let's get a filename
                [RKT_STAGE1_DEFAULT_IMAGE_FILENAME_IN_RKT_DIRECTORY=`basename ${RKT_STAGE1_DEFAULT_LOCATION}`],
        dnl wrong default stage1 image location
        [AC_MSG_ERROR([*** Default stage1 image path is neither a valid URL (with docker://, https://, http:// or file:// scheme) nor an absolute path])])

AS_CASE([${RKT_STAGE1_SETUP_KIND}],
        [detailed],
                  dnl detailed setup - not much to do, default name
                  dnl and version are already specified, warn if
                  dnl default location was not specified
                  [AS_VAR_IF([RKT_STAGE1_DEFAULT_LOCATION],[],
                             [AC_MSG_WARN([* It may be necessary to pass --stage1-{url,path,name,hash,from-dir} flag to run rkt])])],
        [flavor],
                  dnl flavor setup - set the default name and version,
                  dnl set the location if not specified
                  [AS_CASE([${RKT_STAGE1_DEFAULT_FLAVOR}],
                                   dnl for src flavor, we also append systemd version to stage1 name
                                   [src],
                                           [RKT_STAGE1_DEFAULT_NAME="coreos.com/rkt/stage1-${RKT_STAGE1_DEFAULT_FLAVOR}-${RKT_STAGE1_SYSTEMD_VER}"],
                                   dnl other flavors get only flavor name
                                   [RKT_STAGE1_DEFAULT_NAME="coreos.com/rkt/stage1-${RKT_STAGE1_DEFAULT_FLAVOR}"])
                   AS_VAR_IF([RKT_STAGE1_FLAVORS_VERSION_OVERRIDE], [],
                             [RKT_STAGE1_DEFAULT_VERSION="${RKT_VERSION}"],
                             [RKT_STAGE1_DEFAULT_VERSION="${RKT_STAGE1_FLAVORS_VERSION_OVERRIDE}"])
                   AS_VAR_IF([RKT_STAGE1_DEFAULT_LOCATION],[],
                             [RKT_STAGE1_DEFAULT_IMAGE_FILENAME_IN_RKT_DIRECTORY="stage1-${RKT_STAGE1_DEFAULT_FLAVOR}.aci"])],
        [AC_MSG_ERROR([*** Unknown setup kind, should not happen])])

## Functional tests

dnl Check for gpg - required by functional tests. Note that this does
dnl not bail out here if it is not found.
RKT_CHECK_PROG([GPG],[gpg],[gpg])

AS_IF([test "x${RKT_RUN_FUNCTIONAL_TESTS}" = 'xyes' -o "x${RKT_RUN_FUNCTIONAL_TESTS}" = 'xauto'],
      [AC_MSG_CHECKING([whether functional tests can be run])
       rkt_functional_tests_msg=''
       AS_CASE([${host}],
               [*-*-linux*],
                       dnl we are on linux, checks can continue
                       [],
               dnl non-linux host
               [rkt_functional_tests_msg='Functional tests are supported only under linux'])
      AS_VAR_IF([rkt_functional_tests_msg],[],
                dnl we are on linux host; check if we have default
                dnl flavor set
                [AS_VAR_IF([RKT_STAGE1_SETUP_KIND],[flavor],
                           dnl we have flavor setup, checks can
                           dnl continue
                           [:],
                           [rkt_functional_tests_msg='Currently functional tests cannot be run on 3rd party stage1 images'])])
      AS_VAR_IF([rkt_functional_tests_msg],[],
                dnl we have a default flavor, check some special
                dnl prerequisites for host flavor
                [AS_VAR_IF([RKT_STAGE1_DEFAULT_FLAVOR], [host],
                           dnl checking if we are running systemd as init, -.mount is a mount unit for /.
                           [AS_IF([systemctl 2>/dev/null | grep --silent -e '-\.mount'],
                                  dnl systemd runs as init
                                  [sdv=`systemctl --version | head -1 | sed -e 's/^systemd \(@<:@0-9@:>@*\)$/\1/'`
                                   AS_IF([test ${sdv} -lt 227],
                                         [rkt_functional_tests_msg="Cannot run functional tests with ${RKT_STAGE1_DEFAULT_FLAVOR} stage1 flavor - systemd version on host is lower than 227 (currently running ${sdv})"])],
                                  dnl systemd is not an init
                                  [rkt_functional_tests_msg="Cannot run functional tests with ${RKT_STAGE1_DEFAULT_FLAVOR} stage1 flavor - no systemd on host"])])])
      dnl gpg is required for functional tests
      AS_VAR_IF([GPG],[gpg],
                dnl gpg found, alright
                [:],
                dnl gpg not found, running functional tests is impossible
                [rkt_functional_tests_msg='Cannot run functional tests - no gpg found'])
      AS_VAR_IF([rkt_functional_tests_msg], [],
                 dnl no message, tests can be run
                 [RKT_RUN_FUNCTIONAL_TESTS=yes
                  AC_MSG_RESULT([yes])],
                 dnl message available, tests will not be run
                 [AC_MSG_RESULT([no])
                  AS_VAR_IF([RKT_RUN_FUNCTIONAL_TESTS], [auto],
                            dnl functional tests are optional, so only print an info
                            [RKT_RUN_FUNCTIONAL_TESTS=no
                             AC_MSG_NOTICE([${rkt_functional_tests_msg}])],
                            dnl functional tests are required, so bail out
                            [AC_MSG_ERROR([*** ${rkt_functional_tests_msg}])])])])


## Miscellaneous stuff

# Check if arch is supported for coreos/kvm flavor.
AS_CASE([${RKT_ACI_ARCH}],
        dnl supported arches
        [aarch64 | amd64], [],
        dnl unsupported arches
        [RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [coreos,kvm,src],
                           [AC_MSG_ERROR([*** ${RKT_ACI_ARCH}: no arch support for stage1 flavors {coreos,kvm,src}])],
                           [AC_MSG_WARN([* ${RKT_ACI_ARCH}: unsupported arch])])])

# Set the path to the interpreter binary for coreos flavor.
RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [coreos],
                  [AS_CASE([${RKT_ACI_ARCH}],
                           [aarch64],
                                    [RKT_STAGE1_INTERPRETER="/usr/lib/ld-linux-aarch64.so.1"],
                           [amd64],
                                    [RKT_STAGE1_INTERPRETER="/usr/lib/ld-linux-x86-64.so.2"])])

# Set the path to the interpreter binary for src flavor.
RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [src],
                  [AS_CASE([${RKT_ACI_ARCH}],
                           [aarch64],
                                    [RKT_STAGE1_INTERPRETER="/lib64/ld-linux-aarch64.so.1"],
                           [amd64],
                                    [RKT_STAGE1_INTERPRETER="/lib64/ld-linux-x86-64.so.2"])])


# cross_compiling is a standard autoconf variable.  See the autoconf manual for
# details.
AS_VAR_IF([cross_compiling], [no],
          dnl no crosscompilation
          [GOARCH_FOR_BUILD=${GOARCH_FOR_BUILD:-${GOARCH}}],
          dnl we do crosscompilation
          [AC_MSG_CHECKING([build golang arch])
           AS_VAR_IF([GOARCH_FOR_BUILD], [],
                     [AS_CASE([${build_cpu}],
                              [x86_64],
                                      [GOARCH_FOR_BUILD="amd64"],
                              [aarch64],
                                      [GOARCH_FOR_BUILD="arm64"],
                              [powerpc | powerpc64],
                                      [GOARCH_FOR_BUILD="ppc64"],
                              [powerpc64le],
                                      [GOARCH_FOR_BUILD="ppc64le"],
                              [AC_MSG_RESULT([unknown]); AC_MSG_ERROR([*** Unknown build cpu: ${build_cpu}.  Set GOARCH_FOR_BUILD variable.])])
                      AC_MSG_RESULT([${GOARCH_FOR_BUILD}])],
                     [AC_MSG_RESULT([user supplied ${GOARCH_FOR_BUILD}])])])

AC_LANG_PUSH([C])

# check for libc generally
AC_CHECK_LIB([c], [fork],
             dnl libc is there
             [:],
             dnl libc is not there
             [AC_MSG_ERROR([*** No libc found. Try to install glibc-devel or libc6-dev.])])

# Check whether we can build TPM support code
AC_CHECK_HEADER(trousers/tss.h, [HAVE_TPM=yes], [HAVE_TPM=no], [AC_INCLUDES_DEFAULT])

TPM_TAGS=''
AS_CASE([${RKT_ENABLE_TPM}],
        [no],
                dnl nothing to do
                [:],
        [auto],
                [AS_VAR_IF([HAVE_TPM],[yes],
                           [TPM_TAGS=tpm
                            RKT_ENABLE_TPM=yes],
                           [RKT_ENABLE_TPM=no])],
        [yes],
                [AS_VAR_IF([HAVE_TPM],[no],
                           [AC_MSG_ERROR([*** TPM is enabled, but could not find required development files])])
                 TPM_TAGS=tpm],
        [AC_MSG_ERROR([*** Invalid value passed to --enable-tpm, should be either 'yes', 'no' or 'auto'])])
AS_CASE([${RKT_ENABLE_TPM}],
        [no],
                [RKT_ADD_FEATURE '-TPM'],
        [yes],
                [RKT_ADD_FEATURE '+TPM'])

# Check whether we have libsystemd-journal (for go-systemd/sdjournal)
AC_CHECK_HEADER([systemd/sd-journal.h], [HAVE_SDJOURNAL=yes], [HAVE_SDJOURNAL=no], [AC_INCLUDES_DEFAULT])

SDJOURNAL_TAGS=''
AS_CASE([${RKT_ENABLE_SDJOURNAL}],
        [no],
                dnl nothing to do
                [:],
        [auto],
                [AS_VAR_IF([HAVE_SDJOURNAL],[yes],
                           [SDJOURNAL_TAGS=sdjournal
                            RKT_ENABLE_SDJOURNAL=yes],
                           [RKT_ENABLE_SDJOURNAL=no])],
        [yes],
                [AS_VAR_IF([HAVE_SDJOURNAL],[no],
                           [AC_MSG_ERROR([*** Host systemd-journal support enabled, but could not find required development files])])
                 SDJOURNAL_TAGS=sdjournal],
        [AC_MSG_ERROR([*** Invalid value passed to --enable-sdjournal, should be either 'yes', 'no' or 'auto'])])
AS_CASE([${RKT_ENABLE_SDJOURNAL}],
        [no],
                [RKT_ADD_FEATURE '-SDJOURNAL'],
        [yes],
                [RKT_ADD_FEATURE '+SDJOURNAL'])

AC_LANG_POP([C])

AC_PROG_INSTALL
RKT_REQ_PROG([FILE],[file],[file])
RKT_REQ_PROG([GOBINARY],[go],[go])
RKT_REQ_PROG([GOFMTBINARY],[gofmt],[gofmt])
RKT_REQ_ABS_PROG([ABS_GO], [go])

#### LINKER FLAGS

dnl Digits regexp class. Square brackets are used by m4 for quoting,
dnl so to get literal square brackets, m4 provides ugly @<:@ and @:>@
dnl for [ and ].
m4_define([DIGITS],[@<:@0-9@:>@])
m4_define([ALNUM],[@<:@a-z0-9@:>@])

dnl Detect go version. Successfully parse versions like "go1.5",
dnl "go1.5.3", "go1.6rc1", "devel +66330d8 Wed Jan 13 23:40:13 2016
dnl +0000".
GO_VERSION_OUTPUT=`"${ABS_GO}" version`
GO_VERSION=`AS_ECHO(["${GO_VERSION_OUTPUT}"]) | sed -e 's/^go version \(.*\) ALNUM*\/ALNUM*$/\1/'`
AS_CASE([$GO_VERSION],
        [devel*],
                [AC_MSG_WARN([* Using development version of go, all bets are off])],
        [go*],
                [GO_VERSION=`AS_ECHO(["${GO_VERSION}"]) | sed -e 's/^go\(.*\)/\1/'`
                 GO_MAJOR=`AS_ECHO(["${GO_VERSION}"]) | grep -o '^DIGITS\+'`
                 GO_MINOR=`AS_ECHO(["${GO_VERSION}"]) | grep -o '^DIGITS\+\.DIGITS\+' | grep -o 'DIGITS\+$'`
                 GO_MICRO=`AS_ECHO(["${GO_VERSION}"]) | grep -o '^DIGITS\+\.DIGITS\+\.DIGITS\+' | sed -e 's/^DIGITS*\.DIGITS*\.\(DIGITS*\).*/\1/'`
                 AS_VAR_IF([GO_MICRO],[],
                           [GO_MICRO=0])

                 GO_BEST_MAJOR=1
                 GO_BEST_MINOR=5
                 AC_MSG_CHECKING([whether we have go ${GO_BEST_MAJOR}.${GO_BEST_MINOR} or newer])
                 AS_IF([test "${GO_MAJOR}" -gt "${GO_BEST_MAJOR}" || test "${GO_MAJOR}" -eq "${GO_BEST_MAJOR}" -a "${GO_MINOR}" -ge "${GO_BEST_MINOR}"],
                       [AC_MSG_RESULT([yes])],
                       [AC_MSG_RESULT([no])
                        AC_MSG_ERROR([*** go is too old (${GO_VERSION})])])

                 AC_MSG_CHECKING([whether we have a go version without CVE-2015-8618])
                 AS_IF([test "${GO_MAJOR}" -eq "1" -a "${GO_MINOR}" -eq "5" -a "${GO_MICRO}" -lt "3"],
                       [AC_MSG_RESULT([no])
                        insecure_go_warning="go version is vulnerable to CVE-2015-8618 (${GO_VERSION})"
                        AS_VAR_IF([RKT_INSECURE_GO],[no],
                                  [AC_MSG_ERROR([*** ${insecure_go_warning}])],
                                  [AC_MSG_WARN([* ${insecure_go_warning}])])],
                       [AC_MSG_RESULT([yes])])],
        [AC_MSG_ERROR([*** Could not parse the output of go version])])
RKT_STAGE1_DEFAULT_NAME_LDFLAGS="-X 'main.buildDefaultStage1Name=${RKT_STAGE1_DEFAULT_NAME}'"
RKT_STAGE1_DEFAULT_VERSION_LDFLAGS="-X 'main.buildDefaultStage1Version=${RKT_STAGE1_DEFAULT_VERSION}'"
RKT_STAGE1_DEFAULT_LOCATION_LDFLAGS="-X 'main.buildDefaultStage1ImageLoc=${RKT_STAGE1_DEFAULT_LOCATION}'"
RKT_STAGE1_DEFAULT_IMAGE_FILENAME_LDFLAGS="-X 'main.buildDefaultStage1ImageInRktDir=${RKT_STAGE1_DEFAULT_IMAGE_FILENAME_IN_RKT_DIRECTORY}'"
RKT_STAGE1_DEFAULT_IMAGES_DIRECTORY_LDFLAGS="-X 'main.buildDefaultStage1ImagesDir=${RKT_STAGE1_DEFAULT_IMAGES_DIR}'"
RKT_STAGE1_INTERPRETER_LDFLAGS="-X 'main.interpBin=${RKT_STAGE1_INTERPRETER}'"
RKT_VERSION_LDFLAGS="-X 'github.com/coreos/rkt/version.Version=${RKT_VERSION}'"
RKT_FEATURES_LDFLAGS="-X 'main.rktFeatures=${RKT_FEATURES}'"

#### SUBSTITUTIONS

AC_SUBST(RKT_VERSION)
AC_SUBST(RKT_ACI_ARCH)
AC_SUBST(RKT_STAGE1_FLAVORS_VERSION_OVERRIDE)

AC_SUBST(RKT_STAGE1_DEFAULT_NAME)
AC_SUBST(RKT_STAGE1_DEFAULT_VERSION)

AC_SUBST(RKT_STAGE1_ALL_FLAVORS)
AC_SUBST(RKT_STAGE1_FLAVORS)
AC_SUBST(RKT_STAGE1_DEFAULT_FLAVOR)

AC_SUBST(RKT_STAGE1_DEFAULT_LOCATION)
AC_SUBST(RKT_STAGE1_DEFAULT_IMAGES_DIR)
AC_SUBST(RKT_STAGE1_DEFAULT_IMAGE_FILENAME_IN_RKT_DIRECTORY)

AC_SUBST(RKT_STAGE1_SYSTEMD_SRC)
AC_SUBST(RKT_STAGE1_SYSTEMD_VER)
AC_SUBST(RKT_STAGE1_SYSTEMD_REV)

AC_SUBST(RKT_LOCAL_COREOS_PXE_IMAGE_PATH)
AC_SUBST(RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER)

AC_SUBST(RKT_STAGE1_COREOS_BOARD)
AC_SUBST(RKT_STAGE1_INTERPRETER)

AC_SUBST(RKT_RUN_FUNCTIONAL_TESTS)
AC_SUBST(INCREMENTAL_BUILD)

AC_SUBST(GOARCH)
AC_SUBST(GOARM)
AC_SUBST(GOARCH_FOR_BUILD)

AC_SUBST(RKT_STAGE1_DEFAULT_NAME_LDFLAGS)
AC_SUBST(RKT_STAGE1_DEFAULT_VERSION_LDFLAGS)
AC_SUBST(RKT_STAGE1_DEFAULT_LOCATION_LDFLAGS)
AC_SUBST(RKT_STAGE1_DEFAULT_IMAGE_FILENAME_LDFLAGS)
AC_SUBST(RKT_STAGE1_DEFAULT_IMAGES_DIRECTORY_LDFLAGS)
AC_SUBST(RKT_STAGE1_INTERPRETER_LDFLAGS)
AC_SUBST(RKT_VERSION_LDFLAGS)
AC_SUBST(RKT_FEATURES_LDFLAGS)

AC_SUBST(RKT_STAGE1_KVM_ALL_HV)
AC_SUBST(RKT_STAGE1_KVM_HV)
AC_SUBST(TPM_TAGS)
AC_SUBST(SDJOURNAL_TAGS)

AC_SUBST(RKT_DEFINES_FOR_ENTER)

#### FILE GENERATION AND REPORTING

AC_CONFIG_FILES([Makefile])
AC_OUTPUT

AC_MSG_RESULT([
        AC_PACKAGE_NAME ${RKT_VERSION}

        stage1 setup

        type:                                   '${RKT_STAGE1_SETUP_KIND}'
        default stage1 location:                '${RKT_STAGE1_DEFAULT_LOCATION}'
        default stage1 images directory:        '${RKT_STAGE1_DEFAULT_IMAGES_DIR}'])

AS_CASE([${RKT_STAGE1_SETUP_KIND}],
        [flavor],
                [AC_MSG_RESULT([
        built stage1 flavors:                   '${RKT_STAGE1_FLAVORS}'
        default stage1 flavor:                  '${RKT_STAGE1_DEFAULT_FLAVOR}'
        implied default stage1 name:            '${RKT_STAGE1_DEFAULT_NAME}'])
                 AS_VAR_IF([RKT_STAGE1_FLAVORS_VERSION_OVERRIDE], [],
                           [AC_MSG_RESULT([dnl
        implied default stage1 version:         '${RKT_STAGE1_DEFAULT_VERSION}'])],
                           [AC_MSG_RESULT([dnl
        overridden default stage1 version:      '${RKT_STAGE1_DEFAULT_VERSION}'])])],
        [detailed],
                [AC_MSG_RESULT([
        default stage1 name:                    '${RKT_STAGE1_DEFAULT_NAME}'
        default stage1 version:                 '${RKT_STAGE1_DEFAULT_VERSION}'])],
        [AC_MSG_ERROR([*** Unknown setup kind, should not happen])])

RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [coreos,kvm],
                  [AC_MSG_RESULT([
        coreos/kvm flavor specific build parameters

        local CoreOS PXE image path:            '${RKT_LOCAL_COREOS_PXE_IMAGE_PATH}'
        local CoreOS PXE image systemd version: '${RKT_LOCAL_COREOS_PXE_IMAGE_SYSTEMD_VER}'
        stage1 CoreOS board:                    '${RKT_STAGE1_COREOS_BOARD}'])])

RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [src],
                  [AC_MSG_RESULT([
        src flavor specific build parameters

        systemd git repo:                       '${RKT_STAGE1_SYSTEMD_SRC}'
        systemd git revision:                   '${RKT_STAGE1_SYSTEMD_REV}'
        systemd version:                        '${RKT_STAGE1_SYSTEMD_VER}'])])

RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [kvm],
                  [AC_MSG_RESULT([
        kvm flavor specific build parameters

        hypervisors:                            '${RKT_STAGE1_KVM_HV}'])])

RKT_IF_HAS_VALUE([${RKT_STAGE1_FLAVORS}], [coreos,src],
                  [AC_MSG_RESULT([
        coreos flavor specific build parameters

        interpreter:                            '${RKT_STAGE1_INTERPRETER}'])])


AC_MSG_RESULT([
        other build parameters

        functional tests enabled:               '${RKT_RUN_FUNCTIONAL_TESTS}'
        incremental build:                      '${INCREMENTAL_BUILD}'
        features:                               '${RKT_FEATURES}'
        ACI arch:                               '${RKT_ACI_ARCH}'
        go version:                             '${GO_VERSION}'
        go arch:                                '${GOARCH}'])
