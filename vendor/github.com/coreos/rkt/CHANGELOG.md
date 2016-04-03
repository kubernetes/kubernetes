## v1.2.1

This release fixes a couple of bugs we missed in 1.2.0.

#### Bug fixes

- Do not error out if `/dev/ptmx` or `/dev/log` exist ([#2302](https://github.com/coreos/rkt/pull/2302)).
- Vendor a release of go-systemd instead of current master ([#2306](https://github.com/coreos/rkt/pull/2306)).

## v1.2.0

This release is an incremental release with numerous bug fixes.

#### New features and UX changes

- Add `--hostname` option to rkt run/run-prepared ([#2251](https://github.com/coreos/rkt/pull/2251)). This option allows setting the pod host name.

#### Bug fixes

- Fix deadlock while exiting a lkvm rkt pod ([#2191](https://github.com/coreos/rkt/pull/2191)).
- SELinux fixes preparating rkt to work on Fedora with SELinux enabled ([#2247](https://github.com/coreos/rkt/pull/2247) and [#2262](https://github.com/coreos/rkt/pull/2262)).
- Fix bug that occurs for some types of on-disk image corruption, making it impossible for the user run or garbage collect them ([#2180](https://github.com/coreos/rkt/issues/2180)).
- Fix authentication issue when fetching from a private quay.io repository ([#2248](https://github.com/coreos/rkt/issues/2248)).
- Allow concurrent image fetching ([#2239](https://github.com/coreos/rkt/pull/2239)).
- Fix issue mounting volumes on images if the target path includes an absolute symlink ([#2290](https://github.com/coreos/rkt/pull/2290)).
- Clean up dangling symlinks in `/var/log/journal` on garbage collection if running on systemd hosts ([#2289](https://github.com/coreos/rkt/pull/2289)).

#### Note for 3rd party stage1 builders

- The stage1 command line interface is versioned now. See the [implementors guide](https://github.com/coreos/rkt/blob/master/Documentation/devel/stage1-implementors-guide.md) for more information.

## v1.1.0

This release is the first incremental release since 1.0. It includes bugfixes and some UX improvements.

#### New features and UX changes

- Add support for non-numerical UID/GID as specified in the appc spec ([#2159](https://github.com/coreos/rkt/pull/2159)). rkt can now start apps as the user and group specified in the [image manifest](https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema) with three different possible formats: a numeric UID/GID, a username and group name referring to the ACI's /etc/passwd and /etc/group, or a file path in the ACI whose owner will determine the UID/GID.
- When an application terminates with a non-zero exit status, `rkt run` should return that exit status ([#2198](https://github.com/coreos/rkt/pull/2198)). This is now fixed in the [src and host flavors](https://github.com/coreos/rkt/blob/master/Documentation/build-configure.md#--with-stage1-flavors) with [systemd >= v227](https://lists.freedesktop.org/archives/systemd-devel/2015-October/034509.html) but not yet in the shipped coreos flavor.
- Use exit status 2 to report usage errors ([#2149](https://github.com/coreos/rkt/pull/2149)).
- Add support for tuning pod's network via the [CNI tuning plugin](https://github.com/appc/cni/blob/master/Documentation/tuning.md) ([#2140](https://github.com/coreos/rkt/pull/2140)). For example, this allows increasing the size of the listen queue for accepting new TCP connections (`net.core.somaxconn`) in the rkt pod.
- Keep $TERM from the host when entering a pod ([#1962](https://github.com/coreos/rkt/pull/1962)). This fixes the command "clear" which previously was not working.

#### Bug fixes
- Socket activation was not working if the port on the host is different from the app port as set in the image manifest ([#2137](https://github.com/coreos/rkt/pull/2137)).
- Fix an authentication failure when fetching images from private repositories in the official Docker registry ([#2197](https://github.com/coreos/rkt/pull/2197)).
- Set /etc/hostname in kvm pods ([#2190](https://github.com/coreos/rkt/pull/2190)).

## v1.0.0

This marks the first release of rkt recommended for use in production.
The command-line UX and on-disk format are considered stable and safe to develop against.
Any changes to these interfaces will be backwards compatible and subject to formal deprecation.
The API is not yet completely stabilized, but is functional and suitable for use by early adopters.

#### New features and UX changes

- Add pod creation and start times to `rkt list` and `rkt status` ([#2030](https://github.com/coreos/rkt/pull/2030)). See [`rkt list`](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/list.md) and [`rkt status`](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/status.md) documentation.
- The DNS configuration can now be passed to the pod via the command line ([#2040](https://github.com/coreos/rkt/pull/2040)). See [`DNS support`](https://github.com/coreos/rkt/blob/master/Documentation/networking.md#dns-support) documentation.
- Errors are now structured, allowing for better control of the output ([#1937](https://github.com/coreos/rkt/pull/1937)). See [Error & Output](https://github.com/coreos/rkt/blob/master/Documentation/hacking.md#errors--output) for how a developer should use it.
- All output now uses the new log package in `pkg/log` to provide a more clean and consistent output format and more helpful debug output ([#1937](https://github.com/coreos/rkt/pull/1937)).
- Added configuration for stage1 image. Users can drop a configuration file to `/etc/rkt/stage1.d` (or to `stage1.d` in the user configuration directory) to tell rkt to use a different stage1 image name, version and location instead of build-time defaults ([#1977](https://github.com/coreos/rkt/pull/1977)).
- Replaced the `--stage1-image` flag with a new set of flags. `--stage1-url`, `--stage-path`, `--stage1-name` do the usual fetching from remote if the image does not exist in the store. `--stage1-hash` takes the stage1 image directly from the store. `--stage1-from-dir` works together with the default stage1 images directory and is described in the next point ([#1977](https://github.com/coreos/rkt/pull/1977)).
- Added default stage1 images directory. User can use the newly added `--stage1-from-dir` parameter to avoid typing the full path. `--stage1-from-dir` behaves like `--stage1-path` ([#1977](https://github.com/coreos/rkt/pull/1977)).
- Removed the deprecated `--insecure-skip-verify` flag ([#2068](https://github.com/coreos/rkt/pull/2068)).
- Fetched keys are no longer automatically trusted by default, unless `--trust-keys-from-https` is used. Additionally, newly fetched keys have to be explicitly trusted with `rkt trust` if a previous key was trusted for the same image prefix ([#2033](https://github.com/coreos/rkt/pull/2033)).
- Use NAT loopback to make ports forwarded in pods accessible from localhost ([#1256](https://github.com/coreos/rkt/issues/1256)).
- Show a clearer error message when unprivileged users execute commands that require root privileges ([#2081](https://github.com/coreos/rkt/pull/2081)).
- Add a rkt tmpfiles configuration file to make the creation of the rkt data directory on first boot easier ([#2088](https://github.com/coreos/rkt/pull/2088)).
- Remove `rkt install` command. It was replaced with a `setup-data-dir.sh` script ([#2101](https://github.com/coreos/rkt/pull/2101).

#### Bug fixes

- Fix regression when authenticating to v2 Docker registries ([#2008](https://github.com/coreos/rkt/issues/2008)).
- Don't link to libacl, but dlopen it ([#1963](https://github.com/coreos/rkt/pull/1963)). This means that rkt will not crash if libacl is not present on the host, but it will just print a warning.
- Only suppress diagnostic messages, not error messages in stage1 ([#2111](https://github.com/coreos/rkt/pull/2111)).

#### Other changes

- Trusted Platform Module logging (TPM) is now enabled by default ([#1815](https://github.com/coreos/rkt/issues/1815)). This ensures that rkt benefits from security features by default. See rkt's [Build Configuration](https://github.com/coreos/rkt/blob/master/Documentation/build-configure.md#security) documentation.
- Added long descriptions to all rkt commands ([#2098](https://github.com/coreos/rkt/issues/2098)).

#### Migration

- The `--stage1-image` flag was removed. Scripts using it should be updated to use one of `--stage1-url`, `--stage1-path`, `--stage1-name`, `--stage1-hash` or `--stage1-from-dir`
- All uses of the deprecated `--insecure-skip-verify` flag should be replaced with the `--insecure-options` flag which allows user to selectively disable security features.
- The `rkt install` command was removed in favor of the `dist/scripts/setup-data-dir.sh` script.

#### Note for packagers

With this release, `rkt` RPM/dpkg packages should have the following updates:

- Pass `--enable-tpm=no` to configure script, if `rkt` should not use TPM.
- Use the `--with-default-stage1-images-directory` configure flag, if the default is not acceptable and install the built stage1 images there.
- Distributions using systemd: install the new file `dist/init/systemd/tmpfiles.d/rkt.conf` in `/usr/lib/tmpfiles.d/rkt.conf` and then run `systemd-tmpfiles --create rkt.conf`. This can replace running `rkt install` to set the correct ownership and permissions.

## v0.16.0

#### New features and UX changes

- Explicitly allow http connections via a new 'http' option to `--insecure-options` ([#1945](https://github.com/coreos/rkt/pull/1945)). Any data and credentials will be sent in the clear.
- When using `bash`, `rkt` commands can be auto-completed ([#1955](https://github.com/coreos/rkt/pull/1955)).
- The executables given on the command line via the `--exec` parameters don't need to be absolute paths anymore ([#1953](https://github.com/coreos/rkt/pull/1953)). This change reflects an update in the appc spec since [v0.7.2](https://github.com/appc/spec/releases/tag/v0.7.2). See rkt's [rkt run --exec](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/run.md#overriding-executable-to-launch) documentation.
- Add a `--full` flag to rkt fetch so it returns full hash of the image ([#1976](https://github.com/coreos/rkt/pull/1976)).
- There is a new global flag for specifying the user configuration directory, `--user-config`. It overrides whatever is configured in system and local configuration directories. It can be useful for specifying different credentials for fetching images without putting them in a globally visible directory like `/etc/rkt`. See rkt's [Global Options](https://github.com/coreos/rkt/blob/master/Documentation/commands.md#global-options) documentation ([#1981](https://github.com/coreos/rkt/pull/1981)).
- As a temporary fix, search for network plugins in the local configuration directory too ([#2005](https://github.com/coreos/rkt/pull/2005)).
- Pass the environment defined in the image manifest to the application when using the fly stage1 image ([#1989](https://github.com/coreos/rkt/pull/1989)).

#### Build improvements

- Fix vagrant rkt build ([#1960](https://github.com/coreos/rkt/pull/1960)).
- Switch to using unrewritten imports, this will allow rkt packages to be cleanly vendored by other projects ([#2014](https://github.com/coreos/rkt/pull/2014)).

#### API service

- Allow filtering images by name ([#1985](https://github.com/coreos/rkt/pull/1985)).

#### Bug fixes

- Fix bug where the wrong image signature was checked when using dependencies ([#1991](https://github.com/coreos/rkt/pull/1991)).

#### Test improvements

- A new script to run test on AWS makes it easier to test under several distributions: CentOS, Debian, Fedora, Ubuntu ([#1925](https://github.com/coreos/rkt/pull/1925)).
- The functional tests now skip user namespace tests when user namespaces do not work ([#1947](https://github.com/coreos/rkt/pull/1947)).
- Check that rkt is not built with go 1.5.{0,1,2} to make sure it's not vulnerable to CVE-2015-8618 ([#2006](https://github.com/coreos/rkt/pull/2006)).

#### Other changes

- Cleanups in the kvm stage1 ([#1895](https://github.com/coreos/rkt/pull/1895)).
- Document stage1 filesystem layout for developers ([#1832](https://github.com/coreos/rkt/pull/1832)).

#### Note for packagers

With this release, `rkt` RPM/dpkg packages should have the following updates:

- Install the new file `dist/bash_completion/rkt.bash` in `/etc/bash_completion.d/`.

## v0.15.0

rkt v0.15.0 is an incremental release with UX improvements, bug fixes, API service enhancements and new support for Go 1.5.

#### New features and UX changes

- Images can now be deleted from the store by both ID and name ([#1866](https://github.com/coreos/rkt/pull/1866)). See rkt's [rkt image rm](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/image.md#rkt-image-rm) documentation.
- The journals of rkt pods can now be accessed by members of the Unix group rkt ([#1877](https://github.com/coreos/rkt/pull/1877)). See rkt's [journalctl -M](https://github.com/coreos/rkt/blob/master/Documentation/using-rkt-with-systemd.md#journalctl--m) documentation.

#### Improved documentation

- Mention [rkt integration in Nomad](https://github.com/coreos/rkt/blob/master/Documentation/using-rkt-with-nomad.md) ([#1884](https://github.com/coreos/rkt/pull/1884)).
- Document [how to start the api service](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/api-service.md) and add a [program example](https://github.com/coreos/rkt/blob/master/api/v1alpha/client_example.go) explaining how the api service can be used to integrate rkt with other programs ([#1915](https://github.com/coreos/rkt/pull/1915)).

#### API service

- Programs using rkt's API service are now provided with the size of the images stored in rkt's store ([#1916](https://github.com/coreos/rkt/pull/1916)).
- Programs using rkt's API service are now provided with any annotations found in the [image manifest](https://github.com/appc/spec/blob/master/spec/aci.md#image-manifest-schema) and [pod manifest](https://github.com/appc/spec/blob/master/spec/pods.md#pod-manifest-schema) ([#1924](https://github.com/coreos/rkt/pull/1924)).
- Fix a panic in the API service by making the store database thread-safe ([#1892](https://github.com/coreos/rkt/pull/1892)) and by refactoring the API service functions to get the pod state ([#1893](https://github.com/coreos/rkt/pull/1893)).

#### Build improvements

- Add support for building rkt with Go 1.5, which is now the preferred version. rkt can still be built with Go 1.4 as best effort ([#1907](https://github.com/coreos/rkt/pull/1907)). As part of the move to Go 1.5, rkt now has a godep-save script to support Go 1.5 ([#1857](https://github.com/coreos/rkt/pull/1857)).
- Continuous Integration on Travis now builds with both Go 1.4.2 and Go 1.5.2. Go 1.4.3 is avoided to workaround recent problems with go vet ([#1941](https://github.com/coreos/rkt/pull/1941)).

#### Bug fixes

- Fix regression issue when downloading image signatures from quay.io ([#1909](https://github.com/coreos/rkt/pull/1909)).
- Properly cleanup the tap network interface that were not cleaned up in some error cases when using the kvm stage1 ([#1921](https://github.com/coreos/rkt/pull/1921)).
- Fix a bug in the 9p filesystem used by the kvm stage1 that were preventing `apt-get` from working propertly ([#1918](https://github.com/coreos/rkt/pull/1918)).

## v0.14.0

rkt v0.14.0 brings new features like resource isolators in the kvm stage1, a new stage1 flavor called *fly*, bug fixes and improved documentation.
The appc spec version has been updated to v0.7.4

#### New features and UX changes

- The data directory that rkt uses can now be configured with a config file ([#1806](https://github.com/coreos/rkt/pull/1806)). See rkt's [paths configuration](https://github.com/coreos/rkt/blob/master/Documentation/configuration.md#rktkind-paths) documentation.
- CPU and memory resource isolators can be specified on the command line to override the limits specified in the image manifest ([#1851](https://github.com/coreos/rkt/pull/1851), [#1874](https://github.com/coreos/rkt/pull/1874)). See rkt's [overriding isolators](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/run.md#overriding-isolators) documentation.
- CPU and memory resource isolators can now be used within the kvm stage1 ([#1404](https://github.com/coreos/rkt/pull/1404))
- The `rkt image list` command can now display the image size ([#1865](https://github.com/coreos/rkt/pull/1865)).
- A new stage1 flavor has been added: fly; and it represents the first experimental implementation of the upcoming rkt fly feature. ([#1833](https://github.com/coreos/rkt/pull/1833))

#### Build improvements

- It is now possible to build rkt inside rkt ([#1681](https://github.com/coreos/rkt/pull/1681)). This should improve the reproducibility of builds. This release does not use it yet but it is planned for future releases.
- Linux distribution packagers can override the version of stage1 during the build ([#1821](https://github.com/coreos/rkt/pull/1821)). This is needed for any Linux distributions that might carry distro-specific patches along the upstream release. See rkt's documentation about [building stage1 flavors](https://github.com/coreos/rkt/blob/master/Documentation/build-configure.md#--with-stage1-flavors-version-override).
- Smaller build improvements with dep generation ([#1838](https://github.com/coreos/rkt/pull/1838)), error messages on `make clean` ([#1850](https://github.com/coreos/rkt/pull/1850)), dependency checks in the kvm flavor ([#1860](https://github.com/coreos/rkt/pull/1860))

#### Bug fixes

- rkt is now able to override the application command with `--exec` when the application manifest didn't specify any command ([#1843](https://github.com/coreos/rkt/pull/1843)).
- In some cases, user namespaces were not working in Linux distributions without systemd, such as Ubuntu 14.04 LTS. This is fixed by creating a unique cgroup for each pod when systemd is not used ([#1844](https://github.com/coreos/rkt/pull/1844))
- rkt's tar package didn't prefix the destination file correctly when using hard links in images. This was not a issue in rkt itself but was causing acbuild to misbehave ([#1852](https://github.com/coreos/rkt/pull/1852)).
- ACIs with multiple dependencies can end up depending on the same base image through multiple paths. In some of those configuration with multiple dependencies, fetching the image via image discovery was not working. This is fixed and a new test ensures it will keep working ([#1822](https://github.com/coreos/rkt/pull/1822)).
- The pod cgroups were misconfigured when systemd-devel is not installed. This was causing per-app CPU and memory isolators to be ineffective on those systems. This is now fixed but will require an additional fix for NixOS ([#1873](https://github.com/coreos/rkt/pull/1873)).
- During the garbage collection of pods (`rkt gc`), all mounts will be umounted even when the pod is in an inconsistent state ([#1828](https://github.com/coreos/rkt/pull/1828), [#1856](https://github.com/coreos/rkt/pull/1856))

#### Improved documentation

- New documentation about configure flags ([#1824](https://github.com/coreos/rkt/pull/1824)). This also includes formatting and typos fixes and updates. The examples about rkt's configuration files are also clarified ([#1847](https://github.com/coreos/rkt/pull/1847)).
- New documentation explaining [how cgroups are used by rkt](https://github.com/coreos/rkt/blob/master/Documentation/devel/cgroups.md) ([#1870](https://github.com/coreos/rkt/pull/1870)). This should make it easier for software developers to integrate rkt with monitoring software.

#### API service

- The API service is meant to be used by orchestration tools like Kubernetes. The performance of the API service was improved by reducing the round-trips in the ListPods and ListImages requests ([#1786](https://github.com/coreos/rkt/pull/1786)). Those requests also gained multiple filters for more flexibility ([#1853](https://github.com/coreos/rkt/pull/1853)).

## v0.13.0

The primary motivation for this release is to add support for fetching images on the Docker Registry 2.0. It also includes other small improvements.

- docker2aci: support Docker Registry 2.0 ([#1826](https://github.com/coreos/rkt/pull/1826))
- always use https:// when fetching docker images ([#1837](https://github.com/coreos/rkt/pull/1837))
- stage0: add container hash data into TPM ([#1775](https://github.com/coreos/rkt/pull/1775))
- host flavor: fix systemd copying into stage1 for Debian packaging ([#1811](https://github.com/coreos/rkt/pull/1811))
- clarify network error messages ([#1707](https://github.com/coreos/rkt/pull/1707))
- documentation: add more build-time requirements ([#1834](https://github.com/coreos/rkt/pull/1834))

## v0.12.0

rkt v0.12.0 is an incremental release with UX improvements like fine-grained security controls and implicit generation of empty volumes, performance improvements, bug fixes and testing enhancements.

#### New features and UX changes

- implement `rkt cat-manifest` for pods ([#1744](https://github.com/coreos/rkt/pull/1744))
- generate an empty volume if a required one is not provided ([#1753](https://github.com/coreos/rkt/pull/1753))
- make disabling security features granular; `--insecure-skip-verify` is now `--insecure-options={feature(s)-to-disable}` ([#1738](https://github.com/coreos/rkt/pull/1738)). See rkt's [Global Options](https://github.com/coreos/rkt/blob/master/Documentation/commands.md#global-options) documentation.
- allow skipping the on-disk integrity check using `--insecure-options=ondisk`. This greatly speeds up start time. ([#1804](https://github.com/coreos/rkt/pull/1804))
- set empty volumes' permissions following the [spec](https://github.com/appc/spec/blob/master/spec/pods.md#pod-manifest-schema) ([1803](https://github.com/coreos/rkt/pull/1803))
- flannel networking support in kvm flavor ([#1563](https://github.com/coreos/rkt/pull/1563))

#### Bug fixes

- store used MCS contexts on the filesystem ([#1742](https://github.com/coreos/rkt/pull/1742))
- fix Docker images with whiteout-ed hard links ([#1653](https://github.com/coreos/rkt/pull/1653))
- fix Docker images relying on /dev/stdout ([#1617](https://github.com/coreos/rkt/pull/1617))
- use authentication for discovery and trust ([#1801](https://github.com/coreos/rkt/pull/1801))
- fix build in Docker ([#1798](https://github.com/coreos/rkt/pull/1798))
- fix kvm networking ([#1530](https://github.com/coreos/rkt/pull/1530))

#### Improved testing

- add functional tests for rkt api service ([#1761](https://github.com/coreos/rkt/pull/1761))
- fix TestSocketActivation on systemd-v219 ([#1768](https://github.com/coreos/rkt/pull/1768))
- fix the ACE validator test ([#1802](https://github.com/coreos/rkt/pull/1802))

#### Other changes

- Bumped appc spec to 0.7.3 ([#1800](https://github.com/coreos/rkt/pull/1800))

## v0.11.0

rkt v0.11.0 is an incremental release with mostly bug fixes and testing improvements.

#### New features and UX changes

- support resuming ACI downloads ([#1444](https://github.com/coreos/rkt/pull/1444))
- `rkt image gc` now also removes images from the store ([#1697](https://github.com/coreos/rkt/pull/1697))

#### Build

- handle building multiple flavors ([#1683](https://github.com/coreos/rkt/pull/1683))
- verbosity control ([#1685](https://github.com/coreos/rkt/pull/1685), [#1686](https://github.com/coreos/rkt/pull/1686))
- fix bugs in `make clean` ([#1695](https://github.com/coreos/rkt/pull/1695))

#### Improved testing

- nicer output in tests ([#1698](https://github.com/coreos/rkt/pull/1698))
- refactor test code ([#1709](https://github.com/coreos/rkt/pull/1709))
- skip CI tests when the source was not modified ([#1619](https://github.com/coreos/rkt/pull/1619))
- better output when tests fail ([#1728](https://github.com/coreos/rkt/pull/1728))
- fix tests in `10.*` IP range ([#1736](https://github.com/coreos/rkt/pull/1736))
- document how to run functional tests ([#1736](https://github.com/coreos/rkt/pull/1736))

#### Improved documentation

- add some help on how to run rkt as a daemon ([#1684](https://github.com/coreos/rkt/pull/1684))

#### API service

- do not return manifest in `ListPods()` and `ListImages()` ([#1688](https://github.com/coreos/rkt/pull/1688))

#### Bug fixes

- parameter `--mount` fixed in kvm flavour ([#1687](https://github.com/coreos/rkt/pull/1687))
- fix rkt leaking containers in machinectl on CoreOS ([#1694](https://github.com/coreos/rkt/pull/1694), [#1704](https://github.com/coreos/rkt/pull/1704))
- `rkt status` now returns the stage1 pid ([#1699](https://github.com/coreos/rkt/pull/1699))
- fix crash in `rkt status` when an image is removed ([#1701](https://github.com/coreos/rkt/pull/1701))
- fix fd leak in store ([#1716](https://github.com/coreos/rkt/pull/1716))
- fix exec line parsing in ACI manifest ([#1652](https://github.com/coreos/rkt/pull/1652))
- fix build on 32-bit systems ([#1729](https://github.com/coreos/rkt/pull/1729))

## v0.10.0

rkt v0.10.0 is an incremental release with numerous bug fixes and a few small new features and UX improvements.

#### New features and UX changes

- added implementation for basic API service (`rkt api-service`) ([#1508](https://github.com/coreos/rkt/pull/1508))
- mount arbitrary volumes with `--mount` ([#1582](https://github.com/coreos/rkt/pull/1582), [#1678](https://github.com/coreos/rkt/pull/1678))
- `--net=none` only exposes the loopback interface ([#1635](https://github.com/coreos/rkt/pull/1635))
- better formatting for rkt help ([#1597](https://github.com/coreos/rkt/pull/1597))
- metadata service registration (`--mds-register`) disabled by default ([#1635](https://github.com/coreos/rkt/pull/1635))

#### Improved documentation
- [compare rkt and other projects](https://github.com/coreos/rkt/blob/master/Documentation/rkt-vs-other-projects.md) ([#1588](https://github.com/coreos/rkt/pull/1588))
- [Stage 1 systemd Architecture](https://github.com/coreos/rkt/blob/master/Documentation/devel/architecture.md) ([#1631](https://github.com/coreos/rkt/pull/1631))
- [packaging rkt in Linux distributions](https://github.com/coreos/rkt/blob/master/Documentation/packaging.md) ([#1511](https://github.com/coreos/rkt/pull/1511))

#### Improved testing
- new test for user namespaces (`--private-users`) ([#1580](https://github.com/coreos/rkt/pull/1580))
- fix races in tests ([#1608](https://github.com/coreos/rkt/pull/1608))

#### Bug fixes
- suppress unnecessary output when `--debug` is not used ([#1557](https://github.com/coreos/rkt/pull/1557))
- fix permission of rootfs with overlayfs ([#1607](https://github.com/coreos/rkt/pull/1607))
- allow relative path in parameters ([#1615](https://github.com/coreos/rkt/pull/1615))
- fix pod garbage collection failure in some cases ([#1621](https://github.com/coreos/rkt/pull/1621))
- fix `rkt list` when an image was removed ([#1655](https://github.com/coreos/rkt/pull/1655))
- user namespace (`--private-users`) regression with rkt group fixed ([#1654](//github.com/coreos/rkt/pull/1654))

## v0.9.0

rkt v0.9.0 is a significant milestone release with a number of internal and user-facing changes.

There are several notable breaking changes from the previous release:
- The on-disk format for pod trees has changed slightly, meaning that `rkt gc` and `rkt run-prepared` may not work for pods created by previous versions of rkt. To work around this, we recommend removing the pods with an older version of rkt.
- The `--private-net` flag has been renamed to `--net` and its semantic has changed (in particular, it is now enabled by default) - see below for details.
- Several changes to CLI output (e.g. column names) from the `rkt list` and `rkt image list` subcommands.
- The image fetching behaviour has changed, with the introduction of new flags to `rkt run` and `rkt fetch` and the removal of `--local` - see below for details.

#### New features and UX changes

###### `--private-net` --> `--net`, and networking is now private by default
The `--private-net` flag has been changed to `--net`, and has been now made the default behaviour. ([#1532](https://github.com/coreos/rkt/pull/1532), [#1418](https://github.com/coreos/rkt/pull/1418))
That is, a `rkt run` command will now by default set up a private network for the pod.
To achieve the previous default behaviour of the pod sharing the networking namespace of the host, use `--net=host`.
The flag still allows the specification of multiple networks via CNI plugins, and overriding plugin configuration on a per-network basis.
For more details, see the [networking documentation](Documentation/networking.md).

###### New image fetching behaviour
When fetching images during `rkt fetch` or `rkt run`, rkt would previously behave inconsistently for different formats (e.g when performing discovery or when retrieving a Docker image) when deciding whether to use a cached version or not.
`rkt run` featured a `--local` flag to adjust this behaviour but it provided an unintuitive semantic and was not available to the `rkt fetch` command.
Instead, rkt now features two new flags, `--store-only` and `--no-store`, on both the `rkt fetch` and `rkt run` commands, to provide more consistent, controllable, and predictable behaviour regarding when images should be retrieved.
For full details of the new behaviour see the [image fetching documentation](Documentation/image-fetching-behavior.md).

###### Unprivileged users
A number of changes were made to the permissions of rkt's internal store to facilitate unprivileged users to access information about images and pods on the system ([#1542](https://github.com/coreos/rkt/pull/1542), [#1569](https://github.com/coreos/rkt/pull/1569)).
In particular, the set-group-ID bit is applied to the directories touched by `rkt install` so that the `rkt` group (if it exists on the system) can retain read-access to information about pods and images.
This will be used by the rkt API service (targeted for the next release) so that it can run as an unprivileged user on the system.
This support is still considered partially experimental.
Some tasks like `rkt image gc` remain a root-only operation.

###### /etc/hosts support
If no `/etc/hosts` exists in an application filesystem at the time it starts running, rkt will now provide a basic default version of this file.
If rkt detects one already in the app's filesystem (whether through being included in an image, or a volume mounted in), it will make no changes. ([#1541](https://github.com/coreos/rkt/pull/1541))

##### Other new features
- rkt now supports setting supplementary group IDs on processes ([#1514](https://github.com/coreos/rkt/pull/1514)).
- rkt's use of cgroups has been reworked to facilitate rkt running on a variety of operating systems like Void and older non-systemd distributions ([#1437](https://github.com/coreos/rkt/pull/1437), [#1320](https://github.com/coreos/rkt/pull/1320), [#1076](https://github.com/coreos/rkt/pull/1076), [#1042](https://github.com/coreos/rkt/pull/1042))
- If `rkt run` is used with an image that does not have an app section, rkt will now create one if the user provides an `--exec` flag ([#1427](https://github.com/coreos/rkt/pull/1427))
- A new `rkt image gc` command adds initial support for garbage collecting images from the store ([#1487](https://github.com/coreos/rkt/pull/1487)). This removes treeStores not referenced by any non-GCed rkt pod.
- `rkt list` now provides more information including image version and hash ([#1559](https://github.com/coreos/rkt/pull/1559))
- `rkt image list` output now shows shortened hash identifiers by default, and human readable date formats.
  To use the previous output format, use the `--full` flag. ([#1455](https://github.com/coreos/rkt/pull/1455))
- `rkt prepare` gained the `--exec` flag, which restores flag-parity with `rkt run` ([#1410](https://github.com/coreos/rkt/pull/1410))
- lkvm stage1 backend has experimental support for `rkt enter` ([#1303](https://github.com/coreos/rkt/pull/1303))
- rkt now supports empty volume types ([#1502](https://github.com/coreos/rkt/pull/1502))
- An early, experimental read-only API definition has been added ([#1359](https://github.com/coreos/rkt/pull/1359), [#1518](https://github.com/coreos/rkt/pull/1518)).

#### Bug fixes
- Fixed bug in `--stage1-image` option which prevented it from using URLs ([#1524](https://github.com/coreos/rkt/pull/1524))
- Fixed bug in `rkt trust`'s handling of `--root` ([#1494](https://github.com/coreos/rkt/pull/1494))
- Fixed bug when decompressing xz-compressed images ([#1462](https://github.com/coreos/rkt/pull/1462), [#1224](https://github.com/coreos/rkt/pull/1224))
- In earlier versions of rkt, hooks had an implicit timeout of 30 seconds, causing some pre-start jobs which took a long time to be killed. This implicit timeout has been removed. ([#1547](https://github.com/coreos/rkt/pull/1547))
- When running with the lkvm stage1, rkt now sets `$HOME` if it is not already set, working around a bug in the lkvm tool ([#1447](https://github.com/coreos/rkt/pull/1447), [#1393](https://github.com/coreos/rkt/pull/1393))
- Fixed bug preventing `run-prepared` from working if the metadata service was not available ([#1436](https://github.com/coreos/rkt/pull/1436))

#### Other changes
- Bumped appc spec to 0.7.1 ([#1543](https://github.com/coreos/rkt/pull/1543))
- Bumped CNI and netlink dependencies ([#1476](https://github.com/coreos/rkt/pull/1476))
- Bumped ioprogress to a version which prevents the download bar from being drawn when rkt is not drawing to a terminal ([#1423](https://github.com/coreos/rkt/pull/1423), [#1282](https://github.com/coreos/rkt/pull/1282))
- Significantly reworked rkt's internal use of systemd to orchestrate apps, which should facilitate more granular control over pod lifecycles ([#1407](https://github.com/coreos/rkt/pull/1407))
- Reworked rkt's handling of images with non-deterministically dependencies ([#1240](https://github.com/coreos/rkt/pull/1240), [#1198](https://github.com/coreos/rkt/pull/1198)).
- rkt functional tests now run appc's ACE validator, which should ensure that rkt is always compliant with the specification. ([#1473](https://github.com/coreos/rkt/pull/1473))
- A swathe of improvements to the build system
  - `make clean` should now work
  - Different rkt stage1 images are now built with different names ([#1406](https://github.com/coreos/rkt/pull/1406))
  - rkt can now build on older Linux distributions (like CentOS 6) ([#1529](https://github.com/coreos/rkt/pull/1529))
- Various internal improvements to the functional test suite to improve coverage and consolidate code
- The "ACI" field header in `rkt image` output has been changed to "IMAGE NAME"
- `rkt image rm` now exits with status 1 on any failure ([#1486](https://github.com/coreos/rkt/pull/1486))
- Fixed permissions in the default stage1 image ([#1503](https://github.com/coreos/rkt/pull/1503))
- Added documentation for `prepare` and `run-prepared` subcommands ([#1526](https://github.com/coreos/rkt/pull/1526))
- rkt should now report more helpful errors when encountering manifests it does not understand ([#1471](https://github.com/coreos/rkt/pull/1471))


## v0.8.1

rkt v0.8.1 is an incremental release with numerous bug fixes and clean-up to the build system. It also introduces a few small new features and UX improvements.

- New features and UX changes:
  - `rkt rm` is now variadic: it can now remove multiple pods in one command, by UUID
  - The `APPNAME` column in `rkt image list` output has been changed to the more accurate `NAME`. This involves a schema change in rkt's on-disk datastore, but this should be upgraded transparently.
  - Headers are now sent when following HTTP redirects while trying to retrieve an image
  - The default metadata service port number was changed from a registered/reserved IANA port to an arbitrary port in the non-dynamic range
  - Added the ability to override arguments for network plugins
  - rkt will now error out if someone attempts to use `--private-users` with the lkvm backend
- Bug fixes:
  - Fixed creation of /tmp in apps' root filesystems with correct permissions
  - Fixed garbage collection after umounts (for example, if a system reboots before a pod is cleanly destroyed)
  - Fixed a race in interactive mode when using the lkvm backend that could cause a deadlock or segfault
  - Fixed bad parameter being passed to the metadata service ("uid" -> "uuid")
  - Fixed setting of file permissions during stage1 set up
  - Fixed a potential race condition during simultaneous `iptables` invocation
  - Fixed ACI download progress being sent to stderr instead of stdout, now consistent with the output during retrieval of Docker images
  - `rkt help prepare` will now show the correct default stage1 image
  - rkt will refuse to add isolators with nil Limits, preventing a panic caused by an ambiguity in upstream appc schema
- Other changes:
  - Reworked the SELinux implementation to use `systemd-nspawn`'s native context-switching feature
  - Added a workaround for a bug in Docker <1.8 when it is run on the same system as rkt (see https://github.com/coreos/rkt/issues/1210#issuecomment-132793300)
  - Added a `rkt-xxxx-tapN` name to tap devices that rkt creates
  - Functional tests now clean intermediate images between tests
  - Countless improvements and cleanup to the build system
  - Numerous documentation improvements, including splitting out all top-level `rkt` subcommands into their own documents

## v0.8.0

rkt 0.8.0 includes support for running containers under an LKVM hypervisor
and experimental user namespace support.

Full changelog:

- Documentation improvements
- Better integration with systemd:
 - journalctl -M
 - machinectl {reboot,poweroff}
- Update stage1's systemd to v222
- Add more functional tests
- Build system improvements
- Fix bugs with garbage-collection
- LKVM stage1 support with network and volumes
- Smarter image discovery: ETag and Cache-Control support
- Add CNI DHCP plugin
- Support systemd socket activation
- Backup CAS database when migrating
- Improve error messages
- Add the ability to override ACI exec
- Optimize rkt startup times when a stage1 is present in the store
- Trust keys fetched via TLS by default
- Add the ability to garbage-collect a specific pod
- Add experimental user namespace support
- Bugfixes

## v0.7.0

rkt 0.7.0 includes new subcommands for `rkt image` to manipulate images from
the local store.

It also has a new build system based on autotools and integration with SELinux.

Full changelog:

- New subcommands for `rkt image`: extract, render and export
- Metadata service:
  - Auth now based on tokens
  - Registration done by default, unless --mds-register=false is passed
- Build:
  - Remove support for Go 1.3
  - Replace build system with autoconf and make
- Network: fixes for plugins related to mnt namespace
- Signature: clearer error messages
- Security:
  - Support for SELinux
  - Check signature before downloading
- Commands: fix error messages and parameter parsing
- Output: reduce output verbosity
- Systemd integration: fix stop bug
- Tests: Improve tests output

## v0.6.1

The highlight of this release is the support of per-app memory and CPU
isolators. This means that, in addition to restricting a pod's CPU and memory
usage, individual apps inside a pod can also be restricted now.

rkt 0.6.1 also includes a new CLI/subcommand framework, more functional testing
and journalctl integration by default.

Full changelog:

* Updated to v0.6.1 of the appc spec
* support per-app memory and CPU isolators
* allow network selection to the --private-net flag which can be useful for
  grouping certain pods together while separating others
* move to the Cobra CLI/subcommand framework
* per-app logging via journalctl now supported by default
* stage1 runs an unpatched systemd v220
* to help packagers, rkt can generate stage1 from the binaries on the host at
  runtime
* more functional tests
* bugfixes

## v0.5.6

rkt 0.5.6 includes better integration with systemd on the host, some minor bug
fixes and a new ipvlan network plugin.

- Updated to v0.5.2 of the appc spec
- support running from systemd unit files for top-level isolation
- support per-app logging via journalctl. This is only supported if stage1 has
  systemd v219 or v220
- add ipvlan network plugin
- new rkt subcommand: cat-manifest
- extract ACI in a chroot to avoid malformed links modifying the host
  filesystem
- improve rkt error message if the user doesn't provide required volumes
- fix rkt status when using overlayfs
- support for some arm architectures
- documentation improvements


## v0.5.5

rkt 0.5.5 includes a move to [cni](https://github.com/appc/cni) network
plugins, a number of minor bug fixes and two new experimental commands for
handling images: `rkt images` and `rkt rmimage`. 

Full changelog:
- switched to using [cni](https://github.com/appc/cni) based network plugins
- fetch images dependencies recursively when ACIs have dependent images
- fix the progress bar used when downloading images with no content-length
- building the initial stage1 can now be done on various versions of systemd
- support retrying signature downloads in the case of a 202
- remove race in doing a rkt enter
- various documentation fixes to getting started and other guides
- improvements to the functional testing using a new gexpect, testing for
  non-root apps, run context, port test, and more


## v0.5.4

rkt 0.5.4 introduces a number of new features - repository authentication,
per-app arguments + local image signature verification, port forwarding and
more. Further, although we aren't yet guaranteeing API/ABI stability between
releases, we have added important work towards this goal including functional
testing and database migration code.

This release also sees the removal of the `--spawn-metadata-svc` flag to 
`rkt run`. The flag was originally provided as a convenience, making it easy
for users to get started with the metadata service.  In rkt v0.5.4 we removed
it in favor of explicitly starting it via `rkt metadata-service` command. 

Full changelog:
- added configuration support for repository authentication (HTTP Basic Auth,
  OAuth, and Docker repositories). Full details in
  `Documentation/configuration.md`
- `rkt run` now supports per-app arguments and per-image `--signature`
  specifications
- `rkt run` and `rkt fetch` will now verify signatures for local image files
- `rkt run` with `--private-net` now supports port forwarding (using
  `--port=NAME:1234`)
- `rkt run` now supports a `--local` flag to use only local images (i.e. no
  discovery or remote image retrieval will be performed)
- added initial support for running directly from a pod manifest
- the store DB now supports migrations for future versions
- systemd-nspawn machine names are now set to pod UUID
- removed the `--spawn-metadata-svc` option from `rkt run`; this mode was
  inherently racy and really only for convenience. A separate 
  `rkt metadata-service` invocation should be used instead.
- various internal codebase refactoring: "cas" renamed to "store", tasks to
  encapsulate image fetch operations, etc
- bumped docker2aci to support authentication for Docker registries and fix a
  bug when retrieving images from Google Container Registry
- fixed a bug where `--interactive` did not work with arguments
- garbage collection for networking is now embedded in the stage1 image
- when rendering images into the treestore, a global syncfs() is used instead
  of a per-file sync(). This should significantly improve performance when
  first extracting large images
- added extensive functional testing on semaphoreci.com/coreos/rkt
- added a test-auth-server to facilitate testing of fetching images


## v0.5.3
This release contains minor updates over v0.5.2, notably finalising the move to
pods in the latest appc spec and becoming completely name consistent on `rkt`.
- {Container,container} changed globally to {Pod,pod}
- {Rocket,rocket} changed globally to `rkt`
- `rkt install` properly sets permissions for all directories
- `rkt fetch` leverages the cas.Store TmpDir/TmpFile functions (now exported)
  to generate temporary files for downloads
- Pod lifecycle states are now exported for use by other packages
- Metadata service properly synchronizes access to pod state


## v0.5.2

This release is a minor update over v0.5.1, incorporating several bug fixes and
a couple of small new features:
- `rkt enter` works when overlayfs is not available
- `rkt run` now supports the `--no-overlay` option referenced (but not
  implemented!) in the previous release
- the appc-specified environment variables (PATH, HOME, etc) are once again set
  correctly during `rkt run`
- metadata-service no longer manipulates IP tables rules as it connects over a
  unix socket by default
- pkg/lock has been improved to also support regular (non-directory) files
- images in the cas are now locked at runtime (as described in [#460](https://github.com/coreos/rkt/pull/460))


## v0.5.1

This release updates Rocket to follow the latest version of the appc spec,
v0.5.1. This involves the major change of moving to _pods_ and _Pod Manifests_
(which enhance and supplant the previous _Container Runtime Manifest_). The
Rocket codebase has been updated across the board to reflect the schema/spec
change, as well as changing various terminology in other human-readable places:
for example, the previous ambiguous (unqualified) "container" is now replaced
everywhere with "pod".

This release also introduces a number of key features and minor changes:
- overlayfs support, enabled for `rkt run` by default (disable with
  `--no-overlayfs`)
- to facilitate overlayfs, the CAS now features a tree store which stores
  expanded versions of images
- the default stage1 (based on systemd) can now be built from source, instead
  of only derived from an existing binary distribution as previously. This is
  configurable using the new `RKT_STAGE1_USR_FROM` environment variable when
  invoking the build script - see fdcd64947
- the metadata service now uses a Unix socket for registration; this limits who
  can register/unregister pods by leveraging filesystem permissions on the
  socket
- `rkt list` now abbreviates UUIDs by default (configurable with `--full`)
- the ImageManifest's `readOnly` field (for volume mounts) is now overridden by
  the rkt command line
- a simple debug script (in scripts/debug) to facilitate easier debugging of
  applications running under Rocket by injecting Busybox into the pod
- documentation for the metadata service, as well as example systemd unit files


## v0.4.2

- First support for interactive containers, with the `rkt run --interactive`
  flag. This is currently only supported if a container has one app. [#562](https://github.com/coreos/rkt/pull/562) #[601](https://github.com/coreos/rkt/pull/601)
- Add container IP address information to `rkt list`
- Provide `/sys` and `/dev/shm` to apps (per spec)
- Introduce "latest" pattern handling for local image index
- Implement FIFO support in tar package
- Restore atime and mtime during tar extraction
- Bump docker2aci dependency


## v0.4.1

This is primarily a bug fix release with the addition of the `rkt install`
subcommand to help people setup a unprivileged `rkt fetch` based on unix users.

- Fix marshalling error when running containers with resource isolators
- Fixup help text on run/prepare about volumes
- Fixup permissions in `rkt trust` created files
- Introduce the `rkt install` subcommand


## v0.4.0

This release is mostly a milestone release and syncs up with the latest release
of the [appc spec](https://github.com/appc/spec/releases/tag/v0.4.0) yesterday.

Note that due to the introduction of a database for indexing the local CAS,
users upgrading from previous versions of Rocket on a system may need to clear
their local cache by removing the `cas` directory. For example, using the
standard Rocket setup, this would be accomplished with 
`rm -fr /var/lib/rkt/cas`.

Major changes since v0.3.2:
- Updated to v0.4.0 of the appc spec
- Introduced a database for indexing local images in the CAS (based on
  github.com/cznic/ql)
- Refactored container lifecycle to support a new "prepared" state, to
- pre-allocate a container UUID without immediately running the application
- Added support for passing arguments to apps through the `rkt run` CLI
- Implemented ACI rendering for dependencies
- Renamed `rkt metadatasvc` -> `rkt metadata-service`
- Added documentation around networking, container lifecycle, and rkt commands


## v0.3.2

This release introduces much improved documentation and a few new features.

The highlight of this release is that Rocket can now natively run Docker
images. To do this, it leverages the appc/docker2aci library which performs a
straightforward conversion between images in the Docker format and the appc
format.

A simple example:

```
$ rkt --insecure-skip-verify run docker://redis docker://tenstartups/redis-commander
rkt: fetching image from docker://redis
rkt: warning: signature verification has been disabled
Downloading layer: 511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158
```

Note that since Docker images do not support image signature verifications, the
`-insecure-skip-verify` must be used.

Another important change in this release is that the default location for the
stage1 image used by `rkt run` can now be set at build time, by setting the
`RKT_STAGE1_IMAGE` environment variable when invoking the build script. (If
this is not set, `rkt run` will continue with its previous behaviour of looking
for a stage1.aci in the same directory as the binary itself. This makes it
easier for distributions to package Rocket and include the stage1 wherever
they choose (for example, `/usr/lib/rkt/stage1.aci`). For more information, see
https://github.com/coreos/rocket/pull/520


## v0.3.1

The primary motivation for this release is to resynchronise versions with the
appc spec. To minimise confusion in the short term we intend to keep the
major/minor version of Rocket aligned with the version of spec it implements;
hence, since yesterday v0.3.0 of the appc spec was released, today Rocket
becomes v0.3.1. After the spec (and Rocket) reach v1.0.0, we may relax this
restriction.

This release also resolves an upstream bug in the appc discovery code which was
causing rkt trust to fail in certain cases.


## v0.3.0

This is largely a momentum release but it does introduce a few new user-facing
features and some important changes under the hood which will be of interest to
developers and distributors.

First, the CLI has a couple of new commands:
- `rkt trust` can be used to easily add keys to the public keystore for ACI
  signatures (introduced in the previous release). This supports retrieving
  public keys directly from a URL or using discovery to locate public keys - a
  simple example of the latter is `rkt trust --prefix coreos.com/etcd`. See the
  commit for other examples.
- `rkt list` is an extremely simple tool to list the containers on the system

As mentioned, v0.3.0 includes two significant changes to the Rocket build process:
- Instead of embedding the (default) stage1 using go-bindata, Rocket now
  consumes a stage1 in the form of an actual ACI, containing a rootfs and
  stage1 init/exec binaries. By default, Rocket will look for a `stage1.aci` in
  the same directory as the location of the binary itself, but the stage1 can
  be explicitly specified with the new `-stage1-image` flag (which deprecates
  `-stage1-init` and `-stage1-rootfs`). This makes it much more straightforward
  to use alternative stage1 images with rkt and facilitates packing it for
  different distributions like Fedora.
- Rocket now vendors a copy of the appc/spec instead of depending on HEAD. This
  means that Rocket can be built in a self-contained and reproducible way and
  that master will no longer break in response to changes to the spec. It also
  makes explicit the specific version of the spec against which a particular
  release of Rocket is compiled.

As a consequence of these two changes, it is now possible to use the standard
Go workflow to build the Rocket CLI (e.g. `go get github.com/coreos/rocket/rkt`
will build rkt). Note however that this does not implicitly build a stage1, so
that will still need to be done using the included ./build script, or some
other way for those desiring to use a different stage1.


## v0.2.0

This introduces countless features and improvements over v0.1.1. Highlights
include several new commands (`rkt status`, `rkt enter`, `rkt gc`) and
signature validation.


## v0.1.1

The most significant change in this release is that the spec has been split
into its own repository (https://github.com/appc/spec), and significantly
updated since the last release - so many of the changes were to update to match
the latest spec.

Numerous improvements and fixes over v0.1.0:
- Rocket builds on non-Linux (in a limited capacity)
- Fix bug handling uncompressed images
- More efficient image handling in CAS
- mkrootfs now caches and GPG checks images
- stage1 is now properly decoupled from host runtime
- stage1 supports socket activation
- stage1 no longer warns about timezones
- cas now logs download progress to stdout
- rkt run now acquires an exclusive lock on the container directory and records
  the PID of the process


## v0.1.0

- tons of documentation improvements added
- actool introduced along with documentation
- image discovery introduced to rkt run and rkt fetch


## v0.0.0

Initial release.
