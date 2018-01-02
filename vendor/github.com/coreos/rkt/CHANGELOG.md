## 1.25.0

This minor release contains bugfixes and other improvements related to the KVM flavour, which is now using qemu-kvm by default.

## New Features:
- Switch default kvm flavour from lkvm to qemu ([#3562](https://github.com/coreos/rkt/pull/3562)).

### Bug fixes
- stage1/kvm: Change RAM calculation, and increase minimum ([#3572](https://github.com/coreos/rkt/pull/3572)).
- stage1: Ensure ptmx device usable by non-root for all flavours ([#3484](https://github.com/coreos/rkt/pull/3484)).

## Other changes:
- tests: fix TestNonRootReadInfo when $HOME is only accessible by current user ([#3580](https://github.com/coreos/rkt/pull/3580)).
- glide: bump grpc to 1.0.4 ([#3584](https://github.com/coreos/rkt/pull/3584)).
- vendor: bump docker2aci to 0.16.0 ([#3591](https://github.com/coreos/rkt/pull/3591)).

## 1.24.0

This release includes experimental support for attaching to a running application's input and output. It also introduces
a more finely grained pull-policy flag.

## New Features:
- rkt: add experimental support for attachable applications ([#3396](https://github.com/coreos/rkt/pull/3396)).
    It consists of:
    * a new `attach` subcommand
    * a set of per-app flags to control stdin/stdout/stderr modes
    * a stage1 `iottymux` binary for multiplexing and attaching
    * two new templated stage1 services, `iomux` and `ttymux`
- run/prepare/fetch: replace --no-store and --store-only with --pull-policy ([#3554](https://github.com/coreos/rkt/pull/3554)).
    * Replaces the `--no-store` and `--store-only` flags with a singular
    flag `--pull-policy`.
    * can accept one of three things, `never`, `new`, and `update`.
    * `--no-store` has been aliased to `--pull-policy=update`
    * `--store-only` has been aliased to `--pull-policy=never`

### Bug fixes
- image gc: don't remove images that currently running pods were made from ([#3549](https://github.com/coreos/rkt/pull/3549)).
- stage1/fly: evaluate symlinks in mount targets ([#3570](https://github.com/coreos/rkt/pull/3570)).
- lib/app: use runtime app mounts and appVolumes rather than mountpoints ([#3571](https://github.com/coreos/rkt/pull/3571)).

## Other changes:
- kvm/qemu: Update QEMU to v2.8.0 ([#3568](https://github.com/coreos/rkt/pull/3568)).
- stage0/app-add: CLI args should override image ones ([#3566](https://github.com/coreos/rkt/pull/3566)).
- lib/app: use runtime app mounts and appVolumes rather than mountpoints ([#3571](https://github.com/coreos/rkt/pull/3571)).
- kvm/lkvm: update lkvm version to HEAD ([#3569](https://github.com/coreos/rkt/pull/3569)).
- vendor: bump appc to v0.8.10 ([#3574](https://github.com/coreos/rkt/pull/3574)).
- docs: ([#3552](https://github.com/coreos/rkt/pull/3552))

### Build & Test:
- tests: remove gexpect from TestAppUserGroup ([#3561](https://github.com/coreos/rkt/pull/3561)).
- travis: remove "gimme.local" script ([#3556](https://github.com/coreos/rkt/pull/3556)).
- tests: fix when $HOME is only accessible by current user ([#3559](https://github.com/coreos/rkt/pull/3559)).
- makelib: introduce --enable-incremental-build, enabling "go install" ([#3553](https://github.com/coreos/rkt/pull/3553)).

## 1.23.0

This release adds a lot of bugfixes around the rkt fly flavor, garbage collection, kvm, and the sandbox. The new experimental `app` subcommand now follows the semantic of CRI of not quitting prematurely if apps fail or exit. Finally docker2aci received an important update fixing issues with os/arch labels which caused issues on arm architectures, a big thanks here goes to @ybubnov for this contribution.

### New features
- sandbox: don't exit if an app fails ([#3478](https://github.com/coreos/rkt/pull/3478)). In contrast to regular `rkt run` behavior, the sandbox now does not quit if all or single apps fail or exit.

### Bug fixes
- stage1: fix incorrect splitting function ([#3541](https://github.com/coreos/rkt/pull/3541)).
- sandbox/app-add: fix mount targets with absolute symlink targets ([#3490](https://github.com/coreos/rkt/pull/3490)).
- namefetcher: fix nil pointer dereference ([#3536](https://github.com/coreos/rkt/pull/3536)).
- Bump appc/docker2aci library version to 0.15.0 ([#3534](https://github.com/coreos/rkt/pull/3534)). This supports the conversion of images with various os/arch labels.
- stage1: uid shift systemd files ([#3529](https://github.com/coreos/rkt/pull/3529)).
- stage1/kvm/lkvm: chown files and dirs on creation ([#3485](https://github.com/coreos/rkt/pull/3485)).
- stage1/fly: record pgid and let stop fallback to it ([#3523](https://github.com/coreos/rkt/pull/3523)).
- common/overlay: allow data directory name with colon character ([#3505](https://github.com/coreos/rkt/pull/3505)).
- api-service: stop erroring when a pod is running ([#3525](https://github.com/coreos/rkt/pull/3525)).
- stage1/fly: clear FD_CLOEXEC only once ([#3521](https://github.com/coreos/rkt/pull/3521)).
- stage1: Add hostname to /etc/hosts ([#3522](https://github.com/coreos/rkt/pull/3522)).
- gc: avoid erroring in race to deletion ([#3515](https://github.com/coreos/rkt/pull/3515)).
- tests/rkt_stop: Wait for 'stop' command to complete ([#3518](https://github.com/coreos/rkt/pull/3518)).
- pkg/pod: avoid nil panic for missing pods ([#3514](https://github.com/coreos/rkt/pull/3514)).

### Other changes
- stage1: move more logic out of AppUnit ([#3496](https://github.com/coreos/rkt/pull/3496)).
- tests: use appc schema instead of string templates ([#3520](https://github.com/coreos/rkt/pull/3520)).
- stage1: kvm: Update kernel to 4.9.2 ([#3530](https://github.com/coreos/rkt/pull/3530)).
- stage1: remount entire subcgroup r/w, instead of each knob ([#3494](https://github.com/coreos/rkt/pull/3494)).
- tests: update AWS CI setup ([#3509](https://github.com/coreos/rkt/pull/3509)).
- pkg/fileutil: helper function to get major, minor numbers of a device file ([#3500](https://github.com/coreos/rkt/pull/3500)). 
- pkg/log: correctly handle var-arg printf params ([#3516](https://github.com/coreos/rkt/pull/3516)).
- Documentation/stop: describe --uuid-file option ([#3511](https://github.com/coreos/rkt/pull/3511)).

## 1.22.0

This is a stabilization release which includes better support for environments without systemd, improvements to GC behavior in complex scenarios, and several additional fixes.

### New features and UX changes

- rkt/cat-manifest: add support for --uuid-file ([#3498](https://github.com/coreos/rkt/pull/3498)).
- stage1: fallback if systemd cgroup doesn't exist ([#3507](https://github.com/coreos/rkt/pull/3507)).
- vendor: bump gocapability ([#3493](https://github.com/coreos/rkt/pull/3493)). This change renames `sys_psacct` to `sys_pacct`.
- stage0/app: pass debug flag to entrypoints ([#3469](https://github.com/coreos/rkt/pull/3469)).

### Bug fixes
- gc: fix cleaning mounts and files ([#3486](https://github.com/coreos/rkt/pull/3486)). This improves GC behavior in case of busy mounts and other complex scenarios.
- mount: ensure empty volume paths exist for copy-up ([#3468](https://github.com/coreos/rkt/pull/3468)).
- rkt stop/rm: a pod must be closed after PodFromUUIDString() ([#3492](https://github.com/coreos/rkt/pull/3492)).

### Other changes
- stage1/kvm: add a dash in kernel LOCALVERSION ([#3489](https://github.com/coreos/rkt/pull/3489)).
- stage1/kvm: Improve QEMU Makefile rules ([#3474](https://github.com/coreos/rkt/pull/3474)).
- pkg/pod: use IncludeMostDirs bitmask instead of constructing it ([#3506](https://github.com/coreos/rkt/pull/3506)).
- pkg/pod: add WaitReady, dry Sandbox methods ([#3462](https://github.com/coreos/rkt/pull/3462)).
- vendor: bump gexpect to 0.1.1 ([#3467](https://github.com/coreos/rkt/pull/3467)).
- common: fix 'the the' duplication in comment ([#3497](https://github.com/coreos/rkt/pull/3497)).
- docs: multiple updates ([#3479](https://github.com/coreos/rkt/pull/3479), [#3501](https://github.com/coreos/rkt/pull/3501), [#3464](https://github.com/coreos/rkt/pull/3464), [#3495](https://github.com/coreos/rkt/pull/3495)).

## 1.21.0

This release includes bugfixes for the experimental CRI support, more stable integration tests, and some other interesting changes:

- The `default-restricted` network changed from 172.16.28.0/24 to 172.17.0.0/26.
- The detailed roadmap for OCI support has been finalized.


### New features
- Change the subnet for the default-restricted network ([#3440](https://github.com/coreos/rkt/pull/3440)), ([#3459](https://github.com/coreos/rkt/pull/3459)).
- Prepare for writable /proc/sys, and /sys ([#3389](https://github.com/coreos/rkt/pull/3389)).
- Documentation/proposals: add OCI Image Format roadmap ([#3425](https://github.com/coreos/rkt/pull/3425)).

### Bug fixes
- stage1: app add, status didn't work with empty vols ([#3451](https://github.com/coreos/rkt/pull/3451)).
- stage1: properly run defer'd umounts in app add ([#3455](https://github.com/coreos/rkt/pull/3455)).
- cri: correct 'created' timestamp ([#3399](https://github.com/coreos/rkt/pull/3399)).
- fly: ensure the target bin directory exists before building ([#3436](https://github.com/coreos/rkt/pull/3436)).
- rkt: misc systemd-related fixes ([#3418](https://github.com/coreos/rkt/pull/3418)).

### Other changes
- pkg/mountinfo: move mountinfo parser to its own package ([#3415](https://github.com/coreos/rkt/pull/3415)).
- stage1: persist runtime parameters ([#3432](https://github.com/coreos/rkt/pull/3432)), ([#3450](https://github.com/coreos/rkt/pull/3450)).
- stage1: signal supervisor readiness ([#3424](https://github.com/coreos/rkt/pull/3424)), ([#3439](https://github.com/coreos/rkt/pull/3439)).
- sandbox: add missing flagDNSDomain and flagHostsEntries parameters ([#3430](https://github.com/coreos/rkt/pull/3430)).
- pkg/tar: fix variable name in error ([#3433](https://github.com/coreos/rkt/pull/3433)).
- tests: fix TestExport for the KVM+overlay case ([#3435](https://github.com/coreos/rkt/pull/3435)).
- tests: fix some potential gexpect hangs ([#3443](https://github.com/coreos/rkt/pull/3443)).
- tests: add smoke test for app sandbox ([#3371](https://github.com/coreos/rkt/pull/3371)).
- tests: tentative fixes for sporadic host and kvm failures ([#3434](https://github.com/coreos/rkt/pull/3434)).
- rkt: remove empty TODO ([#3417](https://github.com/coreos/rkt/pull/3417)).
- Documentation updates: [#3446](https://github.com/coreos/rkt/pull/3446), ([#3421](https://github.com/coreos/rkt/pull/3421)), ([#3412](https://github.com/coreos/rkt/pull/3412)).

## 1.20.0

This release contains additional bug fixes for the new experimental `app` subcommand, following the path towards the Container Runtime Interface (CRI).
It also adds first step towards OCI by introducing an internal concept called "distribution points", which will allow rkt to recognize multiple image formats internally.
Finally the rkt fly flavor gained support for `rkt enter`.

### New features and UX changes
- stage1/fly: Add a working `rkt enter` implementation ([#3377](https://github.com/coreos/rkt/pull/3377)).

### Bug fixes:
- tests/build-and-run-test.sh: fix systemd revision parameter ([#3395](https://github.com/coreos/rkt/pull/3395)).
- namefetcher: Use ETag in fetchVerifiedURL() ([#3374](https://github.com/coreos/rkt/pull/3374)).
- rkt/run: validates pod manifest to make sure it contains at least one app ([#3363](https://github.com/coreos/rkt/pull/3363)).
- rkt/app: multiple bugfixes ([#3405](https://github.com/coreos/rkt/pull/3405)).

### Other changes
- glide: deduplicate cni entries and update go-systemd ([#3372](https://github.com/coreos/rkt/pull/3372)).
- stage0: improve list --format behavior and flags ([#3403](https://github.com/coreos/rkt/pull/3403)).
- pkg/pod: flatten the pod state if-ladders ([#3404](https://github.com/coreos/rkt/pull/3404)).
- tests: adjust security tests for systemd v232 ([#3401](https://github.com/coreos/rkt/pull/3401)).
- image: export `ImageListEntry` type for image list ([#3383](https://github.com/coreos/rkt/pull/3383)).
- glide: bump gopsutil to v2.16.10 ([#3400](https://github.com/coreos/rkt/pull/3400)).
- stage1: update coreos base to alpha 1235.0.0 ([#3388](https://github.com/coreos/rkt/pull/3388)).
- rkt: Implement distribution points ([#3369](https://github.com/coreos/rkt/pull/3369)). This is the implementation of the distribution concept proposed in [#2953](https://github.com/coreos/rkt/pull/2953).
- build: add --with-stage1-systemd-revision option for src build ([#3362](https://github.com/coreos/rkt/pull/3362)).
- remove isReallyNil() ([#3381](https://github.com/coreos/rkt/pull/3381)). This is cleanup PR, removing some reflection based code.
- vendor: update appc/spec to 0.8.9 ([#3384](https://github.com/coreos/rkt/pull/3384)).
- vendor: Remove direct k8s dependency ([#3312](https://github.com/coreos/rkt/pull/3312)).
- Documentation updates: [#3366](https://github.com/coreos/rkt/pull/3366), [#3376](https://github.com/coreos/rkt/pull/3376), [#3379](https://github.com/coreos/rkt/pull/3379), [#3406](https://github.com/coreos/rkt/pull/3406), [#3410](https://github.com/coreos/rkt/pull/3410).

## 1.19.0

This release contains multiple changes to rkt core, bringing it more in line with the new Container Runtime Interface (CRI) from Kubernetes.

A new experimental `app` subcommand has been introduced, which allows creating a "pod sandbox" and dynamically mutating it at runtime. This feature is not yet completely stabilized, and is currently gated behind an experimental flag.

### New features and UX changes
- rkt: experimental support for pod sandbox ([#3318](https://github.com/coreos/rkt/pull/3318)). This PR introduces an experimental `app` subcommand and many additional app-level options.
- rkt/image: align image selection behavior for the rm subcommand ([#3353](https://github.com/coreos/rkt/pull/3353)).
- stage1/init: leave privileged pods without stage2 mount-ns ([#3290](https://github.com/coreos/rkt/pull/3290)).
- stage0/image: list images output in JSON format ([#3334](https://github.com/coreos/rkt/pull/3334)).
- stage0/arch: initial support for ppc64le platform ([#3315](https://github.com/coreos/rkt/pull/3315)).

### Bug fixes:
- gc: make sure `CNI_PATH` is same for gc and init ([#3348](https://github.com/coreos/rkt/pull/3348)).
- gc: clean up some GC leaks ([#3317](https://github.com/coreos/rkt/pull/3317)).
- stage0: minor wording fixes ([#3351](https://github.com/coreos/rkt/pull/3351)).
- setup-data-dir.sh: fallback to the `mkdir/chmod`s if the rkt.conf doesn't exist ([#3335](https://github.com/coreos/rkt/pull/3335)).
- scripts: add gpg to Debian dependencies ([#3339](https://github.com/coreos/rkt/pull/3339)).
- kvm: fix for breaking change in Debian Sid GCC default options ([#3354](https://github.com/coreos/rkt/pull/3354)).
- image/list: bring back field filtering in plaintext mode ([#3361](https://github.com/coreos/rkt/pull/3361)).

### Other changes
- cgroup/v1: introduce mount flags to mountFsRO ([#3350](https://github.com/coreos/rkt/pull/3350)).
- kvm: update QEMU version to 2.7.0 ([#3341](https://github.com/coreos/rkt/pull/3341)).
- kvm: bump kernel version to 4.8.6, updated config ([#3342](https://github.com/coreos/rkt/pull/3342)). 
- vendor: introduce kr/pretty and bump go-systemd ([#3333](https://github.com/coreos/rkt/pull/3333)).
- vendor: update docker2aci to 0.14.0 ([#3356](https://github.com/coreos/rkt/pull/3356)).
- tests: add the --debug option to more tests ([#3340](https://github.com/coreos/rkt/pull/3340)).
- scripts/build-rir: bump rkt-builder version to 1.1.1 ([#3360](https://github.com/coreos/rkt/pull/3360)).
- Documentation updates: [#3321](https://github.com/coreos/rkt/pull/3321), [#3331](https://github.com/coreos/rkt/pull/3331), [#3325](https://github.com/coreos/rkt/pull/3325).

## 1.18.0

This minor release contains bugfixes, UX enhancements, and other improvements.

### UX changes:
- rkt: gate diagnostic output behind `--debug` ([#3297](https://github.com/coreos/rkt/pull/3297)).
- rkt: Change exit codes to 254 ([#3261](https://github.com/coreos/rkt/pull/3261)). 


### Bug fixes:
- stage1/kvm: correctly bind-mount read-only volumes ([#3304](https://github.com/coreos/rkt/pull/3304)). 
- stage0/cas: apply xattr attributes ([#3305](https://github.com/coreos/rkt/pull/3305)). 
- scripts/install-rkt: add iptables dependency ([#3309](https://github.com/coreos/rkt/pull/3309)). 
- stage0/image: set proxy if InsecureSkipVerify is set ([#3303](https://github.com/coreos/rkt/pull/3303)).

### Other changes
- vendor: update docker2aci to 0.13.0 ([#3314](https://github.com/coreos/rkt/pull/3314)). This fixes multiple fetching and conversion bugs, including two security issues.
- scripts: update glide vendor script ([#3313](https://github.com/coreos/rkt/pull/3313)). 
- vendor: update appc/spec to v0.8.8 ([#3310](https://github.com/coreos/rkt/pull/3310)). 
- stage1: update to CoreOS 1192.0.0 (and update sanity checks) ([#3283](https://github.com/coreos/rkt/pull/3283)). 
- cgroup: introduce proper cgroup/v1, cgroup/v2 packages ([#3277](https://github.com/coreos/rkt/pull/3277)).
- Documentation updates: ([#3281](https://github.com/coreos/rkt/pull/3281)), ([#3319](https://github.com/coreos/rkt/pull/3319)), ([#3308](https://github.com/coreos/rkt/pull/3308)).


## 1.17.0

This is a minor release packaging rkt-api systemd service units, and fixing a bug caused by overly long lines in generated stage1 unit files.

### New features and UX changes
- dist: Add systemd rkt-api service and socket ([#3271](https://github.com/coreos/rkt/pull/3271)).
- dist: package rkt-api unit files ([#3275](https://github.com/coreos/rkt/pull/3275)).

### Bug fixes
- stage1: break down overlong property lines ([#3279](https://github.com/coreos/rkt/pull/3279)).

### Other changes
- stage0: fix typo and some docstring style ([#3266](https://github.com/coreos/rkt/pull/3266)).
- stage0: Create an mtab symlink if not present ([#3265](https://github.com/coreos/rkt/pull/3265)).
- stage1: use systemd protection for kernel tunables ([#3273](https://github.com/coreos/rkt/pull/3273)).
- Documentation updates: ([#3280](https://github.com/coreos/rkt/pull/3280), [#3263](https://github.com/coreos/rkt/pull/3263), [#3268](https://github.com/coreos/rkt/pull/3268), [#3254](https://github.com/coreos/rkt/pull/3254), [#3199](https://github.com/coreos/rkt/pull/3199), [#3256](https://github.com/coreos/rkt/pull/3256))

## 1.16.0

This release contains an important bugfix for the stage1-host flavor, as well as initial internal support for cgroup2 and pod sandboxes as specified by kubernetes CRI (Container Runtime Interface).

### Bug fixes
- stage1/host: fix systemd-nspawn args ordering ([#3216](https://github.com/coreos/rkt/pull/3216)). Fixes https://github.com/coreos/rkt/issues/3215.

### New features
- rkt: support for unified cgroups (cgroup2) ([#3032](https://github.com/coreos/rkt/pull/3032)). This implements support for cgroups v2 along support for legacy version.
- cri: initial implementation of stage1 changes ([#3218](https://github.com/coreos/rkt/pull/3218)). This PR pulls the stage1-based changes from the CRI branch back into
master, leaving out the changes in stage0 (new app subcommands).

### Other changes
- doc/using-rkt-with-systemd: fix the go app example ([#3217](https://github.com/coreos/rkt/pull/3217)).
- rkt: refactor app-level flags handling ([#3209](https://github.com/coreos/rkt/pull/3209)). This is in preparation for https://github.com/coreos/rkt/pull/3205
- docs/distributions: rearrange, add centos ([#3212](https://github.com/coreos/rkt/pull/3212)).
- rkt: Correct typos listed by the tool misspell ([#3208](https://github.com/coreos/rkt/pull/3208)).

## 1.15.0

This relase brings some expanded DNS configuration options, beta support for QEMU, recursive volume mounts, and improved sd_notify support.

### New features
- DNS configuration improvements ([#3161](https://github.com/coreos/rkt/pull/3161)):
    - Respect DNS results from CNI
    - Add --dns=host mode to bind-mount the host's /etc/resolv.conf
    - Add --dns=none mode to ignore CNI DNS
    - Add --hosts-entry (IP=HOSTNAME) to tweak the pod's /etc/hosts
    - Add --hosts-entry=host to bind-mount the host's /etc/hosts
- Introduce QEMU support as an alternative KVM hypervisor ([#2952](https://github.com/coreos/rkt/pull/2952))
- add support for recursive volume/mounts ([#2880](https://github.com/coreos/rkt/pull/2880))
- stage1: allow sd_notify from the app in the container to the host ([#2826](https://github.com/coreos/rkt/pull/2826)).

### Other changes
- rkt-monitor: bunch of improvements ([#3093](https://github.com/coreos/rkt/pull/3093))
- makefile/kvm: add dependency for copied files ([#3197](https://github.com/coreos/rkt/pull/3197))
- store: refactor GetRemote ([#2975](https://github.com/coreos/rkt/pull/2975)).
- build,stage1: include systemd dir when checking libs ([#3186](https://github.com/coreos/rkt/pull/3186))
- tests: volumes: add missing test `volumeMountTestCasesNonRecursive` ([#3165](https://github.com/coreos/rkt/pull/3165))
- kvm/pod: disable insecure-options=paths for kvm flavor ([#3155](https://github.com/coreos/rkt/pull/3155))
- stage0: don't copy image annotations to pod manifest RuntimeApp annotations ([#3100](https://github.com/coreos/rkt/pull/3100))
- stage1: shutdown.service: don't use /dev/console ([#3148](https://github.com/coreos/rkt/pull/3148))
- build: build simple .deb and .rpm packages ([#3177](https://github.com/coreos/rkt/pull/3177)). Add a simple script to build .deb and .rpm packages. This is not a substitute for a proper distro-maintained package.
- Documentation updates: ([#3196](https://github.com/coreos/rkt/pull/3196)) ([#3192](https://github.com/coreos/rkt/pull/3192)) ([#3187](https://github.com/coreos/rkt/pull/3187)) ([#3185](https://github.com/coreos/rkt/pull/3185)) ([#3182](https://github.com/coreos/rkt/pull/3182)) ([#3180](https://github.com/coreos/rkt/pull/3180)) ([#3166](https://github.com/coreos/rkt/pull/3166))
- proposals/app-level-api: add rkt app sandbox subcommand ([#3147](https://github.com/coreos/rkt/pull/3147)). This adds a new subcommand `app init` to create an initial empty pod.

## 1.14.0

This release updates the coreos and kvm flavors, bringing in a newer stable systemd (v231). Several fixes and cgroups-related changes landed in `api-service`, and better heuristics have been introduced to avoid using overlays in non-supported environments. Finally, `run-prepared` now honors options for insecure/privileged pods too.

### New features and UX changes
- stage1: update to CoreOS 1151.0.0 and systemd v231 ([#3122](https://github.com/coreos/rkt/pull/3122)).
- common: fall back to non-overlay with ftype=0 ([#3105](https://github.com/coreos/rkt/pull/3105)).
- rkt: honor insecure-options in run-prepared ([#3138](https://github.com/coreos/rkt/pull/3138)).

#### Bug fixes
- stage0: fix golint warnings ([#3099](https://github.com/coreos/rkt/pull/3099)).
- rkt: avoid possible panic in api-server ([#3111](https://github.com/coreos/rkt/pull/3111)).
- rkt/run: allow --set-env-file files with comments ([#3115](https://github.com/coreos/rkt/pull/3115)).
- scripts/install-rkt: add wget as dependency ([#3124](https://github.com/coreos/rkt/pull/3124)).
- install-rkt.sh: scripts: Fix missing files in .deb when using install-rkt.sh ([#3127](https://github.com/coreos/rkt/pull/3127)). 
- tests: check for run-prepared with insecure options ([#3139](https://github.com/coreos/rkt/pull/3139)).

#### Other changes
- seccomp/docker: update docker whitelist to include mlock ([#3126](https://github.com/coreos/rkt/pull/3126)). This updates the `@docker/default-whitelist` to include mlock-related
syscalls (mlock, mlock2, mlockall).
- build: add PowerPC ([#2936](https://github.com/coreos/rkt/pull/2936)).
- scripts: install-rkt.sh: fail install-pak on errors ([#3150](https://github.com/coreos/rkt/pull/3150)). When install-pak (called from install-rkt.sh) fails at some point
abort packaging.
- api_service: Rework cgroup detection ([#3072](https://github.com/coreos/rkt/pull/3072)). Use the `subcgroup` file hint provided by some stage1s rather than
machined registration.
- Documentation/devel: add make images target ([#3142](https://github.com/coreos/rkt/pull/3142)). This introduces the possibility to generate graphivz based PNG images using
a new `images` make target.
- vendor: update appc/spec to 0.8.7 ([#3143](https://github.com/coreos/rkt/pull/3143)).
- stage1/kvm: avoid writing misleading subcgroup ([#3107](https://github.com/coreos/rkt/pull/3107)).
- vendor: update go-systemd to v12 ([#3125](https://github.com/coreos/rkt/pull/3125)).
- scripts: bump coreos.com/rkt/builder image version ([#3092](https://github.com/coreos/rkt/pull/3092)). This bumps rkt-builder version to 1.0.2, in order to work with
seccomp filtering.
- export: test export for multi-app pods ([#3075](https://github.com/coreos/rkt/pull/3075)).
- Documentation updates: ([#3146](https://github.com/coreos/rkt/pull/3146), [#2954](https://github.com/coreos/rkt/pull/2954), [#3128](https://github.com/coreos/rkt/pull/3128), [#2953](https://github.com/coreos/rkt/pull/2953), [#3103](https://github.com/coreos/rkt/pull/3103), [#3087](https://github.com/coreos/rkt/pull/3087), [#3097](https://github.com/coreos/rkt/pull/3097), [#3096](https://github.com/coreos/rkt/pull/3096), [#3095](https://github.com/coreos/rkt/pull/3095), [#3089](https://github.com/coreos/rkt/pull/3089))

## 1.13.0

This release introduces support for exporting single applications out of multi-app pods. Moreover, it adds additional support to control device manipulation inside pods. Finally all runtime security features can now be optionally disabled at the pod level via new insecure options. This version also contains multiple bugfixes and supports Go 1.7.

### New features and UX changes

- export: name flag for exporting multi-app pods ([#3030](https://github.com/coreos/rkt/pull/3030)).
- stage1: limit device node creation/reading/writing with DevicePolicy= and DeviceAllow= ([#3027](https://github.com/coreos/rkt/pull/3027), [#3058](https://github.com/coreos/rkt/pull/3058)).
- rkt: implements --insecure-options={capabilities,paths,seccomp,run-all} ([#2983](https://github.com/coreos/rkt/pull/2983)).

#### Bug fixes

- kvm: use a properly formatted comment for iptables chains ([#3038](https://github.com/coreos/rkt/pull/3038)). rkt was using the chain name as comment, which could lead to confusion.
- pkg/label: supply mcsdir as function argument to InitLabels() ([#3045](https://github.com/coreos/rkt/pull/3045)).
- api_service: improve machined call error output ([#3059](https://github.com/coreos/rkt/pull/3059)).
- general: fix old appc/spec version in various files ([#3055](https://github.com/coreos/rkt/pull/3055)).
- rkt/pubkey: use custom http client including timeout ([#3084](https://github.com/coreos/rkt/pull/3084)).
- dist: remove quotes from rkt-api.service ExecStart ([#3079](https://github.com/coreos/rkt/pull/3079)).
- build: multiple fixes ([#3042](https://github.com/coreos/rkt/pull/3042), [#3041](https://github.com/coreos/rkt/pull/3041), [#3046](https://github.com/coreos/rkt/pull/3046)).
- configure: disable tests on host flavor with systemd <227 ([#3047](https://github.com/coreos/rkt/pull/3047)).

#### Other changes

- travis: add go 1.7, bump go 1.5/1.6 ([#3077](https://github.com/coreos/rkt/pull/3077)).
- api_service: Add lru cache to cache image info ([#2910](https://github.com/coreos/rkt/pull/2910)).
- scripts: add curl as build dependency ([#3070](https://github.com/coreos/rkt/pull/3070)).
- vendor: use appc/spec 0.8.6 and k8s.io/kubernetes v1.3.0 ([#3063](https://github.com/coreos/rkt/pull/3063)).
- common: use fileutil.IsExecutable() ([#3023](https://github.com/coreos/rkt/pull/3023)).
- build: Stop printing irrelevant invalidation messages ([#3050](https://github.com/coreos/rkt/pull/3050)).
- build: Make generating clean files simpler to do ([#3057](https://github.com/coreos/rkt/pull/3057)).
- Documentation: misc changes ([#3053](https://github.com/coreos/rkt/pull/3053), [#2911](https://github.com/coreos/rkt/pull/2911), [#3035](https://github.com/coreos/rkt/pull/3035), [#3036](https://github.com/coreos/rkt/pull/3036), [#3037](https://github.com/coreos/rkt/pull/3037), [#2945](https://github.com/coreos/rkt/pull/2945), [#3083](https://github.com/coreos/rkt/pull/3083), [#3076](https://github.com/coreos/rkt/pull/3076), [#3033](https://github.com/coreos/rkt/pull/3033), [#3064](https://github.com/coreos/rkt/pull/3064), [#2932](https://github.com/coreos/rkt/pull/2932)).
- functional tests: misc fixes ([#3049](https://github.com/coreos/rkt/pull/3049)).

## 1.12.0

This release introduces support for seccomp filtering via two new seccomp isolators. It also gives a boost to api-service performance by introducing manifest caching. Finally it fixes several regressions related to Docker images handling.

#### New features and UX changes

- cli: rename `--cap-retain` and `--cap-remove` to `--caps-*` ([#2994](https://github.com/coreos/rkt/pull/2994)).
- stage1: apply seccomp isolators ([#2753](https://github.com/coreos/rkt/pull/2753)). This introduces support for appc seccomp isolators.
- scripts: add /etc/rkt owned by group rkt-admin in setup-data-dir.sh ([#2944](https://github.com/coreos/rkt/pull/2944)).
- rkt: add `--caps-retain` and `--caps-remove` to prepare ([#3007](https://github.com/coreos/rkt/pull/3007)).
- store: allow users in the rkt group to delete images ([#2961](https://github.com/coreos/rkt/pull/2961)).
- api_service: cache pod manifest ([#2891](https://github.com/coreos/rkt/pull/2891)). Manifest caching considerably improves api-service performances.
- store: tell the user to run as root on db update ([#2966](https://github.com/coreos/rkt/pull/2966)).
- stage1: disabling cgroup namespace in systemd-nspawn ([#2989](https://github.com/coreos/rkt/pull/2989)). For more information see [systemd#3589](https://github.com/systemd/systemd/pull/3589).
- fly: copy rkt-resolv.conf in the app ([#2982](https://github.com/coreos/rkt/pull/2982)).
- store: decouple aci store and treestore implementations ([#2919](https://github.com/coreos/rkt/pull/2919)).
- store: record ACI fetching information ([#2960](https://github.com/coreos/rkt/pull/2960)).

#### Bug fixes
- stage1/init: fix writing of /etc/machine-id ([#2977](https://github.com/coreos/rkt/pull/2977)).
- rkt-monitor: multiple fixes ([#2927](https://github.com/coreos/rkt/pull/2927), [#2988](https://github.com/coreos/rkt/pull/2988)).
- rkt: don't errwrap cli_apps errors ([#2958](https://github.com/coreos/rkt/pull/2958)).
- pkg/tar/chroot: avoid errwrap in function called by multicall ([#2997](https://github.com/coreos/rkt/pull/2997)).
- networking: apply CNI args to the default networks as well ([#2985](https://github.com/coreos/rkt/pull/2985)).
- trust: provide InsecureSkipTLSCheck to pubkey manager ([#3016](https://github.com/coreos/rkt/pull/3016)).
- api_service: update grpc version ([#3015](https://github.com/coreos/rkt/pull/3015)).
- fetcher: httpcaching fixes ([#2965](https://github.com/coreos/rkt/pull/2965)).

#### Other changes
- build,stage1/init: set interpBin at build time for src flavor ([#2978](https://github.com/coreos/rkt/pull/2978)).
- common: introduce RemoveEmptyLines() ([#3004](https://github.com/coreos/rkt/pull/3004)).
- glide: update docker2aci to v0.12.3 ([#3026](https://github.com/coreos/rkt/pull/3026)). This fixes multiple bugs in layers ordering for Docker images.
- glide: update go-systemd to v11 ([#2970](https://github.com/coreos/rkt/pull/2970)). This fixes a buggy corner-case in journal seeking (implicit seek to head).
- docs: document capabilities overriding ([#2917](https://github.com/coreos/rkt/pull/2917), [#2991](https://github.com/coreos/rkt/pull/2991)).
- issue template: add '\n' to the end of environment output ([#3008](https://github.com/coreos/rkt/pull/3008)).
- functional tests: multiple fixes ([#2999](https://github.com/coreos/rkt/pull/2999), [#2979](https://github.com/coreos/rkt/pull/2979), [#3014](https://github.com/coreos/rkt/pull/3014)).

## 1.11.0

This release sets the ground for the new upcoming KVM qemu flavor. It adds support for exporting a pod to an ACI including all modifications. The rkt API service now also supports systemd socket activation. Finally we have diagnostics back, helping users to find out why their app failed to execute.

#### New features
- KVM: Hypervisor support for KVM flavor focusing on qemu ([#2684](https://github.com/coreos/rkt/pull/2684)). This provides a generic mechanism to use different kvm hypervisors (such as lkvm, qemu-kvm).
- rkt: add command to export a pod to an aci ([#2889](https://github.com/coreos/rkt/pull/2889)). Adds a new `export` command to rkt which generates an ACI from a pod; saving any changes made to the pod.
- rkt/api: detect when run as a `systemd.socket(5)` service ([#2916](https://github.com/coreos/rkt/pull/2916)). This allows rkt to run as a systemd socket-based unit.
- rkt/stop: implement `--uuid-file` ([#2902](https://github.com/coreos/rkt/pull/2902)). So the user can use the value saved on rkt run with `--uuid-file-save`.

#### Bug fixes
- scripts/glide-update: ensure running from $GOPATH ([#2885](https://github.com/coreos/rkt/pull/2885)). glide is confused when it's not running with the rkt repository inside $GOPATH.
- store: fix missing shared storelock acquisition on NewStore ([#2896](https://github.com/coreos/rkt/pull/2896)).
- store,rkt: fix fd leaks ([#2906](https://github.com/coreos/rkt/pull/2906)). Close db lock on store close. If we don't do it, there's a fd leak everytime we open a new Store, even if it was closed.
- stage1/enterexec: remove trailing `\n` in environment variables ([#2901](https://github.com/coreos/rkt/pull/2901)). Loading environment retained the new line character (`\n`), this produced an incorrect evaluation of the environment variables.
- stage1/gc: skip cleaning our own cgroup ([#2914](https://github.com/coreos/rkt/pull/2914)).
- api_service/log: fix file descriptor leak in GetLogs() ([#2930](https://github.com/coreos/rkt/pull/2930)).
- protobuf: fix protoc-gen-go build with vendoring ([#2913](https://github.com/coreos/rkt/pull/2913)).
- build: fix x86 builds ([#2926](https://github.com/coreos/rkt/pull/2926)). This PR fixes a minor issue which leads to x86 builds failing.
- functional tests: add some more volume/mount tests ([#2903](https://github.com/coreos/rkt/pull/2903)).
- stage1/init: link pod's journal in kvm flavor ([#2934](https://github.com/coreos/rkt/pull/2934)). In nspawn flavors, nspawn creates a symlink from `/var/log/journal/${machine-id}` to the pod's journal directory. In kvm we need to do the link ourselves.
- build: Build system fixes ([#2938](https://github.com/coreos/rkt/pull/2938)). This should fix the `expr: syntax error` and useless rebuilds of network plugins.

#### Other changes
- stage1: diagnostic functionality for rkt run ([#2872](https://github.com/coreos/rkt/pull/2872)). If the app exits with `ExecMainStatus == 203`, the app's reaper runs the diagnostic tool and prints the output on stdout. systemd sets `ExecMainstatus` to EXIT_EXEC (203) when execve() fails.
- build: add support for more architectures at configure time ([#2907](https://github.com/coreos/rkt/pull/2907)).
- stage1: update coreos image to 1097.0.0 ([#2884](https://github.com/coreos/rkt/pull/2884)). This is needed for a recent enough version of libseccomp (2.3.0), with support for new syscalls (eg. getrandom).
- api: By adding labels to the image itself, we don't need to pass the manifest to filter function ([#2909](https://github.com/coreos/rkt/pull/2909)). api: Add labels to pod and image type.
- api: optionally build systemd-journal support ([#2868](https://github.com/coreos/rkt/pull/2868)). This introduces a 'sdjournal' tag and corresponding stubs in api_service, turning libsystemd headers into a soft-dependency.
- store: simplify db locking and functions ([#2897](https://github.com/coreos/rkt/pull/2897)). Instead of having a file lock to handle inter process locking and a sync.Mutex to handle locking between multiple goroutines, just create, lock and close a new file lock at every db.Do function.
- stage1/enterexec: Add entry to ASSCB_EXTRA_HEADERS ([#2924](https://github.com/coreos/rkt/pull/2924)). Added entry to ASSCB_EXTRA_HEADERS for better change tracking.
- build: use rkt-builder ACI ([#2923](https://github.com/coreos/rkt/pull/2923)).
- Add hidden 'image fetch' next to the existing 'fetch' option ([#2860](https://github.com/coreos/rkt/pull/2860)).
- stage1: prepare-app: don't mount /sys if path already used ([#2888](https://github.com/coreos/rkt/pull/2888)). When users mount /sys or a sub-directory of /sys as a volume, prepare-app should not mount /sys: that would mask the volume provided by users.
- build,stage1/init: set interpBin at build time to fix other architecture builds (e.g. x86) ([#2950](https://github.com/coreos/rkt/pull/2950)).
- functional tests: re-purpose aws.sh for generating AMIs ([#2736](https://github.com/coreos/rkt/pull/2736)).
- rkt: Add `--cpuprofile` `--memprofile` for profiling rkt ([#2887](https://github.com/coreos/rkt/pull/2887)). Adds two hidden global flags and documentation to enable profiling rkt.
- functional test: check PATH variable for trailer `\n` character ([#2942](https://github.com/coreos/rkt/pull/2942)).
- functional tests: disable TestVolumeSysfs on kvm ([#2941](https://github.com/coreos/rkt/pull/2941)).
- Documentation updates ([#2918](https://github.com/coreos/rkt/pull/2918))

#### Library updates

- glide: update docker2aci to v0.12.1 ([#2873](https://github.com/coreos/rkt/pull/2873)). Includes support for the docker image format v2.2 and OCI image format and allows fetching via digest.

## 1.10.1

This is a minor bug fix release.

#### Bug fixes
- rkt/run: handle malformed environment files ([#2901](https://github.com/coreos/rkt/pull/2901))
- stage1/enterexec: remove trailing `\n` in environment variables ([#2901](https://github.com/coreos/rkt/pull/2901))

## v1.10.0
This release introduces a number of important features and improvements:

- ARM64 support
- A new subcommand `rkt stop` to gracefully stop running pods
- native Go vendoring with Glide
- rkt is now packaged for openSUSE Tumbleweed and Leap

#### New features
- Add ARM64 support ([#2758](https://github.com/coreos/rkt/pull/2758)). This enables ARM64 cross-compliation, fly, and stage1-coreos.
- Replace Godep with Glide, introduce native Go vendoring ([#2735](https://github.com/coreos/rkt/pull/2735)).
- rkt: rkt stop ([#2438](https://github.com/coreos/rkt/pull/2438)). Cleanly stops a running pod. For systemd-nspawn, sends a SIGTERM. For kvm, executes `systemctl halt`.

#### Bug fixes
- stage1/fly: respect runtimeApp App's MountPoints ([#2852](https://github.com/coreos/rkt/pull/2852)). Fixes #2846.
- run: fix sandbox-side metadata service to comply to appc v0.8.1 ([#2863](https://github.com/coreos/rkt/pull/2863)). Fixes #2621.

#### Other changes
- build directory layout change ([#2758](https://github.com/coreos/rkt/pull/2758)): The rkt binary and stage1 image files have been moved from the 'bin' sub-directory to the 'target/bin' sub-directory.
- networking/kvm: add flannel default gateway parsing ([#2859](https://github.com/coreos/rkt/pull/2859)).
- stage1/enterexec: environment file with '\n' as separator (systemd style) ([#2839](https://github.com/coreos/rkt/pull/2839)).
- pkg/tar: ignore global extended headers ([#2847](https://github.com/coreos/rkt/pull/2847)).
- pkg/tar: remove errwrap ([#2848](https://github.com/coreos/rkt/pull/2848)).
- tests: fix abuses of appc types.Isolator ([#2840](https://github.com/coreos/rkt/pull/2840)).
- common: remove unused GetImageIDs() ([#2834](https://github.com/coreos/rkt/pull/2834)).
- common/cgroup: add mountFsRO() helper function ([#2829](https://github.com/coreos/rkt/pull/2829)).
- Documentation updates ([#2732](https://github.com/coreos/rkt/pull/2732), [#2869](https://github.com/coreos/rkt/pull/2869), [#2810](https://github.com/coreos/rkt/pull/2810), [#2865](https://github.com/coreos/rkt/pull/2865), [#2825](https://github.com/coreos/rkt/pull/2825), [#2841](https://github.com/coreos/rkt/pull/2841), [#2732](https://github.com/coreos/rkt/pull/2732))

#### Library updates
- glide: bump ql to v1.0.4 ([#2875](https://github.com/coreos/rkt/pull/2875)). It fixes an occassional panic when doing GC.
- glide: bump gopsutils to 2.1 ([#2876](https://github.com/coreos/rkt/pull/2876)). To include https://github.com/shirou/gopsutil/pull/194 (this adds ARM aarch64 support)
- vendor: update appc/spec to 0.8.5 ([#2854](https://github.com/coreos/rkt/pull/2854)).

## v1.9.1

This is a minor bug fix release.

#### Bug fixes

- Godeps: update go-systemd ([#2837](https://github.com/coreos/rkt/pull/2837)). go-systemd v10 fixes a panic-inducing bug due to returning incorrect
Read() length values.
- stage1/fly: use 0755 to create mountpaths ([#2836](https://github.com/coreos/rkt/pull/2836)). This will allow any user to list the content directories. It does not
have any effect on the permissions on the mounted files itself.

## v1.9.0

This release focuses on bug fixes and developer tooling and UX improvements.

#### New features and UX changes

- rkt/run: added --set-env-file switch and priorities for environments ([#2816](https://github.com/coreos/rkt/pull/2816)). --set-env-file gets an environment variables file path in the format "VAR=VALUE\n...".
- run: add --cap-retain and --cap-remove ([#2771](https://github.com/coreos/rkt/pull/2771)).
- store: print more information on rm as non-root ([#2805](https://github.com/coreos/rkt/pull/2805)).
- Documentation/vagrant: use rkt binary for getting started ([#2808](https://github.com/coreos/rkt/pull/2808)). 
- docs: New file in documentation - instruction for new developers in rkt ([#2639](https://github.com/coreos/rkt/pull/2639)).
- stage0/trust: change error message if prefix/root flag missing ([#2661](https://github.com/coreos/rkt/pull/2661)).

#### Bug fixes

- rkt/uuid: fix match when uuid is an empty string ([#2807](https://github.com/coreos/rkt/pull/2807)).
- rkt/api_service: fix fly pods ([#2799](https://github.com/coreos/rkt/pull/2799)).
- api/client_example: fix panic if pod has no apps ([#2766](https://github.com/coreos/rkt/pull/2766)). Fixes the concern expressed in https://github.com/coreos/rkt/pull/2763#discussion_r66409260
- api_service: wait until a pod regs with machined ([#2788](https://github.com/coreos/rkt/pull/2788)).

#### Other changes

- stage1: update coreos image to 1068.0.0 ([#2821](https://github.com/coreos/rkt/pull/2821)). 
- KVM: Update LKVM patch to mount with mmap mode ([#2795](https://github.com/coreos/rkt/pull/2795)).
- stage1: always write /etc/machine-id ([#2440](https://github.com/coreos/rkt/pull/2440)). Prepare rkt for systemd-v230 in stage1.
- stage1/prepare-app: always adjust /etc/hostname ([#2761](https://github.com/coreos/rkt/pull/2761)). 

## v1.8.0

This release focuses on stabilizing the API service, fixing multiple issues in the logging subsystem.

#### New features and UX changes

- api: GetLogs: improve client example with 'Follow' ([#2747](https://github.com/coreos/rkt/pull/2747)).
- kvm: add proxy arp support to macvtap ([#2715](https://github.com/coreos/rkt/pull/2715)).
- stage0/config: add a CLI flag to pretty print json ([#2745](https://github.com/coreos/rkt/pull/2745)).
- stage1: make /proc/bus/ read-only ([#2743](https://github.com/coreos/rkt/pull/2743)).

#### Bug fixes

- api: GetLogs: use the correct type in LogsStreamWriter ([#2744](https://github.com/coreos/rkt/pull/2744)).
- api: fix service panic on incomplete pods ([#2739](https://github.com/coreos/rkt/pull/2739)).
- api: Fix the GetLogs() when appname is given ([#2763](https://github.com/coreos/rkt/pull/2763)).
- pkg/selinux: various fixes ([#2723](https://github.com/coreos/rkt/pull/2723)).
- pkg/fileutil: don't remove the cleanSrc if it equals '.' ([#2731](https://github.com/coreos/rkt/pull/2731)).
- stage0: remove superfluous error verbs ([#2750](https://github.com/coreos/rkt/pull/2750)).

#### Other changes

- Godeps: bump go-systemd ([#2754](https://github.com/coreos/rkt/pull/2754)). Fixes a panic on the api-service when calling GetLogs().
- Documentation updates ([#2756](https://github.com/coreos/rkt/pull/2756), [#2741](https://github.com/coreos/rkt/pull/2741), [#2737](https://github.com/coreos/rkt/pull/2737), [#2742](https://github.com/coreos/rkt/pull/2742), [#2730](https://github.com/coreos/rkt/pull/2730), [#2729](https://github.com/coreos/rkt/pull/2729))
- Test improvements ([#2726](https://github.com/coreos/rkt/pull/2726)).

## v1.7.0

This release introduces some new security features, including a "no-new-privileges" isolator and initial (partial) restrictions on /proc and /sys access.
Cgroups handling has also been improved with regards to setup and cleaning. Many bugfixes and new documentation are included too.

#### New features and UX changes

- stage1: implement no-new-privs linux isolator ([#2677](https://github.com/coreos/rkt/pull/2677)).
- stage0: disable OverlayFS by default when working on ZFS ([#2600](https://github.com/coreos/rkt/pull/2600)).
- stage1: (partially) restrict access to procfs and sysfs paths ([#2683](https://github.com/coreos/rkt/pull/2683)).
- stage1: clean up pod cgroups on GC ([#2655](https://github.com/coreos/rkt/pull/2655)).
- stage1/prepare-app: don't mount /sys/fs/cgroup in stage2 ([#2681](https://github.com/coreos/rkt/pull/2681)).
- stage0: complain and abort on conflicting CLI flags ([#2666](https://github.com/coreos/rkt/pull/2666)).
- stage1: update CoreOS image signing key ([#2659](https://github.com/coreos/rkt/pull/2659)).
- api_service: Implement GetLogs RPC request ([#2662](https://github.com/coreos/rkt/pull/2662)).
- networking: update to CNI v0.3.0 ([#3696](https://github.com/coreos/rkt/pull/2696)).

#### Bug fixes

- api: fix image size reporting ([#2501](https://github.com/coreos/rkt/pull/2501)).
- build: fix build failures on manpages/bash-completion target due to missing GOPATH ([#2646](https://github.com/coreos/rkt/pull/2646)).
- dist: fix "other" permissions so rkt list can work without root/rkt-admin ([#2698](https://github.com/coreos/rkt/pull/2698)).
- kvm: fix logging network plugin type ([#2635](https://github.com/coreos/rkt/pull/2635)).
- kvm: transform flannel network to allow teardown ([#2647](https://github.com/coreos/rkt/pull/2647)).
- rkt: fix panic on rm a non-existing pod with uuid-file ([#2679](https://github.com/coreos/rkt/pull/2679)).
- stage1/init: work around `cgroup/SCM_CREDENTIALS` race ([#2645](https://github.com/coreos/rkt/pull/2645)).
- gc: mount stage1 on GC ([#2704](https://github.com/coreos/rkt/pull/2704)).
- stage1: fix network files leak on GC ([#2319](https://github.com/coreos/rkt/issues/2319)).

#### Other changes

- deps: remove unused dependencies ([#2703](https://github.com/coreos/rkt/pull/2703)).
- deps: appc/spec, k8s, protobuf updates ([#2697](https://github.com/coreos/rkt/pull/2697)).
- deps: use tagged release of github.com/shirou/gopsutil ([#2705](https://github.com/coreos/rkt/pull/2705)).
- deps: bump docker2aci to v0.11.1 ([#2719](https://github.com/coreos/rkt/pull/2719)).
- Documentation updates ([#2620](https://github.com/coreos/rkt/pull/2620), [#2700](https://github.com/coreos/rkt/pull/2700), [#2637](https://github.com/coreos/rkt/pull/2637), [#2591](https://github.com/coreos/rkt/pull/2591), [#2651](https://github.com/coreos/rkt/pull/2651), [#2699](https://github.com/coreos/rkt/pull/2699), [#2631](https://github.com/coreos/rkt/pull/2631)).
- Test improvements ([#2587](https://github.com/coreos/rkt/pull/2587), [#2656](https://github.com/coreos/rkt/pull/2656), [#2676](https://github.com/coreos/rkt/pull/2676), [#2554](https://github.com/coreos/rkt/pull/2554), [#2690](https://github.com/coreos/rkt/pull/2690), [#2674](https://github.com/coreos/rkt/pull/2674), [#2665](https://github.com/coreos/rkt/pull/2665), [#2649](https://github.com/coreos/rkt/pull/2649), [#2643](https://github.com/coreos/rkt/pull/2643), [#2637](https://github.com/coreos/rkt/pull/2637), [#2633](https://github.com/coreos/rkt/pull/2633)).

## v1.6.0

This release focuses on security enhancements. It provides additional isolators, creating a new mount namespace per app. Also a new version of CoreOS 1032.0.0 with systemd v229 is being used in stage1.

#### New features and UX changes

- stage1: implement read-only rootfs ([#2624](https://github.com/coreos/rkt/pull/2624)). Using the Pod manifest readOnlyRootFS option mounts the rootfs of the app as read-only using systemd-exec unit option ReadOnlyDirectories, see [appc/spec](https://github.com/appc/spec/blob/master/spec/pods.md#pod-manifest-schema).
- stage1: capabilities: implement both remain set and remove set ([#2589](https://github.com/coreos/rkt/pull/2589)). It follows the [Linux Isolators semantics from the App Container Executor spec](https://github.com/appc/spec/blob/master/spec/ace.md#linux-isolators), as modified by [appc/spec#600](https://github.com/appc/spec/pull/600).
- stage1/init: create a new mount ns for each app ([#2603](https://github.com/coreos/rkt/pull/2603)). Up to this point, you could escape the app's chroot easily by using a simple program downloaded from the internet [1](http://www.unixwiz.net/techtips/chroot-practices.html). To avoid this, we now create a new mount namespace per each app.
- api: Return the pods even when we failed getting information about them ([#2593](https://github.com/coreos/rkt/pull/2593)).
- stage1/usr_from_coreos: use CoreOS 1032.0.0 with systemd v229 ([#2514](https://github.com/coreos/rkt/pull/2514)).

#### Bug fixes

- kvm: fix flannel network info ([#2625](https://github.com/coreos/rkt/pull/2625)). It wasn't saving the network information on disk.
- stage1: Machine name wasn't being populated with the full UUID ([#2575](https://github.com/coreos/rkt/pull/2575)).
- rkt: Some simple arg doc string fixes ([#2588](https://github.com/coreos/rkt/pull/2588)). Remove some unnecessary indefinite articles from the start of argument doc strings and fixes the arg doc string for run-prepared's --interactive flag.
- stage1: Fix segfault in enterexec ([#2608](https://github.com/coreos/rkt/pull/2608)). This happened if rkt enter was executed without the TERM environment variable set.
- net: fix port forwarding behavior with custom CNI ipMasq'ed networks and allow different hostPort:podPort combinations ([#2387](https://github.com/coreos/rkt/pull/2387)).
- stage0: check and create /etc ([#2599](https://github.com/coreos/rkt/pull/2599)). Checks '/etc' before writing to '/etc/rkt-resolv.conf' and creates it with default permissions if it doesn't exist.

#### Other changes

- godep: update cni to v0.2.3 ([#2618](https://github.com/coreos/rkt/pull/2618)).
- godep: update appc/spec to v0.8.1 ([#2623](https://github.com/coreos/rkt/pull/2623), [#2611](https://github.com/coreos/rkt/pull/2611)).
- dist: Update tmpfiles to create /etc/rkt ([#2472](https://github.com/coreos/rkt/pull/2472)). By creating this directory, users can run `rkt trust` without being root, if the user is in the rkt group.
- Invoke gofmt with simplify-code flag ([#2489](https://github.com/coreos/rkt/pull/2489)). Enables code simplification checks of gofmt.
- Implement composable uid/gid generators ([#2510](https://github.com/coreos/rkt/pull/2510)). This cleans up the code a bit and implements uid/gid functionality for rkt fly.
- stage1: download CoreOS over HTTPS ([#2568](https://github.com/coreos/rkt/pull/2568)). 
- Documentation updates ([#2555](https://github.com/coreos/rkt/pull/2555), [#2609](https://github.com/coreos/rkt/pull/2609), [#2605](https://github.com/coreos/rkt/pull/2605), [#2578](https://github.com/coreos/rkt/pull/2578), [#2614](https://github.com/coreos/rkt/pull/2614), [#2579](https://github.com/coreos/rkt/pull/2579), [#2570](https://github.com/coreos/rkt/pull/2570)).
- Test improvements ([#2613](https://github.com/coreos/rkt/pull/2613), [#2566](https://github.com/coreos/rkt/pull/2566), [#2508](https://github.com/coreos/rkt/pull/2508)).

## v1.5.1

This release is a minor bug fix release.

#### Bug fixes

- rkt: fix bug where rkt errored out if the default data directory didn't exist [#2557](https://github.com/coreos/rkt/pull/2557).
- kvm: fix docker volume semantics ([#2558](https://github.com/coreos/rkt/pull/2558)). When a Docker image exposes a mount point that is not mounted by a host volume, Docker volume semantics expect the files in the directory to be available to the application. This was not working properly in the kvm flavor and it's fixed now.
- kvm: fix net long names ([#2543](https://github.com/coreos/rkt/pull/2543)). Handle network names that are longer than the maximum allowed by iptables in the kvm flavor.

#### Other changes

- minor tests and clean-ups ([#2551](https://github.com/coreos/rkt/pull/2551)).

## v1.5.0

This release switches to pure systemd for running apps within a pod. This lays the foundation to implement enhanced isolation capabilities. For example, starting with v1.5.0, apps are started with more restricted capabilities. User namespace support and the KVM stage1 are not experimental anymore. Resource usage can be benchmarked using the new rkt-monitor tool.

#### New features and UX changes

- stage1: replace appexec with pure systemd ([#2493](https://github.com/coreos/rkt/pull/2493)). Replace functionality implemented in appexec with equivalent systemd options. This allows restricting the capabilities granted to apps in a pod and makes enabling other security features (per-app mount namespaces, seccomp filters...) easier.
- stage1: restrict capabilities granted to apps ([#2493](https://github.com/coreos/rkt/pull/2493)). Apps in a pod receive now a [smaller set of capabilities](https://github.com/coreos/rkt/blob/v1.5.0/stage1/init/common/pod.go#L67).
- rkt/image: render images on fetch ([#2398](https://github.com/coreos/rkt/pull/2398)). On systems with overlay fs support, rkt was delaying rendering images to the tree store until they were about to run for the first time which caused that first run to be slow for big images. When fetching as root, render the images right away so the first run is faster.

#### Bug fixes

- kvm: fix mounts regression ([#2530](https://github.com/coreos/rkt/pull/2530)). Cause - AppRootfsPath called with local "root" value was adding
stage1/rootfs twice. After this change this is made properly.
- rkt/image: strip "Authorization" on redirects to a different host ([#2465](https://github.com/coreos/rkt/pull/2465)). We now don't pass the "Authorization" header if the redirect goes to a different host, it can leak sensitive information to unexpected third parties.
- stage1/init: interpret the string "root" as UID/GID 0 ([#2458](https://github.com/coreos/rkt/pull/2458)). This is a special case and it should work even if the image doesn't have /etc/passwd or /etc/group.

#### Improved documentation

- added benchmarks folder, benchmarks for v1.4.0 ([#2520](https://github.com/coreos/rkt/pull/2520)). Added the `Documentation/benchmarks` folder which includes a README that describes how rkt-monitor works and how to use it, and a file detailing the results of running rkt-monitor on each current workload with rkt v1.4.0.
- minor documentation fixes ([#2455](https://github.com/coreos/rkt/pull/2455), [#2528](https://github.com/coreos/rkt/pull/2528), [#2511](https://github.com/coreos/rkt/pull/2511)).

#### Testing

- kvm: enable functional tests for kvm ([#2007](https://github.com/coreos/rkt/pull/2007)). This includes initial support for running functional tests on the `kvm` flavor.

#### Other changes

- benchmarks: added rkt-monitor benchmarks ([#2324](https://github.com/coreos/rkt/pull/2324)). This includes the code for a golang binary that can start rkt and watch its resource usage and bash scripts for generating a handful of test scenarios.
- scripts: generate a Debian Sid ACI instead of using the Docker hub image ([#2471](https://github.com/coreos/rkt/pull/2471)). This is the first step to having an official release builder.
- pkg/sys: add SYS_SYNCFS definition for ppc64/ppc64le ([#2443](https://github.com/coreos/rkt/pull/2443)). Added missing SYS_SYNCFS definition for ppc64 and ppc64le, fixing build failures on those architectures.
- userns: not experimental anymore ([#2486](https://github.com/coreos/rkt/pull/2486)). Although it requires doing a recursive chown for each app, user namespaces work fine and shouldn't be marked as experimental.
- kvm: not experimental anymore ([#2485](https://github.com/coreos/rkt/pull/2485)). The kvm flavor was initially introduced in rkt v0.8.0, no reason to mark it as experimental.

## v1.4.0

This release includes a number of new features and bugfixes like a new config subcommand, man page, and bash completion generation during build time.

#### New features and UX changes

- config: add config subcommand ([#2405](https://github.com/coreos/rkt/pull/2405)). This new subcommand prints the current rkt configuration. It can be used to get i.e. authentication credentials. See rkt's [config subcommand](https://github.com/coreos/rkt/blob/master/Documentation/subcommands/config.md) documentation.

- run: add `--user`/`--group` app flags to `rkt run` and `rkt prepare` allowing to override the user and group specified in the image manifest ([#2419](https://github.com/coreos/rkt/pull/2419)).

- gc: Add flag 'mark-only' to mark garbage pods without deleting them ([#2400](https://github.com/coreos/rkt/pull/2400), [#2402](https://github.com/coreos/rkt/pull/2402)). This new flag moves exited/aborted pods to the exited-garbage/garbage directory but does not delete them. A third party application can use `rkt gc --mark-only=true` to mark exited pods as garbage without deleting them.

- kvm: Add support for app capabilities limitation ([#2222](https://github.com/coreos/rkt/pull/2222)). By default kvm flavor has got enabled every capability inside pod. This patch adds support for a restricted set of capabilities inside a kvm flavor of rkt.

- stage1/init: return exit code 1 on error ([#2383](https://github.com/coreos/rkt/pull/2383)). On error, stage1/init was returning a non-zero value between 1 and 7. This change makes it return status code 1 only.

- api: Add 'CreatedAt', 'StartedAt' in pod's info returned by api service. ([#2377](https://github.com/coreos/rkt/pull/2377)).

#### Improved documentation

- Minor documentation fixes ([#2413](https://github.com/coreos/rkt/pull/2413), [#2395](https://github.com/coreos/rkt/pull/2395), [#2231](https://github.com/coreos/rkt/pull/2231)).

- functional tests: Add new test with systemd-proxyd ([#2257](https://github.com/coreos/rkt/pull/2257)). Adds a new test and documentation how to use systemd-proxyd with rkt pods.

#### Bug fixes

- kvm: refactor volumes support ([#2328](https://github.com/coreos/rkt/pull/2328)). This allows users to share regular files as volumes in addition to directories.

- kvm: fix rkt status ([#2415](https://github.com/coreos/rkt/pull/2415)). Fixes a regression bug were `rkt status` was no longer reporting the pid of the pod when using the kvm flavor.

- Build actool for the *build* architecture ([#2372](https://github.com/coreos/rkt/pull/2372)). Fixes a cross compilation issue with acbuild.

- rkt: calculate real dataDir path ([#2399](https://github.com/coreos/rkt/pull/2399)). Fixes garbage collection when the data directory specified by `--dir` contains a symlink component.

- stage1/init: fix docker volume semantics ([#2409](https://github.com/coreos/rkt/pull/2409)). Fixes a bug in docker volume semantics when rkt runs with the option `--pod-manifest`. When a Docker image exposes a mount point that is not mounted by a host volume, Docker volume semantics expect the files in the directory to be available to the application. This was partially fixed in rkt 1.3.0 via [#2315](https://github.com/coreos/rkt/pull/2315) but the bug remained when rkt runs with the option `--pod-manifest`. This is now fully fixed.

- rkt/image: check that discovery labels match manifest labels ([#2311](https://github.com/coreos/rkt/pull/2311)).

- store: fix multi process with multi goroutines race on db ([#2391](https://github.com/coreos/rkt/pull/2391)). This was a bug when multiple `rkt fetch` commands were executed concurrently.

- kvm: fix pid vs ppid usage ([#2396](https://github.com/coreos/rkt/pull/2396)). Fixes a bug in `rkt enter` in the kvm flavor causing an infinite loop.

- kvm: Fix connectivity issue in macvtap networks caused by macvlan NICs having incorrect names ([#2181](https://github.com/coreos/rkt/pull/2181)). 

- tests: TestRktListCreatedStarted: fix timing issue causing the test to fail on slow machines ([#2366](https://github.com/coreos/rkt/pull/2366)).

- rkt/image: remove redundant quotes in an error message ([#2379](https://github.com/coreos/rkt/pull/2379)).

- prepare: Support 'ondisk' verification skip as documented by [the global options](https://github.com/coreos/rkt/blob/master/Documentation/commands.md#global-options) ([#2376](https://github.com/coreos/rkt/pull/2376)). Prior to this commit, rkt prepare would check the ondisk image even if the `--insecure-options=ondisk` flag was provided. This corrects that.

#### Other changes

- tests: skip TestSocketProxyd when systemd-socket-proxyd is not installed ([#2436](https://github.com/coreos/rkt/pull/2436)).

- tests: TestDockerVolumeSemantics: more tests with symlinks ([#2394](https://github.com/coreos/rkt/pull/2394)).

- rkt: Improve build shell script used in [continuous integration](https://github.com/coreos/rkt/blob/master/tests/README.md) ([#2394](https://github.com/coreos/rkt/pull/2394)).

- protobuf: generate code using a script ([#2382](https://github.com/coreos/rkt/pull/2382)).

- Generate manpages ([#2373](https://github.com/coreos/rkt/pull/2373)). This adds support for generating rkt man pages using `make manpages` and the bash completion file using `make bash-completion`, see the note for packagers below.

- tests/aws.sh: add test for Fedora 24 ([#2340](https://github.com/coreos/rkt/pull/2340)).

#### Note for packagers

Files generated from sources are no longer checked-in the git repository. Instead, packagers should build them:

- Bash completion file, generated by `make bash-completion`
- Man pages, generated by `make manpages`

## v1.3.0

This release includes a number of new features and bugfixes like the long-awaited propagation of apps' exit status.

#### New features and UX changes

- Propagate exit status from apps inside the pod to rkt ([#2308](https://github.com/coreos/rkt/pull/2308)). Previously, if an app exited with a non-zero exit status, rkt's exit status would still be 0. Now, if an app fails, its exit status will be propagated to the outside. While this was partially implemented in some stage1 flavors since rkt v1.1.0, it now works in the default coreos flavor.
- Check signatures for stage1 images by default, especially useful when stage1 images are downloaded from the Internet ([#2336](https://github.com/coreos/rkt/pull/2336)).
 This doesn't affect the following cases:
  - The stage1 image is already in the store
  - The stage1 image is in the default directory configured at build time
  - The stage1 image is the default one and it is in the same directory as the rkt binary
- Allow downloading of insecure public keys with the `pubkey` insecure option ([#2278](https://github.com/coreos/rkt/pull/2278)).
- Implement Docker volume semantics ([#2315](https://github.com/coreos/rkt/pull/2315)). Docker volumes are initialized with the files in the image if they exist, unless a host directory is mounted there. Implement that behavior in rkt when it runs a Docker converted image.

#### API service

- Return the cgroup when getting information about running pods and add a new cgroup filter ([#2331](https://github.com/coreos/rkt/pull/2331)).

#### Bug fixes

- Avoid configuring more CPUs than the host has in the kvm flavor ([#2321](https://github.com/coreos/rkt/pull/2321)).
- Fix a bug where the proxy configuration wasn't forwarded to docker2aci ([docker2aci#147](https://github.com/appc/docker2aci/pull/147)).

#### Notes

- This release drops support for go1.4.

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
- [Stage1 systemd Architecture](https://github.com/coreos/rkt/blob/master/Documentation/devel/architecture.md) ([#1631](https://github.com/coreos/rkt/pull/1631))
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
