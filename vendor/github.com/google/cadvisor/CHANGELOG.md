# Changelog

### 0.28.3 (2017-12-7)
- Add timeout for docker calls
- Fix prometheus label consistency

### 0.28.2 (2017-11-21)
- Fix GPU init race condition

### 0.28.1 (2017-11-20)
- Add containerd support
- Fix fsnotify regression from 0.28.0
- Add on demand metrics

### 0.28.0 (2017-11-06)
- Add container nvidia GPU metrics
- Expose container memory max_usage_in_bytes
- Add container memory reservation to prometheus

### 0.27.1 (2017-09-06)
- Add CRI-O support

### 0.27.0 (2017-09-01)
- Fix journalctl leak
- Fix container memory rss
- Add hugepages support
- Fix incorrect CPU usage with 4.7 kernel
- OOM parser uses kmsg
- Add tmpfs support

### 0.26.1 (2017-06-21)
- Fix prometheus metrics.

### 0.26.0 (2017-05-31)
- Fix disk partition discovery for brtfs
- Add ZFS support
- Add UDP metrics (collection disabled by default)
- Improve diskio prometheus metrics
- Update Prometheus godeps to v0.8
- Add overlay2 storage driver support

### 0.25.0 (2017-03-09)
- Disable thin_ls due to excessive iops
- Ignore .mount cgroups, fixing dissappearing stats
- Fix wc goroutine leak
- Update aws-sdk-go dependency to 1.6.10
- Update to go 1.7 for releases

### 0.24.1 (2016-10-10)

- Fix issue with running cAdvisor in a container on some distributions.

### 0.24.0 (2016-09-19)

- Added host-level inode stats (total & available)
- Improved robustness to partial failures
- Metrics collector improvements
  - Added ability to directly use endpoints from the container itself
  - Allow SSL endpoint access
  - Ability to provide a certificate which is exposed to custom endpoints
- Lots of bug fixes, including:
  - Devicemapper thin_ls fixes
  - Prometheus metrics fixes
  - Fixes for missing stats (memory reservation, FS usage, etc.)

### 0.23.9 (2016-08-09)

- Cherry-pick release:
  - Ensure minimum kernel version for thin_ls

### 0.23.8 (2016-08-02)

- Cherry-pick release:
  - Prefix Docker labels & env vars in Prometheus metrics to prevent conflicts

### 0.23.7 (2016-07-18)

- Cherry-pick release:
  - Modify working set memory stats calculation

### 0.23.6 (2016-06-23)

- Cherry-pick release:
  - Updating inotify to fix memory leak v0.23 cherrypick

### 0.23.5 (2016-06-22)

- Cherry-pick release:
  - support LVM based device mapper storage drivers

### 0.23.4 (2016-06-16)
- Cherry-pick release:
  - Check for thin_is binary in path for devicemapper when using ThinPoolWatcher
  - Fix uint64 overflow issue for CPU stats

### 0.23.3 (2016-06-08)
- Cherry-pick release:
  - Cap the maximum consecutive du commands
  - Fix a panic when a prometheus endpoint ends with a newline

### 0.23.2 (2016-05-18)
- Handle kernel log rotation
- More rkt support: poll rkt service for new containers
- Better handling of partial failures when fetching subcontainers
- Devicemapper thin_ls support (requires Device Mapper kernel module and supporting utilities)

### 0.23.1 (2016-05-11)
- Add multi-container charts to the UI
- Add TLS options for Kafka storage driver
- Switch to official Docker client
- Systemd:
  - Ignore .mount cgroups on systemd
  - Better OOM monitoring
- Bug: Fix broken -disable_metrics flag
- Bug: Fix openstack identified as AWS
- Bug: Fix EventStore when limit is 0

### 0.23.0 (2016-04-21)
- Docker v1.11 support
- Preliminary rkt support
- Bug: Fix file descriptor leak

### 0.22.0 (2016-02-25)
- Disk usage calculation bug fixes
- Systemd integration bug fixes
- Instance ID support for Azure and AWS
- Limit number of custom metrics
- Support opt out for disk and network metrics

### 0.21.0 (2016-02-03)
- Support for filesystem stats with docker v1.10
- Bug fixes.

### 0.20.5 (2016-01-27)
- Breaking: Use uint64 for memory stats
- Bug: Fix devicemapper partition labelling
- Bug: Fix network stats when using new Docker network functionality
- Bug: Fix env var label mapping initialization
- Dependencies: libcontainer update

### 0.20.4 (2016-01-20)
- Godep updates

### 0.20.3 (2016-01-19)
- Bug fixes
- Jitter added to housekeeping to smooth CPU usage.

### 0.20.2 (2016-01-15)
- New v2.1 API with better filesystem stats
- Internal refactoring
- Bug fixes.

### 0.18.0 (2015-09-23)
- Large bunch of bug-fixes
- Fixed networking stats for newer docker versions using libnetwork.
- Added application-specific metrics

## 0.16.0 (2015-06-26)
- Misc fixes.

## 0.15.1 (2015-06-10)
- Fix longstanding memory leak.
- Fix UI on newest Chrome.

## 0.15.0 (2015-06-08)
- Expose multiple network intefaces in UI and API.
- Add support for XFS.
- Fixes in inotify watches.
- Fixes on PowerPC machines.
- Fixes for newer systems with systemd.
- Extra debuging informaiton in /validate.

## 0.14.0 (2015-05-21)
- Add process stats to container pages in the UI.
- Serve UI from relative paths (allows reverse proxying).
- Minor fixes to events API.
- Add bytes available to FS info.
- Adding Docker status and image information to UI.
- Basic Redis storage backend.
- Misc reliability improvements.

## 0.13.0 (2015-05-01)
- Added `--docker_only` to limit monitoring to only Docker containers.
- Added support for Docker labels.
- Added limit for events storage.
- Fixes for OOM event monitoring.
- Changed event type to a string in the API.
- Misc fixes.

## 0.12.0 (2015-04-15)
- Added support for Docker 1.6.
- Split OOM event into OOM kill and OOM.
- Made EventData a concrete type in returned events.
- Enabled CPU load tracking (experimental).

## 0.11.0 (2015-03-27)
- Export all stats as [Prometheus](https://prometheus.io/) metrics.
- Initial support for [events](docs/api.md): creation, deletion, and OOM.
- Adding machine UUID information.
- Beta release of the cAdvisor [2.0 API](docs/api_v2.md).
- Improve handling of error conditions.
- Misc fixes and improvements.

## 0.10.1 (2015-02-27)
- Disable OOM monitoring which is using too much CPU.
- Fix break in summary stats.

## 0.10.0 (2015-02-24)
- Adding Start and End time for ContainerInfoRequest.
- Various misc fixes.

## 0.9.0 (2015-02-06)
- Support for more network devices (all non-eth).
- Support for more partition types (btrfs, device-mapper, whole-disk).
- Added reporting of DiskIO stats.
- Adding container creation time to ContainerSpec.
- More robust handling of stats failures.
- Various misc fixes.

## 0.8.0 (2015-01-09)
- Added ethernet device information.
- Added machine-wide networking statistics.
- Misc UI fixes.
- Fixes for partially-isolated containers.

## 0.7.1 (2014-12-23)
- Avoid repeated logging of container errors.
- Handle non identify mounts for cgroups.

## 0.7.0 (2014-12-18)
- Support for HTTP basic auth.
- Added /validate to perform basic checks and determine support for cAdvisor.
- All stats in the UI are now updated.
- Added gauges for filesystem usage.
- Added device information to machine info.
- Fixes to container detection.
- Fixes for systemd detection.
- ContainerSpecs are now cached.
- Performance improvements.

## 0.6.2 (2014-11-20)
- Fixes for Docker API and UI endpoints.
- Misc UI bugfixes.

## 0.6.1 (2014-11-18)
- Bug fix in InfluxDB storage driver. Container name and hostname will be exported.

## 0.6.0 (2014-11-17)
- Adding /docker UI endpoint for Docker containers.
- Fixes around handling Docker containers.
- Performance enhancements.
- Embed all external dependencies.
- ContainerStats Go struct has been flattened. The wire format remains unchanged.
- Misc bugfixes and cleanups.

## 0.5.0 (2014-10-28)
- Added disk space stats. On by default for root, available on AUFS Docker containers.
- Introduced v1.2 remote API with new "docker" resource for Docker containers.
- Added "ContainerHints" file based interface to inject extra information about containers.

## 0.4.1 (2014-09-29)
- Support for Docker containers in systemd systems.
- Adding DiskIO stats
- Misc bugfixes and cleanups

## 0.4.0 (2014-09-19)
- Various performance enhancements: brings CPU usage down 85%+
- Implemented dynamic sampling through dynamic housekeeping.
- Memory storage driver is always on, BigQuery and InfluxDB are now optional storage backends.
- Fix for DNS resolution crashes when contacting InfluxDB.
- New containers are now detected using inotify.
- Added pprof HTTP endpoint.
- UI bugfixes.

## 0.3.0 (2014-09-05)
- Support for Docker with LXC backend.
- Added BigQuery storage driver.
- Performance and stability fixes for InfluxDB storage driver.
- UI fixes and improvements.
- Configurable stats gathering interval (default: 1s).
- Improvements to startup and CPU utilization.
- Added /healthz endpoint for determining whether cAdvisor is healthy.
- Bugfixes and performance improvements.

## 0.2.2 (2014-08-13)
- Improvements to influxDB plugin.
	Table name is now 'stats'.
	Network stats added.
	Detailed cpu and memory stats are no longer exported to reduce the load on the DB.
	Docker container alias now exported - It is now possible to aggregate stats across multiple nodes.
- Make the UI independent of the storage backend by caching recent stats in memory.
- Switched to glog.
- Bugfixes and performance improvements.
- Introduced v1.1 remote API with new "subcontainers" resource.

## 0.2.1 (2014-07-25)
- Handle old Docker versions.
- UI fixes and other bugfixes.

## 0.2.0 (2014-07-24)
- Added network stats to the UI.
- Added support for CoreOS and RHEL.
- Bugfixes and reliability fixes.

## 0.1.4 (2014-07-22)
- Add network statistics to REST API.
- Add "raw" driver to handle non-Docker containers.
- Remove lmctfy in favor of the raw driver.
- Bugfixes for Docker containers and logging.

## 0.1.3 (2014-07-14)
- Add support for systemd systems.
- Fixes for UI with InfluxDB storage driver.

## 0.1.2 (2014-07-10)
- Added Storage Driver concept (flag: storage_driver), default is the in-memory driver
- Implemented InfluxDB storage driver
- Support in REST API for specifying number of stats to return
- Allow running without lmctfy (flag: allow_lmctfy)
- Bugfixes

## 0.1.0 (2014-06-14)
- Support for container aliases
- Sampling historical usage and exporting that in the REST API
- Bugfixes for UI

## 0.0.0 (2014-06-10)
- Initial version of cAdvisor
- Web UI with auto-updating stats
- v1.0 REST API with container and machine information
- Support for Docker containers
- Support for lmctfy containers
