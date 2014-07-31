# Changelog

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
