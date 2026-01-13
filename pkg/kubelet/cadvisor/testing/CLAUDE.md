# Package testing

Package testing provides fake and mock implementations of the cadvisor.Interface for unit testing.

## Key Types

- `Fake`: Simple fake implementation returning static/empty data for testing

## Fake Implementation

- `MachineInfo`: Returns fake machine with 1 core, ~3.75GB memory
- `VersionInfo`: Returns fake kernel (3.16.0), OS (Debian 7), Docker (1.13.1) versions
- `ContainerInfoV2`: Returns empty container info map
- `ImagesFsInfo/RootFsInfo/ContainerFsInfo`: Return empty FsInfo

## Constants

- `FakeKernelVersion`: "3.16.0-0.bpo.4-amd64"
- `FakeContainerOSVersion`: "Debian GNU/Linux 7 (wheezy)"

## Design Notes

- Implements full cadvisor.Interface for compilation compatibility
- Provides predictable values useful for Kubemark and other simulated environments
- `mocks.go` contains mockery-generated mocks for more flexible test scenarios
