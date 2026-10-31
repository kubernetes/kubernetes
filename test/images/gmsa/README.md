# GMSA test image

This image provides the Windows command-line tools used by the GMSA end-to-end
tests.

Windows Server 2019 and 2022 variants remain based on Nano Server. The Windows
Server 2025 variant uses Server Core because Nano Server LTSC 2025 does not
include the RemoteFS components required for the Workstation service and GMSA
secure-channel operations.

| Windows version | Base image |
| --- | --- |
| Windows Server 2019 | `mcr.microsoft.com/windows/nanoserver:1809` |
| Windows Server 2022 | `mcr.microsoft.com/windows/nanoserver:ltsc2022` |
| Windows Server 2025 | `mcr.microsoft.com/windows/servercore:ltsc2025` |

The final Windows stage contains no `RUN` instructions, so the image can be
assembled by the existing Linux-hosted Kubernetes image pipeline.

The toolbox preparation follows `../busybox/Dockerfile_windows` and should stay
synchronized with it. The image copies `nltest` and its dependencies into
`C:\gmsa-tools` for every variant, including Server Core 2025 where those files
already exist, so tests use the same command path on all supported versions.

Build and push all variants with:

```shell
make -C test/images all-build-and-push WHAT=gmsa REGISTRY=<registry>
```
