# ARM support

The ARM support should be considered experimental. It will be extended step by step in the coming weeks.

Building a Docker Development Image works in the same fashion as for Intel platform (x86-64).
Currently we have initial support for 32bit ARMv7 devices.

To work with the Docker Development Image you have to clone the Docker/Docker repo on a supported device.
It needs to have a Docker Engine installed to build the Docker Development Image.

From the root of the Docker/Docker repo one can use make to execute the following make targets:
- make validate
- make binary
- make build
- make deb
- make bundles
- make default
- make shell
- make test-unit
- make test-integration-cli
- make

The Makefile does include logic to determine on which OS and architecture the Docker Development Image is built.
Based on OS and architecture it chooses the correct Dockerfile.
For the ARM 32bit architecture it uses `Dockerfile.armhf`.

So for example in order to build a Docker binary one has to:
1. clone the Docker/Docker repository on an ARM device `git clone https://github.com/docker/docker.git`  
2. change into the checked out repository with `cd docker`  
3. execute `make binary` to create a Docker Engine binary for ARM  

## Kernel modules
A few libnetwork integration tests require that the kernel be
configured with "dummy" network interface and has the module
loaded. However, the dummy module may be not loaded automatically.

To load the kernel module permanently, run these commands as `root`.

    modprobe dummy
    echo "dummy" >> /etc/modules

On some systems you also have to sync your kernel modules.

    oc-sync-kernel-modules
    depmod
