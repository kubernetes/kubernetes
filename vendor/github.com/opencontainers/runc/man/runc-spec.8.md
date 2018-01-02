# NAME
   runc spec - create a new specification file

# SYNOPSIS
   runc spec [command options] [arguments...]

# DESCRIPTION
   The spec command creates the new specification file named "config.json" for
the bundle.

The spec generated is just a starter file. Editing of the spec is required to
achieve desired results. For example, the newly generated spec includes an args
parameter that is initially set to call the "sh" command when the container is
started. Calling "sh" may work for an ubuntu container or busybox, but will not
work for containers that do not include the "sh" program.

# EXAMPLE
  To run docker's hello-world container one needs to set the args parameter
in the spec to call hello. This can be done using the sed command or a text
editor. The following commands create a bundle for hello-world, change the
default args parameter in the spec from "sh" to "/hello", then run the hello
command in a new hello-world container named container1:

    mkdir hello
    cd hello
    docker pull hello-world
    docker export $(docker create hello-world) > hello-world.tar
    mkdir rootfs
    tar -C rootfs -xf hello-world.tar
    runc spec
    sed -i 's;"sh";"/hello";' config.json
    runc start container1

In the start command above, "container1" is the name for the instance of the
container that you are starting. The name you provide for the container instance
must be unique on your host.

An alternative for generating a customized spec config is to use "oci-runtime-tool", the
sub-command "oci-runtime-tool generate" has lots of options that can be used to do any
customizations as you want, see [runtime-tools](https://github.com/opencontainers/runtime-tools)
to get more information.

When starting a container through runc, runc needs root privilege. If not
already running as root, you can use sudo to give runc root privilege. For
example: "sudo runc start container1" will give runc root privilege to start the
container on your host.

Alternatively, you can start a rootless container, which has the ability to run without root privileges. For this to work, the specification file needs to be adjusted accordingly. You can pass the parameter --rootless to this command to generate a proper rootless spec file.

# OPTIONS
   --bundle value, -b value     path to the root of the bundle directory
   --rootless                   generate a configuration for a rootless container
