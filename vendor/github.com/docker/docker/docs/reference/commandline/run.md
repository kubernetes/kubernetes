<!--[metadata]>
+++
title = "run"
description = "The run command description and usage"
keywords = ["run, command, container"]
[menu.main]
parent = "smn_cli"
weight=1
+++
<![end-metadata]-->

# run

    Usage: docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

    Run a command in a new container

      -a, --attach=[]            Attach to STDIN, STDOUT or STDERR
      --add-host=[]              Add a custom host-to-IP mapping (host:ip)
      --blkio-weight=0           Block IO weight (relative weight)
      -c, --cpu-shares=0         CPU shares (relative weight)
      --cap-add=[]               Add Linux capabilities
      --cap-drop=[]              Drop Linux capabilities
      --cgroup-parent=""         Optional parent cgroup for the container
      --cidfile=""               Write the container ID to the file
      --cpu-period=0             Limit CPU CFS (Completely Fair Scheduler) period
      --cpu-quota=0              Limit CPU CFS (Completely Fair Scheduler) quota
      --cpuset-cpus=""           CPUs in which to allow execution (0-3, 0,1)
      --cpuset-mems=""           Memory nodes (MEMs) in which to allow execution (0-3, 0,1)
      -d, --detach=false         Run container in background and print container ID
      --device=[]                Add a host device to the container
      --dns=[]                   Set custom DNS servers
      --dns-search=[]            Set custom DNS search domains
      -e, --env=[]               Set environment variables
      --entrypoint=""            Overwrite the default ENTRYPOINT of the image
      --env-file=[]              Read in a file of environment variables
      --expose=[]                Expose a port or a range of ports
      --group-add=[]             Add additional groups to run as
      -h, --hostname=""          Container host name
      --help=false               Print usage
      -i, --interactive=false    Keep STDIN open even if not attached
      --ipc=""                   IPC namespace to use
      -l, --label=[]             Set metadata on the container (e.g., --label=com.example.key=value)
      --label-file=[]            Read in a file of labels (EOL delimited)
      --link=[]                  Add link to another container
      --log-driver=""            Logging driver for container
      --log-opt=[]               Log driver specific options
      --lxc-conf=[]              Add custom lxc options
      -m, --memory=""            Memory limit
      --mac-address=""           Container MAC address (e.g. 92:d0:c6:0a:29:33)
      --memory-swap=""           Total memory (memory + swap), '-1' to disable swap
      --memory-swappiness=""     Tune a container's memory swappiness behavior. Accepts an integer between 0 and 100.
      --name=""                  Assign a name to the container
      --net="bridge"             Set the Network mode for the container
      --oom-kill-disable=false   Whether to disable OOM Killer for the container or not
      -P, --publish-all=false    Publish all exposed ports to random ports
      -p, --publish=[]           Publish a container's port(s) to the host
      --pid=""                   PID namespace to use
      --privileged=false         Give extended privileges to this container
      --read-only=false          Mount the container's root filesystem as read only
      --restart="no"             Restart policy (no, on-failure[:max-retry], always)
      --rm=false                 Automatically remove the container when it exits
      --security-opt=[]          Security Options
      --sig-proxy=true           Proxy received signals to the process
      -t, --tty=false            Allocate a pseudo-TTY
      -u, --user=""              Username or UID (format: <name|uid>[:<group|gid>])
      --ulimit=[]                Ulimit options
      --uts=""                   UTS namespace to use
      -v, --volume=[]            Bind mount a volume
      --volumes-from=[]          Mount volumes from the specified container(s)
      -w, --workdir=""           Working directory inside the container

The `docker run` command first `creates` a writeable container layer over the
specified image, and then `starts` it using the specified command. That is,
`docker run` is equivalent to the API `/containers/create` then
`/containers/(id)/start`. A stopped container can be restarted with all its
previous changes intact using `docker start`. See `docker ps -a` to view a list
of all containers.

There is detailed information about `docker run` in the [Docker run reference](
/reference/run/).

The `docker run` command can be used in combination with `docker commit` to
[*change the command that a container runs*](#commit-an-existing-container).

See the [Docker User Guide](/userguide/dockerlinks/) for more detailed
information about the `--expose`, `-p`, `-P` and `--link` parameters,
and linking containers.

## Examples

    $ docker run --name test -it debian
    root@d6c0fe130dba:/# exit 13
    $ echo $?
    13
    $ docker ps -a | grep test
    d6c0fe130dba        debian:7            "/bin/bash"         26 seconds ago      Exited (13) 17 seconds ago                         test

This example runs a container named `test` using the `debian:latest` 
image. The `-it` instructs Docker to allocate a pseudo-TTY connected to
the container's stdin; creating an interactive `bash` shell in the container.
In the example, the `bash` shell is quit by entering
`exit 13`. This exit code is passed on to the caller of
`docker run`, and is recorded in the `test` container's metadata.

    $ docker run --cidfile /tmp/docker_test.cid ubuntu echo "test"

This will create a container and print `test` to the console. The `cidfile`
flag makes Docker attempt to create a new file and write the container ID to it.
If the file exists already, Docker will return an error. Docker will close this
file when `docker run` exits.

    $ docker run -t -i --rm ubuntu bash
    root@bc338942ef20:/# mount -t tmpfs none /mnt
    mount: permission denied

This will *not* work, because by default, most potentially dangerous kernel
capabilities are dropped; including `cap_sys_admin` (which is required to mount
filesystems). However, the `--privileged` flag will allow it to run:

    $ docker run --privileged ubuntu bash
    root@50e3f57e16e6:/# mount -t tmpfs none /mnt
    root@50e3f57e16e6:/# df -h
    Filesystem      Size  Used Avail Use% Mounted on
    none            1.9G     0  1.9G   0% /mnt

The `--privileged` flag gives *all* capabilities to the container, and it also
lifts all the limitations enforced by the `device` cgroup controller. In other
words, the container can then do almost everything that the host can do. This
flag exists to allow special use-cases, like running Docker within Docker.

    $ docker  run -w /path/to/dir/ -i -t  ubuntu pwd

The `-w` lets the command being executed inside directory given, here
`/path/to/dir/`. If the path does not exists it is created inside the container.

    $ docker  run  -v `pwd`:`pwd` -w `pwd` -i -t  ubuntu pwd

The `-v` flag mounts the current working directory into the container. The `-w`
lets the command being executed inside the current working directory, by
changing into the directory to the value returned by `pwd`. So this
combination executes the command using the container, but inside the
current working directory.

    $ docker run -v /doesnt/exist:/foo -w /foo -i -t ubuntu bash

When the host directory of a bind-mounted volume doesn't exist, Docker
will automatically create this directory on the host for you. In the
example above, Docker will create the `/doesnt/exist`
folder before starting your container.

    $ docker run --read-only -v /icanwrite busybox touch /icanwrite here

Volumes can be used in combination with `--read-only` to control where
a container writes files. The `--read-only` flag mounts the container's root
filesystem as read only prohibiting writes to locations other than the
specified volumes for the container.

    $ docker run -t -i -v /var/run/docker.sock:/var/run/docker.sock -v ./static-docker:/usr/bin/docker busybox sh

By bind-mounting the docker unix socket and statically linked docker
binary (such as that provided by [https://get.docker.com](
https://get.docker.com)), you give the container the full access to create and
manipulate the host's Docker daemon.

    $ docker run -p 127.0.0.1:80:8080 ubuntu bash

This binds port `8080` of the container to port `80` on `127.0.0.1` of
the host machine. The [Docker User Guide](/userguide/dockerlinks/)
explains in detail how to manipulate ports in Docker.

    $ docker run --expose 80 ubuntu bash

This exposes port `80` of the container for use within a link without
publishing the port to the host system's interfaces. The [Docker User
Guide](/userguide/dockerlinks) explains in detail how to manipulate
ports in Docker.

    $ docker run -e MYVAR1 --env MYVAR2=foo --env-file ./env.list ubuntu bash

This sets environmental variables in the container. For illustration all three
flags are shown here. Where `-e`, `--env` take an environment variable and
value, or if no `=` is provided, then that variable's current value is passed
through (i.e. `$MYVAR1` from the host is set to `$MYVAR1` in the container).
When no `=` is provided and that variable is not defined in the client's
environment then that variable will be removed from the container's list of
environment variables.
All three flags, `-e`, `--env` and `--env-file` can be repeated.

Regardless of the order of these three flags, the `--env-file` are processed
first, and then `-e`, `--env` flags. This way, the `-e` or `--env` will
override variables as needed.

    $ cat ./env.list
    TEST_FOO=BAR
    $ docker run --env TEST_FOO="This is a test" --env-file ./env.list busybox env | grep TEST_FOO
    TEST_FOO=This is a test

The `--env-file` flag takes a filename as an argument and expects each line
to be in the `VAR=VAL` format, mimicking the argument passed to `--env`. Comment
lines need only be prefixed with `#`

An example of a file passed with `--env-file`

    $ cat ./env.list
    TEST_FOO=BAR

    # this is a comment
    TEST_APP_DEST_HOST=10.10.0.127
    TEST_APP_DEST_PORT=8888
    _TEST_BAR=FOO
    TEST_APP_42=magic
    helloWorld=true
    # 123qwe=bar <- is not valid

    # pass through this variable from the caller
    TEST_PASSTHROUGH
    $ TEST_PASSTHROUGH=howdy docker run --env-file ./env.list busybox env
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    HOSTNAME=5198e0745561
    TEST_FOO=BAR
    TEST_APP_DEST_HOST=10.10.0.127
    TEST_APP_DEST_PORT=8888
    _TEST_BAR=FOO
    TEST_APP_42=magic
    helloWorld=true
    TEST_PASSTHROUGH=howdy
    HOME=/root

    $ docker run --env-file ./env.list busybox env
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    HOSTNAME=5198e0745561
    TEST_FOO=BAR
    TEST_APP_DEST_HOST=10.10.0.127
    TEST_APP_DEST_PORT=8888
    _TEST_BAR=FOO
    TEST_APP_42=magic
    helloWorld=true
    TEST_PASSTHROUGH=
    HOME=/root

> **Note**: Environment variables names must consist solely of letters, numbers,
> and underscores - and cannot start with a number.

A label is a a `key=value` pair that applies metadata to a container. To label a container with two labels:

    $ docker run -l my-label --label com.example.foo=bar ubuntu bash

The `my-label` key doesn't specify a value so the label defaults to an empty
string(`""`). To add multiple labels, repeat the label flag (`-l` or `--label`).

The `key=value` must be unique to avoid overwriting the label value. If you
specify labels with identical keys but different values, each subsequent value
overwrites the previous. Docker uses the last `key=value` you supply.

Use the `--label-file` flag to load multiple labels from a file. Delimit each
label in the file with an EOL mark. The example below loads labels from a
labels file in the current directory:

    $ docker run --label-file ./labels ubuntu bash

The label-file format is similar to the format for loading environment
variables. (Unlike environment variables, labels are not visible to processes
running inside a container.) The following example illustrates a label-file
format:

    com.example.label1="a label"

    # this is a comment
    com.example.label2=another\ label
    com.example.label3

You can load multiple label-files by supplying multiple  `--label-file` flags.

For additional information on working with labels, see [*Labels - custom
metadata in Docker*](/userguide/labels-custom-metadata/) in the Docker User
Guide.

    $ docker run --link /redis:redis --name console ubuntu bash

The `--link` flag will link the container named `/redis` into the newly
created container with the alias `redis`. The new container can access the
network and environment of the `redis` container via environment variables.
The `--link` flag will also just accept the form `<name or id>` in which case
the alias will match the name. For instance, you could have written the previous
example as:

    $ docker run --link redis --name console ubuntu bash

The `--name` flag will assign the name `console` to the newly created
container.

    $ docker run --volumes-from 777f7dc92da7 --volumes-from ba8c0c54f0f2:ro -i -t ubuntu pwd

The `--volumes-from` flag mounts all the defined volumes from the referenced
containers. Containers can be specified by repetitions of the `--volumes-from`
argument. The container ID may be optionally suffixed with `:ro` or `:rw` to
mount the volumes in read-only or read-write mode, respectively. By default,
the volumes are mounted in the same mode (read write or read only) as
the reference container.

Labeling systems like SELinux require that proper labels are placed on volume
content mounted into a container. Without a label, the security system might
prevent the processes running inside the container from using the content. By
default, Docker does not change the labels set by the OS.

To change the label in the container context, you can add either of two suffixes
`:z` or `:Z` to the volume mount. These suffixes tell Docker to relabel file
objects on the shared volumes. The `z` option tells Docker that two containers
share the volume content. As a result, Docker labels the content with a shared
content label. Shared volume labels allow all containers to read/write content.
The `Z` option tells Docker to label the content with a private unshared label.
Only the current container can use a private volume.

The `-a` flag tells `docker run` to bind to the container's `STDIN`, `STDOUT`
or `STDERR`. This makes it possible to manipulate the output and input as
needed.

    $ echo "test" | docker run -i -a stdin ubuntu cat -

This pipes data into a container and prints the container's ID by attaching
only to the container's `STDIN`.

    $ docker run -a stderr ubuntu echo test

This isn't going to print anything unless there's an error because we've
only attached to the `STDERR` of the container. The container's logs
still store what's been written to `STDERR` and `STDOUT`.

    $ cat somefile | docker run -i -a stdin mybuilder dobuild

This is how piping a file into a container could be done for a build.
The container's ID will be printed after the build is done and the build
logs could be retrieved using `docker logs`. This is
useful if you need to pipe a file or something else into a container and
retrieve the container's ID once the container has finished running.

    $ docker run --device=/dev/sdc:/dev/xvdc --device=/dev/sdd --device=/dev/zero:/dev/nulo -i -t ubuntu ls -l /dev/{xvdc,sdd,nulo}
    brw-rw---- 1 root disk 8, 2 Feb  9 16:05 /dev/xvdc
    brw-rw---- 1 root disk 8, 3 Feb  9 16:05 /dev/sdd
    crw-rw-rw- 1 root root 1, 5 Feb  9 16:05 /dev/nulo

It is often necessary to directly expose devices to a container. The `--device`
option enables that. For example, a specific block storage device or loop
device or audio device can be added to an otherwise unprivileged container
(without the `--privileged` flag) and have the application directly access it.

By default, the container will be able to `read`, `write` and `mknod` these devices.
This can be overridden using a third `:rwm` set of options to each `--device`
flag:


    $ docker run --device=/dev/sda:/dev/xvdc --rm -it ubuntu fdisk  /dev/xvdc

    Command (m for help): q
    $ docker run --device=/dev/sda:/dev/xvdc:ro --rm -it ubuntu fdisk  /dev/xvdc
    You will not be able to write the partition table.

    Command (m for help): q

    $ docker run --device=/dev/sda:/dev/xvdc --rm -it ubuntu fdisk  /dev/xvdc

    Command (m for help): q

    $ docker run --device=/dev/sda:/dev/xvdc:m --rm -it ubuntu fdisk  /dev/xvdc
    fdisk: unable to open /dev/xvdc: Operation not permitted

> **Note:**
> `--device` cannot be safely used with ephemeral devices. Block devices
> that may be removed should not be added to untrusted containers with
> `--device`.

**A complete example:**

    $ docker run -d --name static static-web-files sh
    $ docker run -d --expose=8098 --name riak riakserver
    $ docker run -d -m 100m -e DEVELOPMENT=1 -e BRANCH=example-code -v $(pwd):/app/bin:ro --name app appserver
    $ docker run -d -p 1443:443 --dns=10.0.0.1 --dns-search=dev.org -v /var/log/httpd --volumes-from static --link riak --link app -h www.sven.dev.org --name web webserver
    $ docker run -t -i --rm --volumes-from web -w /var/log/httpd busybox tail -f access.log

This example shows five containers that might be set up to test a web
application change:

1. Start a pre-prepared volume image `static-web-files` (in the background)
   that has CSS, image and static HTML in it, (with a `VOLUME` instruction in
   the Dockerfile to allow the web server to use those files);
2. Start a pre-prepared `riakserver` image, give the container name `riak` and
   expose port `8098` to any containers that link to it;
3. Start the `appserver` image, restricting its memory usage to 100MB, setting
   two environment variables `DEVELOPMENT` and `BRANCH` and bind-mounting the
   current directory (`$(pwd)`) in the container in read-only mode as `/app/bin`;
4. Start the `webserver`, mapping port `443` in the container to port `1443` on
   the Docker server, setting the DNS server to `10.0.0.1` and DNS search
   domain to `dev.org`, creating a volume to put the log files into (so we can
   access it from another container), then importing the files from the volume
   exposed by the `static` container, and linking to all exposed ports from
   `riak` and `app`. Lastly, we set the hostname to `web.sven.dev.org` so its
   consistent with the pre-generated SSL certificate;
5. Finally, we create a container that runs `tail -f access.log` using the logs
   volume from the `web` container, setting the workdir to `/var/log/httpd`. The
   `--rm` option means that when the container exits, the container's layer is
   removed.

## Restart policies

Use Docker's `--restart` to specify a container's *restart policy*. A restart
policy controls whether the Docker daemon restarts a container after exit.
Docker supports the following restart policies:

<table>
  <thead>
    <tr>
      <th>Policy</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>no</strong></td>
      <td>
        Do not automatically restart the container when it exits. This is the
        default.
      </td>
    </tr>
    <tr>
      <td>
        <span style="white-space: nowrap">
          <strong>on-failure</strong>[:max-retries]
        </span>
      </td>
      <td>
        Restart only if the container exits with a non-zero exit status.
        Optionally, limit the number of restart retries the Docker
        daemon attempts.
      </td>
    </tr>
    <tr>
      <td><strong>always</strong></td>
      <td>
        Always restart the container regardless of the exit status.
        When you specify always, the Docker daemon will try to restart
        the container indefinitely.
      </td>
    </tr>
  </tbody>
</table>

    $ docker run --restart=always redis

This will run the `redis` container with a restart policy of **always**
so that if the container exits, Docker will restart it.

More detailed information on restart policies can be found in the
[Restart Policies (--restart)](/reference/run/#restart-policies-restart)
section of the Docker run reference page.

## Adding entries to a container hosts file

You can add other hosts into a container's `/etc/hosts` file by using one or
more `--add-host` flags. This example adds a static address for a host named
`docker`:

    $ docker run --add-host=docker:10.180.0.1 --rm -it debian
    $$ ping docker
    PING docker (10.180.0.1): 48 data bytes
    56 bytes from 10.180.0.1: icmp_seq=0 ttl=254 time=7.600 ms
    56 bytes from 10.180.0.1: icmp_seq=1 ttl=254 time=30.705 ms
    ^C--- docker ping statistics ---
    2 packets transmitted, 2 packets received, 0% packet loss
    round-trip min/avg/max/stddev = 7.600/19.152/30.705/11.553 ms

Sometimes you need to connect to the Docker host from within your
container. To enable this, pass the Docker host's IP address to
the container using the `--add-host` flag. To find the host's address,
use the `ip addr show` command.

The flags you pass to `ip addr show` depend on whether you are
using IPv4 or IPv6 networking in your containers. Use the following
flags for IPv4 address retrieval for a network device named `eth0`:

    $ HOSTIP=`ip -4 addr show scope global dev eth0 | grep inet | awk '{print \$2}' | cut -d / -f 1`
    $ docker run  --add-host=docker:${HOSTIP} --rm -it debian

For IPv6 use the `-6` flag instead of the `-4` flag. For other network
devices, replace `eth0` with the correct device name (for example `docker0`
for the bridge device).

### Setting ulimits in a container

Since setting `ulimit` settings in a container requires extra privileges not
available in the default container, you can set these using the `--ulimit` flag.
`--ulimit` is specified with a soft and hard limit as such:
`<type>=<soft limit>[:<hard limit>]`, for example:

    $ docker run --ulimit nofile=1024:1024 --rm debian ulimit -n
    1024

> **Note:**
> If you do not provide a `hard limit`, the `soft limit` will be used
> for both values. If no `ulimits` are set, they will be inherited from
> the default `ulimits` set on the daemon.  `as` option is disabled now.
> In other words, the following script is not supported:
> `$ docker run -it --ulimit as=1024 fedora /bin/bash`

The values are sent to the appropriate `syscall` as they are set.
Docker doesn't perform any byte conversion. Take this into account when setting the values.

#### For `nproc` usage:

Be careful setting `nproc` with the `ulimit` flag as `nproc` is designed by Linux to set the
maximum number of processes available to a user, not to a container.  For example, start four
containers with `daemon` user:


    docker run -d -u daemon --ulimit nproc=3 busybox top
    docker run -d -u daemon --ulimit nproc=3 busybox top
    docker run -d -u daemon --ulimit nproc=3 busybox top
    docker run -d -u daemon --ulimit nproc=3 busybox top

The 4th container fails and reports "[8] System error: resource temporarily unavailable" error. 
This fails because the caller set `nproc=3` resulting in the first three containers using up 
the three processes quota set for the `daemon` user.
