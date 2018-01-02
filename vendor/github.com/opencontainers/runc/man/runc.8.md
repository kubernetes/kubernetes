# NAME
   runc - Open Container Initiative runtime

# SYNOPSIS
   runc [global options] command [command options] [arguments...]
   
# DESCRIPTION
runc is a command line client for running applications packaged according to
the Open Container Initiative (OCI) format and is a compliant implementation of the
Open Container Initiative specification.

runc integrates well with existing process supervisors to provide a production
container runtime environment for applications. It can be used with your
existing process monitoring tools and the container will be spawned as a
direct child of the process supervisor.

Containers are configured using bundles. A bundle for a container is a directory
that includes a specification file named "config.json" and a root filesystem.
The root filesystem contains the contents of the container. 

To start a new instance of a container:

    # runc start [ -b bundle ] <container-id>

Where "<container-id>" is your name for the instance of the container that you
are starting. The name you provide for the container instance must be unique on
your host. Providing the bundle directory using "-b" is optional. The default
value for "bundle" is the current directory.

# COMMANDS
   checkpoint   checkpoint a running container
   delete       delete any resources held by the container often used with detached containers
   events       display container events such as OOM notifications, cpu, memory, IO and network stats
   exec         execute new process inside the container
   init         initialize the namespaces and launch the process (do not call it outside of runc)
   kill         kill sends the specified signal (default: SIGTERM) to the container's init process
   list         lists containers started by runc with the given root
   pause        pause suspends all processes inside the container
   ps           displays the processes running inside a container
   restore      restore a container from a previous checkpoint
   resume       resumes all processes that have been previously paused
   run          create and run a container
   spec         create a new specification file
   start        executes the user defined process in a created container
   state        output the state of a container
   update       update container resource constraints
   help, h      Shows a list of commands or help for one command
   
# GLOBAL OPTIONS
   --debug              enable debug output for logging
   --log value          set the log file path where internal debug information is written (default: "/dev/null")
   --log-format value   set the format used by logs ('text' (default), or 'json') (default: "text")
   --root value         root directory for storage of container state (this should be located in tmpfs) (default: "/run/runc")
   --criu value         path to the criu binary used for checkpoint and restore (default: "criu")
   --systemd-cgroup     enable systemd cgroup support, expects cgroupsPath to be of form "slice:prefix:name" for e.g. "system.slice:runc:434234"
   --help, -h           show help
   --version, -v        print the version
