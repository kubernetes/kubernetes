## nsinit

`nsinit` is a cli application which demonstrates the use of libcontainer.  
It is able to spawn new containers or join existing containers.  

### How to build?

First add the `libcontainer/vendor` into your GOPATH. It's because libcontainer
vendors all its dependencies, so it can be built predictably.

```
export GOPATH=$GOPATH:/your/path/to/libcontainer/vendor
```

Then get into the nsinit folder and get the imported file. Use `make` command
to make the nsinit binary.

```
cd libcontainer/nsinit
go get
make
```

We have finished compiling the nsinit package, but a root filesystem must be
provided for use along with a container configuration file.

Choose a proper place to run your container. For example we use `/busybox`.

```
mkdir /busybox 
curl -sSL 'https://github.com/jpetazzo/docker-busybox/raw/buildroot-2014.11/rootfs.tar' | tar -xC /busybox
```

Then you may need to write a configuration file named `container.json` in the
`/busybox` folder. Environment, networking, and different capabilities for
the container are specified in this file. The configuration is used for each
process executed inside the container.

See the `sample_configs` folder for examples of what the container configuration
should look like.

```
cp libcontainer/sample_configs/minimal.json /busybox/container.json
cd /busybox
```

You can customize `container.json` per your needs. After that, nsinit is
ready to work.

To execute `/bin/bash` in the current directory as a container just run the
following **as root**:

```bash
nsinit exec --tty --config container.json /bin/bash
```

If you wish to spawn another process inside the container while your current
bash session is running, run the same command again to get another bash shell
(or change the command).  If the original process (PID 1) dies, all other
processes spawned inside the container will be killed and the namespace will
be removed. 

You can identify if a process is running in a container by looking to see if
`state.json` is in the root of the directory.
   
You may also specify an alternate root directory from where the `container.json`
file is read and where the `state.json` file will be saved.

### How to use?

Currently nsinit has 9 commands. Type `nsinit -h` to list all of them. 
And for every alternative command, you can also use `--help` to get more 
detailed help documents. For example, `nsinit config --help`.

`nsinit` cli application is implemented using [cli.go](https://github.com/codegangsta/cli). 
Lots of details are handled in cli.go, so the implementation of `nsinit` itself 
is very clean and clear.

*   **config**	
It will generate a standard configuration file for a container.  By default, it 
will generate as the template file in [config.go](https://github.com/docker/libcontainer/blob/master/nsinit/config.go#L192). 
It will modify the template if you have specified some configuration by options.
*   **exec**	
Starts a container and execute a new command inside it. Besides common options, it
has some special options as below.
	- `--tty,-t`: allocate a TTY to the container.
	- `--config`: you can specify a configuration file. By default, it will use 
	template configuration.
	- `--id`: specify the ID for a container. By default, the id is "nsinit".
	- `--user,-u`: set the user, uid, and/or gid for the process. By default the 
	value is "root".
	- `--cwd`: set the current working dir.
	- `--env`: set environment variables for the process.
*   **init**		
It's an internal command that is called inside the container's namespaces to 
initialize the namespace and exec the user's process. It should not be called 
externally.
*   **oom**		
Display oom notifications for a container, you should specify container id.
*   **pause**	
Pause the container's processes, you should specify container id. It will use 
cgroup freeze subsystem to help.
*   **unpause**		
Unpause the container's processes. Same with `pause`.
*   **stats**	
Display statistics for the container, it will mainly show cgroup and network 
statistics.
*   **state**	
Get the container's current state. You can also read the state from `state.json`
 in your container_id folder.
*   **help, h**		
Shows a list of commands or help for one command.
