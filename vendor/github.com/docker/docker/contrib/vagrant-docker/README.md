# Vagrant integration

Currently there are at least 4 different projects that we are aware of that deals
with integration with [Vagrant](http://vagrantup.com/) at different levels. One
approach is to use Docker as a [provisioner](http://docs.vagrantup.com/v2/provisioning/index.html)
which means you can create containers and pull base images on VMs using Docker's
CLI and the other is to use Docker as a [provider](http://docs.vagrantup.com/v2/providers/index.html),
meaning you can use Vagrant to control Docker containers.


### Provisioners

* [Vocker](https://github.com/fgrehm/vocker)
* [Ventriloquist](https://github.com/fgrehm/ventriloquist)

### Providers

* [docker-provider](https://github.com/fgrehm/docker-provider)
* [vagrant-shell](https://github.com/destructuring/vagrant-shell)

## Setting up Vagrant-docker with the Engine API

The initial Docker upstart script will not work because it runs on `127.0.0.1`, which is not accessible to the host machine. Instead, we need to change the script to connect to `0.0.0.0`. To do this, modify `/etc/init/docker.conf` to look like this:

```
description     "Docker daemon"

start on filesystem
stop on runlevel [!2345]

respawn

script
    /usr/bin/dockerd -H=tcp://0.0.0.0:2375
end script
```

Once that's done, you need to set up an SSH tunnel between your host machine and the vagrant machine that's running Docker. This can be done by running the following command in a host terminal:

```
ssh -L 2375:localhost:2375 -p 2222 vagrant@localhost
```

(The first 2375 is what your host can connect to, the second 2375 is what port Docker is running on in the vagrant machine, and the 2222 is the port Vagrant is providing for SSH. If VirtualBox is the VM you're using, you can see what value "2222" should be by going to: Network > Adapter 1 > Advanced > Port Forwarding in the VirtualBox GUI.)

Note that because the port has been changed, to run docker commands from within the command line you must run them like this:

```
sudo docker -H 0.0.0.0:2375 < commands for docker >
```
