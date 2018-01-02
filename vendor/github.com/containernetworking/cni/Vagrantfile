# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|
  config.vm.box = "bento/ubuntu-16.04"

  config.vm.synced_folder ".", "/go/src/github.com/containernetworking/cni"

  config.vm.provision "shell", inline: <<-SHELL
    set -e -x -u

    apt-get update -y || (sleep 40 && apt-get update -y)
    apt-get install -y git

    wget -qO- https://storage.googleapis.com/golang/go1.8.3.linux-amd64.tar.gz | tar -C /usr/local -xz

    echo 'export GOPATH=/go; export PATH=/usr/local/go/bin:$GOPATH/bin:$PATH' >> /root/.bashrc
    eval `tail -n1 /root/.bashrc`

    go get github.com/tools/godep

    cd /go/src/github.com/containernetworking/cni
    godep restore

  SHELL
end
