# -*- mode: ruby -*-
# vi: set ft=ruby :

# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

$consul=<<SCRIPT
apt-get update
apt-get -y install wget
wget -qO- https://experimental.docker.com/ | sh
gpasswd -a vagrant docker
service docker restart
docker run -d -p 8500:8500 -p 8300-8302:8300-8302/tcp -p 8300-8302:8300-8302/udp -h consul progrium/consul -server -bootstrap
SCRIPT

$bootstrap=<<SCRIPT
apt-get update
apt-get -y install wget curl
apt-get -y install bridge-utils
wget -qO- https://experimental.docker.com/ | sh
gpasswd -a vagrant docker
echo DOCKER_OPTS=\\"--cluster-store=consul://192.168.33.10:8500 --cluster-advertise=${1}:0\\" >> /etc/default/docker
cp /vagrant/docs/vagrant-systemd/docker.service /etc/systemd/system/
systemctl daemon-reload
systemctl restart docker.service
SCRIPT

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  config.ssh.shell = "bash -c 'BASH_ENV=/etc/profile exec bash'"
  num_nodes = 2
  base_ip = "192.168.33."
  net_ips = num_nodes.times.collect { |n| base_ip + "#{n+11}" }

  config.vm.define "consul-server" do |consul|
    consul.vm.box = "ubuntu/trusty64"
    consul.vm.hostname = "consul-server"
    consul.vm.network :private_network, ip: "192.168.33.10"
    consul.vm.provider "virtualbox" do |vb|
     vb.customize ["modifyvm", :id, "--memory", "512"]
    end
    consul.vm.provision :shell, inline: $consul
  end

  num_nodes.times do |n|
    config.vm.define "net-#{n+1}" do |net|
      net.vm.box = "ubuntu/xenial64"
      net_ip = net_ips[n]
      net_index = n+1
      net.vm.hostname = "net-#{net_index}"
      net.vm.provider "virtualbox" do |vb|
        vb.customize ["modifyvm", :id, "--memory", "1024"]
      end
      net.vm.network :private_network, ip: "#{net_ip}"
      net.vm.provision :shell, inline: $bootstrap, :args => "#{net_ip}"
    end
  end

end
