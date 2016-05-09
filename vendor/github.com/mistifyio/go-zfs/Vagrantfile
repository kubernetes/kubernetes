
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
    config.vm.box = "ubuntu/trusty64"
	config.ssh.forward_agent = true

	config.vm.synced_folder ".", "/home/vagrant/go/src/github.com/mistifyio/go-zfs", create: true

    config.vm.provision "shell", inline: <<EOF
cat << END > /etc/profile.d/go.sh
export GOPATH=\\$HOME/go
export PATH=\\$GOPATH/bin:/usr/local/go/bin:\\$PATH
END

chown -R vagrant /home/vagrant/go

apt-get update
apt-get install -y software-properties-common curl
apt-add-repository --yes ppa:zfs-native/stable
apt-get update
apt-get install -y ubuntu-zfs

cd /home/vagrant
curl -z go1.3.3.linux-amd64.tar.gz -L -O https://storage.googleapis.com/golang/go1.3.3.linux-amd64.tar.gz
tar -C /usr/local -zxf /home/vagrant/go1.3.3.linux-amd64.tar.gz

cat << END > /etc/sudoers.d/go
Defaults env_keep += "GOPATH"
END

EOF

end
