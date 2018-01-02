Vagrant.configure('2') do |config|
    config.vm.box = "ubuntu/xenial64" # Ubuntu 16.04

    # fix issues with slow dns http://serverfault.com/a/595010
    config.vm.provider :virtualbox do |vb, override|
        vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
        vb.customize ["modifyvm", :id, "--natdnsproxy1", "on"]
        vb.customize ["modifyvm", :id, "--memory", "1024"]
    end

    config.vm.provider :libvirt do |libvirt, override|
        libvirt.memory = 1024
    end

    config.vm.network "private_network", type: "dhcp"
    config.vm.provision :shell, :privileged => true, :path => "scripts/install-rkt.sh"
    config.vm.provision :shell, :inline => "usermod -a -G rkt-admin ubuntu"
    config.vm.provision :shell, :inline => "usermod -a -G rkt ubuntu"
end
