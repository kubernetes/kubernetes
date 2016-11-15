# -*- mode: ruby -*-
# vi: set ft=ruby :

# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

# Require a recent version of vagrant otherwise some have reported errors setting host names on boxes
Vagrant.require_version ">= 1.7.4"

if ARGV.first == "up" && ENV['USING_KUBE_SCRIPTS'] != 'true'
  raise Vagrant::Errors::VagrantError.new, <<END
Calling 'vagrant up' directly is not supported.  Instead, please run the following:

  export KUBERNETES_PROVIDER=vagrant
  export VAGRANT_DEFAULT_PROVIDER=providername
  ./cluster/kube-up.sh
END
end

# The number of nodes to provision
$num_node = (ENV['NUM_NODES'] || 1).to_i

# ip configuration
$master_ip = ENV['MASTER_IP']
$node_ip_base = ENV['NODE_IP_BASE'] || ""
$node_ips = $num_node.times.collect { |n| $node_ip_base + "#{n+3}" }

# Determine the OS platform to use
$kube_os = ENV['KUBERNETES_OS'] || "fedora"

# Determine whether vagrant should use nfs to sync folders
$use_nfs = ENV['KUBERNETES_VAGRANT_USE_NFS'] == 'true'

# To override the vagrant provider, use (e.g.):
#   KUBERNETES_PROVIDER=vagrant VAGRANT_DEFAULT_PROVIDER=... .../cluster/kube-up.sh
# To override the box, use (e.g.):
#   KUBERNETES_PROVIDER=vagrant KUBERNETES_BOX_NAME=... .../cluster/kube-up.sh
# You can specify a box version:
#   KUBERNETES_PROVIDER=vagrant KUBERNETES_BOX_NAME=... KUBERNETES_BOX_VERSION=... .../cluster/kube-up.sh
# You can specify a box location:
#   KUBERNETES_PROVIDER=vagrant KUBERNETES_BOX_NAME=... KUBERNETES_BOX_URL=... .../cluster/kube-up.sh
# KUBERNETES_BOX_URL and KUBERNETES_BOX_VERSION will be ignored unless
# KUBERNETES_BOX_NAME is set

# Default OS platform to provider/box information
$kube_provider_boxes = {
  :parallels => {
    'fedora' => {
      # :box_url and :box_version are optional (and mutually exclusive);
      # if :box_url is omitted the box will be retrieved by :box_name (and
      # :box_version if provided) from
      # http://atlas.hashicorp.com/boxes/search (formerly
      # http://vagrantcloud.com/); this allows you override :box_name with
      # your own value so long as you provide :box_url; for example, the
      # "official" name of this box is "rickard-von-essen/
      # opscode_fedora-20", but by providing the URL and our own name, we
      # make it appear as yet another provider under the "kube-fedora22"
      # box
      :box_name => 'kube-fedora23',
      :box_url => 'https://opscode-vm-bento.s3.amazonaws.com/vagrant/parallels/opscode_fedora-23_chef-provisionerless.box'
    }
  },
  :virtualbox => {
    'fedora' => {
      :box_name => 'kube-fedora23',
      :box_url => 'https://opscode-vm-bento.s3.amazonaws.com/vagrant/virtualbox/opscode_fedora-23_chef-provisionerless.box'
    }
  },
  :libvirt => {
    'fedora' => {
      :box_name => 'kube-fedora23',
      :box_url => 'https://dl.fedoraproject.org/pub/fedora/linux/releases/23/Cloud/x86_64/Images/Fedora-Cloud-Base-Vagrant-23-20151030.x86_64.vagrant-libvirt.box'
    }
  },
  :vmware_desktop => {
    'fedora' => {
      :box_name => 'kube-fedora23',
      :box_url => 'https://opscode-vm-bento.s3.amazonaws.com/vagrant/vmware/opscode_fedora-23_chef-provisionerless.box'
    }
  },
  :vsphere => {
    'fedora' => {
      :box_name => 'vsphere-dummy',
      :box_url => 'https://github.com/deromka/vagrant-vsphere/blob/master/vsphere-dummy.box?raw=true'
    }
  }
}

# Give access to all physical cpu cores
# Previously cargo-culted from here:
# http://www.stefanwrobel.com/how-to-make-vagrant-performance-not-suck
# Rewritten to actually determine the number of hardware cores instead of assuming
# that the host has hyperthreading enabled.
host = RbConfig::CONFIG['host_os']
if host =~ /darwin/
  $vm_cpus = `sysctl -n hw.physicalcpu`.to_i
elsif host =~ /linux/
  #This should work on most processors, however it will fail on ones without the core id field.
  #So far i have only seen this on a raspberry pi. which you probably don't want to run vagrant on anyhow...
  #But just in case we'll default to the result of nproc if we get 0 just to be safe.
  $vm_cpus = `cat /proc/cpuinfo | grep 'core id' | sort -u | wc -l`.to_i
  if $vm_cpus < 1
      $vm_cpus = `nproc`.to_i
  end
else # sorry Windows folks, I can't help you
  $vm_cpus = 2
end

# Give VM 1024MB of RAM by default
# In Fedora VM, tmpfs device is mapped to /tmp.  tmpfs is given 50% of RAM allocation.
# When doing Salt provisioning, we copy approximately 200MB of content in /tmp before anything else happens.
# This causes problems if anything else was in /tmp or the other directories that are bound to tmpfs device (i.e /run, etc.)
$vm_master_mem = (ENV['KUBERNETES_MASTER_MEMORY'] || ENV['KUBERNETES_MEMORY'] || 1280).to_i
$vm_node_mem = (ENV['KUBERNETES_NODE_MEMORY'] || ENV['KUBERNETES_MEMORY'] || 2048).to_i

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  if Vagrant.has_plugin?("vagrant-proxyconf")
    $http_proxy = ENV['KUBERNETES_HTTP_PROXY'] || ""
    $https_proxy = ENV['KUBERNETES_HTTPS_PROXY'] || ""
    $no_proxy = ENV['KUBERNETES_NO_PROXY'] || "127.0.0.1"
    config.proxy.http     = $http_proxy
    config.proxy.https    = $https_proxy
    config.proxy.no_proxy = $no_proxy
  end

  # this corrects a bug in 1.8.5 where an invalid SSH key is inserted.
  if Vagrant::VERSION == "1.8.5"
    config.ssh.insert_key = false
  end

  def setvmboxandurl(config, provider)
    if ENV['KUBERNETES_BOX_NAME'] then
      config.vm.box = ENV['KUBERNETES_BOX_NAME']

      if ENV['KUBERNETES_BOX_URL'] then
        config.vm.box_url = ENV['KUBERNETES_BOX_URL']
      end

      if ENV['KUBERNETES_BOX_VERSION'] then
        config.vm.box_version = ENV['KUBERNETES_BOX_VERSION']
      end
    else
      config.vm.box = $kube_provider_boxes[provider][$kube_os][:box_name]

      if $kube_provider_boxes[provider][$kube_os][:box_url] then
        config.vm.box_url = $kube_provider_boxes[provider][$kube_os][:box_url]
      end

      if $kube_provider_boxes[provider][$kube_os][:box_version] then
        config.vm.box_version = $kube_provider_boxes[provider][$kube_os][:box_version]
      end
    end
  end

  def customize_vm(config, vm_mem)

    if $use_nfs then
      config.vm.synced_folder ".", "/vagrant", nfs: true
    end

    # Try VMWare Fusion first (see
    # https://docs.vagrantup.com/v2/providers/basic_usage.html)
    config.vm.provider :vmware_fusion do |v, override|
      setvmboxandurl(override, :vmware_desktop)
      v.vmx['memsize'] = vm_mem
      v.vmx['numvcpus'] = $vm_cpus
    end

    # configure libvirt provider
    config.vm.provider :libvirt do |v, override|
      setvmboxandurl(override, :libvirt)
      v.memory = vm_mem
      v.cpus = $vm_cpus
      v.nested = true
      v.volume_cache = 'none'
    end

    # Then try VMWare Workstation
    config.vm.provider :vmware_workstation do |v, override|
      setvmboxandurl(override, :vmware_desktop)
      v.vmx['memsize'] = vm_mem
      v.vmx['numvcpus'] = $vm_cpus
    end

    # Then try Parallels
    config.vm.provider :parallels do |v, override|
      setvmboxandurl(override, :parallels)
      v.memory = vm_mem # v.customize ['set', :id, '--memsize', vm_mem]
      v.cpus = $vm_cpus # v.customize ['set', :id, '--cpus', $vm_cpus]

      # Don't attempt to update the Parallels tools on the image (this can
      # be done manually if necessary)
      v.update_guest_tools = false # v.customize ['set', :id, '--tools-autoupdate', 'off']

      # Set up Parallels folder sharing to behave like VirtualBox (i.e.,
      # mount the current directory as /vagrant and that's it)
      v.customize ['set', :id, '--shf-guest', 'off']
      v.customize ['set', :id, '--shf-guest-automount', 'off']
      v.customize ['set', :id, '--shf-host', 'on']

      # Synchronize VM clocks to host clock (Avoid certificate invalid issue)
      v.customize ['set', :id, '--time-sync', 'on']

      # Remove all auto-mounted "shared folders"; the result seems to
      # persist between runs (i.e., vagrant halt && vagrant up)
      override.vm.provision :shell, :inline => (%q{
        set -ex
        if [ -d /media/psf ]; then
          for i in /media/psf/*; do
            if [ -d "${i}" ]; then
              umount "${i}" || true
              rmdir -v "${i}"
            fi
          done
          rmdir -v /media/psf
        fi
        exit
      }).strip
    end

    # Then try vsphere
    config.vm.provider :vsphere do |vsphere, override|
      setvmboxandurl(override, :vsphere)

       #config.vm.hostname = ENV['MASTER_NAME']

       config.ssh.username = ENV['MASTER_USER']
       config.ssh.password = ENV['MASTER_PASSWD']

       config.ssh.pty = true
       config.ssh.insert_key = true
       #config.ssh.private_key_path = '~/.ssh/id_rsa_vsphere'
      
      # Don't attempt to update the tools on the image (this can
      # be done manually if necessary)
      # vsphere.update_guest_tools = false # v.customize ['set', :id, '--tools-autoupdate', 'off']

      # The vSphere host we're going to connect to
      vsphere.host = ENV['VAGRANT_VSPHERE_URL']

      # The ESX host for the new VM
      vsphere.compute_resource_name = ENV['VAGRANT_VSPHERE_RESOURCE_POOL']

      # The resource pool for the new VM
      #vsphere.resource_pool_name = 'Comp'

      # path to folder where new VM should be created, if not specified template's parent folder will be used
      vsphere.vm_base_path = ENV['VAGRANT_VSPHERE_BASE_PATH']

      # The template we're going to clone
      vsphere.template_name = ENV['VAGRANT_VSPHERE_TEMPLATE_NAME']

      # The name of the new machine
      #vsphere.name = ENV['MASTER_NAME']

      # vSphere login
      vsphere.user = ENV['VAGRANT_VSPHERE_USERNAME']

      # vSphere password
      vsphere.password = ENV['VAGRANT_VSPHERE_PASSWORD']

      # cpu count
      vsphere.cpu_count = $vm_cpus

      # memory in MB
      vsphere.memory_mb = vm_mem

      # If you don't have SSL configured correctly, set this to 'true'
      vsphere.insecure = ENV['VAGRANT_VSPHERE_INSECURE']
    end


    # Don't attempt to update Virtualbox Guest Additions (requires gcc)
    if Vagrant.has_plugin?("vagrant-vbguest") then
      config.vbguest.auto_update = false
    end
    # Finally, fall back to VirtualBox
    config.vm.provider :virtualbox do |v, override|
      setvmboxandurl(override, :virtualbox)
      v.memory = vm_mem # v.customize ["modifyvm", :id, "--memory", vm_mem]
      v.cpus = $vm_cpus # v.customize ["modifyvm", :id, "--cpus", $vm_cpus]

      # Use faster paravirtualized networking
      v.customize ["modifyvm", :id, "--nictype1", "virtio"]
      v.customize ["modifyvm", :id, "--nictype2", "virtio"]
    end
  end

  # Kubernetes master
  config.vm.define "master" do |c|
    customize_vm c, $vm_master_mem
    if ENV['KUBE_TEMP'] then
      script = "#{ENV['KUBE_TEMP']}/master-start.sh"
      c.vm.provision "shell", run: "always", path: script
    end
    c.vm.network "private_network", ip: "#{$master_ip}"
  end

  # Kubernetes node
  $num_node.times do |n|
    node_vm_name = "node-#{n+1}"

    config.vm.define node_vm_name do |node|
      customize_vm node, $vm_node_mem

      node_ip = $node_ips[n]
      if ENV['KUBE_TEMP'] then
        script = "#{ENV['KUBE_TEMP']}/node-start-#{n}.sh"
        node.vm.provision "shell", run: "always", path: script
      end
      node.vm.network "private_network", ip: "#{node_ip}"
    end
  end
end
