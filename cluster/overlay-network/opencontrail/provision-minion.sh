source /etc/contrail/opencontrail-rc

LOGFILE=/var/log/contrail/provision_minion.log
OS_TYPE="none"
REDHAT="redhat"
UBUNTU="ubuntu"
VROUTER="vrouter"
VHOST="vhost0"
MINION_OVERLAY_NET_IP="none"

timestamp() {
    date
}

if [ ! -f /var/log/contrail/provision_minion.log ]; then
   mkdir -p /var/log/contrail
   touch /var/log/contrail/provision_minion.log
fi

log_error_msg() {
    msg=$1
    echo "$(timestamp): ERROR: $msg" >> $LOGFILE
}

log_warn_msg() {
    msg=$1
    echo "$(timestamp): WARNING: $msg" >> $LOGFILE
}

log_info_msg() {
    msg=$1
    echo "$(timestamp): INFO: $msg" >> $LOGFILE
}


function detect_os()
{
   OS=`uname`
   if [ "${OS}" = "Linux" ]; then
      if [ -f /etc/redhat-release ]; then
         OS_TYPE="redhat"
      elif [ -f /etc/debian_version ]; then
         OS_TYPE="ubuntu"
      fi
   fi
}

function prep_to_build()
{
  if [ $OS_TYPE == $REDHAT ]; then
    yum update
    yum install -y automake flex bison gcc gcc-c++ boost boost-devel scons kernel-devel-`uname -r` libxml2-devel
  elif [ $OS_TYPE == $UBUNTU ]; then
    apt-get update
    apt-get install -y automake flex bison g++ gcc make libboost-all-dev scons linux-headers-`uname -r` libxml2-dev
  fi
}

function build_vrouter()
{
  rm -rf ~/vrouter-build
  mkdir -p vrouter-build/tools
  cd ~/vrouter-build && `git clone https://github.com/Juniper/contrail-vrouter` && `mv contrail-vrouter vrouter`
  cd ~/vrouter-build/tools && `git clone https://github.com/Juniper/contrail-build` && `mv contrail-build build`
  cd ~/vrouter-build/tools && `git clone https://github.com/Juniper/contrail-sandesh` && `mv contrail-sandesh sandesh` 
  cp ~/vrouter-build/tools/build/SConstruct ~/vrouter-build
  cd ~/vrouter-build && `scons vrouter`
}

function modprobe_vrouter()
{
  vr=$(lsmod | grep vrouter | awk '{print $1}')
  if [[ $vr == $VROUTER ]]; then
     `rmmod vrouter`
      rm -rf /lib/modules/`uname -r`/extra/net/vrouter
  fi
  mkdir -p /lib/modules/`uname -r`/extra/net/vrouter
  mv ~/vrouter-build/vrouter/vrouter.ko /lib/modules/`uname -r`/extra/net/vrouter
  mv ~/vrouter-build/build/debug/vrouter/utils/vif /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/rt /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/dropstats /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/flow /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/mirror /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/mpls /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/nh /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/vxlan /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/vrfstats /usr/bin
  cd /lib/modules/`uname -r` && depmod && cd
  `modprobe vrouter`
  vr=$(lsmod | grep vrouter | awk '{print $1}')
  if [[ $vr == $VROUTER ]]; then
     log_info_msg "Latest version of Opencontrail kernel module - $vr instaled"
  else
     log_info_msg "Installing Opencontrail kernel module - $vr failed"
  fi 
}


function setup_vhost()
{
  phy_itf=$(ip a |grep $MINION_OVERLAY_NET_IP | awk '{print $7}')
  mask=$(ifconfig $phy_itf | grep -i '\(netmask\|mask\)' | awk '{print $4}' | cut -d ":" -f 2)
  mac=$(ifconfig $phy_itf | grep HWaddr | awk '{print $5}')
  echo $mac >> /etc/contrail/default_pmac
  if [ $OS_TYPE == $REDHAT ]; then
    if [ $phy_itf != $VHOST ]; then
      intf="/etc/sysconfig/network-scripts/ifcfg-$phy_itf"
      sed -i '/IPADDR/d' $intf
      sed -i '/NETMASK/d' $intf
      sed -i '/DNS/d' $intf
      grep -q 'NM_CONTROLLED=no' $intf || echo "NM_CONTROLLED=no" >> $intf
    
      # create and configure vhost0
      touch /etc/sysconfig/network-scripts/ifcfg-$VHOST
      ivhost0="/etc/sysconfig/network-scripts/ifcfg-$VHOST"
      grep -q '#Contrail vhost0' $ivhost0 || echo "#Contrail vhost0" >> $ivhost0
      grep -q 'DEVICE=vhost0' $ivhost0 || echo "DEVICE=vhost0" >> $ivhost0
      grep -q 'ONBOOT=yes' $ivhost0 || echo "ONBOOT=yes" >> $ivhost0
      grep -q 'BOOTPROTO=none' $ivhost0 || echo "BOOTPROTO=none" >> $ivhost0
      grep -q 'IPV6INIT=no' $ivhost0 || echo "IPV6INIT=no" >> $ivhost0
      grep -q 'USERCTL=yes' || echo "USERCTL=yes" >> $ivhost0
      grep -q 'IPADDR=$MINION_OVERLAY_NET_IP' $ivhost0 || echo "IPADDR=$MINION_OVERLAY_NET_IP" >> $ivhost0
      grep -q 'NETMASK=$mask' $ivhost0 || echo "NETMASK=$mask" >> $ivhost0
      grep -q 'NM_CONTROLLED=no' $ivhost0 || echo "NM_CONTROLLED=no" >> $ivhost0

      # move any routes on intf to vhost0
      if [ -f /etc/sysconfig/network-scripts/route-$phy_itf ]; then
         mv /etc/sysconfig/network-scripts/route-$phy_itf /etc/sysconfig/network-scripts/route-$VHOST
         sed -i 's/$phy_itf/$VHOST/g' /etc/sysconfig/network-scripts/route-$VHOST
      fi
    fi
  elif [ $OS_TYPE == $UBUNTU ]; then
     if [ $phy_itf != $VHOST ]; then
        itf="/etc/network/interfaces"
        rt=$(cat $itf | grep route |grep $phy_itf)
        rtv=$(sed "s/$phy_itf/$VHOST/g" <<<"$rt")
        grep -q "iface eth1 inet manual" $itf || sed -i 's/^iface eth1 inet.*/iface eth1 inet manual \n    pre-up ifconfig eth1 up\n    post-down ifconfig eth1 down/' $itf
        grep -vwE "(address $MINION_OVERLAY_NET_IP|netmask $mask)" $itf > /tmp/interface
        mv /tmp/interface $itf
    
        # create and configure vhost0
        grep -q 'auto vhost0' $itf || echo "auto vhost0" >> $itf
        grep -q 'iface vhost0 inet static' $itf || echo "iface vhost0 inet static" >> $itf
        grep -q 'pre-up' $itf || echo "    pre-up /opt/contrail/bin/if-vhost0" >> $itf
        grep -q 'netmask $mask' $itf || echo "    netmask $mask" >> $itf
        grep -q 'address $MINION_OVERLAY_NET_IP' $itf || echo "    address $MINION_OVERLAY_NET_IP" >> $itf
        grep -q 'network_name application' $itf || echo "    network_name application" >> $itf
        grep -q "$rtv" $itf || echo "    $rtv" >> $itf
     fi
  fi   
}

function setup_opencontrail_kubelet()
{
  if [ $OS_TYPE == $UBUNTU ]; then
     apt-get install -y python-setuptools
     apt-get install -y software-properties-common
     add-apt-repository -y ppa:opencontrail/ppa
     add-apt-repository -y ppa:opencontrail/r2.20
     apt-get update
     apt-get install -y python-contrail
  elif [ $OS_TYPE == $REDHAT ]; then
     yum install -y python-setuptools
     yum install -y software-properties-common
     yum install -y ppa:opencontrail/ppa
     yum install -y ppa:opencontrail/r2.20
     yum update
     yum install -y python-contrail
  fi
  mkdir ockube
  cd ~/ockube && `git clone https://github.com/Juniper/contrail-controller` && cd
  cd ~/ockube && `git clone https://github.com/Juniper/contrail-kubernetes` && cd
  (cd ~/ockube/contrail-controller/src/vnsw/contrail-vrouter-api; python setup.py install) && cd
  (cd ~/ockube/contrail-kubernetes/scripts/opencontrail-kubelet; python setup.py install) && cd

  mkdir -p /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail
}

function update_restart_kubelet()
{
  #check for manifests in kubelet config
  kubeappendoc=" --network-plugin=opencontrail"
  kubeappendpv=" --allow_privileged=true"
  kubeappendmf=" --config=/etc/kubernetes/manifests"
  source /etc/default/kubelet; kubecf=`echo $KUBELET_OPTS`
  kubepid=$(ps -ef|grep kubelet |grep manifests | awk '{print $2}')
  if [[ $kubepid != `pidof kubelet` ]]; then
      mkdir -p /etc/kubernetes/manifests
      kubecf="$kubecf $kubeappendmf"
  fi
  kubepid=$(ps -ef|grep kubelet |grep allow_privileged | awk '{print $2}')
  if [[ $kubepid != `pidof kubelet` ]]; then 
     kubecf="$kubecf $kubeappendpv"
  fi
  kubepid=$(ps -ef|grep kubelet |grep opencontrail | awk '{print $2}')
  if [[ $kubepid != `pidof kubelet` ]]; then
     kubecf="$kubecf $kubeappendoc"
  fi
  sed -i '/KUBELET_OPTS/d' /etc/default/kubelet
  echo 'KUBELET_OPTS="'$kubecf'"' > /etc/default/kubelet
}

function stop_kube_svcs()
{
   if [[ -n `pidof kube-proxy` ]]; then
      log_info_msg "Kube-proxy is running. Opencontrail does not use kube-proxy as it provides the function. Stoping it."
      `service kube-proxy stop`
   fi

   if [[ -n `pidof flanneld` ]]; then
      log_info_msg "flanneld is running. Opencontrail does not use flannel as it provides the function. Stoping it."
      service flanneld stop
      intf=$(ifconfig flannel | awk 'NR==1{print $1}')
      if [ $intf == "flannel0" ]; then
         `ifconfig $intf down`
         `ifdown $intf`
      fi
   fi
}

function cleanup()
{
  if [ $OS_TYPE == $REDHAT ]; then
    yum remove -y flex bison gcc gcc-c++ boost boost-devel scons kernel-devel-`uname -r` libxml2-devel
  elif [ $OS_TYPE == $UBUNTU ]; then
    apt-get remove -y flex bison g++ gcc make libboost-all-dev scons linux-headers-`uname -r` libxml2-dev
  fi
  rm -rf ~/vrouter-build
  rm -rf ~/ockube
}


function main()
{
   MINION_OVERLAY_NET_IP=$(/sbin/ifconfig eth1 | grep "inet addr" | awk -F: '{print $2}' | awk '{print $1}')  
   detect_os
   prep_to_build
   build_vrouter
   modprobe_vrouter
   setup_vhost
   setup_opencontrail_kubelet
   update_restart_kubelet
   stop_kube_svcs
   cleanup
}

main
