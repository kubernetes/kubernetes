#!/bin/bash

##############################################################
# opencontrail-kubernetes minion setup and provisioning script. 
# For more info, please refer to the following link
# https://github.com/Juniper/contrail-kubernetes
##############################################################
source /etc/contrail/opencontrail-rc

readonly PROGNAME=$(basename "$0")

ocver=$1

LOGFILE=/var/log/contrail/provision_minion.log
OS_TYPE="none"
REDHAT="redhat"
UBUNTU="ubuntu"
VROUTER="vrouter"
VHOST="vhost0"
if [[ -z $ocver ]]; then
   ocver="R2.20"
fi
rcfile="/etc/contrail/opencontrail-rc"
if [[ -z $OPENCONTRAIL_CONTROLLER_IP ]]; then
   kube_api_port=$(cat /etc/default/kubelet | grep -o 'api-servers=[^;]*' | awk -F// '{print $2}' | awk '{print $1}')
   kube_api_ip=$(echo $kube_api_port| awk -F':' '{print $1}')
   OPENCONTRAIL_CONTROLLER_IP=$kube_api_ip
   echo "OPENCONTRAIL_CONTROLLER_IP=$kube_api_ip" >> $rcfile
fi
if [[ -z $OPENCONTRAIL_VROUTER_INTF ]];then
   OPENCONTRAIL_VROUTER_INTF="eth0"
   echo "OPENCONTRAIL_VROUTER_INTF="eth0"" >> $rcfile
fi
MINION_OVERLAY_NET_IP=$(/sbin/ifconfig $OPENCONTRAIL_VROUTER_INTF | grep "inet addr" | awk -F: '{print $2}' | awk '{print $1}')

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
  if [ "$OS_TYPEi" == $REDHAT ]; then
    yum update
    yum install -y automake flex bison gcc gcc-c++ boost boost-devel scons kernel-devel-`uname -r` libxml2-devel python-lxml
  elif [ "$OS_TYPE" == $UBUNTU ]; then
    apt-get update
    apt-get install -y automake flex bison g++ gcc make libboost-all-dev scons linux-headers-`uname -r` libxml2-dev python-lxml
  fi
}

function build_vrouter()
{
  rm -rf ~/vrouter-build
  mkdir -p vrouter-build/tools
  cd ~/vrouter-build && `git clone -b $ocver https://github.com/Juniper/contrail-vrouter` && `mv contrail-vrouter vrouter`
  cd ~/vrouter-build/tools && `git clone https://github.com/Juniper/contrail-build` && `mv contrail-build build`
  cd ~/vrouter-build/tools && `git clone -b $ocver https://github.com/Juniper/contrail-sandesh` && `mv contrail-sandesh sandesh` 
  cp ~/vrouter-build/tools/build/SConstruct ~/vrouter-build
  cd ~/vrouter-build && `scons vrouter` 2>&1 | tee $LOGFILE
}

function modprobe_vrouter()
{
  vr=$(lsmod | grep vrouter | awk '{print $1}')
  if [ "$vr" == $VROUTER ]; then
     `rmmod vrouter`
      if [ "$OS_TYPE" == $REDHAT ]; then
         rm -rf /lib/modules/`uname -r`/extra/net/vrouter
      elif [ "$OS_TYPE" == $UBUNTU ]; then
         rm -rf /lib/modules/`uname -r`/updates/dkms/vrouter.ko
      fi
  fi
  if [ "$OS_TYPE" == $REDHAT ]; then
     mkdir -p /lib/modules/`uname -r`/extra/net/vrouter
     mv ~/vrouter-build/vrouter/vrouter.ko /lib/modules/`uname -r`/extra/net/vrouter
  elif [ "$OS_TYPE" == $UBUNTU ]; then
      mkdir -p /lib/modules/`uname -r`/updates/dkms
      mv ~/vrouter-build/vrouter/vrouter.ko /lib/modules/`uname -r`/updates/dkms
  fi
  mv ~/vrouter-build/build/debug/vrouter/utils/vif /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/rt /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/dropstats /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/flow /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/mirror /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/mpls /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/nh /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/vxlan /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/vrfstats /usr/bin
  mv ~/vrouter-build/build/debug/vrouter/utils/vrouter /usr/bin
  cd /lib/modules/`uname -r` && depmod && cd
  `modprobe vrouter`
  vr=$(lsmod | grep vrouter | awk '{print $1}')
  if [ "$vr" == $VROUTER ]; then
     log_info_msg "Latest version of Opencontrail kernel module - $vr instaled"
  else
     log_info_msg "Installing Opencontrail kernel module - $vr failed"
  fi 
}


function setup_vhost()
{
  if [ -z $MINION_OVERLAY_NET_IP ]; then 
     log_info_msg "MINION_OVERLAY_NET_IP is empty. Please check the interface for IP. If it is already assigned to vhost0, there will be no change required"
     exit
  fi
  phy_itf=$(ip a |grep $MINION_OVERLAY_NET_IP | awk '{print $7}')
  mask=$(ifconfig $phy_itf | grep -i '\(netmask\|mask\)' | awk '{print $4}' | cut -d ":" -f 2)
  mac=$(ifconfig $phy_itf | grep HWaddr | awk '{print $5}')
  echo $mac >> /etc/contrail/default_pmac
  if [ "$OS_TYPE" == $REDHAT ]; then
    if [ "$phy_itf" != $VHOST ]; then
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
      grep -q 'DEVICETYPE=vhost' $ivhost0 || echo "DEVICETYPE=vhost" >> $ivhost0
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
  elif [ "$OS_TYPE" == $UBUNTU ]; then
     if [ "$phy_itf" != $VHOST ]; then
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
  rm -rf ~/ockube
  mkdir ockube
  cd ~/ockube && `git clone https://github.com/Juniper/contrail-kubernetes` && cd
  if [ "$OS_TYPE" == $UBUNTU ]; then
     apt-get install -y python-setuptools
     apt-get install -y python-pip
     apt-get install -y software-properties-common
     add-apt-repository -y ppa:opencontrail/ppa
     add-apt-repository -y ppa:opencontrail/r2.20
     apt-get update
     apt-get install -y python-contrail python-contrail-vrouter-api 
  elif [ "$OS_TYPE" == $REDHAT ]; then
     yum install -y python-setuptools
     yum install -y python-pip
     #TODO get vn-api and python-contrail-vrouter-api
  fi
  ockub=$(pip freeze | grep kubelet | awk -F= '{print $1}')
  if [ ! -z "$ockub" ]; then
     pip uninstall -y opencontrail-kubelet
  fi
  (cd ~/ockube/contrail-kubernetes/scripts/opencontrail-kubelet; python setup.py install) && cd
  
  mkdir -p /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail
  if [ ! -f /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail/config ]; then
     touch /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail/config
  fi
  config="/usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail/config"
  ocp="/usr/local/bin/opencontrail-kubelet-plugin"
  if [ ! -f "$ocp" ]; then
     log_info_msg "Opencontrail-kubelet-plugin not found. Please check the package opencontrail-kubelet"
     exit 1
  fi
  grep -q 'DEFAULTS' $config || echo "[DEFAULTS]" >> $config
  sed -i '/api_server/d' $config
  echo "api_server=$OPENCONTRAIL_CONTROLLER_IP" >> $config
  (cd /usr/libexec/kubernetes/kubelet-plugins/net/exec/opencontrail; `ln -s $ocp opencontrail`) && cd
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
      `update-rc.d -f kube-proxy disable`
      `update-rc.d -f kube-proxy remove`
   fi

   if [[ -n `pidof flanneld` ]]; then
      log_info_msg "flanneld is running. Opencontrail does not use flannel as it provides the function. Stoping it."
      service flanneld stop
      intf=$(ifconfig flannel | awk 'NR==1{print $1}')
      if [ $intf == "flannel0" ]; then
         `ifconfig $intf down`
         `ifdown $intf`
         `update-rc.d -f flanneld disable`
         `update-rc.d -f flanneld remove`
      fi
   fi
}

function update_vhost_pre_up()
{
  preup="/etc/network/if-pre-up.d"
  if [ "$OS_TYPE" == $REDHAT ]; then
     preup="/etc/sysconfig/network-scripts"
  fi
  cp ~/ockube/contrail-kubernetes/scripts/opencontrail-install/ifup-vhost $preup
  `chmod +x $preup/ifup-vhost`
}

function cleanup()
{
  if [ "$OS_TYPE" == $REDHAT ]; then
    yum remove -y flex bison gcc gcc-c++ boost boost-devel scons libxml2-devel
  elif [ "$OS_TYPE" == $UBUNTU ]; then
    apt-get remove -y flex bison g++ gcc make libboost-all-dev scons libxml2-dev
  fi
  rm -rf ~/vrouter-build
  rm -rf ~/ockube
}


function main()
{
   detect_os
   prep_to_build
   build_vrouter
   modprobe_vrouter
   setup_vhost
   setup_opencontrail_kubelet
   update_restart_kubelet
   stop_kube_svcs
   update_vhost_pre_up
   cleanup
}

main
