Name: kubelet
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Cluster Manager

URL: https://kubernetes.io

Requires: iptables >= 1.4.21
Requires: kubernetes-cni >= 0.5.1
Requires: socat
Requires: util-linux
Requires: ethtool
Requires: iproute
Requires: ebtables

%description
The node agent of Kubernetes, the container cluster manager.

%install

install -m 755 -d %{buildroot}%{_bindir}
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/
install -m 755 -d %{buildroot}%{_sysconfdir}/kubernetes/manifests/
install -p -m 755 -t %{buildroot}%{_bindir} kubelet
install -p -m 755 -t %{buildroot}%{_sysconfdir}/systemd/system/ kubelet.service

%files
%{_bindir}/kubelet
%{_sysconfdir}/systemd/system/kubelet.service
%{_sysconfdir}/kubernetes/manifests/
