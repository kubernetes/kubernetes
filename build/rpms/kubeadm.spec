Name: kubeadm
Version: OVERRIDE_THIS
Release: 00
License: ASL
Summary: Container Cluster Manager
Requires: kubelet >= 1.6.0
Requires: kubectl >= 1.6.0
Requires: kubernetes-cni

URL: https://kubernetes.io

%description
Command-line utility for administering a Kubernetes cluster.

%prep
find .

%install

install -m 755 -d %{buildroot}%{_bindir}
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/
install -m 755 -d %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/
install -p -m 755 -t %{buildroot}%{_bindir} kubeadm
install -p -m 755 -t %{buildroot}%{_sysconfdir}/systemd/system/kubelet.service.d/ 10-kubeadm.conf

%files
%{_bindir}/kubeadm
%{_sysconfdir}/systemd/system/kubelet.service.d/10-kubeadm.conf
