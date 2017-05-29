Name: kubectl
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Cluster Manager

URL: https://kubernetes.io

%description
Command-line utility for interacting with a Kubernetes cluster.

%install

install -m 755 -d %{buildroot}%{_bindir}
install -p -m 755 -t %{buildroot}%{_bindir} kubectl

%files
%{_bindir}/kubectl
