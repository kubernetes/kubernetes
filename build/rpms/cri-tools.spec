Name: cri-tools
Version: OVERRIDE_THIS
Release: 00
License: ASL 2.0
Summary: Container Runtime Interface tools

URL: https://kubernetes.io

%description
Binaries to interface with the container runtime.

%prep
# This has to be hard coded because bazel does a path substitution before rpm's %{version} is substituted.
tar -xzf {crictl-v1.12.0-linux-amd64.tar.gz}

%install
install -m 755 -d %{buildroot}%{_bindir}
install -p -m 755 -t %{buildroot}%{_bindir} crictl

%files
%{_bindir}/crictl
