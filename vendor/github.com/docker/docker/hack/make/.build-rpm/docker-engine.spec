Name: docker-engine
Version: %{_version}
Release: %{_release}%{?dist}
Summary: The open-source application container engine
Group: Tools/Docker

License: ASL 2.0
Source: %{name}.tar.gz

URL: https://dockerproject.org
Vendor: Docker
Packager: Docker <support@docker.com>

# is_systemd conditional
%if 0%{?fedora} >= 21 || 0%{?centos} >= 7 || 0%{?rhel} >= 7 || 0%{?suse_version} >= 1210
%global is_systemd 1
%endif

# required packages for build
# most are already in the container (see contrib/builder/rpm/ARCH/generate.sh)
# only require systemd on those systems
%if 0%{?is_systemd}
%if 0%{?suse_version} >= 1210
BuildRequires: systemd-rpm-macros
%{?systemd_requires}
%else
%if 0%{?fedora} >= 25
# Systemd 230 and up no longer have libsystemd-journal (see https://bugzilla.redhat.com/show_bug.cgi?id=1350301)
BuildRequires: pkgconfig(systemd)
Requires: systemd-units
%else
BuildRequires: pkgconfig(systemd)
Requires: systemd-units
BuildRequires: pkgconfig(libsystemd-journal)
%endif
%endif
%else
Requires(post): chkconfig
Requires(preun): chkconfig
# This is for /sbin/service
Requires(preun): initscripts
%endif

# required packages on install
Requires: /bin/sh
Requires: iptables
%if !0%{?suse_version}
Requires: libcgroup
%else
Requires: libcgroup1
%endif
Requires: tar
Requires: xz
%if 0%{?fedora} >= 21 || 0%{?centos} >= 7 || 0%{?rhel} >= 7 || 0%{?oraclelinux} >= 7 || 0%{?amzn} >= 1
# Resolves: rhbz#1165615
Requires: device-mapper-libs >= 1.02.90-1
%endif
%if 0%{?oraclelinux} >= 6
# Require Oracle Unbreakable Enterprise Kernel R4 and newer device-mapper
Requires: kernel-uek >= 4.1
Requires: device-mapper >= 1.02.90-2
%endif

# docker-selinux conditional
%if 0%{?fedora} >= 20 || 0%{?centos} >= 7 || 0%{?rhel} >= 7 || 0%{?oraclelinux} >= 7
%global with_selinux 1
%endif

# DWZ problem with multiple golang binary, see bug
# https://bugzilla.redhat.com/show_bug.cgi?id=995136#c12
%if 0%{?fedora} >= 20 || 0%{?rhel} >= 7 || 0%{?oraclelinux} >= 7
%global _dwz_low_mem_die_limit 0
%endif

# start if with_selinux
%if 0%{?with_selinux}

%if 0%{?centos} >= 7 || 0%{?rhel} >= 7 || 0%{?fedora} >= 25
Requires: container-selinux >= 2.9
%endif# centos 7, rhel 7, fedora 25

%if 0%{?oraclelinux} >= 7
%global selinux_policyver 3.13.1-102.0.3.el7_3.15
%endif # oraclelinux 7
%if 0%{?fedora} == 24
%global selinux_policyver 3.13.1-191
%endif # fedora 24 -- container-selinux on fedora24 does not properly set dockerd, for now just carry docker-engine-selinux for it
%if 0%{?oraclelinux} >= 7 || 0%{?fedora} == 24
Requires: selinux-policy >= %{selinux_policyver}
Requires(pre): %{name}-selinux >= %{version}-%{release}
%endif # selinux-policy for oraclelinux-7, fedora-24

%endif # with_selinux

# conflicting packages
Conflicts: docker
Conflicts: docker-io
Conflicts: docker-engine-cs

%description
Docker is an open source project to build, ship and run any application as a
lightweight container.

Docker containers are both hardware-agnostic and platform-agnostic. This means
they can run anywhere, from your laptop to the largest EC2 compute instance and
everything in between - and they don't require you to use a particular
language, framework or packaging system. That makes them great building blocks
for deploying and scaling web apps, databases, and backend services without
depending on a particular stack or provider.

%prep
%if 0%{?centos} <= 6 || 0%{?oraclelinux} <=6
%setup -n %{name}
%else
%autosetup -n %{name}
%endif

%build
export DOCKER_GITCOMMIT=%{_gitcommit}
./hack/make.sh dynbinary
# ./man/md2man-all.sh runs outside the build container (if at all), since we don't have go-md2man here

%check
./bundles/%{_origversion}/dynbinary-daemon/dockerd -v

%install
# install binary
install -d $RPM_BUILD_ROOT/%{_bindir}
install -p -m 755 bundles/%{_origversion}/dynbinary-daemon/dockerd-%{_origversion} $RPM_BUILD_ROOT/%{_bindir}/dockerd

# install proxy
install -p -m 755 /usr/local/bin/docker-proxy $RPM_BUILD_ROOT/%{_bindir}/docker-proxy

# install containerd
install -p -m 755 /usr/local/bin/docker-containerd $RPM_BUILD_ROOT/%{_bindir}/docker-containerd
install -p -m 755 /usr/local/bin/docker-containerd-shim $RPM_BUILD_ROOT/%{_bindir}/docker-containerd-shim
install -p -m 755 /usr/local/bin/docker-containerd-ctr $RPM_BUILD_ROOT/%{_bindir}/docker-containerd-ctr

# install runc
install -p -m 755 /usr/local/bin/docker-runc $RPM_BUILD_ROOT/%{_bindir}/docker-runc

# install tini
install -p -m 755 /usr/local/bin/docker-init $RPM_BUILD_ROOT/%{_bindir}/docker-init

# install udev rules
install -d $RPM_BUILD_ROOT/%{_sysconfdir}/udev/rules.d
install -p -m 644 contrib/udev/80-docker.rules $RPM_BUILD_ROOT/%{_sysconfdir}/udev/rules.d/80-docker.rules

# add init scripts
install -d $RPM_BUILD_ROOT/etc/sysconfig
install -d $RPM_BUILD_ROOT/%{_initddir}


%if 0%{?is_systemd}
install -d $RPM_BUILD_ROOT/%{_unitdir}
install -p -m 644 contrib/init/systemd/docker.service.rpm $RPM_BUILD_ROOT/%{_unitdir}/docker.service
%else
install -p -m 644 contrib/init/sysvinit-redhat/docker.sysconfig $RPM_BUILD_ROOT/etc/sysconfig/docker
install -p -m 755 contrib/init/sysvinit-redhat/docker $RPM_BUILD_ROOT/%{_initddir}/docker
%endif
# add bash, zsh, and fish completions
install -d $RPM_BUILD_ROOT/usr/share/bash-completion/completions
install -d $RPM_BUILD_ROOT/usr/share/zsh/vendor-completions
install -d $RPM_BUILD_ROOT/usr/share/fish/vendor_completions.d
install -p -m 644 contrib/completion/bash/docker $RPM_BUILD_ROOT/usr/share/bash-completion/completions/docker
install -p -m 644 contrib/completion/zsh/_docker $RPM_BUILD_ROOT/usr/share/zsh/vendor-completions/_docker
install -p -m 644 contrib/completion/fish/docker.fish $RPM_BUILD_ROOT/usr/share/fish/vendor_completions.d/docker.fish

# install manpages
install -d %{buildroot}%{_mandir}/man1
install -p -m 644 man/man1/*.1 $RPM_BUILD_ROOT/%{_mandir}/man1
install -d %{buildroot}%{_mandir}/man5
install -p -m 644 man/man5/*.5 $RPM_BUILD_ROOT/%{_mandir}/man5
install -d %{buildroot}%{_mandir}/man8
install -p -m 644 man/man8/*.8 $RPM_BUILD_ROOT/%{_mandir}/man8

# add vimfiles
install -d $RPM_BUILD_ROOT/usr/share/vim/vimfiles/doc
install -d $RPM_BUILD_ROOT/usr/share/vim/vimfiles/ftdetect
install -d $RPM_BUILD_ROOT/usr/share/vim/vimfiles/syntax
install -p -m 644 contrib/syntax/vim/doc/dockerfile.txt $RPM_BUILD_ROOT/usr/share/vim/vimfiles/doc/dockerfile.txt
install -p -m 644 contrib/syntax/vim/ftdetect/dockerfile.vim $RPM_BUILD_ROOT/usr/share/vim/vimfiles/ftdetect/dockerfile.vim
install -p -m 644 contrib/syntax/vim/syntax/dockerfile.vim $RPM_BUILD_ROOT/usr/share/vim/vimfiles/syntax/dockerfile.vim

# add nano
install -d $RPM_BUILD_ROOT/usr/share/nano
install -p -m 644 contrib/syntax/nano/Dockerfile.nanorc $RPM_BUILD_ROOT/usr/share/nano/Dockerfile.nanorc

# list files owned by the package here
%files
%doc AUTHORS CHANGELOG.md CONTRIBUTING.md LICENSE MAINTAINERS NOTICE README.md
/%{_bindir}/docker
/%{_bindir}/dockerd
/%{_bindir}/docker-containerd
/%{_bindir}/docker-containerd-shim
/%{_bindir}/docker-containerd-ctr
/%{_bindir}/docker-proxy
/%{_bindir}/docker-runc
/%{_bindir}/docker-init
/%{_sysconfdir}/udev/rules.d/80-docker.rules
%if 0%{?is_systemd}
/%{_unitdir}/docker.service
%else
%config(noreplace,missingok) /etc/sysconfig/docker
/%{_initddir}/docker
%endif
/usr/share/bash-completion/completions/docker
/usr/share/zsh/vendor-completions/_docker
/usr/share/fish/vendor_completions.d/docker.fish
%doc
/%{_mandir}/man1/*
/%{_mandir}/man5/*
/%{_mandir}/man8/*
/usr/share/vim/vimfiles/doc/dockerfile.txt
/usr/share/vim/vimfiles/ftdetect/dockerfile.vim
/usr/share/vim/vimfiles/syntax/dockerfile.vim
/usr/share/nano/Dockerfile.nanorc

%post
%if 0%{?is_systemd}
%systemd_post docker
%else
# This adds the proper /etc/rc*.d links for the script
/sbin/chkconfig --add docker
%endif
if ! getent group docker > /dev/null; then
    groupadd --system docker
fi

%preun
%if 0%{?is_systemd}
%systemd_preun docker
%else
if [ $1 -eq 0 ] ; then
    /sbin/service docker stop >/dev/null 2>&1
    /sbin/chkconfig --del docker
fi
%endif

%postun
%if 0%{?is_systemd}
%systemd_postun_with_restart docker
%else
if [ "$1" -ge "1" ] ; then
    /sbin/service docker condrestart >/dev/null 2>&1 || :
fi
%endif

%changelog
