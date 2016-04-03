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

# docker builds in a checksum of dockerinit into docker,
# # so stripping the binaries breaks docker
%global __os_install_post %{_rpmconfigdir}/brp-compress
%global debug_package %{nil}

# is_systemd conditional
%if 0%{?fedora} >= 21 || 0%{?centos} >= 7 || 0%{?rhel} >= 7
%global is_systemd 1
%endif

# required packages for build
# most are already in the container (see contrib/builder/rpm/generate.sh)
# only require systemd on those systems
%if 0%{?is_systemd}
BuildRequires: pkgconfig(systemd)
Requires: systemd-units
%else
Requires(post): chkconfig
Requires(preun): chkconfig
# This is for /sbin/service
Requires(preun): initscripts
%endif

# required packages on install
Requires: /bin/sh
Requires: iptables
Requires: libcgroup
Requires: tar
Requires: xz
%if 0%{?fedora} >= 21
# Resolves: rhbz#1165615
Requires: device-mapper-libs >= 1.02.90-1
%endif

# conflicting packages
Conflicts: docker
Conflicts: docker-io

%description
Docker is an open source project to pack, ship and run any application as a
lightweight container

Docker containers are both hardware-agnostic and platform-agnostic. This means
they can run anywhere, from your laptop to the largest EC2 compute instance and
everything in between - and they don't require you to use a particular
language, framework or packaging system. That makes them great building blocks
for deploying and scaling web apps, databases, and backend services without
depending on a particular stack or provider.

%prep
%if 0%{?centos} <= 6
%setup -n %{name}
%else
%autosetup -n %{name}
%endif

%build
./hack/make.sh dynbinary
# ./man/md2man-all.sh runs outside the build container (if at all), since we don't have go-md2man here

%check
./bundles/%{_origversion}/dynbinary/docker -v

%install
# install binary
install -d $RPM_BUILD_ROOT/%{_bindir}
install -p -m 755 bundles/%{_origversion}/dynbinary/docker-%{_origversion} $RPM_BUILD_ROOT/%{_bindir}/docker

# install dockerinit
install -d $RPM_BUILD_ROOT/%{_libexecdir}/docker
install -p -m 755 bundles/%{_origversion}/dynbinary/dockerinit-%{_origversion} $RPM_BUILD_ROOT/%{_libexecdir}/docker/dockerinit

# install udev rules
install -d $RPM_BUILD_ROOT/%{_sysconfdir}/udev/rules.d
install -p -m 755 contrib/udev/80-docker.rules $RPM_BUILD_ROOT/%{_sysconfdir}/udev/rules.d/80-docker.rules

# add init scripts
install -d $RPM_BUILD_ROOT/etc/sysconfig
install -d $RPM_BUILD_ROOT/%{_initddir}


%if 0%{?is_systemd}
install -d $RPM_BUILD_ROOT/%{_unitdir}
install -p -m 644 contrib/init/systemd/docker.service $RPM_BUILD_ROOT/%{_unitdir}/docker.service
install -p -m 644 contrib/init/systemd/docker.socket $RPM_BUILD_ROOT/%{_unitdir}/docker.socket
%endif

install -p -m 644 contrib/init/sysvinit-redhat/docker.sysconfig $RPM_BUILD_ROOT/etc/sysconfig/docker
install -p -m 755 contrib/init/sysvinit-redhat/docker $RPM_BUILD_ROOT/%{_initddir}/docker

# add bash completions
install -d $RPM_BUILD_ROOT/usr/share/bash-completion/completions
install -d $RPM_BUILD_ROOT/usr/share/zsh/vendor-completions
install -d $RPM_BUILD_ROOT/usr/share/fish/completions
install -p -m 644 contrib/completion/bash/docker $RPM_BUILD_ROOT/usr/share/bash-completion/completions/docker
install -p -m 644 contrib/completion/zsh/_docker $RPM_BUILD_ROOT/usr/share/zsh/vendor-completions/_docker
install -p -m 644 contrib/completion/fish/docker.fish $RPM_BUILD_ROOT/usr/share/fish/completions/docker.fish

# install manpages
install -d %{buildroot}%{_mandir}/man1
install -p -m 644 man/man1/*.1 $RPM_BUILD_ROOT/%{_mandir}/man1
install -d %{buildroot}%{_mandir}/man5
install -p -m 644 man/man5/*.5 $RPM_BUILD_ROOT/%{_mandir}/man5

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
/%{_bindir}/docker
/%{_libexecdir}/docker/dockerinit
/%{_sysconfdir}/udev/rules.d/80-docker.rules
%if 0%{?is_systemd}
/%{_unitdir}/docker.service
/%{_unitdir}/docker.socket
%endif
/etc/sysconfig/docker
/%{_initddir}/docker
/usr/share/bash-completion/completions/docker
/usr/share/zsh/vendor-completions/_docker
/usr/share/fish/completions/docker.fish
%doc
/%{_mandir}/man1/*
/%{_mandir}/man5/*
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
