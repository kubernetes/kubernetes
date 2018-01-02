# Installing a different version of rkt in CoreOS

If a different version of rkt is required than what ships with CoreOS, a
oneshot systemd unit can be used to download and install an alternate version
on boot.

The following unit will use curl to download rkt, its signature, and the CoreOS
app signing key. The downloaded rkt is then verified with its signature, and
extracted to /opt/rkt.

```
[Unit]
Description=rkt installer
Requires=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/mkdir -p /opt/rkt
ExecStart=/usr/bin/curl --silent -L -o /opt/rkt.tar.gz <rkt-url>
ExecStart=/usr/bin/curl --silent -L -o /opt/rkt.tar.gz.sig <rkt-sig-url>
ExecStart=/usr/bin/curl --silent -L -o /opt/coreos-app-signing-key.gpg https://coreos.com/dist/pubkeys/app-signing-pubkey.gpg
ExecStart=/usr/bin/gpg --keyring /tmp/gpg-keyring --no-default-keyring --import /opt/coreos-app-signing-key.gpg
ExecStart=/usr/bin/gpg --keyring /tmp/gpg-keyring --no-default-keyring --verify /opt/rkt.tar.gz.sig /opt/rkt.tar.gz
ExecStart=/usr/bin/tar --strip-components=1 -xf /opt/rkt.tar.gz -C /opt/rkt
```

The URLs in this unit must be filled in before the unit is installed. Valid
URLs can be found on [rkt's releases page][rkt-releases].

This unit should be installed with either [ignition][ignition] or a [cloud config][cloud-config].
Other units being added can then contain a `After=rkt-install.service` (or
whatever the service was named) to delay their running until rkt has been
installed.

[rkt-releases]: https://github.com/coreos/rkt/releases
[ignition]: https://coreos.com/ignition/docs/latest/
[cloud-config]: https://coreos.com/os/docs/latest/cloud-config.html
