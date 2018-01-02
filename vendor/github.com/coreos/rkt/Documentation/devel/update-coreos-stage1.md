# Update coreos flavor stage1

This guide will guide you through updating the version of the coreos flavor of stage1.
We usually want to do this to update the systemd version used by the stage1.

The process is quite manual because it's not done often, but improvements are welcomed.

## Extract the root filesystem of the image

Let's assume you want to update from version 991.0.0 to version 1032.0.0.

First, you need to download and verify the image.
Make sure you trust the [CoreOS Image Signing Key][coreos-key].

Since 1032.0.0 is currently only available in the Alpha channel, we'll use the alpha URL:

```
$ mkdir /tmp/coreos-image
$ curl -O https://alpha.release.core-os.net/amd64-usr/1032.0.0/coreos_production_pxe_image.cpio.gz
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  222M  100  222M    0     0  7769k      0  0:00:29  0:00:29 --:--:-- 7790k
$ curl -O http://alpha.release.core-os.net/amd64-usr/1032.0.0/coreos_production_pxe_image.cpio.gz.sig
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   543  100   543    0     0    495      0  0:00:01  0:00:01 --:--:--   495
$ gpg --verify coreos_production_pxe_image.cpio.gz.sig
gpg: assuming signed data in 'coreos_production_pxe_image.cpio.gz'
gpg: Signature made Thu 28 Apr 2016 04:54:00 AM CEST using RSA key ID 1CB5FA26
gpg: checking the trustdb
gpg: marginals needed: 3  completes needed: 1  trust model: PGP
gpg: depth: 0  valid:   5  signed:   5  trust: 0-, 0q, 0n, 0m, 0f, 5u
gpg: depth: 1  valid:   5  signed:   0  trust: 3-, 0q, 0n, 0m, 2f, 0u
gpg: next trustdb check due at 2017-01-19
gpg: Good signature from "CoreOS Buildbot (Offical Builds) <buildbot@coreos.com>" [ultimate]
```

Then you need to extract it:

```
$ gunzip coreos_production_pxe_image.cpio.gz
$ cpio -i < coreos_production_pxe_image.cpio
457785 blocks
$ unsquashfs usr.squashfs
Parallel unsquashfs: Using 4 processors
13445 inodes (14861 blocks) to write


write_xattr: could not write xattr security.capability for file squashfs-root/bin/arping because you're not superuser!

write_xattr: to avoid this error message, either specify -user-xattrs, -no-xattrs, or run as superuser!

Further error messages of this type are suppressed!
[======================================================================================================================================-] 14861/14861 100%

created 12391 files
created 1989 directories
created 722 symlinks
created 0 devices
created 0 fifos
```

You should have now the rootfs of the image in the `squashfs-root` directory.

## Update the manifest files

Back to the rkt repo, in the directory `stage1/usr_from_coreos/manifest.d`, there are some manifest files that define which files are copied from the CoreOS image to the stage1 image.

You need to go through all of them and check that the files listed correspond to files that are in the actual rootfs of the image (which we extracted in the previous step). Do this from your root directory:

```bash
for f in $(cat stage1/usr_from_coreos/manifest-amd64-usr.d/*.manifest); do
	fspath=/tmp/coreos-image/squashfs-root/$f
	if [ ! -e $fspath -a ! -h $fspath ]; then
		echo missing: $f
	fi
done
```

Usually, there are some updated libraries which need an update on their version numbers.
In our case, there are no updates and all the files mentioned in the manifest are present in the updated CoreOS image.

## Update the coreos flavor version used by the build system

In the file `stage1/usr_from_coreos/coreos-common.mk`, we define which CoreOS image version we use for the coreos flavor.
Update `CCN_IMG_RELEASE` to 1032.0.0 and `CCN_SYSTEMD_VERSION` to the systemd version shipped with the image (in our case, v229).

```diff
diff --git a/stage1/usr_from_coreos/coreos-common.mk b/stage1/usr_from_coreos/coreos-common.mk
index b5bfa77..f864f56 100644
--- a/stage1/usr_from_coreos/coreos-common.mk
+++ b/stage1/usr_from_coreos/coreos-common.mk
@@ -9,9 +9,9 @@ _CCN_INCLUDED_ := x
 $(call setup-tmp-dir,CCN_TMPDIR)
 
 # systemd version in coreos image
-CCN_SYSTEMD_VERSION := v225
+CCN_SYSTEMD_VERSION := v229
 # coreos image version
-CCN_IMG_RELEASE := 991.0.0
+CCN_IMG_RELEASE := 1032.0.0
 # coreos image URL
 CCN_IMG_URL := https://alpha.release.core-os.net/amd64-usr/$(CCN_IMG_RELEASE)/coreos_production_pxe_image.cpio.gz
 # path to downloaded pxe image
```

# Check that things work

Once you're finished updating the manifest files and `coreos-common.mk`, we'll do some sanity checks.

First, do a clean build.


### Test all binaries
Make sure that every binary links:

```bash
for f in $(cat stage1/usr_from_coreos/manifest-amd64-usr.d/*.manifest); do
	if [[ $f =~ ^bin/ ]]; then
		sudo chroot build*/aci-for-coreos-flavor/rootfs /usr/lib64/ld-linux-x86-64.so.2 --list $f >/dev/null
		st=$?
		if [ $st -ne 0 ] ; then
			echo $f failed with exit code $st
			break
		fi
	fi
done
```

### run rkt
Run a quick smoketest:

```bash
sudo build*/target/bin/rkt run quay.io/coreos/alpine-sh
```


## Fixing errors
If there are some new libraries missing from the image, you need to add them to the correspoding manifest file.

For example, this update breaks systemd.
When you try to run rkt, you get this error:

```
/usr/lib/systemd/systemd: error while loading shared libraries: libpam.so.0: cannot open shared object file: No such file or directory
```

This means that we need to add libpam to the systemd manifest file:

```diff
diff --git a/stage1/usr_from_coreos/manifest.d/systemd.manifest b/stage1/usr_from_coreos/manifest.d/systemd.manifest
index fca30bb..51d5fbc 100644
--- a/stage1/usr_from_coreos/manifest.d/systemd.manifest
+++ b/stage1/usr_from_coreos/manifest.d/systemd.manifest
@@ -61,6 +61,9 @@ lib64/libmount.so.1
 lib64/libmount.so.1.1.0
 lib64/libnss_files-2.21.so
 lib64/libnss_files.so.2
+lib64/libpam.so
+lib64/libpam.so.0
+lib64/libpam.so.0.84.1
 lib64/libpcre.so
 lib64/libpcre.so.1
 lib64/libpcre.so.1.2.4
```

Then build and test again.


[coreos-key]: https://coreos.com/security/image-signing-key/
