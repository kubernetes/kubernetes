# go-vhd

Go package and CLI to work with VHD images (https://technet.microsoft.com/en-us/virtualization/bb676673.aspx)

**Highly Experimental**

* Support for printing VHD headers
* Create Dynamic (sparse) VHD images
* Convert a RAW image to a fixed VHD

```
govhd create foo.vhd 80GiB
```

```
govhd info foo.vhd
Cookie:                  0x636f6e6563746978 (conectix)
Features:                0x00000002
File format version:     0x00010000
Data offset:             0x0000000000000200 (512 bytes)
Timestamp:               2015-02-10 14:17:25 +0100 CET
Creator application:     go-v
Creator version:         0x00000000
Creator OS:              Wi2k
Original size:           0x0000001400000000 ( 85899345920 bytes )
Current size:            0x0000001400000000 ( 85899345920 bytes )
Disk geometry:           0xa0a010ff (c: 41120, h: 16, s: 255) (85898035200 bytes)
Disk type:               0x00000003 (Dynamic)
Checksum:                0xffffee82
UUID:                    16a1614a-f6f9-1708-a42a-3bf58ada0942
Saved state:             0

Reading dynamic/differential VHD header...
Cookie:                  0x6378737061727365 (cxsparse)
Data offset:             0xffffffffffffffff
Table offset:            0x0000000000000600
Header version:          0x00010000
Max table entries:       0x0000a000
Block size:              0x00200000
Checksum:                0xfffff3d7
Parent UUID:             00000000-0000-0000-0000-000000000000
Parent timestamp:        2000-01-01 01:00:00 +0100 CET
Reserved:                0x00000000
Parent Name:
Reserved2:               0
```

```
$ dd if=/dev/null of=img.raw bs=1 seek=8M
0+0 records in
0+0 records out
0 bytes (0 B) copied, 0,000172473 s, 0,0 kB/s

$ go-vhd raw2fixed img.raw

$ go-vhd info img.vhd

VHD footer
==========
Cookie:                  0x636f6e6563746978 (conectix)
Features:                0x00000002
File format version:     0x00010000
Data offset:             0xffffffffffffffff (18446744073709551615 bytes)
Timestamp:               2015-03-17 14:04:14 +0100 CET
Creator application:     go-v
Creator version:         0x00000000
Creator OS:              Wi2k
Original size:           0x0000000000800000 ( 8388608 bytes )
Current size:            0x0000000000800000 ( 8388608 bytes )
Disk geometry:           0x00f00411 (c: 240, h: 4, s: 17) (8355840 bytes)
Disk type:               0x00000002 (Fixed)
Checksum:                0xffffe70b
UUID:                    54d36115-6bd7-5237-615a-b53a3a912557
Saved state:             0
```

## Building

You'll need golang installed. Tested with go >= 1.2.

```
make
```

### Creating a Debian source package

The following dependencies are required/recommended:

```
apt-get install build-essential debuild dpkg-dev devscripts
```

Create the source deb, that will be moved to `~/debian/go-vhd`:

```
make srcdeb
```
