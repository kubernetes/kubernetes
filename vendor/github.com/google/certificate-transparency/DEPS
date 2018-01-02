deps = {
     "gflags":  	 			 "https://github.com/gflags/gflags.git@v2.1.2",
     "glog":             "https://github.com/benlaurie/glog.git@0.3.4-fix",
     "googlemock": 			 "https://github.com/google/googlemock.git@release-1.7.0",
     "googlemock/gtest": "https://github.com/google/googletest.git@release-1.7.0",
     "icu4c":            "https://github.com/icu-project/icu4c.git@bbd17a792336de5873550794f8304a4b548b0663",
     "json-c": 					 "https://github.com/AlCutter/json-c.git@json-c-0.12-20140410-fix",
     "ldns":             "https://github.com/benlaurie/ldns.git@1.6.17-fix",
     "leveldb": 				 "https://github.com/google/leveldb.git@v1.18",
     "libevent": 				 "https://github.com/libevent/libevent.git@release-2.0.22-stable",
     "libevhtp": 				 "https://github.com/ellzey/libevhtp.git@a89d9b3f9fdf2ebef41893b3d5e4466f4b0ecfda",
     "certificate-transparency/third_party/objecthash":
                         "https://github.com/benlaurie/objecthash.git@798f66bd8c5313da226aa7a60c114147910a7407",
     "openssl": 				 "https://github.com/openssl/openssl.git@OpenSSL_1_0_2d",
     "protobuf":         "https://github.com/google/protobuf.git@v2.6.1",
     "protobuf/gtest":   "https://github.com/google/googletest.git@release-1.7.0",
     "libsnappy":        "https://github.com/google/snappy.git@1.1.3",
     # Randomly chosen github mirror
     "sqlite3-export": 	 "http://repo.or.cz/sqlite-export.git",
     "sqlite3": 				 "http://repo.or.cz/sqlite.git@version-3.8.10.1",
		 "tcmalloc":				 "https://github.com/gperftools/gperftools.git@gperftools-2.4"
}

# Can't use deps_os for this because it doesn't know about freebsd :/
deps_overrides = {
  "freebsd10": {
     "googlemock": 			 "https://github.com/AlCutter/googlemock-fbsd.git@1.7.0",
     "googlemock/gtest": "https://github.com/AlCutter/googletest-fbsd.git@1.7.0",
     "protobuf":         "https://github.com/benlaurie/protobuf.git@2.6.1-fix",
     "protobuf/gtest":   "https://github.com/AlCutter/googletest-fbsd.git@1.7.0",
     "libunwind":        "git://git.sv.gnu.org/libunwind.git@v1.1",
  },
  "linux2": {
     "libunwind":        "git://git.sv.gnu.org/libunwind.git@v1.1",
  },
  "darwin": {
     "ldns": "https://github.com/benlaurie/ldns.git@1.6.17-fix",
  }
}

make_os = {
	"freebsd10": "gmake",
	"darwin": "gnumake"
}

import os
import multiprocessing
import sys

print("Host platform is %s" % sys.platform)
if sys.platform in deps_overrides:
  print("Have %d overrides for platform" % len(deps_overrides[sys.platform]))
  deps.update(deps_overrides[sys.platform])
if sys.platform in make_os:
	make = make_os[sys.platform]
else:
	make = "make"

num_cores = multiprocessing.cpu_count()

print("Using make %s with %d jobs" % (make, num_cores))

here = os.getcwd()

hooks = [
    {
        "name": "libunwind",
        "pattern": "^libunwind/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_libunwind" ],
    },
    {
        "name": "tcmalloc",
        "pattern": "^tcmalloc/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_tcmalloc" ],
    },
    {
        "name": "openssl",
        "pattern": "^openssl/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_openssl" ],
    },
    {
        "name": "libevent",
        "pattern": "^libevent/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_libevent" ],
    },
    {
        "name": "libevhtp",
        "pattern": "^libevhtp/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_libevhtp" ],
    },
    {
        "name": "gflags",
        "pattern": "^gflags/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_gflags" ],
    },
    {
        "name": "glog",
        "pattern": "^glog/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_glog" ],
    },
    {
        "name": "protobuf",
        "pattern": "^protobuf/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_protobuf" ],
    },
    {
        "name": "ldns",
        "pattern": "^ldns/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_ldns" ],
    },
    {
        "name": "sqlite3",
        "pattern": "^sqlite3/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_sqlite3" ],
    },
    {
        "name": "libsnappy",
        "pattern": "^libsnappy/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_libsnappy" ],
    },
    {
        "name": "leveldb",
        "pattern": "^leveldb/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_leveldb" ],
    },
    {
        "name": "json-c",
        "pattern": "^json-c/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_json-c" ],
    },
    {
        "name": "icu4c",
        "pattern": "^icu4c/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_icu4c" ],
    },
    {
        "name": "objecthash",
        "pattern": "^certificate-transparency/third_party/objecthash/",
        "action": [ make, "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_objecthash" ],
    },
    # Do this last
    {
        "name": "ct",
        "pattern": "^certificate-transparency/",
        "action": [ make, "-j", str(num_cores), "-f", os.path.join(here, "certificate-transparency/build.gclient"), "_configure-ct" ],
    }
]
