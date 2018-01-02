On disk format
==============

The data directory is `/var/lib/rkt`, unless configured otherwise.
For details, see [the `paths` kind in configuration documentation][rktkind-paths].
The `--dir` command line option can be used to change this location.

#### CAS database

The CAS database is stored in `/var/lib/rkt/cas/db`.
The database schema can be migrated to newer versions ([#706][rkt-706]).

#### CAS

The CAS also uses other directories in `/var/lib/rkt/cas/`.
To ensure stability for the CAS, we need to make sure we don't remove any of those directories or make any destructive changes to them.
Future version of rkt will retain compatibility with older CAS versions.

#### Pods

The pods are stored in `/var/lib/rkt/pods/` as explained in [Life-cycle of a pod][pod-lifecycle]

The stability of prepared and exited pods is desirable, but not as critical as the CAS.

#### Configuration

The [configuration][configuration] on-disk format is documented separately.


[configuration]: ../configuration.md
[pod-lifecycle]: pod-lifecycle.md
[rkt-706]: https://github.com/coreos/rkt/issues/706
[rktkind-paths]: ../configuration.md#rktkind-paths
