# Rollback workflow

Build it in this directory.
Make sure you have etcd dependency ready. Last time we use etcd v3.0.7.
```
$ go build .
```


Run it:
```
$ ./rollback2 --data-dir $ETCD_DATA_DIR --ttl 1h
```

This will rollback KV pairs from v3 into v2.
If a key was attached to a lease before, it will be created with given TTL (default to 1h).

On success, it will print at the end:
```
Finished successfully
```

Repeat this on all etcd members.

You can do simple check on keys (if any exists):
```
etcdctl ls /
```

Important Note
------

This tool isn't recommended to use if any problem comes up in etcd3 backend.
Please report bugs and we will fix it soon.

If it's still preferred to run this tool, please backup all your data beforehand.
This tool will also back up datadir to same path with ".rollback.backup" suffix.

Caveats:
- The tool doesn't preserve versions of keys.
- If any v2 data exists before rollback, they will be wiped out.
- v3 data only exists in the backup after successful rollback.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/images/etcd/rollback/README.md?pixel)]()