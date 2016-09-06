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
Finished.
```

You can do simple check on keys (if any exists):
```
etcdctl ls /
```

Important Note
------

This tool isn't recommended to use if problem comes up. Please report bugs and we will fix it soon.

If it's still needed to run this tool, please backup all your data beforehand.

Caveats:
- No guarantee on versions.
- If any v2 data exists before rollback, they will be wiped out.
- No v3 data left after rollback.
