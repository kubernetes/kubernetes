# etcd2-backup-coreos

Remote backup and multi-node restore services for etcd2 clusters on CoreOS Linux.

**Warning:** This package is only intended for use on CoreOS Linux.

## Terminology

**Founding member** : The node which is the first member of the new recovered cluster. It is this node's rclone backup data (only) that will be used to restore the cluster. The rest of the nodes will join the cluster with no data, and simply catch up with the **founding member**.

## Configuration

Before installing etcd2-backup, you need to configure `30-etcd2-backup-restore.conf`.

```
[Service]
Environment="ETCD_RESTORE_MASTER_ADV_PEER_URLS=<http://host:port>"
Environment="RCLONE_ENDPOINT=remote-name:path/to/backups"
```

Assuming you're deploying to CoreOS with etcd2, you should only need to change

* `ETCD_RESTORE_MASTER_ADV_PEER_URLS`
   This is the new advertised peer url of the new etcd2 node that will be the founding member of the new restored cluster. We will call this node the **founding member**.

*  `RCLONE_ENDPOINT`
    The rclone endpoint to which backups will be stored.

    Feel free to point any number of machines at the same RCLONE_ENDPOINT, path and all. Backups for each machine are stored in a sub-folder named with the machine ID (%m in systemd parlance)

*  `./rclone.conf`
    The rclone configuration file which will be installed. Must list a `[section]` which matches `RCLONE_ENDPOINT`'s remote-name component.

    An easy way to generate this config file is to [install rclone](http://rclone.org/install/) on your local machine. Then follow the [configuration instructions](http://rclone.org/docs/) to generate an `rclone.conf` file.

If you want to adjust backup frequency, edit `./etcd2-backup.timer`

## Installation

Once you've got those things configured, you can run `./build`.

The `build` script generates a tarball for copying to CoreOS instances. The tarball contains the `etcd2-backup-install` script.

After extracting the contents of the tar file and running the install script, three new systemd services are added. One service, `etcd2-backup`, performs periodic etcd backups, while the other two services, `etcd2-restore` and `etcd2-join`, handle restore procedures.

* `etcd2-backup.service`
   A oneshot service which calls `etcdctl backup` and syncs the backups to the rclone endpoint (using an rclone container, of course). `etcd2-backup.timer` is responsible for periodically running this service.

* `etcd2-restore.service`
    A oneshot service which wipes all etcd2 data and restores a single-node cluster from the rclone backup. This is for restoring the **founding member** only.

* `etcd2-join.service`
   A oneshot service which wipes all etcd2 data and re-joins the new cluster. This is for adding members **after** the **founding member** has succesfully established the new cluster via `etcd2-restore.service`

## Recovery

This assumes that your cluster has lost quorum, and is not recoverable. Otherwise you should probably try to heal your cluster first.

### Backup Freshness

Two factors contribute to the relative freshness or staleness of a backup. The `etcd2-backup.timer` takes a backup every 30 seconds by default, and the etcd `snapshot-count` option controls how many transactions are committed between each write of the snapshot to permanent storage. Given those parameters, we can compute the upper bound on the outdatedness of a backup.
Assumptions:
* transaction rate is a constant `1000 transactions / second`
* `etcd2-backup.timer` is configured for a 30 second interval
* `etcd2 snapshot-count=10000`

```
max-missed-seconds= (10000 transactions / (1000 transactions / second)) + 30 seconds = 40 seconds
```

### Recovery Procedure

1. Make sure `etcd2.service` and `etcd2-backup.timer` are stopped on all nodes in the cluster

2. Restore the **founding member** by starting `etcd2-restore.service` and then, if successful, `etcd2.service`

3. Restore the rest of the cluster **one at a time**. Start `etcd2-join.service`, and then, if successful, `etcd2.service`. Please verify with `etcdctl cluster-health` that the expected set of nodes is present and healthy after each node joins.

4. Verify that your data is sane (enough). If so, kick off `etcd2-backup.timer` on all nodes and, hopefully, go back to bed.

## Retroactively change the founding member

It is necessary to change the cluster's founding member in order to restore a cluster from any other node's data.

Change the value of `ETCD_RESTORE_MASTER_ADV_PEER_URLS` in `30-etcd2-backup-restore.conf` to the advertised peer url of the new founding member. Repeat the install process above on all nodes in the cluster, then proceed with the [recovery procedure](README.md#recovery-procedure).

## Example

Let's pretend that we have an initial 3 node CoreOS cluster that we want to back up to S3.


| ETCD_NAME  | ETCD_ADVERTISED_PEER_URL |
| ------------- |:-------------:|
| e1   | http://172.17.4.51:2379 |
| e2   | http://172.17.4.52:2379 |
| e3   | http://172.17.4.53:2379 |

In the event that the cluster fails, we want to restore from `e1`'s backup

## Configuration

```
[Service]
Environment="ETCD_RESTORE_MASTER_ADV_PEER_URLS=http://172.17.4.51:2379"
Environment="RCLONE_ENDPOINT=s3-testing-conf:s3://etcd2-backup-bucket/backups"
```

The `./rclone.conf` file must contain a `[section]` matching `RCLONE_ENDPOINTS`'s remote-name component.

```
[s3-testing-conf]
type = s3
access_key_id = xxxxxxxx
secret_access_key = xxxxxx
region = us-west-1
endpoint =
location_constraint =
```

## Installation

```sh
cd etcd2-backup
./build
scp etcd2-backup.tgz core@e1:~/
ssh core@e1
e1 $  mkdir -p ~/etcd2-backup
e1 $  mv etcd2-backup.tgz etcd2-backup/
e1 $ cd etcd2-backup
e1 $ tar zxvf ~/etcd2-backup.tgz
e1 $ ./etcd2-backup-install
# Only do the following two commands if this node should generate backups
e1 $ sudo systemctl enable etcd2-backup.timer
e1 $ sudo systemctl start etcd2-backup.timer

e1 $ exit
```

Now `e1`'s etcd data will be backed up to `s3://etcd2-backup-bucket/backups/<e1-machine-id>/` according to the schedule described in `etcd2-backup.timer`.

You should repeat the process for `e2` and `e3`. If you do not want a node to generate backups, omit enabling and starting `etcd2-backup.timer`.

## Restore the cluster

Let's assume that a mischievous friend decided it would be a good idea to corrupt the etcd2 data-dir on ALL of your nodes (`e1`,`e2`,`e3`). You simply want to restore the cluster from `e1`'s backup.

Here's how you would recover:

```sh
# First, ENSURE etcd2 and etcd2-backup are not running on any nodes
for node in e{1..3};do
    ssh core@$node "sudo systemctl stop etcd2.service etcd2-backup.{timer,service}"
done

ssh core@e1 "sudo systemctl start etcd2-restore.service && sudo systemctl start etcd2.service"

for node in e{2..3};do
    ssh core@$node "sudo systemctl start etcd2-join.service && sudo systemctl start etcd2.service"
    sleep 10
done
```

After e2 and e3 finish catching up, your cluster should be back to normal.

## Migrate the cluster

The same friend who corrupted your etcd2 data-dirs decided that you have not had enough fun. This time, your friend dumps coffee on the machines hosting `e1`, `e2` and `e3`. There is a horrible smell, and the machines are dead.

Luckily, you have a new 3-node etcd2 cluster ready to go, along with the S3 backup for `e1` from your old cluster.

The new cluster configuration looks like this. Assume that etcd2-backup is not installed. (If it is, you NEED to make sure it's not running on any nodes)

| ETCD_NAME  | ETCD_ADVERTISED_PEER_URL |
| ------------- |:-------------:|
| q1   | http://172.17.8.201:2379 |
| q2   | http://172.17.8.202:2379 |
| q3   | http://172.17.8.203:2379 |

We will assume `q1` is the chosen founding member, though you can pick any node you like.

## Migrate the remote backup

First, you need to copy your backup from `e1`'s backup folder to `q1`'s backup folder. I will show the S3 example.

```sh
# Make sure to remove q1's backup directory, if it exists already
aws s3 rm --recursive s3://etcd2-backup-bucket/backups/<q1-machine-id>
aws s3 cp --recursive s3://etcd2-backup-bucket/backups/<e1-machine-id> s3://etcd2-backup-bucket/backups/<q1-machine-id>
```

## Configure the New Cluster

```
[Service]
Environment="ETCD_RESTORE_MASTER_ADV_PEER_URLS=http://172.17.8.201:2379"
Environment="RCLONE_ENDPOINT=s3-testing-conf:s3://etcd2-backup-bucket/backups"
```

Since this is a new cluster, each new node will have new `machine-id` and will not clobber your backups from the old cluster, even though `RCLONE_ENDPOINT` is the same for both the old `e` cluster and the new `q` cluster.

## Installation

We first want to install the configured etcd2-backup package on all nodes, but not start any services yet.

```sh
cd etcd2-backup
./build
for node in q{1..3};do
    scp etcd2-backup.tgz core@$node:~/
    ssh core@$node "mkdir -p ~/etcd2-backup"
    ssh core@$node "mv etcd2-backup.tgz etcd2-backup/"
    ssh core@$node " cd etcd2-backup"
    ssh core@$node " tar zxvf ~/etcd2-backup.tgz"
    ssh core@$node " ./etcd2-backup-install"
done
```

## Migrate the Cluster

With `q1` as the founding member.

```sh
# First, make SURE etcd2 and etcd2-backup are not running on any nodes

for node in q{1..3};do
    ssh core@$node "sudo systemctl stop etcd2.service"
done

ssh core@q1 "sudo systemctl start etcd2-restore.service && sudo systemctl start etcd2.service"

for node in q{2..3};do
    ssh core@$node "sudo systemctl start etcd2-join.service && sudo systemctl start etcd2.service"
    sleep 10
done
```

Once you've verifed the cluster has migrated properly, start and enable `etcd2-backup.timer` on at least one node.

```sh
ssh core@q1 "sudo systemctl enable etcd2-backup.service && sudo systemctl start etcd2-backup.service"
```

You should now have periodic backups going to: `s3://etcd2-backup-bucket/backups/<q1-machine-id>`

## Words of caution

1. Notice the `sleep 10` commands that follow starting `etcd2-join.service` and then `etcd2.service`. This sleep is there to allow the member that joined to cluster time to catch up on the cluster state before we attempt to add the next member. This involves sending the entire snapshot over the network. If you're dataset is large, or the network between nodes is slow, or your disks are already bogged down, etc- you may need to turn the sleep time up.

   In the case of large data sets, it is recommended that you copy the data directory produced by `etcd2-restore` on the founding member to the other nodes before running `etcd2-join` on them. This will avoid etcd transferring the entire snapshot to every node after it joins the cluster.

2. It is not recommended clients be allowed to access the etcd2 cluster **until** all members have been added and finished catching up.
