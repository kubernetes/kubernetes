# go-git + aerospike: a git repository backed by a database

<img src="https://upload.wikimedia.org/wikipedia/en/2/2b/Aerospike_logo.png" align="right"/> This is an example of a [go-git](https://github.com/src-d/go-git) repository backed by [Aerospike](http://www.aerospike.com/). 




### and what this means ...
*git* has as very well defined storage system, the `.git` directory, present on any repository. This is the place where `git` stores al the [`objects`](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects), [`references`](https://git-scm.com/book/es/v2/Git-Internals-Git-References) and [`configuration`](https://git-scm.com/docs/git-config#_configuration_file). This information is stored in plain files.

Our original **go-git** version was designed to work in memory, some time after we added support to read the `.git`, and now we have added support for fully customized  [storages](https://godoc.org/github.com/src-d/go-git#Storer).

This means that the internal database of any repository can be saved and accessed on any support, databases, distributed filesystems, etc. This functionality is pretty similar to the [libgit2 backends](http://blog.deveo.com/your-git-repository-in-a-database-pluggable-backends-in-libgit2/)


Installation
------------

What do you need? You need an *aerospike* server. The easiest way to get one for testing is running the official **docker** container provided by Aerospike:

```
docker run -d -p 3000:3000 --name aerospike aerospike/aerospike-server
```

Now, we need the sample code.

```
go get -u github.com/mcuadros/go-git-aerospike/...
```

Running this command will make the binary `go-git-aerospike`. if you have `GOPATH` on your `PATH`, you are ready to go. If not, this is a great moment.

Usage
-----

### Cloning the repository into the database

Running the command `go-git-aerospike` with the `clone` option followed by the URL of a git repository clones the repository into the database, storing all the git objects in it:

```sh
go-git-aerospike clone https://github.com/src-d/flamingo.git
```

The repository is stored in the aerospike database. This means that all the internal objects like commits, trees, blobs and tags are `records` in different `sets` in the `test` namespace:

```sql
aql> SELECT hash, type, url FROM test.commit
```

```
+--------------------------------------------+----------+-------+-----------------------------------+
| hash                                       | type     | blob  | url                               |
+--------------------------------------------+----------+-------+-----------------------------------+
| "c94450c805876e49b38d2ff1103b8c09cdd2aef4" | "commit" | 00 00 | ...github.com/src-d/flamingo.git" |
| "7f71640877608ee9cfe584fac216f03f9aebb523" | "commit" | 00 00 | ...github.com/src-d/flamingo.git" |
| "255f097450dd91812c4eb7b9e0d3a4f034f2acaf" | "commit" | 00 00 | ...github.com/src-d/flamingo.git" |
+--------------------------------------------+----------+-------+-----------------------------------+
102 rows in set (0.071 secs)
```

And also the references and the configuration (remotes) are stored in it.

```sql
aql> SELECT name, target, url FROM test.reference
```
```
+------------------------------+--------------------------------------------+-----------------------+
| name                         | target                                     | url                   |
+------------------------------+--------------------------------------------+-----------------------+
| "HEAD"                       | "ref: refs/heads/master"                   | ...rc-d/flamingo.git" |
| "refs/heads/master"          | "ed3e1aa2e46584cb803ed356cb5d8855f6d05660" | ...rc-d/flamingo.git" |
| "refs/remotes/origin/master" | "ed3e1aa2e46584cb803ed356cb5d8855f6d05660" | ...rc-d/flamingo.git" |
+------------------------------+--------------------------------------------+-----------------------+
3 rows in set (0.046 secs)
```

### Reading the repository

Running the `log` command, a `git log --online` like result is printed:

```sh
go-git-aerospike log https://github.com/src-d/flamingo.git
```

The URL of the repository is the way we identify the objects in the `set`s, since we can clone several repositories to the same database.

```
ed3e1aa ID is also allowed in SendFormTo and SayTo
2031f3e Handle close of message channel in WaitForMessage
e784495 Add SendFormTo and SayTo
447748a Make text in attachments accept markdown
595b4e7 Form author name and author icon and text groupfield
0f2e315 Test for InvokeAction
0dc7c9a Handle closing of channel
b3f167b Implement InvokeAction
```

The process has read all the commits and all the needed objects from the aerospike sets.

### Playing with the database

If you want to explore the database, you can execute the `aql` tool and run some queries:

```sh
docker run -it aerospike/aerospike-tools aql -h 172.17.0.1
aql> SELECT * FROM test;
```
